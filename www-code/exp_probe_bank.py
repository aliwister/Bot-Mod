"""Fit the probe bank on the TRAIN split: base rates, observation model, tau.

Everything the moderator needs in order to select probes and update on answers is
estimated here, once, offline, on `www-dataset/train/dataset-train.json` -- which is disjoint from
the ID test files. Nothing is fit on test.

Three things come out:

  mu[q][h]   per-token base rate of hypothesis h's persona on answers to probe q,
             estimated from *mismatched* answers (the derangement control E3 used).
             Subtracting it is the DCPMI / contextual-calibration correction; without
             it, 58% of E3's apparent gain was a base-rate shift.
  M[q][h][o] P(the calibrated likelihood's argmax is o | true intent is h), by
             counting, Laplace-smoothed. This is the noisy observation model that
             makes probe selection ECED-shaped: a probe reweights hypotheses, it does
             not cut them cleanly.
  tau        evidence temperature, fit by minimizing the posterior's NLL against the
             truth over single-probe updates. Without it b ∝ b·P(r|h) saturates on the
             first answer.

Cost: one elicitation per (train row, probe) plus 5 prefills, and the same again for
the derangement. Everything else is counting.

Writes www-results/probe_bank.json
"""
import json, math, os, random, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import INTENTS  # noqa: E402
from common import enrich_content, _is_organic  # noqa: E402
from exp_e2_logprob_posterior import read_posterior, entropy  # noqa: E402
from exp_e3_teacher_forced_likelihood import elicit, loglik, tok  # noqa: E402
from calibration import temper, fit_temperature  # noqa: E402
from moderator_v2 import PROBE_BANK, BANK_PATH  # noqa: E402

from paths import TRAIN  # noqa: E402
PER_INTENT = 45          # cap per intent; spam has only 6 in train (stated as a caveat)
LAPLACE = 1.0
OUT = BANK_PATH
K = len(INTENTS)
ORGANIC = next(i for i, t in enumerate(INTENTS) if _is_organic(t))


def sample_rows(rng):
    by_intent = {t: [] for t in INTENTS}
    for row in json.load(open(TRAIN, encoding="utf-8")):
        t = row["label_intent"].strip().lower()
        if t in by_intent:
            by_intent[t].append(row)
    out = []
    for t, rows in by_intent.items():
        rng.shuffle(rows)
        out += rows[:PER_INTENT]
    rng.shuffle(out)
    return out, {t: min(len(v), PER_INTENT) for t, v in by_intent.items()}


def build(args):
    """-> one (row, probe) record: prior, answer, and the 5 raw logliks."""
    row, probe = args
    content, mode = enrich_content(row)
    community = row["community"].strip()
    lp = read_posterior(content, community)
    if lp is None:
        return None
    answer = elicit(row, PROBE_BANK[probe])
    if not answer:
        return None
    lls, n = [], 0
    for h in INTENTS:
        ll, nt = loglik(content, community, PROBE_BANK[probe], answer, h)
        if ll is None:
            return None
        lls.append(ll)
        n = nt
    return {"probe": probe, "y": INTENTS.index(row["label_intent"].strip().lower()),
            "mode": mode, "community": community, "prior_raw": lp["p_a"],
            "answer": answer, "loglik": lls, "answer_tokens": n,
            "content": content}


def derange(items, rng):
    """Score each item's likelihood against another item's answer, within probe.

    This is what mu is estimated from -- a hypothesis's score on answers that are not
    this item's. Same construction as E3's negative control, reused here as the
    calibration reference so it costs nothing extra.
    """
    by_probe = {}
    for it in items:
        by_probe.setdefault(it["probe"], []).append(it)
    jobs = []
    for probe, group in by_probe.items():
        if len(group) < 2:
            continue
        order = list(range(len(group)))
        for _ in range(200):                     # reject fixed points
            rng.shuffle(order)
            if all(i != j for i, j in enumerate(order)):
                break
        for i, j in enumerate(order):
            jobs.append((group[i], group[j]))

    def one(pair):
        it, other = pair
        lls, n = [], 0
        for h in INTENTS:
            ll, nt = loglik(it["content"], it["community"], PROBE_BANK[it["probe"]],
                            other["answer"], h)
            if ll is None:
                return None
            lls.append(ll)
            n = nt
        return it, lls, n

    with ThreadPoolExecutor(max_workers=32) as ex:
        for got in ex.map(one, jobs):
            if got is None:
                continue
            it, lls, n = got
            it["loglik_mismatch"] = lls
            it["mismatch_tokens"] = n


def fit_mu(items):
    """mu[probe][h]: per-token mean loglik h gives to *other* items' answers.

    A probe with fewer than two scored items has no derangement and therefore no
    estimate of its own; it falls back to the base rate pooled over all probes. The
    base rates are close across probes (they are a property of the persona, not of the
    question), so the fallback is mild -- but it must exist, or one sparse probe
    silently takes the whole bank down.
    """
    by_probe = {}
    for it in items:
        if "loglik_mismatch" in it:
            by_probe.setdefault(it["probe"], []).append(it)

    def rate(group):
        tot_tok = sum(it["mismatch_tokens"] for it in group) or 1
        return [sum(it["loglik_mismatch"][j] for it in group) / tot_tok
                for j in range(K)]

    pooled = rate([it for g in by_probe.values() for it in g]) if by_probe \
        else [0.0] * K
    return {probe: rate(by_probe[probe]) if probe in by_probe else list(pooled)
            for probe in PROBE_BANK}


def evidence_of(it, mu):
    n = max(it["answer_tokens"], 1)
    m = mu[it["probe"]]
    return [(it["loglik"][j] - m[j] * n) / n for j in range(K)]


def fit_M(items, mu):
    """M[probe][h][o], Laplace-smoothed counts of the calibrated argmax."""
    M = {}
    for probe in PROBE_BANK:
        group = [it for it in items if it["probe"] == probe]
        counts = [[LAPLACE] * K for _ in range(K)]
        for it in group:
            ev = evidence_of(it, mu)
            o = max(range(K), key=lambda j: ev[j])
            counts[it["y"]][o] += 1
        M[probe] = [[c / sum(rowc) for c in rowc] for rowc in counts]
    return M


def fit_tau(items, mu, prior_temp, lo=0.02, hi=200.0, iters=70):
    """tau minimizing the mean NLL of the one-probe posterior against the truth."""
    def post(it, tau):
        b = temper(it["prior_raw"], prior_temp)
        ev = evidence_of(it, mu)
        logb = [math.log(max(b[j], 1e-12)) + ev[j] / tau for j in range(K)]
        mx = max(logb)
        ex = [math.exp(v - mx) for v in logb]
        z = sum(ex)
        return [v / z for v in ex]

    def loss(tau):
        return sum(-math.log(max(post(it, tau)[it["y"]], 1e-12))
                   for it in items) / len(items)

    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if loss(c) < loss(d):
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
    tau = (a + b) / 2
    return tau, loss(tau), post


def main():
    rng = random.Random(5)
    t0 = time.time()
    tok()                                        # warm before the pool (thread-unsafe)

    rows, avail = sample_rows(rng)
    jobs = [(r, q) for r in rows for q in PROBE_BANK]
    print(f"probe bank: {len(PROBE_BANK)} probes x {len(rows)} train rows "
          f"= {len(jobs)} pairs; train availability {avail}", flush=True)

    with ThreadPoolExecutor(max_workers=32) as ex:
        items = [x for x in ex.map(build, jobs) if x is not None]
    print(f"  elicited + scored {len(items)} pairs in {time.time() - t0:.0f}s", flush=True)

    derange(items, rng)
    print(f"  mismatch reference on {sum(1 for i in items if 'loglik_mismatch' in i)}"
          f" pairs in {time.time() - t0:.0f}s", flush=True)

    # initial prior temperature, fit on the sampled train rows (exp_e15_train_refit.py --apply
    # then refits T on every train item and tau given that T)
    uniq = {(tuple(it["prior_raw"]), it["y"]) for it in items}
    prior_temp, _ = fit_temperature([{"p_raw": list(p), "y_intent": y} for p, y in uniq])
    print(f"prior temperature T = {prior_temp:.3f} (train sample)", flush=True)

    mu = fit_mu(items)
    M = fit_M(items, mu)
    tau, tau_loss, post = fit_tau(items, mu, prior_temp)

    # how good is the evidence, per probe, on train? (a sanity read, not the result --
    # the result is measured on test by the system eval)
    per_probe = {}
    for probe in PROBE_BANK:
        group = [it for it in items if it["probe"] == probe]
        if not group:
            continue
        acc_ev = sum(1 for it in group
                     if max(range(K), key=lambda j: evidence_of(it, mu)[j]) == it["y"])
        # does the class marginal move the right way?
        moved = 0
        for it in group:
            b0 = temper(it["prior_raw"], prior_temp)
            b1 = post(it, tau)
            truth_mal = it["y"] != ORGANIC
            d = sum(b1[j] for j in range(K) if j != ORGANIC) - \
                sum(b0[j] for j in range(K) if j != ORGANIC)
            moved += 1 if (d > 0) == truth_mal else 0
        per_probe[probe] = {
            "n": len(group),
            "acc_evidence_argmax": acc_ev / len(group),
            "frac_class_marginal_moved_correctly": moved / len(group),
            "mean_answer_tokens": sum(it["answer_tokens"] for it in group) / len(group),
            "mu": mu[probe],
        }

    d = {
        "meta": {
            "n_train_rows": len(rows), "n_pairs": len(items),
            "per_intent_cap": PER_INTENT, "train_availability": avail,
            "laplace": LAPLACE, "seconds": time.time() - t0,
            "tau_train_nll": tau_loss,
            "note": "fit on www-dataset/train only; disjoint from TEST_FILES",
        },
        "prior_temperature": prior_temp,
        "tau": tau,
        "mu": mu,
        "M": M,
        "per_probe_train": per_probe,
    }
    json.dump(d, open(OUT, "w"), indent=2)

    print(f"\nfitted tau = {tau:.3f}  (train posterior NLL {tau_loss:.4f})")
    hdr = f"\n{'probe':<26}{'n':>5}{'ev acc':>9}{'class dir':>11}{'ans tok':>9}"
    print(hdr)
    print("-" * len(hdr.strip("\n")))
    for probe, s in per_probe.items():
        print(f"{probe:<26}{s['n']:>5}{s['acc_evidence_argmax']:>9.3f}"
              f"{s['frac_class_marginal_moved_correctly']:>11.3f}"
              f"{s['mean_answer_tokens']:>9.0f}")

    print(f"\nobservation model M[q][h][o], rows = true intent, cols = argmax:")
    for probe in PROBE_BANK:
        diag = sum(M[probe][h][h] for h in range(K)) / K
        print(f"  {probe:<26} mean diagonal {diag:.3f}")
    print(f"\nwrote {OUT}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
