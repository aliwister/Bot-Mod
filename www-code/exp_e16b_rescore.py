"""Per-answer scoring of the training interrogations, and the train-side fit of β and α.

The agent's answers are kept exactly as exp_e16_multiturn.py collected them (full history on
the agent side); the moderator scores each answer alone,

    log M(a_k | persona(h), content, q_k)        (exp_e3_teacher_forced_likelihood.loglik)

for every hypothesis h, and likewise for the deranged answers that give the base rates μ.
`fit_variant` then fits β = 1/τ (NLL of the one-probe posterior, exp_probe_bank.fit_tau) and the
stopping level α (best train F1_val on the grid) on train.

    BOTMOD_PORT_BASE=18000 python exp_e16b_rescore.py run train varied
        -> www-results/e16b_rescore_varied_train.json
"""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import INTENTS  # noqa: E402
from common import compute_metrics  # noqa: E402
from exp_e3_teacher_forced_likelihood import loglik, tok  # noqa: E402
import exp_e16_multiturn as E  # noqa: E402
from exp_probe_bank import fit_tau  # noqa: E402
from moderator_v2 import ProbeBank  # noqa: E402
from exp_e12_head_to_head import LOGS  # noqa: E402

K = len(INTENTS)


def out_path(split, cond="varied"):
    return os.path.join(LOGS, f"e16b_rescore_{cond}_{split}.json")


def nohist_all(content, community, q, answer):
    lls = []
    for h in INTENTS:
        ll, _ = loglik(content, community, q, answer, h)
        if ll is None:
            return None
        lls.append(ll)
    return lls


def stage_run(split, cond="varied"):
    tok()
    src = json.load(open(E.out_path(split, cond)))
    recs = src["records"]
    t0 = time.time()
    if split == "train":                     # the same derangement E16 used, rebuilt with its seed
        rng = random.Random(5)
        perm = {}
        for k in range(E.TURNS):
            order = list(range(len(recs)))
            for _ in range(200):
                rng.shuffle(order)
                if all(i != j for i, j in enumerate(order)):
                    break
            perm[k] = order

    def one(i):
        r = recs[i]
        for k, t in enumerate(r["turns"]):
            t["loglik_nohist"] = nohist_all(r["content"], r["community"], t["q"], t["answer"])
            if split == "train":
                other = recs[perm[k][i]]["turns"][k]["answer"]
                t["mismatch_nohist"] = nohist_all(r["content"], r["community"], t["q"], other)
                t["mismatch_answer_tokens"] = len(tok().encode(other, add_special_tokens=False))
        return i

    done = 0
    with ThreadPoolExecutor(max_workers=E.WORKERS) as ex:
        for _ in ex.map(one, range(len(recs))):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    json.dump({"config": {**src["config"], "rescored": ["nohist"], "seconds": time.time() - t0},
               "records": recs}, open(out_path(split, cond), "w"))
    print(f"[{split}] rescored {len(recs)} in {time.time() - t0:.0f}s", flush=True)


# ── fit on train ─────────────────────────────────────────────────────────────

def turn_ll(t, mismatch=False):
    return t.get("mismatch_nohist") if mismatch else t["loglik_nohist"]


def turn_tokens(t, mismatch=False):
    return (t.get("mismatch_tokens") or t.get("mismatch_answer_tokens")) if mismatch else t["answer_tokens"]


def to_replay(recs, prior):
    """-> moderator_v2 replay records, technique slots s1..s3 in order."""
    out = []
    for r in recs:
        if r["item_id"] in prior and all(turn_ll(t) is not None for t in r["turns"]):
            out.append({"item_id": r["item_id"], "prior_raw": prior[r["item_id"]], "mode": r["mode"],
                        "community": r["community"], "user_model": r["user_model"], "y": r["y"],
                        "y_binary": r["y_binary"],
                        "probes": {f"s{k + 1}": {"loglik": turn_ll(t), "answer_tokens": t["answer_tokens"]}
                                   for k, t in enumerate(r["turns"])}})
    return out


def fit_variant(train, prior, T):
    """-> (bank, alpha, info): base rates μ per technique slot from the deranged answers, τ by
    NLL of the one-probe posterior, α = the grid level with the best train F1_val."""
    mu = {}
    for k in range(E.TURNS):
        g = [(turn_ll(r["turns"][k], True), turn_tokens(r["turns"][k], True)) for r in train]
        g = [(ll, n) for ll, n in g if ll is not None and n]
        tot = sum(n for _, n in g) or 1
        mu[f"s{k + 1}"] = [sum(ll[j] for ll, _ in g) / tot for j in range(K)]
    pairs = [{"prior_raw": prior[r["item_id"]], "y": r["y"], "probe": f"s{k + 1}",
              "loglik": turn_ll(t), "answer_tokens": t["answer_tokens"]}
             for r in train if r["item_id"] in prior for k, t in enumerate(r["turns"])
             if turn_ll(t) is not None]
    tau, _, _ = fit_tau(pairs, mu, T)
    uni = [[1.0 / K] * K for _ in range(K)]      # the fixed probe order needs no observation model
    bank = ProbeBank({"mu": mu, "M": {f"s{k + 1}": uni for k in range(E.TURNS)}, "tau": tau,
                      "prior_temperature": T})
    tr = to_replay(train, prior)
    sweep = {a: compute_metrics(E.replay(bank, tr, a))["val_f1"] for a in E.ALPHAS}
    alpha = max(E.ALPHAS, key=lambda a: sweep[a])
    return bank, alpha, {"tau": tau, "alpha": alpha, "train_val_f1": sweep[alpha]}


if __name__ == "__main__":
    if sys.argv[1] == "run":
        stage_run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "varied")
    else:
        sys.exit("want: run train varied")
