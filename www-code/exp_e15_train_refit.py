"""E15: refit Bot-Mod's prior temperature T, likelihood scale tau and stopping level alpha on TRAIN.

The earlier constants were not all fit on train: T came from an earlier calibration study that fit it
on half of the in-distribution TEST files, and alpha was the best of a sweep scored on 384
in-distribution TEST items (an earlier policy study). tau was fit on train, but conditioned on
that T. This script refits all three from the training split alone, with the same procedures:

    T      golden-section NLL fit of the tempered label prior (calibration.fit_temperature)
    tau    NLL of the one-probe posterior, on the probe bank's own train sample
           (exp_probe_bank.sample_rows, seed 5, <= 45 rows per intent), given the new T
    alpha  the stopping level in eval_common.ALPHAS with the best train F1_val, fixed
           probe order, at most 3 probes (the paper's configuration)

Input:  www-results/e12_matrix_train_own.json   (exp_e12_head_to_head.py matrix train, own agent)
Output: www-results/e15_train_refit.json; with --apply, writes the refit T and tau into
        www-results/probe_bank.json (mu and M are unchanged: they were fit on train and do not
        depend on T) and alpha into the refit file that exp_e12_head_to_head.py reads.

    .venv/bin/python exp_e15_train_refit.py [--apply]
"""
import json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moderator_v2 import BayesianModerator, ProbeBank, BANK_PATH, PROBE_BANK  # noqa: E402
from calibration import fit_temperature  # noqa: E402
from exp_probe_bank import sample_rows, fit_tau  # noqa: E402
from eval_common import ALPHAS, MAX_TURNS  # noqa: E402
from exp_e12_head_to_head import LOGS, load_items, to_records  # noqa: E402
from common import compute_metrics  # noqa: E402

OUT = os.path.join(LOGS, "e15_train_refit.json")


def _key(row):
    return (row.get("system_prompt", ""), row.get("text", ""), row.get("mode", ""))


def main(apply=False):
    recs = json.load(open(os.path.join(LOGS, "e12_matrix_train_own.json")))["records"]
    bank_d = json.load(open(BANK_PATH))
    old = {"prior_temperature": bank_d["prior_temperature"], "tau": bank_d["tau"],
           "alpha": 0.02}

    # T: the e5 procedure, on every train item
    T, T_loss = fit_temperature([{"p_raw": r["prior_raw"], "y_intent": r["y"]} for r in recs])

    # tau: the probe-bank procedure on its own train sample, given T
    id_of = {_key(r): r["_item_id"] for r in load_items("train")}
    sample, _ = sample_rows(random.Random(5))
    want = {id_of[_key(r)] for r in sample if _key(r) in id_of}
    pairs = [{"prior_raw": r["prior_raw"], "y": r["y"], "probe": q,
              "loglik": p["loglik"], "answer_tokens": p["answer_tokens"]}
             for r in recs if r["item_id"] in want for q, p in r["probes"].items()]
    tau, tau_loss, _ = fit_tau(pairs, bank_d["mu"], T)
    tau_oldT, _, _ = fit_tau(pairs, bank_d["mu"], old["prior_temperature"])  # reproduction check

    # alpha: replay the paper's configuration on train, pick the best F1_val
    bank = ProbeBank(dict(bank_d, prior_temperature=T, tau=tau))
    sweep = {}
    for a in ALPHAS:
        mod = BayesianModerator(bank, alpha=a, max_turns=MAX_TURNS, selection="fixed")
        rows = to_records(recs, [mod.moderate(record=r, rng=random.Random(3)) for r in recs])
        sweep[str(a)] = compute_metrics(rows)["val_f1"]
    alpha = max(ALPHAS, key=lambda a: sweep[str(a)])
    prior_rows = to_records(recs, [BayesianModerator(bank, alpha=1.0, max_turns=0)
                                   .moderate(record=r, rng=random.Random(3)) for r in recs])

    out = {"n_train_items": len(recs), "n_tau_pairs": len(pairs), "n_tau_rows": len(want),
           "previous": old,
           "refit": {"prior_temperature": T, "tau": tau, "alpha": alpha},
           "fit_loss": {"T_nll": T_loss, "tau_nll": tau_loss},
           "tau_with_previous_T": tau_oldT,
           "train_val_f1_by_alpha": sweep,
           "train_val_f1_prior": compute_metrics(prior_rows)["val_f1"]}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))

    if apply:
        bank_d["prior_temperature"], bank_d["tau"] = T, tau
        bank_d.setdefault("meta", {})["refit"] = (
            "prior_temperature, tau and alpha refit on train by exp_e15_train_refit.py; "
            f"previous T={old['prior_temperature']:.4f} (fit on ID test), tau={old['tau']:.4f}")
        json.dump(bank_d, open(BANK_PATH, "w"), indent=1)
        print(f"wrote T={T:.4f}, tau={tau:.4f} to {BANK_PATH}; alpha={alpha} in {OUT}")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
