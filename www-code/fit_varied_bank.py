"""Fit Bot-Mod's calibration constants on train.

Reads the live training interrogations, scored per answer (exp_e16b_rescore.py run train varied
-> e16b_rescore_varied_train.json), and writes

    www-results/probe_bank_varied.json   per-technique base rates μ, τ (β = 1/τ), temperature T
    www-results/e17_varied_refit.json    T, τ and the stopping level α (best train F1_val)

T is the prior temperature fit on train (exp_e15_train_refit.py, stored in probe_bank.json).

    python fit_varied_bank.py
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_e16_multiturn as E  # noqa: E402
import exp_e16b_rescore as R  # noqa: E402
from moderator_v2 import ProbeBank  # noqa: E402
from exp_e12_head_to_head import LOGS  # noqa: E402


def main():
    T = ProbeBank.load().prior_temperature
    train = json.load(open(R.out_path("train", "varied")))["records"]
    bank, alpha, info = R.fit_variant(train, E.priors(), T)
    out = {"prior_temperature": bank.prior_temperature, "tau": bank.tau, "mu": bank.mu, "M": bank.M,
           "meta": {"protocol": "live agent session, varied wordings (exp_e16.VARIED), per-answer scoring",
                    "fit": "exp_e16b_rescore.py nohist on e16b_rescore_varied_train.json (train only)",
                    "alpha": alpha, "train_val_f1": info["train_val_f1"]}}
    json.dump(out, open(os.path.join(LOGS, "probe_bank_varied.json"), "w"), indent=1)
    json.dump({"refit": {"prior_temperature": T, "tau": bank.tau, "alpha": alpha},
               "source": "varied wordings, fit on varied train conversations"},
              open(os.path.join(LOGS, "e17_varied_refit.json"), "w"), indent=1)
    print(f"T={T:.4f} tau={bank.tau:.4f} (beta={1 / bank.tau:.2f}) alpha={alpha} train F1_val={info['train_val_f1']:.4f}")


if __name__ == "__main__":
    main()
