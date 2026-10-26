"""Shared by the experiments: the stopping-level grid and one-item answer collection with every
probe in the bank (the agent is reset between probes; used for the content prior and the
train-side calibration)."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import INTENTS  # noqa: E402
from common import enrich_content, _is_organic  # noqa: E402
from exp_e2_logprob_posterior import read_posterior  # noqa: E402
from exp_e3_teacher_forced_likelihood import elicit, loglik  # noqa: E402
from moderator_v2 import PROBE_BANK  # noqa: E402

ALPHAS = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001]
MAX_TURNS = 3
K = len(INTENTS)
ORGANIC = next(i for i, t in enumerate(INTENTS) if _is_organic(t))


def collect(row):
    """One item: the raw prior plus, for every probe, the answer and K logliks."""
    content, mode = enrich_content(row)
    community = row["community"].strip()
    lp = read_posterior(content, community)
    if lp is None:
        return None
    rec = {"y": INTENTS.index(row["label_intent"].strip().lower()),
           "y_binary": row["label_binary"].strip().upper(),
           "mode": mode, "community": community, "src": row["_src"],
           "prior_raw": lp["p_a"], "probes": {}}
    for probe, text in PROBE_BANK.items():
        answer = elicit(row, text)
        if not answer:
            continue
        lls, n = [], 0
        ok = True
        for h in INTENTS:
            ll, nt = loglik(content, community, text, answer, h)
            if ll is None:
                ok = False
                break
            lls.append(ll)
            n = nt
        if ok:
            rec["probes"][probe] = {"loglik": lls, "answer_tokens": n,
                                    "answer_chars": len(answer)}
    return rec if rec["probes"] else None
