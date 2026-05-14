"""Self-consistency baseline: sample N intent predictions and majority vote.

Isolates what self-consistency voting alone contributes, with no probing/critique.
"""
import sys
from collections import Counter

from baselines.utils._common import eval_files, files_for_mode, parse_intent, _is_organic

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from prepare import INTENTS, llm_mod  # noqa: E402

N_SAMPLES = 11
TEMP = 0.7

SYSTEM_PROMPT = (
    f"You are a bot moderator for Moltbook.\n"
    f"Given a post or comment, output ONLY the most likely intent from: {INTENTS}"
)


def classify(content, community):
    user = (
        f"Community: {community}\n"
        f"Content: {content}\n\n"
        f"Output ONLY the most likely intent from: {INTENTS}"
    )
    votes = []
    for _ in range(N_SAMPLES):
        r = llm_mod(SYSTEM_PROMPT, user, temp=TEMP)
        votes.append(parse_intent(r))
    t = Counter(votes).most_common(1)[0][0]
    y = "benign" if _is_organic(t) else "malicious"
    return y, t


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    eval_files(files_for_mode(mode), classify, label="self_consistency")
