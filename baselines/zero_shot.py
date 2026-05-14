"""Zero-shot baseline: single greedy LLM call, no reasoning, no self-critique, no voting.

Pure one-shot classification of content -> intent.
"""
import sys

from baselines.utils._common import eval_files, files_for_mode, parse_intent, _is_organic

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from prepare import INTENTS, llm_mod  # noqa: E402


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
    resp = llm_mod(SYSTEM_PROMPT, user, temp=0.0)
    t = parse_intent(resp)
    y = "benign" if _is_organic(t) else "malicious"
    return y, t


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    eval_files(files_for_mode(mode), classify, label="zero_shot")
