"""Self-Refine baseline (Madaan et al., 2023): produce an initial prediction,
generate one self-feedback critique, then revise.

Isolates what one shot of self-critique contributes, without iterative probing or
a user loop. Three LLM calls per item: initial, feedback, refine.
"""
import sys

from baselines.utils._common import eval_files, files_for_mode, parse_intent, _is_organic

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from prepare import INTENTS, llm_mod  # noqa: E402


INIT_SYSTEM = (
    f"You are a bot moderator for Moltbook.\n"
    f"Given a post or comment, output ONLY the most likely intent from: {INTENTS}"
)

FEEDBACK_SYSTEM = (
    "You are a careful reviewer. Given content and an initial intent label, "
    "critique the label: identify any evidence in the content that contradicts it "
    "or suggests a different intent. Be concise (one or two sentences)."
)

REFINE_SYSTEM = (
    f"You are a bot moderator for Moltbook.\n"
    f"Given content, an initial intent label, and reviewer feedback, output ONLY "
    f"the final most likely intent from: {INTENTS}. "
    f"Take the feedback into account; you may keep or change the label."
)


def classify(content, community):
    base = f"Community: {community}\nContent: {content}\n\n"

    init_resp = llm_mod(INIT_SYSTEM, base + f"Output ONLY the most likely intent from: {INTENTS}", temp=0.0)
    t0 = parse_intent(init_resp)

    fb_user = (
        f"{base}Initial intent: {t0}\n\n"
        "Critique the initial intent in one or two sentences. "
        "Does the content support it, or is a different intent more likely?"
    )
    feedback = llm_mod(FEEDBACK_SYSTEM, fb_user, temp=0.3)

    ref_user = (
        f"{base}Initial intent: {t0}\n"
        f"Reviewer feedback: {feedback}\n\n"
        f"Output ONLY the final most likely intent from: {INTENTS}"
    )
    ref_resp = llm_mod(REFINE_SYSTEM, ref_user, temp=0.0)
    t = parse_intent(ref_resp)

    y = "benign" if _is_organic(t) else "malicious"
    return y, t


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    eval_files(files_for_mode(mode), classify, label="self_refine")
