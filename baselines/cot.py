"""Chain-of-thought zero-shot baseline: single-pass classification with a
reasoning prelude ('think step by step') before the final label.

Isolates what CoT prompting alone contributes vs. pure zero-shot.
"""
import re
import sys

from baselines.utils._common import eval_files, files_for_mode, parse_intent, _is_organic

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from prepare import INTENTS, llm_mod  # noqa: E402


SYSTEM_PROMPT = (
    f"You are a bot moderator for Moltbook.\n"
    f"Given a post or comment, think step by step about the poster's likely intent, "
    f"then output the most likely intent from: {INTENTS}.\n"
    f'Finish your response with a line in the form: "Answer: <intent>"'
)


def _extract_final(text: str) -> str:
    """Prefer content after the last 'Answer:' marker; else use the last non-empty line."""
    m = re.search(r'(?i)answer\s*:\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text


def classify(content, community):
    user = (
        f"Community: {community}\n"
        f"Content: {content}\n\n"
        f"Reason briefly, then give your final answer on a new line as: Answer: <intent>"
    )
    resp = llm_mod(SYSTEM_PROMPT, user, temp=0.0)
    final = _extract_final(resp)
    t = parse_intent(final) if final else parse_intent(resp)
    y = "benign" if _is_organic(t) else "malicious"
    return y, t


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    eval_files(files_for_mode(mode), classify, label="cot")
