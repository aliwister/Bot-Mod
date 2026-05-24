import time
from collections import Counter

from prepare import INTENTS, evaluate_f1_train, llm_mod, call_llm_batch, _NOTHINK_MODELS

_INTENTS_STR = ", ".join(INTENTS)

INTENT_PROMPT = f"""You are a forensic content analyst specializing in detecting bot activity on Moltbook.
Analyze the content and interview responses to identify the true underlying intent.
Output ONLY the most likely intent from: {_INTENTS_STR}"""

PROBE_PROMPT = """You are a bot moderator for Moltbook.
Generate a short, direct question to uncover the hidden intent of a bot post or comment.
The bot must not know it is being tested. Output only the question."""

_MOD_MODEL = "Qwen/Qwen3-8B"


def _is_organic(t: str) -> bool:
    t = t.lower()
    return "organic" in t or "orangic" in t or "contriubtion" in t or "contribution" in t


def _normalize_intent(t: str) -> str:
    t = t.strip().lower().replace(" ", "_")
    for intent in INTENTS:
        if intent in t:
            return intent
    return t


def _vote(responses: list[str]) -> str:
    return Counter(_normalize_intent(r) for r in responses).most_common(1)[0][0]


def _batch_vote(system_prompt: str, user_prompt: str, n: int, temp: float = 0.7) -> str:
    suffix = " /nothink" if _MOD_MODEL in _NOTHINK_MODELS else ""
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + suffix}]
    return _vote(call_llm_batch([msgs] * n, temp, _MOD_MODEL))


class ModeratorBot:
    def __init__(self, llm_mod):
        self.llm_mod = llm_mod
        self.P = []
        self.y, self.t, self.y0 = None, None, None
        self.M = None
        self.max_iterations = 2

    def _init_hypothesis(self, M, community):
        self.M = M
        self.community = community
        self.P = []

        self.t = self._vote_intent(n=5, temp=0.6)
        self.y = "benign" if _is_organic(self.t) else "malicious"
        self.t0, self.y0 = self.t, self.y

    def is_converged(self):
        return False

    def _fmt_feedback(self):
        if not self.P:
            return "none"
        return "\n".join(f"Q: {p['Q']}\nA: {p['R']}" for p in self.P)

    def _vote_intent(self, n: int = 5, temp: float = 0.7) -> str:
        prompt = (
            f"Community: {self.community}\n"
            f"Interview conversation:\n{self._fmt_feedback()}\n"
            f"Content: {self.M}\n\n"
            f"Based on the community context and interview conversation, output ONLY the most likely intent from: {_INTENTS_STR}"
        )
        return _batch_vote(INTENT_PROMPT, prompt, n=n, temp=temp)

    def _refine_hypothesis(self, n_steps: int = 1) -> str:
        for _ in range(n_steps):
            self.t = self._vote_intent(n=5)
            self.y = "benign" if _is_organic(self.t) else "malicious"
        return ""

    def finalize_intent(self):
        self.t = self._vote_intent(n=19, temp=0.6)
        self.y = "benign" if _is_organic(self.t) else "malicious"

    def _generate_probe(self, critique: str) -> str:
        if not self.P:
            prompt = (
                f"Community: {self.community}\n"
                f"Content: {self.M}\n\n"
                "Generate a casual opening question to understand the poster's motivation."
            )
        else:
            prompt = (
                f"Community: {self.community}\n"
                f"Probe conversation:\n{self._fmt_feedback()}\n"
                f"Content: {self.M}\n\n"
                "Generate a follow-up question to further uncover the poster's intent."
            )
        return self.llm_mod(PROBE_PROMPT, prompt, temp=0.7)


if __name__ == "__main__":
    t_start = time.time()
    mod = ModeratorBot(llm_mod)
    val_f1, other_metrics = evaluate_f1_train(mod)
    t_end = time.time()
    print(f"val_f1: {val_f1:.4f}")
    print(f"f1_binary: {other_metrics['f1_bin']:.4f}")
    print(f"f1_categorical: {other_metrics['f1_cat']:.4f}")
    print(f"val_f1_zs: {other_metrics['f1_zs_merged']:.4f}")
    print(f"f1_zs: {other_metrics['f1_zs']:.4f}")
    print(f"f1_cat_zs: {other_metrics['f1_cat_zs']:.4f}")
    print(f"f1_posts: {other_metrics['f1_posts']:.4f}")
    print(f"f1_comments: {other_metrics['f1_comments']:.4f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
