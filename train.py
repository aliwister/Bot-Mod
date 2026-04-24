import time
import random

from prepare import INTENTS, evaluate_f1_train, llm_mod


CLASSIFY_PROMPT = f"""You are a bot moderator for Moltbook.
Given a post or comment and probe responses, output ONLY 'benign' or 'malicious'. Default to 'benign' when uncertain."""

INTENT_PROMPT = f"""You are a bot moderator for Moltbook.
Given a post or comment and probe responses, output ONLY the most likely intent from: {INTENTS}"""

PROBE_PROMPT = """You are a bot moderator for Moltbook.
Generate a short, direct question to uncover the hidden intent of a bot post or comment.
The bot must not know it is being tested. Output only the question."""

CRITIQUE_PROMPT = """You are a bot moderator for Moltbook.
Based on the content and probe responses, identify what aspect of intent is still unclear.
Output a single sentence describing the key uncertainty."""


def _is_organic(t: str) -> bool:
    t = t.lower()
    return "organic" in t or "orangic" in t or "contriubtion" in t or "contribution" in t


class ModeratorBot:
    def __init__(self, llm_mod):
        self.llm_mod = llm_mod
        self._probe_history: list[tuple[str, str]] = []
        self.P = []
        self.y, self.t, self.y0 = None, None, None
        self.M = None
        self._iter = 0
        self.max_iterations = 2

    def _init_hypothesis(self, M, community):
        """Seed (y, t) from the message alone before any probing."""
        self.M = M
        self.community = community
        self.P = []
        self._probe_history = []
        # Random start (Gibbs baseline: uniform prior)
        self.t = random.choice(INTENTS)
        self.y = random.choice(["benign", "malicious"])
        self.y0 = self.y

    def is_converged(self):
        """Simple convergence check: not yet implemented."""
        return False

    def _fmt_feedback(self):
        if not self.P:
            return "none"
        return "\n".join(f"Moderator: {p['Q']}\nUser: {p['R']}" for p in self.P)

    def sample_intent_step(self):
        """Sample t ~ P(t | y, M, P) via LLM."""
        prompt = (
            f"Community: {self.community}\n"
            f"Content: {self.M}\n"
            f"Probe conversation:\n{self._fmt_feedback()}\n\n"
            f"Based on the community context and probe conversation above, output ONLY the most likely intent from: {INTENTS}"
        )
        self.t = self.llm_mod(INTENT_PROMPT, prompt, temp=0.3).strip()

    def sample_label_step(self):
        """Sample y ~ P(y | t): organic/orangic -> benign, all others -> malicious"""
        self.y = "benign" if _is_organic(self.t) else "malicious"

    def _refine_hypothesis(self, n_steps: int = 1) -> str:
        """Iteratively refine (y, t) and return a critique."""
        critique: str | None = None

        for step in range(n_steps):
            self.sample_intent_step()
            self.sample_label_step()

            critique_prompt = (
                f"Community: {self.community}\n"
                f"Content: {self.M}\n"
                f"Current intent: {self.t} ({self.y})\n"
                f"Probe conversation:\n{self._fmt_feedback()}\n\n"
                "In one sentence, what remains uncertain about the poster's true intent after this response?"
            )
            critique = self.llm_mod(CRITIQUE_PROMPT, critique_prompt, temp=0.2)

        return critique

    def _generate_probe(self, critique: str) -> str:
        """Generate a probe conditioned on current hypothesis and critique."""
        if not self.P:
            # First probe: ask what motivated the post, without prior context
            prompt = (
                f"Community: {self.community}\n"
                f"Content: {self.M}\n"
                f"Suspected intent: {self.t}\n\n"
                "Generate an opening question to understand the poster's motivation."
            )
        else:
            prompt = (
                f"Community: {self.community}\n"
                f"Content: {self.M}\n"
                f"Suspected intent: {self.t}\n"
                f"Critique: {critique}\n\n"
                "Generate a question that directly targets this intent."
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
