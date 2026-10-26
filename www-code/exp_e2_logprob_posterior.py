"""The content prior: the moderator LLM's label distribution, read from one forward pass.

The moderator is asked for the most likely intent of the content; the top-k log-probabilities at
the position where the label starts are renormalized over the five intents. One call per item,
temperature 0. Temperature scaling of this distribution is in calibration.py.
"""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import INTENTS, _client, _NOTHINK_MODELS  # noqa: E402

MODEL = "Qwen/Qwen3-8B"
TOP_LOGPROBS = 20    # vLLM's cap for the OpenAI-compatible endpoint
VOTE_TEMP = 0.7

SYSTEM_PROMPT = (f"You are a bot moderator for Moltbook.\n"
                 f"Given a post or comment, output ONLY the most likely intent from: {INTENTS}")
K = len(INTENTS)
_SUFFIX = " /nothink" if MODEL in _NOTHINK_MODELS else ""


def _user(content, community):
    return (f"Community: {community}\nContent: {content}\n\n"
            f"Output ONLY the most likely intent from: {INTENTS}{_SUFFIX}")


def _match(tok):
    """Token -> index of the unique intent it prefixes, else None.

    The 5 intents' first tokens are distinct ('sub', 'n', 'spam', 'elic', 'organic') but a
    shorter candidate can straddle two of them ('s' prefixes both subtle_promotion and spam), so
    ambiguous tokens are dropped rather than guessed; the mass they carry is reported in
    `mass_captured`.
    """
    s = tok.strip().lower()
    if not s:
        return None
    hits = [i for i, t in enumerate(INTENTS) if t.startswith(s)]
    return hits[0] if len(hits) == 1 else None


def read_posterior(content, community):
    """-> dict with p_a (the label distribution over INTENTS), text, mass_captured, anchor_pos,
    tail_bound; None if no generated position resolves to an intent.

    /nothink still emits an empty think block, so the label is not at position 0. The anchor is
    the first generated position whose own token resolves to exactly one intent; `top_logprobs`
    there is the label distribution, renormalized over the 5 intents. top_logprobs is capped at
    20, so an intent absent from the list gets p=0; `tail_bound` = exp(smallest logprob
    returned) bounds any missing intent's mass.
    """
    r = _client(MODEL).chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": _user(content, community)}],
        temperature=0.0, max_tokens=24, logprobs=True, top_logprobs=TOP_LOGPROBS)
    ch = r.choices[0]
    text = (ch.message.content or "").strip()
    toks = ch.logprobs.content if ch.logprobs else []

    for pos, tl in enumerate(toks):
        if _match(tl.token) is None:
            continue
        lp = {}                       # intent idx -> summed probability mass
        seen = 0.0
        floor = 0.0
        for cand in tl.top_logprobs:
            p = math.exp(cand.logprob)
            seen += p
            floor = p                 # top_logprobs is returned in descending order
            j = _match(cand.token)
            if j is not None:
                lp[j] = lp.get(j, 0.0) + p
        if not lp:
            continue
        mass = sum(lp.values())
        p1 = [lp.get(j, 0.0) / mass for j in range(K)]
        raw = {j: math.log(v) for j, v in lp.items()}
        mx = max(raw.values())
        sc = {j: math.exp((v - mx) / VOTE_TEMP) for j, v in raw.items()}
        z = sum(sc.values())
        p07 = [sc.get(j, 0.0) / z for j in range(K)]
        return {"p_a": p1, "p_a07": p07, "text": text, "anchor_pos": pos,
                "mass_captured": mass / seen if seen else 0.0,
                "n_intents_seen": len(lp),
                "tail_bound": floor / mass if mass else 1.0}
    return None


def entropy(p):
    """Normalized entropy in [0, 1]."""
    return -sum(q * math.log(q) for q in p if q > 0) / math.log(K)


EPS = 0.1                            # uniform mixture weight for the smoothed NLL


def nll(p, y, eps=0.0):
    """-log p(y), optionally under the mixture (1-eps)*p + eps/K."""
    if eps:
        p = [(1 - eps) * q + eps / K for q in p]
    return -math.log(max(p[y], 1e-12))
