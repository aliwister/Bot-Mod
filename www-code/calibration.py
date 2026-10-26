"""Temperature scaling of the content prior: p^(1/T), and T fit by NLL on train."""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_e2_logprob_posterior import nll, EPS  # noqa: E402


def temper(p, t):
    """p^(1/t), renormalized. t > 1 softens, t < 1 sharpens."""
    lg = [math.log(max(v, 1e-12)) / t for v in p]
    mx = max(lg)
    ex = [math.exp(v - mx) for v in lg]
    z = sum(ex)
    return [v / z for v in ex]


def fit_temperature(items, lo=0.25, hi=25.0, iters=60):
    """T minimizing smoothed NLL, by golden-section search on a unimodal 1-D loss."""
    def loss(t):
        return sum(nll(temper(it["p_raw"], t), it["y_intent"], eps=EPS)
                   for it in items) / len(items)
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if loss(c) < loss(d):
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
    t = (a + b) / 2
    return t, loss(t)
