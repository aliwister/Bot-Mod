"""Bot-Mod: the moderator as a posterior over intents.

  prior      P_0(t) ∝ M(t | x, c)^(1/T)        the moderator LLM's label distribution for the
                                               content, temperature-scaled (T fit on train)
  update     P_k(t) ∝ P_{k-1}(t) exp(β [ℓ_k(t)/|a_k| − μ_q(t)])
                                               the agent's answer a_k to probe q scored by its
                                               per-token log-likelihood under a persona prompt for
                                               each intent t, minus the base rate μ_q(t) estimated
                                               on train from deranged answers; β = 1/τ fit on train
  stopping   stop when max(P(benign), P(malicious)) ≥ 1 − α, or after max_turns probes
  decision   verdict from the class marginal; intent type = argmax within the chosen class

`moderate(row=...)` runs the interrogation live; `moderate(record=...)` replays the same
selection, update, stopping and decision code on stored likelihoods (the per-item records the
collection stages write), which is how every arm in the paper is scored on identical answers.
"""
import json, math, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import INTENTS  # noqa: E402
from common import enrich_content, _is_organic  # noqa: E402
from exp_e2_logprob_posterior import read_posterior, entropy  # noqa: E402
from exp_e3_teacher_forced_likelihood import (  # noqa: E402
    PROBES as E3_PROBES, elicit, loglik, tok)
from calibration import temper  # noqa: E402

K = len(INTENTS)
ORGANIC = next(i for i, t in enumerate(INTENTS) if _is_organic(t))
MALICIOUS = [j for j in range(K) if j != ORGANIC]
from paths import RESULTS as _LOGS  # noqa: E402
BANK_PATH = os.path.join(_LOGS, "probe_bank.json")   # the six-probe bank fit on train (exp_probe_bank.py, exp_e15_train_refit.py)


# ── the probe bank ───────────────────────────────────────────────────────────
# The three techniques of exp_e3 plus three more, so that adaptive probe selection has
# something to select between. Each is one named technique from the deception-detection
# literature and targets a different intent's weak point. The paper's protocol asks the
# first three in fixed order (selection="fixed"); the six-probe bank is used for the
# content-prior matrix and the train-side temperature fit.

PROBE_BANK = dict(E3_PROBES)
PROBE_BANK.update({
    # Vrij et al. 2009, unanticipated questions -- the competitor variant, aimed at
    # subtle_promotion: a genuine participant can name a rival freely, a promoter
    # cannot without working against its own objective.
    "UNANTICIPATED_COMPETITOR": (
        "Name one case where you would advise someone here *against* the thing you "
        "have been recommending, and name a better alternative for that case. Be "
        "specific about the alternative."),
    # Falsification / devil's advocate, aimed at narrative_pushing: state the
    # strongest case against your own position and what would change your mind.
    "FALSIFICATION": (
        "Give me the strongest argument against the position you have been putting "
        "forward here, and tell me what specific evidence would make you abandon it."),
    # Aimed at elicitation: ask what the asker will do with the answer. An
    # information-gathering agent has to either reveal the purpose or refuse.
    "PURPOSE_OF_ASK": (
        "You have been asking questions in this community. Tell me exactly what you "
        "do with the answers you get: where do they go, and who else sees them?"),
})


# ── the observation model, fit offline on train ──────────────────────────────

class ProbeBank:
    """Per-probe base rates mu_h, observation model M_q, and the fitted taus.

    M_q[h][o] = P(the calibrated likelihood's argmax is o | the true intent is h),
    estimated by counting on train with Laplace smoothing. This is the *noisy*
    observation model that makes the selection objective ECED-shaped rather than
    EC2-shaped: probes do not cut hypotheses cleanly, they reweight them.
    """

    def __init__(self, d):
        self.mu = d["mu"]                  # probe -> [K] per-token base rate
        self.M = d["M"]                    # probe -> [K][K] observation model
        self.tau = d["tau"]
        self.prior_temperature = d["prior_temperature"]
        self.probes = list(self.M)
        self.meta = d.get("meta", {})

    @classmethod
    def load(cls, path=BANK_PATH):
        return cls(json.load(open(path)))

    def calibrated_evidence(self, probe, lls, n_tokens):
        """Raw summed logliks -> per-token, base-rate-corrected evidence in nats."""
        mu = self.mu[probe]
        n = max(n_tokens, 1)
        return [(lls[j] - mu[j] * n) / n for j in range(K)]


# ── the moderator ────────────────────────────────────────────────────────────

def class_posterior(b):
    """-> P(malicious). The equivalence partition the metric actually cares about."""
    return sum(b[j] for j in MALICIOUS)


def cross_class_weight(b):
    """EC2's objective: total probability mass on benign-malicious hypothesis pairs.

    Zero exactly when b is confined to one class, so driving it down *is* deciding
    the binary question, and splitting within a class earns nothing -- which is the
    property EC2 has and plain information gain does not.
    """
    return b[ORGANIC] * sum(b[j] for j in MALICIOUS)


class BayesianModerator:
    def __init__(self, bank, alpha=0.1, max_turns=3, min_turns=0,
                 selection="eced", verbose=False):
        self.bank = bank
        self.alpha = alpha              # MSPRT: stop at class confidence >= 1-alpha
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.selection = selection      # "eced" | "fixed" | "random" (for ablations)
        self.verbose = verbose

    # ── step 1: the calibrated prior, one call ──────────────────────────────
    def prior(self, content, community):
        lp = read_posterior(content, community)
        if lp is None:
            return None
        return temper(lp["p_a"], self.bank.prior_temperature)

    # ── step 3: probe selection, zero calls ────────────────────────────────
    def expected_residual_weight(self, b, probe):
        """E_o[ w(b|o) ] under the fitted observation model. Lower is better."""
        M = self.bank.M[probe]
        tot = 0.0
        for o in range(K):
            # P(o) = sum_h b(h) M[h][o];  b|o (h) ∝ b(h) M[h][o]
            joint = [b[h] * M[h][o] for h in range(K)]
            p_o = sum(joint)
            if p_o <= 0:
                continue
            tot += p_o * cross_class_weight([v / p_o for v in joint])
        return tot

    def select_probe(self, b, asked, rng=None):
        avail = [q for q in self.bank.probes if q not in asked]
        if not avail:
            return None
        if self.selection == "fixed":
            return avail[0]                       # bank order, no adaptation
        if self.selection == "random":
            return avail[rng.randrange(len(avail))]
        return min(avail, key=lambda q: self.expected_residual_weight(b, q))

    # ── step 4: the Bayesian update on the answer's likelihood ─────────────
    def update(self, b, content, community, probe, answer, scored=None):
        """`scored` = precomputed (lls, n_tokens), for offline policy evaluation.

        The live path scores the answer with K prefills here. Evaluation instead
        precomputes the full (item x probe) matrix once and replays policies against it.

        Replay is **not** bit-exact and must not be described as such. Measured on this
        server (measured in the stability study): generation at temperature 0 is not
        reproducible -- continuous batching perturbs the logits enough that a
        200-token answer occasionally diverges mid-sequence into a substantively
        different answer -- and re-scoring a *fixed* answer moves the summed loglik by
        up to ~1.3 nats. The likelihood jitter is harmless (evidence is per-token, so
        it is ~0.07 in the log-posterior), but a resampled answer is not.

        What replay actually buys is a *paired* comparison: every policy sees the same
        realization of each answer, so a difference between two arms is attributable to
        the policy rather than to answer-sampling noise. That is common random numbers,
        the standard variance-reduction device for simulation comparison -- stronger
        than independent live runs for ranking arms, not a claim that the live run
        would have reproduced the trace. Run-to-run variability is reported separately
        rather than assumed away. The call count still comes from the trace, i.e. from
        what the policy *would* have spent online.
        """
        if scored is not None:
            lls, n = scored
        else:
            lls, n = [], 0
            for h in INTENTS:
                ll, nt = loglik(content, community, PROBE_BANK[probe], answer, h)
                if ll is None:
                    return b, None
                lls.append(ll)
                n = nt
        ev = self.bank.calibrated_evidence(probe, lls, n)
        tau = max(self.bank.tau, 1e-6)
        logb = [math.log(max(b[j], 1e-12)) + ev[j] / tau for j in range(K)]
        mx = max(logb)
        ex = [math.exp(v - mx) for v in logb]
        z = sum(ex)
        return [v / z for v in ex], {"evidence": ev, "answer_tokens": n}

    # ── step 5: the stopping rule ──────────────────────────────────────────
    def should_stop(self, b, turn):
        if turn < self.min_turns:
            return False
        p_mal = class_posterior(b)
        return max(p_mal, 1.0 - p_mal) >= 1.0 - self.alpha

    # ── step 6: the decision, coherent by construction ─────────────────────
    def decide(self, b):
        """Verdict from the class marginal; intent from within the chosen class.

        The binary call uses the whole malicious mass, and the subtype is only ever named
        *within* the class that was chosen, so verdict and intent can never contradict each other.
        """
        if class_posterior(b) < 0.5:
            return "BENIGN", INTENTS[ORGANIC]
        return "MALICIOUS", INTENTS[max(MALICIOUS, key=lambda j: b[j])]

    # ── the loop ───────────────────────────────────────────────────────────
    def moderate(self, row=None, rng=None, record=None):
        """Run the interrogation. Live when given `row`; replayed when given `record`.

        `record` is one entry of the precomputed matrix: {prior_raw, mode, community,
        probes: {name: {loglik, answer_tokens}}}. Both paths run the *same* selection,
        update, stopping and decision code -- only the source of the likelihoods
        differs -- so an ablation cannot accidentally measure a different algorithm.
        See `update` for what replay does and does not guarantee.
        """
        if record is not None:
            content, mode = None, record["mode"]
            community = record["community"]
            b = temper(record["prior_raw"], self.bank.prior_temperature)
        else:
            content, mode = enrich_content(row)
            community = row["community"].strip()
            b = self.prior(content, community)
        if b is None:
            return None
        trace = {"prior": list(b), "mode": mode, "turns": [],
                 "community": community}
        asked = []
        turn = 0
        while turn < self.max_turns and not self.should_stop(b, turn):
            # in replay, a probe missing from the matrix is treated as unavailable
            # rather than silently scored from another probe's likelihoods
            missing = [q for q in self.bank.probes if q not in record["probes"]] \
                if record is not None else []
            probe = self.select_probe(b, asked + missing, rng)
            if probe is None:
                break
            if record is not None:
                pr = record["probes"][probe]
                scored = (pr["loglik"], pr["answer_tokens"])
                answer = True
            else:
                scored = None
                answer = elicit(row, PROBE_BANK[probe])
            if not answer:
                break
            asked.append(probe)
            b_new, info = self.update(b, content, community, probe, answer,
                                      scored=scored)
            turn += 1
            trace["turns"].append({
                "probe": probe, "answer_tokens": info["answer_tokens"] if info else 0,
                "entropy": entropy(b_new), "p_malicious": class_posterior(b_new),
                "expected_residual_weight": self.expected_residual_weight(b, probe),
            })
            b = b_new
        verdict, intent = self.decide(b)
        trace.update({"posterior": list(b), "n_turns": turn, "asked": asked,
                      "verdict": verdict, "intent": intent,
                      "stopped_early": turn < self.max_turns,
                      # calls: 1 prior + per turn (1 elicit + K prefills)
                      "llm_calls": 1 + turn * (1 + K)})
        return trace
