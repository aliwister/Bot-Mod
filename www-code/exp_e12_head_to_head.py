"""The main experiment: the content prior, Bot-Mod and Bot-Mod-judge on identical items.

Two stages:

    matrix <split>   [GPU]  one content-prior read per item (exp_e2_logprob_posterior) plus, for
                            every probe in the bank, a reset-agent answer and its K likelihoods.
                            The prior is what every later stage starts from; the answers feed the
                            train-side calibration (exp_probe_bank.py, exp_e15_train_refit.py).
                            -> www-results/e12_matrix_<split>_own.json
    score  <split>   [CPU]  replay Bot-Mod and the prior-only arm on the live interrogations
                            (exp_e17_live.py), add Bot-Mod-judge's verdicts, score every arm on
                            the common items with a paired item bootstrap.
                            -> www-results/e12_head_to_head_<split>.json

Splits: train, id, id_spam, ood, ood_spam; the paper's test sets are `id+spam` and `ood+spam`
(the two parts concatenated). Arms:

  bot-mod-v2         calibrated content prior, per-answer likelihood update, fixed probe order,
                     stopping level alpha from fit_varied_bank.py, at most 3 probes   (Ours)
  prior-only         the same posterior with zero questions
  fixed-bank-judge   Bot-Mod-judge: the same transcript read by the moderator LLM as a judge

Every number in the paper's main table that concerns these three arms comes from `score`.
"""
import json, os, random, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import INTENTS  # noqa: E402
from common import TEST_FILES, OOD_FILES, compute_metrics, _is_organic  # noqa: E402
from exp_e3_teacher_forced_likelihood import tok  # noqa: E402
from eval_common import collect  # noqa: E402
from moderator_v2 import PROBE_BANK, ProbeBank, BayesianModerator  # noqa: E402
from paths import RESULTS as LOGS, DATASET as _DS, TRAIN  # noqa: E402

TAG = "_own"                                   # each item answered by the model that wrote it
MTAG = "_varied_live"                          # Bot-Mod's live-session likelihoods (exp_e17_live.py)
BANK_VARIED = os.path.join(LOGS, "probe_bank_varied.json")
# Extra spam personas, test-only. Scored together with the original split as `id+spam` / `ood+spam`.
SPAM_FILES = {s: [os.path.join(_DS, f"{s}-spam", f"{kind}-spam-extra-{s}{mid}-generated-{m}.json")
                  for m in ("qwen", "mistral", "llama")
                  for kind, mid in (("posts", ""), ("comments", "-matched"))]
              for s in ("id", "ood")}
SPLITS = {"id": TEST_FILES, "ood": OOD_FILES,
          "id_spam": SPAM_FILES["id"], "ood_spam": SPAM_FILES["ood"],
          "train": [TRAIN]}                    # train only for fitting, never scored
# the stopping level, fit on train by fit_varied_bank.py
_REFIT = os.path.join(LOGS, "e17_varied_refit.json")
BEST_ALPHA = json.load(open(_REFIT))["refit"]["alpha"] if os.path.exists(_REFIT) else 0.02
MAX_TURNS = 3
N_BOOT = 400
WORKERS = 48
K = len(INTENTS)
ORGANIC = next(i for i, t in enumerate(INTENTS) if _is_organic(t))

# the moderator arms as kwargs, so replay and bootstrap cannot disagree on what an arm is
MOD_ARMS = {
    "prior-only": dict(alpha=1.0, max_turns=0, selection="eced"),
    "bot-mod-v2": dict(alpha=BEST_ALPHA, max_turns=MAX_TURNS, selection="fixed"),
}


def load_items(split):
    """The whole split, in file order, each row tagged with a stable item id."""
    items = []
    for p in SPLITS[split]:
        base = os.path.basename(p)
        for i, row in enumerate(json.load(open(p, encoding="utf-8"))):
            if row["label_intent"].strip().lower() not in INTENTS:
                continue
            items.append(dict(row, _src=base, _item_id=f"{base}:{i}"))
    return items


# ── stage 1: the content prior and the reset-agent answer matrix (GPU) ───────

def stage_matrix(split, limit=None, workers=WORKERS):
    out = f"{LOGS}/e12_matrix_{split}{TAG}.json"
    items = load_items(split)
    if limit:
        items = items[:limit]
    tok()
    t0 = time.time()
    print(f"[{split}] matrix: {len(items)} items x {len(PROBE_BANK)} probes = "
          f"{len(items) * len(PROBE_BANK)} answers, {workers} workers", flush=True)

    done = [0]

    def one(row):
        # a single bad item must not cost the whole hour
        try:
            rec = collect(row)
        except Exception as e:                                  # noqa: BLE001
            rec = None
            print(f"  ! {row['_item_id']}: {type(e).__name__}: {e}", flush=True)
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"  {done[0]}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
        if rec is not None:
            rec["item_id"] = row["_item_id"]
            rec["user_model"] = row.get("user_model", "")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        recs = [r for r in ex.map(one, items) if r is not None]
    full = [r for r in recs if len(r["probes"]) == len(PROBE_BANK)]
    json.dump({"config": {"split": split, "n_requested": len(items),
                          "n_collected": len(recs), "n_full": len(full),
                          "probes": list(PROBE_BANK), "seconds": time.time() - t0},
               "records": full}, open(out, "w"))
    print(f"[{split}] {len(full)}/{len(items)} items with all {len(PROBE_BANK)} probes "
          f"in {time.time() - t0:.0f}s -> {out}", flush=True)


# ── stage 2: score everything on the common items (no GPU) ───────────────────

def to_records(recs, traces):
    """Moderator traces -> the per-item record shape every arm shares."""
    return [{"item_id": r["item_id"], "verdict": t["verdict"],
             "intent_type": r["y_binary"], "intent": INTENTS[r["y"]],
             "t": t["intent"], "mode": r["mode"],
             "user_model": r.get("user_model", ""),
             "n_turns": t["n_turns"], "llm_calls": t["llm_calls"]}
            for r, t in zip(recs, traces) if t is not None]


def metrics(rows):
    m = compute_metrics(rows)
    m["n"] = len(rows)
    calls = [r["llm_calls"] for r in rows if "llm_calls" in r]
    turns = [r["n_turns"] for r in rows if "n_turns" in r]
    if calls:
        m["mean_llm_calls"] = sum(calls) / len(calls)
    if turns:
        m["mean_turns"] = sum(turns) / len(turns)
    m["acc_5way"] = sum(1 for r in rows if r["t"] == r["intent"]) / len(rows)
    return m


def paired_bootstrap(by_arm, a, b, rng, n_boot=N_BOOT):
    """val_f1(a) - val_f1(b), resampling items and scoring both arms on the draw."""
    ra, rb = by_arm[a], by_arm[b]
    n = len(ra)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = compute_metrics([ra[i] for i in idx])["val_f1"]
        db = compute_metrics([rb[i] for i in idx])["val_f1"]
        draws.append(da - db)
    draws.sort()
    return {"delta_mean": sum(draws) / len(draws),
            "ci95": [draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot) - 1]],
            "frac_positive": sum(1 for d in draws if d > 0) / n_boot}


def build_by_arm(split):
    """-> (by_arm, meta): every arm's per-row records, item-aligned on the common items."""
    # `id+spam` -> the parts `id` and `id_spam`, concatenated
    parts = [split[:-5], split[:-5] + "_spam"] if split.endswith("+spam") else [split]
    recs, n_full, part_arms = [], 0, []
    for p in parts:
        mat = json.load(open(f"{LOGS}/e12_matrix_{p}{MTAG}.json"))
        recs += mat["records"]
        n_full += mat["config"]["n_full"]
        part_arms.append(json.load(open(f"{LOGS}/e17_fbj_varied_live_{p}.json"))["arms"])
    base = {"arms": {k: [r for a in part_arms for r in a[k]] for k in part_arms[0]
                     if all(k in a for a in part_arms)}}
    bank = ProbeBank.load(BANK_VARIED)

    # the common item set: every arm scored on exactly the same rows
    ids = set(r["item_id"] for r in recs)
    for name, rows in base["arms"].items():
        ids &= set(r["item_id"] for r in rows)
    recs = [r for r in recs if r["item_id"] in ids]
    order = {r["item_id"]: i for i, r in enumerate(recs)}
    print(f"[{split}] common items: {len(ids)}  "
          f"(matrix {n_full}, "
          + ", ".join(f"{k} {len(v)}" for k, v in base["arms"].items()) + ")")

    by_arm = {}
    for name, cfg in MOD_ARMS.items():
        mod = BayesianModerator(bank, **cfg)
        traces = [mod.moderate(record=r, rng=random.Random(3)) for r in recs]
        by_arm[name] = to_records(recs, traces)
    for name, rows in base["arms"].items():
        rows = [r for r in rows if r["item_id"] in ids]
        rows.sort(key=lambda r: order[r["item_id"]])
        by_arm[name] = rows

    n = len(recs)
    assert all(len(v) == n for v in by_arm.values()), \
        {k: len(v) for k, v in by_arm.items()}
    assert all(all(x["item_id"] == y["item_id"] for x, y in zip(v, by_arm["bot-mod-v2"]))
               for v in by_arm.values()), "arms are not item-aligned"
    return by_arm, {"n": n, "tau": bank.tau,
                    "prior_temperature": bank.prior_temperature,
                    "n_full": n_full}


def stage_score(split):
    by_arm, meta = build_by_arm(split)
    n = meta["n"]
    rng = random.Random(19)
    out = {"config": {"split": split, "n_items": n, "alpha": BEST_ALPHA,
                      "max_turns": MAX_TURNS, "n_boot": N_BOOT, "tau": meta["tau"],
                      "prior_temperature": meta["prior_temperature"]},
           "label_distribution": dict(Counter(r["intent"] for r in by_arm["bot-mod-v2"])),
           "arms": {}, "deltas": {}, "by_mode": {}, "by_user_model": {}}

    for name, rows in by_arm.items():
        m = metrics(rows)
        m.setdefault("mean_llm_calls", 1.0 + 3.0)      # the judge: three agent turns + one read
        per_file = {}
        for src in sorted(set(r["item_id"].split(":")[0] for r in rows)):
            sub = [r for r in rows if r["item_id"].startswith(src + ":")]
            per_file[src] = compute_metrics(sub)["val_f1"]
        m["val_f1_per_file"] = per_file
        m["val_f1_file_mean"] = sum(per_file.values()) / len(per_file)
        out["arms"][name] = m
        for mode in ("post", "comment"):
            sub = [r for r in rows if r["mode"] == mode]
            if sub:
                out["by_mode"].setdefault(name, {})[mode] = compute_metrics(sub)["val_f1"]
        for um in sorted(set(r.get("user_model", "") for r in rows)):
            sub = [r for r in rows if r.get("user_model", "") == um]
            if sub and um:
                out["by_user_model"].setdefault(name, {})[um] = \
                    compute_metrics(sub)["val_f1"]

    for other in [a for a in by_arm if a != "bot-mod-v2"]:
        out["deltas"][f"bot-mod-v2 - {other}"] = \
            paired_bootstrap(by_arm, "bot-mod-v2", other, rng)

    json.dump(out, open(f"{LOGS}/e12_head_to_head_{split}.json", "w"), indent=2)

    # ── report ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 92}\n[{split.upper()}] {n} items, natural distribution: "
          f"{out['label_distribution']}\n{'=' * 92}")
    hdr = (f"{'arm':<20}{'F1_val':>9}{'F1_y':>9}{'F1_t':>9}{'FPR':>8}{'acc5':>8}"
           f"{'posts':>9}{'comments':>10}{'probes':>8}")
    print(hdr + "\n" + "-" * len(hdr))
    for name, m in sorted(out["arms"].items(), key=lambda kv: -kv[1]["val_f1"]):
        bm = out["by_mode"].get(name, {})
        print(f"{name:<20}{m['val_f1']:>9.4f}{m['f1_bin']:>9.4f}{m['f1_cat']:>9.4f}"
              f"{m['fpr']:>8.3f}{m['acc_5way']:>8.3f}{bm.get('post', 0):>9.4f}"
              f"{bm.get('comment', 0):>10.4f}{m.get('mean_turns', 3.0):>8.2f}")
    print(f"\npaired bootstrap over items, {N_BOOT} draws:")
    for k, d in out["deltas"].items():
        print(f"  {k:<34}{d['delta_mean']:>+9.4f}  "
              f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
              f"{d['frac_positive']:.0%} positive")
    print(f"\nwrote {LOGS}/e12_head_to_head_{split}.json")


if __name__ == "__main__":
    stage = sys.argv[1]
    split = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    if stage == "matrix":
        stage_matrix(split, limit)
    elif stage == "score":
        stage_score(split)
    else:
        sys.exit(f"unknown stage {stage!r}; want matrix|score")
