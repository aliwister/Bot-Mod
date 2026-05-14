"""Shared eval utilities for baselines. Output format matches eval_run.py."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import f1_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare import INTENTS  # noqa: E402

_ALPHA = 0.7


def _is_organic(t: str) -> bool:
    t = t.lower()
    return "organic" in t or "contribution" in t


def parse_intent(text: str) -> str:
    """Extract one of INTENTS from a free-form LLM response. Fallback: the raw text (will miss as a category)."""
    low = text.lower()
    for intent in INTENTS:
        if intent in low:
            return intent
    for intent in INTENTS:
        spaced = intent.replace("_", " ")
        if spaced in low:
            return intent
    return low.strip()[:40]


def enrich_content(row):
    """Return moderator-facing text. For comments, prepend parent post title."""
    text = row.get("text", "").strip()
    mode = row.get("mode", "post").strip() or "post"
    if mode != "comment":
        return text, mode
    ctx = row.get("context", "")
    if isinstance(ctx, str) and ctx:
        try:
            ctx = json.loads(ctx)
        except Exception:
            ctx = None
    if isinstance(ctx, dict):
        parent_title = ctx.get("post", {}).get("title", "")
        if parent_title:
            return f"[POST] {parent_title}\n{text}", mode
    return text, mode


def _metric_split(results):
    tp = sum(1 for r in results if r["verdict"] == "MALICIOUS" and r["intent_type"] == "MALICIOUS")
    fp = sum(1 for r in results if r["verdict"] == "MALICIOUS" and r["intent_type"] == "BENIGN")
    fn = sum(1 for r in results if r["verdict"] == "BENIGN"    and r["intent_type"] == "MALICIOUS")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1_bin = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    mal_tp = [r for r in results if r["intent_type"] == "MALICIOUS" and r["verdict"] == "MALICIOUS"]
    if mal_tp:
        y_true = [r["intent"] for r in mal_tp]
        y_pred = [r["t"] for r in mal_tp]
        f1_cat = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
    else:
        f1_cat = 0.0
    return f1_bin, f1_cat


def compute_metrics(results):
    f1_bin, f1_cat = _metric_split(results)
    val_f1 = (f1_bin ** _ALPHA) * (f1_cat ** (1 - _ALPHA))

    def per_mode(m):
        sub = [r for r in results if r["mode"] == m]
        if not sub:
            return 0.0
        b, c = _metric_split(sub)
        return (b ** _ALPHA) * (c ** (1 - _ALPHA))

    return {
        "val_f1": val_f1,
        "f1_bin": f1_bin,
        "f1_cat": f1_cat,
        "f1_posts": per_mode("post"),
        "f1_comments": per_mode("comment"),
    }


def run_eval(classify_fn, filename, max_workers=64):
    """classify_fn(content, community) -> (verdict, intent)"""
    with open(filename, encoding="utf-8") as f:
        rows = json.load(f)

    def process(args):
        idx, row = args
        community = row["community"].strip()
        truth     = row["label_binary"].strip().lower()
        intent    = row["label_intent"].strip().lower()
        content, mode = enrich_content(row)
        verdict, t = classify_fn(content, community)
        return {
            "verdict": verdict.upper(),
            "intent_type": truth.upper(),
            "intent": intent.lower(),
            "t": t.lower(),
            "mode": mode,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(process, enumerate(rows)))
    return results


def eval_files(files_with_split, classify_fn, label):
    """Iterate files, print per-file metrics matching eval_run.py format."""
    for split, path in files_with_split:
        name = path.split("/")[-1]
        print(f"\n{'='*60}\n[{split}] {name}\n{'='*60}", flush=True)
        t0 = time.time()
        results = run_eval(classify_fn, path)
        m = compute_metrics(results)
        elapsed = time.time() - t0
        print(f"val_f1:          {m['val_f1']:.4f}")
        print(f"f1_binary:       {m['f1_bin']:.4f}")
        print(f"f1_categorical:  {m['f1_cat']:.4f}")
        print(f"f1_posts:        {m['f1_posts']:.4f}")
        print(f"f1_comments:     {m['f1_comments']:.4f}")
        print(f"time:            {elapsed:.1f}s", flush=True)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ID_DIR  = os.path.join(_REPO_ROOT, "data", "dataset", "id")
_OOD_DIR = os.path.join(_REPO_ROOT, "data", "dataset", "ood")

TEST_FILES = [
    os.path.join(_ID_DIR, "posts-test-generated-qwen.json"),
    os.path.join(_ID_DIR, "comments-test-generated-qwen.json"),
    os.path.join(_ID_DIR, "posts-test-generated-mistral.json"),
    os.path.join(_ID_DIR, "comments-test-generated-mistral.json"),
    os.path.join(_ID_DIR, "posts-test-generated-llama.json"),
    os.path.join(_ID_DIR, "comments-test-generated-llama.json"),
]
OOD_FILES = [
    os.path.join(_OOD_DIR, "posts-ood-posts-generated-qwen.json"),
    os.path.join(_OOD_DIR, "comments-ood-comments-generated-qwen.json"),
    os.path.join(_OOD_DIR, "posts-ood-posts-generated-mistral.json"),
    os.path.join(_OOD_DIR, "comments-ood-comments-generated-mistral.json"),
    os.path.join(_OOD_DIR, "posts-ood-posts-generated-llama.json"),
    os.path.join(_OOD_DIR, "comments-ood-comments-generated-llama.json"),
]


def files_for_mode(mode):
    files = []
    if mode in ("test", "both"):
        files += [("TEST", p) for p in TEST_FILES]
    if mode in ("ood", "both"):
        files += [("OOD", p) for p in OOD_FILES]
    return files
