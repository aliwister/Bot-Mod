"""Shared by every module: item text, the metric, and the test file lists."""
import json, os, sys
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import INTENTS  # noqa: E402
from paths import DATASET as _DS  # noqa: E402

_ALPHA = 0.7          # F1_val = F1_y^0.7 * F1_t^0.3


def _is_organic(t: str) -> bool:
    t = t.lower()
    return "organic" in t or "contribution" in t


def parse_intent(text: str) -> str:
    """Extract one of INTENTS from a free-form LLM response."""
    low = text.lower()
    for intent in INTENTS:
        if intent in low:
            return intent
    for intent in INTENTS:
        if intent.replace("_", " ") in low:
            return intent
    return low.strip()[:40]


def enrich_content(row):
    """The moderator-facing text. For comments, the parent post's title is prepended."""
    text = row.get("text", "").strip()
    mode = row.get("mode", "post").strip() or "post"
    if mode != "comment":
        return text, mode
    ctx = row.get("context", "")
    if isinstance(ctx, str) and ctx:
        try:
            ctx = json.loads(ctx)
        except Exception:                                      # noqa: BLE001
            ctx = None
    if isinstance(ctx, dict):
        parent_title = ctx.get("post", {}).get("title", "")
        if parent_title:
            return f"[POST] {parent_title}\n{text}", mode
    return text, mode


def _metric_split(results):
    """-> (F1_y, F1_t). F1_y: binary F1 of the malicious verdict. F1_t: macro F1 of the intent
    type over the items that are malicious and were flagged."""
    tp = sum(1 for r in results if r["verdict"] == "MALICIOUS" and r["intent_type"] == "MALICIOUS")
    fp = sum(1 for r in results if r["verdict"] == "MALICIOUS" and r["intent_type"] == "BENIGN")
    fn = sum(1 for r in results if r["verdict"] == "BENIGN" and r["intent_type"] == "MALICIOUS")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1_bin = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    mal_tp = [r for r in results if r["intent_type"] == "MALICIOUS" and r["verdict"] == "MALICIOUS"]
    if mal_tp:
        f1_cat = f1_score([r["intent"] for r in mal_tp], [r["t"] for r in mal_tp],
                          average="macro", zero_division=0.0)
    else:
        f1_cat = 0.0
    return f1_bin, f1_cat


def fpr(results):
    """False positive rate: benign items flagged malicious / benign items."""
    ben = [r for r in results if r["intent_type"] == "BENIGN"]
    return sum(1 for r in ben if r["verdict"] == "MALICIOUS") / len(ben) if ben else 0.0


def compute_metrics(results):
    f1_bin, f1_cat = _metric_split(results)
    val_f1 = (f1_bin ** _ALPHA) * (f1_cat ** (1 - _ALPHA))

    def per_mode(m):
        sub = [r for r in results if r["mode"] == m]
        if not sub:
            return 0.0
        b, c = _metric_split(sub)
        return (b ** _ALPHA) * (c ** (1 - _ALPHA))

    return {"val_f1": val_f1, "f1_bin": f1_bin, "f1_cat": f1_cat, "fpr": fpr(results),
            "f1_posts": per_mode("post"), "f1_comments": per_mode("comment")}


_ID_DIR = os.path.join(_DS, "id")
_OOD_DIR = os.path.join(_DS, "ood")
TEST_FILES = [os.path.join(_ID_DIR, f"{kind}-test-generated-{m}.json")
              for m in ("qwen", "mistral", "llama") for kind in ("posts", "comments")]
OOD_FILES = [os.path.join(_OOD_DIR, f"{kind}-ood-{kind}-generated-{m}.json")
             for m in ("qwen", "mistral", "llama") for kind in ("posts", "comments")]
