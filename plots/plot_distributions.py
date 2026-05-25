import csv
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── data ──────────────────────────────────────────────────────────────────────
SPLITS = {
    "Train": "/data/home/aha112/bot-moderator/dataset-curation/final/train/train.csv",
    "Test":  "/data/home/aha112/bot-moderator/dataset-curation/final/test/test.csv",
    "OOD":   "/data/home/aha112/bot-moderator/dataset-curation/final/ood/ood-posts.csv",
}

INTENTS = ["organic_contribution", "subtle_promotion", "narrative_pushing", "elicitation", "spam"]
INTENT_LABELS = ["Organic\nContribution", "Subtle\nPromotion", "Narrative\nPushing", "Elicitation", "Spam"]

data = {}
for split, path in SPLITS.items():
    with open(path) as f:
        rows = list(csv.DictReader(f))
    intent_cnt   = Counter(r["intent"]      for r in rows)
    benign_cnt   = Counter(r["intent"]      for r in rows if r["intent_type"] == "Benign")
    malicious_cnt = Counter(r["intent"]     for r in rows if r["intent_type"] == "Malicious")
    data[split] = {
        "intent":    intent_cnt,
        "benign":    benign_cnt,
        "malicious": malicious_cnt,
        "total":     len(rows),
    }

# ── layout ────────────────────────────────────────────────────────────────────
SPLIT_NAMES = list(SPLITS.keys())
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
fig.suptitle("Intent & Intent-Type Distributions by Split", fontsize=13, fontweight="bold", y=1.01)

COLORS = {"Benign": "#4C9BE8", "Malicious": "#E8694C"}

x = np.arange(len(INTENTS))
bar_w = 0.55

for ax, split in zip(axes, SPLIT_NAMES):
    d = data[split]
    benign_vals   = [d["benign"].get(i, 0)    for i in INTENTS]
    malicious_vals = [d["malicious"].get(i, 0) for i in INTENTS]

    bars_b = ax.bar(x, benign_vals,   bar_w, color=COLORS["Benign"],    label="Benign",    zorder=3)
    bars_m = ax.bar(x, malicious_vals, bar_w, bottom=benign_vals,
                    color=COLORS["Malicious"], label="Malicious", zorder=3)

    # value labels
    for bar, bv, mv in zip(bars_b, benign_vals, malicious_vals):
        total = bv + mv
        if total > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, total + 1,
                    str(total), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title(f"{split}  (n={d['total']})", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(INTENT_LABELS, fontsize=8)
    ax.set_ylabel("Count" if split == "Train" else "", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

# shared legend
legend_patches = [
    mpatches.Patch(color=COLORS["Benign"],    label="Benign"),
    mpatches.Patch(color=COLORS["Malicious"], label="Malicious"),
]
fig.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.02, 1.0),
           fontsize=9, title="Intent Type", title_fontsize=9)

plt.tight_layout()
plt.savefig("/data/home/aha112/bot-moderator/dataset-curation/distributions.png",
            dpi=150, bbox_inches="tight")
print("Saved distributions.png")
