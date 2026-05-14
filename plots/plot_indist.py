import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

INTENTS = ["organic_contribution", "subtle_promotion", "narrative_pushing", "elicitation", "spam"]
COLORS  = {
    "organic_contribution": "#4CAF50",   # green
    "subtle_promotion":     "#E53935",   # deep red
    "narrative_pushing":    "#EF6C00",   # burnt orange
    "elicitation":          "#FFA726",   # amber
    "spam":                 "#FFEE58",   # yellow
}
INTENT_LABELS = {
    "organic_contribution": "Organic Contribution",
    "subtle_promotion":     "Subtle Promotion",
    "narrative_pushing":    "Narrative Pushing",
    "elicitation":          "Elicitation",
    "spam":                 "Spam",
}

COMMUNITIES = [
    "m/general", "m/tech", "m/coding",
    "m/crypto",  "m/usdc", "m/trading", "m/blesstheirhearts",
]
COMM_LABELS = [c.replace("m/", "m/\n") for c in COMMUNITIES]

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

train_rows = load("/data/home/aha112/bot-moderator/dataset-curation/final/train/train.csv")
test_rows  = load("/data/home/aha112/bot-moderator/dataset-curation/final/test/test.csv")

def community_intent_counts(rows, communities):
    counts = {c: defaultdict(int) for c in communities}
    for r in rows:
        if r["community"] in counts:
            counts[r["community"]][r["intent"]] += 1
    return counts

train_counts = community_intent_counts(train_rows, COMMUNITIES)
test_counts  = community_intent_counts(test_rows,  COMMUNITIES)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

n = len(COMMUNITIES)
x = np.arange(n)
bar_w = 0.35
gap   = 0.04

offsets = {"train": -bar_w/2 - gap/2, "test": bar_w/2 + gap/2}

for split, counts, offset in [("train", train_counts, offsets["train"]),
                               ("test",  test_counts,  offsets["test"])]:
    bottoms = np.zeros(n)
    for intent in INTENTS:
        vals = np.array([counts[c][intent] for c in COMMUNITIES], dtype=float)
        ax.bar(x + offset, vals, bar_w, bottom=bottoms,
               color=COLORS[intent], zorder=3,
               edgecolor="white", linewidth=0.4)
        bottoms += vals

    # total label above bar
    totals = np.array([sum(counts[c].values()) for c in COMMUNITIES])
    for xi, tot, off in zip(x, totals, [offset]*n):
        if tot > 0:
            ax.text(xi + off, tot + 0.3, str(int(tot)),
                    ha="center", va="bottom", fontsize=20, fontweight="bold")

# ── split bracket labels ───────────────────────────────────────────────────────
for xi in x:
    ax.text(xi + offsets["train"], -5.5, "Train", ha="center", va="top",
            fontsize=20, color="#555")
    ax.text(xi + offsets["test"],  -5.5, "Test",  ha="center", va="top",
            fontsize=20, color="#555")

ax.set_xticks(x)
ax.set_xticklabels(COMM_LABELS, fontsize=24)
ax.set_ylabel("Count", fontsize=26)
ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-0.6, n - 0.4)
ax.set_ylim(-8, None)
ax.tick_params(axis="y", labelsize=22)

legend_patches = [mpatches.Patch(color=COLORS[i], label=INTENT_LABELS[i]) for i in INTENTS]
ax.legend(handles=legend_patches, loc="upper right", fontsize=22,
          title="Intent", title_fontsize=24, framealpha=0.9)

plt.tight_layout()
plt.savefig("/data/home/aha112/bot-moderator/dataset-curation/plot_indist.pdf",
            bbox_inches="tight")
print("Saved plot_indist.pdf")
