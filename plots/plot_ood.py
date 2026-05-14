import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

INTENTS = ["organic_contribution", "subtle_promotion", "narrative_pushing", "elicitation", "spam"]
COLORS  = {
    "organic_contribution": "#4CAF50",
    "subtle_promotion":     "#E53935",
    "narrative_pushing":    "#EF6C00",
    "elicitation":          "#FFA726",
    "spam":                 "#FFEE58",
}
INTENT_LABELS = {
    "organic_contribution": "Organic Contribution",
    "subtle_promotion":     "Subtle Promotion",
    "narrative_pushing":    "Narrative Pushing",
    "elicitation":          "Elicitation",
    "spam":                 "Spam",
}

COMMUNITIES = ["m/art", "m/philosophy", "m/politics", "m/travel", "m/consciousness", "m/shitposts"]
COMM_LABELS  = [c.replace("m/", "m/\n") for c in COMMUNITIES]

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

ood_rows = load("/data/home/aha112/bot-moderator/dataset-curation/final/ood/ood-posts.csv")

counts = {c: defaultdict(int) for c in COMMUNITIES}
for r in ood_rows:
    if r["community"] in counts:
        counts[r["community"]][r["intent"]] += 1

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 5))

n = len(COMMUNITIES)
x = np.arange(n)
bar_w = 0.65

bottoms = np.zeros(n)
for intent in INTENTS:
    vals = np.array([counts[c][intent] for c in COMMUNITIES], dtype=float)
    ax.bar(x, vals, bar_w, bottom=bottoms,
           color=COLORS[intent], zorder=3,
           edgecolor="white", linewidth=0.4)
    bottoms += vals

totals = np.array([sum(counts[c].values()) for c in COMMUNITIES])
for xi, tot in zip(x, totals):
    if tot > 0:
        ax.text(xi, tot + 0.3, str(int(tot)),
                ha="center", va="bottom", fontsize=1, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(COMM_LABELS, fontsize=20)
ax.set_ylabel("Count", fontsize=20)
ax.yaxis.grid(True, linestyle="--", alpha=0.45, linewidth=1.5, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=18)

legend_patches = [mpatches.Patch(color=COLORS[i], label=INTENT_LABELS[i]) for i in INTENTS]
ax.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.25),
          ncol=len(INTENTS), fontsize=18, framealpha=0.9, handletextpad=0.5, columnspacing=1.0)

plt.tight_layout()
plt.savefig("/data/home/aha112/bot-moderator/dataset-curation/plot_ood.pdf",
            bbox_inches="tight")
print("Saved plot_ood.pdf")
