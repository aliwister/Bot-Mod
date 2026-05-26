#!/usr/bin/env python3
"""Plot validation F1 progression over experiments from results-avg.tsv."""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text


def plot_val_f1_progression(tsv_path: str, out_prefix: str = "progress_avg") -> None:
    df = pd.read_csv(tsv_path, sep="\t")
    for col in ["val_f1", "f1_bin", "f1_cat", "val_f1_zs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["status"] = df["status"].str.strip().str.upper()

    fig, ax = plt.subplots(figsize=(16, 8))

    valid = df[df["status"].isin(["KEEP", "DISCARD"])].copy().reset_index(drop=True)
    baseline_f1 = valid.loc[0, "val_f1"]

    above = valid[valid["val_f1"] >= baseline_f1 - 0.0005]

    disc = above[above["status"] == "DISCARD"]
    ax.scatter(disc.index, disc["val_f1"],
               c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded")

    # Cleanup-only keeps (perf-neutral) — kept for code simplicity, not a real improvement
    cleanup_pat = r"simpler,?\s*equivalent|same\s+perf|equivalent\)"
    valid["is_cleanup"] = (
        valid["description"].fillna("").str.contains(cleanup_pat, case=False, regex=True)
    )

    kept_v = above[above["status"] == "KEEP"]
    ax.scatter(kept_v.index, kept_v["val_f1"],
               c="#2ecc71", s=50, zorder=4, label="Kept",
               edgecolors="black", linewidths=0.5)

    kept_mask = valid["status"] == "KEEP"
    kept_idx = valid.index[kept_mask]
    kept_f1 = valid.loc[kept_mask, "val_f1"]

    # Running best excludes cleanup-only keeps (their single-run val_f1 is noise,
    # not a real perf change — including them makes the frontier look like it regresses)
    eligible_mask = kept_mask & ~valid["is_cleanup"]
    eligible_idx = valid.index[eligible_mask]
    eligible_f1 = valid.loc[eligible_mask, "val_f1"]

    # Only points that actually raise the running max are on the frontier line
    cummax = eligible_f1.cummax()
    is_new_high = eligible_f1 > cummax.shift(1).fillna(-np.inf)
    frontier_idx = eligible_idx[is_new_high.to_numpy()]
    frontier_f1 = eligible_f1[is_new_high.to_numpy()]
    running_max = frontier_f1.cummax()

    step_x = np.append(frontier_idx, valid.index.max())
    step_y = np.append(running_max, running_max.iloc[-1])
    ax.step(step_x, step_y, where="post", color="#27ae60",
            linewidth=2, alpha=0.7, zorder=3, label="Running best")

    # Labels seeded below their point (hint to adjust_text to keep them under)
    force_below = {101}

    texts = []
    for idx, f1 in zip(frontier_idx, frontier_f1):
        desc = str(valid.loc[idx, "description"]).strip()
        exp_col = str(valid.loc[idx, "exp"]).strip().lstrip("exp") if "exp" in valid.columns else str(idx)
        wrapped = "\n".join(textwrap.wrap(desc, width=22))
        label = f"$\\mathbf{{exp\\ {exp_col}}}$\n{wrapped}"
        y0 = f1 - 0.03 if idx in force_below else f1
        va = "top" if idx in force_below else "baseline"
        t = ax.text(idx, y0, label, fontsize=12.0, color="#1a7a3a", va=va)
        texts.append(t)

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#1a7a3a", lw=0.8, alpha=0.5),
    )

    ax.set_xlabel("Experiment #", fontsize=18)
    ax.set_ylabel(r"F1$_{\mathrm{val}}$", fontsize=18)
    ax.legend(loc="lower right", fontsize=16, markerscale=2)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.2)

    span = frontier_f1.max() - baseline_f1
    ax.set_ylim(baseline_f1 - span * 0.15, 0.75)

    plt.tight_layout()
    png_path = f"{out_prefix}.png"
    pdf_path = f"{out_prefix}.pdf"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved to {png_path} and {pdf_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    tsv = repo_root / "results" / "results-avg.tsv"
    out = repo_root / "plots" / "progress"
    plot_val_f1_progression(str(tsv), str(out))
