#!/usr/bin/env python3
"""Regenerate results-avg.tsv from results.tsv.

Groups the raw per-run rows in results.tsv into experiments and emits one row
per experiment with numeric columns averaged. A group is started by each
keep/discard row; subsequent `-`-status rows attach to a prior group when:
  - first col is `expN-...`      -> attach to that exp
  - first col is `<hash>-rerun*` -> attach to the group with that commit hash
Orphan `-` rows whose commit has no keep/discard row become their own group
(status `-`), matching the legacy behavior.
"""

import re
from pathlib import Path

NUM_COLS = ["val_f1", "f1_bin", "f1_cat", "val_f1_zs", "f1_zs",
            "f1_cat_zs", "f1_posts", "f1_comments", "time"]
OUT_COLS = ["exp", "commit", *NUM_COLS, "status", "description"]


def main() -> None:
    here = Path(__file__).resolve().parent
    src = here / "results.tsv.bak"
    dst = here / "results-avg.tsv"

    with src.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]

    col = {name: header.index(name) for name in
           ["commit", "status", "description", *NUM_COLS]}

    groups = []  # list of dicts
    by_commit = {}
    by_exp = {}
    next_exp = 0

    for r in rows:
        first = r[col["commit"]]
        status = r[col["status"]]

        if status in ("keep", "discard"):
            g = {"exp": f"exp{next_exp}", "hash": first, "rows": [r]}
            groups.append(g)
            by_commit[first] = len(groups) - 1
            by_exp[g["exp"]] = len(groups) - 1
            next_exp += 1
            continue

        # status == "-": attach to existing group or start orphan
        m_exp = re.match(r"^(exp\d+)(?:-|$)", first)
        m_rerun = re.match(r"^([0-9a-f]{7,})-rerun", first)
        target = None
        if m_exp and m_exp.group(1) in by_exp:
            target = by_exp[m_exp.group(1)]
        elif m_rerun and m_rerun.group(1) in by_commit:
            target = by_commit[m_rerun.group(1)]

        if target is not None:
            groups[target]["rows"].append(r)
        else:
            g = {"exp": f"exp{next_exp}", "hash": first, "rows": [r]}
            groups.append(g)
            by_commit[first] = len(groups) - 1
            by_exp[g["exp"]] = len(groups) - 1
            next_exp += 1

    lines = ["\t".join(OUT_COLS)]
    for g in groups:
        n = len(g["rows"])
        first = g["rows"][0]
        avg = {}
        for c in NUM_COLS:
            vals = [float(r[col[c]]) for r in g["rows"] if r[col[c]] != ""]
            avg[c] = sum(vals) / len(vals) if vals else float("nan")
        commit_str = f"{g['hash']} ({n})"
        out = [g["exp"], commit_str]
        for c in NUM_COLS:
            out.append(f"{avg[c]:.4f}" if c != "time" else f"{avg[c]:.1f}")
        out.append(first[col["status"]])
        out.append(first[col["description"]])
        lines.append("\t".join(out))

    dst.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(groups)} experiments to {dst}")


if __name__ == "__main__":
    main()
