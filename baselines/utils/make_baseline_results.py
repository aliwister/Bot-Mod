"""Parse logs/baselines/*/run*.log and emit baseline_results.csv."""
import re
import os
from pathlib import Path

BASELINES_ORDER = ["zero_shot", "self_consistency", "cot", "self_refine"]
LLMS = ["qwen", "mistral", "llama"]
LOG_ROOT = Path("/run/user/1000/autoresearch3/logs/baselines")
OUT = Path("/run/user/1000/autoresearch3/baseline_results.csv")


def parse_log(path):
    """Return {(posts|comments, llm): {metric: value}}."""
    out = {}
    text = path.read_text()
    blocks = re.split(r"={60}\n\[(TEST|OOD)\] (\S+)\n={60}", text)
    for i in range(1, len(blocks) - 2, 3):
        split = blocks[i]
        fname = blocks[i + 1]
        body = blocks[i + 2]
        kind = "posts" if fname.startswith("posts") else "comments"
        llm = next((l for l in LLMS if l in fname), None)
        metrics = {}
        for m in re.finditer(
            r"^(val_f1|f1_binary|f1_categorical|val_f1_zs|f1_zs|f1_cat_zs):\s+([\d.]+)",
            body,
            re.M,
        ):
            metrics[m.group(1)] = float(m.group(2))
        out[(split, kind, llm)] = metrics
    return out


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else ""


def write_block(f, title, runs_by_split):
    """runs_by_split: {split: {(run, llm): {"posts": metrics, "comments": metrics}}}"""
    for split in ("TEST", "OOD"):
        data = runs_by_split.get(split, {})
        if not data:
            continue
        f.write(f"{title} [{split}]:\n")
        f.write(
            "run\tllm\t"
            "posts val_f1\tposts f1_binary\tposts f1_categorical\t\t"
            "comments val_f1\tcomments f1_binary\tcomments f1_categorical\n"
        )
        for run in sorted({r for r, _ in data.keys()}):
            for llm in LLMS:
                if (run, llm) not in data:
                    continue
                p = data[(run, llm)]["posts"]
                c = data[(run, llm)]["comments"]
                f.write(
                    "\t".join(
                        [
                            str(run),
                            llm,
                            fmt(p.get("val_f1")),
                            fmt(p.get("f1_binary")),
                            fmt(p.get("f1_categorical")),
                            "",
                            fmt(c.get("val_f1")),
                            fmt(c.get("f1_binary")),
                            fmt(c.get("f1_categorical")),
                        ]
                    )
                    + "\n"
                )
        f.write("\n")


with OUT.open("w") as f:
    for bl in BASELINES_ORDER:
        bl_dir = LOG_ROOT / bl
        if not bl_dir.exists():
            continue
        runs_by_split = {"TEST": {}, "OOD": {}}
        for run in (1, 2, 3):
            log = bl_dir / f"run{run}.log"
            if not log.exists():
                continue
            parsed = parse_log(log)
            for llm in LLMS:
                for split in ("TEST", "OOD"):
                    p = parsed.get((split, "posts", llm), {})
                    c = parsed.get((split, "comments", llm), {})
                    if not p and not c:
                        continue
                    runs_by_split[split][(run, llm)] = {"posts": p, "comments": c}

        title = {
            "zero_shot": "Zero-shot",
            "self_consistency": "Self-consistency",
            "cot": "Chain-of-thought",
            "self_refine": "Self-refine",
        }[bl]
        write_block(f, title, runs_by_split)

print(f"wrote {OUT}")
