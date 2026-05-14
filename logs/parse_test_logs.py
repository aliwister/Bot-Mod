import csv
import glob
import os
import re
import statistics

BASE = "logs/test"
TRAIN_TSV = "results/results-avg.tsv"
OUT_TSV = "autoresearch.tsv"

posts_re = re.compile(r"f1_posts:\s+([\d.]+)")
comments_re = re.compile(r"f1_comments:\s+([\d.]+)")
dir_re = re.compile(r"^exp(\d+)\(([^)]+)\)$")


def discover_dirs():
    entries = []
    for name in os.listdir(BASE):
        m = dir_re.match(name)
        if not m:
            continue
        entries.append((int(m.group(1)), name))
    entries.sort()
    return entries


def parse_dir(path):
    posts, comments = [], []
    for log in sorted(glob.glob(os.path.join(path, "*.log"))):
        with open(log) as f:
            text = f.read()
        for m in posts_re.finditer(text):
            v = float(m.group(1))
            if v > 0:
                posts.append(v)
        for m in comments_re.finditer(text):
            v = float(m.group(1))
            if v > 0:
                comments.append(v)
    return posts, comments


def load_train():
    train = {}
    with open(TRAIN_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            exp = row["exp"].replace("exp", "")
            train[exp] = {
                "posts": float(row["f1_posts"]),
                "comments": float(row["f1_comments"]),
                "all": float(row["val_f1"]),
            }
    return train


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


train = load_train()
dirs = discover_dirs()

columns = [
    "exp",
    "train_posts",
    "train_comments",
    "train_all",
    "test_avg_posts",
    "test_std_posts",
    "test_avg_comments",
    "test_std_comments",
    "test_avg_all",
    "n_posts",
    "n_comments",
]

rows = []
for exp_num, d in dirs:
    full = os.path.join(BASE, d)
    posts, comments = parse_dir(full)
    all_vals = posts + comments
    t = train.get(str(exp_num), {"posts": float("nan"), "comments": float("nan"), "all": float("nan")})
    rows.append({
        "exp": str(exp_num),
        "train_posts": f"{t['posts']:.4f}",
        "train_comments": f"{t['comments']:.4f}",
        "train_all": f"{t['all']:.4f}",
        "test_avg_posts": f"{mean(posts):.4f}",
        "test_std_posts": f"{std(posts):.4f}",
        "test_avg_comments": f"{mean(comments):.4f}",
        "test_std_comments": f"{std(comments):.4f}",
        "test_avg_all": f"{mean(all_vals):.4f}",
        "n_posts": str(len(posts)),
        "n_comments": str(len(comments)),
    })

with open(OUT_TSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

header = (
    f"{'Exp#':<5} | "
    f"{'Train Posts':>11} {'Train Cmts':>10} {'Train All':>9} | "
    f"{'Avg Posts':>9} {'Std Posts':>9} {'Avg Cmts':>8} {'Std Cmts':>8} {'Avg All':>8}"
)
print(header)
print("-" * len(header))
for r in rows:
    print(
        f"{r['exp']:<5} | "
        f"{r['train_posts']:>11} {r['train_comments']:>10} {r['train_all']:>9} | "
        f"{r['test_avg_posts']:>9} {r['test_std_posts']:>9} {r['test_avg_comments']:>8} {r['test_std_comments']:>8} {r['test_avg_all']:>8}"
    )
print(f"\nWrote {len(rows)} rows to {OUT_TSV}")
