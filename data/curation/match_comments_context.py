"""
Build comment JSON splits from judge-result CSVs.

For each row, matches it with one post object from the corresponding community JSON
in moltbook-export/<community>.json (cycling through posts if rows > posts).
Reads judge CSVs (true_intent/true_intent_type columns + pass filter).
"""

import csv
import json
import random
from pathlib import Path

DATA_DIR = Path("moltbook-export")
SPLITS = [
    ("raw-gpt-generated-sys-prompts/dataset-id-judge_results.csv",  "comments-id-matched.json"),
    ("raw-gpt-generated-sys-prompts/dataset-ood-judge_results.csv", "comments-ood-matched.json"),
]

random.seed(42)


def resolve_json_path(name: str) -> Path:
    path = DATA_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No data file for community '{name}' in {DATA_DIR}")
    return path


def is_judge_csv(fieldnames: list[str]) -> bool:
    return "true_intent" in fieldnames


def passes_judge(row: dict) -> bool:
    return (
        row.get("intent_type_match", "True") in ("True", True) and
        row.get("intent_match",      "True") in ("True", True) and
        row.get("community_match",   "True") in ("True", True)
    )


def normalize_context(raw: dict) -> dict:
    """
    Strip raw community JSON down to the shape used in comment-test.json:
      post_id, submolt.name, post.{title,content,created_at,like_counts,comment_count,author},
      comments[].{content, author.{name,karma,follower_count}}
    """
    post = raw.get("post", {})
    author = post.get("author", {})
    raw_comments = raw.get("comments", [])

    return {
        "post_id": raw.get("post_id"),
        "submolt": {
            "name": raw.get("submolt", {}).get("name"),
        },
        "post": {
            "title":         post.get("title"),
            "content":       post.get("content"),
            "created_at":    post.get("created_at"),
            "like_counts":   post.get("like_counts"),
            "comment_count": post.get("comment_count"),
            "author": {
                "name":             author.get("name"),
                "description":      author.get("description"),
                "karma":            author.get("karma"),
                "follower_count":   author.get("follower_count"),
                "following_count":  author.get("following_count"),
                "you_follow":       author.get("you_follow", False),
            },
        },
        "comments": [
            {
                "content": c.get("content"),
                "author": {
                    "name":           c.get("author", {}).get("name"),
                    "karma":          c.get("author", {}).get("karma"),
                    "follower_count": c.get("author", {}).get("follower_count"),
                },
            }
            for c in raw_comments
        ],
    }


def build(csv_path: str, out_path: str):
    community_cache: dict[str, list] = {}
    community_indices: dict[str, int] = {}
    entries = []
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        judge_mode = is_judge_csv(reader.fieldnames or [])
        for row in reader:
            if judge_mode and not passes_judge(row):
                skipped += 1
                continue

            community = row["community"].strip()
            name = community.removeprefix("m/")

            if name not in community_cache:
                posts = json.loads(resolve_json_path(name).read_text(encoding="utf-8"))
                random.shuffle(posts)
                community_cache[name] = posts
                community_indices[name] = 0

            idx = community_indices[name] % len(community_cache[name])
            community_indices[name] += 1

            intent      = row.get("true_intent",      row.get("intent"))
            intent_type = row.get("true_intent_type", row.get("intent_type"))

            entries.append({
                "system_prompt": row["system_prompt"],
                "community":     community,
                "intent":        intent,
                "intent_type":   intent_type,
                "context":       normalize_context(community_cache[name][idx]),
            })

    Path(out_path).write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    msg = f"Wrote {len(entries)} entries → {out_path}"
    if skipped:
        msg += f"  (skipped {skipped} failed-judge rows)"
    print(msg)


for csv_path, out_path in SPLITS:
    build(csv_path, out_path)
