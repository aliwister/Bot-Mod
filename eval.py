import sys
import time
from prepare import evaluate_f1, llm_mod
from train import ModeratorBot

TEST_FILES = [
    "data/dataset/id/posts-test-generated-qwen.json",
    "data/dataset/id/comments-test-generated-qwen.json",
    "data/dataset/id/posts-test-generated-mistral.json",
    "data/dataset/id/comments-test-generated-mistral.json",
    "data/dataset/id/posts-test-generated-llama.json",
    "data/dataset/id/comments-test-generated-llama.json",
]

OOD_FILES = [
    "data/dataset/ood/posts-ood-posts-generated-qwen.json",
    "data/dataset/ood/comments-ood-comments-generated-qwen.json",
    "data/dataset/ood/posts-ood-posts-generated-mistral.json",
    "data/dataset/ood/comments-ood-comments-generated-mistral.json",
    "data/dataset/ood/posts-ood-posts-generated-llama.json",
    "data/dataset/ood/comments-ood-comments-generated-llama.json",
]

mode = sys.argv[1] if len(sys.argv) > 1 else "both"
files = []
if mode in ("test", "both"):
    files += [("TEST", p) for p in TEST_FILES]
if mode in ("ood", "both"):
    files += [("OOD", p) for p in OOD_FILES]

mod = ModeratorBot(llm_mod)

for split, path in files:
    name = path.split("/")[-1]
    print(f"\n{'='*60}\n[{split}] {name}\n{'='*60}", flush=True)
    t0 = time.time()
    f1_merged, m = evaluate_f1(mod, filename=path)
    elapsed = time.time() - t0
    print(f"val_f1:          {f1_merged:.4f}")
    print(f"f1_binary:       {m['f1_bin']:.4f}")
    print(f"f1_categorical:  {m['f1_cat']:.4f}")
    print(f"val_f1_zs:       {m['f1_zs_merged']:.4f}")
    print(f"f1_zs:           {m['f1_zs']:.4f}")
    print(f"f1_cat_zs:       {m['f1_cat_zs']:.4f}")
    print(f"f1_posts:        {m['f1_posts']:.4f}")
    print(f"f1_comments:     {m['f1_comments']:.4f}")
    print(f"time:            {elapsed:.1f}s", flush=True)
