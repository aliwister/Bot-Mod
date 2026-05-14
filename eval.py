import time
from prepare import evaluate_f1, llm_mod
from train import ModeratorBot

FILES = [
    "/run/user/1000/autoresearch2/cache/test/posts-test-generated-qwen.json",
    "/run/user/1000/autoresearch2/cache/test/comments-test-generated-qwen.json",

    "/run/user/1000/autoresearch2/cache/test/posts-test-generated-mistral.json",
    "/run/user/1000/autoresearch2/cache/test/comments-test-generated-mistral.json",

    "/run/user/1000/autoresearch2/cache/test/posts-test-generated-llama.json",
    "/run/user/1000/autoresearch2/cache/test/comments-test-generated-llama.json",
]

mod = ModeratorBot(llm_mod)

for path in FILES:
    name = path.split("/")[-1]
    print(f"\n{'='*60}\n{name}\n{'='*60}")
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
    print(f"time:            {elapsed:.1f}s")
