
import time

from prepare import evaluate_f1_test, llm_mod
from train import ModeratorBot

t_start = time.time()
mod = ModeratorBot(llm_mod)
val_f1, other_metrics = evaluate_f1_test(mod)
t_end = time.time()
print(f"val_f1: {val_f1:.4f}")
print(f"f1_binary: {other_metrics['f1_bin']:.4f}")
print(f"f1_categorical: {other_metrics['f1_cat']:.4f}")
print(f"val_f1_zs: {other_metrics['f1_zs_merged']:.4f}")
print(f"f1_zs: {other_metrics['f1_zs']:.4f}")
print(f"f1_cat_zs: {other_metrics['f1_cat_zs']:.4f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
