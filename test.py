#!/usr/bin/env python3
import json
import copy
from concurrent.futures import ThreadPoolExecutor
import prepare
from train import ModeratorBot
from prepare import llm_mod, metric

MODEL_MAP = {
    "qwen":    "Qwen/Qwen3-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
}


def evaluate_file(mod, data_file, llm_key, mode):
    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {llm_key.upper()} - {mode.upper()}")
    print(f"File: {data_file}")
    print(f"{'=' * 70}")

    with open(data_file, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} items")

    for row in rows:
        if mode == "comment" and "context" in row:
            if isinstance(row["context"], str):
                row["context"] = json.loads(row["context"])
        elif "context" not in row:
            row["context"] = ""

    model_name = MODEL_MAP[llm_key]

    def process_row(args):
        idx, row = args
        community      = row["community"].strip()
        system_prompt  = row["system_prompt"].strip()
        truth          = row["label_binary"].strip().lower()
        intent         = row["label_intent"].strip().lower()
        context        = row.get("context", "")
        generated_text = row.get("text", "").strip()

        print(f"[{idx+1}/{len(rows)}]  {mode}  intent={intent}  truth={truth}")

        m = copy.deepcopy(mod)

        if mode == "comment" and isinstance(context, dict):
            context_str = json.dumps(context, ensure_ascii=False)
        else:
            context_str = context if isinstance(context, str) else ""

        user = prepare.UserBot(
            community, system_prompt,
            mode=mode, context=context_str,
            model=model_name, pregenerated_text=generated_text,
        )
        m._init_hypothesis(user.M, community)

        y0 = m.y
        t0 = getattr(m, "t", intent)

        for _ in range(m.max_iterations):
            m.sample_intent_step()
            m.sample_label_step()
            critique = m._refine_hypothesis()
            question = m._generate_probe(critique)
            response = user._call_user(question)
            m.P.append({"Critique": critique, "Q": question, "R": response})
            if m.is_converged():
                print(f"[{idx+1}] Converged")
                break

        verdict = m.y
        correct = verdict == truth
        print(f"[{idx+1}]  verdict={verdict}  truth={truth}  correct={correct}")
        return (
            {"verdict": verdict.upper(), "intent_type": truth.upper(), "correct": correct,
             "intent": intent, "t": m.t.strip().lower(),
             "community": community, "M": user.M, "context": context_str},
            {"verdict": y0.upper(), "intent_type": truth.upper(), "correct": y0 == truth,
             "intent": intent, "t": t0.strip().lower(),
             "community": community, "M": user.M, "context": context_str},
        )

    with ThreadPoolExecutor() as executor:
        pairs = list(executor.map(process_row, enumerate(rows)))

    results = [p[0] for p in pairs]

    f1_bin, f1_cat = metric(results)
    _lambda = 0.7
    val_f1 = _lambda * f1_bin + (1 - _lambda) * f1_cat

    print(f"\n  F1 Binary:      {f1_bin:.4f}")
    print(f"  F1 Categorical: {f1_cat:.4f}")
    print(f"  Val F1:         {val_f1:.4f}")

    return results, f1_bin, f1_cat, val_f1


def main():
    mod = ModeratorBot(llm_mod)

    print("=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    models = ["qwen", "mistral", "llama"]
    all_results = {}

    for llm_key in models:
        for mode_plural, mode in [("posts", "post"), ("comments", "comment")]:
            data_file = f"cache/test/{mode_plural}_test_generated_{llm_key}.json"
            results, f1_bin, f1_cat, val_f1 = evaluate_file(mod, data_file, llm_key, mode)
            all_results[f"{llm_key}_{mode_plural}"] = {
                "f1_bin": f1_bin, "f1_cat": f1_cat, "val_f1": val_f1,
            }

    print(f"\n{'=' * 70}")
    print("AGGREGATED RESULTS")
    print(f"{'=' * 70}")

    print(f"\n{'Model':<10} {'Mode':<10} {'Val F1':<10} {'F1 Bin':<10} {'F1 Cat':<10}")
    print("-" * 50)
    for llm_key in models:
        for mode_plural in ["posts", "comments"]:
            m = all_results[f"{llm_key}_{mode_plural}"]
            print(f"{llm_key:<10} {mode_plural:<10} {m['val_f1']:<10.4f} {m['f1_bin']:<10.4f} {m['f1_cat']:<10.4f}")

    posts_val_f1    = [all_results[f"{llm}_posts"]["val_f1"]    for llm in models]
    comments_val_f1 = [all_results[f"{llm}_comments"]["val_f1"] for llm in models]
    all_val_f1      = [m["val_f1"] for m in all_results.values()]
    all_f1_bin      = [m["f1_bin"] for m in all_results.values()]
    all_f1_cat      = [m["f1_cat"] for m in all_results.values()]

    avg_posts_val_f1    = sum(posts_val_f1)    / len(posts_val_f1)
    avg_comments_val_f1 = sum(comments_val_f1) / len(comments_val_f1)
    avg_val_f1_all      = sum(all_val_f1)      / len(all_val_f1)
    avg_f1_bin_all      = sum(all_f1_bin)      / len(all_f1_bin)
    avg_f1_cat_all      = sum(all_f1_cat)      / len(all_f1_cat)

    print(f"\navg_posts_val_f1:    {avg_posts_val_f1:.4f}")
    print(f"avg_comments_val_f1: {avg_comments_val_f1:.4f}")
    print(f"avg_val_f1_all:      {avg_val_f1_all:.4f}  <- ACCEPTANCE CRITERION #1")
    print(f"avg_f1_bin_all:      {avg_f1_bin_all:.4f}  <- ACCEPTANCE CRITERION #2")
    print(f"avg_f1_cat_all:      {avg_f1_cat_all:.4f}")

    output = {
        "per_file": {k: {"val_f1": v["val_f1"], "f1_bin": v["f1_bin"], "f1_cat": v["f1_cat"]}
                     for k, v in all_results.items()},
        "aggregated": {
            "avg_posts_val_f1":    avg_posts_val_f1,
            "avg_comments_val_f1": avg_comments_val_f1,
            "avg_val_f1_all":      avg_val_f1_all,
            "avg_f1_bin_all":      avg_f1_bin_all,
            "avg_f1_cat_all":      avg_f1_cat_all,
        },
    }

    print(output)
    return output


if __name__ == "__main__":
    main()
