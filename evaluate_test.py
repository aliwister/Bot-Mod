#!/usr/bin/env python3
"""
Evaluate moderator on all 6 test files (posts + comments for qwen, mistral, llama).
Reports metrics for each file and aggregated statistics.
"""
import json
import copy
from concurrent.futures import ThreadPoolExecutor
import prepare

MODEL_MAP = {
    "qwen": "Qwen/Qwen3-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

# Reverse mapping for file lookup
MODEL_KEY_MAP = {v: k for k, v in MODEL_MAP.items()}

def evaluate_file(mod, data_file, llm_key, mode):
    """
    Evaluate moderator on a single file.
    Returns: (results, f1_binary, f1_categorical, val_f1)
    """
    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {llm_key.upper()} - {mode.upper()}")
    print(f"File: {data_file}")
    print(f"{'=' * 80}")

    # Load data
    with open(data_file, encoding="utf-8") as f:
        rows = json.load(f)

    print(f"Loaded {len(rows)} items")

    # Ensure context is properly formatted for comments
    for row in rows:
        if mode == "comment" and "context" in row:
            if isinstance(row["context"], str):
                row["context"] = json.loads(row["context"])
        else:
            if "context" not in row:
                row["context"] = ""

    model_name = MODEL_MAP[llm_key]

    def process_row(args):
        idx, row = args
        community     = row["community"].strip()
        system_prompt = row["system_prompt"].strip()
        truth         = row["label_binary"].strip().lower()
        intent        = row["label_intent"].strip().lower()
        context       = row.get("context", "")
        generated_text = row.get("text", "").strip()

        print(f"[{idx+1}/{len(rows)}]  {mode}  intent={intent}  truth={truth}")

        m = copy.deepcopy(mod)

        # Serialize context for comment mode
        if mode == "comment" and isinstance(context, dict):
            context_str = json.dumps(context, ensure_ascii=False)
        else:
            context_str = context if isinstance(context, str) else ""

        # Create UserBot with pre-generated text
        user = prepare.UserBot(
            community,
            system_prompt,
            mode=mode,
            context=context_str,
            model=model_name,
            pregenerated_text=generated_text
        )
        m._init_hypothesis(user.M, community)

        # Store initial hypothesis
        y0 = m.y
        t0 = getattr(m, "t", intent)

        for i in range(m.max_iterations):
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
             "intent": intent.lower(), "t": m.t.strip().lower(), "community": community,
             "M": user.M, "context": context_str},
            {"verdict": y0.upper(), "intent_type": truth.upper(), "correct": y0 == truth,
             "intent": intent.lower(), "t": t0.strip().lower(), "community": community,
             "M": user.M, "context": context_str},
        )

    with ThreadPoolExecutor() as executor:
        pairs = list(executor.map(process_row, enumerate(rows)))

    results          = [p[0] for p in pairs]
    results_zeroshot = [p[1] for p in pairs]

    # Calculate metrics
    f1_bin, f1_cat = prepare.metric(results)
    f1_zs, f1_cat_zs = prepare.metric(results_zeroshot)

    # val_f1 = weighted combination
    _lambda = 0.7
    val_f1 = _lambda * f1_bin + (1 - _lambda) * f1_cat

    print(f"\n📊 RESULTS:")
    print(f"   F1 Binary:      {f1_bin:.4f}")
    print(f"   F1 Categorical: {f1_cat:.4f}")
    print(f"   Val F1:         {val_f1:.4f}")
    print(f"   Zero-shot F1:   {f1_zs:.4f}")

    return results, f1_bin, f1_cat, val_f1

def main():
    """Main evaluation loop"""
    # Import the moderator class
    # NOTE: You need to replace this with your actual moderator class
    # For now, we'll create a mock moderator that needs to be replaced
    class MockModerator:
        def __init__(self):
            self.max_iterations = 3
            self.y = "benign"  # Initial hypothesis
            self.t = "organic_contribution"
            self.P = []

        def _init_hypothesis(self, message, community):
            # Mock implementation - replace with actual
            self.y = "benign"
            self.t = "organic_contribution"

        def sample_intent_step(self):
            # Mock implementation - replace with actual
            pass

        def sample_label_step(self):
            # Mock implementation - replace with actual
            pass

        def _refine_hypothesis(self):
            # Mock implementation - replace with actual
            return "Critique of current hypothesis"

        def _generate_probe(self, critique):
            # Mock implementation - replace with actual
            return "What is your experience with this topic?"

        def is_converged(self):
            # Mock implementation - replace with actual
            return len(self.P) >= 2

    # TODO: Replace MockModerator with your actual moderator class
    # e.g., from your_module import YourModeratorClass
    # mod = YourModeratorClass()
    mod = MockModerator()

    print("=" * 80)
    print("TEST EVALUATION")
    print("=" * 80)
    print("\n⚠️  NOTE: Using MockModerator. Replace with actual moderator class.")
    print("   Edit evaluate_test.py and replace MockModerator with your moderator.\n")

    all_results = {}
    models = ["qwen", "mistral", "llama"]

    # Evaluate each file
    for llm_key in models:
        for mode in ["posts", "comments"]:
            data_file = f"cache/test/{mode}_test_generated_{llm_key}.json"

            results, f1_bin, f1_cat, val_f1 = evaluate_file(
                mod, data_file, llm_key, mode.rstrip('s')  # Remove 's' from 'posts'/'comments'
            )

            all_results[f"{llm_key}_{mode}"] = {
                "results": results,
                "f1_bin": f1_bin,
                "f1_cat": f1_cat,
                "val_f1": val_f1
            }

    # Calculate aggregated metrics
    print(f"\n{'=' * 80}")
    print("AGGREGATED RESULTS")
    print(f"{'=' * 80}")

    print("\n📊 PER-FILE METRICS:")
    print(f"\n{'Model':<10} {'Mode':<10} {'Val F1':<10} {'F1 Binary':<12} {'F1 Categorical':<15}")
    print("-" * 80)

    for llm_key in models:
        for mode in ["posts", "comments"]:
            key = f"{llm_key}_{mode}"
            metrics = all_results[key]
            print(f"{llm_key:<10} {mode:<10} {metrics['val_f1']:<10.4f} {metrics['f1_bin']:<12.4f} {metrics['f1_cat']:<15.4f}")

    # Calculate averages
    print("\n📈 AGGREGATED AVERAGES:")

    # Average val_f1 for posts
    posts_val_f1 = [all_results[f"{llm}_posts"]["val_f1"] for llm in models]
    avg_posts_val_f1 = sum(posts_val_f1) / len(posts_val_f1)
    print(f"   Average Val F1 (Posts):       {avg_posts_val_f1:.4f}")

    # Average val_f1 for comments
    comments_val_f1 = [all_results[f"{llm}_comments"]["val_f1"] for llm in models]
    avg_comments_val_f1 = sum(comments_val_f1) / len(comments_val_f1)
    print(f"   Average Val F1 (Comments):    {avg_comments_val_f1:.4f}")

    # Average val_f1 for ALL (posts + comments)
    all_val_f1 = [metrics["val_f1"] for metrics in all_results.values()]
    avg_val_f1_all = sum(all_val_f1) / len(all_val_f1)
    print(f"   Average Val F1 (All):         {avg_val_f1_all:.4f}")

    # Average f1_bin for all
    all_f1_bin = [metrics["f1_bin"] for metrics in all_results.values()]
    avg_f1_bin = sum(all_f1_bin) / len(all_f1_bin)
    print(f"   Average F1 Binary (All):      {avg_f1_bin:.4f}")

    # Average f1_cat for all
    all_f1_cat = [metrics["f1_cat"] for metrics in all_results.values()]
    avg_f1_cat = sum(all_f1_cat) / len(all_f1_cat)
    print(f"   Average F1 Categorical (All): {avg_f1_cat:.4f}")

    # Acceptance criteria
    print(f"\n{'=' * 80}")
    print("⚠️  ACCEPTANCE CRITERIA")
    print(f"{'=' * 80}")
    print("\n📋 To accept an improvement over baseline, it must WIN on BOTH:")
    print(f"   1. Average Val F1 (All):     {avg_val_f1_all:.4f}  ← Must be higher than baseline")
    print(f"   2. Average F1 Binary (All):  {avg_f1_bin:.4f}  ← Must be higher than baseline")
    print("\n💡 Run this evaluation after each improvement to compare against baseline.")
    print("   Only accept the improvement if BOTH metrics improve.")

    # Save results
    output_file = "test_evaluation_results.json"
    output_data = {
        "per_file": {
            key: {
                "val_f1": metrics["val_f1"],
                "f1_bin": metrics["f1_bin"],
                "f1_cat": metrics["f1_cat"]
            }
            for key, metrics in all_results.items()
        },
        "aggregated": {
            "avg_posts_val_f1": avg_posts_val_f1,
            "avg_comments_val_f1": avg_comments_val_f1,
            "avg_val_f1_all": avg_val_f1_all,
            "avg_f1_bin_all": avg_f1_bin,
            "avg_f1_cat_all": avg_f1_cat
        },
        "acceptance_criteria": {
            "avg_val_f1_all": avg_val_f1_all,
            "avg_f1_bin_all": avg_f1_bin
        }
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print("=" * 80)

    return output_data

if __name__ == "__main__":
    main()
