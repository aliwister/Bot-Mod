#!/usr/bin/env python3
"""
Create combined confusion matrix plots (3 models side-by-side) with blue color scheme
"""

import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_log_file(log_path):
    """Parse the log file and extract intent predictions vs truth for each JSON file.
    Apply correction: if verdict=benign, set intent_pred to organic_contribution
    """

    with open(log_path, 'r') as f:
        lines = f.readlines()

    results = {}
    current_file = None

    header_re = re.compile(r'^\[TEST\]\s+(\S+\.json)\s*$')
    verdict_re = re.compile(
        r'^\[(\d+)\]\s+verdict=(\w+)\s+truth=\w+\s+correct=\w+\s+'
        r'intent=(\w+)\s+predicted=(\w+)'
    )

    for line in lines:
        line = line.strip()

        header_match = header_re.match(line)
        if header_match:
            current_file = header_match.group(1)
            results[current_file] = []
            continue

        match = verdict_re.match(line)
        if match and current_file:
            sample_id = match.group(1)
            verdict = match.group(2)
            intent_truth = match.group(3)
            intent_pred = match.group(4)

            # CORRECTION: If verdict is benign, set intent_pred to organic_contribution
            if verdict == 'benign':
                intent_pred = 'organic_contribution'

            results[current_file].append((intent_pred, intent_truth))

    return results

INTENT_SHORT = {
    'subtle_promotion': 'SP',
    'narrative_pushing': 'NP',
    'spam': 'S',
    'elicitation': 'E',
    'organic_contribution': 'OC',
}

def create_confusion_matrix(predictions):
    """Create a confusion matrix from list of (pred, truth) tuples."""

    all_intents = sorted(set([p for p, t in predictions] + [t for p, t in predictions]))

    matrix = defaultdict(lambda: defaultdict(int))
    for pred, truth in predictions:
        matrix[truth][pred] += 1

    matrix_array = np.zeros((len(all_intents), len(all_intents)))
    for i, truth_intent in enumerate(all_intents):
        for j, pred_intent in enumerate(all_intents):
            matrix_array[i, j] = matrix[truth_intent][pred_intent]

    return matrix_array, all_intents

def plot_combined_matrices(results_dict, dataset_type):
    """Plot confusion matrices for all models side by side with blue color scheme."""

    filtered = {k: v for k, v in results_dict.items() if dataset_type.lower() in k.lower()}

    if not filtered:
        return

    model_order = ['llama', 'mistral', 'qwen']
    sorted_items = sorted(filtered.items(), key=lambda x: next((i for i, m in enumerate(model_order) if m in x[0].lower()), 999))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, (json_file, (matrix, labels)) in enumerate(sorted_items):
        ax = axes[idx]

        if 'llama' in json_file:
            model = 'LLaMA'
        elif 'mistral' in json_file:
            model = 'Mistral'
        elif 'qwen' in json_file:
            model = 'Qwen'
        else:
            model = 'Unknown'

        # Blue color scheme
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar=True, ax=ax, vmin=0, vmax=matrix.max(),
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'shrink': 0.8})

        total = matrix.sum()
        correct = np.trace(matrix)
        accuracy = correct / total if total > 0 else 0

        # Minimal subplot title (just model name)
        ax.set_title(f'{model}', fontsize=14, fontweight='bold', pad=10)

        # Labels
        ax.set_xlabel('Predicted', fontsize=11)
        if idx == 0:
            ax.set_ylabel('True', fontsize=11)
        else:
            ax.set_ylabel('')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    # No main title
    plt.tight_layout()
    filename = f'confusion_matrices_{dataset_type.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_normalized_matrices(results_dict, dataset_type):
    """Plot normalized (percentage) confusion matrices with blue color scheme."""

    filtered = {k: v for k, v in results_dict.items() if dataset_type.lower() in k.lower()}

    if not filtered:
        return

    model_order = ['llama', 'mistral', 'qwen']
    sorted_items = sorted(filtered.items(), key=lambda x: next((i for i, m in enumerate(model_order) if m in x[0].lower()), 999))

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.subplots_adjust(wspace=0.08, right=0.88)

    for idx, (json_file, (matrix, labels)) in enumerate(sorted_items):
        ax = axes[idx]

        if 'llama' in json_file:
            model = 'LLaMA'
        elif 'mistral' in json_file:
            model = 'Mistral'
        elif 'qwen' in json_file:
            model = 'Qwen'
        else:
            model = 'Unknown'

        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(matrix, row_sums, where=row_sums!=0) * 100

        short_labels = [INTENT_SHORT.get(l, l) for l in labels]
        def fmt_val(v):
            if v == 0:
                return '0'
            s = f'{v:.1f}'
            return s[:-2] if s.endswith('.0') else s
        annots = np.vectorize(fmt_val)(normalized_matrix)
        sns.heatmap(normalized_matrix, annot=annots, fmt='', cmap='Blues',
                    xticklabels=short_labels, yticklabels=short_labels,
                    cbar=False, ax=ax, vmin=0, vmax=100,
                    linewidths=0.5, linecolor='white',
                    annot_kws={'fontsize': 26})

        ax.set_title(f'{model}', fontsize=36, fontweight='bold', pad=15)
        ax.set_aspect('equal')

        if idx == 1:
            ax.set_xlabel('Predicted', fontsize=36)
        else:
            ax.set_xlabel('')
        if idx == 0:
            ax.set_ylabel('True', fontsize=36)
        else:
            ax.set_ylabel('')

        plt.setp(ax.get_xticklabels(), rotation=0, fontsize=32)
        if idx == 0:
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=32)
        else:
            ax.set_yticklabels([])

    cbar = fig.colorbar(axes[0].collections[0], ax=axes, location='right', shrink=0.8)
    cbar.set_label('%', fontsize=32)
    cbar.ax.tick_params(labelsize=28)
    if dataset_type.lower() != 'comments':
        cbar.ax.set_visible(False)
    filename = f'normalized_confusion_matrices_{dataset_type.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def main():
    log_path = '../logs/78(2de8985)/r1.log'

    print(f"Parsing {log_path}...")
    print("Applying correction: verdict=benign → intent_pred=organic_contribution\n")
    results = parse_log_file(log_path)

    print(f"Found {len(results)} JSON files with predictions\n")

    # Create separate entries per model and dataset type
    results_dict = {}
    if results:
        for json_file, predictions in results.items():
            if predictions:
                m, l = create_confusion_matrix(predictions)
                results_dict[json_file] = (m, l)

        print("Creating normalized confusion matrices (blue color scheme, larger text)...\n")
        plot_normalized_matrices(results_dict, 'posts')
        plot_normalized_matrices(results_dict, 'comments')

        print("\n" + "="*70)
        print("Generated 2 plots:")
        print("  • normalized_confusion_matrices_posts.png")
        print("  • normalized_confusion_matrices_comments.png")
        print("="*70)

if __name__ == "__main__":
    main()
