#!/usr/bin/env python3
"""
Plot experiment progression showing train vs test performance
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

TSV_PATH = Path(__file__).resolve().parent.parent / 'results' / 'autoresearch.tsv'


def load_experiments(tsv_path=TSV_PATH):
    """Load experiments, keeping only those that set a new best train score."""
    experiments, train_scores, test_scores, test_stds = [], [], [], []
    best_train = float('-inf')
    with open(tsv_path, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            train = float(row['train_all'])
            if train <= best_train:
                continue
            best_train = train
            experiments.append(f"exp{row['exp']}")
            train_scores.append(train)
            test_scores.append(float(row['test_avg_all']))
            std_p = float(row['test_std_posts'])
            std_c = float(row['test_std_comments'])
            test_stds.append(np.sqrt((std_p ** 2 + std_c ** 2) / 2.0))
    return experiments, train_scores, test_scores, test_stds


experiments, train_scores, test_scores, test_stds = load_experiments()

def plot_experiment_comparison():
    """Create a grouped bar chart comparing train vs test across experiments."""

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(experiments))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, train_scores, width,
                   label='Train', color='#4A90E2', alpha=0.9, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_scores, width,
                   label='Test (Mean)', color='#50C878', alpha=0.9, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)

    # Customize plot
    ax.set_xlabel('Experiment', fontsize=22, fontweight='bold')
    ax.set_ylabel(r'F1$_{\mathrm{val}}$', fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=12, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=18, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.85)

    plt.tight_layout()
    plt.savefig('experiment_progression.pdf', bbox_inches='tight')
    print("Saved: experiment_progression.pdf")
    plt.close()

def plot_experiment_lines():
    """Create a line plot showing train and test trends."""

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create lines
    ax.plot(experiments, train_scores, 'o-',
            linewidth=3, markersize=12, color='#4A90E2',
            label='Train', markeredgecolor='black', markeredgewidth=2)
    ax.errorbar(experiments, test_scores, yerr=test_stds, fmt='s-',
                linewidth=3, markersize=12, color='#50C878',
                ecolor='#50C878', elinewidth=1.5, capsize=5, capthick=1.5,
                label='Test (Mean)',
                markeredgecolor='black', markeredgewidth=2)

    # Customize plot first
    ax.set_xlabel('Experiment', fontsize=26, fontweight='bold')
    ax.set_ylabel(r'F1$_{\mathrm{val}}$', fontsize=26, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(fontsize=22, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.4, 0.8)

    # Add value labels with automatic positioning using adjustText
    texts = []
    for i, (exp, train, test) in enumerate(zip(experiments, train_scores, test_scores)):
        # Create text objects for test values
        t2 = ax.text(i, test, f'{test:.3f}',
                    ha='center', va='center', fontsize=16, fontweight='bold', color='#50C878',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#50C878', alpha=0.9, linewidth=1.2))
        texts.append(t2)

    # Automatically adjust text positions to avoid overlaps and points
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.6),
                expand_points=(2, 2),
                expand_text=(1.2, 1.2),
                force_text=(0.5, 0.5),
                force_points=(0.5, 0.5))

    plt.tight_layout()
    plt.savefig('experiment_progression_lines.pdf', bbox_inches='tight')
    print("Saved: experiment_progression_lines.pdf")
    plt.close()

def plot_combined():
    """Create a combined plot with both bar and line views."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    # Left: Bar chart
    x = np.arange(len(experiments))
    width = 0.35
    bars1 = ax1.bar(x - width/2, train_scores, width,
                    label='Train', color='#4A90E2', alpha=0.9, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, test_scores, width,
                    label='Test (Mean)', color='#50C878', alpha=0.9, edgecolor='black', linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

    ax1.set_xlabel('Experiment', fontsize=18, fontweight='bold')
    ax1.set_ylabel(r'F1$_{\mathrm{val}}$', fontsize=18, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments, fontsize=10, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelsize=14)
    ax1.legend(fontsize=16, loc='upper left', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 0.85)

    # Right: Line plot
    ax2.plot(experiments, train_scores, 'o-',
             linewidth=3, markersize=10, color='#4A90E2',
             label='Train', markeredgecolor='black', markeredgewidth=2)
    ax2.errorbar(experiments, test_scores, yerr=test_stds, fmt='s-',
                 linewidth=3, markersize=10, color='#50C878',
                 ecolor='#50C878', elinewidth=1.5, capsize=4, capthick=1.5,
                 label='Test (Mean)',
                 markeredgecolor='black', markeredgewidth=2)

    ax2.set_xlabel('Experiment', fontsize=18, fontweight='bold')
    ax2.set_ylabel(r'F1$_{\mathrm{val}}$', fontsize=18, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', labelsize=10)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.legend(fontsize=16, loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0.4, 0.8)

    plt.tight_layout()
    plt.savefig('experiment_progression_combined.pdf', bbox_inches='tight')
    print("Saved: experiment_progression_combined.pdf")
    plt.close()

def main():
    print("Creating experiment progression plots...\n")

    plot_experiment_comparison()
    plot_experiment_lines()
    plot_combined()

    print("\n" + "="*70)
    print("Generated 3 plots:")
    print("  • experiment_progression.pdf (bar chart)")
    print("  • experiment_progression_lines.pdf (line plot)")
    print("  • experiment_progression_combined.pdf (both views)")
    print("="*70)

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Train: {min(train_scores):.4f} → {max(train_scores):.4f} (Δ {max(train_scores)-min(train_scores):.4f})")
    print(f"  Test:  {min(test_scores):.4f} → {max(test_scores):.4f} (Δ {max(test_scores)-min(test_scores):.4f})")
    print(f"  Best test performance: {max(test_scores):.4f} ({experiments[test_scores.index(max(test_scores))]})")

if __name__ == "__main__":
    main()
