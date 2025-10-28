#!/usr/bin/env python3
"""
Plot the progression of MI values through layers for different VGG architectures.
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Set up Times New Roman font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def load_results(data_dir='data', architectures=['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19']):
    """
    Load MI results from JSON files.

    Returns:
        dict: {architecture: {layer_idx: {'mean_delta_mi': [...], 'std_delta_mi': [...],
                                          'baseline_mi': [...], 'layer_name': str}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: {'mean_delta_mi': [], 'std_delta_mi': [],
                                                        'baseline_mi': [], 'layer_name': None}))

    for arch in architectures:
        pattern = f"{data_dir}/mi_results_{arch}_seed*.json"
        files = glob.glob(pattern)

        if not files:
            print(f"Warning: No files found for {arch}")
            continue

        print(f"Found {len(files)} files for {arch}")

        for filepath in files:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Get the architecture data (key should match the architecture name)
            arch_data = data.get(arch, data.get(list(data.keys())[0]))

            for layer in arch_data['layers']:
                layer_idx = layer['layer_idx']

                # Check if this is new format (with mean_delta_mi) or old format (with mean_mi)
                if 'mean_delta_mi' in layer:
                    # New format
                    results[arch][layer_idx]['mean_delta_mi'].append(layer['mean_delta_mi'])
                    results[arch][layer_idx]['std_delta_mi'].append(layer['std_delta_mi'])
                    results[arch][layer_idx]['baseline_mi'].append(arch_data['baseline_mi'])
                else:
                    # Old format - skip this file
                    print(f"Skipping {filepath} - old format (needs re-evaluation)")
                    break

                if results[arch][layer_idx]['layer_name'] is None:
                    results[arch][layer_idx]['layer_name'] = layer['layer_name']

    return results


def plot_mi_progression(results, output_path='plots/mi_progression.png'):
    """
    Plot MI progression through layers for all architectures.
    Values show ΔMI = I(Y; Ŷ_full) - I(Y; Ŷ_masked), the information lost due to masking.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Publication-quality color scheme
    colors = {
        'vgg9': '#F39B7F',   # Peach
        'vgg11': '#E64B35',  # Red-orange
        'vgg13': '#4DBBD5',  # Cyan
        'vgg16': '#00A087',  # Teal
        'vgg19': '#3C5488'   # Blue
    }

    for arch in sorted(results.keys()):
        layer_indices = sorted(results[arch].keys())

        mean_delta_mis = []
        std_delta_mis = []

        for layer_idx in layer_indices:
            # Average across seeds
            mean_delta_mi = np.mean(results[arch][layer_idx]['mean_delta_mi'])
            # Std across seeds
            std_delta_mi = np.std(results[arch][layer_idx]['mean_delta_mi'])

            mean_delta_mis.append(mean_delta_mi)
            std_delta_mis.append(std_delta_mi)

        mean_delta_mis = np.array(mean_delta_mis)
        std_delta_mis = np.array(std_delta_mis)

        # Plot with error bars (no markers)
        ax.plot(layer_indices, mean_delta_mis,
                color=colors.get(arch, None),
                linewidth=2.5,
                label=arch.upper(),
                alpha=0.85)

        ax.fill_between(layer_indices,
                        mean_delta_mis - std_delta_mis,
                        mean_delta_mis + std_delta_mis,
                        color=colors.get(arch, None),
                        alpha=0.2)

    # Publication-quality styling
    ax.set_xlabel('Layer Index', fontsize=16, fontweight='normal')
    ax.set_ylabel('Subset Synergy', fontsize=16, fontweight='normal')
    ax.set_title('Information Lost by Masking Across Layers', fontsize=16, fontweight='bold', pad=10)

    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Legend with updated styling
    ax.legend(fontsize=13, loc='best', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1)

    # Remove grid lines
    ax.grid(False)

    # Black border (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # Set background color to white
    ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_path}")

    return fig, ax


def print_summary_stats(results):
    """Print summary statistics for each architecture."""
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    for arch in sorted(results.keys()):
        # Get baseline MI (should be same across all layers for a given arch)
        baseline_mi = np.mean(results[arch][0]['baseline_mi'])

        print(f"\n{arch.upper()}:")
        print(f"Baseline MI (unmasked): {baseline_mi:.4f}")
        print(f"{'Layer':<15} {'Layer Name':<20} {'Mean ΔMI':<12} {'Std (across seeds)':<20}")
        print("-" * 80)

        layer_indices = sorted(results[arch].keys())
        for layer_idx in layer_indices:
            layer_name = results[arch][layer_idx]['layer_name']
            mean_delta_mi = np.mean(results[arch][layer_idx]['mean_delta_mi'])
            std_delta_mi = np.std(results[arch][layer_idx]['mean_delta_mi'])

            print(f"{layer_idx:<15} {layer_name:<20} {mean_delta_mi:<12.4f} {std_delta_mi:<20.4f}")


if __name__ == '__main__':
    # Load results
    results = load_results()

    if not results:
        print("No results found! Please check that JSON files exist in the data/ directory.")
        exit(1)

    # Print summary statistics
    print_summary_stats(results)

    # Create plot
    plot_mi_progression(results)

    print("\nDone!")
