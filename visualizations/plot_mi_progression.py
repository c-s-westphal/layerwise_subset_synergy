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


def load_results(data_dir='data', architectures=['vgg11', 'vgg13', 'vgg16', 'vgg19']):
    """
    Load MI results from JSON files.

    Returns:
        dict: {architecture: {layer_idx: {'mean_mi': [...], 'std_mi': [...], 'layer_name': str}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: {'mean_mi': [], 'std_mi': [], 'layer_name': None}))

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
                results[arch][layer_idx]['mean_mi'].append(layer['mean_mi'])
                results[arch][layer_idx]['std_mi'].append(layer['std_mi'])

                if results[arch][layer_idx]['layer_name'] is None:
                    results[arch][layer_idx]['layer_name'] = layer['layer_name']

    return results


def plot_mi_progression(results, output_path='plots/mi_progression.png'):
    """
    Plot MI progression through layers for all architectures.
    Values are transformed as ln(10) - MI to show decreasing synergy.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'vgg11': '#1f77b4', 'vgg13': '#ff7f0e', 'vgg16': '#2ca02c', 'vgg19': '#d62728'}
    markers = {'vgg11': 'o', 'vgg13': 's', 'vgg16': '^', 'vgg19': 'D'}

    ln_10 = np.log(10)

    for arch in sorted(results.keys()):
        layer_indices = sorted(results[arch].keys())

        mean_mis = []
        std_mis = []

        for layer_idx in layer_indices:
            # Average across seeds
            mean_mi = np.mean(results[arch][layer_idx]['mean_mi'])
            # Std across seeds
            std_mi = np.std(results[arch][layer_idx]['mean_mi'])

            mean_mis.append(mean_mi)
            std_mis.append(std_mi)

        mean_mis = np.array(mean_mis)
        std_mis = np.array(std_mis)

        # Transform: ln(10) - MI
        transformed_mean = ln_10 - mean_mis

        # Plot with error bars
        ax.plot(layer_indices, transformed_mean,
                marker=markers.get(arch, 'o'),
                color=colors.get(arch, None),
                linewidth=2,
                markersize=8,
                label=arch.upper(),
                alpha=0.8)

        ax.fill_between(layer_indices,
                        transformed_mean - std_mis,
                        transformed_mean + std_mis,
                        color=colors.get(arch, None),
                        alpha=0.2)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('ln(10) - MI (bits)', fontsize=12)
    ax.set_title('Synergy Progression Through Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    return fig, ax


def print_summary_stats(results):
    """Print summary statistics for each architecture."""
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    for arch in sorted(results.keys()):
        print(f"\n{arch.upper()}:")
        print(f"{'Layer':<15} {'Layer Name':<20} {'Mean MI':<12} {'Std (across seeds)':<20}")
        print("-" * 80)

        layer_indices = sorted(results[arch].keys())
        for layer_idx in layer_indices:
            layer_name = results[arch][layer_idx]['layer_name']
            mean_mi = np.mean(results[arch][layer_idx]['mean_mi'])
            std_mi = np.std(results[arch][layer_idx]['mean_mi'])

            print(f"{layer_idx:<15} {layer_name:<20} {mean_mi:<12.4f} {std_mi:<20.4f}")


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
