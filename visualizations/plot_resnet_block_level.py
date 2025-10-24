#!/usr/bin/env python3
"""
Plot MI progression for ResNet models at the block level only.
Only shows values after each residual block (relu2 layers), not intermediate relu1 layers.
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def is_block_level_relu(layer_name):
    """
    Check if a layer is a block-level ReLU (after residual addition).

    Block-level ReLUs are:
    - 'relu' (initial conv)
    - 'layer*.*.relu2' (end of residual blocks)

    Intermediate ReLUs (relu1) are excluded.
    """
    if layer_name == 'relu':
        return True
    if 'relu2' in layer_name:
        return True
    return False


def load_resnet_results(data_dir='data', architectures=['resnet20', 'resnet32', 'resnet56', 'resnet74']):
    """
    Load MI results from JSON files, filtering to block-level ReLUs only.

    Returns:
        dict: {architecture: {block_idx: {'mean_delta_mi': [...], 'std_delta_mi': [...],
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

            # Get the architecture data
            arch_data = data.get(arch, data.get(list(data.keys())[0]))

            block_idx = 0  # Counter for block-level layers only
            for layer in arch_data['layers']:
                layer_name = layer['layer_name']

                # Skip if not a block-level ReLU
                if not is_block_level_relu(layer_name):
                    continue

                # This is a block-level layer
                if 'mean_delta_mi' in layer:
                    results[arch][block_idx]['mean_delta_mi'].append(layer['mean_delta_mi'])
                    results[arch][block_idx]['std_delta_mi'].append(layer['std_delta_mi'])
                    results[arch][block_idx]['baseline_mi'].append(arch_data['baseline_mi'])
                else:
                    # Old format - skip this file
                    print(f"Skipping {filepath} - old format (needs re-evaluation)")
                    break

                if results[arch][block_idx]['layer_name'] is None:
                    results[arch][block_idx]['layer_name'] = layer_name

                block_idx += 1

    return results


def plot_block_level_mi(results, output_path='plots/resnet_block_level_mi.png'):
    """
    Plot MI progression through blocks for ResNet architectures.
    Only shows block-level ReLUs (after residual addition).
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        'resnet20': '#9467bd',
        'resnet32': '#8c564b',
        'resnet56': '#e377c2',
        'resnet74': '#7f7f7f'
    }
    markers = {
        'resnet20': 'o',
        'resnet32': 's',
        'resnet56': '^',
        'resnet74': 'D'
    }

    for arch in sorted(results.keys()):
        block_indices = sorted(results[arch].keys())

        mean_delta_mis = []
        std_delta_mis = []

        for block_idx in block_indices:
            # Average across seeds
            mean_delta_mi = np.mean(results[arch][block_idx]['mean_delta_mi'])
            # Std across seeds
            std_delta_mi = np.std(results[arch][block_idx]['mean_delta_mi'])

            mean_delta_mis.append(mean_delta_mi)
            std_delta_mis.append(std_delta_mi)

        mean_delta_mis = np.array(mean_delta_mis)
        std_delta_mis = np.array(std_delta_mis)

        # Plot with error bars
        ax.plot(block_indices, mean_delta_mis,
                marker=markers.get(arch, 'o'),
                color=colors.get(arch, None),
                linewidth=2.5,
                markersize=10,
                label=arch.upper(),
                alpha=0.8)

        ax.fill_between(block_indices,
                        mean_delta_mis - std_delta_mis,
                        mean_delta_mis + std_delta_mis,
                        color=colors.get(arch, None),
                        alpha=0.2)

    ax.set_xlabel('Block Index (after residual addition)', fontsize=13)
    ax.set_ylabel('ΔMI = I(Y; Ŷ_full) - I(Y; Ŷ_masked) (nats)', fontsize=13)
    ax.set_title('Information Lost by Masking at Residual Block Outputs', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")

    return fig, ax


def print_block_level_stats(results):
    """Print summary statistics for block-level layers only."""
    print("\n" + "="*90)
    print("Block-Level MI Statistics (relu2 only - after residual addition)")
    print("="*90)

    for arch in sorted(results.keys()):
        # Get baseline MI
        baseline_mi = np.mean(results[arch][0]['baseline_mi'])

        print(f"\n{arch.upper()}:")
        print(f"Baseline MI (unmasked): {baseline_mi:.4f}")
        print(f"{'Block':<8} {'Layer Name':<25} {'Mean ΔMI':<12} {'Std (across seeds)':<20}")
        print("-" * 90)

        block_indices = sorted(results[arch].keys())
        for block_idx in block_indices:
            layer_name = results[arch][block_idx]['layer_name']
            mean_delta_mi = np.mean(results[arch][block_idx]['mean_delta_mi'])
            std_delta_mi = np.std(results[arch][block_idx]['mean_delta_mi'])

            print(f"{block_idx:<8} {layer_name:<25} {mean_delta_mi:<12.4f} {std_delta_mi:<20.4f}")


if __name__ == '__main__':
    # Load block-level results only
    print("Loading ResNet results (block-level ReLUs only)...\n")
    results = load_resnet_results()

    if not results:
        print("No results found! Please check that JSON files exist in the data/ directory.")
        exit(1)

    # Print summary statistics
    print_block_level_stats(results)

    # Create plot
    plot_block_level_mi(results)

    print("\nDone!")
