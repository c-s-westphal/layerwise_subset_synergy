#!/usr/bin/env python3
"""
Plot the progression of MI values through layers for both VGG and MLP architectures.
Creates side-by-side subplots for comparison.
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
                                          'baseline_mi': [], 'layer_name': str}}}
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

                # Store data
                results[arch][layer_idx]['mean_delta_mi'].append(layer['mean_delta_mi'])
                results[arch][layer_idx]['std_delta_mi'].append(layer['std_delta_mi'])
                results[arch][layer_idx]['baseline_mi'].append(arch_data['baseline_mi'])

                if results[arch][layer_idx]['layer_name'] is None:
                    results[arch][layer_idx]['layer_name'] = layer['layer_name']

    return results


def plot_combined_mi_progression(vgg_results, mlp_results, output_path='plots/combined_mi_progression.png'):
    """
    Plot MI progression for both VGG and MLP architectures in side-by-side subplots.
    """
    # Create figure with 1 row, 2 columns - side by side
    fig, (ax_vgg, ax_mlp) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Color schemes
    vgg_colors = {
        'vgg9': '#F39B7F',   # Peach
        'vgg11': '#E64B35',  # Red-orange
        'vgg13': '#4DBBD5',  # Cyan
        'vgg16': '#00A087',  # Teal
        'vgg19': '#3C5488'   # Blue
    }

    mlp_colors = {
        'mlp2': '#F39B7F',   # Peach
        'mlp3': '#E64B35',   # Red-orange
        'mlp4': '#4DBBD5',   # Cyan
        'mlp5': '#00A087',   # Teal
        'mlp6': '#3C5488'    # Blue
    }

    # Plot VGG results
    for arch in sorted(vgg_results.keys()):
        layer_indices = sorted(vgg_results[arch].keys())

        mean_delta_mis = []
        std_delta_mis = []

        for layer_idx in layer_indices:
            # Average across seeds
            mean_delta_mi = np.mean(vgg_results[arch][layer_idx]['mean_delta_mi'])
            # Std across seeds
            std_delta_mi = np.std(vgg_results[arch][layer_idx]['mean_delta_mi'])

            mean_delta_mis.append(mean_delta_mi)
            std_delta_mis.append(std_delta_mi)

        mean_delta_mis = np.array(mean_delta_mis)
        std_delta_mis = np.array(std_delta_mis)

        # Plot with error bars (no markers)
        ax_vgg.plot(layer_indices, mean_delta_mis,
                    color=vgg_colors.get(arch, None),
                    linewidth=2.5,
                    label=arch.upper(),
                    alpha=0.85)

        ax_vgg.fill_between(layer_indices,
                            mean_delta_mis - std_delta_mis,
                            mean_delta_mis + std_delta_mis,
                            color=vgg_colors.get(arch, None),
                            alpha=0.2)

    # Plot MLP results
    for arch in sorted(mlp_results.keys()):
        layer_indices = sorted(mlp_results[arch].keys())

        mean_delta_mis = []
        std_delta_mis = []

        for layer_idx in layer_indices:
            # Average across seeds
            mean_delta_mi = np.mean(mlp_results[arch][layer_idx]['mean_delta_mi'])
            # Std across seeds
            std_delta_mi = np.std(mlp_results[arch][layer_idx]['mean_delta_mi'])

            mean_delta_mis.append(mean_delta_mi)
            std_delta_mis.append(std_delta_mi)

        mean_delta_mis = np.array(mean_delta_mis)
        std_delta_mis = np.array(std_delta_mis)

        # Plot with error bars (no markers)
        ax_mlp.plot(layer_indices, mean_delta_mis,
                    color=mlp_colors.get(arch, None),
                    linewidth=2.5,
                    label=arch.upper(),
                    alpha=0.85)

        ax_mlp.fill_between(layer_indices,
                            mean_delta_mis - std_delta_mis,
                            mean_delta_mis + std_delta_mis,
                            color=mlp_colors.get(arch, None),
                            alpha=0.2)

    # Styling for VGG subplot
    ax_vgg.set_xlabel('Layer Index', fontsize=16, fontweight='normal')
    ax_vgg.set_ylabel('Subset Synergy', fontsize=16, fontweight='normal')
    ax_vgg.set_title('VGG on CIFAR-10', fontsize=16, fontweight='bold', pad=8)
    ax_vgg.tick_params(axis='both', which='major', labelsize=14)
    ax_vgg.legend(fontsize=12, loc='best', frameon=True, fancybox=False,
                  edgecolor='black', framealpha=1)
    ax_vgg.grid(False)
    for spine in ax_vgg.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    ax_vgg.set_facecolor('white')

    # Styling for MLP subplot
    ax_mlp.set_xlabel('Layer Index', fontsize=16, fontweight='normal')
    ax_mlp.set_ylabel('Subset Synergy', fontsize=16, fontweight='normal')
    ax_mlp.set_title('MLP on MNIST', fontsize=16, fontweight='bold', pad=8)
    ax_mlp.tick_params(axis='both', which='major', labelsize=14)
    ax_mlp.legend(fontsize=12, loc='best', frameon=True, fancybox=False,
                  edgecolor='black', framealpha=1)
    ax_mlp.grid(False)
    for spine in ax_mlp.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    ax_mlp.set_facecolor('white')

    plt.tight_layout()

    # Save as PNG
    output_path_png = output_path if str(output_path).endswith('.png') else str(output_path) + '.png'
    plt.savefig(output_path_png, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_png}")

    # Save as PDF
    output_path_pdf = str(output_path).replace('.png', '.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_pdf}")

    plt.close()

    return fig, (ax_vgg, ax_mlp)


def print_summary_stats(results, title):
    """Print summary statistics for each architecture."""
    print(f"\n{'='*80}")
    print(f"{title} Summary Statistics")
    print(f"{'='*80}")

    for arch in sorted(results.keys()):
        # Get baseline MI (should be same across all layers for a given arch)
        if 0 in results[arch] and results[arch][0]['baseline_mi']:
            baseline_mi = np.mean(results[arch][0]['baseline_mi'])
        else:
            baseline_mi = 0.0

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
    # Load VGG results
    print("Loading VGG results...")
    vgg_results = load_results(data_dir='data', architectures=['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19'])

    # Load MLP results
    print("\nLoading MLP results...")
    mlp_results = load_results(data_dir='data', architectures=['mlp2', 'mlp3', 'mlp4', 'mlp5', 'mlp6'])

    if not vgg_results and not mlp_results:
        print("No results found! Please check that JSON files exist in the data/ directory.")
        exit(1)

    # Print summary statistics
    if vgg_results:
        print_summary_stats(vgg_results, "VGG")
    if mlp_results:
        print_summary_stats(mlp_results, "MLP")

    # Create combined plot
    if vgg_results and mlp_results:
        print("\nCreating combined plot...")
        plot_combined_mi_progression(vgg_results, mlp_results)
        print("\n" + "="*70)
        print("Combined plots generated successfully!")
        print(f"PNG (600 DPI): plots/combined_mi_progression.png")
        print(f"PDF: plots/combined_mi_progression.pdf")
        print("="*70)
    else:
        print("\nWarning: Need both VGG and MLP results to create combined plot")

    print("\nDone!")
