import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import json
from tqdm import tqdm
import numpy as np

from models import vgg9, vgg11, vgg13, vgg16, vgg19
from models.resnet import ResNet20, ResNet32, ResNet56, ResNet74
from masking import ActivationMasker, generate_random_subsets, forward_with_masked_activations
from mutual_information import calculate_mi_per_layer


def get_cifar10_train_loader(batch_size=128):
    """
    Get CIFAR-10 train data loader (without augmentation).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader


def load_model(model_name, checkpoint_path, device='cuda'):
    """
    Load a trained model from checkpoint.
    """
    model_dict = {
        'vgg9': vgg9,
        'vgg11': vgg11,
        'vgg13': vgg13,
        'vgg16': vgg16,
        'vgg19': vgg19,
        'resnet20': ResNet20,
        'resnet32': ResNet32,
        'resnet56': ResNet56,
        'resnet74': ResNet74
    }

    model = model_dict[model_name]().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Loaded {model_name} from {checkpoint_path}")
    print(f"Train Acc: {checkpoint['train_acc']:.2f}%, Test Acc: {checkpoint['test_acc']:.2f}%")

    return model


def get_baseline_mi(model, trainloader, device='cuda'):
    """
    Calculate MI for the unmasked (baseline) model.

    Args:
        model: Trained model
        trainloader: Train data loader
        device: Device to run on

    Returns:
        Baseline MI: I(Y; Ŷ_full)
    """
    model.eval()
    all_predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    true_labels = np.array(true_labels)

    from mutual_information import calculate_mutual_information
    baseline_mi = calculate_mutual_information(true_labels, all_predictions)

    return baseline_mi


def evaluate_layer_masking(model, trainloader, layer_idx, masker, baseline_mi, n_subsets=1000, device='cuda'):
    """
    Evaluate a single layer by masking random subsets of activation channels.

    Args:
        model: Trained model
        trainloader: Train data loader
        layer_idx: Index of the layer to evaluate
        masker: ActivationMasker instance with computed stats
        baseline_mi: Baseline MI from unmasked model (I(Y; Ŷ_full))
        n_subsets: Number of random subsets to test
        device: Device to run on

    Returns:
        Tuple of (mean_delta_mi, std_delta_mi, mean_masked_mi, std_masked_mi, n_channels)
    """
    layer_name, n_channels = masker.get_layer_info()[layer_idx]

    # Generate random subsets
    subsets = generate_random_subsets(n_channels, n_subsets)

    # Get true labels (only need to do this once)
    true_labels = []
    for _, labels in trainloader:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    # Collect predictions for each subset (without progress bar)
    all_predictions = []

    for channels_to_mask in subsets:
        predictions, _ = forward_with_masked_activations(
            model, trainloader, layer_idx, channels_to_mask, masker, device
        )
        all_predictions.append(predictions)

    # Calculate MI for this layer
    from mutual_information import calculate_average_mi
    mean_masked_mi, std_masked_mi = calculate_average_mi(all_predictions, true_labels)

    # Calculate delta: baseline - masked (information lost)
    mean_delta_mi = baseline_mi - mean_masked_mi
    std_delta_mi = std_masked_mi  # Std remains the same

    return mean_delta_mi, std_delta_mi, mean_masked_mi, std_masked_mi, n_channels


def evaluate_model(model_name, checkpoint_path, n_subsets=1000,
                   batch_size=128, device='cuda'):
    """
    Evaluate a trained model by masking activation channels in each layer.

    Args:
        model_name: Name of the model (vgg11, vgg13, vgg16, vgg19, resnet20, resnet32, resnet56, resnet74)
        checkpoint_path: Path to model checkpoint
        n_subsets: Number of random subsets per layer
        batch_size: Batch size for evaluation
        device: Device to run on

    Returns:
        Dictionary with MI results per layer
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")

    # Load model
    model = load_model(model_name, checkpoint_path, device)
    trainloader = get_cifar10_train_loader(batch_size)

    # Calculate baseline MI (unmasked model)
    print("Calculating baseline MI (unmasked model)...", end=' ', flush=True)
    baseline_mi = get_baseline_mi(model, trainloader, device)
    print(f"Baseline MI = {baseline_mi:.4f}")

    # Initialize masker
    masker = ActivationMasker(model, device)
    layer_info = masker.get_layer_info()

    print(f"\nFound {len(layer_info)} convolutional layers")
    print()

    # Compute activation statistics for all layers
    print("Computing activation statistics...")
    for layer_idx in range(len(layer_info)):
        masker.compute_activation_stats(trainloader, layer_idx)
    print()

    # Evaluate each layer
    mi_results = {}

    for layer_idx in range(len(layer_info)):
        layer_name, n_channels = layer_info[layer_idx]
        print(f"Evaluating layer {layer_idx}: {layer_name} ({n_channels} channels)...", end=' ', flush=True)

        mean_delta_mi, std_delta_mi, mean_masked_mi, std_masked_mi, _ = evaluate_layer_masking(
            model, trainloader, layer_idx, masker, baseline_mi, n_subsets, device
        )
        mi_results[layer_idx] = (mean_delta_mi, std_delta_mi, mean_masked_mi, std_masked_mi)

        # Print MI immediately after layer is done
        print(f"ΔMI = {mean_delta_mi:.4f} ± {std_delta_mi:.4f} (masked MI = {mean_masked_mi:.4f})")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Baseline MI (unmasked): {baseline_mi:.4f}")
    print(f"{'-'*60}")
    print(f"{'Layer':<8} {'Name':<20} {'Channels':<10} {'ΔMI (mean ± std)':<25} {'Masked MI'}")
    print(f"{'-'*60}")

    for layer_idx, (layer_name, n_channels) in enumerate(layer_info):
        mean_delta_mi, std_delta_mi, mean_masked_mi, std_masked_mi = mi_results[layer_idx]
        print(f"{layer_idx:<8} {layer_name:<20} {n_channels:<10} {mean_delta_mi:.4f} ± {std_delta_mi:.4f}        {mean_masked_mi:.4f}")

    print(f"{'='*60}\n")

    # Save results
    results = {
        'model_name': model_name,
        'checkpoint_path': checkpoint_path,
        'n_subsets': n_subsets,
        'baseline_mi': float(baseline_mi),
        'layers': []
    }

    for layer_idx, (layer_name, n_channels) in enumerate(layer_info):
        mean_delta_mi, std_delta_mi, mean_masked_mi, std_masked_mi = mi_results[layer_idx]
        results['layers'].append({
            'layer_idx': layer_idx,
            'layer_name': layer_name,
            'n_channels': n_channels,
            'mean_delta_mi': float(mean_delta_mi),
            'std_delta_mi': float(std_delta_mi),
            'mean_masked_mi': float(mean_masked_mi),
            'std_masked_mi': float(std_masked_mi)
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VGG and ResNet models with channel masking')
    parser.add_argument('--model', type=str, default='all',
                       choices=['vgg11', 'vgg13', 'vgg16', 'vgg19',
                               'resnet20', 'resnet32', 'resnet56', 'resnet74', 'all'],
                       help='Model to evaluate (default: all)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint file, or "best"/"final" for standard names (default: None)')
    parser.add_argument('--n_subsets', '--n-subsets', type=int, default=1000,
                       dest='n_subsets',
                       help='Number of random subsets per layer (default: 1000)')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=128,
                       dest='batch_size',
                       help='Batch size (default: 128)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Seed used for naming outputs (default: None)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--max_layer_batches', type=int, default=None,
                       help='Maximum batches per layer (unused, for compatibility)')
    parser.add_argument('--keep_checkpoint', action='store_true',
                       help='Keep checkpoint after evaluation (unused, for compatibility)')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Determine which models to evaluate
    if args.model == 'all':
        models_to_eval = ['vgg11', 'vgg13', 'vgg16', 'vgg19',
                         'resnet20', 'resnet32', 'resnet56', 'resnet74']
    else:
        models_to_eval = [args.model]

    # Evaluate each model
    all_results = {}

    for model_name in models_to_eval:
        # Determine checkpoint path
        if args.checkpoint is None:
            # Default: look for seed-specific checkpoint
            if args.seed is not None:
                checkpoint_path = f'checkpoints/{model_name}_seed{args.seed}/checkpoint_latest.pth'
            else:
                checkpoint_path = f'checkpoints/{model_name}_best.pth'
        elif args.checkpoint in ['best', 'final']:
            # Legacy mode: best or final
            checkpoint_path = f'checkpoints/{model_name}_{args.checkpoint}.pth'
        else:
            # Direct path provided
            checkpoint_path = args.checkpoint

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            print(f"Skipping {model_name}")
            continue

        results = evaluate_model(
            model_name,
            checkpoint_path,
            n_subsets=args.n_subsets,
            batch_size=args.batch_size,
            device=args.device
        )

        all_results[model_name] = results

    # Save all results
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        output_filename = f'mi_results_{args.model}_seed{args.seed}.json'
    elif args.checkpoint in ['best', 'final']:
        output_filename = f'mi_results_{args.checkpoint}.json'
    else:
        output_filename = 'mi_results.json'

    output_path = os.path.join(args.output_dir, output_filename)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
