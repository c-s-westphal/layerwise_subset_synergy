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

from models import vgg11, vgg13, vgg16, vgg19
from masking import ChannelMasker, generate_random_subsets, forward_with_mask
from mutual_information import calculate_mi_per_layer


def get_cifar10_test_loader(batch_size=128):
    """
    Get CIFAR-10 test data loader.
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader


def load_model(model_name, checkpoint_path, device='cuda'):
    """
    Load a trained model from checkpoint.
    """
    model_dict = {
        'vgg11': vgg11,
        'vgg13': vgg13,
        'vgg16': vgg16,
        'vgg19': vgg19
    }

    model = model_dict[model_name]().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Loaded {model_name} from {checkpoint_path}")
    print(f"Train Acc: {checkpoint['train_acc']:.2f}%, Test Acc: {checkpoint['test_acc']:.2f}%")

    return model


def evaluate_layer_masking(model, testloader, layer_idx, n_subsets=100, device='cuda'):
    """
    Evaluate a single layer by masking random subsets of channels.

    Args:
        model: Trained model
        testloader: Test data loader
        layer_idx: Index of the layer to evaluate
        n_subsets: Number of random subsets to test
        device: Device to run on

    Returns:
        List of predictions for each subset, true labels
    """
    masker = ChannelMasker(model, device)
    layer_name, n_channels = masker.get_layer_info()[layer_idx]

    print(f"  Layer {layer_idx}: {layer_name} ({n_channels} channels)")

    # Generate random subsets
    subsets = generate_random_subsets(n_channels, n_subsets)
    print(f"  Generated {len(subsets)} unique random subsets")

    # Get true labels (only need to do this once)
    true_labels = []
    for _, labels in testloader:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    # Collect predictions for each subset
    all_predictions = []

    for subset_idx, channels_to_mask in enumerate(tqdm(subsets, desc=f"  Masking subsets")):
        predictions, _ = forward_with_mask(
            model, testloader, layer_idx, channels_to_mask, device
        )
        all_predictions.append(predictions)

    return all_predictions, true_labels


def evaluate_model(model_name, checkpoint_path, n_subsets=100,
                   batch_size=128, device='cuda'):
    """
    Evaluate a trained model by masking channels in each layer.

    Args:
        model_name: Name of the model (vgg11, vgg13, vgg16, vgg19)
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
    testloader = get_cifar10_test_loader(batch_size)

    # Get layer info
    masker = ChannelMasker(model, device)
    layer_info = masker.get_layer_info()

    print(f"\nFound {len(layer_info)} convolutional layers")
    print()

    # Evaluate each layer
    predictions_by_layer = {}

    for layer_idx in range(len(layer_info)):
        print(f"Evaluating layer {layer_idx}...")
        predictions, labels = evaluate_layer_masking(
            model, testloader, layer_idx, n_subsets, device
        )
        predictions_by_layer[layer_idx] = predictions

    # Calculate MI for each layer
    print("\nCalculating mutual information...")
    mi_results = calculate_mi_per_layer(predictions_by_layer, labels)

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'Name':<30} {'Channels':<10} {'MI (mean ± std)'}")
    print(f"{'-'*60}")

    for layer_idx, (layer_name, n_channels) in enumerate(layer_info):
        mean_mi, std_mi = mi_results[layer_idx]
        print(f"{layer_idx:<8} {layer_name:<30} {n_channels:<10} {mean_mi:.4f} ± {std_mi:.4f}")

    print(f"{'='*60}\n")

    # Save results
    results = {
        'model_name': model_name,
        'checkpoint_path': checkpoint_path,
        'n_subsets': n_subsets,
        'layers': []
    }

    for layer_idx, (layer_name, n_channels) in enumerate(layer_info):
        mean_mi, std_mi = mi_results[layer_idx]
        results['layers'].append({
            'layer_idx': layer_idx,
            'layer_name': layer_name,
            'n_channels': n_channels,
            'mean_mi': float(mean_mi),
            'std_mi': float(std_mi)
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VGG models with channel masking')
    parser.add_argument('--model', type=str, default='all',
                       choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'all'],
                       help='Model to evaluate (default: all)')
    parser.add_argument('--checkpoint', type=str, default='best',
                       choices=['best', 'final'],
                       help='Which checkpoint to use (default: best)')
    parser.add_argument('--n-subsets', type=int, default=100,
                       help='Number of random subsets per layer (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Determine which models to evaluate
    if args.model == 'all':
        models_to_eval = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    else:
        models_to_eval = [args.model]

    # Evaluate each model
    all_results = {}

    for model_name in models_to_eval:
        checkpoint_path = f'checkpoints/{model_name}_{args.checkpoint}.pth'

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
    os.makedirs('results', exist_ok=True)
    output_path = f'results/mi_results_{args.checkpoint}.json'

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
