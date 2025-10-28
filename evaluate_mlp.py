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
from sklearn.metrics import mutual_info_score

from models import mlp2, mlp3, mlp4, mlp5, mlp6


def get_mnist_train_loader(batch_size=512):
    """
    Get MNIST train data loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader


def load_model(model_name, checkpoint_path, dropout=0.0, device='cuda'):
    """
    Load a trained MLP model from checkpoint.
    """
    model_dict = {
        'mlp2': mlp2,
        'mlp3': mlp3,
        'mlp4': mlp4,
        'mlp5': mlp5,
        'mlp6': mlp6
    }

    model = model_dict[model_name](dropout=dropout).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Loaded {model_name} from {checkpoint_path}")
    print(f"Train Acc: {checkpoint['train_acc']:.2f}%, Test Acc: {checkpoint['test_acc']:.2f}%")

    return model


def generate_random_neuron_masks(n_neurons, n_masks, seed=None):
    """
    Generate random boolean masks for neuron subsets.

    Args:
        n_neurons: Total number of neurons in the layer
        n_masks: Number of random masks to generate
        seed: Random seed for reproducibility

    Returns:
        List of boolean numpy arrays of shape (n_neurons,)
        True = keep neuron, False = mask (zero out) neuron
    """
    if seed is not None:
        np.random.seed(seed)

    masks = []
    for _ in range(n_masks):
        # Random subset size from 1 to n_neurons - 1
        subset_size = np.random.randint(1, n_neurons)

        # Create mask (True = keep, False = mask out)
        mask = np.zeros(n_neurons, dtype=bool)
        indices_to_keep = np.random.choice(n_neurons, size=subset_size, replace=False)
        mask[indices_to_keep] = True

        masks.append(mask)

    return masks


class NeuronMaskingHook:
    """
    Forward hook that masks (zeros out) specific neurons in a layer.
    """
    def __init__(self, mask):
        """
        Args:
            mask: Boolean numpy array of shape (n_neurons,)
                  True = keep neuron, False = zero out neuron
        """
        self.mask = torch.from_numpy(mask).bool()

    def __call__(self, module, input, output):
        """
        Apply mask to layer output.

        Args:
            output: Tensor of shape (batch_size, n_neurons)

        Returns:
            Masked output with same shape
        """
        # Multiply output by mask
        masked_output = output * self.mask.to(output.device).float()
        return masked_output


def get_predictions_and_labels(model, dataloader, device, layer_idx=None, mask=None):
    """
    Get model predictions and labels, optionally with masking applied to a specific layer.

    Args:
        model: The MLP model
        dataloader: DataLoader for evaluation
        device: Device to use
        layer_idx: Index of layer to mask (0 = input_proj, 1+ = residual blocks)
        mask: Boolean array indicating which neurons to keep (True) or mask (False)

    Returns:
        predictions: numpy array of predicted class labels
        labels: numpy array of true labels
    """
    model.eval()
    all_predictions = []
    all_labels = []

    # Register hook if masking is requested
    hook_handle = None
    if mask is not None and layer_idx is not None:
        maskable_layers = model.get_maskable_layers()

        if layer_idx >= len(maskable_layers):
            raise ValueError(f"layer_idx {layer_idx} out of range for model with {len(maskable_layers)} maskable layers")

        _, _, target_module = maskable_layers[layer_idx]
        hook = NeuronMaskingHook(mask)
        hook_handle = target_module.register_forward_hook(hook)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.numpy())

    # Remove hook
    if hook_handle is not None:
        hook_handle.remove()

    return np.array(all_predictions), np.array(all_labels)


def calculate_mi_for_layer(model, dataloader, device, layer_idx, n_subsets=100, seed=None):
    """
    Calculate mutual information for a specific layer with random neuron masking.

    Args:
        model: The MLP model
        dataloader: DataLoader for evaluation
        device: Device to use
        layer_idx: Index of layer to evaluate
        n_subsets: Number of random neuron subsets to test
        seed: Random seed for reproducibility

    Returns:
        dict with:
            - baseline_mi: MI with no masking
            - mean_delta_mi: Mean difference in MI when neurons are masked
            - std_delta_mi: Std of MI differences
            - n_neurons: Number of neurons in the layer
    """
    print(f"  Evaluating layer {layer_idx}...")

    # Get baseline predictions (no masking)
    predictions_full, labels = get_predictions_and_labels(model, dataloader, device)

    # Calculate baseline MI
    baseline_mi = mutual_info_score(labels, predictions_full)
    print(f"    Baseline MI (no masking): {baseline_mi:.4f}")

    # Get number of neurons in this layer
    n_neurons = 256  # All layers have 256 neurons

    # Generate random masks
    masks = generate_random_neuron_masks(n_neurons, n_subsets, seed=seed)

    # Calculate MI for each masked version
    masked_mis = []
    for mask_idx, mask in enumerate(tqdm(masks, desc=f"    Layer {layer_idx} masks", leave=False)):
        predictions_masked, _ = get_predictions_and_labels(
            model, dataloader, device, layer_idx=layer_idx, mask=mask
        )

        mi_masked = mutual_info_score(labels, predictions_masked)
        masked_mis.append(mi_masked)

    masked_mis = np.array(masked_mis)

    # Calculate delta MI (information lost due to masking)
    delta_mis = baseline_mi - masked_mis
    mean_delta_mi = np.mean(delta_mis)
    std_delta_mi = np.std(delta_mis)

    print(f"    Mean ΔMI: {mean_delta_mi:.4f} ± {std_delta_mi:.4f}")

    return {
        'baseline_mi': baseline_mi,
        'mean_delta_mi': mean_delta_mi,
        'std_delta_mi': std_delta_mi,
        'n_neurons': n_neurons
    }


def evaluate_all_layers(model, model_name, dataloader, device, n_subsets=100, seed=None):
    """
    Evaluate all maskable layers in the model.

    Returns:
        dict with layer-wise results
    """
    print(f"\nEvaluating all layers for {model_name}...")

    maskable_layers = model.get_maskable_layers()
    print(f"Found {len(maskable_layers)} maskable layers")

    results = {
        'model': model_name,
        'n_layers': model.n_layers,
        'hidden_dim': model.hidden_dim,
        'baseline_mi': None,  # Will be set from first layer
        'layers': []
    }

    for layer_idx, layer_name, _ in maskable_layers:
        layer_results = calculate_mi_for_layer(
            model, dataloader, device, layer_idx, n_subsets, seed
        )

        # Set baseline MI from first layer (should be same for all)
        if results['baseline_mi'] is None:
            results['baseline_mi'] = layer_results['baseline_mi']

        results['layers'].append({
            'layer_idx': layer_idx,
            'layer_name': layer_name,
            'mean_delta_mi': layer_results['mean_delta_mi'],
            'std_delta_mi': layer_results['std_delta_mi'],
            'n_neurons': layer_results['n_neurons']
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MLP with layer-wise neuron masking')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, required=True,
                       choices=['mlp2', 'mlp3', 'mlp4', 'mlp5', 'mlp6'],
                       help='Model architecture')
    parser.add_argument('--n_subsets', type=int, default=100,
                       help='Number of random neuron subsets to evaluate per layer')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for evaluation')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability (should match training)')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    model = load_model(args.model, args.checkpoint, args.dropout, args.device)

    # Get data loader
    print("\nLoading MNIST training data...")
    trainloader = get_mnist_train_loader(args.batch_size)

    # Evaluate all layers
    results = evaluate_all_layers(
        model, args.model, trainloader, args.device,
        n_subsets=args.n_subsets, seed=args.seed
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
    output_file = os.path.join(args.output_dir, f"mi_results_{args.model}{seed_suffix}.json")

    with open(output_file, 'w') as f:
        json.dump({args.model: results}, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Number of layers: {model.n_layers}")
    print(f"Baseline MI: {results['baseline_mi']:.4f}")
    print(f"\nLayer-wise ΔMI:")
    for layer_result in results['layers']:
        print(f"  Layer {layer_result['layer_idx']} ({layer_result['layer_name']}): "
              f"{layer_result['mean_delta_mi']:.4f} ± {layer_result['std_delta_mi']:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
