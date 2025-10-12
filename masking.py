import torch
import numpy as np
from typing import List, Set, Tuple, Dict


def generate_random_subsets(n_channels: int, n_subsets: int = 1000) -> List[Set[int]]:
    """
    Generate random subsets of channel indices with no duplicates.

    Args:
        n_channels: Total number of channels in the layer
        n_subsets: Number of unique random subsets to generate (default: 1000)

    Returns:
        List of sets, where each set contains indices of channels to mask
    """
    subsets = []
    seen_subsets = set()

    max_attempts = n_subsets * 100  # Prevent infinite loop
    attempts = 0

    while len(subsets) < n_subsets and attempts < max_attempts:
        attempts += 1

        # Random subset size from 1 to n_channels - 1 (leave at least 1 channel)
        subset_size = np.random.randint(1, n_channels)

        # Randomly select channel indices to mask
        channels_to_mask = np.random.choice(
            n_channels, size=subset_size, replace=False
        )

        # Convert to frozenset for hashing (to check uniqueness)
        subset_frozen = frozenset(channels_to_mask)

        if subset_frozen not in seen_subsets:
            seen_subsets.add(subset_frozen)
            subsets.append(set(channels_to_mask))

    if len(subsets) < n_subsets:
        print(f"Warning: Could only generate {len(subsets)} unique subsets out of {n_subsets} requested")

    return subsets


class ActivationMasker:
    """
    Handles masking of channel activations in convolutional layers using hooks.
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.conv_layers = self._get_conv_layers()
        self.activation_stats = {}  # Store min/max per layer
        self.hook_handle = None
        self.channels_to_mask = None
        self.noise_range = None

    def _get_conv_layers(self) -> List[Tuple[str, torch.nn.Conv2d]]:
        """Get all convolutional layers from the model."""
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))
        return conv_layers

    def get_layer_info(self) -> List[Tuple[str, int]]:
        """
        Get information about each conv layer.

        Returns:
            List of tuples (layer_name, n_output_channels)
        """
        info = []
        for name, layer in self.conv_layers:
            n_channels = layer.out_channels
            info.append((name, n_channels))
        return info

    def compute_activation_stats(self, dataloader, layer_idx: int):
        """
        Compute min/max activation values for a specific layer.

        Args:
            dataloader: DataLoader for the dataset
            layer_idx: Index of the layer in self.conv_layers
        """
        if layer_idx >= len(self.conv_layers):
            raise ValueError(f"Layer index {layer_idx} out of range")

        layer_name, layer = self.conv_layers[layer_idx]

        # Skip if already computed
        if layer_name in self.activation_stats:
            return

        # Collect activations
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())

        # Register hook
        handle = layer.register_forward_hook(hook_fn)

        # Forward pass on a subset of data to get statistics
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if i >= 10:  # Use first 10 batches for stats
                    break
                inputs = inputs.to(self.device)
                _ = self.model(inputs)

        # Remove hook
        handle.remove()

        # Compute min/max across all collected activations
        all_activations = torch.cat(activations, dim=0)
        layer_min = all_activations.min().item()
        layer_max = all_activations.max().item()

        self.activation_stats[layer_name] = (layer_min, layer_max)
        print(f"  Computed activation range for {layer_name}: [{layer_min:.4f}, {layer_max:.4f}]")

    def _masking_hook(self, module, input, output):
        """
        Hook function that masks selected channels with uniform noise.
        """
        if self.channels_to_mask is None or self.noise_range is None:
            return output

        # Clone output to avoid in-place modification issues
        masked_output = output.clone()

        # Replace selected channels with uniform noise
        for channel_idx in self.channels_to_mask:
            noise = torch.empty_like(masked_output[:, channel_idx, :, :]).uniform_(
                self.noise_range[0], self.noise_range[1]
            )
            masked_output[:, channel_idx, :, :] = noise

        return masked_output

    def mask_activations(self, layer_idx: int, channels_to_mask: Set[int]):
        """
        Set up masking for specified channels in a layer.

        Args:
            layer_idx: Index of the layer in self.conv_layers
            channels_to_mask: Set of channel indices to mask
        """
        if layer_idx >= len(self.conv_layers):
            raise ValueError(f"Layer index {layer_idx} out of range")

        layer_name, layer = self.conv_layers[layer_idx]

        # Ensure we have activation statistics
        if layer_name not in self.activation_stats:
            raise ValueError(f"Must compute activation stats for {layer_name} first")

        # Store masking configuration
        self.channels_to_mask = channels_to_mask
        self.noise_range = self.activation_stats[layer_name]

        # Remove existing hook if any
        if self.hook_handle is not None:
            self.hook_handle.remove()

        # Register new hook
        self.hook_handle = layer.register_forward_hook(self._masking_hook)

    def remove_hook(self):
        """Remove the masking hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        self.channels_to_mask = None
        self.noise_range = None


def forward_with_masked_activations(model, dataloader, layer_idx: int,
                                     channels_to_mask: Set[int],
                                     masker: ActivationMasker,
                                     device='cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform forward pass with masked channel activations and collect predictions.

    Args:
        model: The neural network model
        dataloader: DataLoader for the dataset
        layer_idx: Index of the layer to mask
        channels_to_mask: Set of channel indices to mask
        masker: ActivationMasker instance with computed stats
        device: Device to run on

    Returns:
        Tuple of (predictions, true_labels) as numpy arrays
    """
    # Set up masking hook
    masker.mask_activations(layer_idx, channels_to_mask)

    # Forward pass
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Remove hook
    masker.remove_hook()

    return np.array(all_predictions), np.array(all_labels)
