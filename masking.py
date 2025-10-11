import torch
import numpy as np
from typing import List, Set, Tuple


def generate_random_subsets(n_channels: int, n_subsets: int = 100) -> List[Set[int]]:
    """
    Generate random subsets of channel indices with no duplicates.

    Args:
        n_channels: Total number of channels in the layer
        n_subsets: Number of unique random subsets to generate (default: 100)

    Returns:
        List of sets, where each set contains indices of channels to mask
    """
    subsets = []
    seen_subsets = set()

    max_attempts = n_subsets * 100  # Prevent infinite loop
    attempts = 0

    while len(subsets) < n_subsets and attempts < max_attempts:
        attempts += 1

        # Random subset size from 1 to n_channels
        subset_size = np.random.randint(1, n_channels + 1)

        # Randomly select channels to mask
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


class ChannelMasker:
    """
    Handles masking of channels in convolutional layers.
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.original_weights = {}
        self.conv_layers = self._get_conv_layers()

    def _get_conv_layers(self) -> List[torch.nn.Conv2d]:
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
            List of tuples (layer_name, n_channels)
        """
        info = []
        for name, layer in self.conv_layers:
            n_channels = layer.out_channels
            info.append((name, n_channels))
        return info

    def mask_channels(self, layer_idx: int, channels_to_mask: Set[int]):
        """
        Mask specified channels in a specific layer by setting their weights to zero.

        Args:
            layer_idx: Index of the layer in self.conv_layers
            channels_to_mask: Set of channel indices to mask
        """
        if layer_idx >= len(self.conv_layers):
            raise ValueError(f"Layer index {layer_idx} out of range")

        layer_name, layer = self.conv_layers[layer_idx]

        # Save original weights if not already saved
        if layer_name not in self.original_weights:
            self.original_weights[layer_name] = layer.weight.data.clone()

        # Mask the specified channels
        with torch.no_grad():
            for channel_idx in channels_to_mask:
                layer.weight.data[channel_idx] = 0

    def unmask_layer(self, layer_idx: int):
        """
        Restore original weights for a specific layer.

        Args:
            layer_idx: Index of the layer in self.conv_layers
        """
        if layer_idx >= len(self.conv_layers):
            raise ValueError(f"Layer index {layer_idx} out of range")

        layer_name, layer = self.conv_layers[layer_idx]

        if layer_name in self.original_weights:
            with torch.no_grad():
                layer.weight.data = self.original_weights[layer_name].clone()

    def unmask_all(self):
        """Restore all layers to their original weights."""
        for layer_idx in range(len(self.conv_layers)):
            self.unmask_layer(layer_idx)


def forward_with_mask(model, dataloader, layer_idx: int,
                     channels_to_mask: Set[int], device='cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform forward pass with masked channels and collect predictions.

    Args:
        model: The neural network model
        dataloader: DataLoader for the dataset
        layer_idx: Index of the layer to mask
        channels_to_mask: Set of channel indices to mask
        device: Device to run on

    Returns:
        Tuple of (predictions, true_labels) as numpy arrays
    """
    masker = ChannelMasker(model, device)

    # Apply mask
    masker.mask_channels(layer_idx, channels_to_mask)

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

    # Unmask
    masker.unmask_layer(layer_idx)

    return np.array(all_predictions), np.array(all_labels)
