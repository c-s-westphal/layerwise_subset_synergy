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
    Handles masking of channel activations after ReLU layers using hooks.
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.relu_layers = self._get_relu_layers()
        self.activation_stats = {}  # No longer needed but kept for compatibility
        self.hook_handle = None
        self.channels_to_mask = None
        self.noise_range = None  # No longer needed but kept for compatibility

    def _get_relu_layers(self) -> List[Tuple[str, torch.nn.ReLU]]:
        """Get all ReLU layers from the model (after Conv2d and BN)."""
        relu_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                relu_layers.append((name, module))
        return relu_layers

    def get_layer_info(self) -> List[Tuple[str, int]]:
        """
        Get information about each ReLU layer by running a dummy forward pass.

        Returns:
            List of tuples (layer_name, n_output_channels)
        """
        info = []

        # Create a dummy input (batch_size=1, channels=3, height=32, width=32 for CIFAR-10)
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)

        # Store outputs from each ReLU layer
        outputs = {}
        handles = []

        def create_hook(layer_name):
            def hook(module, input, output):
                outputs[layer_name] = output
            return hook

        # Register hooks on all ReLU layers
        for name, layer in self.relu_layers:
            handle = layer.register_forward_hook(create_hook(name))
            handles.append(handle)

        # Run forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(dummy_input)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Extract channel counts from outputs
        for name, _ in self.relu_layers:
            if name in outputs:
                # Output shape is (batch, channels, height, width)
                n_channels = outputs[name].shape[1]
                info.append((name, n_channels))

        return info

    def compute_activation_stats(self, dataloader, layer_idx: int):
        """
        No-op: Activation statistics are not needed when replacing with zeros.

        Args:
            dataloader: DataLoader for the dataset
            layer_idx: Index of the layer in self.relu_layers
        """
        # No longer needed since we replace with zeros
        if layer_idx >= len(self.relu_layers):
            raise ValueError(f"Layer index {layer_idx} out of range")

        layer_name, layer = self.relu_layers[layer_idx]

        # Mark as "computed" for compatibility
        if layer_name not in self.activation_stats:
            self.activation_stats[layer_name] = None
            print(f"  Layer {layer_name}: Will replace masked channels with zeros")

    def _masking_hook(self, module, input, output):
        """
        Hook function that masks selected channels with zeros.
        """
        if self.channels_to_mask is None:
            return output

        # Clone output to avoid in-place modification issues
        masked_output = output.clone()

        # Replace selected channels with zeros
        for channel_idx in self.channels_to_mask:
            masked_output[:, channel_idx, :, :] = 0.0

        return masked_output

    def mask_activations(self, layer_idx: int, channels_to_mask: Set[int]):
        """
        Set up masking for specified channels in a layer.

        Args:
            layer_idx: Index of the layer in self.relu_layers
            channels_to_mask: Set of channel indices to mask
        """
        if layer_idx >= len(self.relu_layers):
            raise ValueError(f"Layer index {layer_idx} out of range")

        layer_name, layer = self.relu_layers[layer_idx]

        # Ensure we have "computed" activation stats (for compatibility)
        if layer_name not in self.activation_stats:
            raise ValueError(f"Must compute activation stats for {layer_name} first")

        # Store masking configuration
        self.channels_to_mask = channels_to_mask

        # Remove existing hook if any
        if self.hook_handle is not None:
            self.hook_handle.remove()

        # Register new hook on the ReLU layer
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
