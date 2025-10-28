import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with pre-activation LayerNorm.

    Architecture:
        identity = x
        x = LayerNorm(x)
        x = Linear(hidden_dim → hidden_dim)
        x = ReLU(x)
        x = Dropout(x)
        x = x + identity  # Residual connection
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x + identity
        return x


class MLP_Unified(nn.Module):
    """
    Unified MLP with variable depth (n_layers = 2-6) for MNIST.

    Architecture:
        Input (784) → Input Projection → (n_layers - 1) Residual Blocks → Final LayerNorm → Classifier (10)

    For n_layers = L:
        - 1 input projection layer
        - L - 1 residual blocks
        - 1 final LayerNorm
        - 1 classifier layer

    Each layer output (input projection + residual blocks) is 256-dimensional.
    """
    def __init__(self, n_layers, input_dim=784, hidden_dim=256, num_classes=10, dropout=0.0):
        super(MLP_Unified, self).__init__()

        assert n_layers >= 2, "n_layers must be >= 2 for masking experiments"

        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)

        # Initialize input projection
        nn.init.kaiming_normal_(self.input_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.input_proj.bias)

        # Residual blocks (n_layers - 1 blocks)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(n_layers - 1)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)  # (batch_size, 784)

        # Input projection
        x = self.input_proj(x)
        x = self.input_relu(x)
        x = self.input_dropout(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Classifier
        x = self.classifier(x)

        return x

    def get_maskable_layers(self):
        """
        Returns a list of all layers where we can apply masking.
        This includes:
          - Output of input projection (after ReLU, before dropout)
          - Output of each residual block

        Returns:
            List of (layer_idx, layer_name, module) tuples
        """
        maskable = []

        # Layer 0: Input projection output (after input_relu)
        maskable.append((0, 'input_proj', self.input_relu))

        # Layers 1 to n_layers-1: Residual block outputs
        for i, block in enumerate(self.blocks):
            maskable.append((i + 1, f'block_{i}', block))

        return maskable

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def mlp2(input_dim=784, hidden_dim=256, num_classes=10, dropout=0.0):
    """2-layer MLP: input projection + 1 residual block"""
    return MLP_Unified(n_layers=2, input_dim=input_dim, hidden_dim=hidden_dim,
                       num_classes=num_classes, dropout=dropout)


def mlp3(input_dim=784, hidden_dim=256, num_classes=10, dropout=0.0):
    """3-layer MLP: input projection + 2 residual blocks"""
    return MLP_Unified(n_layers=3, input_dim=input_dim, hidden_dim=hidden_dim,
                       num_classes=num_classes, dropout=dropout)


def mlp4(input_dim=784, hidden_dim=256, num_classes=10, dropout=0.0):
    """4-layer MLP: input projection + 3 residual blocks"""
    return MLP_Unified(n_layers=4, input_dim=input_dim, hidden_dim=hidden_dim,
                       num_classes=num_classes, dropout=dropout)


def mlp5(input_dim=784, hidden_dim=256, num_classes=10, dropout=0.0):
    """5-layer MLP: input projection + 4 residual blocks"""
    return MLP_Unified(n_layers=5, input_dim=input_dim, hidden_dim=hidden_dim,
                       num_classes=num_classes, dropout=dropout)


def mlp6(input_dim=784, hidden_dim=256, num_classes=10, dropout=0.0):
    """6-layer MLP: input projection + 5 residual blocks"""
    return MLP_Unified(n_layers=6, input_dim=input_dim, hidden_dim=hidden_dim,
                       num_classes=num_classes, dropout=dropout)
