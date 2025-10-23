"""
ResNet models (20, 32, 56) for CIFAR-10 with configurable batch normalization.

Based on "Deep Residual Learning for Image Recognition" (He et al., 2016)
and adapted for CIFAR-10 following the original paper's CIFAR-10 implementation.

Architecture:
- Initial conv layer (3×3, 16 filters)
- 3 stages with [n, n, n] blocks
- Filters: [16, 32, 64]
- Downsampling via stride=2 at start of stage 2 and stage 3
- Global Average Pooling + FC layer

Total layers = 1 (initial) + 6n + 1 (FC) = 6n + 2
- ResNet-20: n=3, layers=20
- ResNet-32: n=5, layers=32
- ResNet-56: n=9, layers=56
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-20/32/56 on CIFAR-10."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(BasicBlock, self).__init__()
        self.use_batchnorm = use_batchnorm

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(planes) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=False)  # inplace=False for masking compatibility

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(planes) if use_batchnorm else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # Option A: zero-padding for increased dimensions (original paper for CIFAR)
            # Option B: 1x1 conv projection (more common in modern implementations)
            # We use Option B for better performance
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=not use_batchnorm),
                nn.BatchNorm2d(self.expansion * planes) if use_batchnorm else nn.Identity()
            )

        self.relu2 = nn.ReLU(inplace=False)  # inplace=False for masking compatibility

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    """ResNet model for CIFAR-10.

    Args:
        block: Residual block type (BasicBlock)
        num_blocks: List of number of blocks in each stage [n, n, n]
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        use_batchnorm: Whether to use batch normalization (default: True)
        use_dropout: Whether to use dropout before classifier (default: False)
    """
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=True, use_dropout=False):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        # Initial conv block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(16) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=False)  # inplace=False for masking compatibility

        # Residual layers
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, use_batchnorm=use_batchnorm)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, use_batchnorm=use_batchnorm)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, use_batchnorm=use_batchnorm)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(64 * block.expansion, num_classes)
            )
        else:
            self.classifier = nn.Linear(64 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride, use_batchnorm):
        """Create a residual layer with num_blocks blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv
        out = self.relu(self.bn1(self.conv1(x)))

        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Global average pooling + classifier
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def ResNet20(num_classes=10, use_batchnorm=True, use_dropout=False):
    """ResNet-20 (n=3): 6*3 + 2 = 20 layers."""
    return ResNet(BasicBlock, [3, 3, 3], num_classes, use_batchnorm, use_dropout)


def ResNet32(num_classes=10, use_batchnorm=True, use_dropout=False):
    """ResNet-32 (n=5): 6*5 + 2 = 32 layers."""
    return ResNet(BasicBlock, [5, 5, 5], num_classes, use_batchnorm, use_dropout)


def ResNet56(num_classes=10, use_batchnorm=True, use_dropout=False):
    """ResNet-56 (n=9): 6*9 + 2 = 56 layers."""
    return ResNet(BasicBlock, [9, 9, 9], num_classes, use_batchnorm, use_dropout)


def ResNet74(num_classes=10, use_batchnorm=True, use_dropout=False):
    """ResNet-74 (n=12): 6*12 + 2 = 74 layers."""
    return ResNet(BasicBlock, [12, 12, 12], num_classes, use_batchnorm, use_dropout)


if __name__ == '__main__':
    # Test model creation
    for arch_name, arch_fn in [
        ('resnet20', ResNet20),
        ('resnet32', ResNet32),
        ('resnet56', ResNet56),
        ('resnet74', ResNet74),
    ]:
        print(f"\n{arch_name.upper()}:")

        # With batch norm
        model_bn = arch_fn(num_classes=10, use_batchnorm=True, use_dropout=False)
        total_params = sum(p.numel() for p in model_bn.parameters())
        print(f"  With BN: {total_params:,} parameters")

        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        y = model_bn(x)
        assert y.shape == (1, 10), f"Expected shape (1, 10), got {y.shape}"
        print(f"  Forward pass: OK (output shape: {y.shape})")
