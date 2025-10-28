import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm

from models import mlp2, mlp3, mlp4, mlp5, mlp6


def get_mnist_loaders(batch_size=512):
    """
    Get MNIST train and test data loaders.
    Uses full 60,000 training samples.
    """
    # MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def separate_parameters_for_weight_decay(model, weight_decay):
    """
    Separate parameters into two groups:
    - decay_params: Linear layer weights (apply weight_decay)
    - no_decay_params: Biases and LayerNorm parameters (no weight_decay)
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay on biases or LayerNorm parameters
        if 'bias' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            # Apply weight decay to Linear weights
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def train_epoch(model, trainloader, criterion, optimizer, device, grad_clip=1.0):
    """
    Train for one epoch.
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(trainloader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # Optimizer step
        optimizer.step()

        # Statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': train_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    return train_loss / len(trainloader), 100. * correct / total


def test(model, testloader, criterion, device):
    """
    Evaluate on test set.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss / len(testloader), 100. * correct / total


def train_model(model_name, epochs=600, batch_size=512, lr=1e-3,
                target_train_acc=99.99, weight_decay=0.0, dropout=0.0,
                warmup_epochs=5, grad_clip=1.0, device='cuda', seed=None,
                checkpoint_dir='checkpoints', log_dir='logs'):
    """
    Train an MLP model with the specified training regime.

    Training hyperparameters:
    - Optimizer: AdamW with selective weight decay
    - Learning rate schedule: Linear warmup (5 epochs) + Cosine annealing
    - Gradient clipping: 1.0
    - Early stopping: Stop when train_acc >= target_train_acc
    """
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    if seed is not None:
        print(f"Seed: {seed}")
    print(f"{'='*50}\n")

    # Initialize model
    model_dict = {
        'mlp2': mlp2,
        'mlp3': mlp3,
        'mlp4': mlp4,
        'mlp5': mlp5,
        'mlp6': mlp6
    }

    model = model_dict[model_name](dropout=dropout).to(device)

    print(f"Model: {model_name}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Hidden dim: 256")
    print(f"Dropout: {dropout}")
    print(f"Target train accuracy: {target_train_acc}%")
    print()

    # Get data loaders
    trainloader, testloader = get_mnist_loaders(batch_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Separate parameters for selective weight decay
    param_groups = separate_parameters_for_weight_decay(model, weight_decay)
    optimizer = optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate schedule: Warmup + Cosine annealing
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # Create checkpoint directory
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}{seed_suffix}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Training loop
    best_train_acc = 0.0
    epoch_reached_target = -1

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device, grad_clip
        )
        test_loss, test_acc = test(model, testloader, criterion, device)

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_loss': train_loss,
                'test_loss': test_loss
            }
            torch.save(checkpoint, os.path.join(checkpoint_path, 'checkpoint_best.pth'))

        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss
        }
        torch.save(checkpoint, os.path.join(checkpoint_path, 'checkpoint_latest.pth'))

        # Early stopping
        if train_acc >= target_train_acc:
            print(f"\n{'='*50}")
            print(f"Target accuracy {target_train_acc}% reached at epoch {epoch}!")
            print(f"{'='*50}\n")
            epoch_reached_target = epoch
            break

    # Final evaluation
    final_train_loss, final_train_acc = test(model, trainloader, criterion, device)
    final_test_loss, final_test_acc = test(model, testloader, criterion, device)

    print(f"\nFinal Results:")
    print(f"Train Acc: {final_train_acc:.2f}%")
    print(f"Test Acc: {final_test_acc:.2f}%")
    print(f"Generalization Gap: {final_train_acc - final_test_acc:.2f}%")

    return model, epoch_reached_target != -1


def main():
    parser = argparse.ArgumentParser(description='Train MLP models on MNIST')

    parser.add_argument('--model', type=str, default='mlp2',
                       choices=['mlp2', 'mlp3', 'mlp4', 'mlp5', 'mlp6', 'all'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=600,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--target_train_acc', type=float, default=99.99,
                       help='Target training accuracy for early stopping')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (applied only to Linear weights)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Train models
    if args.model == 'all':
        models_to_train = ['mlp2', 'mlp3', 'mlp4', 'mlp5', 'mlp6']
    else:
        models_to_train = [args.model]

    results = {}
    for model_name in models_to_train:
        model, acc_reached = train_model(
            model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            target_train_acc=args.target_train_acc,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            warmup_epochs=args.warmup_epochs,
            grad_clip=args.grad_clip,
            device=args.device,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
        results[model_name] = acc_reached

    # Summary
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    for model_name, acc_reached in results.items():
        status = "✓" if acc_reached else "✗"
        print(f"{model_name}: {status} (Target {args.target_train_acc}% train acc)")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
