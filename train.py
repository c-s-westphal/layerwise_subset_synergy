import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment, RandAugment
import os
import argparse
from tqdm import tqdm
import numpy as np

from models import vgg9, vgg11, vgg13, vgg16, vgg19
from models.resnet import ResNet20, ResNet32, ResNet56, ResNet74


def get_cifar100_loaders(batch_size=256, return_clean_train=False):
    """
    Get CIFAR-100 train and test data loaders with data augmentation.
    Training uses RandomCrop, RandomHorizontalFlip, and RandAugment.

    Args:
        batch_size: Batch size for data loaders
        return_clean_train: If True, also return a clean (non-augmented) train loader

    Returns:
        trainloader, testloader (and optionally clean_trainloader)
    """
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Test transforms - only normalization (also used for clean train evaluation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    if return_clean_train:
        # Clean train set for early stopping evaluation (no augmentation)
        clean_trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=test_transform)
        clean_trainloader = DataLoader(
            clean_trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        return trainloader, testloader, clean_trainloader

    return trainloader, testloader


def cutmix_data(x, y, alpha=0.5):
    """
    Apply CutMix augmentation.

    Args:
        x: Input images (batch)
        y: Labels (batch)
        alpha: CutMix hyperparameter

    Returns:
        Mixed inputs, labels_a, labels_b, lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Generate random box
    _, _, H, W = x.size()
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling of center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def train_epoch(model, trainloader, criterion, optimizer, device, use_cutmix=True, cutmix_alpha=0.5):
    """
    Train for one epoch with CutMix augmentation and gradient clipping.
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(trainloader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply CutMix with 50% probability
        if use_cutmix and np.random.rand() < 0.5:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            # For accuracy, use the primary target
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().item() +
                       (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

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


def train_model(model_name, epochs=500, batch_size=256, lr=0.001,
                target_train_acc=None, device='cuda', seed=None,
                checkpoint_dir='checkpoints', log_dir='logs', warmup_epochs=10):
    """
    Train a VGG or ResNet model with AdamW, warmup, and cosine annealing learning rate schedule.
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

    model = model_dict[model_name](num_classes=100).to(device)

    # Get data loaders (including clean train for early stopping evaluation)
    trainloader, testloader, clean_trainloader = get_cifar100_loaders(batch_size, return_clean_train=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001, betas=(0.9, 0.999))

    # Learning rate scheduler: warmup (10 epochs) + cosine annealing (490 epochs)
    # Warmup from 1e-6 to 1e-3, then cosine decay to 1e-6
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    train_acc_reached = False

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}/{epochs}")

        # Train on augmented data
        train_loss, train_acc_aug = train_epoch(
            model, trainloader, criterion, optimizer, device)

        # Evaluate on clean train data (for early stopping)
        clean_train_loss, clean_train_acc = test(model, clean_trainloader, criterion, device)

        # Evaluate on test data
        test_loss, test_acc = test(model, testloader, criterion, device)

        print(f"Train Loss (aug): {train_loss:.3f} | Train Acc (aug): {train_acc_aug:.2f}%")
        print(f"Train Acc (clean): {clean_train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # Step the learning rate scheduler
        scheduler.step()

        # Check if target train accuracy reached (if specified) - STOP IMMEDIATELY
        # Use CLEAN train accuracy for early stopping
        if target_train_acc is not None and clean_train_acc >= target_train_acc:
            print(f"\n{'*'*60}")
            print(f"Target train accuracy {target_train_acc}% reached!")
            print(f"Epoch: {epoch + 1}, Clean Train Acc: {clean_train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
            print(f"Stopping training and saving checkpoint...")
            print(f"{'*'*60}\n")

            # Save checkpoint immediately
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'train_acc': clean_train_acc,
                'train_acc_aug': train_acc_aug,
                'test_acc': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            if seed is not None:
                save_dir = os.path.join(checkpoint_dir, f'{model_name}_seed{seed}')
                os.makedirs(save_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'))
            else:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_name}_latest.pth'))

            train_acc_reached = True
            break

    # Save final checkpoint after all epochs
    if not train_acc_reached:
        if target_train_acc is not None:
            print(f"\n{'!'*60}")
            print(f"WARNING: Target train accuracy {target_train_acc}% NOT reached")
            print(f"Final Clean Train Acc: {clean_train_acc:.2f}% after {epochs} epochs")
            print(f"Saving final checkpoint anyway...")
            print(f"{'!'*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"Training completed after {epochs} epochs")
            print(f"Final Clean Train Acc: {clean_train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
            print(f"Saving final checkpoint...")
            print(f"{'='*60}\n")

        checkpoint = {
            'model': model.state_dict(),
            'epoch': epochs,
            'train_acc': clean_train_acc,
            'train_acc_aug': train_acc_aug,
            'test_acc': test_acc,
            'optimizer': optimizer.state_dict(),
        }
        if seed is not None:
            save_dir = os.path.join(checkpoint_dir, f'{model_name}_seed{seed}')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'))
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_name}_latest.pth'))

    print(f"\nTraining complete for {model_name}")
    print(f"Final Clean Train Acc: {clean_train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    return model, train_acc_reached


def main():
    parser = argparse.ArgumentParser(description='Train VGG and ResNet models on CIFAR-100 with AdamW')
    parser.add_argument('--model', type=str, default='all',
                       choices=['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                               'resnet20', 'resnet32', 'resnet56', 'resnet74', 'all'],
                       help='Model to train (default: all)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Max number of epochs (default: 500)')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=256,
                       dest='batch_size',
                       help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate for AdamW (default: 0.001)')
    parser.add_argument('--target_train_acc', '--target-acc', type=float, default=99.0,
                       dest='target_train_acc',
                       help='Target train accuracy for early stopping (default: 99.0)')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Number of warmup epochs (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory (default: logs)')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Train models
    if args.model == 'all':
        models_to_train = ['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                          'resnet20', 'resnet32', 'resnet56', 'resnet74']
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
            device=args.device,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            warmup_epochs=args.warmup_epochs
        )
        results[model_name] = acc_reached

    # Summary
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    for model_name, acc_reached in results.items():
        if args.target_train_acc is not None:
            status = "✓" if acc_reached else "✗"
            print(f"{model_name}: {status} (Target {args.target_train_acc}% train acc)")
        else:
            status = "✓"
            print(f"{model_name}: {status} (Completed {args.epochs} epochs)")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
