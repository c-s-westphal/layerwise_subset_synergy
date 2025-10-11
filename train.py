import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm

from models import vgg11, vgg13, vgg16, vgg19


def get_cifar10_loaders(batch_size=128):
    """
    Get CIFAR-10 train and test data loaders (no augmentation).
    """
    # Only normalization for both train and test
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def train_epoch(model, trainloader, criterion, optimizer, device):
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

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
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


def train_model(model_name, epochs=200, batch_size=128, lr=0.01,
                target_train_acc=None, device='cuda', seed=None,
                weight_decay=5e-4, momentum=0.9, optimizer_name='sgd',
                lr_schedule=False, lr_milestones=None, lr_gamma=0.1,
                checkpoint_dir='checkpoints', log_dir='logs'):
    """
    Train a VGG model.
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
        'vgg11': vgg11,
        'vgg13': vgg13,
        'vgg16': vgg16,
        'vgg19': vgg19
    }

    model = model_dict[model_name]().to(device)

    # Get data loaders
    trainloader, testloader = get_cifar10_loaders(batch_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                             momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Learning rate scheduler
    if lr_schedule and lr_milestones is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=lr_milestones,
                                                   gamma=lr_gamma)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_test_acc = 0
    train_acc_reached = False

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}/{epochs}")

        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test(model, testloader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_test_acc:
            print(f"Saving best model (Test Acc: {test_acc:.2f}%)")
            best_test_acc = test_acc
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            # Save to seed-specific directory if seed is provided
            if seed is not None:
                save_dir = os.path.join(checkpoint_dir, f'{model_name}_seed{seed}')
                os.makedirs(save_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'))
            else:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_name}_best.pth'))

        # Check if target train accuracy reached (optional)
        if target_train_acc is not None and train_acc >= target_train_acc and not train_acc_reached:
            print(f"\n{'*'*50}")
            print(f"Target train accuracy {target_train_acc}% reached!")
            print(f"Epoch: {epoch + 1}, Train Acc: {train_acc:.2f}%")
            print(f"{'*'*50}\n")
            train_acc_reached = True

    # Final save
    checkpoint = {
        'model': model.state_dict(),
        'epoch': epochs,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'optimizer': optimizer.state_dict(),
    }
    if seed is not None:
        save_dir = os.path.join(checkpoint_dir, f'{model_name}_seed{seed}')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epochs}.pth'))
    else:
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_name}_final.pth'))

    print(f"\nTraining complete for {model_name}")
    print(f"Best Test Acc: {best_test_acc:.2f}%")
    print(f"Final Train Acc: {train_acc:.2f}%")

    return model, train_acc_reached


def main():
    parser = argparse.ArgumentParser(description='Train VGG models on CIFAR-10')
    parser.add_argument('--model', type=str, default='all',
                       choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'all'],
                       help='Model to train (default: all)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs (default: 200)')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=128,
                       dest='batch_size',
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--target_train_acc', '--target-acc', type=float, default=None,
                       dest='target_train_acc',
                       help='Target train accuracy (default: None - disabled)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay (default: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam'],
                       help='Optimizer (default: sgd)')
    parser.add_argument('--lr_schedule', action='store_true',
                       help='Use MultiStepLR scheduler with milestones')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=None,
                       help='Learning rate milestones (default: None)')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                       help='Learning rate decay gamma (default: 0.1)')
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
        models_to_train = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
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
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            optimizer_name=args.optimizer,
            lr_schedule=args.lr_schedule,
            lr_milestones=args.lr_milestones,
            lr_gamma=args.lr_gamma,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
        results[model_name] = acc_reached

    # Summary
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    if args.target_train_acc is not None:
        for model_name, acc_reached in results.items():
            status = "✓" if acc_reached else "✗"
            print(f"{model_name}: {status} (Target {args.target_train_acc}% train acc)")
    else:
        for model_name in results.keys():
            print(f"{model_name}: ✓ Training complete")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
