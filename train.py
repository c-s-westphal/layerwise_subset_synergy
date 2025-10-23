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
from models.resnet import ResNet20, ResNet32, ResNet56, ResNet74


def get_cifar10_loaders(batch_size=128):
    """
    Get CIFAR-10 train and test data loaders with standard augmentations.
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalization only for test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
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


def train_model(model_name, epochs=500, batch_size=128, lr=0.001,
                target_train_acc=99.99, device='cuda', seed=None,
                checkpoint_dir='checkpoints', log_dir='logs'):
    """
    Train a VGG or ResNet model with AdamW and cosine annealing learning rate schedule.
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
        'vgg19': vgg19,
        'resnet20': ResNet20,
        'resnet32': ResNet32,
        'resnet56': ResNet56,
        'resnet74': ResNet74
    }

    model = model_dict[model_name]().to(device)

    # Get data loaders
    trainloader, testloader = get_cifar10_loaders(batch_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler: cosine annealing over full training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    train_acc_reached = False

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}/{epochs}")

        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test(model, testloader, criterion, device)

        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")

        # Step the learning rate scheduler
        scheduler.step()

        # Check if target train accuracy reached - STOP IMMEDIATELY
        if train_acc >= target_train_acc:
            print(f"\n{'*'*60}")
            print(f"Target train accuracy {target_train_acc}% reached!")
            print(f"Epoch: {epoch + 1}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
            print(f"Stopping training and saving checkpoint...")
            print(f"{'*'*60}\n")

            # Save checkpoint immediately
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'train_acc': train_acc,
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

    # If target not reached, save final checkpoint
    if not train_acc_reached:
        print(f"\n{'!'*60}")
        print(f"WARNING: Target train accuracy {target_train_acc}% NOT reached")
        print(f"Final Train Acc: {train_acc:.2f}% after {epochs} epochs")
        print(f"Saving final checkpoint anyway...")
        print(f"{'!'*60}\n")

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
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pth'))
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'{model_name}_latest.pth'))

    print(f"\nTraining complete for {model_name}")
    print(f"Final Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    return model, train_acc_reached


def main():
    parser = argparse.ArgumentParser(description='Train VGG and ResNet models on CIFAR-10 with AdamW')
    parser.add_argument('--model', type=str, default='all',
                       choices=['vgg11', 'vgg13', 'vgg16', 'vgg19',
                               'resnet20', 'resnet32', 'resnet56', 'resnet74', 'all'],
                       help='Model to train (default: all)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Max number of epochs (default: 500)')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=128,
                       dest='batch_size',
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate for AdamW (default: 0.001)')
    parser.add_argument('--target_train_acc', '--target-acc', type=float, default=99.99,
                       dest='target_train_acc',
                       help='Target train accuracy for early stopping (default: 99.99)')
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
        models_to_train = ['vgg11', 'vgg13', 'vgg16', 'vgg19',
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
