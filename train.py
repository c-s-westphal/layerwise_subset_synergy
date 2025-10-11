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


def train_model(model_name, epochs=200, batch_size=128, lr=0.01,
                target_train_acc=99.99, device='cuda'):
    """
    Train a VGG model until it reaches target train accuracy.
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
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
    optimizer = optim.SGD(model.parameters(), lr=lr,
                         momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler
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
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(checkpoint, f'checkpoints/{model_name}_best.pth')

        # Check if target train accuracy reached
        if train_acc >= target_train_acc and not train_acc_reached:
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
    torch.save(checkpoint, f'checkpoints/{model_name}_final.pth')

    if not train_acc_reached:
        print(f"\n{'!'*50}")
        print(f"WARNING: Target train accuracy {target_train_acc}% NOT reached!")
        print(f"Final Train Acc: {train_acc:.2f}%")
        print(f"Consider training for more epochs.")
        print(f"{'!'*50}\n")

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
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--target-acc', type=float, default=99.99,
                       help='Target train accuracy (default: 99.99)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

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
            target_train_acc=args.target_acc,
            device=args.device
        )
        results[model_name] = acc_reached

    # Summary
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    for model_name, acc_reached in results.items():
        status = "✓" if acc_reached else "✗"
        print(f"{model_name}: {status} (Target {args.target_acc}% train acc)")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
