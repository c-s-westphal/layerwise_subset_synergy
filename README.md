# Layerwise Subset Synergy

This repository implements VGG architectures (VGG11, VGG13, VGG16, VGG19) for CIFAR-10 classification and analyzes the mutual information between true labels and predictions when random subsets of channels are masked at each layer.

## Overview

The project:
1. Trains VGG models on CIFAR-10 until reaching 99.99% train accuracy
2. Masks random subsets of channels in each convolutional layer independently
3. Performs forward passes with masked models
4. Calculates mutual information between true labels and masked predictions
5. Analyzes how channel masking affects information flow at each layer

## Repository Structure

```
.
├── models/                 # VGG model implementations
│   ├── __init__.py
│   └── vgg.py             # VGG11, 13, 16, 19 architectures
├── data/                   # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/           # Saved model checkpoints
├── results/               # Evaluation results (JSON)
├── train.py               # Training script
├── evaluate.py            # Evaluation script with masking
├── masking.py             # Channel masking utilities
├── mutual_information.py  # MI calculation utilities
└── requirements.txt       # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd layerwise_subset_synergy

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Training Models

Train all VGG models (11, 13, 16, 19):

```bash
python train.py --model all
```

Train a specific model:

```bash
python train.py --model vgg16
```

Training options:
- `--model`: Model to train (vgg11, vgg13, vgg16, vgg19, all)
- `--epochs`: Number of epochs (default: 200)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.01)
- `--target-acc`: Target train accuracy threshold (default: 99.99)
- `--device`: Device to use (cuda or cpu, default: cuda)

The script will:
- Train with SGD (momentum=0.9, weight_decay=5e-4)
- Use cosine annealing learning rate scheduler
- Apply standard data augmentation (random crop, horizontal flip)
- Monitor train accuracy and flag when 99.99% is reached
- Save best model (by test accuracy) and final model

### 2. Evaluating with Channel Masking

Evaluate all trained models:

```bash
python evaluate.py --model all
```

Evaluate a specific model:

```bash
python evaluate.py --model vgg16 --checkpoint best
```

Evaluation options:
- `--model`: Model to evaluate (vgg11, vgg13, vgg16, vgg19, all)
- `--checkpoint`: Which checkpoint to use (best or final, default: best)
- `--n-subsets`: Number of random subsets per layer (default: 100)
- `--batch-size`: Batch size (default: 128)
- `--device`: Device to use (cuda or cpu, default: cuda)

The script will:
- Load the trained model
- For each convolutional layer independently:
  - Generate 100 unique random subsets of channels (size 1 to n_channels)
  - Mask those channels (set weights to 0)
  - Perform forward pass on test set
  - Collect predictions
- Calculate mutual information between true labels and predictions for each subset
- Compute mean and std of MI across all subsets for each layer
- Save results to `results/mi_results_{checkpoint}.json`

## Training Hyperparameters

Standard hyperparameters used:
- **Optimizer**: SGD with momentum=0.9
- **Learning Rate**: 0.01 (cosine annealing)
- **Batch Size**: 128
- **Epochs**: 200
- **Weight Decay**: 5e-4
- **Data Augmentation**: Random crop (32x32 with padding=4), horizontal flip
- **Normalization**: CIFAR-10 standard (mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

## Channel Masking Strategy

For each layer with `n` channels:
1. Generate 100 unique random subsets
2. Each subset is created by:
   - Randomly selecting a subset size `k` from 1 to `n`
   - Randomly selecting `k` channels to mask
   - Ensuring no duplicate subsets
3. Average subset size ≈ `n/2`
4. Mask channels by setting their convolutional weights to 0
5. Only one layer is masked at a time

## Mutual Information Calculation

- Uses discrete MI: `I(Y; Y_hat) = Σ p(y, y_hat) log(p(y, y_hat) / (p(y)p(y_hat)))`
- Measures how much information predictions provide about true labels
- Calculated using `sklearn.metrics.mutual_info_score`
- Results include mean and standard deviation across all subsets per layer

## Output

Results are saved in JSON format:

```json
{
  "vgg16": {
    "model_name": "vgg16",
    "checkpoint_path": "checkpoints/vgg16_best.pth",
    "n_subsets": 100,
    "layers": [
      {
        "layer_idx": 0,
        "layer_name": "features.0",
        "n_channels": 64,
        "mean_mi": 1.2345,
        "std_mi": 0.0123
      },
      ...
    ]
  }
}
```

## Example Workflow

```bash
# 1. Train all models
python train.py --model all --epochs 200

# 2. Evaluate with best checkpoints
python evaluate.py --model all --checkpoint best --n-subsets 100

# 3. Check results
cat results/mi_results_best.json
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- NumPy 1.24+
- scikit-learn 1.3+
- tqdm 4.65+
- matplotlib 3.7+

## Notes

- Training to 99.99% train accuracy may require more than 200 epochs for some models
- If target accuracy is not reached, a warning will be displayed
- CUDA is recommended for faster training and evaluation
- CIFAR-10 dataset (~170MB) will be automatically downloaded on first run
