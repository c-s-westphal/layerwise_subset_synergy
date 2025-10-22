#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=vgg11
#SBATCH --array=1-10
#SBATCH --output=logs/vgg11_%A_%a.out
#SBATCH --error=logs/vgg11_%A_%a.err
set -euo pipefail

hostname
date

number=$SLURM_ARRAY_TASK_ID
paramfile="scripts/jobs_vgg11.txt"

# ---------------------------------------------------------------------
# 1.  Activate virtual environment
# ---------------------------------------------------------------------
# Activate venv using full path
source ~/vgg11_dropout_robustness/venv/bin/activate

# ---------------------------------------------------------------------
# 2.  Keep Matplotlib out of home quota
# ---------------------------------------------------------------------
export MPLCONFIGDIR="$TMPDIR/mplcache"
mkdir -p "$MPLCONFIGDIR"

# ---------------------------------------------------------------------
# 3.  Create output directories
# ---------------------------------------------------------------------
mkdir -p checkpoints
mkdir -p results
mkdir -p data
mkdir -p logs

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format: seed
seed=$(sed -n ${number}p "$paramfile" | awk '{print $1}')

date
echo "Training: VGG11 | Seed: $seed"

# ---------------------------------------------------------------------
# 5.  Run training
# ---------------------------------------------------------------------
echo "Starting training..."
python -u train.py \
    --model vgg11 \
    --seed $seed \
    --epochs 500 \
    --batch_size 128 \
    --lr 0.001 \
    --target_train_acc 99.99 \
    --checkpoint_dir checkpoints \
    --log_dir logs

date
echo "Training completed successfully: VGG11 | Seed $seed"

# ---------------------------------------------------------------------
# 6.  Run evaluation with layer-wise masking
# ---------------------------------------------------------------------
echo "Starting evaluation with layer-wise masking..."

checkpoint_dir="checkpoints/vgg11_seed${seed}"

# Try checkpoint_latest.pth first, then fall back to checkpoint_epoch_200.pth
if [ -f "$checkpoint_dir/checkpoint_latest.pth" ]; then
    checkpoint_file="$checkpoint_dir/checkpoint_latest.pth"
elif [ -f "$checkpoint_dir/checkpoint_epoch_200.pth" ]; then
    checkpoint_file="$checkpoint_dir/checkpoint_epoch_200.pth"
else
    echo "Error: No checkpoint file found in $checkpoint_dir"
    echo "Contents of checkpoint directory:"
    ls -la "$checkpoint_dir" || true
    exit 1
fi

echo "Found checkpoint: $checkpoint_file"

# Run layer-wise evaluation (1000 subsets per layer)
# Note: checkpoint will be kept (not deleted after evaluation)
python -u evaluate.py \
    --checkpoint "$checkpoint_file" \
    --model vgg11 \
    --n_subsets 100 \
    --output_dir data \
    --device cuda \
    --seed $seed

date
echo "Evaluation completed successfully: VGG11 | Seed $seed"
