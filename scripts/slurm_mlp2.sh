#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mlp2
#SBATCH --array=1-10
#SBATCH --output=logs/mlp2_%A_%a.out
#SBATCH --error=logs/mlp2_%A_%a.err
set -euo pipefail

hostname
date

number=$SLURM_ARRAY_TASK_ID
paramfile="scripts/jobs_mlp2.txt"

# ---------------------------------------------------------------------
# 1.  Activate virtual environment
# ---------------------------------------------------------------------
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
seed=$(sed -n ${number}p "$paramfile" | awk '{print $1}')

date
echo "Training: MLP2 | Seed: $seed"

# ---------------------------------------------------------------------
# 5.  Run training
# ---------------------------------------------------------------------
echo "Starting training..."
python -u train_mlp.py \
    --model mlp2 \
    --seed $seed \
    --epochs 600 \
    --batch_size 512 \
    --lr 0.001 \
    --target_train_acc 99.99 \
    --weight_decay 0.0 \
    --dropout 0.0 \
    --warmup_epochs 5 \
    --grad_clip 1.0 \
    --checkpoint_dir checkpoints \
    --log_dir logs

date
echo "Training completed successfully: MLP2 | Seed $seed"

# ---------------------------------------------------------------------
# 6.  Run evaluation with layer-wise neuron masking
# ---------------------------------------------------------------------
echo "Starting evaluation with layer-wise neuron masking..."

checkpoint_dir="checkpoints/mlp2_seed${seed}"

# Try checkpoint_latest.pth first, then fall back to checkpoint_best.pth
if [ -f "$checkpoint_dir/checkpoint_latest.pth" ]; then
    checkpoint_file="$checkpoint_dir/checkpoint_latest.pth"
elif [ -f "$checkpoint_dir/checkpoint_best.pth" ]; then
    checkpoint_file="$checkpoint_dir/checkpoint_best.pth"
else
    echo "Error: No checkpoint file found in $checkpoint_dir"
    echo "Contents of checkpoint directory:"
    ls -la "$checkpoint_dir" || true
    exit 1
fi

echo "Found checkpoint: $checkpoint_file"

# Run layer-wise evaluation
python -u evaluate_mlp.py \
    --checkpoint "$checkpoint_file" \
    --model mlp2 \
    --n_subsets 100 \
    --batch_size 512 \
    --dropout 0.0 \
    --output_dir data \
    --device cuda \
    --seed $seed

date
echo "Evaluation completed successfully: MLP2 | Seed $seed"
