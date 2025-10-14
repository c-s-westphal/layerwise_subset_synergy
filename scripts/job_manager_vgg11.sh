#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N vgg11
#$ -t 1-3
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_vgg11.txt"

# ---------------------------------------------------------------------
# 1.  Load toolchains and activate virtual-env
# ---------------------------------------------------------------------
source /share/apps/source_files/python/python-3.9.5.source
source /share/apps/source_files/cuda/cuda-11.8.source
source /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# ---------------------------------------------------------------------
# 2.  Keep Matplotlib out of your home quota
# ---------------------------------------------------------------------
export MPLCONFIGDIR="$TMPDIR/mplcache"
mkdir -p "$MPLCONFIGDIR"

# ---------------------------------------------------------------------
# 3.  Create output directories
# ---------------------------------------------------------------------
mkdir -p checkpoints
mkdir -p results
mkdir -p data

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
python3.9 -u train.py \
    --model vgg11 \
    --seed $seed \
    --epochs 500 \
    --batch_size 128 \
    --lr 0.001 \
    --target_train_acc 95.0 \
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
python3.9 -u evaluate.py \
    --checkpoint "$checkpoint_file" \
    --model vgg11 \
    --n_subsets 1000 \
    --output_dir data \
    --device cuda \
    --seed $seed

date
echo "Evaluation completed successfully: VGG11 | Seed $seed"
