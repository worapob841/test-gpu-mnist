#!/bin/bash
#SBATCH --job-name=mnist_transfer
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load required modules
module load python/3.10
module load cuda/12.1
module load cudnn/8.9

# Activate virtual environment if exists
# source /path/to/venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
mkdir -p logs

# Run the training script
srun python transfer_learning_mnist.py \
    --data_dir $SLURM_TMPDIR/data \
    --batch_size 64 \
    --epochs 10 \
    --lr 0.001 \
    --train_samples 10000 \
    --save_dir $SLURM_SUBMIT_DIR/checkpoints