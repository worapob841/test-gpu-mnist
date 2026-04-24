#!/bin/bash
#SBATCH --job-name=mnist_ddp
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --distribution=block

# Load required modules
module load python/3.10
module load cuda/12.1
module load cudnn/8.9

# Set environment variables
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO

# Create logs directory
mkdir -p logs

# Change to script directory
cd $SLURM_SUBMIT_DIR

# Run DDP training with 8 GPUs
srun python train_ddp.py \
    --data_dir $SLURM_TMPDIR/data \
    --batch_size 32 \
    --epochs 10 \
    --lr 0.001 \
    --train_samples 10000 \
    --save_dir $SLURM_SUBMIT_DIR/checkpoints