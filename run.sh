#!/bin/bash
srun python transfer_learning_mnist.py --data_dir ./data --batch_size 64 --epochs 10 --lr 0.001 --train_samples 10000 --save_dir ./checkpoints