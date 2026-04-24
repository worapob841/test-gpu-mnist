# MNIST Transfer Learning

Transfer learning on MNIST using ResNet18 pretrained on ImageNet.

## Usage

### Local Training
```bash
python transfer_learning_mnist.py --epochs 10 --batch_size 64
```

### SLURM Cluster
```bash
sbatch train_mnist.sbatch
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Data directory |
| `--batch_size` | `64` | Batch size |
| `--epochs` | `10` | Number of epochs |
| `--lr` | `0.001` | Learning rate |
| `--train_samples` | `10000` | Training samples |
| `--save_dir` | `./checkpoints` | Model save directory |

## Requirements

- PyTorch
- torchvision
- tqdm