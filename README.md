# MNIST Transfer Learning

Transfer learning on MNIST using ViT-Large (Vision Transformer) pretrained on ImageNet.

## Usage

### Local Training
```bash
python transfer_learning_mnist.py --epochs 10 --batch_size 32
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

- Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```