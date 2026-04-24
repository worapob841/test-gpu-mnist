import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import argparse
import os
import time

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return train_transform, val_transform

def load_data(data_dir, batch_size, train_samples=10000, world_size=1, rank=0):
    train_transform, val_transform = get_transforms()
    
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )
    
    if train_samples < len(full_train_dataset):
        indices = torch.randperm(len(full_train_dataset))[:train_samples]
        train_dataset = Subset(full_train_dataset, indices)
    else:
        train_dataset = full_train_dataset
    
    val_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=val_transform,
        download=True
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler

def create_model(num_classes=10):
    model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
    
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train] GPU-{rank}", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, val_loader, criterion, device, epoch, rank):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Eval] GPU-{rank}", leave=False)
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(val_loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='DDP Transfer Learning on MNIST')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model')
    args = parser.parse_args()
    
    init_process_group(backend='nccl')
    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"World size: {world_size}, Rank: {rank}")
        print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_loader, val_loader, train_sampler = load_data(
        args.data_dir, args.batch_size, args.train_samples, world_size, rank
    )
    
    if rank == 0:
        print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        print("Creating model with transfer learning from ViT-L/16...")
    
    model = create_model(num_classes=10).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_acc = 0.0
    
    if rank == 0:
        print(f"\nStarting training for {args.epochs} epochs...\n")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch, rank)
        
        scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print("-" * 50)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save(model.module.state_dict(), save_path)
                print(f"  Saved best model to {save_path}")
    
    total_time = time.time() - start_time
    
    if rank == 0:
        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Total time: {total_time/60:.2f} minutes")
    
    destroy_process_group()

if __name__ == '__main__':
    main()