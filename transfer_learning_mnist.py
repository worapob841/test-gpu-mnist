import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import argparse
import os
import time

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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

def load_data(data_dir, batch_size, train_samples=10000, device=None):
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
    
    use_pin_memory = device is not None and device.type in ('cuda', 'mps')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader

def create_model(num_classes=10):
    model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
    
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
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

def evaluate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Eval]", leave=False)
    
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
    parser = argparse.ArgumentParser(description='Transfer Learning on MNIST')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader = load_data(args.data_dir, args.batch_size, args.train_samples, device)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    print("Creating model with transfer learning from ViT-L/16...")
    model = create_model(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_acc = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...\n")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print("-" * 50)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model to {save_path}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Total time: {total_time/60:.2f} minutes")

if __name__ == '__main__':
    main()