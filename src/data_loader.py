import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from . import config

def get_transforms(image_size=config.IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataloaders(
    data_dir=config.PROCESSED_DIR,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS
):
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    test_dir = Path(data_dir) / "test"
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, train_dataset.classes
