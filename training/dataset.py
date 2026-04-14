import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, img_size=380):
    """
    Creates training and validation data loaders with heavy augmentation
    for crop disease images.
    """
    
    # Advanced Augmentation for Agricultural Data
    # Handles variations in lighting, drone angle, and weather
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # Drone images can be any orientation
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(img_size + 20),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    # Check if we have a downloaded path from kagglehub
    path_file = "./training/dataset_path.txt"
    if os.path.exists(path_file):
        with open(path_file, "r") as f:
            data_dir = f.read().strip()
        print(f"Using Kaggle dataset path: {data_dir}")
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Warning: Data directories not found at {data_dir}. Please organize your data into 'train' and 'val' folders.")
        return None, None

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
