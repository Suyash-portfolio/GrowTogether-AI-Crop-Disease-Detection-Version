import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from training.dataset import get_data_loaders
from training.train_classifier import train_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    data_dir = './training/data'
    num_classes = 38
    batch_size = 32
    num_epochs = 20

    # Load Data
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size, img_size=224) # Standard CNN size
    if not train_loader: return
    
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Initialize Custom CNN (ResNet50 as backbone)
    model = models.resnet50(pretrained=True)
    
    # Modify the final layer for our specific crop classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train
    print("Training CNN Feature Extractor...")
    model, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, device=device)

    # Save the backbone only (feature extractor)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    torch.save(feature_extractor.state_dict(), 'ml_models/cnn_backbone_trained.pth')
    print("Feature extractor saved to ml_models/cnn_backbone_trained.pth")

if __name__ == "__main__":
    main()
