import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from training.dataset import get_data_loaders
import time
import copy

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda'):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    data_dir = './training/data'
    num_classes = 38 # PlantVillage standard
    batch_size = 16
    num_epochs = 30
    learning_rate = 0.001

    # Load Data
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size)
    if not train_loader: return
    
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Initialize Model (EfficientNet-B4)
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Differential Learning Rates: Lower for backbone, higher for head
    optimizer = optim.Adam([
        {'params': model._fc.parameters(), 'lr': learning_rate},
        {'params': model.extract_features.parameters(), 'lr': learning_rate / 10}
    ])

    # Train
    model, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, device=device)

    # Save
    torch.save(model.state_dict(), 'ml_models/efficientnet_model_trained.pth')
    print("Model saved to ml_models/efficientnet_model_trained.pth")

if __name__ == "__main__":
    main()
