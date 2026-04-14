import torch
import torch.nn as nn
from torchvision import models

class CNNFeatureExtractor:
    def __init__(self):
        # Using ResNet backbone for feature extraction
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

    def extract(self, image_tensor):
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.flatten()
