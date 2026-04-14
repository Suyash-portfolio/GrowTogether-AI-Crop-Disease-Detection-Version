import cv2
import numpy as np
import torch
from torchvision import transforms

def get_efficientnet_transform():
    """
    MANDATORY: Must match the training normalization for EfficientNet-B4.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((380, 380)), # EfficientNet-B4 standard resolution
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet Mean
            std=[0.229, 0.224, 0.225]   # ImageNet Std
        )
    ])

def preprocess_drone_image(image_bytes):
    """
    Fixes BGR -> RGB and handles initial decoding.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # CRITICAL FIX: Convert to RGB for PyTorch/EfficientNet
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb, img_bgr
