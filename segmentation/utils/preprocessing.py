import cv2
import torch
import numpy as np
from torchvision import transforms

# Using float32 tensors since I don't know the datatype we'll end up using

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image, transforms=True):
    if transforms:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1280, 720)), # Size of images from BDD100K dataset
            transforms.ToTensor(),          # Divides by 255, transposes dimensions to (C, H, W) and Converts to tensor
        ])
        return transform(image)
    else:
        # Basic processing if not using transforms (No resizing)
        image = image.astype(np.float32) / 255.0 
        return torch.from_numpy(image.transpose(2, 0, 1)) # Transpose to (C, H, W) and convert to tensor

def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask

def preprocess_mask(mask, target_size=(1280, 720)):
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    return torch.from_numpy(mask[None, ...])  # Add a channel dimension (None adds a new axis in the beginning and the ... keeps the other dimensions as is)
