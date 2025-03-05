import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .preprocessing import load_image

class RegressionDataset(Dataset):
    
    def __init__(self, image_dir, degradation_values, transform=None):
        self.image_dir = image_dir
        self.degradation_values = degradation_values
        self.transform = transform # I don't think we'll need transforms since it's a small input, but I'll leave this here nevertheless

        # get all filenames in given dataset location
        self.file_names = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.file_names)  # Total number of samples

    def __getitem__(self, idx):
        
        filename = self.file_names[idx]
        
        # Image
        image = load_image(os.path.join(self.image_dir, filename))
        # Corresponding degradation value
        degradation = self.degradation_values[idx]

        if self.transform:
            image = self.transform(image)

        return image, degradation
