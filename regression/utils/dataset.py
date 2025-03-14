import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .preprocessing import load_image
import pandas as pd
from torchvision.transforms import ToTensor

class RegressionDataset(Dataset):
    
    def __init__(self, image_dir, degradation_values_csv, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(degradation_values_csv, header=None, names=["filename", "value"])
        self.transform = transform # I don't think we'll need transforms since it's a small input, but I'll leave this here nevertheless

        # get all filenames in given dataset location
        self.file_names = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.data)  # Total number of samples

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = load_image(img_path)

        # Load regression target
        degradation_value = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        image = ToTensor()(image)

        return image, degradation_value
