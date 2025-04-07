import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot, pad
from torchvision import transforms
from .preprocessing import load_image
import pandas as pd
from torchvision.transforms import ToTensor

class RegressionDataset(Dataset):
    
    def __init__(self, image_dir, degradation_values_csv, transform=None, subset_size=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(degradation_values_csv)
        self.transform = transform # I don't think we'll need transforms since it's a small input, but I'll leave this here nevertheless

        if subset_size is not None:
            subset_size = min(len(self.data), subset_size)
            self.data = self.data.sample(n=subset_size, ignore_index=True)


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


class ClassificationDataset(Dataset):

    def __init__(self, image_dir, degradation_values_csv, transform=None, num_classes=3, subset_size=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(degradation_values_csv)
        self.num_classes = num_classes
        self.transform = transform  # I don't think we'll need transforms since it's a small input, but I'll leave this here nevertheless

        if subset_size is not None:
            subset_size = min(len(self.data), subset_size)
            self.data = self.data.sample(n=subset_size, ignore_index=True)

    def __len__(self):
        return len(self.data)  # Total number of samples

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = load_image(img_path)

        # Load regression target
        degradation_target = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
        degradation_target = one_hot(degradation_target, num_classes=self.num_classes)

        if self.transform:
            image = self.transform(image)

        image = ToTensor()(image)

        return image, degradation_target



def collate_fn(batch):

    images, targets = zip(*batch)
    print(images.shape)
    # Get max dimensions in the batch
    max_h = max([img.shape[2] for img in images])
    max_w = max([img.shape[1] for img in images])

    padded_images = []
    for img in images:
        # Pad dimensions (left, right, top, bottom)
        pad_w = max_w - img.shape[2]
        pad_h = max_h - img.shape[1]
        padded_img = pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded_img)

    images = torch.stack(padded_images)
    targets = torch.stack(targets)
    return images, targets