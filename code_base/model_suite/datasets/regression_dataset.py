import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from ..utils.preprocessing import load_image


class RegressionDataset(Dataset):

    def __init__(self, image_dir, degradation_csv, img_transform=None, subset_size=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(degradation_csv)
        self.transform = img_transform  # I don't think we'll need transforms since it's a small input, but I'll leave this here nevertheless

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
        else:
            image = ToTensor()(image)

        return image, degradation_value

