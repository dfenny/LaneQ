import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.nn.functional import one_hot, pad

from ..utils.preprocessing import load_image


class ClassificationDataset(Dataset):

    def __init__(self, image_dir, degradation_csv, img_transform=None, num_classes=3, subset_size=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(degradation_csv)
        self.num_classes = num_classes
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
        degradation_target = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long)
        degradation_target = one_hot(degradation_target, num_classes=self.num_classes)

        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)

        return image, degradation_target


def collate_pad_fn(batch):
    # Determine the maximum width and height of images in the batch
    max_height = max([x[0].size(1) for x in batch])
    max_width = max([x[0].size(2) for x in batch])

    padded_images = []
    labels = []
    for image, label in batch:
        # Pad dimensions (left, right, top, bottom)
        pad_w = max_width - image.shape[2]
        pad_h = max_height - image.shape[1]
        padded_image = pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded_image)
        labels.append(label)

    # Stack all images to form a batch
    return torch.stack(padded_images), torch.stack(labels)
