import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.nn.functional import one_hot, pad

from ..utils.preprocessing import load_image


class ClassificationDataset(Dataset):
    """
    Dataset for loading images and their degradation labels for classification.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the image files.
    degradation_csv : str
        Path to a CSV file with two columns: filename and integer label.
    img_transform : callable, optional
        A transform to apply to each image (e.g., torchvision transforms).  
        If None, images will be converted to tensors via `ToTensor()`.
    num_classes : int, default=3
        Number of classes for one-hot encoding the labels.
    subset_size : int, optional
        If provided, randomly sample this many examples (or fewer if the CSV
        has fewer rows) from the full dataset.

    Attributes
    ----------
    image_dir : str
        As above.
    data : pandas.DataFrame
        The loaded (and possibly subsampled) CSV.
    transform : callable or None
        The image transform to apply.
    num_classes : int
        Number of classes for one-hot encoding.
    """

    def __init__(self, image_dir, degradation_csv, img_transform=None, num_classes=3, subset_size=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(degradation_csv)
        self.num_classes = num_classes
        self.transform = img_transform  # I don't think we'll need transforms since it's a small input, but I'll leave this here nevertheless

        if subset_size is not None:
            subset_size = min(len(self.data), subset_size)
            self.data = self.data.sample(n=subset_size, ignore_index=True)

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of image–label pairs available.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load and return a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            The image tensor, shape [C, H, W].
        degradation_target : torch.Tensor
            One-hot encoded label tensor, shape [num_classes].
        """
        # Build full image path and load
        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = load_image(img_path)

        # Load regression target
        degradation_target = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long)
        degradation_target = one_hot(degradation_target, num_classes=self.num_classes).float()

        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)

        return image, degradation_target


def collate_pad_fn(batch):
    """
    Custom collate function that pads a batch of images to the same size.

    Parameters
    ----------
    batch : list of tuples
        Each element is (image_tensor, label_tensor), where
        - image_tensor is a float Tensor of shape [C, H, W]
        - label_tensor is a one-hot encoded Tensor of shape [num_classes]

    Returns
    -------
    images : torch.Tensor
        A tensor of shape [N, C, H_max, W_max] where each image has been
        zero-padded (on right and bottom) to the maximum height and width
        within the batch.
    labels : torch.Tensor
        A tensor of shape [N, num_classes] stacking all label tensors.
    """
    # Find maximum spatial dimensions in this batch
    max_height = max(img.size(1) for img, _ in batch)
    max_width = max(img.size(2) for img, _ in batch)

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
