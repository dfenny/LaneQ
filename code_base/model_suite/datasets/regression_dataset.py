import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from ..utils.preprocessing import load_image


class RegressionDataset(Dataset):
    """
    Dataset for loading images and continuous degradation labels for regression.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the image files.
    degradation_csv : str
        Path to a CSV file with two columns: filename and float target value.
    img_transform : callable, optional
        A transform to apply to each image (e.g., torchvision transforms).
        If None, images will be converted to tensors via `ToTensor()`.
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
    """

    def __init__(self, image_dir, degradation_csv, img_transform=None, subset_size=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(degradation_csv)
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
            Number of image–value pairs available.
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
        degradation_value : torch.Tensor
            The continuous target value as a 0-dim float tensor.
        """
        # Build full image path and load
        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = load_image(img_path)

        # Load regression target
        degradation_value = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)

        return image, degradation_value

