import os
import random
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torchvision.transforms import ToTensor
from ..utils.preprocessing import load_image, load_mask, apply_img_preprocessing, resize_mask, rgb_to_label_mask


class SegmentationDataset(Dataset):
    """
    Dataset for loading image–mask pairs for semantic segmentation tasks.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing input image files.
    mask_dir : str
        Path to the directory containing corresponding mask files.
    img_transform : callable, optional
        A transform to apply to each input image. If None, no additional transform is applied.
    mask_reshape : tuple of int (width, height), optional
        If provided, masks will be resized to this (width, height). Must be a tuple.
    rgb_mask : bool, default=False
        If True, loads masks as RGB images and converts them via `rgb_label_map`.
    rgb_label_map : dict, optional
        Mapping from RGB tuple to integer class label. Required if `rgb_mask` is True.
    one_hot_target : bool, default=False
        If True, one-hot encodes the mask into shape `[num_classes, H, W]`.
    num_classes : int, default=2
        Number of classes for one-hot encoding. Ignored if `one_hot_target` is False.
    subset_size : int, optional
        If provided, randomly sample this many filenames (or fewer if dataset is smaller).
    """

    def __init__(self, image_dir, mask_dir, img_transform=None, mask_reshape=None,
                 rgb_mask=False, rgb_label_map=None, one_hot_target=False,
                 num_classes=2, subset_size=None):
        """
        Initialize the SegmentationDataset.

        Reads image filenames, applies optional subsampling, and validates mask settings.

        Parameters
        ----------
        image_dir : str
            Directory with input images.
        mask_dir : str
            Directory with mask images.
        img_transform : callable or None
            Transform to apply to images.
        mask_reshape : tuple of int or None
            Desired mask size (width, height).
        rgb_mask : bool
            Whether masks are RGB images.
        rgb_label_map : dict or None
            RGB-to-label mapping if `rgb_mask` is True.
        one_hot_target : bool
            Whether to one-hot encode masks.
        num_classes : int
            Number of classes for one-hot encoding.
        subset_size : int or None
            Number of samples to randomly select.
        
        Raises
        ------
        ValueError
            If `mask_reshape` is not a tuple when provided, or if `rgb_mask` is True but
            `rgb_label_map` is not a valid dict.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = img_transform

        # get all filenames in given dataset location
        self.file_names = sorted(os.listdir(image_dir))
        if subset_size is not None:
            # randomly select subset
            k = min(len(self.file_names), subset_size)
            self.file_names = random.sample(self.file_names, k)

        if mask_reshape is not None and (not isinstance(mask_reshape, tuple)):
            raise ValueError("mask_reshape should be tuple of (width, height)")

        self.mask_reshape = mask_reshape

        if rgb_mask and (not isinstance(rgb_label_map, dict) or rgb_label_map is None):
            raise ValueError("If rgb_mask is True, rgb_label_map must be a valid dictionary.")

        self.rgb_mask = rgb_mask
        self.rgb_label_map = rgb_label_map
        self.one_hot_target = one_hot_target
        self.num_classes = num_classes

    def __len__(self):
        """
        Return the total number of samples.

        Returns
        -------
        int
            Number of image–mask pairs in the dataset.
        """
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Load and return the image and its corresponding mask at the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        img : torch.Tensor
            Preprocessed image tensor of shape [C, H, W].
        mask : torch.Tensor
            Mask tensor of shape [num_classes, H, W] if `one_hot_target` is True,
            otherwise [1, H, W].
        """
        filename = self.file_names[idx]
        img = load_image(os.path.join(self.image_dir, filename))

        # IMP: ONLY for dataset used during code test - change extension for mask image
        filename = filename.replace(".jpg", ".png")

        if self.rgb_mask:
            mask = load_image(os.path.join(self.mask_dir, filename))
            if self.mask_reshape is not None:
                mask = resize_mask(mask, self.mask_reshape)           # reshape if required
            mask = rgb_to_label_mask(rgb_mask=mask, label_map=self.rgb_label_map)    # convert RGB to label mask
        else:
            mask = load_mask(os.path.join(self.mask_dir, filename))
            if self.mask_reshape is not None:
                mask = resize_mask(mask, self.mask_reshape)           # reshape if required

        # in case target needs to be one hot encoded
        if self.one_hot_target:
            # IMP: find better option to do this
            # for now use torch functionality for which numpy needs to converted to tensor and back to numpy
            # tensor is converted back to numpy to avoid issues if self.transform as ToTensor() functions
            mask = torch.from_numpy(mask)
            mask = mask.long()     # Convert to LongTensor as required by one_hot encoding function
            mask = one_hot(mask, num_classes=self.num_classes).numpy()

        # apply the transformations to image
        img = apply_img_preprocessing(img, transform=self.transform)
        mask = ToTensor()(mask)

        # return a tuple of the image and its mask
        return img, mask
