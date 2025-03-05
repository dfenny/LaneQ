import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torchvision.transforms import ToTensor
from .preprocessing import load_image, load_mask, apply_img_preprocessing, resize_mask, rgb_to_label_mask


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, img_transform=None, mask_reshape=None, rgb_mask=False, rgb_label_map=None,
                 one_hot_target=False, num_classes=2):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = img_transform

        # get all filenames in given dataset location
        self.file_names = sorted(os.listdir(image_dir))

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
        return len(self.file_names)

    def __getitem__(self, idx):

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
