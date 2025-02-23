import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from utils.preprocessing import *

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_ids = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        image_path = os.path.join(self.image_dir, image_id)
        mask_path = os.path.join(self.mask_dir, image_id)
        
        image = load_image(image_path)
        mask = load_mask(mask_path)

        image = preprocess_image(image, transforms=self.transforms)
        mask = preprocess_mask(mask)
        
        data_item = {'image': image, 'mask': mask}
        
        return data_item
