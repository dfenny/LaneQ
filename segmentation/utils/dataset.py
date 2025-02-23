import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

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
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalizing the image and mask (I'm not sure what datatype we'll end up using, so I've left it as float32)
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        data_item = {'image': image, 'mask': mask}
        
        # In case we want to augment data etc.
        if self.transforms:
            data_item = self.transforms(data_item)
        
        # Converting to torch tensors
        data_item['image'] = torch.from_numpy(data_item['image'].transpose(2, 0, 1)) # This transposes the image to (C, H, W) format, and then converts it to a tensor
        data_item['mask'] = torch.from_numpy(data_item['mask'][None, ...]) # This transposes the mask to (1, H, W) a 1-channel image, and then into a tensor (Needed help for this)
        
        return data_item
