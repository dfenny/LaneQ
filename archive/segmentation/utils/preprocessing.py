import cv2
import torch
import numpy as np
from torchvision import transforms

# Using float32 tensors since I don't know the datatype we'll end up using

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(mask_path):
    """Assumption that grayscale image as only two unique values {0, 255} or {0, 1}"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)    # change to float (need for few cases)

    # if range of image is [0, 255] normalize it
    if np.max(mask) > 1:
        mask = mask / 255   # normalize

    return mask


def apply_img_preprocessing(image, transform=None):

    # apply basic transformations
    if transform is None or (isinstance(transform, transforms.Compose) and len(transform.transforms)==0):
        transform = transforms.Compose([
            transforms.ToTensor(),  # Divides by 255, transposes dimensions to (C, H, W) and Converts to tensor
        ])

    # else apply provided transformations
    return transform(image)


def resize_mask(mask, target_size):
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return mask


def rgb_to_label_mask(rgb_mask, label_map):

    # Create empty mask
    # IMP: dtype as float to avoid normalization due to transforms.ToTensor()
    mask = np.zeros(rgb_mask.shape[:2], dtype=np.float32)

    for label, info in label_map.items():
        color_code = info["color_code"]  # RGB color code of label
        train_label = info["train_id"]  # training label for this class

        # Assign class label based on color match
        mask[np.all(rgb_mask == color_code, axis=-1)] = train_label

    return mask


def get_img_transform(resize_height, resize_width):

    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor()
    ])
    return preprocess_transform