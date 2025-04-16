import cv2
import torch
import numpy as np
from torchvision import transforms

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def apply_img_preprocessing(image, transform=None):

    # apply basic transformations
    if transform is None or (isinstance(transform, transforms.Compose) and len(transform.transforms)==0):
        transform = transforms.Compose([
            transforms.ToTensor(),  # Divides by 255, transposes dimensions to (C, H, W) and Converts to tensor
        ])

    # else apply provided transformations
    return transform(image)


def get_img_transform(resize_height, resize_width):

    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor()
    ])
    return preprocess_transform