import cv2
import torch
import numpy as np
from torchvision import transforms

# Using float32 tensors since I don't know the datatype we'll end up using

def load_image(image_path):
    """
    Load an image from disk and convert it from BGR to RGB.

    Parameters
    ----------
    image_path : str
        Path to the image file to load.

    Returns
    -------
    ndarray
        The loaded image as a NumPy array in RGB format (H, W, C).
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(mask_path):
    """
    Load a grayscale mask image, normalize its values to [0,1], and return as float32.

    Assumes the mask has only two unique values {0, 255} or {0, 1}.

    Parameters
    ----------
    mask_path : str
        Path to the mask image file (grayscale).

    Returns
    -------
    ndarray
        The mask as a float32 NumPy array with values in [0, 1].
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)    # change to float (need for few cases)

    # if range of image is [0, 255] normalize it
    if np.max(mask) > 1:
        mask = mask / 255   # normalize

    return mask


def apply_img_preprocessing(image, transform=None):
    """
    Apply image preprocessing transforms, defaulting to ToTensor if none provided.

    If `transform` is None or an empty Compose, this function will apply:
      transforms.ToTensor() (scales to [0,1] and converts to [C, H, W] torch.Tensor).

    Parameters
    ----------
    image : ndarray or PIL.Image
        The input image to preprocess.
    transform : torchvision.transforms.Compose or None, optional
        A composition of image transforms. If None or empty, defaults to ToTensor.

    Returns
    -------
    torch.Tensor
        The preprocessed image tensor of shape [C, H, W].
    """
    if transform is None or (isinstance(transform, transforms.Compose) and len(transform.transforms) == 0):
        transform = transforms.Compose([
            transforms.ToTensor(),  # Divides by 255, transposes dimensions to (C, H, W) and Converts to tensor
        ])

    # else apply provided transformations
    return transform(image)


def get_img_transform(resize_height, resize_width):
    """
    Create a transform pipeline that resizes an image and converts it to a tensor.

    Parameters
    ----------
    resize_height : int
        Target height in pixels.
    resize_width : int
        Target width in pixels.

    Returns
    -------
    torchvision.transforms.Compose
        A composed transform: ToPILImage -> Resize -> ToTensor.
    """
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor()
    ])
    return preprocess_transform


def resize_mask(mask, target_size):
    """
    Resize a mask array to the given target size using nearest-neighbor interpolation.

    Parameters
    ----------
    mask : ndarray
        Input mask array (H, W).
    target_size : tuple of int (width, height)
        Desired output size.

    Returns
    -------
    ndarray
        Resized mask array of shape (target_height, target_width).
    """
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return mask


def rgb_to_label_mask(rgb_mask, label_map):
    """
    Convert an RGB mask image to a 2D label mask using a color-to-label mapping.

    Parameters
    ----------
    rgb_mask : ndarray
        RGB mask image as a NumPy array of shape (H, W, 3).
    label_map : dict
        Mapping from label names to dicts with keys:
          - "color_code": list or tuple of 3 ints (RGB).
          - "train_id": int, the label value to assign.

    Returns
    -------
    ndarray
        2D mask array of dtype float32 where each pixel value is the train_id
        corresponding to its RGB color in `rgb_mask`.
    """
    mask = np.zeros(rgb_mask.shape[:2], dtype=np.float32)

    for label, info in label_map.items():
        color_code = info["color_code"]  # RGB color code of label
        train_label = info["train_id"]  # training label for this class

        # Assign class label based on color match
        mask[np.all(rgb_mask == color_code, axis=-1)] = train_label

    return mask


