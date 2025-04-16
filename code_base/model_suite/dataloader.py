from .datasets import SegmentationDataset, RegressionDataset, ClassificationDataset, collate_pad_fn
from .utils.preprocessing import get_img_transform
from torch.utils.data import DataLoader
import json


def generate_dataloader(dataset_type, data_loc, dataloader_config, preprocess_config=None, img_transform=None,
                        num_class=3):
    """
    Create a PyTorch DataLoader for a specified dataset type.

    Parameters
    ----------
    dataset_type : str
        One of {'segmentation', 'regression', 'classification'}. Determines which Dataset subclass to use.
    data_loc : dict
        Paths and settings for the data:
          - segmentation: keys 'img_dir', 'mask_dir', optional 'random_subset'
          - regression/classification: keys 'img_dir', 'degradation_csv', optional 'random_subset'
    dataloader_config : dict
        Configuration for DataLoader:
          - 'batch_size' (int)
          - 'num_workers' (int)
          - shuffle is set based on 'shuffle' key if present
    preprocess_config : dict, optional
        Preprocessing options for segmentation:
          - 'resize_width' (int), 'resize_height' (int)
          - 'RGB_mask' (bool), 'RGB_labelmap' (str path to JSON)
          - 'one_hot_mask' (bool), 'num_classes' (int)
    img_transform : callable, optional
        A torchvision-compatible transform to apply to images after loading.
    num_class : int, default=3
        Number of classes for classification one-hot encoding.

    Returns
    -------
    tuple or None
        If a valid dataset_type is provided, returns a tuple:
          - DataLoader: configured DataLoader instance
          - int: total number of samples in the dataset
        Otherwise, returns None.
    """
    dataset = None
    if dataset_type == 'segmentation':

        preprocess_config = {} if preprocess_config is None else preprocess_config
        resize_width = preprocess_config.get('resize_width', None)
        resize_height = preprocess_config.get('resize_height', None)
        mask_reshape = None
        if resize_width is not None and resize_height is not None:
            img_transform = get_img_transform(resize_height=resize_height, resize_width=resize_width)
            mask_reshape = (resize_width, resize_height)

        label_map = None
        if preprocess_config.get("RGB_mask", False):
            try:
                with open(preprocess_config["RGB_labelmap"], 'r') as json_file:
                    label_map = json.load(json_file)
            except Exception:
                print("Error loading labelmap")
                label_map = None

        # initialize dataset object
        dataset = SegmentationDataset(image_dir=data_loc["img_dir"],
                                      mask_dir=data_loc["mask_dir"],
                                      rgb_mask=preprocess_config.get("RGB_mask", False),
                                      rgb_label_map=label_map,
                                      img_transform=img_transform,
                                      mask_reshape=mask_reshape,
                                      one_hot_target=preprocess_config.get("one_hot_mask", False),
                                      num_classes=preprocess_config.get("num_classes", 2),
                                      subset_size=data_loc.get('random_subset', None))

    elif dataset_type == 'regression':

        dataset = RegressionDataset(image_dir=data_loc["img_dir"],
                                    degradation_csv=data_loc["degradation_csv"],
                                    img_transform=img_transform,
                                    subset_size=data_loc.get('random_subset', None))

    elif dataset_type == 'classification':

        dataset = ClassificationDataset(image_dir=data_loc["img_dir"],
                                        degradation_csv=data_loc["degradation_csv"],
                                        img_transform=img_transform,
                                        num_classes=num_class,
                                        subset_size=data_loc.get('random_subset', None))

    if dataset is None:
        return

    # generate dataloader
    batch_size = dataloader_config.get('batch_size', 1)
    collate_fn = collate_pad_fn if (batch_size > 1 and dataset_type == "classification") else None
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=dataloader_config.get('num_workers', False),
                            num_workers=dataloader_config.get('num_workers', 0),
                            collate_fn=collate_fn)
    return dataloader, len(dataset)
