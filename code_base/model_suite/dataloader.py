from .datasets import SegmentationDataset, RegressionDataset, ClassificationDataset, collate_pad_fn
from .utils.preprocessing import get_img_transform
from torch.utils.data import DataLoader
import json


def generate_dataloader(dataset_type, data_loc, dataloader_config, preprocess_config=None, img_transform=None,
                        num_class=3):
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
        if preprocess_config["RGB_mask"]:
            try:
                with open(preprocess_config["RGB_labelmap"], 'r') as json_file:
                    label_map = json.load(json_file)
            except Exception as e:
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
