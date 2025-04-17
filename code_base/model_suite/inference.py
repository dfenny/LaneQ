import os
import cv2
import json
import time
import argparse
import numpy as np
import torch
from torchvision import transforms

from .utils.common import add_bbox, box_coco_to_corner
from .utils.preprocessing import load_image
from .utils.inference_utils import load_saved_model, pred_segmentation_mask, pred_degradation_category, generate_individual_segments_n_annotations


# class colors
_color_map = {0: (0, 255, 0), 1: (255, 255, 0), 2: (255, 0, 0)}   # RGB
_target_name_to_code = {'ignore': -1, 'Good': 0, 'Slight': 1, 'Severe': 2}
_target_code_to_name =  {v: k for k, v in _target_name_to_code.items()}


class DegradationDetector:
    """
    Pipeline for detecting and classifying degradation segments in images.

    Combines a segmentation model to find regions of interest, then classifies
    each segment with a classification model, and finally annotates and saves
    the results.
    """

    def __init__(self, segmentation_weights_path, classification_weights_path,
                 output_dir, min_segment_dim=10):
        """
        Initialize the DegradationDetector with pre-trained models and settings.

        Parameters
        ----------
        segmentation_weights_path : str
            Filesystem path to the saved weights for the segmentation model.
        classification_weights_path : str
            Filesystem path to the saved weights for the classification model.
        output_dir : str
            Directory where all prediction outputs (masks, JSON, annotated images)
            will be saved.
        min_segment_dim : int, default=10
            Minimum width or height (in pixels) of a detected segment to classify;
            smaller segments will be skipped.
        """
        # model configs
        seg_model_name = "unet"
        segmentation_model_config = {'in_channels': 3, 'out_channels': 1}

        classification_model_name = "cnn_sppf"
        classification_model_config = {'in_channels': 3, 'out_dim': 3}

        # initialize and load saved model
        self.segmentation_model = load_saved_model(model_name=seg_model_name,
                                                   saved_weight_path=segmentation_weights_path,
                                                   **segmentation_model_config)
        self.classification_model = load_saved_model(model_name=classification_model_name,
                                                     saved_weight_path=classification_weights_path,
                                                     **classification_model_config)

        self.output_dir = output_dir
        self.min_segment_dim = max(10, min_segment_dim)

        self.seg_input_height = 720
        self.seg_input_width = 1280
        self.seg_img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.seg_input_height, self.seg_input_width)),
            transforms.ToTensor()
        ])

    def predict(self, img_path, device="cpu"):
        """
        Run the full degradation detection pipeline on a single image.

        Steps:
          1. Load image and run segmentation to get a binary mask.
          2. Extract individual segments via connected components.
          3. Classify each sufficiently large segment.
          4. Annotate the original image with colored bounding boxes.
          5. Save the predicted mask, JSON annotations, and annotated image.

        Parameters
        ----------
        img_path : str
            Path to the input image file.
        device : str or torch.device, default="cpu"
            Device on which to perform model inference.

        Returns
        -------
        dict
            Contains "save_path" key pointing to the directory where outputs are saved.
        """
        # required during saving output
        img_name = os.path.basename(img_path)
        img_name_without_ext = os.path.splitext(img_name)[0]

        test_img = load_image(img_path)

        # detect lane segments using segmentation model
        pred_mask = pred_segmentation_mask(model=self.segmentation_model, test_img=test_img,
                                           img_transform=self.seg_img_transform,
                                           add_batch_dim=True, device=device)
        pred_mask = np.squeeze(pred_mask)   # remove batch dim
        pred_mask = pred_mask.astype(np.uint8) * 255

        # identify each of the detected lane segments using connected components to process them individually
        # this function also gives pre-processing segments as required by the classification model
        annotations_dict = generate_individual_segments_n_annotations(img=test_img, mask=pred_mask,
                                                                      annot_prefix=img_name_without_ext)
        annotations_dict["image"] = img_name  # add the full image name in annotations

        # run classification inference on each detected segments
        for segment in annotations_dict["annotations"]:
            topx, topy, width, height = segment["bounding_box"]
            if width < self.min_segment_dim or height < self.min_segment_dim:  # do not process smaller segments
                continue

            segment_img = segment["segment_crop"]
            pred_deg_class = pred_degradation_category(model=self.classification_model, test_img=segment_img,
                                                       add_batch_dim=True, device=device)
            segment["degradation"] = pred_deg_class

        # remove segment arrays from annotation dict to save results
        for segment in annotations_dict["annotations"]:
            segment.pop('segment_crop', None)

        # annotated the model results with colored bbox
        annoted_img = test_img.copy()
        annoted_img = cv2.resize(annoted_img, (self.seg_input_width, self.seg_input_height))
        for segment in annotations_dict["annotations"]:
            coco_bbox = segment["bounding_box"]
            corner_bbox = box_coco_to_corner(coco_bbox)  # convert to corner format
            degradation_class = segment.get("degradation", -1)
            if degradation_class >= 0:
                label = _target_code_to_name[degradation_class]
                rgb_code = _color_map[degradation_class]
                annoted_img = add_bbox(img=annoted_img, bbox=corner_bbox, label=label, bbox_color=rgb_code,
                                       text_color=rgb_code)

        # save all the results

        # Make a folder for every image
        save_path = os.path.join(self.output_dir, img_name_without_ext)
        os.makedirs(save_path, exist_ok=True)

        # save predicted mask
        mask_name = f"{img_name_without_ext}_pred_mask.png"
        mask_path = os.path.join(save_path, mask_name)
        cv2.imwrite(mask_path, pred_mask)

        # save annotation json
        annotations_dict["predicted_mask"] = mask_name
        annot_name = f"{img_name_without_ext}.json"
        annot_path = os.path.join(save_path, annot_name)
        with open(annot_path, "w") as f:
            json.dump(annotations_dict, f)

        # save final annotated image
        annot_img_name = f"{img_name_without_ext}_annotated.jpg"
        annot_img_path = os.path.join(save_path, annot_img_name)
        annoted_img = cv2.cvtColor(annoted_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(annot_img_path, annoted_img)

        return {"save_path": save_path}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("output_dir", type=str, help="Directory to save output files.")
    parser.add_argument("--seg_weights", type=str,
                        default="segmentation/experiment_results/checkpoints/unet_final_2025-04-06_02-50-25.pth",
                        help="Path to segmentation model weights.")
    parser.add_argument("--classification_weights", type=str,
                        default="regression/experiment_results/classification_7april_best_weights/checkpoints/cnn_sppf_checkpoint_epoch_45.pth",
                        help="Path to classification model weights.")

    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using DEVICE: {DEVICE}")

    detector_model = DegradationDetector(segmentation_weights_path=args.seg_weights,
                                         classification_weights_path=args.classification_weights,
                                         output_dir=args.output_dir)

    tic = time.time()
    result_path = detector_model.predict(img_path=args.image_path, device=DEVICE)
    toc = time.time()
    print(f"Execution time: {round((toc - tic), 4)} sec")
    print("Pipeline result saved at:", result_path['save_path'])

