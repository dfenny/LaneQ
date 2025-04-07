# Importing all the necessary funtions from segmentation and regression
import os
import cv2
import json
import torch
import numpy as np
import argparse
from torchvision import transforms

from segmentation.utils.preprocessing import load_image
from segmentation.inference import load_saved_model as load_saved_segmentation_model, pred_segmentation_mask
from regression.inference import load_saved_model as load_saved_regression_model, pred_degradation_value

from degradation_calculation.calculate_degradation import generate_individual_segments_and_dict, generate_individual_segments_and_dict_v2


def run_inference(input_image: str, 
                  output_dir: str, 
                  seg_checkpoint: str = "segmentation/experiment_results/checkpoints/unet_final_2025-04-06_02-50-25.pth", 
                  reg_checkpoint: str = "regression/experiment_results/classification_7april_best_weights/checkpoints/cnn_sppf_checkpoint_epoch_45.pth",
                  mild_threshold: float = 0.33, 
                  moderate_threshold: float = 0.66) -> dict:
    """
    Runs inference for an input image and stores the results in specified directory.
    
    Args:   
        input_image (str): Path to the input image.
        output_dir (str): Directory to save output images and JSON files.
        seg_checkpoint (str): Path to the segmentation model checkpoint.
        reg_checkpoint (str): Path to the regression model checkpoint.
        mild_threshold (float): Threshold for mild degradation.
        moderate_threshold (float): Threshold for moderate degradation.
    
    Returns:
        dict: Dictionary containing paths to output files.
    """

    os.makedirs(output_dir, exist_ok=True)
    
    # Transforms
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((720, 1280)),
        transforms.ToTensor()
    ])
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model configs
    segmentation_model_config = {'in_channels': 3, 'out_channels': 1}
    regression_model_config = {'in_channels': 3, 'out_dim': 3}

    # Load models
    segmentation_model = load_saved_segmentation_model(model_name="unet", saved_weight_path=seg_checkpoint, **segmentation_model_config)
    regression_model = load_saved_regression_model(model_name="cnn_sppf", saved_weight_path=reg_checkpoint, **regression_model_config)
    
    # Load image
    input_img = load_image(input_image)
    
    # Predict segmentation mask
    predicted_mask = pred_segmentation_mask(model=segmentation_model, test_img=input_img, img_transform=img_transform, add_batch_dim=True, device=DEVICE)
    predicted_mask = np.squeeze(predicted_mask).astype(np.uint8) * 255
    
    filename = os.path.basename(input_image)
    filename_without_ext = os.path.splitext(filename)[0]
    
    # Make a folder for every image
    os.makedirs(output_dir + "/" + filename_without_ext, exist_ok=True)

    mask_path = os.path.join(output_dir + "/" + filename_without_ext, f"{filename_without_ext}_mask.jpg")
    cv2.imwrite(mask_path, predicted_mask)
    
    # Generate segment annotations
    annotations_dict = generate_individual_segments_and_dict_v2(img=input_img, mask=predicted_mask, filename=filename)

    # Run regression model on each segment
    for segment in annotations_dict["annotations"]:
        x, y, w, h = segment["bounding_box"]
        # segment_img = input_img[y:y+h, x:x+w]
        segment_img = segment["segment_crop"]
        
        # Do not pass small segments into the regression model
        if w < 10 or h < 10:
            continue
        
        degradation_value = pred_degradation_value(model=regression_model, test_img=segment_img, img_transform=None, add_batch_dim=True, device=DEVICE)
        segment["degradation"] = degradation_value

    new_annot = []
    for annot in annotations_dict["annotations"]:
        del annot['segment_crop']
        new_annot.append(annot.copy())

    annotations_dict["annotations"] = new_annot.copy()

    # Save JSON file
    json_path = os.path.join(output_dir + "/" + filename_without_ext, f"{filename_without_ext}.json")
    with open(json_path, "w") as f:
        json.dump(annotations_dict, f)
    
    # Annotate and save image
    input_img_annotated = cv2.imread(input_image)
    input_img_annotated = cv2.resize(input_img_annotated, (1280, 720))
    
    # Plot stuff
    for segment in annotations_dict["annotations"]:
        x, y, w, h = segment["bounding_box"]
        degradation = segment["degradation"]
        
        # if degradation < mild_threshold and degradation > 0:
        if degradation == 0:
            cv2.rectangle(input_img_annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.putText(input_img_annotated, f"{degradation:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(input_img_annotated, f"Good", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # elif mild_threshold <= degradation < moderate_threshold:
        elif degradation == 1:
            cv2.rectangle(input_img_annotated, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(input_img_annotated, f"Slight", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        # elif degradation > mild_threshold:
        elif degradation == 2:
            cv2.rectangle(input_img_annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(input_img_annotated, f"Severe", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        
    
    annotated_img_path = os.path.join(output_dir + "/" + filename_without_ext, f"{filename_without_ext}_annotated.jpg")
    cv2.imwrite(annotated_img_path, input_img_annotated)
    
    # return {
    #     "mask_path": mask_path,
    #     "json_path": json_path,
    #     "annotated_img_path": annotated_img_path
    # }

    return { "Path to results": output_dir + "/" + filename_without_ext}

def main():
    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument("input_image", type=str, help="Path to the input image.")
    parser.add_argument("output_dir", type=str, help="Directory to save output files.")
    parser.add_argument("--seg_checkpoint", type=str, default="segmentation/experiment_results/checkpoints/unet_final_2025-04-06_02-50-25.pth", help="Path to segmentation model weights.")
    parser.add_argument("--reg_checkpoint", type=str, default="regression/experiment_results/classification_7april_best_weights/checkpoints/cnn_sppf_checkpoint_epoch_45.pth", help="Path to regression model weights.")
    parser.add_argument("--mild_threshold", default=0.33, type=float, help="Threshold for mild degradation.")
    parser.add_argument("--moderate_threshold", default=0.66, type=float, help="Threshold for moderate degradation.")
    
    args = parser.parse_args()
    
    results = run_inference(
        output_dir=args.output_dir,
        input_image=args.input_image,
        mild_threshold=args.mild_threshold,
        moderate_threshold=args.moderate_threshold,
        seg_checkpoint=args.seg_checkpoint,
        reg_checkpoint=args.reg_checkpoint
    )

    for k, v in results.items():
        print(f"{k} : {v}")

if __name__ == "__main__":
    main()