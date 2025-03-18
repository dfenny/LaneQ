# Importing all the necessary funtions from segmentation and regression
import cv2
import json
import torch
import numpy as np
from torchvision import transforms

from segmentation.inference import load_saved_model as load_saved_segmentation_model, pred_segmentation_mask
from regression.inference import load_saved_model as load_saved_regression_model, pred_degradation_value
from segmentation.utils.preprocessing import load_image, get_img_transform, load_mask, apply_img_preprocessing

from degradation_calculation.calculate_degradation import generate_individual_segments_and_dict


# ==============================================
# Threshold values for degradation

mild_degradation_threshold = 0.3
moderate_degradation_threshold = 0.6

# ==============================================

# Transforms
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((720, 1280)),   # ensure resize same is used for mask by setting preprocess_config
    transforms.ToTensor()
])

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Segmentation model
seg_saved_weight_path = "segmentation/experiment_results/checkpoints/unet_checkpoint_epoch_5.pth"
segmentation_model_name = "unet"
segmentation_model_config = {'in_channels': 3, 'out_channels': 1}

# Regression model    
reg_saved_weight_path = "regression/experiment_results/checkpoints/cnn_sppf_final_2025-03-17_23-57-21.pth"
regression_model_name = "cnn_sppf"
regression_model_config = {'in_channels': 3, 'out_dim': 1}

# Load the saved models
segmentation_model = load_saved_segmentation_model(model_name=segmentation_model_name, saved_weight_path=seg_saved_weight_path, **segmentation_model_config)
regression_model = load_saved_regression_model(model_name=regression_model_name, saved_weight_path=reg_saved_weight_path, **regression_model_config)

# Now I'll be running inference of segmetnation model and storing it's results in a JSON file for each input image

# Loading image
input_image_path = "D:/Ankith B V/Documents/UofU/Datasets/BDD100k/cs6945share/retro_project/bdd100k/images/test/cabc9045-b3349548.jpg"
input_img = load_image(input_image_path)

# Predicting the segmentation mask
predicted_mask = pred_segmentation_mask(model=segmentation_model, test_img=input_img, img_transform=img_transform, add_batch_dim=True, device=DEVICE)
predicted_mask = np.squeeze(predicted_mask)
predicted_mask = predicted_mask.astype(np.uint8)
# print(predicted_mask.shape)

# Sample segmentation mask
# sample_segmentation_mask_path = "damage_ratio_calc_data/damage_ratio_masks/SegmentationClass/183_jpg.rf.c3a79aad316f75fe23536c3bbbd6da51.png"
# predicted_mask_0 = load_mask(sample_segmentation_mask_path)
# predicted_mask_0 = cv2.imread(sample_segmentation_mask_path, cv2.IMREAD_GRAYSCALE)
# print(predicted_mask_0.shape)

# Convert the detctions into individual segments and store information in a JSON file
filename = input_image_path.split("/")[-1]
print(filename)

annotations_dict = generate_individual_segments_and_dict(img=input_img, mask=predicted_mask, filename=filename)
# On second thought, do we even need to store the individual segments in a separate folder? We can just store the segment coords and dynamically generate the segments during regression

print(annotations_dict)

# Store the segment information in a JSON file for the input image (Converting this to a JSON file after the regression model inference makes more sense)
# with open(f"damage_ratio_calc_data/{"".join(filename.split(".")[0:-1])}.json", "w") as f:
#     json.dump(annotations_dict, f)

# Next I'll be running inference of regression model on these individual segments and storing it's results in the same JSON file of the input image
for segment in annotations_dict["annotations"]:
    
    # TODO: I have a doubt here. Sicne we're only using the coordinates of the bounding box to crop the segment, we might crop in nearby lane lines too. Does anyone know what we can do here?
    x, y, w, h = segment["bounding_box"]  # Extract bounding box
    segment_img = input_img[y:y+h, x:x+w]

    # Predicting the degradation value
    degradation_value = pred_degradation_value(model=regression_model, test_img=segment_img, img_transform=None, add_batch_dim=True, device=DEVICE)

    # Setting the degradation value to the segment predicted value
    segment["degradation"] = degradation_value
    # segment["degradation"] = -1


# Store all the information in a JSON file of the input image
filename_without_ext = "".join(filename.split(".")[0:-1])
with open(f"damage_ratio_calc_data/{filename_without_ext}.json", "w") as f:
    json.dump(annotations_dict, f)

# Read the JSON file and write down bounding boxes and degradation values on the image
with open(f"damage_ratio_calc_data/{filename_without_ext}.json", "r") as f:
    annotations_dict = json.load(f)

input_img = cv2.imread(input_image_path)
for segment in annotations_dict["annotations"]:
    x, y, w, h = segment["bounding_box"]
    cv2.rectangle(input_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(input_img, f"{segment['degradation']:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if segment["degradation"] < mild_degradation_threshold:
        cv2.rectangle(input_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    elif segment["degradation"] < moderate_degradation_threshold:
        cv2.rectangle(input_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
    else:
        cv2.rectangle(input_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imwrite(f"damage_ratio_calc_data/{filename_without_ext}_annotated.jpg", input_img)