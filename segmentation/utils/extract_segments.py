import json
import os
import cv2
import numpy as np

def get_masks_and_bounding_boxes(mask_path, output_dir="."):
    """ Takes mask.png and produces output in a dir of individual mask.pngs and json info

    INPUTS:
    mask_path: path-like (could be string representing path) to mask.png
    output_dir: The directory you want all of the outputs in. Please note that
                a sub-directory will be created to store the info for the mask
                provided. Defaults to the directory you are running the script
                from.

    Example Usage:
    get_masks_and_bounding_boxes(mask_path="0dsfk2-32jklfd.png", output_dir="data/refined_masks")
    """
    img_name = os.path.basename(mask_path)
    img_name_no_file_type = img_name.split(".")[0]
    individual_img_output_dir = os.path.join(output_dir, img_name_no_file_type)
    os.makedirs(individual_img_output_dir, exist_ok=True)

    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the image is binary (if not already)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Perform connected component analysis
    # The function returns:
    #   num_labels: number of labels (including background)
    #   labels: image where each pixel has a label number
    #   stats: statistics for each label (e.g., bounding box, area)
    #   centroids: center of each component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    results = []
    # Loop over each component (skip label 0, which is the background)
    for i in range(1, num_labels):
        # Create a mask for the current object
        mask = (labels == i).astype(np.uint8) * 255
        # Write the mask to the output dir
        mask_out_path = os.path.join(individual_img_output_dir, f"mask_{i}.png")
        cv2.imwrite(mask_out_path, mask)
        mask_dict = {
            'id': i,
            'bounding_box': stats[i][:-1].tolist()
        }
        results.append(mask_dict)
    annotations_dict = {
        'image': img_name,
        'annotations': results
    }
    json_out_path = os.path.join(individual_img_output_dir, f"{img_name_no_file_type}.json")
    with open(json_out_path, 'w') as jspot:
        json.dump(annotations_dict, jspot)
