import os
import time
import json
import argparse
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd

import degradation_utils as hp


def fetch_all_components_degradation(gray_img, label_mask, dilate_kernel=13):
    degradation_info = {}
    for idx in np.unique(label_mask):
        if idx == 0:
            continue

        component_mask = (label_mask == idx).astype(np.uint8)  # Create a mask for the current object
        degradation_ratio = hp.cal_degradation_ratio(grayscale_img=gray_img, component_mask=component_mask,
                                                     dilate_kernel=dilate_kernel)
        degradation_info[idx] = degradation_ratio
    return degradation_info


def get_degradation_annotations_n_segment_labels(img, mask, segment_output_dir, save_name, bev_shape=(640, 640),
                                                 min_area=100, min_roi_overlap=0.6, dilated_kernel=13):

    # generate connected components
    num_labels, label_mask, bboxes = hp.generate_connected_components(mask, connectivity=8)

    # get ROI for transformation
    roi_points = hp.get_roi_points()

    # filtering of smaller components and components with less overlap with roi
    filtered_label_mask = hp.filter_connected_components(label_mask=label_mask, roi_points=roi_points,
                                                         min_area=min_area, min_roi_overlap=min_roi_overlap)

    # get transformation matrix
    matrix = hp.generate_perspective_matrix(roi_pts=roi_points, output_shape=bev_shape)

    # Apply transformation on image
    img_bev = hp.apply_perspective_transform(img=img.copy(), matrix=matrix, output_shape=bev_shape)
    gray_img_bev = cv2.cvtColor(img_bev, cv2.COLOR_RGB2GRAY)   # converting image to grayscale

    # Apply transformation on labelled mask
    filtered_label_mask_bev = hp.apply_perspective_transform(img=filtered_label_mask.copy(), matrix=matrix,
                                                             output_shape=bev_shape,
                                                             interpolation=cv2.INTER_NEAREST)

    # get degradation ratio for each valid components
    component_degradation = fetch_all_components_degradation(gray_img=gray_img_bev, label_mask=filtered_label_mask_bev,
                                                             dilate_kernel=dilated_kernel)

    # Loop over each component (skip label 0, which is the background)
    segment_labels = []
    annot_results = []
    for idx in range(1, num_labels):

        degradation_ratio = component_degradation.get(idx, -1)   # get degradation ratio (-1 if not calculated)
        coco_bbox = bboxes[idx].tolist()   # get bbox

        # save separate segment as png if degradation ratio is calculated
        if degradation_ratio >= 0:
            xmin, ymin, xmax, ymax = hp.box_coco_to_corner(coco_bbox)

            # get segment in original image
            orig_mask = (label_mask == idx).astype(np.uint8)
            segment = cv2.bitwise_and(img, img, mask=orig_mask)         # apply mask
            segment = segment[ymin:ymax + 1, xmin:xmax + 1].copy()      # get specific segment crop

            # Write the segment to the output dir
            segment_name = f"{save_name}_{idx}.png"
            segment_path = os.path.join(segment_output_dir, segment_name)
            cv2.imwrite(segment_path, segment)

            segment_info = {
                "name": segment_name,
                'degradation': degradation_ratio
            }
            segment_labels.append(segment_info)

        # store annotations
        mask_dict = {
            'id': idx,
            'bounding_box': coco_bbox,
            'degradation': degradation_ratio
        }
        annot_results.append(mask_dict)

    return annot_results, segment_labels



def generate_degradation_annotations(image_dir, mask_dir, segment_output_dir, annotations_output_dir, bev_shape=(640, 640),
                                     min_area=100, min_roi_overlap=0.6, dilated_kernel=13):

    # ensure all necessary folders are available
    os.makedirs(segment_output_dir, exist_ok=True)
    os.makedirs(annotations_output_dir, exist_ok=True)

    # get all file names
    file_list = sorted(os.listdir(image_dir))
    segment_label_list = []

    main_tic = time.time()
    for filename in tqdm(file_list):

        # read image and mask
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        file_initials = ".".join(filename.split(".")[:-1])  # remove extension
        mask_name = file_initials + ".png"
        mask_path = os.path.join(mask_dir, mask_name)

        mask_img = cv2.imread(mask_path)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        # if range of image is [0, 255] normalize it
        if np.max(mask_img) > 1:
            mask_img = mask_img / 255  # normalize
        mask_img = mask_img.astype(np.uint8)

        # get annotation and segment labels to save in to json and
        annotations, segment_labels = get_degradation_annotations_n_segment_labels(img=img, mask=mask_img,
                                                                                   segment_output_dir=segment_output_dir,
                                                                                   save_name=file_initials,
                                                                                   bev_shape=bev_shape,
                                                                                   min_area=min_area,
                                                                                   min_roi_overlap=min_roi_overlap,
                                                                                   dilated_kernel=dilated_kernel)

        segment_label_list.extend(segment_labels)

        # save individual annotations to JSON file
        annotations_dict = {
            'image': filename,
            'annotations': annotations
        }
        json_out_path = os.path.join(annotations_output_dir, f"{file_initials}.json")
        with open(json_out_path, 'w') as jspot:
            json.dump(annotations_dict, jspot)

    main_toc = time.time()
    print(f"Degradation labelling Completed! Total time: {round((main_toc - main_tic) / 60, 4)} min")

    # convert segments labels to csv and save
    segment_df = pd.DataFrame(segment_label_list)
    fn = f"degradation_segment_labels.csv"
    fn = os.path.join(segment_output_dir, fn)
    segment_df.to_csv(fn, index=False)
    print(f"Segment label file saved at {fn}")



if __name__ == '__main__':

    # Parsing the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='Path to folder containing images')
    parser.add_argument('--mask_dir', type=str, help='Path to folder containing segmentation mask of images')
    parser.add_argument('--segment_output_dir', type=str, help='Path to folder where individual segments needs to be saved')
    parser.add_argument('--annotations_output_dir', type=str, help='Path to folder where annotations needs to be saved')
    args = parser.parse_args()

    # start generating labels
    generate_degradation_annotations(image_dir=args.image_dir, mask_dir=args.mask_dir,
                                     segment_output_dir=args.segment_output_dir,
                                     annotations_output_dir=args.annotations_output_dir)
