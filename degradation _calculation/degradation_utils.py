import cv2
import numpy as np
from skimage.filters import threshold_otsu

def get_roi_points():
    # roi generation
    push_top_corner_wrt_road_region = 0.35    # percentage of additional offset to top corners of roi
    roi_side_slope_theta = 25            # slope angle in degree for slides of trapezoid roi

    upper_line = 350    # set upper edge of ROI
    lower_line = 720    # set base of ROI

    start_col = 256    # set start point (in x) for roi
    end_col = 1024     # set end point (in x) for roi

    # calculate additional corner offset
    roi_start, roi_end = start_col, end_col
    if push_top_corner_wrt_road_region > 0:
        offset = int(start_col * push_top_corner_wrt_road_region)
        roi_start, roi_end = (start_col + offset), (end_col - offset)

    # calculate x coordinate of lower corners of roi based on roi height and slope
    roi_height = lower_line - upper_line
    d = int(roi_height * np.tan(np.radians(90 - roi_side_slope_theta)))

    # calculate 4 corners of ROI in order: tl, tr, br, bl
    roi_points = [
        (roi_start, upper_line),
        (roi_end, upper_line),
        (roi_end + d, lower_line),
        (roi_start - d, lower_line)
    ]
    roi_points = np.array(roi_points)
    return roi_points


def generate_connected_components(binary_img, connectivity=8):

    # Perform connected component analysis
    # The function returns:
    #   num_labels: number of labels (including background)
    #   labels: image where each pixel has a label number
    #   stats: statistics for each label (e.g., bounding box, area)
    #         -> top-left-x, top-left-y, width, height, area
    #   centroids: center of each component
    num_labels, label_mask, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=connectivity)

    # keep only bounding box in stats
    stats = stats[:, :-1]

    return num_labels, label_mask, stats


def filter_connected_components(label_mask, roi_points, min_area=100, min_roi_overlap=0.6):

    # generate a ROI area mask to get pixels which are part of roi for transformation
    inside_roi = np.zeros(label_mask.shape[:2])
    inside_roi = cv2.fillPoly(inside_roi, pts=[roi_points], color=1).astype(np.uint8)
    inside_roi = cv2.bitwise_and(label_mask, label_mask, mask=inside_roi)

    # Calculate area(frequency) of labels in label_mask and inside_roi
    label_count = len(np.unique(label_mask))
    orig_area = np.bincount(label_mask.ravel(), minlength=label_count)
    inside_roi_area = np.bincount(inside_roi.ravel(), minlength=label_count)
    orig_area = np.delete(orig_area, 0)     # remove 0 for background
    inside_roi_area = np.delete(inside_roi_area, 0)   # remove 0 for background

    # Step: Create a mask for labels that meet the conditions
    # Condition 1: Original area must be greater than min_area
    # Condition 2: The ratio of area in ROI to the original area must be >= min_roi_overlap_area
    valid_labels = (orig_area > min_area) & ((inside_roi_area / orig_area) >= min_roi_overlap)
    labels = np.arange(1, label_count)
    valid_labels = labels[valid_labels]    # get list of valid connect component labels

    # set invalid labels/connected components to zero
    label_mask = np.where(np.isin(label_mask, valid_labels), label_mask, 0)
    return label_mask


def generate_perspective_matrix(roi_pts, output_shape):

    assert len(roi_pts) == 4, "roi_pts should contains four (x, y) points"

    original_points = np.array(roi_pts, np.float32)

    # generate points corresponding in transformed view
    output_w, output_h = output_shape
    transformed_pts = [(0, 0), (output_w, 0), (output_w, output_h), (0, output_h)]
    transformed_pts = np.array(transformed_pts, np.float32)

    matrix = cv2.getPerspectiveTransform(original_points, transformed_pts)
    return matrix


def apply_perspective_transform(img, matrix, output_shape, interpolation=cv2.INTER_CUBIC):
    bev = cv2.warpPerspective(img, matrix, output_shape, flags=interpolation)
    return bev


def cal_degradation_ratio(grayscale_img, component_mask, dilate_kernel=13):

    # dilate the mask to include surrounding road region near lane marking for calculating threshold
    kernel = np.ones((dilate_kernel, dilate_kernel))
    dilated_mask = cv2.dilate(component_mask, kernel, iterations=1)

    # use this dialted region and apply otsu to find a good threshold to identify lane paint
    temp = cv2.bitwise_and(grayscale_img, grayscale_img, mask=dilated_mask)
    good_threshold = threshold_otsu(temp[temp > 0])  # use only non-zero intensities

    # crop given component from the grayscale image using mask
    component_region = cv2.bitwise_and(grayscale_img, grayscale_img, mask=component_mask)

    # find good intensity pixels
    undegraded_region = (component_region >= good_threshold)

    # calculate degradation ratio
    total_area = np.sum(component_mask)
    undegraded_area = np.sum(undegraded_region)
    degradation_ratio = 1 - (undegraded_area / total_area)
    degradation_ratio = round(degradation_ratio, 4)
    return degradation_ratio


def add_bbox(img, bbox, label=None, bbox_color=(255, 255, 255), bbox_thickness=2, text_color=(0, 0, 0), font_scale=0.7):
    """
    Adds given bounding box to image with provided label (if any)

    Args:
        img (np.ndarray): image data
        bbox (list|tuple): list or tuple containing xmin, ymin, xmax, ymax for bounding box
        label (str): Label of bounding box
        bbox_color (tuple): RGB color code for bounding box
        bbox_thickness (int): bounding box line thickness
        text_color (tuple): RGB color code for label text
        font_scale (float): Scale for label text font

    Returns:
        np.ndarray: image with added bbox
    """
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    # For bounding box
    x1, y1, x2, y2 = bbox
    img = cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=bbox_color, thickness=bbox_thickness)

    if label is not None:
        # For the text background - finds space required by the text
        (w, h), _ = cv2.getTextSize(text=label, fontFace=_FONT, fontScale=font_scale, thickness=1)

        # Add a rect-background for label text (negative thickness to fill rectangle)
        lrect_pt1 = x1 - 1, y1 - (h + bbox_thickness)
        lrect_pt2 = x1 + w, y1 - bbox_thickness
        img = cv2.rectangle(img=img, pt1=lrect_pt1, pt2=lrect_pt2, color=bbox_color, thickness=-1)
        img = cv2.putText(img=img, text=label, org=(x1, y1 - bbox_thickness), fontFace=_FONT, fontScale=font_scale,
                          color=text_color, thickness=1)

    return img


def box_coco_to_corner(bbox):
    """Convert from (upper-left, width, height) to (upper-left, bottom-right)"""
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    boxes = (x1, y1, x2, y2)
    return boxes
