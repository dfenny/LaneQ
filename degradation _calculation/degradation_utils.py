import cv2
import numpy as np
from skimage.measure import label, regionprops


def generate_trapezoid_roi(img_shape, trap_height, trap_lower_width, trap_corner_offset, bottom_offset, center_offset=0):
    """
    Generate the coordinates of a symmetric trapezoidal region of interest (ROI) within an image.
    This function calculates the four corner coordinates of a trapezoidal region within an image.
    The trapezoid is defined by the following parameters: height, bottom width, corner offset, and bottom offset.
    
    Args:
        img_shape (tuple): shape of the image, given as a tuple (height, width)
        trap_height (int): The height of the trapezoid (distance between the top and bottom edges).
        trap_lower_width (int): The width of the bottom side of the trapezoid.
        trap_corner_offset (int): horizontal offset from the bottom corners to the top corners, effectively making the top of the trapezoid narrower.
        bottom_offset (int): vertical distance from the bottom of the image to the lower edge of the trapezoid.
        center_offset (int): additional horizontal offset to shift the entire trapezoid left or right. Defaults to 0, meaning no shift.

    Returns:
        list: list of four tuples, each containing the (x, y) coordinates of a corner of the trapezoid in the following order:
        top-left, top-right, bottom-right, bottom-left.
    """
    img_h, img_w = img_shape
    center = img_w // 2  # get image center

    # get horizontal levels for trapezoid
    low_y = img_h - bottom_offset
    top_y = low_y - trap_height

    # find x coord for lower corners of trapezoid

    # cal offset from center
    left_w = trap_lower_width // 2
    right_w = trap_lower_width - left_w

    # lower x corners
    low_leftx = center - left_w
    low_rightx = center + right_w

    # find top x corner
    top_leftx = low_leftx + trap_corner_offset
    top_rightx = low_rightx - trap_corner_offset

    # push RoI from center if required
    if center_offset != 0:
        low_leftx += center_offset
        low_rightx += center_offset
        top_leftx += center_offset
        top_rightx += center_offset

    # corners in order: tl, tr, br, bl
    trap_points = [(top_leftx, top_y) , (top_rightx, top_y), (low_rightx, low_y), (low_leftx, low_y)]
    return trap_points


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


def generate_connected_components(binary_img, connectivity=2, closing=True, kernel=5):

    if closing:
        kernel = np.ones((kernel, kernel), np.uint8)              # Define the structuring element
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)    # Perform closing

    # find connected components
    label_mask, label_count = label(binary_img, connectivity=connectivity, return_num=True)
    return label_mask, label_count


def cal_component_level_degradation(component_img, grayscale_img, good_threshold):

    # store degradation ratio
    degradation = []

    label_count = len(np.unique(component_img))    # 0 is background
    for l in range(1, label_count):

        # filter single component
        component_mask = (component_img == l).astype(np.uint8)

        # crop segmented region for each component
        component_region = cv2.bitwise_and(grayscale_img, grayscale_img, mask=component_mask)

        # find good intensity pixels
        undegraded_region = component_region >= good_threshold

        # calculate degradation ratio
        total_area = np.sum(component_mask)
        undegraded_area = np.sum(undegraded_region)
        degradation_ratio = 1 - (undegraded_area / total_area)
        degradation_ratio = round(degradation_ratio, 4)

        # calculate bbox for this component
        bbox = regionprops(component_mask, cache=False)[0].bbox
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]    # ensure proper sequence (xmin, ymin, xmax, ymax)

        degradation.append({"degradation": degradation_ratio, "bbox": bbox})

    return degradation


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


def get_bbox_corners(bbox):

    xmin, ymin, xmax, ymax = bbox  # unpack
    top_left = (xmin, ymin)
    top_right = (xmax, ymin)
    bottom_left = (xmin, ymax)
    bottom_right = (xmax, ymax)

    return top_left, top_right, bottom_right, bottom_left


def apply_bbox_inv_perspective_transform(degradation, orig_matrix):

    new_degradation = []

    # Compute the inverse perspective transform matrix
    matrix_inv = np.linalg.inv(orig_matrix)

    for component in degradation:
        ratio, bev_bbox = component["degradation"], component["bbox"]

        # need all four corners to perform perspective transform
        bbox_four = get_bbox_corners(bev_bbox)     # tl, tr, br, bl
        bbox_four = np.float32(bbox_four)
        bev_bbox = bbox_four.reshape(-1, 1, 2).astype(np.float32)    # reshape (N, 1, 2) format

        # Transform all four corners back to the original perspective view
        original_bbox = cv2.perspectiveTransform(bev_bbox, matrix_inv)     # (N, 1, 2)
        original_bbox = np.int32(np.squeeze(original_bbox))                # (N, 2)

        # above bbox is wrt to ROI, so also calculate bbox wrt image axes
        xmin, ymin = np.min(original_bbox[:, 0]), np.min(original_bbox[:, 1])
        xmax, ymax = np.max(original_bbox[:, 0]), np.max(original_bbox[:, 1])
        normal_bbox = np.int32([xmin, ymin, xmax, ymax]).tolist()

        new_degradation.append({"degradation": ratio, "bbox": normal_bbox, "oriented_bbox": original_bbox.tolist()})

    return new_degradation


_demo_settings = {

    "b1d0091f-75824d0d.jpg": {
        "roi_h": 280, "roi_lw": 1200, "rot_co": 440, "roi_bo": 0, "roi_ceo": 50,
        "threshold": 100
    },

    "b1d4b62c-60aab822.jpg": {
        "roi_h": 250, "roi_lw": 1280, "rot_co": 480, "roi_bo": 0, "roi_ceo": -60,
        "threshold": 45
    },

    "b5b02b31-f19988fb.jpg": {
        "roi_h": 300, "roi_lw": 1300, "rot_co": 480, "roi_bo": 50, "roi_ceo": 90,
        "threshold": 100
    },

    "b1ebfc3c-740ec84a.jpg": {
        "roi_h": 250, "roi_lw": 1300, "rot_co": 480, "roi_bo": 0, "roi_ceo": -60,
        "threshold": 100
    },

    "b6bdb46e-3709d206.jpg": {
        "roi_h": 250, "roi_lw": 1280, "rot_co": 480, "roi_bo": 0, "roi_ceo": -60,
        "threshold": 180
    },

    "b1d22449-117aa773.jpg": {
        "roi_h": 240, "roi_lw": 1280, "rot_co": 480, "roi_bo": 80, "roi_ceo": 30,
        "threshold": 180
    },

    "191_jpg.rf.e27c030e763e58ce48964e670158b6e7.jpg": {
        "roi_h": 340, "roi_lw": 850, "rot_co": 320, "roi_bo": 0, "roi_ceo": 60,
        "threshold": 180
    },

    "184_jpg.rf.9edad8ca8d25dc9949b48968cca6e41c.jpg": {
        "roi_h": 370, "roi_lw": 680, "rot_co": 200, "roi_bo": 0, "roi_ceo": 90,
        "threshold": 160
    }

}