import cv2
import numpy as np
from skimage.filters import threshold_li

def get_roi_points():
    """
    Generates a set of four points defining a trapezoidal Region of Interest (ROI) in an image,
    typically used for perspective transformations (e.g., bird's-eye view in road scenes).
    Currently, ROI location is fixed

    The ROI is defined as a trapezoid with:
        - A top edge (upper_line)
        - A base edge (lower_line)
        - Optional inward offset to top corners (push_top_corner_wrt_road_region)
        - Sloped sides defined by an angle (roi_side_slope_theta)

    Returns:
        np.ndarray: An array of shape (4, 2) containing the coordinates of the ROI's four corners in the following order:
            - Top-left (tl), Top-right (tr), Bottom-right (br), Bottom-left (bl)

    Notes:
        - Coordinates are returned in pixel space.
        - Adjusting `push_top_corner_wrt_road_region` controls how much the top corners are shifted inward.
        - `roi_side_slope_theta` controls the angle of the trapezoid's side walls.
    """
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
    """
    Performs connected component analysis on a binary image and extracts bounding boxes for each component.

    Parameters:
        binary_img (np.ndarray): Binary image (2D array) where non-zero pixels are considered foreground.
        connectivity (int, optional): Connectivity used for grouping pixels into components.
            - 4 for 4-way connectivity (up, down, left, right)
            - 8 for 8-way connectivity (includes diagonals)
            Default is 8.

    Returns:
        tuple:
            - num_labels (int): Total number of connected components found, including background (label 0).
            - label_mask (np.ndarray): Labeled image of the same shape as `binary_img`, where each pixel is assigned
              a label corresponding to a connected component.
            - stats (np.ndarray): Array of shape (num_labels, 4), where each row contains the bounding box of a component:
                [top-left-x, top-left-y, width, height]

    Notes:
        - The background is typically assigned label 0.
        - The `stats` output excludes the area column and only returns bounding box information.
    """

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
    """
    Filters connected components in a labeled mask based on size and overlap with a defined Region of Interest (ROI).

    Parameters:
       label_mask (np.ndarray): Labeled mask image (2D array), where each non-zero integer represents a unique component.
       roi_points (np.ndarray): Array of shape (4, 2) defining a polygon (typically trapezoidal) ROI.
       min_area (int, optional): Minimum number of pixels required for a component to be retained. Default is 100.
       min_roi_overlap (float, optional): Minimum ratio of component area that must lie within the ROI. Default is 0.6.

    Returns:
       np.ndarray: A filtered labeled mask of the same shape as `label_mask`, where:
           - Only components meeting both the area and ROI overlap conditions are retained.
           - All other component pixels are set to 0 (background).

    Notes:
       - ROI is defined using polygon fill and applied as a binary mask.
       - Background label (0) is excluded from filtering conditions.
       - This function is typically used before further processing like degradation analysis or perspective transformation.
    """

    # generate a ROI area mask to get pixels which are part of roi for transformation
    inside_roi = np.zeros(label_mask.shape[:2])
    inside_roi = cv2.fillPoly(inside_roi, pts=[roi_points], color=1).astype(np.uint8)
    inside_roi = cv2.bitwise_and(label_mask, label_mask, mask=inside_roi)

    # Calculate area(frequency) of labels in label_mask and inside_roi
    label_count = len(np.unique(label_mask))
    if label_count == 1:
        return label_mask

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
    """
    Generates a perspective transformation matrix to map a quadrilateral ROI to a rectangular output view.

    Parameters:
        roi_pts (array-like): List or array of four (x, y) points defining the source quadrilateral ROI.
                              The points should be ordered as:
                              [top-left, top-right, bottom-right, bottom-left].
        output_shape (tuple): Desired output shape as (width, height) of the transformed view.

    Returns:
        np.ndarray: 3x3 perspective transformation matrix (homography) that maps `roi_pts` to a rectangular region
                    of size `output_shape`.

    Notes:
        - This matrix can be used with `cv2.warpPerspective` to perform the actual transformation.
        - Ensure consistent point order between source and target for accurate results.
    """
    assert len(roi_pts) == 4, "roi_pts should contains four (x, y) points"

    original_points = np.array(roi_pts, np.float32)

    # generate points corresponding in transformed view
    output_w, output_h = output_shape
    transformed_pts = [(0, 0), (output_w, 0), (output_w, output_h), (0, output_h)]
    transformed_pts = np.array(transformed_pts, np.float32)

    matrix = cv2.getPerspectiveTransform(original_points, transformed_pts)
    return matrix


def apply_perspective_transform(img, matrix, output_shape, interpolation=cv2.INTER_CUBIC):
    """
    Applies a perspective transformation to an image using a given transformation matrix.

    Parameters:
        img (np.ndarray): Input image to be transformed (can be grayscale or color).
        matrix (np.ndarray): 3x3 perspective transformation matrix (usually from cv2.getPerspectiveTransform).
        output_shape (tuple): Desired output size as (width, height).
        interpolation (int, optional): Interpolation method used during warping. Default is `cv2.INTER_CUBIC`.

    Returns:
        np.ndarray: The perspective-transformed image with dimensions `output_shape`.
    """
    bev = cv2.warpPerspective(img, matrix, output_shape, flags=interpolation)
    return bev


def cal_degradation_ratio(grayscale_img, component_mask, dilate_kernel=13, comp_len_limit=100, sub_comp_len=80,
                          relax_threshold=0.05):
    """
    Calculates the degradation ratio of a component in a grayscale image by comparing its intensity with
    surrounding road regions using adaptive thresholding.

    Parameters:
        grayscale_img (np.ndarray): 2D grayscale image containing the scene.
        component_mask (np.ndarray): Binary mask of the component (same shape as `grayscale_img`), where non-zero pixels
                                     represent the region of interest (e.g., a lane marking).
        dilate_kernel (int, optional): Size of the square kernel used to dilate the component mask to include nearby road.
                                       Default is 13.
        comp_len_limit (int, optional): Vertical length threshold to determine whether to split the component into subregions.
                                        Default is 100.
        sub_comp_len (int, optional): Height of each sub-region when the component is divided for localized analysis.
                                      Default is 80.
        relax_threshold (float, optional): Percentage by which the adaptive threshold is relaxed to be more tolerant
                                           in detecting un-degraded pixels. Default is 0.05.

    Returns:
        float: Degradation ratio in the range [0, 1], where:
               - 0 means fully undegraded (all pixels above threshold)
               - 1 means fully degraded (none of the pixels above threshold)

    Method:
        - Dilation is applied to include neighboring road region.
        - Adaptive thresholding (Li's method) is used on the region around the component.
        - Each pixel in the component is classified as "undegraded" or not.
        - The degradation ratio is computed as the fraction of degraded pixels in the component.

    Notes:
        - If the component is long enough (based on `comp_len_limit`), it is divided into horizontal bins
          and degradation is assessed per bin.
        - Li’s thresholding is applied only on non-zero pixels in the road+lane cropped region.
    """

    # dilate the mask to include surrounding road region near lane marking for calculating threshold
    kernel = np.ones((dilate_kernel, dilate_kernel))
    dilated_mask = cv2.dilate(component_mask, kernel, iterations=1)

    # use this dialted region to get neighboring road color info
    road_plus_lane = cv2.bitwise_and(grayscale_img, grayscale_img, mask=dilated_mask)

    # crop given component from the grayscale image using mask
    component_region = cv2.bitwise_and(grayscale_img, grayscale_img, mask=component_mask)

    # placeholder to label each pixel as good or bad
    type_mask = component_mask.copy()

    # if lane is too long calculate degradation by dividing in horizontal bins
    non_zero_y, non_zero_x = np.nonzero(component_mask)
    ymin, ymax = non_zero_y.min(), non_zero_y.max()
    if (ymax - ymin) > comp_len_limit:
        # divide the component into horizontal bins
        sub_comp_bins = np.arange(ymax, ymin, -sub_comp_len)
        if sub_comp_bins[-1] > ymin:
            # if final bin doesnt cover the end of component add it
            sub_comp_bins = np.append(sub_comp_bins, ymin)

        # each element is a component division y value as (ymax, ymin)
        sub_comp_bins = list(zip(sub_comp_bins, sub_comp_bins[1:]))

    else:
        sub_comp_bins = [(ymax, ymin)]

    # do thresholding on each subcomponent separately:
    for sub_bin in sub_comp_bins:

        # crop subcomponents containing both road and lane
        road_lane_crop = road_plus_lane[sub_bin[1]:sub_bin[0], :]

        # apply li thresholding to find threshold to separate lane color and road
        good_threshold = threshold_li(road_lane_crop[road_lane_crop > 0])  # use only non-zero intensities

        if relax_threshold:
            # relax the threshold further if required
            good_threshold = good_threshold * (1 - relax_threshold)

        # find good intensity pixels
        component_crop = component_region[sub_bin[1]:sub_bin[0], :]
        undegraded_region = (component_crop >= good_threshold)

        # add the info to type mask
        type_mask[sub_bin[1]:sub_bin[0], :][undegraded_region] = 2

        # calculate the total degradation
    total_area = np.sum(type_mask > 0)  # count of non zero pixels
    undegraded_area = np.sum(type_mask == 2)  # pixels marked as good (label:2)
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
                          color=text_color, thickness=2)

    return img


def box_coco_to_corner(bbox):
    """Convert from (upper-left, width, height) to (upper-left, bottom-right)"""
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    boxes = (x1, y1, x2, y2)
    return boxes


def expand_bbox(coco_bbox, image_width, image_height, padding=10):
    """
    Expands the COCO bounding box by adding padding around it.

    Args:
        coco_bbox (list): Original bounding box in COCO format [x_min, y_min, width, height].
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        padding (int): The amount of padding to add around the bounding box.

    Returns:
        list: The new expanded bounding box [x_min, y_min, width, height].
    """
    # Unpack original bounding box
    x_min, y_min, width, height = coco_bbox

    # Expand the bounding box by the padding amount
    x_min_expanded = max(x_min - padding, 0)  # Ensure the x_min is not less than 0
    y_min_expanded = max(y_min - padding, 0)  # Ensure the y_min is not less than 0
    width_expanded = min(x_min + width + padding, image_width) - x_min_expanded  # Ensure width doesn't go beyond image
    height_expanded = min(y_min + height + padding,
                          image_height) - y_min_expanded  # Ensure height doesn't go beyond image

    # Return the expanded bounding box
    return [x_min_expanded, y_min_expanded, width_expanded, height_expanded]