from itertools import combinations
import json
import os
import numpy as np
import cv2
import bezier
from shapely import Polygon, LineString
from PIL import Image
from tqdm import trange

import matplotlib.pyplot as plt

# Directory paths
image_dir = os.path.expanduser("~/Downloads/bdd100k_images/train/")
json_dir = os.path.expanduser("~/Downloads/bdd100k_annotations/train/")
segmaps_dir = os.path.expanduser("~/Downloads/bdd100k_seg_maps/color_labels/train")

# Get all images
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
segmaps_files = sorted([f for f in os.listdir(segmaps_dir) if f.endswith(".png")])

lane_colors = {
    # "lane/crosswalk": "blue",
    "lane/double other": "purple",
    "lane/double white": "white",
    "lane/double yellow": "yellow",
    # "lane/road curb": "pink",
    "lane/single other": "olive",
    "lane/single white": "white",
    "lane/single yellow": "orange"
}


def construct_spline(curve_points):
    """Constructs a smooth spline curve."""
    curve = bezier.Curve.from_nodes(curve_points)
    space = np.linspace(0, 1, 100)
    eval = curve.evaluate_multi(space)
    xs = eval[0].tolist()
    ys = eval[1].tolist()
    return xs, ys


def group_lines(points):
    """Groups lane marking points into line or curve segments."""
    groups = []
    if "C" in [t for x, y, t in points]:
        groups.append({"C": [(x, y) for x, y, t in points]})
    else:
        groups.append({"L": [(x, y) for x, y, t in points]})
    return groups

def construct_line(points):
    """Processes lane marking points, constructing curves where necessary."""
    # groups = group_lines(points)
    if "C" in [t for x, y, t in points]:
        return construct_spline(
            np.array(
                [(x, y) for x, y, t in points]
            ).T
        )
    else:
        return [x for x, _, _ in points], [y for _, y, _ in points]

def construct_line_tuples(poly2d):
    """Constructs xs and ys, then feeds construct_line, and returns outputs like:
    
    [(x, y), ... , (x, y)]
    """
    points = [(p[0], p[1], p[2]) for p in poly2d]
    xs, ys = construct_line(points)
    return [(x, y) for x, y in zip(xs, ys)]

def get_largest_area_polygon(points1, points2):
    """ Checks both orders of two sets of points (two lines) to see which 
    creates the polygon with the smallest area.

    points must be in format [(x, y), ... , (x, y)]
    """
    line_1 = LineString(points1)
    line_2 = LineString(points2)
    greatest_allowable_hd = 100
    if line_1.hausdorff_distance(line_2) > greatest_allowable_hd:
        return None
    shape_1 = points1 + points2
    shape_2 = points1 + points2[::-1]
    poly_1 = Polygon(shape_1)
    poly_2 = Polygon(shape_2)
    return poly_1 if poly_1.is_valid else poly_2 if poly_2.is_valid else None

def sort_through_lines_to_find_matches(objects):
    """ Given a list of objects, find the lane types we're interested in and
    sort them out and then work with each group individually
    """
    # First group the objects into similar lane types
    obj_groups = {}
    for obj in objects:
        obj_groups.setdefault(
            (obj["category"], obj["attributes"]["direction"],
             obj["attributes"]["style"]), []
            ).append(
            construct_line_tuples(obj["poly2d"])
        )
    # Second for each line in each lane type find the right groupings
    polygons = {}
    for group_key, group in obj_groups.items():
        for possible_polygon in combinations(group, 2):
            polygon = get_largest_area_polygon(*possible_polygon)
            if polygon:
                polygons[polygon] = group_key
    # (make sure no polygons overlap or intersect)
    for poly1, poly2 in combinations(polygons, 2):
        intersection = poly1.intersection(poly2)
        if intersection.length > 0 or intersection.area > 0:
            if poly1.area < poly2.area:
                gone_poly = poly2
            else:
                gone_poly = poly1
            if gone_poly in polygons:
                del polygons[gone_poly]
    # Third pass back the areas covered by the groupings
    return polygons

def process_polygon_into_mask(polygon, lanetype, img, road_color):
    ln_cat, ln_dir, ln_style = lanetype
    paint_color = ln_cat.split(" ")[-1]
    # Create mask from polygon:
    image_shape = img.shape[:2]
    mask = np.zeros(image_shape, dtype=np.uint8)
    contour = np.array(polygon.exterior.coords, dtype=np.int32)
    cv2.fillPoly(mask, [contour], 255)
    # Handle solid lane lines first (they are simplest)
    if ln_style == "solid":
        return mask
    elif ln_dir == "parallel" and ln_style == "dashed":
        mask = refine_mask_by_color_distance(mask, img, road_color, tolerance=50)
    # mask = refine_mask_by_color_distance(mask, img, road_color, tolerance=50)
    return mask

def refine_mask_by_color_distance(mask, img, road_color, tolerance):
    # Convert image and road_color to int16 for safe subtraction
    img_int = img.astype(np.int16)
    road_color = np.array(road_color, dtype=np.int16)
    
    # Compute Euclidean distance for each pixel from road_color
    diff = np.linalg.norm(img_int - road_color, axis=2)
    
    # Create a mask where pixels beyond the tolerance are marked (True)
    distance_mask = (diff > tolerance).astype(np.uint8) * 255
    
    # Combine with the original mask
    refined_mask = cv2.bitwise_and(mask, distance_mask)
    return refined_mask

def extract_annotations(index):
    """Extracts the lane marking annotations from the JSON file for the given index."""
    
    img_path = os.path.join(image_dir, image_files[index])
    json_path = os.path.join(json_dir, json_files[index])
    segmaps_path = os.path.join(segmaps_dir, segmaps_files[index])

    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        img = np.ones((800, 1200, 3), dtype=np.uint8) * 255  # Default white image if loading fails
    
    segmap_mask = cv2.imread(segmaps_path)
    segmap_mask = cv2.cvtColor(segmap_mask, cv2.COLOR_BGR2RGB)

    # Load JSON annotations
    with open(json_path, "r") as f:
        data = json.load(f)

    objects = []
    for frame in data["frames"]:
        for obj in frame["objects"]:
            if obj["category"] in lane_colors and "poly2d" in obj:
                objects.append(obj)

    # Get polygons from lane markings
    polygons = sort_through_lines_to_find_matches(objects)

    # Get median road color
    road_mask_val = (128, 64, 128)
    rgb_array = np.array(road_mask_val, dtype=np.uint8)
    road_mask = np.all(segmap_mask == rgb_array, axis=2).astype(np.uint8) * 255
    road_pixels = img[road_mask > 0]
    road_color = np.median(road_pixels, axis=0)

    masks = []
    for polygon, lane_type in polygons.items():
        masks.append(
            process_polygon_into_mask(polygon, lane_type, img, road_color)
        )
    
    # Combine all masks:
    image_shape = img.shape[:2]
    combined_mask = np.zeros(image_shape, dtype=np.uint8)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    return img, objects, polygons, combined_mask

if __name__ == "__main__":
    out_dir = "masks"
    os.makedirs(out_dir, exist_ok=True)
    for ind in trange(len(image_files)):
        _, _, _, mask = extract_annotations(ind)
        msk_img = Image.fromarray(mask)
        img_name = image_files[ind]
        img_name = img_name.replace("jpg", "png")
        msk_img.save(os.path.join(out_dir, img_name))