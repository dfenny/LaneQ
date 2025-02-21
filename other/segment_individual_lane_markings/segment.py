from itertools import combinations
import json
import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import Label
import bezier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely import Polygon, LineString

# Directory paths
image_dir = os.path.expanduser("~/Downloads/bdd100k_images/train/")
json_dir = os.path.expanduser("~/Downloads/bdd100k_annotations/train/")

# Get all images
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

# Index tracker
current_index = 29

# Tkinter window
root = tk.Tk()
root.title("Lane Marking Viewer")

# Matplotlib figure
fig, ax = plt.subplots()

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

def load_image_and_annotations(index):
    """Loads the image and lane markings for the given index."""
    global ax

    ax.clear()

    # Load image
    img_path = os.path.join(image_dir, image_files[index])
    json_path = os.path.join(json_dir, json_files[index])

    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
    except:
        h, w = 800, 1200
        img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Load JSON annotations
    with open(json_path, "r") as f:
        data = json.load(f)

    # Show the image
    ax.imshow(img)

    # Define colors for each lane type
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

    # Store handles for legend
    legend_handles = []

    # Extract lane markings and plot them
    for frame in data["frames"]:
        objects = []
        for obj in frame["objects"]:
            if obj["category"] in lane_colors and "poly2d" in obj:
                objects.append(obj)
                points = [(p[0], p[1], p[2]) for p in obj["poly2d"]]
                xs, ys = construct_line(points)
                # if "C" in [c for x, y, c in obj["poly2d"]]:
                #     color = "red"
                # else:
                #     color = "blue"
                color = lane_colors.get(obj["category"], "red")
                line, = ax.plot(xs, ys, '--', label=obj["category"], color=color, lw=1)

                # Only add to legend if not already included
                if obj["category"] not in [handle.get_label() for handle in legend_handles]:
                    legend_handles.append(line)
        
        # get polygons from lane markings:
        polygons = sort_through_lines_to_find_matches(objects)

        for polygon, lane_type in polygons.items():
            x, y = polygon.exterior.xy
            color = lane_colors.get(lane_type, "red")
            # line, = ax.plot(x, y, label=lane_type, color=color, linewidth=1)
            ax.fill(x, y, color=color, alpha=0.5)
            # Only add to legend if not already included
            # if lane_type not in [handle.get_label() for handle in legend_handles]:
            #     legend_handles.append(line)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, title="Lane Markings")
    ax.set_title(f"Image {index+1}/{len(image_files)}: {image_files[index]}")
    ax.set_xticks([])
    ax.set_yticks([])

    canvas.draw()

def prev_image():
    """Loads the previous image."""
    global current_index
    if current_index > 0:
        current_index -= 1
        load_image_and_annotations(current_index)

def next_image():
    """Loads the next image."""
    global current_index
    if current_index < len(image_files) - 1:
        current_index += 1
        load_image_and_annotations(current_index)

# Create a Tkinter frame
frame = tk.Frame(root)
frame.pack()

# Embed Matplotlib figure into Tkinter
canvas = FigureCanvasTkAgg(fig, master=frame)

# Frame for navigation buttons
# button_frame = tk.Frame(root)
# button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

prev_button = tk.Button(frame, text="Previous Photo", command=prev_image)
prev_button.pack(side=tk.LEFT, padx=10, pady=5, expand=True)

next_button = tk.Button(frame, text="Next Photo", command=next_image)
next_button.pack(side=tk.RIGHT, padx=10, pady=5, expand=True)

canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
# Load the first image
load_image_and_annotations(current_index)

# Start the Tkinter main loop
root.mainloop()