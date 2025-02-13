import json
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import splprep, splev

# Load JSON file
json_file = os.path.expanduser("~/Downloads/bdd100k_annotations/train/0a0a0b1a-7c39d841.json")
with open(json_file, "r") as f:
    data = json.load(f)

# Load an image (optional) or create a blank canvas
image_path = os.path.expanduser("~/Downloads/bdd100k_images/train/0a0a0b1a-7c39d841.jpg")
try:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    h, w, _ = img.shape
except:
    h, w = 800, 1200  # Default canvas size if no image
    img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

# Plot the image
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(img)

def construct_spline(curve_points):
    tck, u = splprep([curve_points[:, 0], curve_points[:, 1]], s=0, k=2)
    curve_x, curve_y = splev(u, tck)
    return curve_x, curve_y

def group_lines(points):
    # points should be list of lists with nested lists three items each
    # x, y, point_type
    groups = []
    for j in range(len(points)):
        current_type = points[j][2]
        prev_type = points[j - 1][2] if j != 0 else None
        if current_type == prev_type:
            groups[-1][current_type].append(points[j][:-1])
        else:
            groups.append({current_type: [points[j][:-1]]})
    return groups

def construct_line(points):
    groups = group_lines(points)
    x_list = []
    y_list = []
    for group_dict in groups:
        for group_type, group_points in group_dict.items():
            if group_type == "L":
                x_list += [x for x, _ in group_points]
                y_list += [y for _, y in group_points]
            elif group_type == "C":
                spline_points = np.array(group_points, dtype=np.int32)
                xs, ys = construct_spline(spline_points)
                x_list += xs.tolist()
                y_list += ys.tolist()
            else:
                warnings.warn(f"Unexpected data type: {group_type}")
    return x_list, y_list


# Extract poly2d data and plot only lane-related categories
for frame in data["frames"]:
    for obj in frame["objects"]:
        if "lane" in obj["category"] and "poly2d" in obj:
            points = [(p[0], p[1], p[2]) for p in obj["poly2d"]]

            xs, ys = construct_line(points)
            ax.plot(xs, ys, label=obj["category"])


# Show the image with annotations
ax.set_title("Annotated Scene with Lane Markings")
ax.set_xticks([])
ax.set_yticks([])
plt.show()