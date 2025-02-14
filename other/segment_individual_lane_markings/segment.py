import json
import os
import numpy as np
import cv2
import warnings
import tkinter as tk
from tkinter import Label
import bezier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
fig, ax = plt.subplots(figsize=(8, 6))

def construct_spline(curve_points):
    """Constructs a smooth spline curve."""
    curve = bezier.Curve.from_nodes(curve_points)
    space = np.linspace(0, 1, 100)
    eval = curve.evaluate_multi(space)
    xs = eval[0]
    ys = eval[1]
    return xs, ys
    # tck, u = splprep([curve_points[:, 0], curve_points[:, 1]], s=0, k=2)
    # u_fine = np.linspace(0, 1, 100)
    # curve_x, curve_y = splev(u_fine, tck)
    # return curve_x, curve_y

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
    # for group_dict in groups:
    #     for group_type, group_points in group_dict.items():
    #         if group_type == "L":
    #             x_list += [x for x, _ in group_points]
    #             y_list += [y for _, y in group_points]
    #         elif group_type == "C":
    #             spline_points = np.array(group_points, dtype=np.int32)
    #             xs, ys = construct_spline(spline_points.T)
    #             x_list += xs.tolist()
    #             y_list += ys.tolist()
    #             # x_list += [x for x, _ in group_points]
    #             # y_list += [y for _, y in group_points]
    #         else:
    #             warnings.warn(f"Unexpected data type: {group_type}")
    return x_list, y_list

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

    # Extract lane markings and plot them
    for frame in data["frames"]:
        for obj in frame["objects"]:
            if "lane" in obj["category"] and "poly2d" in obj:
                points = [(p[0], p[1], p[2]) for p in obj["poly2d"]]
                xs, ys = construct_line(points)
                if "C" in [c for x, y, c in obj["poly2d"]]:
                    color = "red"
                else:
                    color = "blue"
                ax.plot(xs, ys, label=obj["category"], color=color)

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