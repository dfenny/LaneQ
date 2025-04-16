from segment import *
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Index tracker
current_index = 29


# Tkinter window
root = tk.Tk()
root.title("Lane Marking Viewer")

# Matplotlib figure
fig, ax = plt.subplots()

def plot_image_and_annotations(img, objects, polygons, mask, index):
    """Plots the image and lane markings using the extracted annotation data."""
    global ax
    ax.clear()

    # Show the image
    ax.imshow(img)
    if plot_new_masks.get():
        ax.imshow(mask, cmap="Reds", alpha=0.5)

    if plot_original_annotations.get():
        # Store handles for legend
        legend_handles = []

        # Extract lane markings and plot them
        for obj in objects:
            points = [(p[0], p[1], p[2]) for p in obj["poly2d"]]
            xs, ys = construct_line(points)
            color = lane_colors.get(obj["category"], "red")
            line, = ax.plot(xs, ys, '--', label=obj["category"], color=color, lw=1)

            # Only add to legend if not already included
            if obj["category"] not in [handle.get_label() for handle in legend_handles]:
                legend_handles.append(line)

        # # Plot polygons
        # for polygon, lane_type in polygons.items():
        #     x, y = polygon.exterior.xy
        #     color = lane_colors.get(lane_type, "red")
        #     ax.fill(x, y, color=color, alpha=0.5)

        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, title="Lane Markings")

    ax.set_title(f"Image {index+1}/{len(image_files)}: {image_files[index]}")
    ax.set_xticks([])
    ax.set_yticks([])

    canvas.draw()

def load_image_and_annotations(index):
    """Loads the image and annotations, then plots them."""
    img, objects, polygons, mask = extract_annotations(index)
    plot_image_and_annotations(img, objects, polygons, mask, index)

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

# Create a Tkinter frame for the buttons
frame_buttons = tk.Frame(root, height=50)
frame_buttons.pack(side=tk.BOTTOM, fill=tk.X)

# Create a Tkinter frame for the plot
frame_plot = tk.Frame(root)
frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


# Embed Matplotlib figure into Tkinter
canvas = FigureCanvasTkAgg(fig, master=frame_plot)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Spacer on the left
left_spacer = tk.Frame(frame_buttons)
left_spacer.pack(side=tk.LEFT, expand=True, fill=tk.X)

# Buttons centered together
prev_button = tk.Button(frame_buttons, text="Previous Photo", command=prev_image)
prev_button.pack(side=tk.LEFT, padx=5, pady=5)

next_button = tk.Button(frame_buttons, text="Next Photo", command=next_image)
next_button.pack(side=tk.LEFT, padx=5, pady=5)

plot_original_annotations = tk.IntVar()
show_annotations_chkbx = tk.Checkbutton(
    frame_buttons,
    text="Show original annotations",
    variable=plot_original_annotations,
    command=lambda: load_image_and_annotations(current_index)
)
show_annotations_chkbx.pack(side=tk.LEFT, padx=5, pady=5)

plot_new_masks = tk.IntVar(value=1)
show_masks_chkbx = tk.Checkbutton(
    frame_buttons,
    text="Show generated masks",
    variable=plot_new_masks,
    command=lambda: load_image_and_annotations(current_index)
)
show_masks_chkbx.pack(side=tk.LEFT, padx=5, pady=5)

# Spacer on the right
right_spacer = tk.Frame(frame_buttons)
right_spacer.pack(side=tk.LEFT, expand=True, fill=tk.X)

# Load the first image
load_image_and_annotations(current_index)

# Start the Tkinter main loop
root.mainloop()