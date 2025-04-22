import os

import cv2
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch

from .model_suite.inference import DegradationDetector

output_dir = "laneq_output"
dt = DegradationDetector(output_dir=output_dir)

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)

# Index tracker
current_index = 0
image_files = []
input_dir = None

# Tkinter window
root = tk.Tk()
root.title("LaneQ")

# Matplotlib figure
fig, ax = plt.subplots()

def select_dir():
    global input_dir
    input_dir = tk.filedialog.askdirectory()
    if input_dir:
        # Load images from the selected directory
        global image_files
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg'))]
        load_image_and_annotations(index=current_index, no_msk=True, no_cls=True)

def plot_image_and_annotations(img, index):
    """Plots the image and lane markings using the extracted annotation data."""
    global ax
    ax.clear()

    # Show the image
    ax.imshow(img)
    
    if show_seg_model_res.get():
        # Predict segmentation mask
        pred_mask = cv2.imread(os.path.join(output_dir, image_files[index].replace(".jpg", ""), f"{image_files[index].split('.')[0]}_pred_mask.png"))
        ax.imshow(pred_mask, cmap="Blues", alpha=0.5)
    
    if show_class_model_res.get():
        # Predict classification mask
        pred_mask = cv2.imread(os.path.join(output_dir, image_files[index].replace(".jpg", ""), f"{image_files[index].split('.')[0]}_annotated.jpg"))
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2RGB)
        ax.imshow(pred_mask)

    ax.set_title(f"Image {index+1}/{len(image_files)}: {image_files[index]}")
    ax.set_xticks([])
    ax.set_yticks([])

    canvas.draw()

def load_image_and_annotations(index, no_msk=False, no_cls=False):
    """Loads the image and annotations, then plots them."""
    if no_msk:
        show_seg_mod_chkbx.deselect()
    if no_cls:
        show_class_mod_chkbx.deselect()
    if image_files:
        img = cv2.imread(os.path.join(input_dir, image_files[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not os.path.exists(os.path.join(output_dir, image_files[index].replace(".jpg", ""), f"{image_files[index].split('.')[0]}_pred_mask.png")):
            dt.predict(os.path.join(input_dir, image_files[index]), device=DEVICE)
        plot_image_and_annotations(img, index)
    else:
        ax.clear()
        ax.set_title("No images loaded.")
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

select_dir_button = tk.Button(frame_buttons, text="Select Directory", command=select_dir)
select_dir_button.pack(side=tk.LEFT, padx=5, pady=5)

# Bind left and right arrow keys to navigate photos
root.bind('<Left>', lambda event: prev_image())
root.bind('<Right>', lambda event: next_image())


show_seg_model_res = tk.IntVar(value=0)
show_seg_mod_chkbx = tk.Checkbutton(
    frame_buttons,
    text="Show segmentation model results",
    variable=show_seg_model_res,
    command=lambda: load_image_and_annotations(current_index, no_cls=True)
)
show_seg_mod_chkbx.pack(side=tk.LEFT, padx=5, pady=5)

show_class_model_res = tk.IntVar(value=1)
show_class_mod_chkbx = tk.Checkbutton(
    frame_buttons,
    text="Show classificaiton model results",
    variable=show_class_model_res,
    command=lambda: load_image_and_annotations(current_index, no_msk=True)
)
show_class_mod_chkbx.pack(side=tk.LEFT, padx=5, pady=5)

# Spacer on the right
right_spacer = tk.Frame(frame_buttons)
right_spacer.pack(side=tk.LEFT, expand=True, fill=tk.X)

# Load the first image
load_image_and_annotations(current_index)

def run_gui():
    # Start the Tkinter main loop
    root.mainloop()