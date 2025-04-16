import os
import time
import json
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# Helper function to load the hyperparameters from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def visualize_learning_curve(history, save_path, timestamp=""):

    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training loss")

    fn = os.path.join(save_path, f"learning_curve_{timestamp}.png")
    plt.savefig(fn, bbox_inches='tight')
    plt.close()
    print(f"   Learning curve saved to {fn}")

    # save history for future use:
    fn = os.path.join(save_path, f"learning_history_{timestamp}.json")
    with open(fn, "w") as file:
        json.dump(history, file, indent=4)
    print(f"   Learning history saved to {fn}")


def visualize_confusion_matrix(train_cm, val_cm, label_names=None, save_path=None):

    # Create subplots with 1 row and 2 columns
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Adjust fig-size as needed

    # Plot the train confusion matrix
    train_cm = np.round(train_cm, 2)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=train_cm, display_labels=label_names)
    disp_train.plot(ax=ax[0], cmap='PuBu')
    ax[0].set_title("Train Confusion Matrix")

    # Plot the validation confusion matrix
    val_cm = np.round(val_cm, 2)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=label_names)
    disp_val.plot(ax=ax[1], cmap='PuBu')  # Plot on the second subplot
    ax[1].set_title("Validation Confusion Matrix")

    # save history for future use:
    if save_path is not None:
        current_time = time.localtime()  # Get current time
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', current_time)  # Format as 'YYYY-MM-DD_HH-MM-SS'
        fn = os.path.join(save_path, f"confusion_matrices_{timestamp}.json")
        plt.savefig(fn, dpi=300, transparent=True)
        plt.close()
        print(f"   Confusion matrix saved at {fn}")


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


def box_coco_to_corner(bbox):
    """Convert from (upper-left, width, height) to (upper-left, bottom-right)"""
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    boxes = (x1, y1, x2, y2)
    return boxes


def add_bbox(img, bbox, label=None, bbox_color=(255, 255, 255), bbox_thickness=2, text_color=(0, 0, 0), font_scale=1):
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    # For bounding box
    x1, y1, x2, y2 = bbox
    img = cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=bbox_color, thickness=bbox_thickness)

    if label is not None:
        img = cv2.putText(img=img, text=label, org=(x1, y1 - 5), fontFace=_FONT, fontScale=font_scale,
                          color=text_color, thickness=2, lineType=cv2.LINE_AA)

    return img