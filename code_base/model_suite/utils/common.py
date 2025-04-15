import os
import time
import json
import yaml
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
