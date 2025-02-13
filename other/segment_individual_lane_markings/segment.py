import json
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load JSON file
json_file = os.path.expanduser(
    "~/Downloads/bdd100k_annotations/train/0a0a0b1a-7c39d841.json"
)
with open(json_file, "r") as f:
    data = json.load(f)

# Load an image (optional) or create a blank canvas
image_path = os.path.expanduser(
    "~/Downloads/bdd100k_images/train/0a0a0b1a-7c39d841.jpg"
)
image_path = os.path.expanduser(image_path)
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

# Extract poly2d data
for frame in data["frames"]:
    for obj in frame["objects"]:
        if "lane" in obj["category"]:
        # if "poly2d" in obj:
            points = np.array([(p[0], p[1]) for p in obj["poly2d"]], dtype=np.int32)

            # Shade the polygon area
            ax.fill(points[:, 0], points[:, 1], alpha=0.4, label=obj["category"])

            # Draw polygon edges
            ax.plot(points[:, 0], points[:, 1], linestyle='-', linewidth=2, label=obj["category"])

# Show the image with annotations
ax.set_title("Annotated Scene with poly2d Areas")
ax.set_xticks([])
ax.set_yticks([])
# plt.legend()
plt.show()