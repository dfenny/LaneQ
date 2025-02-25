import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '/Users/douglasfenwick/Downloads/bdd100k_images/train/00232de3-19eca24a.jpg'  # Change this to your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Resize the image for faster processing (optional)
# image_small = cv2.resize(image, (300, 300))
image_small = image

# Reshape image to a 2D array of pixels
pixels = image_small.reshape(-1, 3)

# Apply K-Means Clustering
num_colors = 8  # Adjust for more/less detail
kmeans = cv2.kmeans(
    np.float32(pixels), num_colors, None, 
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
    10, cv2.KMEANS_RANDOM_CENTERS
)

# Assign each pixel to the closest cluster
centers = np.uint8(kmeans[2])  # Cluster centers (RGB values)
segmented_img = centers[kmeans[1].flatten()]  # Assign colors

# Reshape back to image shape
segmented_img = segmented_img.reshape(image_small.shape)

# Display the output
plt.figure(figsize=(8, 8))
plt.imshow(segmented_img)
plt.axis("off")
plt.title("Color by Number Effect")
plt.show()