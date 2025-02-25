import cv2
import numpy as np
import matplotlib.pyplot as plt

def reduce_color_depth(image: np.ndarray, bit_depth: int = 4) -> np.ndarray:
    """
    Reduces the color depth of an image.
    
    Parameters:
        image (np.ndarray): Input image as a NumPy array.
        bit_depth (int): Target bit depth per channel (1 to 8). Default is 4-bit.
    
    Returns:
        np.ndarray: Image with reduced color depth.
    """
    if bit_depth < 1 or bit_depth > 8:
        raise ValueError("bit_depth must be between 1 and 8")

    # Calculate the divisor based on bit depth
    bit_divisor = 256 // (2 ** bit_depth)
    
    # Reduce color depth
    reduced_image = (image // bit_divisor) * bit_divisor
    
    return reduced_image

# Load image
image_path = "/Users/douglasfenwick/Downloads/bdd100k_images/train/00232de3-19eca24a.jpg"  # Replace with the path to your image
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

# Convert image to RGB for Matplotlib display
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reduce color depth
bit_depth = 3  # Adjust as needed (1-8)
reduced_img = reduce_color_depth(img, bit_depth=bit_depth)

# Display images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(reduced_img)
axes[1].set_title(f"Reduced to {bit_depth}-bit per channel")
axes[1].axis("off")

plt.show()