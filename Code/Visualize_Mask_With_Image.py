import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def visualize_mask_with_image(image_path, mask_path, class_colors):
    """
    Visualizes a multi-channel mask as an overlay on the original image.

    Args:
        image_path (str): Path to the original image file.
        mask_path (str): Path to the .npy mask file.
        class_colors (dict): Dictionary of class IDs to RGB tuples.
                             Example: {0: (255, 0, 0), 1: (0, 255, 0), ...}
    """
    # Load the image and mask
    image = np.array(Image.open(image_path))
    mask = np.load(mask_path)

    # Initialize an overlay image
    overlay = np.zeros_like(image, dtype=np.uint8)

    # Apply each class mask with a unique color
    for class_id, color in class_colors.items():
        if class_id < mask.shape[0]:  # Ensure the class_id exists in the mask
            class_mask = mask[class_id]
            overlay[class_mask > 0] = color

    # Blend the image and overlay
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # Plot the result
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Overlay of Multi-Channel Mask on Image")
    plt.show()

# Example usage
image_path = "/path/to/original/image.jpg"  # Replace with path to the original image
mask_path = "/path/to/generated_mask.npy"  # Replace with path to the .npy mask file
class_colors = {
    0: (255, 0, 0),  # Red for class 0
    1: (0, 255, 0),  # Green for class 1
    2: (0, 0, 255),  # Blue for class 2
    3: (255, 255, 0),  # Yellow for class 3
}

visualize_mask_with_image(image_path, mask_path, class_colors)
