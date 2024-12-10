def visualize_mask_with_image(image_path, mask_path, class_colors):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2

    # Load the image and mask
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.load(mask_path)

    # Initialize an overlay image
    overlay = np.zeros_like(image, dtype=np.uint8)

    # Apply each class mask with a unique color
    for class_id, color in class_colors.items():
        if class_id < mask.shape[0]:  # Ensure the channel exists
            class_mask = mask[class_id]
            # Apply the color to the overlay
            overlay[class_mask > 0] = color

    # Blend the original image with the overlay
    blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    cv2.imwrite("F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Mask/overlay_output.jpg", blended)

    # Display the blended image
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Overlay of Multi-Channel Mask on Image")
    plt.show()

# Example usage
image_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas/CPDataset/images/0a05af59-10-19.3.jpg"  
mask_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Mask/0a05af59-10-19_mask.npy" 

class_colors = {
    0: (255, 0, 0),    # R 
    1: (0, 255, 0),    # G
    2: (0, 0, 255),    # B
    3: (255, 255, 0),  # Y
}

visualize_mask_with_image(image_path, mask_path, class_colors)
