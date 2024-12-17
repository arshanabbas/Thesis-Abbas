
image_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas/CPDataset/images/0a05af59-10-19.3.jpg"  
mask_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Mask/0a05af59-10-19_mask.npy" 
def visualize_mask_with_transparency(image_path, mask_path, class_colors, alpha=0.5):
    """
    Visualizes a multi-channel mask as a transparent overlay on the original image.

    Args:
        image_path (str): Path to the original image file.
        mask_path (str): Path to the .npy mask file.
        class_colors (dict): Dictionary of class IDs to RGB tuples.
        alpha (float): Transparency level for overlays (0.0 to 1.0).
    """
    # Load the image and mask
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.load(mask_path)

    # Convert image to float for transparency blending
    image = image.astype(np.float32) / 255.0

    # Initialize the final overlay
    overlay = np.zeros_like(image, dtype=np.float32)

    # Apply each class mask with transparency
    for class_id, color in class_colors.items():
        if class_id < mask.shape[0]:  # Ensure the channel exists
            class_mask = mask[class_id]
            color_normalized = np.array(color) / 255.0
            class_overlay = np.zeros_like(image, dtype=np.float32)
            for i in range(3):  # Apply color to R, G, B channels
                class_overlay[:, :, i] = class_mask * color_normalized[i]
            overlay = overlay + alpha * class_overlay

    # Blend the overlay with the original image
    blended = (1 - alpha) * image + overlay
    blended = (blended * 255).astype(np.uint8)

    # Display the blended image
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Overlay of Multi-Channel Mask with Transparency")
    plt.show()

# Example usage
visualize_mask_with_transparency(image_path, mask_path, class_colors, alpha=0.5)