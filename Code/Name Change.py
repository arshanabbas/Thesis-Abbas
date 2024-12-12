import os

# Directory paths
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/images"
mask_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Mask"
output_image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Output/Images"
output_mask_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Output/Mask"

# File extension for images and masks
image_ext = ".jpg"
mask_ext = ".npy"

# Ensure output directories exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Function to rename and move files
def rename_and_move_files(image_dir, mask_dir, output_image_dir, output_mask_dir, image_ext, mask_ext):
    # List all images and masks
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(image_ext)])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(mask_ext)])

    if len(image_files) != len(mask_files):
        print("Error: The number of images and masks do not match.")
        return

    for i, (image, mask) in enumerate(zip(image_files, mask_files)):
        # New names
        new_image_name = f"image_{i+1:03d}{image_ext}"
        new_mask_name = f"mask_{i+1:03d}{mask_ext}"

        # Current file paths
        image_path = os.path.join(image_dir, image)
        mask_path = os.path.join(mask_dir, mask)

        # New file paths in output directories
        new_image_path = os.path.join(output_image_dir, new_image_name)
        new_mask_path = os.path.join(output_mask_dir, new_mask_name)

        # Copy and rename files
        os.rename(image_path, new_image_path)
        os.rename(mask_path, new_mask_path)

rename_and_move_files(image_dir, mask_dir, output_image_dir, output_mask_dir, image_ext, mask_ext)