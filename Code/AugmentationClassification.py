import os
import cv2
import glob
import albumentations as A
import shutil  # To copy the original files
import numpy as np

# Define augmentation pipeline with high-quality transformations
transform = A.Compose([
    # Augmentation Techniques
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.HueSaturationValue(p=0.3),
    
    # Image Quality Enhancements
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),  # Contrast Limited Adaptive Histogram Equalization
    A.RandomGamma(gamma_limit=(80, 120), p=0.1),  # Random Gamma for contrast enhancement
    A.Sharpen(p=0.1),  # Sharpening the image for better details
    A.Equalize(p=0.1)  # General histogram equalization
])

# Path to the main folder containing the 19 classes
main_folder = "D:/Abbas/ubergabe/Data/val"
output_folder = "D:/Abbas/ubergabe/Playground/val"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through each class (subfolder) in the main folder
for class_folder in os.listdir(main_folder):
    class_path = os.path.join(main_folder, class_folder)
    
    # Skip non-directory files (if any)
    if not os.path.isdir(class_path):
        continue
    
    # Create an output folder for the class in the augmented data directory
    output_class_folder = os.path.join(output_folder, class_folder)
    os.makedirs(output_class_folder, exist_ok=True)
    
    # Get all image files in the class folder
    image_files = glob.glob(os.path.join(class_path, "*.jpg"))
    
    # Loop through each image in the class folder
    for img_file in image_files:
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Apply augmentation with enhancement
        augmented = transform(image=img)
        augmented_img = augmented["image"]
        
        # Save the augmented image
        aug_img_filename = os.path.join(output_class_folder, "aug_" + os.path.basename(img_file))
        cv2.imwrite(aug_img_filename, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
        
        # Also copy the original image with a modified name to avoid conflict
        original_img_filename = os.path.join(output_class_folder, os.path.basename(img_file).replace(".jpg", "_original.jpg"))
        shutil.copy(img_file, original_img_filename)
    
    print(f"Augmentation complete for class: {class_folder}")

print("All augmentations complete! Augmented images and original files saved.")
