import os
import cv2
import glob
import albumentations as A
from shutil import copyfile

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.HueSaturationValue(p=0.3)
])

# Path
main_folder = "R:/Arshan Abbas/Übergabe/Playground/Input"
output_folder = "R:/Arshan Abbas/Übergabe/Playground/Ouput"

# Loop through each class in the main folder
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
        
        # Apply augmentation
        augmented = transform(image=img)
        augmented_img = augmented["image"]
        
        # Save the augmented image
        aug_img_filename = os.path.join(output_class_folder, "aug_" + os.path.basename(img_file))
        cv2.imwrite(aug_img_filename, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
    
    print(f"Augmentation complete for class: {class_folder}")

print("All augmentations complete! Augmented images saved.")