import os
import cv2
import json
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pathlib import Path
from tqdm import tqdm

# original dataset
original_images_folder = r"F:\Arshan_Abbas\Fabian\Task3\Code\Thesis-Abbas\CPDataset\images"
coco_annotation_path = r"F:\Arshan_Abbas\Fabian\Task3\Code\Thesis-Abbas\CPDataset\new_result.json"

# To save augmented images and masks
augmented_images_folder = r"F:\Arshan_Abbas\Fabian\Task3\Code\NoSync\Augmented\images"
augmented_masks_folder = r"F:\Arshan_Abbas\Fabian\Task3\Code\NoSync\Augmented\masks"

# Ensuring folder 
Path(augmented_images_folder).mkdir(parents=True, exist_ok=True)
Path(augmented_masks_folder).mkdir(parents=True, exist_ok=True)

# Load the COCO annotations
with open(coco_annotation_path, 'r') as file:
    coco_data = json.load(file)

# Initialize COCO object
coco = COCO(coco_annotation_path)

# Create a list
image_ids = coco.getImgIds()

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Horizontal Flip
    A.VerticalFlip(p=0.5),    # Vertical Flip
    A.RandomRotate90(p=0.5),  # Rotate image and mask 90 degrees
    A.RandomBrightnessContrast(p=0.2),  # Random brightness and contrast adjustment
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),  # Random color change
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Gaussian Blur
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),  # Elastic transform for more flexibility
    A.PadIfNeeded(min_height=500, min_width=500, p=1),  # Ensure a minimum size for the image
    ToTensorV2(),  # Convert the images and masks to PyTorch tensors
], additional_targets={'mask': 'mask'})  # Ensures transformations applied to the mask

# Function to generate the mask from COCO annotations
def create_mask_from_annotations(image_info, annotations):
    mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
    
    for ann in annotations:
        segmentation = ann['segmentation'][0]
        points = np.array(segmentation).reshape((-1, 2))
        points = points.astype(np.int32)  # Ensure the points are of the correct type
        cv2.fillPoly(mask, [points], color=255)
    
    return mask

# Function to apply the augmentations
def augment_and_save(image_path, image, mask, augmented_image_folder, augmented_mask_folder, mask_filename):
    # Apply the augmentation transformations
    augmented = transform(image=image, mask=mask)
    augmented_image = augmented['image']
    augmented_mask = augmented['mask']

    # Convert the augmented image and mask from tensor to numpy arrays
    augmented_image = augmented_image.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
    augmented_mask = augmented_mask.cpu().numpy()  # Directly convert to numpy for 2D mask

    # Generate the new filenames using the augmented image folder
    new_image_filename = os.path.join(augmented_image_folder, os.path.basename(image_path))
    new_mask_filename = os.path.join(augmented_masks_folder, mask_filename)

    # Save the augmented image and mask
    cv2.imwrite(new_image_filename, augmented_image)
    cv2.imwrite(new_mask_filename, augmented_mask)

# Target size
target_image_count = 2000
current_image_count = len(os.listdir(original_images_folder))

# Iterate through images and augment them
augmented_count = 0
for img_id in tqdm(image_ids, desc="Augmenting images"):
    img_info = coco.loadImgs([img_id])[0]
    img_filename = img_info['file_name']

    # Get the corresponding mask file
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    annotations = coco.loadAnns(ann_ids)

    # Generate mask image from COCO annotations
    mask = create_mask_from_annotations(img_info, annotations)

    # Load the image
    image_path = os.path.join(original_images_folder, img_filename)
    image = cv2.imread(image_path)

    # Match the image name with extension
    mask_filename = img_filename.replace('.jpg', '.png')  

    # Augment and save images until we reach the target size
    augment_and_save(image_path, image, mask, augmented_images_folder, augmented_masks_folder, mask_filename)
    augmented_count += 1

    # Stop when target count is reached
    if augmented_count >= (target_image_count - current_image_count):
        break

print(f"Augmentation completed! Total images augmented to: {augmented_count + current_image_count}")