import os
import cv2
import glob
import albumentations as A
import numpy as np
from ultralytics import YOLO

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.HueSaturationValue(p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Paths to images and labels
input_images_path = "./images"
input_labels_path = "./labels"
output_images_path = "./augmented_images"
output_labels_path = "./augmented_labels"

os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)

# Process each image
image_files = glob.glob(os.path.join(input_images_path, "*.jpg"))

for img_file in image_files:
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    
    # Load corresponding YOLO label
    label_file = os.path.join(input_labels_path, os.path.basename(img_file).replace(".jpg", ".txt"))
    if not os.path.exists(label_file):
        continue
    
    with open(label_file, "r") as f:
        lines = f.readlines()
    
    bboxes = []
    class_labels = []
    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        x_center, y_center, w, h = map(float, values[1:])
        bboxes.append([x_center, y_center, w, h])
        class_labels.append(class_id)
    
    # Apply augmentation
    augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
    augmented_img = augmented["image"]
    augmented_bboxes = augmented["bboxes"]
    
    # Save augmented image
    aug_img_filename = os.path.join(output_images_path, "aug_" + os.path.basename(img_file))
    cv2.imwrite(aug_img_filename, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
    
    # Save augmented label
    aug_label_filename = os.path.join(output_labels_path, "aug_" + os.path.basename(label_file))
    with open(aug_label_filename, "w") as f:
        for bbox, cls in zip(augmented_bboxes, class_labels):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")
    
print("Augmentation complete! Augmented images and labels saved.")