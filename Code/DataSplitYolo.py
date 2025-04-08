import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths
image_folder = 'D:/Abbas/GitHub/PolygontoYOLO/crack_dataset/images'        # Path to your 'images' folder
annotation_folder = 'D:/Abbas/GitHub/PolygontoYOLO/crack_dataset/labels'  # Path to your 'annotations' folder

# Define structure
dataset_folder = 'D:/Abbas/GitHub/PolygontoYOLO/dataset_crackyolo'
train_image_folder = os.path.join(dataset_folder, 'images', 'train')
val_image_folder = os.path.join(dataset_folder, 'images', 'val')
train_label_folder = os.path.join(dataset_folder, 'labels', 'train')
val_label_folder = os.path.join(dataset_folder, 'labels', 'val')

# Create the necessary directories
os.makedirs(train_image_folder, exist_ok=True)
os.makedirs(val_image_folder, exist_ok=True)
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

# List all image files in the image folder
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
annotation_files = [f for f in os.listdir(annotation_folder) if os.path.isfile(os.path.join(annotation_folder, f))]

# Make sure every image file has a corresponding label file
image_files = [f for f in image_files if f.replace('.jpg', '.txt') in annotation_files or f.replace('.png', '.txt') in annotation_files]

# Split the data (80% train, 20% val)
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# Function to copy files
def copy_files(files, src_folder, dst_folder):
    for file in files:
        shutil.copy(os.path.join(src_folder, file), dst_folder)

# Copy the images to the train and val folders
copy_files(train_images, image_folder, train_image_folder)
copy_files(val_images, image_folder, val_image_folder)

# Copy the labels to the train and val folders
for image in train_images:
    label_file = image.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.copy(os.path.join(annotation_folder, label_file), train_label_folder)

for image in val_images:
    label_file = image.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.copy(os.path.join(annotation_folder, label_file), val_label_folder)

print("Dataset has been successfully organized into train and val folders.")