import cv2
import matplotlib.pyplot as plt
import json
import os
from pycocotools.coco import COCO

# Path to the folder containing cropped images and the updated COCO annotation JSON
cropped_images_folder = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\COCO1\images_cropped"
coco_annotation_path = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\COCO1\Test\new_result.json"
visualized_folder = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\COCO1\Visualized"

# Ensure the Visualized folder exists
if not os.path.exists(visualized_folder):
    os.makedirs(visualized_folder)

# Load the COCO annotations
with open(coco_annotation_path, 'r') as file:
    coco_data = json.load(file)

# Initialize COCO object to handle the annotations
coco = COCO(coco_annotation_path)

# Iterate through each image in the cropped images folder
for img_filename in os.listdir(cropped_images_folder):
    if img_filename.endswith('.jpg'):  # Only process jpg images
        # Construct the full image path
        img_path = os.path.join(cropped_images_folder, img_filename)

        # Load the image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for displaying with Matplotlib

        # Get the image ID from the cropped image path
        image_id = None
        for img_info in coco.imgs.values():
            if img_info['file_name'] == img_filename:
                image_id = img_info['id']
                break

        if image_id is None:
            continue  # Skip images that are not found in the annotation file

        # Get the annotations (if any) associated with this image
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))

        # Plotting the image and its segmentation
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')  # Hide the axes

        # Draw the segmentation polygons on the image
        for ann in annotations:
            segmentation = ann['segmentation'][0]  # Assuming each annotation has a segmentation polygon

            # Segmentation is a flat list, so we need to split it into x and y coordinates
            x_coords = segmentation[0::2]  # Even indices (x coordinates)
            y_coords = segmentation[1::2]  # Odd indices (y coordinates)

            # Close the polygon by connecting the last point with the first
            x_coords.append(x_coords[0])
            y_coords.append(y_coords[0])

            # Plot the segmentation polygon over the image
            plt.plot(x_coords, y_coords, marker='o', color='r', linewidth=2)  # Red polygon for segmentation

        # Save the visualized image
        visualized_image_path = os.path.join(visualized_folder, img_filename)
        plt.savefig(visualized_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to avoid memory overload

        print(f"Processed and saved visualized image: {img_filename}")
