import cv2
import matplotlib.pyplot as plt
import json
import os
from pycocotools.coco import COCO
import random

# Path to the folder 
cropped_images_folder = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\Output\Images"
coco_annotation_path = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\Output\updated_result.json"
visualized_folder = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\Output\UpdatedVisualized"

# Ensure the Visualized folder exists
if not os.path.exists(visualized_folder):
    os.makedirs(visualized_folder)

# Load the COCO annotations
with open(coco_annotation_path, 'r') as file:
    coco_data = json.load(file)

# Initialize COCO object to handle the annotations
coco = COCO(coco_annotation_path)

# Get the class names from the annotation file
categories = coco.loadCats(coco.getCatIds())
category_names = {category['id']: category['name'] for category in categories}

# Generate random colors for each class and normalize to [0, 1]
colors = {category_id: [random.randint(0, 255)/255.0, random.randint(0, 255)/255.0, random.randint(0, 255)/255.0] for category_id in category_names.keys()}

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

        # Create a list for legend entries (class names and colors)
        legend_entries = []

        # Draw the segmentation polygons on the image and include class labels
        for ann in annotations:
            segmentation = ann['segmentation'][0]  # Assuming each annotation has a segmentation polygon
            category_id = ann['category_id']
            class_name = category_names[category_id]

            # Segmentation is a flat list, so we need to split it into x and y coordinates
            x_coords = segmentation[0::2]  # Even indices (x coordinates)
            y_coords = segmentation[1::2]  # Odd indices (y coordinates)

            # Close the polygon by connecting the last point with the first
            x_coords.append(x_coords[0])
            y_coords.append(y_coords[0])

            # Plot the segmentation polygon over the image with a specific color
            plt.plot(x_coords, y_coords, marker='o', color=colors[category_id], linewidth=2)  # Colored polygon for segmentation
            
            # Annotate with class label at the centroid of the polygon
            centroid_x = sum(x_coords) / len(x_coords)
            centroid_y = sum(y_coords) / len(y_coords)
            plt.text(centroid_x, centroid_y, class_name, color=colors[category_id], fontsize=10, ha='center', va='center')

            # Add the class to the legend if it's not already added
            if class_name not in legend_entries:
                legend_entries.append((class_name, colors[category_id]))

        # Create a legend for the classes
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for _, color in legend_entries]
        labels = [label for label, _ in legend_entries]
        plt.legend(handles=handles, labels=labels, loc='upper right', title="Classes")

        # Save the visualized image
        visualized_image_path = os.path.join(visualized_folder, img_filename)
        plt.savefig(visualized_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to avoid memory overload

        print(f"Processed and saved visualized image: {img_filename}")