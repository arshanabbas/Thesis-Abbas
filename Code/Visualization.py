import cv2
import matplotlib.pyplot as plt
import json
import os
from pycocotools.coco import COCO
import random

# Paths
cropped_images_folder = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\CPDataset\images"
coco_annotation_path = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\CPDataset\new_result.json"
visualized_folder = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\Output\Vold"

# Ensure the Visualized folder exists
os.makedirs(visualized_folder, exist_ok=True)

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
        img_path = os.path.join(cropped_images_folder, img_filename)

        # Load the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_filename}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for displaying with Matplotlib

        # Get the image ID from the cropped image path
        image_id = next((img_info['id'] for img_info in coco.imgs.values() if img_info['file_name'] == img_filename), None)
        if image_id is None:
            print(f"Warning: No matching image ID found for {img_filename}")
            continue

        # Get the annotations associated with this image
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        if not annotations:
            print(f"Warning: No annotations found for {img_filename}")
            continue

        # Plotting the image and its segmentation
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')  # Hide the axes

        # Create a list for legend entries (class names and colors)
        legend_entries = []

        for ann in annotations:
            segmentation = ann.get('segmentation')
            if not segmentation or len(segmentation) == 0:
                print(f"Warning: No segmentation data for annotation {ann['id']} in {img_filename}")
                continue

            # Assume the first polygon in the segmentation
            polygon = segmentation[0]
            x_coords = polygon[0::2]  # Even indices (x coordinates)
            y_coords = polygon[1::2]  # Odd indices (y coordinates)

            # Plot the polygon and label
            category_id = ann['category_id']
            class_name = category_names[category_id]
            plt.fill(x_coords, y_coords, color=colors[category_id], alpha=0.4)  # Transparent fill
            plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], color=colors[category_id], linewidth=2)  # Boundary

            # Annotate with class label
            centroid_x = sum(x_coords) / len(x_coords)
            centroid_y = sum(y_coords) / len(y_coords)
            plt.text(centroid_x, centroid_y, class_name, color='white', fontsize=10, ha='center', va='center')

            # Add class name and color to the legend
            if class_name not in [entry[0] for entry in legend_entries]:
                legend_entries.append((class_name, colors[category_id]))

        # Add the legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for _, color in legend_entries]
        labels = [label for label, _ in legend_entries]
        plt.legend(handles=handles, labels=labels, loc='upper right', title="Classes")

        # Save the visualized image
        visualized_image_path = os.path.join(visualized_folder, img_filename)
        plt.savefig(visualized_image_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Save with high resolution
        plt.close()  # Close the plot to avoid memory overload

        print(f"Processed and saved visualized image: {img_filename}")