import cv2
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO

# Path to the cropped image 
cropped_image_path = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\COCO1\images_cropped\your_image_name.jpg"
coco_annotation_path = r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\COCO1\Test\new_result.json"

# Load
image = cv2.imread(cropped_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for displaying with Matplotlib

# Load the COCO annotations
with open(coco_annotation_path, 'r') as file:
    coco_data = json.load(file)

# Initialize COCO object to handle the annotations
coco = COCO(coco_annotation_path)

# Get the image id from the cropped image path (Assuming your image name is consistent with COCO format)
image_id = None
for img_info in coco.imgs.values():
    if img_info['file_name'] == cropped_image_path.split("\\")[-1]:
        image_id = img_info['id']
        break

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

plt.show()
