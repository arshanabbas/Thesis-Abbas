import cv2
import json
import os

# Load file
with open('F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/COCO1/Test/result.json') as f:
    annotations = json.load(f)

# Define the paths 
input_dir = (r"F:\\Pomodoro\\Work\\TIME\\Script\\Thesis-Abbas-Segmentation\\COCO1\\images")
output_dir = (r"F:\\Pomodoro\\Work\\TIME\\Script\\Thesis-Abbas-Segmentation\\COCO1\\Cropped")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a dictionary 
annotations_by_image_id = {ann['image_id']: ann for ann in annotations['annotations']}

# Iterate through image
for img_filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_filename)
    img = cv2.imread(img_path)
    
    # Skip non-image files
    if img is None:
        continue

    height, width = img.shape[:2]
    h = width  # height of the cropped region
    y_offset = int((height - h) / 2)  # Calculate the starting Y-coordinate for cropping

    # Crop the image
    cropped_img = img[y_offset:y_offset + h, :]
    cropped_img_path = os.path.join(output_dir, img_filename)
    cv2.imwrite(cropped_img_path, cropped_img)

    # Adjust annotations
    image_id = img_filename.split('.')[0]  # Assuming image ID matches filename
    if image_id in annotations_by_image_id:
        annotation = annotations_by_image_id[image_id]
        
        # Adjust the y-coordinate of the mask points
        new_segmentation = []
        for point in annotation['segmentation'][0]:
            x, y_coord = point[0], point[1] - y_offset
            new_segmentation.append([x, y_coord])
