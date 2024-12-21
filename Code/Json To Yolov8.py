import json
import os
from pathlib import Path

# Function to normalize polygon coordinates
def normalize_polygon(polygon, image_width, image_height):
    normalized_polygon = []
    for i in range(0, len(polygon), 2):
        x_norm = polygon[i] / image_width
        y_norm = polygon[i + 1] / image_height
        normalized_polygon.extend([x_norm, y_norm])
    return normalized_polygon

# Main conversion function
def convert_annotations_to_yolov8_segmentation(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for image_info in data['images']:
        image_id = image_info['id']
        image_width = image_info['width']
        image_height = image_info['height']
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

        output_file = Path(output_dir) / f"{Path(image_info['file_name']).stem}.txt"

        with open(output_file, 'w') as out_f:
            for ann in annotations:
                category_id = ann['category_id'] - 1  # YOLO class IDs start from 0
                segmentation = ann['segmentation'][0]  # Assuming one segmentation per annotation
                normalized_segmentation = normalize_polygon(segmentation, image_width, image_height)

                out_f.write(f"{category_id} {' '.join(map(str, normalized_segmentation))}\n")

    print(f"Segmentation annotations converted and saved to {output_dir}")

# Example usage
json_file = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/new_result.json"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/YOLOv8"
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/images"  # Optional, in case you need image dimensions from files

convert_annotations_to_yolov8_segmentation(json_file, output_dir)