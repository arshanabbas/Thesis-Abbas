import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Define color for Class 2 (Blue)
CLASS_2_COLOR = (0, 0, 255)  # Blue for Class 2
CIRCLE_RADIUS = 5  # Radius of the circles
CIRCLE_COLOR = (0, 0, 255)  # Blue border for circles
CIRCLE_THICKNESS = 1  # Thin outline

# Function to check if a point is inside a polygon
def is_point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Visualization
def visualize_class2_segmentation(image_dir, annotation_dir, output_dir=None):
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    for annotation_file in os.listdir(annotation_dir):
        if not annotation_file.endswith(".txt"):
            continue

        image_name = os.path.splitext(annotation_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        annotation_path = os.path.join(annotation_dir, annotation_file)

        if not os.path.exists(image_path):
            print(f"Image {image_name} not found. Skipping...")
            continue

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse annotation file
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                polygon = list(map(float, parts[1:]))

                # Only process Class 2 (Nebenbereich)
                if class_id != 2:
                    continue

                # Normalize and map coordinates to image dimensions
                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0]))
                          for i in range(0, len(polygon), 2)]
                points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

                # Draw polygon outline instead of filling
                cv2.polylines(image, [points], isClosed=True, color=CLASS_2_COLOR, thickness=1)

                # Draw class ID at the first point
                cv2.putText(image, "2", tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Generate random empty circles inside the polygon
                x_min, y_min = np.min(points, axis=0)[0]
                x_max, y_max = np.max(points, axis=0)[0]

                num_circles = 10  # Adjust number of circles as needed
                for _ in range(num_circles):
                    for _ in range(50):  # Max attempts to find a valid point
                        rand_x = random.randint(x_min, x_max)
                        rand_y = random.randint(y_min, y_max)

                        if is_point_inside_polygon((rand_x, rand_y), points):
                            cv2.circle(image, (rand_x, rand_y), CIRCLE_RADIUS, CIRCLE_COLOR, thickness=CIRCLE_THICKNESS)
                            break  # Move to next circle after placing one

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')

        # Remove white border when saving
        if output_dir:
            output_path = os.path.join(output_dir, image_name)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

# Example usage
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/visualisation" 

visualize_class2_segmentation(image_dir, annotation_dir, output_dir)
