import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Define colors and pore parameters
CLASS_2_COLOR = (0, 0, 0)  # Blue for Class 2 outline
PORE_COLOR = (0, 0, 0)  # Blue outline for pores

CIRCLE_THICKNESS = 1  # Thin outline
MIN_PORE_RADIUS = 2  # Minimum pore size
MAX_PORE_RADIUS = 5  # Maximum pore size
PORE_DENSITY = 0.02  # Percentage of the polygon area covered by pores

# Function to check if a point is inside a polygon
def is_point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Function to calculate the area of a polygon
def polygon_area(polygon):
    return 0.5 * abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))

# Function to generate realistic, irregular pores
def generate_realistic_pores(polygon, num_pores, img_shape):
    pores = []
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]

    max_attempts = 50  # Attempts per pore to prevent overlap
    for _ in range(num_pores):
        attempts = 0
        while attempts < max_attempts:
            rand_x = random.randint(x_min, x_max)
            rand_y = random.randint(y_min, y_max)
            rand_w = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
            rand_h = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
            angle = random.randint(0, 180)  # Random rotation angle for natural look

            # Check if the center is inside the polygon
            if is_point_inside_polygon((rand_x, rand_y), polygon):
                # Ensure pores do not overlap too closely
                if all(np.linalg.norm(np.array([rand_x, rand_y]) - np.array(p[:2])) > (max(rand_w, rand_h) + p[2] + 3) for p in pores):
                    pores.append((rand_x, rand_y, rand_w, rand_h, angle))
                    break  # Move to next pore after a valid placement
            attempts += 1
    return pores

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

                # Only process Class 3 
                if class_id != 3:
                    continue

                # Normalize and map coordinates to image dimensions
                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0]))
                          for i in range(0, len(polygon), 2)]
                points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

                # Draw polygon outline instead of filling
                cv2.polylines(image, [points], isClosed=True, color=CLASS_2_COLOR, thickness=2)

                # Draw class ID at the first point
                cv2.putText(image, "2", tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Calculate polygon area
                area = polygon_area(points[:, 0, :])
                num_pores = max(3, int(area * PORE_DENSITY))  # Adjust based on polygon size

                # Generate realistic, irregular pores
                pores = generate_realistic_pores(points, num_pores, image.shape)

                # Draw the pores
                for (x, y, w, h, angle) in pores:
                    cv2.ellipse(image, (x, y), (w, h), angle, 0, 360, PORE_COLOR, thickness=CIRCLE_THICKNESS)

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
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore3"

visualize_class2_segmentation(image_dir, annotation_dir, output_dir)
