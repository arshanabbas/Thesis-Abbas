import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Color and thickness
CLASS_3_COLOR = (0, 0, 0)  # Black for Class 3 outline
CRACK_COLOR = (0, 0, 0)    # Black for cracks
CRACK_THICKNESS = 1        # Thin cracks

# Crack parameters
MIN_CRACK_LENGTH = 30  # Min number of steps
MAX_CRACK_LENGTH = 80  # Max number of steps
STEP_SIZE = 5  # Length of each segment
ANGLE_VARIATION = 30  # Max deviation in angle (degrees)
NUM_CRACKS = 5  # Number of cracks per polygon

# Function to check if a point is inside a polygon
def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Function to generate a random crack inside polygon
def generate_random_crack(polygon):
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]

    # Find random starting point inside polygon
    while True:
        start_x = random.randint(x_min, x_max)
        start_y = random.randint(y_min, y_max)
        if is_point_inside_polygon((start_x, start_y), polygon):
            break

    # Initial random direction
    angle = random.uniform(0, 360)
    crack_points = [(start_x, start_y)]

    # Random length
    length = random.randint(MIN_CRACK_LENGTH, MAX_CRACK_LENGTH)

    for _ in range(length):
        angle += random.uniform(-ANGLE_VARIATION, ANGLE_VARIATION)  # Add some randomness to direction
        dx = STEP_SIZE * math.cos(math.radians(angle))
        dy = STEP_SIZE * math.sin(math.radians(angle))
        new_x = int(crack_points[-1][0] + dx)
        new_y = int(crack_points[-1][1] + dy)

        # Stop if out of polygon
        if not is_point_inside_polygon((new_x, new_y), polygon):
            break

        crack_points.append((new_x, new_y))

    return crack_points

# Main visualization function
def visualize_class3_with_cracks(image_dir, annotation_dir, output_dir=None):
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

                # Draw polygon outline
                cv2.polylines(image, [points], isClosed=True, color=CLASS_3_COLOR, thickness=1)

                # Generate cracks
                for _ in range(NUM_CRACKS):
                    crack = generate_random_crack(points)
                    if len(crack) > 1:
                        for i in range(len(crack) - 1):
                            cv2.line(image, crack[i], crack[i + 1], CRACK_COLOR, thickness=CRACK_THICKNESS)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')

        # Save output
        if output_dir:
            output_path = os.path.join(output_dir, image_name)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

# Example usage
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/crack1"

visualize_class3_with_cracks(image_dir, annotation_dir, output_dir)