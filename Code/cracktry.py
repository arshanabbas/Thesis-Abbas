import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# --------------------- CONFIGURATION -----------------------

# Color and thickness
CLASS_3_COLOR = (128, 128, 128)  # Gray outline for Class 3
CRACK_COLOR = (0, 0, 0)          # Black for cracks
CRACK_THICKNESS = 5              # Starting thickness of cracks (can be adjusted)

# Crack parameters
MIN_CRACK_LENGTH = 30  # Min number of steps
MAX_CRACK_LENGTH = 80  # Max number of steps
STEP_SIZE = 5          # Length of each segment
ANGLE_VARIATION = 30   # Max deviation in angle (degrees)
NUM_CRACKS = 5         # Number of cracks per polygon

# --------------------- HELPER FUNCTIONS -----------------------

# Function to check if a point is inside a polygon
def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Function to get a random point on the polygon's edge
def get_random_point_on_edge(polygon):
    polygon = polygon.reshape((-1, 2))  # Simplify shape
    idx = random.randint(0, len(polygon) - 1)
    pt1 = polygon[idx]
    pt2 = polygon[(idx + 1) % len(polygon)]  # Wrap around
    t = random.random()  # Random fraction along edge
    x = int(pt1[0] * (1 - t) + pt2[0] * t)
    y = int(pt1[1] * (1 - t) + pt2[1] * t)
    return x, y

# Function to generate a random crack starting from edge
def generate_random_crack(polygon):
    # Start from edge
    start_x, start_y = get_random_point_on_edge(polygon)

    # Random initial direction
    angle = random.uniform(0, 360)
    crack_points = [(start_x, start_y)]

    # Random length
    length = random.randint(MIN_CRACK_LENGTH, MAX_CRACK_LENGTH)

    for _ in range(length):
        angle += random.uniform(-ANGLE_VARIATION, ANGLE_VARIATION)  # Random angle adjustment
        dx = STEP_SIZE * math.cos(math.radians(angle))
        dy = STEP_SIZE * math.sin(math.radians(angle))
        new_x = int(crack_points[-1][0] + dx)
        new_y = int(crack_points[-1][1] + dy)

        # Stop if out of polygon
        if not is_point_inside_polygon((new_x, new_y), polygon):
            break

        crack_points.append((new_x, new_y))

    return crack_points

# --------------------- MAIN FUNCTION -----------------------

def visualize_class3_with_cracks(image_dir, annotation_dir, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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

                # Process only Class 3
                if class_id != 3:
                    continue

                # Convert normalized points to image dimensions
                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0]))
                          for i in range(0, len(polygon), 2)]
                points_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

                # Draw polygon outline
                cv2.polylines(image, [points_np], isClosed=True, color=CLASS_3_COLOR, thickness=2)

                # --------------------- CRACK GENERATION -----------------------

                for _ in range(NUM_CRACKS):
                    crack = generate_random_crack(points_np)
                    if len(crack) > 1:
                        total_segments = len(crack) - 1
                        for i in range(total_segments):
                            # Progress ratio from start (0) to end (1)
                            progress = i / total_segments

                            # Thickness fades out smoothly (power 2 for smoothness)
                            thickness = max(1, int(CRACK_THICKNESS * ((1 - progress) ** 2)))

                            # Draw crack segment with varying thickness
                            cv2.line(image, crack[i], crack[i + 1], CRACK_COLOR, thickness=thickness)

        # --------------------- DISPLAY OR SAVE -----------------------

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')

        if output_dir:
            output_path = os.path.join(output_dir, image_name)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

# --------------------- USAGE -----------------------

# Example directories (replace with your own)
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/Randomwalk_cracks"

# Run visualization
visualize_class3_with_cracks(image_dir, annotation_dir, output_dir)
