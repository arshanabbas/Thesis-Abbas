import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from noise import pnoise1  # Perlin noise function

# Constants
CLASS_3_COLOR = (0, 0, 0)
CRACK_COLOR = (0, 0, 0)
CRACK_THICKNESS = 1
NUM_CRACKS = 5
STEP_SIZE = 5  # Step distance for each crack movement
NOISE_SCALE = 0.1  # Controls Perlin noise smoothness
JAGGEDNESS = 0.4  # Higher = more jaggedness
BRANCH_PROBABILITY = 0.3  # Chance of cracks branching
BRANCH_ANGLE_VARIATION = 45  # Random angle deviation for branches
BRANCH_LENGTH = 20  # Length of branches


# Check if a point is inside a polygon
def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    point = (float(point[0]), float(point[1]))  # Ensure float type
    return cv2.pointPolygonTest(polygon, point, False) >= 0


# Generate Perlin noise-based crack
def generate_perlin_crack(image, polygon, seed):
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]

    # Find a valid starting point inside the polygon
    while True:
        start_x = random.randint(x_min, x_max)
        start_y = random.randint(y_min, y_max)
        if is_point_inside_polygon((start_x, start_y), polygon):
            break

    # Initialize crack path
    current_x, current_y = start_x, start_y
    angle = random.uniform(0, 360)
    noise_offset = random.uniform(0, 100)  # Different noise per crack

    crack_points = [(current_x, current_y)]

    for i in range(50):  # Max length of the crack
        angle_offset = pnoise1((i + noise_offset) * NOISE_SCALE) * JAGGEDNESS * 360  # Get Perlin noise value
        angle += angle_offset

        # Compute next point
        new_x = int(current_x + STEP_SIZE * np.cos(np.radians(angle)))
        new_y = int(current_y + STEP_SIZE * np.sin(np.radians(angle)))

        if not is_point_inside_polygon((new_x, new_y), polygon):
            break  # Stop crack if outside polygon

        crack_points.append((new_x, new_y))
        current_x, current_y = new_x, new_y

        # Branching logic
        if random.random() < BRANCH_PROBABILITY and len(crack_points) > 10:
            branch_angle = angle + random.uniform(-BRANCH_ANGLE_VARIATION, BRANCH_ANGLE_VARIATION)
            branch_end = (
                int(current_x + BRANCH_LENGTH * np.cos(np.radians(branch_angle))),
                int(current_y + BRANCH_LENGTH * np.sin(np.radians(branch_angle)))
            )
            if is_point_inside_polygon(branch_end, polygon):
                cv2.line(image, (current_x, current_y), branch_end, CRACK_COLOR, thickness=1)

    # Draw the main crack
    for i in range(len(crack_points) - 1):
        cv2.line(image, crack_points[i], crack_points[i + 1], CRACK_COLOR, thickness=CRACK_THICKNESS)


# Wrapper to generate multiple cracks in a polygon
def generate_perlin_cracks_in_polygon(image, polygon, num_cracks=NUM_CRACKS):
    for seed in range(num_cracks):  # Different seed for different cracks
        generate_perlin_crack(image, polygon, seed)


# Main function for Class 3 with Perlin cracks
def visualize_class3_with_perlin_cracks(image_dir, annotation_dir, output_dir=None):
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

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                polygon = list(map(float, parts[1:]))

                if class_id != 3:
                    continue  # Process only Class 3

                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0]))
                          for i in range(0, len(polygon), 2)]
                points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

                # Draw polygon outline
                cv2.polylines(image, [points], isClosed=True, color=CLASS_3_COLOR, thickness=1)

                # Generate Perlin noise cracks inside polygon
                generate_perlin_cracks_in_polygon(image, points, num_cracks=NUM_CRACKS)

        # Plot and save
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')

        if output_dir:
            output_path = os.path.join(output_dir, image_name)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()


# Example usage
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/perlin_cracks"

visualize_class3_with_perlin_cracks(image_dir, annotation_dir, output_dir)