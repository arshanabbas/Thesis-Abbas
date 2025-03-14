import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Define colors and parameters
CLASS_3_COLOR = (128,128,128)  # Black outline for Class 3
CRACK_COLOR = (0, 0, 0)  # Black cracks
CRACK_THICKNESS = 1  # Thin crack line

# Crack generation parameters
STEP_SIZE = 7  # Length of each crack segment
ANGLE_VARIATION = 40  # Angle deviation for crack direction
MAX_BRANCH_DEPTH = 3  # How many times cracks can branch
BRANCH_PROBABILITY = 0.2  # Probability of branching
NUM_CRACKS = 3  # Number of cracks per polygon
MIN_CRACK_LENGTH = 10  # Min crack length
MAX_CRACK_LENGTH = 20  # Max crack length


# Function to check if a point is inside a polygon
def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(polygon, point, False) >= 0


# Recursive function to generate branching cracks
def generate_branching_crack(image, start_point, angle, length, polygon, depth=0):
    if depth > MAX_BRANCH_DEPTH or length <= 0:
        return  # Stop if depth or length exceeded

    current_point = start_point

    for _ in range(length):
        angle += random.uniform(-ANGLE_VARIATION, ANGLE_VARIATION)  # Add angular deviation
        dx = int(STEP_SIZE * math.cos(math.radians(angle)))
        dy = int(STEP_SIZE * math.sin(math.radians(angle)))
        next_point = (current_point[0] + dx, current_point[1] + dy)

        if not is_point_inside_polygon(next_point, polygon):
            break  # Stop if outside polygon

        # Draw crack segment
        cv2.line(image, current_point, next_point, CRACK_COLOR, thickness=CRACK_THICKNESS)

        current_point = next_point

        # Random branching
        if random.random() < BRANCH_PROBABILITY:
            branch_angle = angle + random.uniform(-60, 60)
            branch_length = random.randint(10, 30)
            generate_branching_crack(image, current_point, branch_angle, branch_length, polygon, depth + 1)


# Function to generate multiple cracks within a polygon
def generate_cracks_in_polygon(image, polygon, num_cracks=NUM_CRACKS):
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]

    for _ in range(num_cracks):
        # Find a random starting point inside the polygon
        for _ in range(50):  # Limit attempts
            start_x = random.randint(x_min, x_max)
            start_y = random.randint(y_min, y_max)
            if is_point_inside_polygon((start_x, start_y), polygon):
                break

        # Random initial angle and length
        angle = random.uniform(0, 360)
        length = random.randint(MIN_CRACK_LENGTH, MAX_CRACK_LENGTH)
        generate_branching_crack(image, (start_x, start_y), angle, length, polygon)


# Main visualization function for Class 3 with cracks
def visualize_class3_with_branching_cracks(image_dir, annotation_dir, output_dir=None):
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

        # Load and convert image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process annotation file
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                polygon = list(map(float, parts[1:]))

                # Process only Class 3
                if class_id != 3:
                    continue

                # Normalize and map coordinates to image dimensions
                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0]))
                          for i in range(0, len(polygon), 2)]
                points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

                # Draw Class 3 polygon outline
                cv2.polylines(image, [points], isClosed=True, color=CLASS_3_COLOR, thickness=2)

                # Generate branching cracks inside polygon
                generate_cracks_in_polygon(image, points, num_cracks=NUM_CRACKS)

        # Plot and save output
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')

        if output_dir:
            output_path = os.path.join(output_dir, image_name)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()


# Example usage:
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/branching_cracks"

visualize_class3_with_branching_cracks(image_dir, annotation_dir, output_dir)