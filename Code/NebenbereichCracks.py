import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from noise import pnoise1

# === Directory Paths ===
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/visualisation"

os.makedirs(output_dir, exist_ok=True)

# === Configurations ===
CLASS_2_COLOR = (0, 0, 255)  # Blue outline for Class 2
CRACK_COLOR = (0, 0, 0)  # Black crack
CRACK_THICKNESS = 1  # Thin cracks
NUM_CRACKS = 1  # Only one primary crack
MAX_CRACK_LENGTH = 120  # Length of the crack
BRANCH_PROBABILITY = 0.2  # Small branches

# === Utility Functions ===
def is_point_inside_polygon(point, polygon):
    """Check if a point is inside the Nebenbereich polygon."""
    point = np.array(point, dtype=np.float32)  # Convert point to float32
    polygon = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))  # Ensure correct shape
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def generate_main_crack(start_x, start_y, polygon, length=MAX_CRACK_LENGTH):
    """Generate a main vertical crack using Perlin noise."""
    crack_path = [(start_x, start_y)]
    
    for i in range(length):
        noise = pnoise1(i * 0.1)  # Perlin noise for natural waviness
        new_x = start_x + int(noise * 5)  # Small horizontal deviations
        new_y = start_y + i  # Vertical growth

        if is_point_inside_polygon((new_x, new_y), polygon):  # Ensure it's inside
            crack_path.append((new_x, new_y))
        else:
            break  # Stop if it reaches outside

    return crack_path

def add_side_branches(crack_path, polygon, branch_prob=BRANCH_PROBABILITY):
    """Create small branching cracks along the main fracture."""
    branched_paths = [crack_path]

    for i, (x, y) in enumerate(crack_path):
        if random.random() < branch_prob:
            branch_length = random.randint(10, 30)
            branch = []
            for j in range(branch_length):
                new_x = x + random.choice([-1, 1]) * j  # Small horizontal move
                new_y = y + random.randint(-1, 1)  # Slight vertical move
                if is_point_inside_polygon((new_x, new_y), polygon):
                    branch.append((new_x, new_y))
                else:
                    break
            branched_paths.append(branch)

    return branched_paths

# === Main Processing ===
def generate_realistic_crack(image, polygon):
    """Generate a central jagged crack inside the Nebenbereich region."""
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]

    # Place crack roughly in the center of the polygon
    start_x = (x_min + x_max) // 2
    start_y = y_min + 10  # Start slightly inside

    # Generate main crack
    crack_path = generate_main_crack(start_x, start_y, polygon)
    branched_paths = add_side_branches(crack_path, polygon)

    # Draw cracks on the image
    for path in branched_paths:
        for i in range(1, len(path)):
            thickness = max(1, random.randint(1, 2))  # Slight variation in thickness
            cv2.line(image, path[i - 1], path[i], CRACK_COLOR, thickness)

# === Main Visualization Function ===
def visualize_class2_segmentation():
    """Load images, process Nebenbereich, and generate cracks."""
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

        # Read annotation file
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                polygon = list(map(float, parts[1:]))

                if class_id != 2:
                    continue  # Process only "Nebenbereich" (Class 2)

                # Convert polygon points to image coordinates
                points = np.array([(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0]))
                          for i in range(0, len(polygon), 2)], dtype=np.int32).reshape((-1, 1, 2))

                # Draw Nebenbereich boundary
                cv2.polylines(image, [points], isClosed=True, color=CLASS_2_COLOR, thickness=1)

                # Generate realistic crack inside Nebenbereich
                generate_realistic_crack(image, points)

        # Save & Display
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Processed and saved: {output_path}")

# === Run the Function ===
visualize_class2_segmentation()