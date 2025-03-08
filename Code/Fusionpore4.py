import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Define colors and pore parameters
CLASS_2_COLOR = (0, 0, 255)  # Blue for Class 2 outline
PORE_COLOR = (0, 0, 255)  # Blue outline for pores

CIRCLE_THICKNESS = 1  # Thin outline
MIN_PORE_RADIUS = 2  # Minimum pore size
MAX_PORE_RADIUS = 5  # Maximum pore size
MIN_TOTAL_PORES = 25  # Minimum pores per image
MAX_TOTAL_PORES = 40  # Maximum pores per image
MIN_CLUSTERS = 2  # At least two clusters per image
MAX_CLUSTERS = 3  # Maximum number of clusters
MIN_DISTANCE_BETWEEN_PORES = 2  # Minimum distance to prevent overlap

# Function to check if a point is inside a polygon
def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))  # Ensure correct shape
    point = (float(point[0]), float(point[1]))  # Convert to float
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Function to calculate the area of a polygon
def polygon_area(polygon):
    return 0.5 * abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))

# Function to generate pores across the whole Nebenbereich with small clusters and scattered pores, avoiding overlap
def generate_balanced_pores(polygon, img_shape):
    pores = []
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))  # Ensure correct format
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]
    
    max_attempts = 50  # Attempts per pore to prevent overlap
    num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)  # At least two clusters per image
    cluster_centers = []
    
    # Generate cluster centers
    for _ in range(num_clusters):
        cluster_x = random.randint(x_min, x_max)
        cluster_y = random.randint(y_min, y_max)
        cluster_centers.append((cluster_x, cluster_y))
    
    # Define the total number of pores
    num_pores = random.randint(MIN_TOTAL_PORES, MAX_TOTAL_PORES)
    cluster_pore_count = min(random.randint(5, 10) * num_clusters, num_pores - 5)  # At least some scattered pores
    scattered_pore_count = num_pores - cluster_pore_count  # Remaining pores are scattered
    
    def is_far_enough(new_x, new_y, new_r):
        for (x, y, w, h, _) in pores:
            existing_r = max(w, h)  # Approximate radius from width and height
            if np.linalg.norm(np.array([new_x, new_y]) - np.array([x, y])) < (existing_r + new_r + MIN_DISTANCE_BETWEEN_PORES):
                return False
        return True
    
    for _ in range(cluster_pore_count):
        attempts = 0
        while attempts < max_attempts:
            cluster_x, cluster_y = random.choice(cluster_centers)
            rand_x = int(cluster_x + random.randint(-10, 10))
            rand_y = int(cluster_y + random.randint(-10, 10))
            rand_w = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
            rand_h = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
            angle = random.randint(0, 180)

            if is_point_inside_polygon((rand_x, rand_y), polygon) and is_far_enough(rand_x, rand_y, max(rand_w, rand_h)):
                pores.append((rand_x, rand_y, rand_w, rand_h, angle))
                break
            attempts += 1
    
    for _ in range(scattered_pore_count):
        attempts = 0
        while attempts < max_attempts:
            rand_x = random.randint(x_min, x_max)
            rand_y = random.randint(y_min, y_max)
            rand_w = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
            rand_h = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
            angle = random.randint(0, 180)

            if is_point_inside_polygon((rand_x, rand_y), polygon) and is_far_enough(rand_x, rand_y, max(rand_w, rand_h)):
                pores.append((rand_x, rand_y, rand_w, rand_h, angle))
                break
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

                # Only process Class 2 (Nebenbereich)
                if class_id != 2:
                    continue

                # Normalize and map coordinates to image dimensions
                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0]))
                          for i in range(0, len(polygon), 2)]
                points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

                # Draw polygon outline instead of filling
                cv2.polylines(image, [points], isClosed=True, color=CLASS_2_COLOR, thickness=1)

                # Generate realistic, irregular pores across the full region
                pores = generate_balanced_pores(points, image.shape)

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
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore5"

visualize_class2_segmentation(image_dir, annotation_dir, output_dir)
