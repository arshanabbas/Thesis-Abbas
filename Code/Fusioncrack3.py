import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
CLASS_3_COLOR = (128,128,128)  # Black outline for Class 3
CRACK_COLOR = (0, 0, 0)  # Black cracks
CRACK_THICKNESS = 1  # Thin line
NUM_CRACKS = 5  # Number of cracks per polygon
NUM_CONTROL_POINTS = 4  # Cubic Bezier curve (start, end, two control points)
NUM_BEZIER_POINTS = 100  # Number of points to sample along the curve


# Check if a point is inside a polygon
def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(polygon, point, False) >= 0


# Generate a Bezier curve from control points
def generate_bezier_curve(control_points, num_points=100):
    control_points = np.array(control_points)
    n = len(control_points) - 1

    def bernstein_poly(i, n, t):
        return comb(n, i) * (t ** i) * (1 - t) ** (n - i)

    def comb(n, k):
        from math import comb as math_comb  # Use math's comb for binomial coefficient
        return math_comb(n, k)

    t_values = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2), dtype=np.float32)

    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t_values), control_points[i])

    return curve.astype(int)


# Function to generate a single bezier crack within a polygon
def generate_bezier_crack(polygon, num_control_points=NUM_CONTROL_POINTS):
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]

    control_points = []
    attempts = 0

    while len(control_points) < num_control_points and attempts < 100:
        rand_x = random.randint(x_min, x_max)
        rand_y = random.randint(y_min, y_max)
        if is_point_inside_polygon((rand_x, rand_y), polygon):
            control_points.append((rand_x, rand_y))
        attempts += 1

    if len(control_points) < num_control_points:
        return None  # Not enough points found inside polygon

    return generate_bezier_curve(control_points, num_points=NUM_BEZIER_POINTS)


# Wrapper to generate multiple cracks per polygon
def generate_bezier_cracks_in_polygon(image, polygon, num_cracks=NUM_CRACKS):
    for _ in range(num_cracks):
        bezier_crack = generate_bezier_crack(polygon)
        if bezier_crack is not None:
            for i in range(len(bezier_crack) - 1):
                pt1 = tuple(bezier_crack[i])
                pt2 = tuple(bezier_crack[i + 1])
                cv2.line(image, pt1, pt2, CRACK_COLOR, thickness=CRACK_THICKNESS)


# Main visualization function for Class 3 with Bezier cracks
def visualize_class3_with_bezier_cracks(image_dir, annotation_dir, output_dir=None):
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

                # Generate Bezier cracks inside polygon
                generate_bezier_cracks_in_polygon(image, points, num_cracks=NUM_CRACKS)

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
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/bezier_cracks"

visualize_class3_with_bezier_cracks(image_dir, annotation_dir, output_dir)