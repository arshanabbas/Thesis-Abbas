import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
CLASS_3_COLOR = (0, 0, 0)
CRACK_COLOR = (0, 0, 0)
CRACK_THICKNESS = 1
NUM_CRACKS = 5
NUM_CONTROL_POINTS = 4
NUM_BEZIER_POINTS = 100
NOISE_INTENSITY = 3  # Perturbation intensity
BRANCH_PROBABILITY = 0.3  # Probability of branching
BRANCH_LENGTH = 20  # Length of branch cracks


# Check if a point is inside a polygon (fixed)
def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    point = (float(point[0]), float(point[1]))  # Ensure float type
    return cv2.pointPolygonTest(polygon, point, False) >= 0


# Bezier curve generator
def generate_bezier_curve(control_points, num_points=100):
    control_points = np.array(control_points)
    n = len(control_points) - 1

    def bernstein_poly(i, n, t):
        return comb(n, i) * (t ** i) * (1 - t) ** (n - i)

    def comb(n, k):
        from math import comb as math_comb
        return math_comb(n, k)

    t_values = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2), dtype=np.float32)

    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t_values), control_points[i])

    return curve.astype(int)


# Perturb curve to make it jagged (fixed)
def perturb_curve(curve, intensity, polygon):
    perturbed = []
    for pt in curve:
        dx = random.randint(-intensity, intensity)
        dy = random.randint(-intensity, intensity)
        new_pt = (pt[0] + dx, pt[1] + dy)
        if is_point_inside_polygon((float(new_pt[0]), float(new_pt[1])), polygon):
            perturbed.append(new_pt)
        else:
            perturbed.append(pt)  # Keep original if outside
    return perturbed


# Generate Bezier crack with perturbation and branches
def generate_realistic_bezier_crack(image, polygon):
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]

    control_points = []
    while len(control_points) < NUM_CONTROL_POINTS:
        rand_x = random.randint(x_min, x_max)
        rand_y = random.randint(y_min, y_max)
        if is_point_inside_polygon((rand_x, rand_y), polygon):
            control_points.append((rand_x, rand_y))

    curve = generate_bezier_curve(control_points, num_points=NUM_BEZIER_POINTS)
    curve = perturb_curve(curve, NOISE_INTENSITY, polygon)

    # Draw main crack
    for i in range(len(curve) - 1):
        thickness = max(1, int(CRACK_THICKNESS * (1 - i / len(curve))))  # Optional thinning
        cv2.line(image, curve[i], curve[i + 1], CRACK_COLOR, thickness=thickness)

    # Add branches
    for i in range(5, len(curve) - 5, 10):  # Every few points
        if random.random() < BRANCH_PROBABILITY:
            branch_angle = random.uniform(0, 360)
            branch_end = (
                int(curve[i][0] + BRANCH_LENGTH * np.cos(np.radians(branch_angle))),
                int(curve[i][1] + BRANCH_LENGTH * np.sin(np.radians(branch_angle)))
            )
            if is_point_inside_polygon((float(branch_end[0]), float(branch_end[1])), polygon):  # Fixed check
                cv2.line(image, curve[i], branch_end, CRACK_COLOR, thickness=1)


# Wrapper to generate multiple cracks in a polygon
def generate_bezier_cracks_in_polygon(image, polygon, num_cracks=NUM_CRACKS):
    for _ in range(num_cracks):
        generate_realistic_bezier_crack(image, polygon)


# Main visualization function for Class 3 with cracks
def visualize_class3_with_improved_bezier_cracks(image_dir, annotation_dir, output_dir=None):
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

                # Generate improved cracks inside polygon
                generate_bezier_cracks_in_polygon(image, points, num_cracks=NUM_CRACKS)

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


# ✅ Example usage
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/bezier_improved_cracks"

visualize_class3_with_improved_bezier_cracks(image_dir, annotation_dir, output_dir)