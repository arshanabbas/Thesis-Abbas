import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# --------------------- CONFIGURATION -----------------------

CLASS_3_COLOR = (128, 128, 128)  # Gray outline for Class 3
CRACK_CORE_COLOR = (128, 128, 128)  # Normal grey crack color
CRACK_EDGE_COLOR = (180, 180, 180)  # Light grey edges
CRACK_THICKNESS = 2  # Core thickness
CRACK_EDGE_THICKNESS = CRACK_THICKNESS + 2  # Edge thickness (slightly bigger)

# Crack parameters
MIN_CRACK_LENGTH = 50
MAX_CRACK_LENGTH = 80
STEP_SIZE = 5
ANGLE_VARIATION = 15
MIN_CRACKS = 1
MAX_CRACKS = 3

# --------------------- HELPER FUNCTIONS -----------------------

def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def get_random_point_on_edge(polygon):
    polygon = polygon.reshape((-1, 2))
    idx = random.randint(0, len(polygon) - 1)
    pt1 = polygon[idx]
    pt2 = polygon[(idx + 1) % len(polygon)]
    t = random.random()
    x = int(pt1[0] * (1 - t) + pt2[0] * t)
    y = int(pt1[1] * (1 - t) + pt2[1] * t)
    return x, y

def get_polygon_center(polygon):
    polygon = polygon.reshape((-1, 2))
    center_x = int(np.mean(polygon[:, 0]))
    center_y = int(np.mean(polygon[:, 1]))
    return center_x, center_y

def generate_random_crack(polygon, existing_cracks):
    start_x, start_y = get_random_point_on_edge(polygon)
    center_x, center_y = get_polygon_center(polygon)

    angle = math.degrees(math.atan2(center_y - start_x, center_x - start_y))  # Aim at center
    crack_points = [(start_x, start_y)]
    length = random.randint(MIN_CRACK_LENGTH, MAX_CRACK_LENGTH)

    for _ in range(length):
        angle += random.uniform(-ANGLE_VARIATION, ANGLE_VARIATION)
        dx = STEP_SIZE * math.cos(math.radians(angle))
        dy = STEP_SIZE * math.sin(math.radians(angle))
        new_x = int(crack_points[-1][0] + dx)
        new_y = int(crack_points[-1][1] + dy)

        if not is_point_inside_polygon((new_x, new_y), polygon):
            break

        crack_points.append((new_x, new_y))

    return crack_points if len(crack_points) > 1 else None  # Ensure valid crack

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

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        existing_cracks = []

        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                polygon = list(map(float, parts[1:]))

                if class_id != 3:
                    continue

                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0]))
                          for i in range(0, len(polygon), 2)]
                points_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

                cv2.polylines(image, [points_np], isClosed=True, color=CLASS_3_COLOR, thickness=2)

                # Ensure 1 to 3 cracks
                num_cracks = random.randint(MIN_CRACKS, MAX_CRACKS)
                generated_cracks = 0
                retries = 10  # Prevent infinite loops

                while generated_cracks < num_cracks and retries > 0:
                    crack = generate_random_crack(points_np, existing_cracks)
                    if crack:
                        existing_cracks.append(crack)
                        generated_cracks += 1
                        total_segments = len(crack) - 1

                        for i in range(total_segments):
                            progress = i / total_segments
                            alpha_value = int(255 * (1 - progress))  # Smooth fade

                            # **Draw crack edges first (light grey, slightly thicker)**
                            cv2.line(overlay, crack[i], crack[i + 1], CRACK_EDGE_COLOR + (alpha_value,),
                                     thickness=CRACK_EDGE_THICKNESS)

                            # **Draw crack core second (normal grey, same thickness)**
                            cv2.line(overlay, crack[i], crack[i + 1], CRACK_CORE_COLOR + (alpha_value,),
                                     thickness=CRACK_THICKNESS)

                    retries -= 1

                # If no cracks were generated, force one
                if generated_cracks == 0:
                    crack = generate_random_crack(points_np, existing_cracks)
                    if crack:
                        existing_cracks.append(crack)
                        total_segments = len(crack) - 1
                        for i in range(total_segments):
                            progress = i / total_segments
                            alpha_value = int(255 * (1 - progress))

                            cv2.line(overlay, crack[i], crack[i + 1], CRACK_EDGE_COLOR + (alpha_value,),
                                     thickness=CRACK_EDGE_THICKNESS)
                            cv2.line(overlay, crack[i], crack[i + 1], CRACK_CORE_COLOR + (alpha_value,),
                                     thickness=CRACK_THICKNESS)

        def blend_overlay(base_image, overlay):
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                base_image[:, :, c] = (1 - alpha) * base_image[:, :, c] + alpha * overlay[:, :, c]
            return base_image.astype(np.uint8)

        final_image = blend_overlay(image, overlay)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(final_image)
        ax.axis('off')

        if output_dir:
            output_path = os.path.join(output_dir, image_name)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

# --------------------- USAGE -----------------------

image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/cracksgrey"

visualize_class3_with_cracks(image_dir, annotation_dir, output_dir)