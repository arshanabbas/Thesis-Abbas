import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# --------------------- CONFIGURATION -----------------------

CLASS_3_COLOR = (128, 128, 128)
CRACK_COLOR = (0, 0, 0)
CRACK_THICKNESS = 2

# Crack parameters
MIN_CRACK_LENGTH = 60
MAX_CRACK_LENGTH = 80
STEP_SIZE = 5
ANGLE_VARIATION = 15
MIN_CRACKS = 2
MAX_CRACKS = 2

# YOLOv8 Format Output
YOLO_CLASS_ID = 0
save_yolo_labels = True
yolo_label_output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/crack_dataset/labels"

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

def generate_random_crack(polygon):
    max_attempts = 50
    attempt = 0

    while attempt < max_attempts:
        start_x, start_y = get_random_point_on_edge(polygon)
        center_x, center_y = get_polygon_center(polygon)
        angle = math.degrees(math.atan2(center_y - start_y, center_x - start_x))
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

        if len(crack_points) >= MIN_CRACK_LENGTH:
            return crack_points

        attempt += 1

    return None

def save_crack_bounding_box(cracks, image_shape, label_path, padding=5):
    """Saves bounding boxes for cracks in YOLO format."""
    with open(label_path, "w") as f:
        for crack in cracks:
            crack = np.array(crack)
            x_min = max(0, np.min(crack[:, 0]) - padding)
            y_min = max(0, np.min(crack[:, 1]) - padding)
            x_max = min(image_shape[1], np.max(crack[:, 0]) + padding)
            y_max = min(image_shape[0], np.max(crack[:, 1]) + padding)

            x_center = (x_min + x_max) / 2 / image_shape[1]
            y_center = (y_min + y_max) / 2 / image_shape[0]
            width = (x_max - x_min) / image_shape[1]
            height = (y_max - y_min) / image_shape[0]

            f.write(f"{YOLO_CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# --------------------- MAIN FUNCTION -----------------------

def visualize_class3_with_cracks(image_dir, annotation_dir, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if save_yolo_labels:
        os.makedirs(yolo_label_output_dir, exist_ok=True)

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

                while len(existing_cracks) < MIN_CRACKS:
                    crack = generate_random_crack(points_np)
                    if crack:
                        existing_cracks.append(crack)

        for crack in existing_cracks:
            for i in range(len(crack) - 1):
                cv2.line(image, crack[i], crack[i + 1], CRACK_COLOR, thickness=CRACK_THICKNESS)

        if save_yolo_labels:
            label_path = os.path.join(yolo_label_output_dir, f"{os.path.splitext(image_name)[0]}.txt")
            save_crack_bounding_box(existing_cracks, image.shape, label_path)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')
        if output_dir:
            plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            plt.show()

# --------------------- USAGE -----------------------

image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/crack_dataset/images"

visualize_class3_with_cracks(image_dir, annotation_dir, output_dir)