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

# YOLOv1 Format Output
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

def get_random_point_on_named_edge(polygon, edge_name):
    polygon = polygon.reshape((-1, 2))
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)

    if edge_name == "north":
        y = y_min
        x = random.randint(x_min, x_max)
    elif edge_name == "south":
        y = y_max
        x = random.randint(x_min, x_max)
    elif edge_name == "west":
        x = x_min
        y = random.randint(y_min, y_max)
    elif edge_name == "east":
        x = x_max
        y = random.randint(y_min, y_max)
    else:
        return get_random_point_on_edge(polygon)

    return x, y

def get_polygon_center(polygon):
    polygon = polygon.reshape((-1, 2))
    center_x = int(np.mean(polygon[:, 0]))
    center_y = int(np.mean(polygon[:, 1]))
    return center_x, center_y

def is_overlapping(new_crack, existing_cracks, min_distance=5):
    for crack in existing_cracks:
        for (x1, y1) in crack:
            for (x2, y2) in new_crack:
                if abs(x1 - x2) < min_distance and abs(y1 - y2) < min_distance:
                    return True
    return False

def generate_random_crack(polygon, existing_cracks, edge_name=None, force_generate=False):
    max_attempts = 30
    attempt = 0

    while attempt < max_attempts:
        if edge_name:
            start_x, start_y = get_random_point_on_named_edge(polygon, edge_name)
        else:
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

            new_crack = crack_points + [(new_x, new_y)]
            if is_overlapping(new_crack, existing_cracks) and not force_generate:
                break

            crack_points.append((new_x, new_y))

        if len(crack_points) >= MIN_CRACK_LENGTH:
            return crack_points

        attempt += 1

    if force_generate and len(crack_points) > 1:
        return crack_points
    return None

def convert_to_yolo_bbox(x, y, w, h, image_w, image_h):
    return x / image_w, y / image_h, (2 * w) / image_w, (2 * h) / image_h

def save_crack_bounding_box(crack_points, image_shape):
    crack_points = np.array(crack_points)
    x_min = np.min(crack_points[:, 0])
    y_min = np.min(crack_points[:, 1])
    x_max = np.max(crack_points[:, 0])
    y_max = np.max(crack_points[:, 1])

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_w = (x_max - x_min) / 2
    half_h = (y_max - y_min) / 2

    bx, by, bw, bh = convert_to_yolo_bbox(center_x, center_y, half_w, half_h, image_shape[1], image_shape[0])
    return (YOLO_CLASS_ID, bx, by, bw, bh)

def write_yolo_labels_to_file(label_path, labels):
    with open(label_path, "w") as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

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

        overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        existing_cracks = []
        labels = []

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

                generated_cracks = 0
                available_edges = ["north", "south", "east", "west"]
                random.shuffle(available_edges)
                attempts = 0

                while generated_cracks < MIN_CRACKS and attempts < 100:
                    edge = available_edges[generated_cracks % 4]
                    crack = generate_random_crack(points_np, existing_cracks, edge_name=edge)

                    if crack:
                        existing_cracks.append(crack)
                        generated_cracks += 1

                        total_segments = len(crack) - 1
                        for i in range(total_segments):
                            progress = i / total_segments
                            alpha_value = int(255 * (1 - progress))
                            color_with_alpha = CRACK_COLOR + (alpha_value,)
                            cv2.line(overlay, crack[i], crack[i + 1], color_with_alpha, thickness=CRACK_THICKNESS)

                        if save_yolo_labels:
                            label = save_crack_bounding_box(crack, image.shape)
                            labels.append(label)

                    attempts += 1

                while generated_cracks < MIN_CRACKS:
                    crack = generate_random_crack(points_np, existing_cracks, force_generate=True)
                    if crack:
                        existing_cracks.append(crack)
                        generated_cracks += 1

                        total_segments = len(crack) - 1
                        for i in range(total_segments):
                            progress = i / total_segments
                            alpha_value = int(255 * (1 - progress))
                            color_with_alpha = CRACK_COLOR + (alpha_value,)
                            cv2.line(overlay, crack[i], crack[i + 1], color_with_alpha, thickness=CRACK_THICKNESS)

                        if save_yolo_labels:
                            label = save_crack_bounding_box(crack, image.shape)
                            labels.append(label)

        if save_yolo_labels and labels:
            label_path = os.path.join(yolo_label_output_dir, f"{os.path.splitext(image_name)[0]}.txt")
            write_yolo_labels_to_file(label_path, labels)

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
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/crack_dataset/images"

visualize_class3_with_cracks(image_dir, annotation_dir, output_dir)