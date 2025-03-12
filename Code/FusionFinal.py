import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# ----------------------- Configuration -----------------------
CLASS_3_COLOR = (64, 64, 64)  # Grey for Class 3 outline
CIRCLE_THICKNESS = 1  # Thin outline
MIN_PORE_RADIUS = 2
MAX_PORE_RADIUS = 5
MIN_TOTAL_PORES = 15
MAX_TOTAL_PORES = 30
MIN_CLUSTERS = 2
MAX_CLUSTERS = 3
MIN_DISTANCE_BETWEEN_CLUSTER_PORES = 2
MIN_DISTANCE_BETWEEN_SCATTERED_PORES = 4
PORE_CLASS_ID = 0  # Pore
PORE_NEST_CLASS_ID = 1  # Porennest (cluster)

# ----------------------- Helper Functions -----------------------
def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0

def convert_to_yolo_bbox(x, y, w, h, image_w, image_h):
    return x / image_w, y / image_h, (2 * w) / image_w, (2 * h) / image_h

def save_yolo_labels(output_dir, image_name, labels):
    label_file = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
    with open(label_file, "w") as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

# ----------------------- Pore and Cluster Generation -----------------------
def generate_balanced_pores_with_labels(polygon, img_shape):
    pores, labels = [], []
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]
    max_attempts = 200
    num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)
    cluster_centers = [(random.randint(x_min, x_max), random.randint(y_min, y_max)) for _ in range(num_clusters)]
    num_pores = max(MIN_TOTAL_PORES, min(MAX_TOTAL_PORES, int(cv2.contourArea(polygon) // 50)))
    cluster_pore_count = max(5, min(random.randint(5, 10) * num_clusters, num_pores - 5))
    scattered_pore_count = max(5, num_pores - cluster_pore_count)

    def is_far_enough(nx, ny, nr, existing, min_dist):
        for (x, y, w, h, _) in existing:
            if np.linalg.norm(np.array([nx, ny]) - np.array([x, y])) < (max(w, h) + nr + min_dist):
                return False
        return True

    # -------- Cluster Pores --------
    cluster_pore_positions = []
    for _ in range(cluster_pore_count):
        for _ in range(max_attempts):
            cx, cy = random.choice(cluster_centers)
            x, y = int(cx + random.randint(-10, 10)), int(cy + random.randint(-10, 10))
            w, h, angle = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS), random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS), random.randint(0, 180)
            if is_point_inside_polygon((x, y), polygon) and is_far_enough(x, y, max(w, h), pores, MIN_DISTANCE_BETWEEN_CLUSTER_PORES):
                pores.append((x, y, w, h, angle))
                cluster_pore_positions.append((x, y, w, h))
                break

    # Create bounding box for cluster (Porennest)
    if cluster_pore_positions:
        xs, ys, ws, hs = zip(*cluster_pore_positions)
        min_x, max_x = min(xs) - max(ws), max(xs) + max(ws)
        min_y, max_y = min(ys) - max(hs), max(ys) + max(hs)
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        cluster_w, cluster_h = max_x - min_x, max_y - min_y
        bx, by, bw, bh = convert_to_yolo_bbox(cx, cy, cluster_w / 2, cluster_h / 2, img_shape[1], img_shape[0])
        labels.append((PORE_NEST_CLASS_ID, bx, by, bw, bh))

    # -------- Scattered Pores --------
    for _ in range(scattered_pore_count):
        for _ in range(max_attempts):
            x, y = random.randint(x_min, x_max), random.randint(y_min, y_max)
            w, h, angle = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS), random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS), random.randint(0, 180)
            if is_point_inside_polygon((x, y), polygon) and is_far_enough(x, y, max(w, h), pores, MIN_DISTANCE_BETWEEN_SCATTERED_PORES):
                pores.append((x, y, w, h, angle))
                bx, by, bw, bh = convert_to_yolo_bbox(x, y, w, h, img_shape[1], img_shape[0])
                labels.append((PORE_CLASS_ID, bx, by, bw, bh))
                break

    return pores, labels

# ----------------------- Main Function with Visualization -----------------------
def visualize_class3_and_annotate(image_dir, annotation_dir, output_dir=None):
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
        label_list = []

        with open(annotation_path, 'r') as f:
            for line in f:
                if int(line.strip().split()[0]) != 3: continue  # Only process Class 3
                polygon = list(map(float, line.strip().split()[1:]))
                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0])) for i in range(0, len(polygon), 2)]
                points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [points], True, CLASS_3_COLOR, CIRCLE_THICKNESS)
                pores, labels = generate_balanced_pores_with_labels(points, image.shape)
                label_list.extend(labels)
                for (x, y, w, h, angle) in pores:
                    gray = random.randint(50, 120)
                    cv2.ellipse(image, (x, y), (w, h), angle, 0, 360, (gray, gray, gray), -1)

        # Draw YOLO bounding boxes for visual check
        for class_id, cx, cy, bw, bh in label_list:
            color = (0, 255, 0) if class_id == PORE_CLASS_ID else (255, 0, 0)
            x1, y1, x2, y2 = int((cx - bw/2)*image.shape[1]), int((cy - bh/2)*image.shape[0]), int((cx + bw/2)*image.shape[1]), int((cy + bh/2)*image.shape[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        print(f"Total pores in {image_name}: {len(label_list)}")
        save_yolo_labels(output_dir, image_name, label_list)
        plt.imshow(image), plt.axis('off'), plt.show()

# ----------------------- Example Run -----------------------
# Example usage
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset"