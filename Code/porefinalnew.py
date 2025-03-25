import os
import cv2
import numpy as np
import random

# ----------------------- Configuration -----------------------
CLASS_3_COLOR = (64, 64, 64)
CIRCLE_THICKNESS = 1
MIN_PORE_RADIUS = 3
MAX_PORE_RADIUS = 5
MIN_TOTAL_PORES = 15
MAX_TOTAL_PORES = 30
MIN_CLUSTERS = 2
MAX_CLUSTERS = 3
MIN_DISTANCE_BETWEEN_CLUSTER_PORES = 8
MIN_DISTANCE_BETWEEN_SCATTERED_PORES = 5
PORE_CLASS_ID = 0
PORE_NEST_CLASS_ID = 1
CLUSTER_PADDING = 10
PORE_PADDING = 5
MIN_DISTANCE_BETWEEN_CLUSTERS = 40

# ----------------------- Helper Functions -----------------------
def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0

def convert_to_yolo_bbox(x, y, w, h, image_w, image_h):
    return x / image_w, y / image_h, (2 * w) / image_w, (2 * h) / image_h

def save_yolo_labels(output_labels_dir, image_name, labels):
    label_file = os.path.join(output_labels_dir, os.path.splitext(image_name)[0] + ".txt")
    with open(label_file, "w") as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

def are_clusters_far_enough(new_center, existing_centers, min_distance):
    for center in existing_centers:
        if np.linalg.norm(np.array(new_center) - np.array(center)) < min_distance:
            return False
    return True

def draw_pore(image, x, y, w, h, angle):
    """
    Draws a pore using cv2.circle for small radii (1 or 2) and cv2.ellipse for larger radii (3-5).
    """
    if w <= 2 and h <= 2:
        cv2.circle(image, (x, y), w, (80, 80, 80), -1)  # Filled circle for small pores
    else:
        cv2.ellipse(image, (x, y), (w, h), angle, 0, 360, (80, 80, 80), -1)

def generate_balanced_pores_with_labels(polygon, img_shape):
    pores, labels = [], []
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]
    num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)
    cluster_centers = []
    
    while len(cluster_centers) < num_clusters:
        cx, cy = random.randint(x_min, x_max), random.randint(y_min, y_max)
        if is_point_inside_polygon((cx, cy), polygon) and are_clusters_far_enough((cx, cy), cluster_centers, MIN_DISTANCE_BETWEEN_CLUSTERS):
            cluster_centers.append((cx, cy))
    
    num_pores = random.randint(MIN_TOTAL_PORES, MAX_TOTAL_PORES)
    cluster_pore_positions = [[] for _ in range(num_clusters)]
    for _ in range(num_pores):
        cluster_idx = random.randint(0, num_clusters - 1)
        cx, cy = cluster_centers[cluster_idx]
        angle = random.randint(0, 180)
        w, h = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS), random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        x = cx + random.randint(-10, 10)
        y = cy + random.randint(-10, 10)
        
        if is_point_inside_polygon((x, y), polygon):
            pores.append((x, y, w, h, angle))
            cluster_pore_positions[cluster_idx].append((x, y, w, h))
    
    for cluster_pores in cluster_pore_positions:
        if not cluster_pores:
            continue
        xs, ys, ws, hs = zip(*cluster_pores)
        bx, by, bw, bh = convert_to_yolo_bbox(sum(xs)/len(xs), sum(ys)/len(ys), max(ws), max(hs), img_shape[1], img_shape[0])
        labels.append((PORE_NEST_CLASS_ID, bx, by, bw, bh))
    
    return pores, labels

# ----------------------- Main Function -----------------------
def visualize_class3_and_annotate(image_dir, annotation_dir, output_images_dir, output_labels_dir):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

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
                if int(line.strip().split()[0]) != 3:
                    continue
                polygon = list(map(float, line.strip().split()[1:]))
                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0])) for i in range(0, len(polygon), 2)]
                pores, labels = generate_balanced_pores_with_labels(points, image.shape)
                label_list.extend(labels)
                for (x, y, w, h, angle) in pores:
                    draw_pore(image, x, y, w, h, angle)  # Use the new function

        cv2.imwrite(os.path.join(output_images_dir, image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if label_list:
            save_yolo_labels(output_labels_dir, image_name, label_list)

# ----------------------- Example -----------------------
# Example usage
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/porefinal"
output_labels_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/poreannotations"
visualize_class3_and_annotate(image_dir, annotation_dir, output_images_dir, output_labels_dir)