import os
import cv2
import numpy as np
import random
import math

# ----------------------- Configuration -----------------------
CLASS_3_COLOR = (64, 64, 64)
CIRCLE_THICKNESS = 1
MIN_PORE_RADIUS = 1
MAX_PORE_RADIUS = 5
MIN_TOTAL_PORES = 15
MAX_TOTAL_PORES = 30
MIN_CLUSTERS = 2
MAX_CLUSTERS = 3
MIN_DISTANCE_BETWEEN_CLUSTER_PORES = 8
MIN_DISTANCE_BETWEEN_SCATTERED_PORES = 5
MIN_DISTANCE_BETWEEN_CLUSTERS = 40
CLUSTER_PADDING = 10
PORE_PADDING = 5
PORE_CLASS_ID = 0
PORE_NEST_CLASS_ID = 1
STRICT_CIRCULARITY = 0.85
STRICT_ELONGATION = 1.35
MIN_DISTANCE_BETWEEN_PORES = 7
EDGE_MARGIN = 3

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

def is_valid_pore_shape(w, h, min_circularity=STRICT_CIRCULARITY, max_elongation=STRICT_ELONGATION):
    if w == 0 or h == 0:
        return False
    ratio = max(w, h) / min(w, h)
    area = np.pi * (w / 2) * (h / 2)
    perimeter = 2 * np.pi * np.sqrt((w/2)**2 + (h/2)**2) / 2
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return ratio <= max_elongation and circularity >= min_circularity

def is_far_from_existing(x, y, r, placed_pores, min_dist=MIN_DISTANCE_BETWEEN_PORES):
    for px, py, pr in placed_pores:
        dist = math.hypot(x - px, y - py)
        if dist < (r + pr + min_dist):
            return False
    return True

def is_far_from_polygon_edge(x, y, polygon, margin):
    dist = cv2.pointPolygonTest(np.array(polygon, dtype=np.int32).reshape((-1, 1, 2)), (x, y), True)
    return dist is not None and dist >= margin

def draw_pore(image, x, y, w, h, angle):
    scale = 6
    img_h, img_w = image.shape[:2]
    up_w, up_h = img_w * scale, img_h * scale

    mask = np.ones((up_h, up_w, 3), dtype=np.uint8) * 255
    cx, cy = x * scale, y * scale
    rw, rh = max(1, w * scale), max(1, h * scale)

    center = (int(cx), int(cy))
    axes = (int(rw), int(rh))
    cv2.ellipse(mask, center, axes, angle, 0, 360, (0, 0, 0), -1, lineType=cv2.LINE_AA)

    mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=0.8, sigmaY=0.8)
    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_AREA)
    image[:] = cv2.subtract(image, 255 - mask)

"""def draw_pore(image, x, y, w, h, angle):
    scale = 6
    img_h, img_w = image.shape[:2]
    up_w, up_h = img_w * scale, img_h * scale
    mask = np.ones((up_h, up_w, 3), dtype=np.uint8) * 255
    cx, cy = x * scale, y * scale
    rw, rh = max(1, w * scale), max(1, h * scale)
    center = (int(cx), int(cy))
    axes = (int(rw), int(rh))
    cv2.ellipse(mask, center, axes, angle, 0, 360, (64, 64, 64), -1, lineType=cv2.LINE_AA)
    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_AREA)
    image[:] = cv2.min(image, mask)  # Solid burn-in, no transparency
"""
# ----------------------- Pore and Cluster Generation -----------------------
def generate_balanced_pores_with_labels(polygon, img_shape):
    pores, labels = [], []
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    x_min, y_min = np.min(polygon_np, axis=0)[0]
    x_max, y_max = np.max(polygon_np, axis=0)[0]
    max_attempts = 200
    num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)

    cluster_centers = []
    while len(cluster_centers) < num_clusters:
        cx, cy = random.randint(x_min, x_max), random.randint(y_min, y_max)
        if (is_point_inside_polygon((cx, cy), polygon_np) and 
            are_clusters_far_enough((cx, cy), cluster_centers, MIN_DISTANCE_BETWEEN_CLUSTERS) and 
            is_far_from_polygon_edge(cx, cy, polygon, EDGE_MARGIN)):
            cluster_centers.append((cx, cy))

    target_num_pores = random.randint(MIN_TOTAL_PORES, MAX_TOTAL_PORES)
    cluster_pore_target = target_num_pores // 2
    scatter_pore_target = target_num_pores - cluster_pore_target

    placed_pores = []

    def try_place_pore(x, y, w, h, angle):
        r = max(w, h)
        if not is_point_inside_polygon((x, y), polygon_np):
            return False
        if not is_far_from_polygon_edge(x, y, polygon, EDGE_MARGIN):
            return False
        if is_valid_pore_shape(w, h) and is_far_from_existing(x, y, r, placed_pores):
            pores.append((x, y, w, h, angle))
            placed_pores.append((x, y, r))
            return True
        return False

    attempts = 0
    while len(pores) < cluster_pore_target and attempts < cluster_pore_target * max_attempts:
        attempts += 1
        cx, cy = random.choice(cluster_centers)
        angle_deg = random.uniform(0, 360)
        angle_rad = np.deg2rad(angle_deg)
        placement_radius = random.uniform(5, 20)
        x = int(cx + placement_radius * np.cos(angle_rad))
        y = int(cy + placement_radius * np.sin(angle_rad))
        w = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        h = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        angle = random.randint(0, 180)
        try_place_pore(x, y, w, h, angle)

    attempts = 0
    while len(pores) < target_num_pores and attempts < 2 * target_num_pores * max_attempts:
        attempts += 1
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        w = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        h = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        angle = random.randint(0, 180)
        if try_place_pore(x, y, w, h, angle):
            padded_w, padded_h = w + PORE_PADDING, h + PORE_PADDING
            bx, by, bw, bh = convert_to_yolo_bbox(x, y, padded_w, padded_h, img_shape[1], img_shape[0])
            labels.append((PORE_CLASS_ID, bx, by, bw, bh))

    for cx, cy in cluster_centers:
        cluster_related = [p for p in pores if math.hypot(p[0] - cx, p[1] - cy) < 25]
        if not cluster_related:
            continue
        xs, ys, ws, hs = zip(*[(x, y, w, h) for (x, y, w, h, _) in cluster_related])
        min_x = max(0, min(xs) - max(ws) - CLUSTER_PADDING)
        max_x = min(img_shape[1], max(xs) + max(ws) + CLUSTER_PADDING)
        min_y = max(0, min(ys) - max(hs) - CLUSTER_PADDING)
        max_y = min(img_shape[0], max(ys) + max(hs) + CLUSTER_PADDING)
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        cluster_w, cluster_h = max_x - min_x, max_y - min_y
        bx, by, bw, bh = convert_to_yolo_bbox(cx, cy, cluster_w / 2, cluster_h / 2, img_shape[1], img_shape[0])
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
                print(f"{image_name} → Generated {len(pores)} pores")
                label_list.extend(labels)
                for (x, y, w, h, angle) in pores:
                    draw_pore(image, x, y, w, h, angle)

        cv2.imwrite(os.path.join(output_images_dir, image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if label_list:
            save_yolo_labels(output_labels_dir, image_name, label_list)

# ----------------------- Execution -----------------------
dirs = {
    "image_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images",
    "annotation_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8",
    "output_images_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image",
    "output_labels_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"
}

if __name__ == '__main__':
    visualize_class3_and_annotate(**dirs)