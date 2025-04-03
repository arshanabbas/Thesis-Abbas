import os
import cv2
import numpy as np
import random

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
    w = max(2, w)
    h = max(2, h)
    if random.random() < 0.5:
        h = w  # make circular sometimes for natural variation

    # Core: sharp, dark center
    core_mask = np.zeros_like(image, dtype=np.uint8)
    core_color = (50, 50, 50)
    cv2.ellipse(core_mask, (x, y), (w//2, h//2), angle, 0, 360, core_color, -1)

    # Halo: soft Gaussian for feathered look
    halo_size = max(w, h) * 2
    blob = np.zeros((halo_size, halo_size), dtype=np.uint8)
    cv2.circle(blob, (halo_size//2, halo_size//2), min(w, h), 60, -1)
    blob = cv2.GaussianBlur(blob, (7, 7), 0.8)
    blob = np.clip(blob * 1.5, 0, 255).astype(np.uint8)  # boost contrast
    blob = cv2.merge([blob, blob, blob])

    top = max(0, y - halo_size//2)
    left = max(0, x - halo_size//2)
    bottom = min(image.shape[0], y + halo_size//2)
    right = min(image.shape[1], x + halo_size//2)

    blob_crop = blob[0:(bottom - top), 0:(right - left)]
    roi = image[top:bottom, left:right]

    # Subtract halo first, then overlay sharp core
    image[top:bottom, left:right] = cv2.subtract(roi, blob_crop)
    image[:] = cv2.subtract(image, core_mask)

# ----------------------- Pore and Cluster Generation -----------------------
def generate_balanced_pores_with_labels(polygon, img_shape):
    pores, labels = [], []
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    x_min, y_min = np.min(polygon, axis=0)[0]
    x_max, y_max = np.max(polygon, axis=0)[0]
    max_attempts = 200
    num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)

    cluster_centers = []
    while len(cluster_centers) < num_clusters:
        cx, cy = random.randint(x_min, x_max), random.randint(y_min, y_max)
        if is_point_inside_polygon((cx, cy), polygon) and are_clusters_far_enough((cx, cy), cluster_centers, MIN_DISTANCE_BETWEEN_CLUSTERS):
            cluster_centers.append((cx, cy))

    num_pores = max(MIN_TOTAL_PORES, min(MAX_TOTAL_PORES, int(cv2.contourArea(polygon) // 50)))
    cluster_pore_count = max(5, min(random.randint(5, 10) * num_clusters, num_pores - 5))
    scattered_pore_count = max(5, num_pores - cluster_pore_count)

    def is_far_enough(nx, ny, nr, existing, min_dist):
        for (x, y, w, h, _) in existing:
            if np.linalg.norm(np.array([nx, ny]) - np.array([x, y])) < (max(w, h) + nr + min_dist):
                return False
        return True

    cluster_pore_positions = [[] for _ in range(num_clusters)]
    cluster_success, total_cluster_attempts = 0, 0
    max_total_cluster_attempts = cluster_pore_count * max_attempts

    while cluster_success < cluster_pore_count and total_cluster_attempts < max_total_cluster_attempts:
        total_cluster_attempts += 1
        chosen_cluster_idx = random.randint(0, num_clusters - 1)
        cx, cy = cluster_centers[chosen_cluster_idx]
        angle_deg = random.uniform(0, 360)
        angle_rad = np.deg2rad(angle_deg)
        placement_radius = random.uniform(5, 20)
        x = int(cx + placement_radius * np.cos(angle_rad))
        y = int(cy + placement_radius * np.sin(angle_rad))

        w, h = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS), random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        pore_angle = random.randint(0, 180)

        if is_point_inside_polygon((x, y), polygon) and is_far_enough(x, y, max(w, h), pores, MIN_DISTANCE_BETWEEN_CLUSTER_PORES):
            pores.append((x, y, w, h, pore_angle))
            cluster_pore_positions[chosen_cluster_idx].append((x, y, w, h))
            cluster_success += 1

    for cluster_pores in cluster_pore_positions:
        if not cluster_pores:
            continue
        xs, ys, ws, hs = zip(*cluster_pores)
        min_x = max(0, min(xs) - max(ws) - CLUSTER_PADDING)
        max_x = min(img_shape[1], max(xs) + max(ws) + CLUSTER_PADDING)
        min_y = max(0, min(ys) - max(hs) - CLUSTER_PADDING)
        max_y = min(img_shape[0], max(ys) + max(hs) + CLUSTER_PADDING)
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        cluster_w, cluster_h = max_x - min_x, max_y - min_y
        bx, by, bw, bh = convert_to_yolo_bbox(cx, cy, cluster_w / 2, cluster_h / 2, img_shape[1], img_shape[0])
        labels.append((PORE_NEST_CLASS_ID, bx, by, bw, bh))

    scatter_success, total_scatter_attempts = 0, 0
    max_total_scatter_attempts = scattered_pore_count * max_attempts

    while scatter_success < scattered_pore_count and total_scatter_attempts < max_total_scatter_attempts:
        total_scatter_attempts += 1
        x, y = random.randint(x_min, x_max), random.randint(y_min, y_max)
        w, h, angle = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS), random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS), random.randint(0, 180)

        if is_point_inside_polygon((x, y), polygon) and is_far_enough(x, y, max(w, h), pores, MIN_DISTANCE_BETWEEN_SCATTERED_PORES + MAX_PORE_RADIUS):
            pores.append((x, y, w, h, angle))
            padded_w = w + PORE_PADDING
            padded_h = h + PORE_PADDING
            bx, by, bw, bh = convert_to_yolo_bbox(x, y, padded_w, padded_h, img_shape[1], img_shape[0])
            labels.append((PORE_CLASS_ID, bx, by, bw, bh))
            scatter_success += 1

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

# ----------------------- Test Visualization -----------------------
if __name__ == '__main__':
    test_img = np.ones((256, 256, 3), dtype=np.uint8) * 200
    for _ in range(20):
        x = random.randint(20, 236)
        y = random.randint(20, 236)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        angle = random.randint(0, 180)
        draw_pore(test_img, x, y, w, h, angle)
        cv2.circle(test_img, (x, y), 1, (255, 0, 0), 1)
    cv2.imwrite("test_output.jpg", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    print("Generated test_output.jpg with visible pores.")

# ----------------------- Example -----------------------
dirs = {
    "image_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images",
    "annotation_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8",
    "output_images_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image",
    "output_labels_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"
}
visualize_class3_and_annotate(**dirs)