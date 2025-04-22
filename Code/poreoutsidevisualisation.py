import os
import cv2
import numpy as np
import random
import math

# ----------------------- Configuration -----------------------
CLASS_3_COLOR = (64, 64, 64)
MIN_PORE_RADIUS = 1
MAX_PORE_RADIUS = 5
MIN_TOTAL_PORES = 15
MAX_TOTAL_PORES = 30
MIN_CLUSTERS = 2
MAX_CLUSTERS = 3
MIN_DISTANCE_BETWEEN_PORES = 7
MIN_DISTANCE_BETWEEN_CLUSTERS = 40
STRICT_CIRCULARITY = 0.85
STRICT_ELONGATION = 1.35
BOUNDARY_MARGIN = 3
PORE_CLASS_ID = 0
PORE_NEST_CLASS_ID = 1
PORE_PADDING = 5
CLUSTER_BOX_PADDING = 10


# ----------------------- Helper Functions -----------------------
def is_valid_pore_shape(w, h):
    if w == 0 or h == 0:
        return False
    ratio = max(w, h) / min(w, h)
    area = np.pi * (w / 2) * (h / 2)
    perimeter = 2 * np.pi * np.sqrt((w / 2) ** 2 + (h / 2) ** 2) / 2
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return ratio <= STRICT_ELONGATION and circularity >= STRICT_CIRCULARITY

def convert_to_yolo_bbox(x, y, w, h, image_w, image_h):
    return x / image_w, y / image_h, (2 * w) / image_w, (2 * h) / image_h

def save_yolo_labels(output_labels_dir, image_name, labels):
    label_file = os.path.join(output_labels_dir, os.path.splitext(image_name)[0] + ".txt")
    with open(label_file, "w") as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

def is_far_from_existing(x, y, r, placed_pores):
    for px, py, pr in placed_pores:
        if math.hypot(x - px, y - py) < (r + pr + MIN_DISTANCE_BETWEEN_PORES):
            return False
    return True

def draw_pore(image, x, y, w, h, angle, base_arc_start):
    scale = 6
    img_h, img_w = image.shape[:2]
    up_w, up_h = img_w * scale, img_h * scale
    cx, cy = int(x * scale), int(y * scale)
    rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
    center = (cx, cy)

    core_radius = max(w, h)
    if core_radius <= 2:
        thick_start, thick_end = 2 * scale, 1 * scale
    elif core_radius == 3:
        thick_start, thick_end = int(2.5 * scale), 1 * scale
    else:
        thick_start, thick_end = 3 * scale, 1 * scale

    relative_y = y / img_h
    max_dev = 0.2
    jitter = int((random.uniform(-1, 1) * max_dev) * 360 * relative_y)
    arc_start = (base_arc_start + jitter) % 360

    arc_length = random.randint(180, 270) if core_radius > 2 else 240
    arc_end = arc_start + arc_length
    fade_segments = 6
    angle_step = arc_length // fade_segments

    base = np.zeros((up_h, up_w, 4), dtype=np.uint8)
    for i in range(fade_segments):
        opacity = int(np.interp(i, [0, fade_segments - 1], [255, 50]))
        thickness = int(np.interp(i, [0, fade_segments - 1], [thick_start, thick_end]))
        axes = (rw + int(thickness * 0.75), rh + int(thickness * 0.75))
        color = (180, 180, 180, opacity)
        sa = arc_start + i * angle_step
        ea = sa + angle_step
        cv2.ellipse(base, center, axes, angle, sa, ea, color, thickness=thickness, lineType=cv2.LINE_AA)

    base[:, :, 3] = cv2.GaussianBlur(base[:, :, 3], (5, 5), sigmaX=2.0)
    cv2.ellipse(base, center, (rw, rh), angle, 0, 360, (45, 45, 45, 255), -1, lineType=cv2.LINE_AA)

    final = cv2.resize(base, (img_w, img_h), interpolation=cv2.INTER_AREA)
    rgb, alpha = final[..., :3], final[..., 3:] / 255.0
    for c in range(3):
        image[..., c] = (alpha[..., 0] * rgb[..., c] + (1 - alpha[..., 0]) * image[..., c]).astype(np.uint8)

def group_pores_into_clusters(pores, threshold=25):
    from sklearn.cluster import DBSCAN
    coords = np.array([[x, y] for x, y, *_ in pores])
    clustering = DBSCAN(eps=threshold, min_samples=3).fit(coords)
    clustered = {}
    for label in set(clustering.labels_):
        if label == -1:
            continue
        clustered[label] = [pores[i] for i in range(len(pores)) if clustering.labels_[i] == label]
    return clustered

# ----------------------- Pore Generator -----------------------
def generate_balanced_pores_with_labels(polygon, img_shape):
    pores, labels = [], []
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    x_min, y_min = np.min(polygon_np, axis=0)[0]
    x_max, y_max = np.max(polygon_np, axis=0)[0]

    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_np], 255)
    margin_mask = cv2.erode(mask, np.ones((2 * BOUNDARY_MARGIN + 1, 2 * BOUNDARY_MARGIN + 1), np.uint8))

    total_pores = random.randint(MIN_TOTAL_PORES, MAX_TOTAL_PORES)
    num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)
    pores_per_cluster = 3
    clustered_pores_needed = num_clusters * pores_per_cluster
    scattered_pores_needed = total_pores - clustered_pores_needed

    placed_pores = []
    cluster_centers = []

    # Step 1: Place cluster centers
    while len(cluster_centers) < num_clusters:
        cx, cy = random.randint(x_min, x_max), random.randint(y_min, y_max)
        if mask[cy, cx] == 255 and margin_mask[cy, cx] == 255:
            too_close = any(math.hypot(cx - pcx, cy - pcy) < MIN_DISTANCE_BETWEEN_CLUSTERS for pcx, pcy in cluster_centers)
            if not too_close:
                cluster_centers.append((cx, cy))

    cluster_bounding_boxes = []

    # Step 2: Generate clustered pores
    for cx, cy in cluster_centers:
        cluster_pores = []
        attempts = 0
        while len(cluster_pores) < pores_per_cluster and attempts < 100:
            angle_deg = random.uniform(0, 360)
            radius = random.uniform(5, 15)
            x = int(cx + radius * math.cos(math.radians(angle_deg)))
            y = int(cy + radius * math.sin(math.radians(angle_deg)))
            w = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
            h = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
            angle = random.randint(0, 180)
            r = max(w, h)

            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                if mask[y, x] == 255 and margin_mask[y, x] == 255:
                    if is_valid_pore_shape(w, h) and is_far_from_existing(x, y, r, placed_pores):
                        cluster_pores.append((x, y, w, h, angle))
                        placed_pores.append((x, y, r))
            attempts += 1

        pores.extend(cluster_pores)

        if cluster_pores:
            xs, ys = zip(*[(px, py) for (px, py, *_rest) in cluster_pores])
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            # Apply generous cluster padding
            xmin -= CLUSTER_BOX_PADDING
            xmax += CLUSTER_BOX_PADDING
            ymin -= CLUSTER_BOX_PADDING
            ymax += CLUSTER_BOX_PADDING

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_shape[1], xmax)
            ymax = min(img_shape[0], ymax)

            cx_box = (xmin + xmax) / 2
            cy_box = (ymin + ymax) / 2
            box_w = (xmax - xmin) / 2
            box_h = (ymax - ymin) / 2

            labels.append((PORE_NEST_CLASS_ID, *convert_to_yolo_bbox(cx_box, cy_box, box_w, box_h, img_shape[1], img_shape[0])))
            cluster_bounding_boxes.append((xmin, ymin, xmax, ymax))

    # Step 3: Generate scattered pores
    attempts = 0
    while scattered_pores_needed > 0 and attempts < 1000:
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        w = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        h = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        angle = random.randint(0, 180)
        r = max(w, h)

        if mask[y, x] == 255 and margin_mask[y, x] == 255:
            inside_cluster = any(xmin <= x <= xmax and ymin <= y <= ymax for (xmin, ymin, xmax, ymax) in cluster_bounding_boxes)
            if not inside_cluster and is_valid_pore_shape(w, h) and is_far_from_existing(x, y, r, placed_pores):
                pores.append((x, y, w, h, angle))
                placed_pores.append((x, y, r))
                bx, by, bw, bh = convert_to_yolo_bbox(x, y, w + PORE_PADDING, h + PORE_PADDING, img_shape[1], img_shape[0])
                labels.append((PORE_CLASS_ID, bx, by, bw, bh))
                scattered_pores_needed -= 1
        attempts += 1

    return pores, labels, set()

# ----------------------- Main Execution -----------------------
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
                pores, labels, clustered_ids = generate_balanced_pores_with_labels(points, image.shape)
                print(f"{image_name} → Generated {len(pores)} pores")
                label_list.extend(labels)
                base_arc_start = random.randint(0, 359)
                for p in pores:
                    if id(p) not in clustered_ids:
                        draw_pore(image, *p, base_arc_start)
        cv2.imwrite(os.path.join(output_images_dir, image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if label_list:
            save_yolo_labels(output_labels_dir, image_name, label_list)

# ----------------------- Paths -----------------------
dirs = {
    "image_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images",
    "annotation_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8",
    "output_images_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image",
    "output_labels_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"
}

if __name__ == '__main__':
    visualize_class3_and_annotate(**dirs)
