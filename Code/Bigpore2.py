import os
import cv2
import numpy as np
import random
import math

# -----------------------
# Dataset Paths
# -----------------------
dirs = {
    "image_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images",
    "annotation_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8",
    "output_images_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image",
    "output_labels_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"
}

os.makedirs(dirs["output_images_dir"], exist_ok=True)
os.makedirs(dirs["output_labels_dir"], exist_ok=True)

# -----------------------
# Constants
# -----------------------
PORE_CLASS_ID = 0
BOUNDARY_MARGIN = 4
PORE_PADDING = 6
MIN_DISTANCE_BETWEEN_PORES = 15

# -----------------------
# Utilities
# -----------------------
def load_image(image_path):
    return cv2.imread(image_path)

def parse_yolo_polygon_annotation(annotation_path, image_shape):
    h, w = image_shape[:2]
    polygons = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) != 3:
                continue
            coords = list(map(float, parts[1:]))
            points = [(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])]
            polygons.append(np.array(points, dtype=np.int32))
    return polygons

def polygon_to_mask(image_shape, polygons):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
    return mask

def is_far_from_existing(x, y, r, placed_pores):
    for px, py, pr in placed_pores:
        if math.hypot(x - px, y - py) < (r + pr + MIN_DISTANCE_BETWEEN_PORES):
            return False
    return True

def convert_to_yolo_bbox(x, y, w, h, image_w, image_h):
    return x / image_w, y / image_h, (2 * w) / image_w, (2 * h) / image_h

# -----------------------
# Soft Triangle Drawing
# -----------------------
def draw_soft_triangle_pore(image, x, y, scale=0.5, angle=0):
    # Triangle points
    base_pts = np.array([
        [150, 50],
        [70, 220],
        [230, 220]
    ], dtype=np.float32)

    radius = 28
    arc_points = 30
    edge_curve_points = 20
    contour = []

    for i in range(len(base_pts)):
        p0 = base_pts[i - 1]
        p1 = base_pts[i]
        p2 = base_pts[(i + 1) % len(base_pts)]

        v1 = p0 - p1
        v2 = p2 - p1
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        angle_between = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) / 2
        offset = radius / np.tan(angle_between)
        p_start = p1 + v1 * offset
        p_end = p1 + v2 * offset

        arc = [(1 - t) * p_start + t * p_end for t in np.linspace(0, 1, arc_points)]
        edge_vec = p_start - p0
        edge = [p0 + t * edge_vec for t in np.linspace(0, 1, edge_curve_points)]

        contour.extend(edge)
        contour.extend(arc)

    # Rotate and scale
    contour = np.array(contour) - [150, 150]
    theta = np.radians(angle)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    transformed = np.dot(contour * scale, rot_matrix.T) + [x, y]
    contour = transformed.astype(np.int32).reshape((-1, 1, 2))

    # Draw
    cv2.fillPoly(image, [contour], color=(30, 30, 30), lineType=cv2.LINE_AA)

# -----------------------
# Pipeline
# -----------------------
def run_pipeline():
    for filename in os.listdir(dirs["image_dir"]):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(dirs["image_dir"], filename)
        annotation_path = os.path.join(dirs["annotation_dir"], base_name + ".txt")

        if not os.path.exists(annotation_path):
            print(f"⚠️ Skipping {filename} (no annotation)")
            continue

        image = load_image(image_path)
        if image is None:
            continue

        polygons = parse_yolo_polygon_annotation(annotation_path, image.shape)
        mask = polygon_to_mask(image.shape, polygons)
        margin_mask = cv2.erode(mask, np.ones((2 * BOUNDARY_MARGIN + 1, 2 * BOUNDARY_MARGIN + 1), np.uint8))

        placed_pores = []
        yolo_labels = []
        used_configs = set()

        for _ in range(2):  # Two triangle pores
            for _ in range(1000):
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                if margin_mask[y, x] != 255 or not is_far_from_existing(x, y, 10, placed_pores):
                    continue

                angle = random.randint(0, 360)
                scale = random.uniform(0.45, 0.6)
                config = (round(scale, 2), angle // 10)
                if config in used_configs:
                    continue
                used_configs.add(config)

                draw_soft_triangle_pore(image, x, y, scale=scale, angle=angle)
                bw, bh = int(30 * scale) + PORE_PADDING, int(30 * scale) + PORE_PADDING
                bbox = convert_to_yolo_bbox(x, y, bw, bh, image.shape[1], image.shape[0])
                yolo_labels.append((PORE_CLASS_ID, *bbox))
                placed_pores.append((x, y, 10))
                break

        # Save results
        cv2.imwrite(os.path.join(dirs["output_images_dir"], filename), image)
        with open(os.path.join(dirs["output_labels_dir"], base_name + ".txt"), "w") as f:
            for label in yolo_labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

        print(f"✅ Processed {filename}")

# -----------------------
# Run it
# -----------------------
if __name__ == "__main__":
    run_pipeline()