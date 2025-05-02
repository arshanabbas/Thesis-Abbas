import os
import cv2
import numpy as np
import math
import random

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
# Configuration
# -----------------------
MIN_PORE_RADIUS = 2
MAX_PORE_RADIUS = 6
MIN_DISTANCE_BETWEEN_PORES = 15
MIN_TOTAL_SINGULAR_PORES = 15
MAX_TOTAL_SINGULAR_PORES = 30
PORE_CLASS_ID = 0
PORE_PADDING = 6
BOUNDARY_MARGIN = 4

# -----------------------
# Helper Functions
# -----------------------

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def parse_yolo_polygon_annotation(annotation_path, image_shape):
    h, w = image_shape[:2]
    polygons = []

    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            class_id = int(parts[0])
            if class_id != 3:
                continue

            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue

            points = [(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])]
            polygons.append(np.array(points, dtype=np.int32))

    return polygons

def polygon_to_mask(image_shape, polygons):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, color=255)
    return mask

def is_far_from_existing(x, y, r, placed_pores):
    for px, py, pr in placed_pores:
        if math.hypot(x - px, y - py) < (r + pr + MIN_DISTANCE_BETWEEN_PORES):
            return False
    return True

def convert_to_yolo_bbox(x, y, w, h, image_w, image_h):
    return x / image_w, y / image_h, (2 * w) / image_w, (2 * h) / image_h

def generate_single_pore(mask, margin_mask, placed_pores, image_shape):
    h, w = image_shape[:2]
    attempts = 0
    max_attempts = 1000

    while attempts < max_attempts:
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        if margin_mask[y, x] != 255:
            attempts += 1
            continue

        rw = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        rh = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        r = max(rw, rh)
        angle = random.randint(0, 180)

        if is_far_from_existing(x, y, r, placed_pores):
            placed_pores.append((x, y, r))
            return (x, y, rw, rh, angle)

        attempts += 1

    return None

def draw_pore(image, x, y, rw, rh, angle):
    cv2.ellipse(image, (x, y), (rw, rh), angle, 0, 360, (50, 50, 50), -1, lineType=cv2.LINE_AA)

# -----------------------
# Main Pipeline
# -----------------------

def run_pipeline():
    for filename in os.listdir(dirs["image_dir"]):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(dirs["image_dir"], filename)
        annotation_path = os.path.join(dirs["annotation_dir"], base_name + ".txt")

        if not os.path.exists(annotation_path):
            print(f"⚠️ Skipping {filename} (no annotation found)")
            continue

        image = load_image(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            continue

        polygons = parse_yolo_polygon_annotation(annotation_path, image.shape)
        mask = polygon_to_mask(image.shape, polygons)

        kernel_size = 2 * BOUNDARY_MARGIN + 1
        margin_mask = cv2.erode(mask, np.ones((kernel_size, kernel_size), np.uint8))

        num_pores = random.randint(MIN_TOTAL_SINGULAR_PORES, MAX_TOTAL_SINGULAR_PORES)
        placed_pores = []
        yolo_labels = []

        for _ in range(num_pores):
            pore = generate_single_pore(mask, margin_mask, placed_pores, image.shape)
            if pore:
                x, y, rw, rh, angle = pore
                draw_pore(image, x, y, rw, rh, angle)
                bx, by, bw, bh = convert_to_yolo_bbox(x, y, rw + PORE_PADDING, rh + PORE_PADDING, image.shape[1], image.shape[0])
                yolo_labels.append((PORE_CLASS_ID, bx, by, bw, bh))

        output_img_path = os.path.join(dirs["output_images_dir"], filename)
        output_lbl_path = os.path.join(dirs["output_labels_dir"], base_name + ".txt")
        cv2.imwrite(output_img_path, image)

        with open(output_lbl_path, "w") as f:
            for label in yolo_labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

        print(f"✅ Pores placed: {len(yolo_labels)} — {output_img_path}")

if __name__ == "__main__":
    run_pipeline()