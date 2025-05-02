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
PORE_CLASS_ID = 0
BOUNDARY_MARGIN = 4
MIN_DISTANCE_BETWEEN_PORES = 15
PORE_PADDING = 6

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

def place_custom_pore(mask, margin_mask, placed_pores, image_shape):
    h, w = image_shape[:2]
    max_attempts = 1000
    for _ in range(max_attempts):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        if margin_mask[y, x] != 255:
            continue
        r = 10
        if is_far_from_existing(x, y, r, placed_pores):
            placed_pores.append((x, y, r))
            return x, y
    return None, None

# -----------------------
# Pore Drawing Functions
# -----------------------

def draw_elliptical_pore(image, x, y, short_axis, angle):
    elongation_ratio = random.uniform(2.5, 3.5)
    long_axis = int(short_axis * elongation_ratio)
    axes = (long_axis, short_axis)
    cv2.ellipse(image, (x, y), axes, angle, 0, 360, (30, 30, 30), -1, lineType=cv2.LINE_AA)

def draw_crescent_pore(image, x, y, short_axis, angle):
    """
    Crescent pore with high elongation and slim body.
    One sharp end, one open side.
    """
    elongation_ratio = random.uniform(4.0, 5.0)  # more stretched
    width = int(short_axis * elongation_ratio)
    height = int(short_axis * 0.8)  # slimmer

    # Key control points for crescent
    top = (x - width // 2, y)
    bottom = (x + width // 2, y)
    tip = (x, y - height)
    tail = (x, y + height // 3)

    pts = np.array([
        top,
        (x - width // 4, y - height // 3),
        tip,
        (x + width // 4, y - height // 4),
        bottom,
        tail
    ], dtype=np.int32).reshape((-1, 1, 2))

    # Rotate the entire shape
    rot_rad = np.radians(angle)
    rot_matrix = np.array([
        [np.cos(rot_rad), -np.sin(rot_rad)],
        [np.sin(rot_rad),  np.cos(rot_rad)]
    ])
    pts = np.dot(pts.reshape(-1, 2) - [x, y], rot_matrix.T) + [x, y]
    pts = pts.astype(np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(image, [pts], color=(30, 30, 30), lineType=cv2.LINE_AA)

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

        placed_pores = []
        yolo_labels = []

        # First: Crescent-shaped pore
        short_axis = random.randint(4, 6)
        angle = random.randint(0, 180)
        x, y = place_custom_pore(mask, margin_mask, placed_pores, image.shape)
        if x is not None:
            draw_crescent_pore(image, x, y, short_axis, angle)
            bbox_w = int(short_axis * 3.5) + PORE_PADDING
            bbox_h = int(short_axis * 1.5) + PORE_PADDING
            bx, by, bw, bh = convert_to_yolo_bbox(x, y, bbox_w, bbox_h, image.shape[1], image.shape[0])
            yolo_labels.append((PORE_CLASS_ID, bx, by, bw, bh))

        # Second: Sharp elliptical pore
        short_axis = random.randint(5, 6)
        angle = random.randint(0, 180)
        x, y = place_custom_pore(mask, margin_mask, placed_pores, image.shape)
        if x is not None:
            draw_elliptical_pore(image, x, y, short_axis, angle)
            bbox_w = int(short_axis * 3.5) + PORE_PADDING
            bbox_h = int(short_axis * 1.5) + PORE_PADDING
            bx, by, bw, bh = convert_to_yolo_bbox(x, y, bbox_w, bbox_h, image.shape[1], image.shape[0])
            yolo_labels.append((PORE_CLASS_ID, bx, by, bw, bh))

        # Save outputs
        output_img_path = os.path.join(dirs["output_images_dir"], filename)
        output_lbl_path = os.path.join(dirs["output_labels_dir"], base_name + ".txt")
        cv2.imwrite(output_img_path, image)

        with open(output_lbl_path, "w") as f:
            for label in yolo_labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

        print(f"✅ Added 2 pores to {filename} — saved")

if __name__ == "__main__":
    run_pipeline()
