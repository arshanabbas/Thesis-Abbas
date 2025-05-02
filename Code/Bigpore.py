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

# -----------------------
# Pore Drawing Functions
# -----------------------

def draw_elliptical_pore(image, x, y, short_axis, angle):
    """
    Draw an elongated elliptical pore with realistic shading and soft edges.
    """
    elongation_ratio = random.uniform(2.0, 3.0)
    long_axis = int(short_axis * elongation_ratio)
    axes = (long_axis, short_axis)

    h, w = image.shape[:2]
    temp = np.zeros((h, w, 4), dtype=np.uint8)

    # Draw soft-edged ellipse on temp layer (RGBA)
    cv2.ellipse(temp, (x, y), axes, angle, 0, 360, (40, 40, 40, 255), -1, lineType=cv2.LINE_AA)

    # Add Gaussian blur to alpha channel
    temp[:, :, 3] = cv2.GaussianBlur(temp[:, :, 3], (9, 9), sigmaX=3)

    # Alpha blend into image
    alpha = temp[..., 3:] / 255.0
    rgb = temp[..., :3]
    for c in range(3):
        image[..., c] = (alpha[..., 0] * rgb[..., c] + (1 - alpha[..., 0]) * image[..., c]).astype(np.uint8)

def draw_triangular_pore(image, center, size, angle):
    triangle = np.array([
        [0, -size],
        [-size, size],
        [size, size]
    ], dtype=np.float32)
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = triangle @ rot.T + np.array(center)
    mask = np.zeros_like(image[:, :, 0])
    cv2.fillConvexPoly(mask, rotated.astype(np.int32), color=255)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    for c in range(3):
        image[..., c] = np.where(mask > 0, (image[..., c] * (1 - mask / 255) + 40 * (mask / 255)).astype(np.uint8), image[..., c])

def draw_comet_pore(image, center, radius, angle):
    h, w = image.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.ellipse(overlay, center, (radius, radius), 0, 0, 360, (50, 50, 50, 255), -1, lineType=cv2.LINE_AA)
    tail_len = radius * 3
    tail_center = (
        int(center[0] + tail_len * math.cos(math.radians(angle))),
        int(center[1] + tail_len * math.sin(math.radians(angle)))
    )
    cv2.ellipse(overlay, tail_center, (tail_len, radius), angle, 0, 360, (40, 40, 40, 100), -1, lineType=cv2.LINE_AA)
    rgb, alpha = overlay[..., :3], overlay[..., 3:] / 255.0
    for c in range(3):
        image[..., c] = (alpha[..., 0] * rgb[..., c] + (1 - alpha[..., 0]) * image[..., c]).astype(np.uint8)

# -----------------------
# Pore Generator
# -----------------------

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

        # 1. Two elliptical pores
        # 1. Two elliptical pores (stretched)
        for _ in range(2):
            x, y = place_custom_pore(mask, margin_mask, placed_pores, image.shape)
            if x is not None:
                short_axis = random.randint(2, 3)
                angle = random.randint(0, 180)
                draw_elliptical_pore(image, x, y, short_axis, angle)

        # Estimate max elongated bounding box
                bbox_w = int(short_axis * 3.5) + PORE_PADDING
                bbox_h = int(short_axis * 1.5) + PORE_PADDING
                bx, by, bw, bh = convert_to_yolo_bbox(x, y, bbox_w, bbox_h, image.shape[1], image.shape[0])
                yolo_labels.append((PORE_CLASS_ID, bx, by, bw, bh))


        # 2. One triangular pore
        x, y = place_custom_pore(mask, margin_mask, placed_pores, image.shape)
        if x is not None:
            size = random.randint(10, 16)
            angle = random.randint(0, 360)
            draw_triangular_pore(image, (x, y), size, angle)
            bx, by, bw, bh = convert_to_yolo_bbox(x, y, size + PORE_PADDING, size + PORE_PADDING, image.shape[1], image.shape[0])
            yolo_labels.append((PORE_CLASS_ID, bx, by, bw, bh))

        # 3. One comet-shaped pore
        x, y = place_custom_pore(mask, margin_mask, placed_pores, image.shape)
        if x is not None:
            radius = random.randint(4, 6)
            angle = random.randint(0, 360)
            draw_comet_pore(image, (x, y), radius, angle)
            bx, by, bw, bh = convert_to_yolo_bbox(x, y, radius + PORE_PADDING, radius + PORE_PADDING, image.shape[1], image.shape[0])
            yolo_labels.append((PORE_CLASS_ID, bx, by, bw, bh))

        # Save outputs
        output_img_path = os.path.join(dirs["output_images_dir"], filename)
        output_lbl_path = os.path.join(dirs["output_labels_dir"], base_name + ".txt")
        cv2.imwrite(output_img_path, image)

        with open(output_lbl_path, "w") as f:
            for label in yolo_labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

        print(f"✅ Placed 4 pores in {filename} — saved to output")

if __name__ == "__main__":
    run_pipeline()
