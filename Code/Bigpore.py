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
# Crescent Pore Drawing Function
# -----------------------

def draw_asymmetric_crescent_pore(image, x, y, scale=1.0, angle=0):
    """
    Draws a crescent-shaped pore:
    - Sharp northern edge
    - Smooth southern curve
    - Solid and clean, no blur
    """
    # Define top (north) side — angular, structured
    top_pts = np.array([
        (-30, 5),   # far left
        (-15, -8),  # rise
        (0, -12),   # sharp peak
        (18, -6),   # drop
        (25, 4),    # near lower right
    ], dtype=np.float32)

    # Bottom (south) side — gentle arc
    bottom_pts = []
    for t in np.linspace(0, 1, 10):
        xt = (1 - t) * 25 + t * (-10)
        yt = 4 + 6 * np.sin(t * np.pi)  # gentle downward bulge
        bottom_pts.append((xt, yt))
    bottom_pts = np.array(bottom_pts, dtype=np.float32)

    # Combine points and apply scaling
    all_pts = np.vstack([top_pts, bottom_pts])
    all_pts *= scale

    # Rotate and translate to (x, y)
    theta = np.radians(angle)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    rotated = np.dot(all_pts, rot_matrix.T) + [x, y]
    contour = rotated.astype(np.int32).reshape((-1, 1, 2))

    # Draw filled pore
    cv2.fillPoly(image, [contour], color=(30, 30, 30), lineType=cv2.LINE_AA)

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

        # Draw 1–2 crescent pores per image
        for _ in range(2):
            x, y = place_custom_pore(mask, margin_mask, placed_pores, image.shape)
            if x is not None:
                scale = random.uniform(0.8, 1.2)
                angle = random.randint(0, 360)
                draw_asymmetric_crescent_pore(image, x, y, scale=scale, angle=angle)
                bbox_w = int(30 * scale) + PORE_PADDING
                bbox_h = int(15 * scale) + PORE_PADDING
                bx, by, bw, bh = convert_to_yolo_bbox(x, y, bbox_w, bbox_h, image.shape[1], image.shape[0])
                yolo_labels.append((PORE_CLASS_ID, bx, by, bw, bh))

        # Save outputs
        output_img_path = os.path.join(dirs["output_images_dir"], filename)
        output_lbl_path = os.path.join(dirs["output_labels_dir"], base_name + ".txt")
        cv2.imwrite(output_img_path, image)

        with open(output_lbl_path, "w") as f:
            for label in yolo_labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

        print(f"✅ Added crescent pores to {filename}")

if __name__ == "__main__":
    run_pipeline()
