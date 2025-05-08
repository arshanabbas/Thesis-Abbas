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

# -----------------------
# Constants
# -----------------------
PORE_CLASS_ID = 0
BOUNDARY_MARGIN = 4
PORE_PADDING = 6
MIN_DISTANCE_BETWEEN_PORES = 20

os.makedirs(dirs["output_images_dir"], exist_ok=True)
os.makedirs(dirs["output_labels_dir"], exist_ok=True)

# -----------------------
# Utility Functions
# -----------------------
def load_image(path):
    return cv2.imread(path)

def parse_polygons(path, shape):
    h, w = shape[:2]
    polys = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) != 3:
                continue
            coords = list(map(float, parts[1:]))
            pts = [(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])]
            polys.append(np.array(pts, dtype=np.int32))
    return polys

def polygon_to_mask(shape, polys):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if polys:
        cv2.fillPoly(mask, polys, 255)
    return mask

def is_far(x, y, r, placed):
    for px, py, pr in placed:
        if math.hypot(x - px, y - py) < (r + pr + MIN_DISTANCE_BETWEEN_PORES):
            return False
    return True

def convert_to_yolo_bbox(x, y, w, h, iw, ih):
    return x / iw, y / ih, (2 * w) / iw, (2 * h) / ih

# -----------------------
# Bezier-Based Triangle Pore
# -----------------------
def bezier_interp(p0, p1, p2, steps=30):
    return [(1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2 for t in np.linspace(0, 1, steps)]

def draw_bezier_triangle(image, x, y, scale=0.5, angle_deg=0):
    base_pts = np.array([
        [150, 50],
        [70, 220],
        [230, 220]
    ], dtype=np.float32)

    control_offset = 40
    curve_steps = 20
    contour = []

    for i in range(3):
        p0 = base_pts[i]
        p1 = base_pts[(i + 1) % 3]
        p2 = base_pts[(i + 2) % 3]

        d01 = p1 - p0
        d12 = p2 - p1
        d01 /= np.linalg.norm(d01)
        d12 /= np.linalg.norm(d12)

        edge_start = p1 - d01 * control_offset
        edge_end = p1 + d12 * control_offset

        bezier_curve = bezier_interp(edge_start, p1, edge_end, steps=curve_steps)
        contour.extend(bezier_curve)

    contour = np.array(contour, dtype=np.float32)

    # Center, scale, rotate
    center = np.mean(contour, axis=0)
    contour = (contour - center) * scale

    theta = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    contour = np.dot(contour, rot_matrix.T)

    contour += np.array([x, y])
    contour = contour.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [contour], color=(30, 30, 30), lineType=cv2.LINE_AA)

# -----------------------
# Main Dataset Pipeline
# -----------------------
def run_pipeline():
    for fname in os.listdir(dirs["image_dir"]):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue

        base = os.path.splitext(fname)[0]
        image_path = os.path.join(dirs["image_dir"], fname)
        annot_path = os.path.join(dirs["annotation_dir"], base + ".txt")

        if not os.path.exists(annot_path):
            print(f"⚠️ Skipping {fname} (no annotation)")
            continue

        image = load_image(image_path)
        if image is None:
            continue

        polys = parse_polygons(annot_path, image.shape)
        mask = polygon_to_mask(image.shape, polys)
        margin_mask = cv2.erode(mask, np.ones((2 * BOUNDARY_MARGIN + 1, 2 * BOUNDARY_MARGIN + 1), np.uint8))

        labels, placed = [], []
        used_configs = set()

        for _ in range(2):  # Add two triangle pores
            for _ in range(1000):
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                if margin_mask[y, x] != 255 or not is_far(x, y, 12, placed):
                    continue

                scale = random.uniform(0.4, 0.5)
                angle = random.randint(0, 360)
                config = (round(scale, 2), angle // 10)
                if config in used_configs:
                    continue
                used_configs.add(config)

                draw_bezier_triangle(image, x, y, scale=scale, angle_deg=angle)
                bbox = convert_to_yolo_bbox(x, y, 30 * scale + PORE_PADDING, 30 * scale + PORE_PADDING,
                                            image.shape[1], image.shape[0])
                labels.append((PORE_CLASS_ID, *bbox))
                placed.append((x, y, 12))
                break

        # Save results
        cv2.imwrite(os.path.join(dirs["output_images_dir"], fname), image)
        with open(os.path.join(dirs["output_labels_dir"], base + ".txt"), "w") as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

        print(f"✅ Saved {fname} with {len(labels)} triangle pores.")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    run_pipeline()
