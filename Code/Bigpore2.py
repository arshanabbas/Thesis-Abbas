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

PORE_CLASS_ID = 0
BOUNDARY_MARGIN = 4
PORE_PADDING = 6
MIN_DISTANCE_BETWEEN_PORES = 20

os.makedirs(dirs["output_images_dir"], exist_ok=True)
os.makedirs(dirs["output_labels_dir"], exist_ok=True)

# -----------------------
# Utility Functions
# -----------------------
def bezier_interp(p0, p1, p2, steps=30):
    return [(1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2 for t in np.linspace(0, 1, steps)]

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
# Final Pore Shape: Variant 5 (Extreme Distortion)
# -----------------------
def draw_variant5_pore(image, x, y, base_scale=0.45, rotation_deg=0):
    base_pts = np.array([
        [0, -28],
        [-70, -4],
        [-50, 30],
        [65, 22]
    ], dtype=np.float32)

    def extreme_distort_edges(pts, steps=14):
        contour = []
        for i in range(len(pts)):
            p0 = pts[i - 1]
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]

            smooth = 0.2 + random.uniform(-0.5, 0.5)
            smooth = np.clip(smooth, 0.05, 0.5)
            jitter_x = random.uniform(-10, 10)
            jitter_y = random.uniform(-10, 10)

            start = p1 + (p0 - p1) * smooth + np.array([jitter_x, jitter_y])
            end = p1 + (p2 - p1) * smooth + np.array([-jitter_x, -jitter_y])
            bezier = bezier_interp(start, p1, end, steps=steps)
            contour.extend(bezier)
        return np.array(contour, dtype=np.float32)

    contour = extreme_distort_edges(base_pts)
    center = np.mean(contour, axis=0)
    contour -= center
    contour[:, 0] *= base_scale * random.uniform(0.8, 1.2)
    contour[:, 1] *= base_scale * random.uniform(0.65, 1.2)

    theta = np.radians(rotation_deg)
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    contour = np.dot(contour, rot.T)
    contour += np.array([x, y])
    contour = contour.astype(np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(image, [contour], color=(30, 30, 30), lineType=cv2.LINE_AA)

# -----------------------
# Main Pipeline Function
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

        for _ in range(1):  # Add one extreme distorted pore
            for _ in range(1000):
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                if margin_mask[y, x] != 255 or not is_far(x, y, 20, placed):
                    continue

                draw_variant5_pore(image, x, y, rotation_deg=random.randint(0, 360))
                bbox = convert_to_yolo_bbox(x, y, 30 + PORE_PADDING, 30 + PORE_PADDING, image.shape[1], image.shape[0])
                labels.append((PORE_CLASS_ID, *bbox))
                placed.append((x, y, 20))
                break

        # Save
        cv2.imwrite(os.path.join(dirs["output_images_dir"], fname), image)
        with open(os.path.join(dirs["output_labels_dir"], base + ".txt"), "w") as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

        print(f"✅ Saved {fname} with 1 Variant 5 pore.")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    run_pipeline()
