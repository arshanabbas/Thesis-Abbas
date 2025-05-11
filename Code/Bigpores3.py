import cv2
import numpy as np
import os
import random
import math
from scipy.interpolate import splprep, splev

# ---------------------- Configuration ----------------------
dirs = {
    "image_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images",
    "annotation_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8",
    "output_images_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image",
    "output_labels_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"
}

FINAL_SIZE = 64
PORE_CLASS_ID = 0
BOUNDARY_MARGIN = 4
PORE_PADDING = 6
MIN_DISTANCE_BETWEEN_PORES = 20

os.makedirs(dirs["output_images_dir"], exist_ok=True)
os.makedirs(dirs["output_labels_dir"], exist_ok=True)

# ---------------------- Control Points from Sketch ----------------------
control_points = np.array([
    [50, 20],
    [48, 40],
    [47, 55],
    [46, 70],
    [45, 85],
    [50, 100],
    [60, 100],
    [70, 90],
    [80, 75],
    [90, 60],
    [85, 30],
], dtype=np.float32)

# ---------------------- Smoothing Function ----------------------
def create_smoothed_pore_mask(control_pts, upscale=6, final_size=64):
    canvas_size = final_size * upscale
    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    tck, _ = splprep(control_pts.T * upscale, s=0, per=True)
    u_fine = np.linspace(0, 1, 700)
    smooth_curve = np.vstack(splev(u_fine, tck)).T.astype(np.int32)

    cv2.fillPoly(mask, [smooth_curve], 255)
    mask = cv2.GaussianBlur(mask, (3, 3), 1.0)
    final = cv2.resize(mask, (final_size, final_size), interpolation=cv2.INTER_AREA)
    return final

# ---------------------- Utility Functions ----------------------
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

def overlay_mask_on_image(image, mask, center_x, center_y):
    h, w = mask.shape
    top_left_x = center_x - w // 2
    top_left_y = center_y - h // 2

    for i in range(h):
        for j in range(w):
            px = top_left_x + j
            py = top_left_y + i
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                alpha = mask[i, j] / 255.0
                if alpha > 0:
                    image[py, px] = (1 - alpha) * image[py, px] + alpha * np.array([30, 30, 30])

# ---------------------- Main Pipeline ----------------------
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

        image = cv2.imread(image_path)
        if image is None:
            continue

        polys = parse_polygons(annot_path, image.shape)
        mask = polygon_to_mask(image.shape, polys)
        margin_mask = cv2.erode(mask, np.ones((2 * BOUNDARY_MARGIN + 1, 2 * BOUNDARY_MARGIN + 1), np.uint8))

        labels = []
        placed = []

        pore_mask = create_smoothed_pore_mask(control_points)

        for _ in range(1):  # Insert one pore
            for _ in range(1000):
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                if margin_mask[y, x] != 255 or not is_far(x, y, FINAL_SIZE // 2, placed):
                    continue

                overlay_mask_on_image(image, pore_mask, x, y)
                labels.append((PORE_CLASS_ID, *convert_to_yolo_bbox(
                    x, y, FINAL_SIZE // 2 + PORE_PADDING, FINAL_SIZE // 2 + PORE_PADDING,
                    image.shape[1], image.shape[0]
                )))
                placed.append((x, y, FINAL_SIZE // 2))
                break

        cv2.imwrite(os.path.join(dirs["output_images_dir"], fname), image)
        with open(os.path.join(dirs["output_labels_dir"], base + ".txt"), "w") as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

        print(f"✅ Processed {fname}")

# ---------------------- Entry Point ----------------------
if __name__ == "__main__":
    run_pipeline()