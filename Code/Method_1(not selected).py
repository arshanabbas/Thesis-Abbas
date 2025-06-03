#Method 1 (Rough edges and not selected) 
import os
import cv2
import numpy as np
import random
from scipy.interpolate import splprep, splev

# Directories
dirs = {
    "image_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images",
    "annotation_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8",
    "output_images_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image",
    "output_labels_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"
}

os.makedirs(dirs["output_images_dir"], exist_ok=True)
os.makedirs(dirs["output_labels_dir"], exist_ok=True)

# Pore shape points (11 points example from your previous finalized shape)
pore_points = np.array([
    [50, 40], [48, 58], [47, 72], [45, 85], [44, 98], [46, 113],
    [58, 123], [54, 97], [60, 95], [85, 78], [88, 60]
])

# Generate smooth pore shape with anti-jagged method (supersampling)
def generate_pore_shape(scale=1.0):
    pts = pore_points * scale
    tck, _ = splprep(pts.T, s=0.5, per=True)
    u = np.linspace(0, 1.0, 300)
    curve = np.stack(splev(u, tck), axis=-1).astype(np.int32)
    return curve

# Draw pore safely inside image using mask check
def draw_pore_on_image_safely(image, mask, scale=0.4):
    h, w = image.shape[:2]
    eroded_mask = cv2.erode(mask, np.ones((10, 10), np.uint8), iterations=1)
    valid_points = np.column_stack(np.where(eroded_mask == 255))

    random.shuffle(valid_points)  # shuffle to avoid infinite loops
    for attempt, (cy, cx) in enumerate(valid_points):
        # Generate shape
        shape = generate_pore_shape(scale)
        shape[:, 0] += cx - shape[:, 0].mean().astype(int)
        shape[:, 1] += cy - shape[:, 1].mean().astype(int)

        # Create pore mask to check intersection
        temp_mask = np.zeros_like(mask)
        cv2.fillPoly(temp_mask, [shape], 255)
        intersection = cv2.bitwise_and(eroded_mask, temp_mask)

        if cv2.countNonZero(intersection) == cv2.countNonZero(temp_mask):
            # Valid pore found fully inside ROI
            # Supersampling approach to reduce jaggedness
            factor = 4
            upscale = cv2.resize(image, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
            upscale_shape = (shape * factor).astype(np.int32)
            cv2.fillPoly(upscale, [upscale_shape], (0, 0, 0))
            downscale = cv2.resize(upscale, (w, h), interpolation=cv2.INTER_AREA)
            np.copyto(image, downscale)
            return shape
    print("⚠️ No valid placement found")
    return None

# Convert bbox to YOLO format
def to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return 2, x_center, y_center, width, height  # class_id=2

# Main loop
for fname in os.listdir(dirs["image_dir"]):
    if not fname.endswith(".jpg"):
        continue
    img_path = os.path.join(dirs["image_dir"], fname)
    ann_path = os.path.join(dirs["annotation_dir"], fname.replace(".jpg", ".txt"))
    if not os.path.exists(ann_path):
        print(f"⚠️ Missing annotation for {fname}")
        continue

    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) != 3:
                continue
            coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            polygon = coords.astype(np.int32)
            cv2.fillPoly(mask, [polygon], 255)

    if np.count_nonzero(mask) == 0:
        print(f"⚠️ No valid Class 3 ROI in {fname}")
        continue

    pore_shape = draw_pore_on_image_safely(image, mask)
    if pore_shape is None:
        continue  # Skip if no safe placement found

    xmin, ymin = pore_shape.min(axis=0)
    xmax, ymax = pore_shape.max(axis=0)
    label = to_yolo_bbox(xmin, ymin, xmax, ymax, w, h)

    out_img_path = os.path.join(dirs["output_images_dir"], fname)
    out_lbl_path = os.path.join(dirs["output_labels_dir"], fname.replace(".jpg", ".txt"))
    cv2.imwrite(out_img_path, image)
    with open(out_lbl_path, 'w') as f:
        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

    print(f"✅ Saved {fname} with safe pore inside ROI")
