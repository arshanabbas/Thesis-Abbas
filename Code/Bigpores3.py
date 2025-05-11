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

# Final control points (11 points)
pore_points = np.array([
    [50, 40], [48, 58], [47, 72], [45, 85], [44, 98], [46, 113],
    [58, 123], [54, 97], [60, 95], [85, 78], [88, 60]
])

# Generate spline
def generate_pore_shape(scale=1.0):
    pts = pore_points * scale
    tck, _ = splprep(pts.T, s=0.5, per=True)
    u = np.linspace(0, 1.0, 300)
    curve = np.stack(splev(u, tck), axis=-1).astype(np.int32)
    return curve

# Draw pore in an image
def draw_pore_on_image(image, center, scale=0.4):
    h, w = image.shape[:2]
    shape = generate_pore_shape(scale)
    shape[:, 0] += center[0] - shape[:, 0].mean().astype(int)
    shape[:, 1] += center[1] - shape[:, 1].mean().astype(int)
    cv2.fillPoly(image, [shape], (0, 0, 0))
    return shape

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

    valid = np.column_stack(np.where(mask == 255))
    if len(valid) == 0:
        print(f"⚠️ No valid Class 3 region in {fname}")
        continue

    cx, cy = random.choice(valid)
    pore_shape = draw_pore_on_image(image, (cy, cx))

    xmin, ymin = pore_shape.min(axis=0)
    xmax, ymax = pore_shape.max(axis=0)
    label = to_yolo_bbox(xmin, ymin, xmax, ymax, w, h)

    # Save image and label
    out_img_path = os.path.join(dirs["output_images_dir"], fname)
    out_lbl_path = os.path.join(dirs["output_labels_dir"], fname.replace(".jpg", ".txt"))
    cv2.imwrite(out_img_path, image)
    with open(out_lbl_path, 'w') as f:
        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\\n")

    print(f"✅ Saved {fname} with custom pore")
