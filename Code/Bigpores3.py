import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import os
import random

# Directories (New folder for Method 2)
dirs = {
    "image_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images",
    "annotation_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8",
    "output_images_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image_method2",
    "output_labels_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation_method2"
}

os.makedirs(dirs["output_images_dir"], exist_ok=True)
os.makedirs(dirs["output_labels_dir"], exist_ok=True)

pore_points = np.array([
    [50, 40], [48, 58], [47, 72], [45, 85], [44, 98], [46, 113],
    [58, 123], [54, 97], [60, 95], [85, 78], [88, 60]
])

def generate_pore_shape(scale=1.0):
    pts = pore_points * scale
    tck, _ = splprep(pts.T, s=0.5, per=True)
    u = np.linspace(0, 1.0, 300)
    curve = np.stack(splev(u, tck), axis=-1).astype(np.int32)
    return curve

def draw_pore_soft_mask(image, center, scale=0.4):
    h, w = image.shape[:2]
    shape = generate_pore_shape(scale)
    shape[:, 0] += center[0] - shape[:, 0].mean().astype(int)
    shape[:, 1] += center[1] - shape[:, 1].mean().astype(int)

    # Create alpha mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [shape], 255)
    mask_blur = cv2.GaussianBlur(mask, (9, 9), 3)

    # Blend using alpha mask
    mask_blur_float = mask_blur.astype(float) / 255.0
    mask_blur_float = cv2.merge([mask_blur_float]*3)

    black_shape = np.zeros_like(image)
    blended = (mask_blur_float * black_shape + (1 - mask_blur_float) * image).astype(np.uint8)

    return blended, shape

def to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return 2, x_center, y_center, width, height

for fname in os.listdir(dirs["image_dir"]):
    if not fname.endswith(".jpg"):
        continue
    img_path = os.path.join(dirs["image_dir"], fname)
    ann_path = os.path.join(dirs["annotation_dir"], fname.replace(".jpg", ".txt"))
    if not os.path.exists(ann_path):
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
        continue

    cx, cy = random.choice(valid)
    image, pore_shape = draw_pore_soft_mask(image, (cy, cx))

    xmin, ymin = pore_shape.min(axis=0)
    xmax, ymax = pore_shape.max(axis=0)
    label = to_yolo_bbox(xmin, ymin, xmax, ymax, w, h)

    cv2.imwrite(os.path.join(dirs["output_images_dir"], fname), image)
    with open(os.path.join(dirs["output_labels_dir"], fname.replace(".jpg", ".txt")), 'w') as f:
        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

