import os
import cv2
import numpy as np
import random

# Directories
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8"
output_images_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image"
output_labels_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Variant 5 Pore Generator (Distorted)
def generate_pore_shape(scale=0.45, steps=14):
    base_pts = np.array([
        [0, -28],
        [-70, -4],
        [-50, 30],
        [65, 22]
    ], dtype=np.float32)

    contour = []
    for i in range(len(base_pts)):
        p0 = base_pts[i - 1]
        p1 = base_pts[i]
        p2 = base_pts[(i + 1) % len(base_pts)]

        smooth = np.clip(0.2 + random.uniform(-0.5, 0.5), 0.05, 0.5)
        jitter = np.random.uniform(-10, 10, size=2)

        start = p1 + (p0 - p1) * smooth + jitter
        end = p1 + (p2 - p1) * smooth - jitter

        bezier = [(1 - t) ** 2 * start + 2 * (1 - t) * t * p1 + t ** 2 * end for t in np.linspace(0, 1, steps)]
        contour.extend(bezier)

    contour = np.array(contour)
    contour -= np.mean(contour, axis=0)
    contour *= scale * np.array([
        random.uniform(0.8, 1.2),
        random.uniform(0.65, 1.2)
    ])

    theta = np.radians(random.uniform(0, 360))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    contour = np.dot(contour, rot.T)

    return contour

# Convert bbox to YOLO format
def to_yolo_bbox(contour, w, h):
    xmin, ymin = contour.min(axis=0)
    xmax, ymax = contour.max(axis=0)
    x_center = (xmin + xmax) / 2 / w
    y_center = (ymin + ymax) / 2 / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# Get class-3 polygon mask
def get_class3_mask(annotation_path, img_shape):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(annotation_path):
        return mask

    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            if cls_id != 3:
                continue
            coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            polygon = coords.astype(np.int32)
            cv2.fillPoly(mask, [polygon], 255)

    return mask

# Check if full pore is inside class-3 ROI
def is_pore_inside_mask(pore_pts, mask):
    pore_int = np.round(pore_pts).astype(np.int32)
    h, w = mask.shape
    for x, y in pore_int:
        if not (0 <= x < w and 0 <= y < h):
            return False
        if mask[y, x] != 255:
            return False
    return True

# Main loop
for fname in os.listdir(image_dir):
    if not fname.lower().endswith(('.jpg', '.png')):
        continue

    image_path = os.path.join(image_dir, fname)
    annotation_path = os.path.join(annotation_dir, fname.replace('.jpg', '.txt').replace('.png', '.txt'))

    image = cv2.imread(image_path)
    if image is None:
        continue

    h, w = image.shape[:2]
    class3_mask = get_class3_mask(annotation_path, image.shape)

    # Get all pixels that are valid centers
    valid_points = np.column_stack(np.where(class3_mask == 255))
    if len(valid_points) == 0:
        print(f"❌ No class-3 ROI in {fname}")
        continue

    max_attempts = 50
    success = False

    for attempt in range(max_attempts):
        cy, cx = random.choice(valid_points)
        pore = generate_pore_shape()
        shifted = pore + np.array([cx, cy]) - np.mean(pore, axis=0)

        if is_pore_inside_mask(shifted, class3_mask):
            contour_int = shifted.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [contour_int], (30, 30, 30), lineType=cv2.LINE_AA)

            # Save image and label
            label = to_yolo_bbox(shifted, w, h)
            out_image_path = os.path.join(output_images_dir, fname)
            out_label_path = os.path.join(output_labels_dir, fname.replace('.jpg', '.txt').replace('.png', '.txt'))

            cv2.imwrite(out_image_path, image)
            with open(out_label_path, 'w') as f:
                f.write(label + "\n")

            print(f"✅ {fname} saved with safe pore.")
            success = True
            break

    if not success:
        print(f"❌ Failed to place pore safely in {fname}")