import os
import cv2
import numpy as np
import random
import math

# ------------ CONFIGURATION ------------
dirs = {
    "image_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images",
    "annotation_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8",
    "output_images_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image",
    "output_labels_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"
}

CLASS_ID = 0  # Singular pore
BOUNDARY_MARGIN = 3


# ------------ FUNCTION: Generate Pore Mask ------------
def generate_irregular_pore_mask(img_size=(64, 64), base_radius=20, num_lobes=5, lobe_strength=0.25, blur_strength=3):
    h, w = img_size
    center = (w // 2, h // 2)

    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    radii = np.full_like(angles, base_radius, dtype=np.float32)
    lobe_angles = np.linspace(0, 2 * np.pi, num=num_lobes, endpoint=False)
    lobe_angles += np.random.uniform(0, 2 * np.pi / num_lobes, size=num_lobes)

    for lobe_angle in lobe_angles:
        distance = np.abs(np.angle(np.exp(1j * (angles - lobe_angle))))
        lobe_effect = np.exp(-distance**2 / (2 * (np.pi / num_lobes / 2)**2))
        radii += base_radius * lobe_strength * lobe_effect * random.choice([-1, 1])

    points = np.array([
        (int(center[0] + r * np.cos(a)), int(center[1] + r * np.sin(a)))
        for r, a in zip(radii, angles)
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    mask = cv2.GaussianBlur(mask, (blur_strength | 1, blur_strength | 1), sigmaX=2)

    # Optional: add radial shading
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    gradient = 1 - np.clip(dist / (base_radius * 1.5), 0, 1)
    mask = (mask * gradient).astype(np.uint8)

    return mask


# ------------ FUNCTION: Convert to YOLO format ------------
def convert_to_yolo_bbox(x, y, w, h, img_w, img_h):
    return x / img_w, y / img_h, w / img_w, h / img_h


# ------------ FUNCTION: Save YOLO Labels ------------
def save_yolo_labels(output_labels_dir, image_name, labels):
    """
    Save YOLO-format labels to a text file.
    """
    if not labels:
        print(f"[INFO] No labels to save for {image_name}. Skipping label file.")
        return

    label_file_path = os.path.join(output_labels_dir, os.path.splitext(image_name)[0] + ".txt")
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(label_file_path, "w") as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

    print(f"[✓] Saved labels: {label_file_path}")


# ------------ MAIN EXECUTION LOOP ------------
os.makedirs(dirs["output_images_dir"], exist_ok=True)
os.makedirs(dirs["output_labels_dir"], exist_ok=True)

for annotation_file in os.listdir(dirs["annotation_dir"]):
    if not annotation_file.endswith(".txt"):
        continue

    image_name = os.path.splitext(annotation_file)[0] + ".jpg"
    image_path = os.path.join(dirs["image_dir"], image_name)
    annotation_path = os.path.join(dirs["annotation_dir"], annotation_file)

    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path)
    if image is None:
        continue

    image_h, image_w = image.shape[:2]
    label_list = []

    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or int(parts[0]) != 3:
                continue

            polygon = list(map(float, parts[1:]))
            points = [(int(polygon[i] * image_w), int(polygon[i + 1] * image_h)) for i in range(0, len(polygon), 2)]
            polygon_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

            mask = np.zeros((image_h, image_w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_np], 255)
            margin_mask = cv2.erode(mask, np.ones((2 * BOUNDARY_MARGIN + 1, 2 * BOUNDARY_MARGIN + 1), np.uint8))

            # Try placing a pore inside the polygon
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                pore_size = random.randint(40, 70)
                pore_mask = generate_irregular_pore_mask(img_size=(pore_size, pore_size), base_radius=pore_size // 3)

                x = random.randint(0, image_w - pore_size)
                y = random.randint(0, image_h - pore_size)

                region = margin_mask[y:y + pore_size, x:x + pore_size]
                if region.shape[0] != pore_size or region.shape[1] != pore_size:
                    attempts += 1
                    continue

                if np.all(region == 255):
                    roi = image[y:y + pore_size, x:x + pore_size]
                    alpha = pore_mask.astype(np.float32) / 255.0
                    colored_pore = cv2.merge([pore_mask] * 3).astype(np.float32)

                    for c in range(3):
                        roi[..., c] = (alpha * colored_pore[..., c] + (1 - alpha) * roi[..., c]).astype(np.uint8)

                    cx, cy = x + pore_size // 2, y + pore_size // 2
                    bw, bh = pore_size // 2, pore_size // 2
                    label = (CLASS_ID, *convert_to_yolo_bbox(cx, cy, bw * 2, bh * 2, image_w, image_h))
                    label_list.append(label)
                    placed = True

                attempts += 1

    save_img_path = os.path.join(dirs["output_images_dir"], image_name)
    cv2.imwrite(save_img_path, image)
    save_yolo_labels(dirs["output_labels_dir"], image_name, label_list)
