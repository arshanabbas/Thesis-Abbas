import os
import cv2
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

# ----------------------- Configuration -----------------------
CLASS_BIG_PORE_ID = 2
BOUNDARY_MARGIN = 3
MIN_DISTANCE_BETWEEN_BIG_PORES = 30

# ----------------------- Helper Functions -----------------------
def convert_to_yolo_bbox(x, y, w, h, image_w, image_h):
    return x / image_w, y / image_h, (2 * w) / image_w, (2 * h) / image_h

def save_yolo_labels(output_labels_dir, image_name, labels):
    label_file = os.path.join(output_labels_dir, os.path.splitext(image_name)[0] + ".txt")
    with open(label_file, "w") as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

# ----------------------- Big Pore Generator -----------------------
def generate_smooth_noise(shape, scale=10):
    noise = np.random.rand(*shape)
    noise = gaussian_filter(noise, sigma=scale)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

def generate_major_lobed_pore(
    img_size=(64, 64),
    base_radius=20,
    num_lobes=5,
    lobe_strength=0.25,
    fill_color_range=(30, 70),
    blur_strength=5
):
    canvas = np.zeros(img_size, dtype=np.uint8)
    center = (img_size[1]//2, img_size[0]//2)

    lobe_angles = np.linspace(0, 2*np.pi, num=num_lobes, endpoint=False) + np.random.uniform(0, 2*np.pi/num_lobes, num_lobes)
    angles = np.linspace(0, 2*np.pi, 200, endpoint=False)

    radii = np.full_like(angles, base_radius, dtype=np.float32)
    for lobe_angle in lobe_angles:
        distance = np.abs(np.angle(np.exp(1j*(angles - lobe_angle))))
        lobe_effect = np.exp(-distance**2 / (2*(np.pi/num_lobes/2)**2))
        radii += base_radius * lobe_strength * lobe_effect * (np.random.choice([-1, 1]))

    points = np.array([
        (int(center[0] + r * np.cos(a)), int(center[1] + r * np.sin(a)))
        for r, a in zip(radii, angles)
    ], dtype=np.int32)

    cv2.fillPoly(canvas, [points], color=np.random.randint(*fill_color_range))

    # Add internal crater texture with smooth noise
    mask = canvas > 0
    base_texture = np.random.randint(10, 30)
    smooth_noise = generate_smooth_noise(img_size, scale=8)
    radial_gradient = np.zeros_like(canvas, dtype=np.float32)
    for y in range(img_size[0]):
        for x in range(img_size[1]):
            radial_distance = math.hypot(x - center[0], y - center[1]) / (base_radius * 1.5)
            radial_gradient[y, x] = 1 - min(1, radial_distance)
    combined_texture = (smooth_noise * radial_gradient * 0.6 * base_texture).astype(np.int16)
    textured_canvas = np.clip(canvas.astype(np.int16) - combined_texture, 0, 255).astype(np.uint8)
    canvas[mask] = textured_canvas[mask]

    # Add bright edge reflection
    edge_gradient = np.zeros_like(canvas, dtype=np.float32)
    direction = random.choice(['top', 'bottom', 'left', 'right'])
    for y in range(img_size[0]):
        for x in range(img_size[1]):
            if direction == 'top':
                distance = y / img_size[0]
            elif direction == 'bottom':
                distance = (img_size[0] - y) / img_size[0]
            elif direction == 'left':
                distance = x / img_size[1]
            elif direction == 'right':
                distance = (img_size[1] - x) / img_size[1]
            edge_gradient[y, x] = max(0, min(1, distance * 2))
    edge_effect = gaussian_filter(edge_gradient, sigma=10) * 80
    canvas = np.clip(canvas.astype(np.int16) + (edge_effect * mask).astype(np.int16), 0, 255).astype(np.uint8)

    # Slight rotation
    angle = random.randint(0, 360)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    canvas = cv2.warpAffine(canvas, M, (img_size[1], img_size[0]), borderValue=0)

    if blur_strength > 0:
        canvas = cv2.GaussianBlur(canvas, (blur_strength|1, blur_strength|1), sigmaX=2)

    return canvas

# ----------------------- Pore Placement -----------------------
def generate_big_pores_with_labels(polygon, img_shape, num_big_pores=3):
    labels = []
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_np], 255)
    margin_mask = cv2.erode(mask, np.ones((2 * BOUNDARY_MARGIN + 1, 2 * BOUNDARY_MARGIN + 1), np.uint8))

    big_pores_info = []
    centers = []

    attempts = 0
    max_attempts = 500

    while len(big_pores_info) < num_big_pores and attempts < max_attempts:
        pore_size = random.randint(40, 70)
        pore_img = generate_major_lobed_pore(img_size=(pore_size, pore_size), base_radius=pore_size//3)

        x = random.randint(0, img_shape[1] - pore_size)
        y = random.randint(0, img_shape[0] - pore_size)

        region = margin_mask[y:y+pore_size, x:x+pore_size]
        if region.shape[0] != pore_size or region.shape[1] != pore_size:
            attempts += 1
            continue

        bx, by = x + pore_size//2, y + pore_size//2

        # Check distance from existing pores
        too_close = False
        for (cx, cy) in centers:
            if math.hypot(bx - cx, by - cy) < MIN_DISTANCE_BETWEEN_BIG_PORES:
                too_close = True
                break

        if np.all(region == 255) and not too_close:
            big_pores_info.append((x, y, pore_img))
            centers.append((bx, by))

            bw, bh = pore_size//2, pore_size//2
            label = (CLASS_BIG_PORE_ID, *convert_to_yolo_bbox(bx, by, bw, bh, img_shape[1], img_shape[0]))
            labels.append(label)
        attempts += 1

    return big_pores_info, labels

# ----------------------- Main Function -----------------------
def visualize_class3_and_annotate(image_dir, annotation_dir, output_images_dir, output_labels_dir):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    for annotation_file in os.listdir(annotation_dir):
        if not annotation_file.endswith(".txt"):
            continue

        image_name = os.path.splitext(annotation_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        annotation_path = os.path.join(annotation_dir, annotation_file)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_list = []

        with open(annotation_path, 'r') as f:
            for line in f:
                if int(line.strip().split()[0]) != 3:
                    continue

                polygon = list(map(float, line.strip().split()[1:]))
                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0])) for i in range(0, len(polygon), 2)]
                big_pores, labels = generate_big_pores_with_labels(points, image.shape, num_big_pores=random.randint(2, 4))
                label_list.extend(labels)

                for (x, y, pore_img) in big_pores:
                    roi = image[y:y+pore_img.shape[0], x:x+pore_img.shape[1]]
                    pore_img_rgb = cv2.merge([pore_img, pore_img, pore_img])
                    mask = pore_img > 0
                    roi[mask] = pore_img_rgb[mask]

        cv2.imwrite(os.path.join(output_images_dir, image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if label_list:
            save_yolo_labels(output_labels_dir, image_name, label_list)

# ----------------------- Paths -----------------------
dirs = {
    "image_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images",
    "annotation_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8",
    "output_images_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image",
    "output_labels_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"
}

if __name__ == '__main__':
    visualize_class3_and_annotate(**dirs)
