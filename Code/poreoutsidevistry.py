import os
import cv2
import numpy as np
import random
import math

# ----------------------- Configuration -----------------------
CLASS_3_COLOR = (64, 64, 64)
MIN_PORE_RADIUS = 1
MAX_PORE_RADIUS = 5
MIN_TOTAL_PORES = 15
MAX_TOTAL_PORES = 30
MIN_CLUSTERS = 2
MAX_CLUSTERS = 3
MIN_DISTANCE_BETWEEN_PORES = 7
MIN_DISTANCE_BETWEEN_CLUSTERS = 40
STRICT_CIRCULARITY = 0.85
STRICT_ELONGATION = 1.35
BOUNDARY_MARGIN = 3
PORE_CLASS_ID = 0
PORE_NEST_CLASS_ID = 1
PORE_PADDING = 5

# ----------------------- Helper Functions -----------------------
def is_point_inside_polygon(point, polygon):
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0

def convert_to_yolo_bbox(x, y, w, h, image_w, image_h):
    return x / image_w, y / image_h, (2 * w) / image_w, (2 * h) / image_h

def save_yolo_labels(output_labels_dir, image_name, labels):
    label_file = os.path.join(output_labels_dir, os.path.splitext(image_name)[0] + ".txt")
    with open(label_file, "w") as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

def are_clusters_far_enough(new_center, existing_centers, min_distance):
    for center in existing_centers:
        if np.linalg.norm(np.array(new_center) - np.array(center)) < min_distance:
            return False
    return True

def is_valid_pore_shape(w, h):
    if w == 0 or h == 0:
        return False
    ratio = max(w, h) / min(w, h)
    area = np.pi * (w / 2) * (h / 2)
    perimeter = 2 * np.pi * np.sqrt((w / 2) ** 2 + (h / 2) ** 2) / 2
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return ratio <= STRICT_ELONGATION and circularity >= STRICT_CIRCULARITY

def is_far_from_existing(x, y, r, placed_pores):
    for px, py, pr in placed_pores:
        if math.hypot(x - px, y - py) < (r + pr + MIN_DISTANCE_BETWEEN_PORES):
            return False
    return True

def draw_pore(image, x, y, w, h, angle, base_arc_start):
    scale = 6
    img_h, img_w = image.shape[:2]
    up_w, up_h = img_w * scale, img_h * scale

    cx, cy = int(x * scale), int(y * scale)
    rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
    center = (cx, cy)

    core_radius = max(w, h)
    if core_radius <= 2:
        thick_start, thick_end = 2 * scale, 1 * scale
    elif core_radius == 3:
        thick_start, thick_end = int(2.5 * scale), 1 * scale
    else:
        thick_start, thick_end = 3 * scale, 1 * scale

    relative_y = y / img_h
    jitter = int((random.uniform(-1, 1) * 0.2) * 360 * relative_y)
    arc_start = (base_arc_start + jitter) % 360

    if core_radius > 2:
        arc_length = random.randint(180, 270)
    else:
        arc_length = 240

    fade_segments = 6
    angle_step = arc_length // fade_segments

    base = np.zeros((up_h, up_w, 4), dtype=np.uint8)

    for i in range(fade_segments):
        opacity = int(np.interp(i, [0, fade_segments - 1], [255, 50]))
        thickness = int(np.interp(i, [0, fade_segments - 1], [thick_start, thick_end]))
        axes = (rw + int(thickness * 0.75), rh + int(thickness * 0.75))
        color = (180, 180, 180, opacity)

        sa = arc_start + i * angle_step
        ea = sa + angle_step

        if i > 0:
            jitter_angle = random.uniform(-5, 5)
            sa += jitter_angle
            ea += jitter_angle

        cv2.ellipse(
            base,
            center,
            axes,
            angle,
            sa,
            ea,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

    base[:, :, 3] = cv2.GaussianBlur(base[:, :, 3], (5, 5), sigmaX=2.0)
    cv2.ellipse(base, center, (rw, rh), angle, 0, 360, (45, 45, 45, 255), -1, lineType=cv2.LINE_AA)

    final = cv2.resize(base, (img_w, img_h), interpolation=cv2.INTER_AREA)
    rgb, alpha = final[..., :3], final[..., 3:] / 255.0
    for c in range(3):
        image[..., c] = (alpha[..., 0] * rgb[..., c] + (1 - alpha[..., 0]) * image[..., c]).astype(np.uint8)

# ----------------------- Pore and Cluster Generation -----------------------
def generate_balanced_pores_with_labels(polygon, img_shape):
    pores, labels = [], []
    polygon_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    x_min, y_min = np.min(polygon_np, axis=0)[0]
    x_max, y_max = np.max(polygon_np, axis=0)[0]

    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_np], 255)
    margin_mask = cv2.erode(mask, np.ones((2 * BOUNDARY_MARGIN + 1, 2 * BOUNDARY_MARGIN + 1), np.uint8))

    placed_pores = []
    attempts = 0
    while len(pores) < random.randint(MIN_TOTAL_PORES, MAX_TOTAL_PORES) and attempts < 2000:
        attempts += 1
        x, y = random.randint(x_min, x_max), random.randint(y_min, y_max)
        w, h = random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS), random.randint(MIN_PORE_RADIUS, MAX_PORE_RADIUS)
        angle = random.randint(0, 180)
        r = max(w, h)
        if mask[y, x] == 255 and margin_mask[y, x] == 255:
            if is_valid_pore_shape(w, h) and is_far_from_existing(x, y, r, placed_pores):
                pores.append((x, y, w, h, angle))
                placed_pores.append((x, y, r))
                bx, by, bw, bh = convert_to_yolo_bbox(x, y, w + PORE_PADDING, h + PORE_PADDING, img_shape[1], img_shape[0])
                labels.append((PORE_CLASS_ID, bx, by, bw, bh))

    return pores, labels

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
            print(f"Image {image_name} not found. Skipping...")
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
                pores, labels = generate_balanced_pores_with_labels(points, image.shape)
                print(f"{image_name} → Generated {len(pores)} pores")
                label_list.extend(labels)

                base_arc_start = random.randint(0, 359)
                for (x, y, w, h, angle) in pores:
                    draw_pore(image, x, y, w, h, angle, base_arc_start)

        cv2.imwrite(os.path.join(output_images_dir, image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if label_list:
            save_yolo_labels(output_labels_dir, image_name, label_list)

# ----------------------- Execution -----------------------
dirs = {
    "image_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images",
    "annotation_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/yolov8",
    "output_images_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image",
    "output_labels_dir": "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"
}

if __name__ == '__main__':
    visualize_class3_and_annotate(**dirs)


"""def draw_pore(image, x, y, w, h, angle):

    scale = 6
    img_h, img_w = image.shape[:2]
    up_w, up_h = img_w * scale, img_h * scale

    cx, cy = int(x * scale), int(y * scale)
    rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
    center = (cx, cy)

    # Create base with alpha
    base = np.zeros((up_h, up_w, 4), dtype=np.uint8)

    max_core_radius = max(w, h)
    fade_segments = 6

    # === Logic for small and large pores ===
    if max_core_radius <= 2:
        arc_length = 216  # fixed ~60% arc
        thick_start, thick_end = 2 * scale, 1 * scale
    else:
        arc_percentage = random.uniform(0.5, 0.75)  # 50–75%
        arc_length = int(360 * arc_percentage)
        thick_start, thick_end = 3 * scale, 1 * scale

    start_angle = random.randint(0, 360)
    angle_step = arc_length // fade_segments
    current_angle = start_angle

    # === Draw outer ring (fading segments) ===
    for i in range(fade_segments):
        opacity = int(np.interp(i, [0, fade_segments - 1], [255, 50]))
        thickness = int(np.interp(i, [0, fade_segments - 1], [thick_start, thick_end]))
        axes = (rw + int(thickness * 0.75), rh + int(thickness * 0.75))
        segment_color = (180, 180, 180, opacity)

        cv2.ellipse(
            base,
            center,
            axes,
            angle,
            current_angle,
            current_angle + angle_step,
            segment_color,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )
        current_angle += angle_step

    # === Gaussian blur for soft fading ===
    base[:, :, 3] = cv2.GaussianBlur(base[:, :, 3], (5, 5), sigmaX=2)

    # === Draw solid pore core ===
    pore_color = (45, 45, 45)
    core_axes = (rw, rh)
    cv2.ellipse(base, center, core_axes, angle, 0, 360, pore_color + (255,), -1, lineType=cv2.LINE_AA)

    # === Resize and alpha blend ===
    final = cv2.resize(base, (img_w, img_h), interpolation=cv2.INTER_AREA)
    rgb, alpha = final[..., :3], final[..., 3:] / 255.0

    for c in range(3):
        image[..., c] = (alpha[..., 0] * rgb[..., c] + (1 - alpha[..., 0]) * image[..., c]).astype(np.uint8)
"""
"""def draw_pore(image, x, y, w, h, angle):

    scale = 6
    img_h, img_w = image.shape[:2]
    up_w, up_h = img_w * scale, img_h * scale

    cx, cy = int(x * scale), int(y * scale)
    rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
    center = (cx, cy)

    # Create blank canvas with alpha
    base = np.zeros((up_h, up_w, 4), dtype=np.uint8)

    # Ring appearance setup
    arc_span = 360
    start_angle = random.randint(0, 360)
    arc_length = random.randint(210, 240)  # 55–60% visible
    fade_segments = 6  # More gradual fade
    angle_step = arc_length // fade_segments
    current_angle = start_angle

    # Draw gradually fading arc segments
    for i in range(fade_segments):
        opacity = int(np.interp(i, [0, fade_segments - 1], [255, 50]))  # From 100% to ~20%
        thickness = int(np.interp(i, [0, fade_segments - 1], [3 * scale, 1 * scale]))
        axes = (rw + int(thickness * 0.75), rh + int(thickness * 0.75))

        segment_color = (180, 180, 180, opacity)
        cv2.ellipse(base, center, axes, angle, current_angle, current_angle + angle_step,
                    segment_color, thickness=thickness, lineType=cv2.LINE_AA)
        current_angle += angle_step

    # Blur only the ring area
    blurred = cv2.GaussianBlur(base, (5, 5), sigmaX=2.0, sigmaY=2.0)

    # Draw inner core (solid, no alpha)
    pore_color = (45, 45, 45)
    core_axes = (rw, rh)
    cv2.ellipse(blurred, center, core_axes, angle, 0, 360, pore_color + (255,), -1, lineType=cv2.LINE_AA)

    # Resize back to image size
    final = cv2.resize(blurred, (img_w, img_h), interpolation=cv2.INTER_AREA)

    # Blend using alpha channel
    rgb, alpha = final[..., :3], final[..., 3:] / 255.0
    for c in range(3):
        image[..., c] = (alpha[..., 0] * rgb[..., c] + (1 - alpha[..., 0]) * image[..., c]).astype(np.uint8)
"""