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

os.makedirs(dirs["output_images_dir"], exist_ok=True)
os.makedirs(dirs["output_labels_dir"], exist_ok=True)

# -----------------------
# Helper Functions
# -----------------------

PORE_CLASS_ID = 0
BOUNDARY_MARGIN = 4
PORE_PADDING = 6
MIN_DISTANCE_BETWEEN_PORES = 15

def load_image(image_path):
    return cv2.imread(image_path)

def parse_yolo_polygon_annotation(annotation_path, image_shape):
    h, w = image_shape[:2]
    polygons = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) != 3:
                continue
            coords = list(map(float, parts[1:]))
            points = [(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])]
            polygons.append(np.array(points, dtype=np.int32))
    return polygons

def polygon_to_mask(image_shape, polygons):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
    return mask

def is_far_from_existing(x, y, r, placed_pores):
    for px, py, pr in placed_pores:
        if math.hypot(x - px, y - py) < (r + pr + MIN_DISTANCE_BETWEEN_PORES):
            return False
    return True

def convert_to_yolo_bbox(x, y, w, h, image_w, image_h):
    return x / image_w, y / image_h, (2 * w) / image_w, (2 * h) / image_h

# -----------------------
# Final Crescent Drawing Function
# -----------------------

def draw_crescent_pore(image, x, y, scale=0.5, angle=0, r_west=12, r_east=4):
    # WEST (left dome arc)
    western_arc = []
    center_w = (-15, 0)
    for theta in np.linspace(180, 270, 10):
        rad = np.radians(theta)
        xw = center_w[0] + r_west * np.cos(rad)
        yw = center_w[1] + r_west * np.sin(rad)
        western_arc.append((xw, yw))

    # EAST (right dome arc)
    eastern_arc = []
    center_e = (22, 0)
    for theta in np.linspace(270, 360, 10):
        rad = np.radians(theta)
        xe = center_e[0] + r_east * np.cos(rad)
        ye = center_e[1] + r_east * np.sin(rad)
        eastern_arc.append((xe, ye))

    # TOP AND BOTTOM
    mid_top = [(0, -12), (18, -6)]
    bottom = []
    for t in np.linspace(0, 1, 20):
        xt = (1 - t) * (center_e[0] - r_east) + t * (-10)
        yt = 4 + 6 * np.sin(t * np.pi)
        bottom.append((xt, yt))

    # Transform
    all_pts = np.vstack([western_arc, mid_top, eastern_arc, bottom])
    theta_rad = np.radians(angle)
    rot_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])
    transformed = np.dot(all_pts * scale, rot_matrix.T) + [x, y]
    contour = transformed.astype(np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(image, [contour], (30, 30, 30), lineType=cv2.LINE_AA)

# -----------------------
# Pipeline
# -----------------------

def run_pipeline():
    for filename in os.listdir(dirs["image_dir"]):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(dirs["image_dir"], filename)
        annotation_path = os.path.join(dirs["annotation_dir"], base_name + ".txt")

        if not os.path.exists(annotation_path):
            print(f"⚠️ Skipping {filename} (no annotation)")
            continue

        image = load_image(image_path)
        if image is None:
            continue

        polygons = parse_yolo_polygon_annotation(annotation_path, image.shape)
        mask = polygon_to_mask(image.shape, polygons)
        margin_mask = cv2.erode(mask, np.ones((2 * BOUNDARY_MARGIN + 1, 2 * BOUNDARY_MARGIN + 1), np.uint8))

        placed_pores = []
        yolo_labels = []

        for _ in range(2):  # Add two crescent pores per image
            for _ in range(1000):
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                if margin_mask[y, x] == 255 and is_far_from_existing(x, y, 10, placed_pores):
                    angle = random.randint(0, 360)
                    scale = random.uniform(0.45, 0.6)
                    r_west = random.randint(8, 16)
                    r_east = random.randint(2, 5)

                    draw_crescent_pore(image, x, y, scale=scale, angle=angle, r_west=r_west, r_east=r_east)

                    # Estimate bounding box size
                    bw, bh = int(30 * scale) + PORE_PADDING, int(15 * scale) + PORE_PADDING
                    bbox = convert_to_yolo_bbox(x, y, bw, bh, image.shape[1], image.shape[0])
                    yolo_labels.append((PORE_CLASS_ID, *bbox))
                    placed_pores.append((x, y, 10))
                    break

        # Save modified image and label
        cv2.imwrite(os.path.join(dirs["output_images_dir"], filename), image)
        with open(os.path.join(dirs["output_labels_dir"], base_name + ".txt"), "w") as f:
            for label in yolo_labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
        

        print(f"Number of labels for {filename}: {len(yolo_labels)}")

        print(f"✅ Processed {filename}")

if __name__ == "__main__":
    run_pipeline()
