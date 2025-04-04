import cv2
import numpy as np
import random
import os

# ----------------------- Configuration -----------------------
IMAGE_PATH = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/images/0a05af59-10-19.3.jpg"  # Replace with your actual image path
OUTPUT_PATH = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image/0a05af59-10-19.3.jpg"

NUM_PORES = 30
MIN_RADIUS = 1
MAX_RADIUS = 5

# ----------------------- Pore Drawing Function -----------------------
def draw_pore_clean(image, x, y, w, h, angle):
    img_h, img_w = image.shape[:2]
    scale = 4
    up_w, up_h = img_w * scale, img_h * scale

    # Create high-res blank mask
    high_core = np.ones((up_h, up_w, 3), dtype=np.uint8) * 255  # white base
    cx, cy = x * scale, y * scale
    rw = w * scale
    rh = h * scale

    # Draw solid black ellipse core
    cv2.ellipse(high_core, (cx, cy), (rw, rh), angle, 0, 360, (0, 0, 0), -1)

    # Downscale mask
    small_mask = cv2.resize(high_core, (img_w, img_h), interpolation=cv2.INTER_AREA)
    mask_to_subtract = 255 - small_mask  # invert to get dark pore mask

    # Subtract pore directly from image
    image[:] = cv2.subtract(image, mask_to_subtract)

# ----------------------- Main Application -----------------------
def apply_clean_pores(image_path, num_pores):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    for _ in range(num_pores):
        x = random.randint(10, w - 10)
        y = random.randint(10, h - 10)
        r = random.randint(MIN_RADIUS, MAX_RADIUS)
        aspect = random.uniform(0.9, 1.2)
        rw = r
        rh = int(r * aspect)
        angle = random.randint(0, 180)
        draw_pore_clean(image, x, y, rw, rh, angle)

    cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved clean Phase 2 image with pores to: {OUTPUT_PATH}")

# ----------------------- Execute -----------------------
if __name__ == '__main__':
    apply_clean_pores(IMAGE_PATH, NUM_PORES)
