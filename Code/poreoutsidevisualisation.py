import os
import cv2
import numpy as np
import random
import math

# ----------------------- Configuration -----------------------
IMG_SIZE = 256
NUM_PORES = 20
MIN_RADIUS = 1
MAX_RADIUS = 5
BACKGROUND_COLOR = 200
CORE_COLOR = (45, 45, 45)
RING_COLOR = (180, 180, 180)

# ----------------------- Draw Function -----------------------
def draw_pore(image, x, y, w, h, angle):
    scale = 6
    img_h, img_w = image.shape[:2]
    up_h, up_w = img_h * scale, img_w * scale

    mask = np.ones((up_h, up_w, 3), dtype=np.uint8) * BACKGROUND_COLOR

    cx, cy = x * scale, y * scale
    rw, rh = w * scale, h * scale

    # Outer ring
    ring_thickness = int(0.5 * scale) if max(w, h) < 4 else int(1.5 * scale)
    outer_axes = (int(rw + ring_thickness), int(rh + ring_thickness))
    cv2.ellipse(mask, (int(cx), int(cy)), outer_axes, angle, 0, 360, RING_COLOR, -1, lineType=cv2.LINE_AA)

    # Apply Gaussian blur to soften the ring
    blurred_mask = cv2.GaussianBlur(mask, (15, 15), sigmaX=5, sigmaY=5)

    # Draw core (non-blurred) over blurred ring
    cv2.ellipse(blurred_mask, (int(cx), int(cy)), (int(rw), int(rh)), angle, 0, 360, CORE_COLOR, -1, lineType=cv2.LINE_AA)

    # Resize and merge
    resized = cv2.resize(blurred_mask, (img_w, img_h), interpolation=cv2.INTER_AREA)
    image[:] = cv2.min(image, resized)

# ----------------------- Test Image Generator -----------------------
def generate_test_image():
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * BACKGROUND_COLOR
    placed = []
    attempts = 0
    
    while len(placed) < NUM_PORES and attempts < 500:
        attempts += 1
        x = random.randint(10, IMG_SIZE - 10)
        y = random.randint(10, IMG_SIZE - 10)
        w = random.randint(MIN_RADIUS, MAX_RADIUS)
        h = random.randint(MIN_RADIUS, MAX_RADIUS)
        angle = random.randint(0, 180)
        if all(math.hypot(x - px, y - py) > (w + ph + 6) for px, py, pw, ph in placed):
            placed.append((x, y, w, h))
            draw_pore(img, x, y, w, h, angle)

    return img

# ----------------------- Run Test -----------------------
if __name__ == '__main__':
    test_img = generate_test_image()
    cv2.imwrite("test_pore_blurred_ring.jpg", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    print("Generated test_pore_blurred_ring.jpg with updated halo effect.")