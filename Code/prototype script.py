import cv2
import numpy as np
import random
import math

def draw_triangular_pore(canvas, center, size, angle):
    """Draw a soft triangular pore."""
    triangle = np.array([
        [0, -size],
        [-size, size],
        [size, size]
    ], dtype=np.float32)

    # Rotate
    theta = np.radians(angle)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated = triangle @ rot_matrix.T
    rotated += np.array(center)

    # Create mask
    mask = np.zeros_like(canvas, dtype=np.uint8)
    cv2.fillConvexPoly(mask, rotated.astype(np.int32), color=255)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    for c in range(3):
        canvas[..., c] = np.where(mask > 0, (canvas[..., c] * (1 - mask / 255) + 40 * (mask / 255)).astype(np.uint8), canvas[..., c])

def draw_comet_pore(canvas, center, radius, angle):
    """Draw a pore with a comet-like tail."""
    h, w = canvas.shape[:2]
    temp = np.zeros((h, w, 4), dtype=np.uint8)

    # Core ellipse
    cv2.ellipse(temp, center, (radius, radius), 0, 0, 360, (50, 50, 50, 255), -1, lineType=cv2.LINE_AA)

    # Tail ellipse (faded)
    tail_len = radius * 3
    tail_width = radius
    tail_center = (
        int(center[0] + tail_len * math.cos(math.radians(angle))),
        int(center[1] + tail_len * math.sin(math.radians(angle)))
    )

    cv2.ellipse(temp, tail_center, (tail_len, tail_width), angle, 0, 360, (40, 40, 40, 100), -1, lineType=cv2.LINE_AA)

    # Blend
    rgb, alpha = temp[..., :3], temp[..., 3:] / 255.0
    for c in range(3):
        canvas[..., c] = (alpha[..., 0] * rgb[..., c] + (1 - alpha[..., 0]) * canvas[..., c]).astype(np.uint8)

# === Preview
def preview_shapes():
    canvas = np.full((200, 400, 3), 220, dtype=np.uint8)
    draw_triangular_pore(canvas, center=(100, 100), size=20, angle=random.randint(0, 360))
    draw_comet_pore(canvas, center=(300, 100), radius=8, angle=random.randint(0, 360))
    cv2.imwrite("preview_singular_shapes.png", canvas)
    print("✅ Preview saved as 'preview_singular_shapes.png'")

if __name__ == "__main__":
    preview_shapes()