import os
import cv2
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------

image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/crack_dataset/images"
label_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/crack_dataset/labels"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/crack_dataset/testing"

os.makedirs(output_dir, exist_ok=True)

# Class color mapping (BGR)
CLASS_COLORS = {
    0: (64, 64, 64),         # Crack - Black
}

# -------------------- MAIN --------------------

def draw_yolo_boxes(image, label_path):
    h, w, _ = image.shape
    if not os.path.exists(label_path):
        return image  # No labels, return as-is

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed

            class_id, x_c, y_c, width, height = map(float, parts)
            x_center = int(x_c * w)
            y_center = int(y_c * h)
            box_w = int(width * w)
            box_h = int(height * h)

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            color = CLASS_COLORS.get(int(class_id), (128, 128, 128))  # default gray
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, f"Class {int(class_id)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return image

# -------------------- LOOP --------------------

for filename in os.listdir(image_dir):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {filename}")
        continue

    image_with_boxes = draw_yolo_boxes(image, label_path)

    # Save or display
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image_with_boxes)

    # Optional: also display
    # plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    # plt.title(filename)
    # plt.axis('off')
    # plt.show()

print("✅ Bounding box visualization complete.")