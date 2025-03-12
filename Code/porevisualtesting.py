import os
import cv2

# Class color mappings for visualization
CLASS_COLORS = {
    0: (0, 255, 0),   # Green for Pore (Class 0)
    1: (255, 0, 0)    # Red for Porennest (Class 1)
}

# Function to draw bounding boxes from YOLO annotations
def visualize_annotations(images_dir, annotations_dir, output_testing_dir):
    os.makedirs(output_testing_dir, exist_ok=True)

    for image_file in os.listdir(images_dir):
        if not image_file.endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(annotations_dir, os.path.splitext(image_file)[0] + ".txt")

        if not os.path.exists(label_path):
            print(f"Warning: Annotation for {image_file} not found. Skipping.")
            continue

        # Read image
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Read annotation file
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                # Convert YOLO format back to pixel coordinates
                x_center = int(cx * width)
                y_center = int(cy * height)
                box_width = int(w * width)
                box_height = int(h * height)

                x1 = x_center - box_width // 2
                y1 = y_center - box_height // 2
                x2 = x_center + box_width // 2
                y2 = y_center + box_height // 2

                color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Default white if class not found
                label_text = "Pore" if class_id == 0 else "Porennest"

                # Draw rectangle and label
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save annotated visualization
        output_path = os.path.join(output_testing_dir, image_file)
        cv2.imwrite(output_path, image)
        print(f"Saved visualization for {image_file}")

# ----------------------- Example Run -----------------------
images_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/image"
annotations_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/annotation"
output_testing_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/PolygontoYOLO/ErrorPlayground/pore_dataset/testing"

visualize_annotations(images_dir, annotations_dir, output_testing_dir)