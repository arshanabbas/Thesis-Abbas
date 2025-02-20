import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#color
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
]

# Function legend
def add_legend(ax, class_names):
    legend_patches = [plt.Line2D([0], [0], color=np.array(COLORS[i]) / 255, lw=5, label=f'Class {i}')
                      for i in range(len(class_names))]
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(class_names))

# Visualization
def visualize_segmentation(image_dir, annotation_dir, class_names, output_dir=None):
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    for annotation_file in os.listdir(annotation_dir):
        if not annotation_file.endswith(".txt"):
            continue

        image_name = os.path.splitext(annotation_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        annotation_path = os.path.join(annotation_dir, annotation_file)

        if not os.path.exists(image_path):
            print(f"Image {image_name} not found. Skipping...")
            continue

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse annotation file
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                polygon = list(map(float, parts[1:]))

                # Normalize and map coordinates to image dimensions
                points = [(int(polygon[i] * image.shape[1]), int(polygon[i + 1] * image.shape[0]))
                          for i in range(0, len(polygon), 2)]
                points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

                # Draw contour of the polygon
                cv2.polylines(image, [points], isClosed=True, color=COLORS[class_id % len(COLORS)], thickness=2)

                # Optionally, draw the class ID at the first point
                cv2.putText(image, f"{class_id}", points[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Plot with legend
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        ax.axis('off')
        add_legend(ax, class_names)

        # Save or display
        if output_dir:
            output_path = os.path.join(output_dir, image_name)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


#Use
image_dir = "F:/Arshan_Abbas/Fabian/Task3/Code/PolygontoYOLO/images"
annotation_dir = "F:/Arshan_Abbas/Fabian/Task3/Code/PolygontoYOLO/YOLOv8"
output_dir = "F:/Arshan_Abbas/Fabian/Task3/Code/PolygontoYOLO/Contours"
class_names = ["Hintergrund", "Metall", "Fusion", "Nebenbereich"]

visualize_segmentation(image_dir, annotation_dir, class_names, output_dir)