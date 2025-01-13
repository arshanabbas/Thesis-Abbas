import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a color palette for the classes
COLORS = [
    (255, 0, 0),    # Red for Class 0
    (0, 255, 0),    # Green for Class 1
    (0, 0, 255),    # Blue for Class 2
    (255, 255, 0),  # Yellow for Class 3
]

# Function to add a legend
def add_legend(ax, classes):
    legend_patches = [plt.Line2D([0], [0], color=np.array(COLORS[i]) / 255, lw=5, label=f'Class {i}')
                      for i in range(len(classes))]
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(classes))

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

                # Draw filled polygon
                cv2.fillPoly(image, [points], COLORS[class_id % len(COLORS)])

                # Draw class ID at the first point
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


# Example usage
image_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/images"
annotation_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/YOLOv8"
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/CPDataset/visualcheck_Copy" 
class_names = ["Hintergrund", "Metall", "Nebenbereich", "Fusion"] 

visualize_segmentation(image_dir, annotation_dir, class_names, output_dir)