import os
import pandas as pd

# Directories
images_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/SMPDataset/images"
masks_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/SMPDataset/masks"

# Get sorted lists of file paths
image_files = sorted(os.listdir(images_dir))
mask_files = sorted(os.listdir(masks_dir))

# Create a list of dictionaries
data = []
for img, mask in zip(image_files, mask_files):
    data.append({
        "image_path": os.path.join(images_dir, img),
        "mask_path": os.path.join(masks_dir, mask)
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV
csv_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/SMPDataset/dataset_paths.csv"
df.to_csv(csv_path, index=False)

print(f"CSV file saved to {csv_path}")