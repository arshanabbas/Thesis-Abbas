import numpy as np
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
import os

def create_multi_channel_masks(annotation_file, output_dir, num_classes):
    # Load COCO annotations
    print("Creating index...")
    coco = COCO(annotation_file)
    print("Index created!")

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all images
    for img_id in coco.getImgIds():
        # Load image metadata
        img = coco.loadImgs(img_id)[0]
        width, height = img['width'], img['height']
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        # Initialize an empty multi-channel mask
        mask = np.zeros((num_classes, height, width), dtype=np.uint8)

        # Loop
        for ann in anns:
            class_id = ann['category_id']  # Get the class ID (channel index)
            rle = mask_utils.frPyObjects(ann['segmentation'], height, width)
            binary_mask = mask_utils.decode(rle).astype(np.uint8)

            # Use np.where()
            coords = np.where(binary_mask > 0)
            mask[class_id, coords[0], coords[1]] = 1

        # Save the multi-channel mask as a .npy file
        mask_file_name = f"{img['file_name'].split('.')[0]}_mask.npy"
        mask_file_path = os.path.join(output_dir, mask_file_name)
        np.save(mask_file_path, mask)

        print(f"Saved mask for image {img['file_name']} to {mask_file_path}")

# Example Usage
annotation_file = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas/CPDataset/result.json" 
output_dir = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Mask"  
num_classes = 4 

create_multi_channel_masks(annotation_file, output_dir, num_classes)