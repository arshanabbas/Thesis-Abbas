import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset class for semantic segmentation.
    Reads image and mask file paths from a CSV file and loads them as PyTorch tensors.
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image and mask paths.
            transform (callable, optional): Optional transform to be applied on an image-mask pair.
        """
        self.data = pd.read_csv(csv_file)  # Load CSV file
        self.transform = transform         # Transformation for data augmentation

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Fetches a single sample (image and mask) by index."""
        # Get file paths from the CSV
        image_path = self.data.iloc[idx]['image_path']
        mask_path = self.data.iloc[idx]['mask_path']

        # Load image and mask
        image = Image.open(image_path).convert("RGB")  # Load image as RGB
        mask = np.load(mask_path)                     # Load mask as a numpy array

        # Convert to numpy arrays
        image = np.array(image)

        # If the mask has multiple channels, reduce to single-channel
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = np.argmax(mask, axis=0)  # Take the argmax to collapse channels

        mask = mask.astype(np.int64)  # Ensure the mask is of type int64 (class indices)

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        mask = torch.tensor(mask, dtype=torch.long)  # Mask of shape [H, W]

        return image, mask


if __name__ == "__main__":
    # Path to your CSV file
    csv_file = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/SMPDataset/dataset_paths.csv"

    # Initialize the dataset without transforms
    dataset = SegmentationDataset(csv_file=csv_file, transform=None)

    # Fetch a single sample for testing
    image, mask = dataset[0]

    # Print details for verification
    print("Image Tensor Shape:", image.shape)  # Expected: [3, H, W]
    print("Image Tensor Type:", image.dtype)  # Expected: torch.float32
    print("Mask Tensor Shape:", mask.shape)   # Expected: [H, W]
    print("Mask Tensor Type:", mask.dtype)    # Expected: torch.int64
    print("Unique Mask Values (Classes):", torch.unique(mask))  # Check unique class indices

    # Visualization for all available classes
    image_np = image.permute(1, 2, 0).numpy() / 255.0  # Normalize image for display
    mask_np = mask.numpy()

    # Get all unique classes in the mask
    unique_classes = torch.unique(mask)
    print(f"Available Classes in the Mask: {unique_classes.tolist()}")

    # Plot the original image and masks for each class
    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, len(unique_classes) + 1, 1)
    plt.imshow(image_np)
    plt.title("Original Image")
    plt.axis("off")

    # Loop through and visualize each class
    for i, target_class in enumerate(unique_classes):
        class_mask = (mask_np == target_class.item()).astype(np.uint8)  # Binary mask for the class

        plt.subplot(1, len(unique_classes) + 1, i + 2)
        plt.imshow(class_mask, cmap="gray")
        plt.title(f"Class {target_class.item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()