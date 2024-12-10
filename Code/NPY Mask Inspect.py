import numpy as np

# Load the mask file
mask_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Mask/0a05af59-10-19_mask.npy"  # Replace with your mask file path
mask = np.load(mask_path)

# Check each channel for non-zero values
for i in range(mask.shape[0]):
    print(f"Class {i}: {np.sum(mask[i] > 0)} pixels")
