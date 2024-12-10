import numpy as np
import matplotlib.pyplot as plt

# Load the mask
mask_path = "F:/Pomodoro/Work/TIME/Script/Thesis-Abbas-Segmentation/Mask/0a05af59-10-19_mask.npy"  
mask = np.load(mask_path)

# Plot each channel 
for i in range(mask.shape[0]):
    plt.figure()
    plt.title(f"Class {i} Mask")
    plt.imshow(mask[i], cmap='gray')
    plt.axis("off")
    plt.show()
