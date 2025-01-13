import os
from ultralytics import YOLO

# Load the pre-trained model
model = YOLO("yolo11n-seg.pt")  # Ensure the model file is correctly downloaded and accessible

# Path to the dataset configuration file
data = 'C:/Users/arab/Documents/GitHub/Thesis-Abbas/Code/dataset.yaml'

# Start training
results = model.train(data=data, epochs=1, imgsz=1024)  # Ensure the arguments match the expected API