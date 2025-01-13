import os
from ultralytics import YOLO

# Load the pre-trained model
model = YOLO("D:/Abbas/GitHub/PolygontoYOLO/yolo11n-seg.pt")

# Path to the dataset configuration file
data = 'C:/Users/arab/Documents/GitHub/Thesis-Abbas/Code/dataset.yaml'

# Start training
results = model.train(data= "D:/Abbas/GitHub/Thesis-Abbas/Code/dataset.yaml", epochs=1, imgsz=1024)