from ultralytics import YOLO

# Load the pre-trained model
model = YOLO("yolo11n-seg.pt")  # Ensure the model file is correctly downloaded and accessible

# Path to the dataset configuration file
data = 'F:/Arshan_Abbas/Fabian/Task3/Code/Thesis-Abbas/Code/dataset.yaml'

# Start training
results = model.train(data=data, epochs=1, imgsz=640)  # Ensure the arguments match the expected API
