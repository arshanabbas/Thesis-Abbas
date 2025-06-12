from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    train_results = model.train(
        data="train.yaml",  # Path to the dataset configuration file
        epochs=500,  # Set the number of training epochs
        imgsz=640,  # Image size for training (input size for model)
        device=0,  # Specify the device (0 for the first GPU, 'cpu' for CPU)
    )

    # After training, evaluate the model's performance on the validation set
    metrics = model.val()

    # Optionally, print or save evaluation metrics
    print(metrics)

if __name__ == '__main__':
    main()