from ultralytics import YOLO
from pathlib import Path

def main():
    # Load YOLO model
    model = YOLO("yolo11n.pt")

    # Define safe output directory
    save_project_dir = Path("runs/detect").resolve()
    run_name = "train11"

    # Train the model
    model.train(
        data="train.yaml",
        epochs=350,
        imgsz=640,
        device=0,
        name=run_name,
        project=str(save_project_dir),
        save=True,
    )

    # Validate the model and save plots
    metrics = model.val(save=True)

    # Show evaluation metrics
    print(metrics)

if __name__ == '__main__':
    main()
