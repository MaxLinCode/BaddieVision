from pathlib import Path

from ultralytics import YOLO

DATASET_DIR = Path(__file__).resolve().parent

model = YOLO("yolov8n.pt")  # Start from small pretrained model
model.train(
    data=str(DATASET_DIR / "data.yaml"),
    epochs=30,
    imgsz=640,
    batch=8,
    name="shuttle_detector"
)
