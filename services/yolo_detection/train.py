from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"
RUNS_DIR = PROJECT_ROOT / "runs" / "yolo-detection"


def train() -> None:
    """Train YOLOv8 model on the dataset."""
    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=str(DATASET_DIR),
        epochs=30,
        imgsz=320,
        batch=8,
        device="cpu",
        patience=8,
        project=str(RUNS_DIR),
        name="baseline",
        exist_ok=True,
        seed=42,
        workers=0,
        verbose=True,
    )

if __name__ == "__main__":
    train()
