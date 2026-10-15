from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "runs" / "yolo-detection" / "baseline" / "weights" / "best.pt"

def predict(image_path: str) -> list:
    """Predict the class of an image using the trained model."""
    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=image_path, 
                            device="cpu", 
                            verbose=False)

    result = results[0]     # Get the first result (assuming single image input)

    class_id = result.probs.top1
    confidence = float(result.probs.top1conf)

    class_name = result.names[class_id] if class_id in result.names else "Unknown"

    print(f"Prediction : {class_name} (Confidence: {confidence:.2f})")


if __name__ == "__main__":
    test_image_path = PROJECT_ROOT / "dataset" / "test" / "good" / "20260728_064107.jpg"
    predict(str(test_image_path))
    test_image_path = PROJECT_ROOT / "dataset" / "test" / "good" / "20260728_064207.jpg"
    predict(str(test_image_path))
    test_image_path = PROJECT_ROOT / "dataset" / "test" / "defective" / "IMG-20260728-WA0000.jpg"
    predict(str(test_image_path))
    test_image_path = PROJECT_ROOT / "dataset" / "test" / "defective" / "IMG-20260728-WA0001.jpg"
    predict(str(test_image_path))