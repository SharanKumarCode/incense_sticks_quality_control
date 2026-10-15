from pathlib import Path

from ultralytics import YOLO

from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score
    )
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "runs" / "yolo-detection" / "baseline" / "weights" / "best.pt"

TEST_DIR = PROJECT_ROOT / "dataset" / "test"

OUTPUT_DIR = PROJECT_ROOT / "runs" / "yolo-detection" / "evaluation"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def get_test_images() -> list[tuple[Path, str]]:

    images: list[tuple[Path, str]] = []

    for label in ["good", "defective"]:
        class_dir = TEST_DIR / label

        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory '{class_dir}' does not exist.")
        
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((image_path, label))
    
    return images


def evaluate() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' does not exist. Please train the model first.")

    test_images = get_test_images()

    if not test_images:
        raise ValueError(f"No test images found in '{TEST_DIR}'.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating model '{MODEL_PATH}' on {len(test_images)} test images...")

    model = YOLO(str(MODEL_PATH))

    y_true: list[str] = []
    y_pred: list[str] = []

    misclassified:list[tuple[Path, str, str, float]] = []

    print("\nStarting predictions on test images...\n")

    for image_path, actual_label in test_images:
        results = model.predict(source=str(image_path), device="cpu", verbose=False)
        result = results[0]

        class_id = result.probs.top1
        confidence = float(result.probs.top1conf)

        predicted_label = result.names[class_id] if class_id in result.names else "Unknown"

        y_true.append(actual_label)
        y_pred.append(predicted_label)

        correct = actual_label == predicted_label
        status = "PASS" if correct else "FAIL"

        print(f"Image: {image_path.name} | Actual: {actual_label} | Predicted: {predicted_label} | Confidence: {confidence:.2f} | Status: {status}")

        if not correct:
            misclassified.append((image_path, actual_label, predicted_label, confidence))


    print()
    print("=" * 50)
    print("Evaluation Results:")
    print("=" * 50)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="defective", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="defective", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="defective", zero_division=0)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Defective Recall: {recall:.4f}")
    print(f"Defective F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=["good", "defective"], zero_division=0)
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=["good", "defective"])
    good_as_good = cm[0, 0]
    good_as_defective = cm[0, 1]
    defective_as_good = cm[1, 0]
    defective_as_defective = cm[1, 1]

    print("\nConfusion Matrix:")
    print(f"Good as Good: {good_as_good}")
    print(f"Good as Defective: {good_as_defective}")
    print(f"Defective as Good: {defective_as_good}")
    print(f"Defective as Defective: {defective_as_defective}")

    total_defective = defective_as_good + defective_as_defective
    if total_defective > 0:
        false_negative_rate = defective_as_good / total_defective
        print(f"False Negative Rate (Defective as Good): {false_negative_rate:.4f}")

    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["good", "defective"])
    display.plot(cmap=plt.cm.Blues)
    plt.title("Incense Stick  YOLO Classification")

    plt.tight_layout()
    confusion_matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=150)

    plt.close()

    print(f"\nConfusion matrix saved to: {confusion_matrix_path}")

    print("=" * 50)
    print("Misclassified Images:")
    print("=" * 50)

    if not misclassified:
        print("No misclassified images.")
    else:
        for image_path, actual_label, predicted_label, confidence in misclassified:
            print(f"Image: {image_path.name} | Actual: {actual_label} | Predicted: {predicted_label} | Confidence: {confidence:.2f}")


if __name__ == "__main__":
    evaluate()
    