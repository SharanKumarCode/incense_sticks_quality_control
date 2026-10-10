import os
import time
from pathlib import Path
from typing import List, Tuple
from .detector import OpenCVDetector
from ...utils.config import config

class Evaluator:
    def __init__(self):
        self.detector = OpenCVDetector()

    def run_evaluation(self, data_dir: Path) -> dict:
        good_dir = data_dir / config.good_dir
        defective_dir = data_dir / config.defective_dir
        
        results = [] # List of (true_label, predicted_label)
        
        # Process Good Samples
        if good_dir.exists():
            for img_path in good_dir.glob("*.jpg"):
                res = self.detector.detect(str(img_path))
                results.append((False, res.is_defective)) # True label is NOT defective (False)

        # Process Defective Samples
        if defective_dir.exists():
            for img_path in defective_dir.glob("*.jpg"):
                res = self.detector.detect(str(img_path))
                results.append((True, res.is_defective)) # True label is defective (True)

        return self.calculate_metrics(results)

    def calculate_metrics(self, results: List[Tuple[bool, bool]]) -> dict:
        tp = sum(1 for true, pred in results if true and pred)
        tn = sum(1 for true, pred in results if not true and not pred)
        fp = sum(1 for true, pred in results if not true and pred)
        fn = sum(1 for true, pred in results if true and not pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(results) if results else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "confusion_matrix": {
                "tp": tp, "tn": tn, "fp": fp, "fn": fn
            }
        }

if __name__ == "__main__":
    evaluator = Evaluator()
    metrics = evaluator.run_evaluation(config.data_dir)
    print(f"Evaluation Metrics: {metrics}")
