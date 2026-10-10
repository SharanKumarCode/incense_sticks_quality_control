import cv2
import numpy as np
from .models import DetectionResult

class Visualizer:
    @staticmethod
    def draw_results(image_path: str, result: DetectionResult) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            return np.zeros((500, 500, 3), dtype=np.uint8)

        color = (0, 0, 255) if result.is_defective else (0, 255, 0)
        status = "DEFECTIVE" if result.is_defective else "GOOD"
        
        # Overlay status
        cv2.putText(
            image, 
            f"Status: {status} ({result.confidence:.2f})", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            color, 
            2
        )
        
        # Overlay defects
        for i, defect in enumerate(result.detected_defects):
            cv2.putText(
                image, 
                f"- {defect}", 
                (20, 80 + (i * 30)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                color, 
                2
            )
            
        return image

    @staticmethod
    def save_result(image: np.ndarray, output_path: str):
        cv2.imwrite(output_path, image)
