import cv2
import numpy as np
import logging
from .models import DetectionResult
from .preprocessing import ImagePreprocessor
from .feature_extraction import FeatureExtractor
from utils.config import config

logger = logging.getLogger(__name__)

class OpenCVDetector:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor()

    def detect(self, image_path: str) -> DetectionResult:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")

            # Preprocessing
            processed = self.preprocessor.preprocess(image)
            normalized = self.preprocessor.normalize_background(processed)
            binary = self.preprocessor.get_binary_mask(normalized)

            # Feature Extraction
            contour = self.feature_extractor.extract_contour(binary)
            
            if contour is None:
                return DetectionResult(
                    is_defective=True,
                    confidence=0.0,
                    detected_defects=["No stick detected"],
                    measurements={}
                )

            measurements = self.feature_extractor.get_measurements(contour, binary)
            straightness = self.feature_extractor.calculate_straightness(contour)
            measurements["straightness"] = straightness

            # Defect Logic
            defects = []
            
            # 1. Shape and Integrity
            if measurements["solidity"] < 0.95:
                defects.append("Cracked or irregular surface")
            
            # 2. Straightness and Curvature
            if measurements["straightness"] < config.straightness_threshold:
                defects.append("Bent or curved stick")

            # 3. Width Uniformity (Coating)
            if measurements["width_variance"] > config.width_variance_threshold:
                defects.append("Uneven coating or surface damage")

            # 4. Shape Anomalies (Missing sections)
            if measurements["shape_anomaly"] > config.shape_anomaly_threshold:
                defects.append("Missing sections or shape irregularities")

            # 5. Length Variation
            # Using aspect ratio as a proxy for relative length vs width
            if measurements["aspect_ratio"] < 10: 
                 defects.append("Broken stick or significant length variation")

            # 6. Surface detail check via Hough Lines
            # If too many lines are found inside, it might indicate cracks
            if measurements["hough_lines_count"] > 15:
                defects.append("Surface cracks detected")

            is_defective = len(defects) > 0
            
            # Heuristic confidence
            confidence = 1.0 if not defects else 0.85

            return DetectionResult(
                is_defective=is_defective,
                confidence=confidence,
                detected_defects=defects,
                measurements=measurements
            )

        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return DetectionResult(
                is_defective=True,
                confidence=0.0,
                detected_defects=[f"Error: {str(e)}"],
                measurements={}
            )
