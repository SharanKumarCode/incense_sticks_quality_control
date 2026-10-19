Act as a Principal Computer Vision Engineer.
Create an OpenCV-based defect detection module for a Python 3.14.4 application that uses uv as the package manager.
The code must be created ONLY inside:
services/opencv_detection/
Goal
Detect mechanical defects in incense sticks from still mages.
Each image contains exactly one incense stick.
Primary defects:
• Broken stick - Bent stick ⁃ Cracked stick Nissing sections Significant length variation Shape irregularities
Secondary defects:
Missing coating Uneven coating - Surface damage
Dataset assumptions:
data/good/ data/defective/
Images come from a mobile phone.
Lighting varies moderately.
Background varies moderately.
Requirements
Implement:
1. Dataset Loader 
2. Image Preprocessor 
3. Background Normalizer 
4. Contour Extractor 
5. Feature Extractor 
6. Defect Detector 
7. Evaluation Module 
8. Visualization Module

Use OpenCV techniques such as:
• Gaussian Blur _ CLAHP - Adaptive Thresholding - Morphological perations - Edge Detection - Contour Analysis - Convex Hull - Skeletonization - Hough Lines
Feature measurements:
Calculate:
• Estimated stick length - Width variance - Straightness score - Curvature score - Missing segment score - Shape anomaly score
Detection Output
Define a dataclass:
DetectionResult
Fields:
is_defective: bool - confidence: float - detected_defects: list[str] - measurements: dict
Architecture
Create:
services/opencv_detection/ init.py detector.py preprocessing.py feature_extraction.py evaluation.py visualization.py models.py config.py
Detector Interface
class OpenCVDetector:
def detect(self, image_path: str) -> DetectionResult
Evaluation
Generate scripts for:
• Precision - Recall -F1 - Confusion Matrix
Code Quality
Python 3.14.4 Pydantic configuration - Type hints Logging - Clean architecture - No hardcoded paths
136 Do not generate code outside 137 services/opencv_detection/.

