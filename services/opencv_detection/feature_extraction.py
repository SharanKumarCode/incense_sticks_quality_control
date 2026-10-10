import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import distance as dist

class FeatureExtractor:
    @staticmethod
    def extract_contour(binary_mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Return largest contour assuming one stick per image
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
        size = np.size(binary_mask)
        skel = np.zeros(binary_mask.shape, np.uint8)
        
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        
        img = binary_mask.copy()
        
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            
            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True
        return skel

    @staticmethod
    def get_hough_lines_score(image: np.ndarray) -> float:
        # Expects edges or binary image
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is None:
            return 0.0
        return len(lines)

    @staticmethod
    def calculate_width_variance(contour: np.ndarray) -> float:
        # Sample widths along the fitted line
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)

        box = box.astype(np.int64)
        
        # Simple heuristic: distance of contour points to central axis
        [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        distances = []
        for point in contour:
            px, py = point[0]
            d = abs((px - x0) * vy - (py - y0) * vx)
            distances.append(d)
        
        return float(np.var(distances))

    @staticmethod
    def get_measurements(contour: np.ndarray, binary_mask: np.ndarray) -> dict:
        # Bounding box and stats
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect
        length = max(w, h)
        avg_width = min(w, h)
        
        # Convex Hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        solidity = float(contour_area) / hull_area if hull_area > 0 else 0
        
        # Skeletonization for refined length
        # (Crop mask to contour for speed)
        skel = FeatureExtractor.skeletonize(binary_mask)
        skel_points = cv2.countNonZero(skel)
        
        # Hough Lines on the mask
        hough_score = FeatureExtractor.get_hough_lines_score(binary_mask)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Shape anomaly: deviation from a perfect rectangle
        expected_area = length * avg_width
        shape_anomaly = 1.0 - (contour_area / expected_area) if expected_area > 0 else 1.0
        
        width_variance = FeatureExtractor.calculate_width_variance(contour)

        return {
            "length": float(length),
            "skel_length": float(skel_points),
            "avg_width": float(avg_width),
            "width_variance": width_variance,
            "solidity": float(solidity),
            "perimeter": float(perimeter),
            "shape_anomaly": float(shape_anomaly),
            "hough_lines_count": float(hough_score),
            "aspect_ratio": float(length / avg_width) if avg_width > 0 else 0
        }

    @staticmethod
    def calculate_straightness(contour: np.ndarray) -> float:
        # Distance from points to the fitted line
        [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        distances = []
        for point in contour:
            px, py = point[0]
            # Distance from point to line: |(x-x0)*vy - (y-y0)*vx| / sqrt(vx^2 + vy^2)
            d = abs((px - x0) * vy - (py - y0) * vx)
            distances.append(d)
        
        return 1.0 / (1.0 + np.mean(distances)) # Higher is straighter
