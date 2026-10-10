import cv2
import numpy as np
from utils.config import config

class ImagePreprocessor:
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit, 
            tileGridSize=config.clahe_tile_grid_size
        )
        enhanced = clahe.apply(gray)
        
        # Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(
            enhanced, 
            config.gaussian_blur_ksize, 
            config.gaussian_blur_sigma
        )
        
        return blurred

    @staticmethod
    def normalize_background(image: np.ndarray) -> np.ndarray:
        # Simple background subtraction/normalization
        # Morphological opening to estimate background
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Subtract background
        normalized = cv2.absdiff(image, background)
        
        # Invert if background is dark (usually incense sticks are dark on light background or vice versa)
        # We assume stick is darker than background for this normalization or we use adaptive thresholding later
        return normalized

    @staticmethod
    def get_binary_mask(image: np.ndarray) -> np.ndarray:
        # Adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            config.adaptive_thresh_block_size, 
            config.adaptive_thresh_c
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.morph_kernel_size, config.morph_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
