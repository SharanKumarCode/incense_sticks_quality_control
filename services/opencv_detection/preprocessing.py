from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from services.opencv_detection.config import PreprocessingConfig

ImageArray = NDArray[np.uint8]

def load_image(image_path: str | Path) -> ImageArray:
    """Load an image from a file path.

    Args:
        image_path (str | Path): The path to the image file.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image


def resize_image(image: ImageArray, max_width: int) -> ImageArray:
    """Resize an image to a maximum width while maintaining aspect ratio.

    Args:
        image (ImageArray): The input image.
        max_width (int): The maximum width for the resized image.
    """
    height, width = image.shape[:2]

    if width <= max_width:
        return image.copy()  # Return a copy of the original image if width does not exceed max_width

    scale = max_width / width
    new_width = max_width
    new_height  = round(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def to_grayscale(image: ImageArray) -> ImageArray:
    """Convert an image to grayscale.

    Args:
        image (ImageArray): The input image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(grayscale: ImageArray, config: PreprocessingConfig) -> ImageArray:
    """Enhance the contrast of a grayscale image using CLAHE.

    Args:
        grayscale (ImageArray): The input grayscale image.
        config (PreprocessingConfig): The preprocessing configuration.
    """
    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_grid_size)
    return clahe.apply(grayscale)


def blur_image(image: ImageArray, config: PreprocessingConfig) -> ImageArray:
    """Apply Gaussian blur to an image.

    Args:
        image (ImageArray): The input image.
        config (PreprocessingConfig): The preprocessing configuration.
    """
    kernel = config.gaussian_kernel_size
    return cv2.GaussianBlur(image, (kernel, kernel), config.gaussian_sigma)


def create_otsu_mask(image: ImageArray, inverted: bool = False) -> ImageArray:
    """Create a binary mask using Otsu's thresholding method.

    Args:
        image (ImageArray): The input grayscale image.
        inverted (bool): Whether to invert the mask.
    """
    threshold_type = (
        cv2.THRESH_BINARY_INV if inverted else cv2.THRESH_BINARY
    )
    _, mask = cv2.threshold(image, 0, 255, threshold_type | cv2.THRESH_OTSU)
    return mask


def create_adaptive_mask(image: ImageArray, config: PreprocessingConfig, inverted: bool = False) -> ImageArray:
    """Create a binary mask using adaptive thresholding.

    Args:
        image (ImageArray): The input grayscale image.
        config (PreprocessingConfig): The preprocessing configuration.
        inverted (bool): Whether to invert the mask.
    """
    threshold_type = (
        cv2.THRESH_BINARY_INV if inverted else cv2.THRESH_BINARY
    )
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        config.adaptive_block_size,
        config.adaptive_constant,
    )


class ImagePreprocessor:
    """Interface for image preprocessing classes."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config  or PreprocessingConfig()

    def preprocess(self, image_path: str | Path) -> dict[str, ImageArray]:
        """Preprocess an image.

        Args:
            image_path (str | Path): The path to the input image.
        """

        original_image = load_image(image_path)
        resized_image = resize_image(original_image, self.config.max_width)
        grayscale_image = to_grayscale(resized_image)
        enhanced_image = enhance_contrast(grayscale_image, self.config)
        blurred_image = blur_image(enhanced_image, self.config)

        return {
            "original": original_image,
            "resized": resized_image,
            "grayscale": grayscale_image,
            "enhanced": enhanced_image,
            "blurred": blurred_image,

            "otsu": create_otsu_mask(blurred_image),
            "otsu_inverted": create_otsu_mask(blurred_image, inverted=True),
            "adaptive": create_adaptive_mask(blurred_image, self.config),
            "adaptive_inverted": create_adaptive_mask(blurred_image, self.config, inverted=True)
        }