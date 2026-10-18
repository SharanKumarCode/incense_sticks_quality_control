import cv2
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from services.opencv_detection.config import SegmentationConfig

ImageArray = NDArray[np.uint8]
Contour = NDArray[np.int32]

segmentation_config = SegmentationConfig()

@dataclass(frozen=True, slots=True)
class ContourCandidate:
    contour: Contour
    area: float
    x: int
    y: int
    width: int
    height: int

    aspect_ratio: float
    image_area_ratio: float


def find_external_contours(mask: ImageArray) -> list[Contour]:
    """Find external contours in a binary image.

    Args:
        mask (ImageArray): The input binary image.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def describe_contours(contours: list[Contour]) -> None:
    """Describe contours by calculating their area and perimeter.

    Args:
        contours (list[Contour]): A list of contours.
    """
    print(f"Contours found: {len(contours)}")

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for index, contour in enumerate(sorted_contours[:10]):
        area = cv2.contourArea(contour)

        x, y, width, height = cv2.boundingRect(contour)

        print(
            f"{index}: "
            f"Area: {area:.2f}, "
            f"Bounding Box: (x: {x}, y: {y}, width: {width}, height: {height})"
        )

def analyse_contour(contour: Contour, image_shape: tuple[int, ...]) -> ContourCandidate:
    """Analyse a contour and return its properties.

    Args:
        contour (Contour): The input contour.
        image_shape (tuple[int, ...]): The shape of the image (height, width).
    """
    area = cv2.contourArea(contour)
    x, y, width, height = cv2.boundingRect(contour)

    long_side  = max(width, height)
    short_side = max(1, min(width, height))

    aspect_ratio = long_side / short_side

    image_height, image_width = image_shape[:2]

    image_area = image_height * image_width
    image_area_ratio = area / image_area if image_area > 0 else 0.0

    return ContourCandidate(
        contour=contour,
        area=area,
        x=x,
        y=y,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        image_area_ratio=image_area_ratio
    )

def is_plausible_stick(candidate: ContourCandidate) -> bool:
    """Determine if a contour candidate is a plausible stick based on its properties.

    Args:
        candidate (ContourCandidate): The contour candidate to evaluate.
    """
    print(candidate.area, segmentation_config.min_contour_area)
    if candidate.area < segmentation_config.min_contour_area:
        return False

    if candidate.aspect_ratio < segmentation_config.min_aspect_ratio:
        return False

    if candidate.image_area_ratio > segmentation_config.max_contour_area_ratio:
        return False

    return True

def find_stick_candidates(mask: ImageArray) -> list[ContourCandidate]:
    """Find plausible stick candidates in a binary mask.

    Args:
        mask (ImageArray): The input binary mask.
    """
    contours = find_external_contours(mask)
    candidates = [analyse_contour(contour, mask.shape) for contour in contours]
    plausible_candidates = [candidate for candidate in candidates if is_plausible_stick(candidate)]

    return sorted(plausible_candidates, key=lambda c: c.area, reverse=True)

def draw_candidates(image: ImageArray, candidates: list[ContourCandidate]) -> ImageArray:
    """Draw bounding boxes around plausible stick candidates on the image.

    Args:
        image (ImageArray): The input image.
        candidates (list[ContourCandidate]): A list of plausible stick candidates.
    """
    output_image = image.copy()

    for index, candidate in enumerate(candidates):
        colour = (0, 255, 0) if index == 0 else (0, 165, 255) # Orange for others

        cv2.drawContours(
            output_image,
            [candidate.contour],
            -1,
            colour,
            2
        )

        cv2.rectangle(
            output_image,
            (candidate.x, candidate.y),
            (candidate.x + candidate.width, candidate.y + candidate.height),
            colour,
            2
        )

        cv2.putText(
            output_image,
            f"# {index} AR={candidate.aspect_ratio:.2f}",
            (candidate.x, max(candidate.y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
            cv2.LINE_AA
        )

    return output_image


def select_primary_stick(candidates: list[ContourCandidate]) -> ContourCandidate | None:
    """Select the primary stick candidate based on the largest area.

    Args:
        candidates (list[ContourCandidate]): A list of plausible stick candidates.
    """
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda c: c.area, reverse=True)
    return candidates[0]  # The candidates are already sorted by area in descending order


def create_stick_mask(image_shape: tuple[int, ...], contour: Contour) -> ImageArray:
    """Create a binary mask for the stick based on its contour.

    Args:
        image_shape (tuple[int, ...]): The shape of the image (height, width).
        contour (Contour): The contour of the stick.
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask