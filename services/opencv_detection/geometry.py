from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from services.opencv_detection.image_transform_helper import Contour


@dataclass(frozen=True, slots=True)
class GeometryResult:
    area_px: float
    perimeter_px: float

    length_px: float
    width_px: float

    aspect_ratio: float
    orientation_deg: float

    centre_x: float
    centre_y: float


def extract_geometry(contour: Contour) -> GeometryResult:
    """
    Extracts geometric properties
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    rectangle = cv2.minAreaRect(contour)

    centre, size, angle = rectangle
    centre_x, centre_y = centre
    raw_width, raw_height = size
    length = max(raw_width, raw_height)
    width = min(raw_width, raw_height)
    aspect_ratio = length / width if width >  0 else 0.0

    orientation = angle + 90.0 if raw_width < raw_height else angle

    if orientation >=90.0:
        orientation -= 180.0

    if orientation < -90.0:
        orientation += 180.0

    return GeometryResult(
        area_px=area,
        perimeter_px=perimeter,
        length_px=length,
        width_px=width,
        aspect_ratio=aspect_ratio,
        orientation_deg=orientation,
        centre_x=centre_x,
        centre_y=centre_y
    )

def get_rotated_box_points(contour: Contour) -> NDArray[np.int32]:
    rectangle  = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rectangle)

    return np.int32(box)
    