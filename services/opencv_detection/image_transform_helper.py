import cv2
import numpy as np
from numpy.typing import NDArray


ImageArray = NDArray[np.uint8]
Contour = NDArray[np.uint32]


def rotate_mask_horizontal(
        mask: ImageArray,
        orientation_deg: float
):
    height, width = mask.shape[:2]

    centre = (
        width / 2.0,
        height / 2.0
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        centre,
        orientation_deg,
        1.0
    )

    rotated = cv2.warpAffine(
        mask,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated, rotation_matrix