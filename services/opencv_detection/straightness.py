import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from services.opencv_detection.segmentation import Contour, create_stick_mask
from services.opencv_detection.image_transform_helper import rotate_mask_horizontal, ImageArray



@dataclass(frozen=True, slots=True)
class StraightnessResult:
    mean_deviation_px:  float
    max_deviation_px: float

    normalised_mean_deviation: float
    normalised_max_deviation: float

    slope: float
    intercept: float

    centreline_points: NDArray[np.float64]

    rotated_mask: ImageArray


def extract_centreline(
        mask:  ImageArray
) -> NDArray[np.float64]:
    points: list[tuple[float, float]] = []

    _, width  = mask.shape

    for x in range(width):
        y_values = np.flatnonzero(
            mask[:, x] > 0
        )

        if len(y_values) == 0:
            continue

        centre_y = float(
            np.mean(y_values)
        )

        points.append((
            float(x),
            centre_y
        ))

    if not points:
        return np.empty((0, 2), dtype=np.float64)

    return np.asarray(
        points,
        dtype=np.float64
    )

def fit_centreline(
        points: NDArray[np.float64]
) ->  tuple[float,  float]:

    if len(points) < 2:
        raise ValueError(
            "Atleast 2 centreline points are required"
        )

    x = points[:, 0]
    y = points[:, 1]

    slope, intercept = np.polyfit(x, y, 1)

    return (float(slope), float(intercept))


def calculate_deviations(
        points:  NDArray[np.float64],
        slope: float,
        intercept: float
) ->  NDArray[np.float64]:

    x = points[:, 0]
    y = points[:, 1]

    predicted_y = slope * x + intercept

    deviations = np.abs(y - predicted_y)

    return deviations


def analyse_straightness(
        contour: Contour,
        image_shape: tuple[int, ...],
        orientation_deg: float,
        length_px: float
) -> StraightnessResult:
    mask = create_stick_mask(
        image_shape,
        contour
    )

    rotated_mask, _ = rotate_mask_horizontal(
        mask,
        orientation_deg
    )

    points = extract_centreline(
        rotated_mask
    )

    if len(points) < 2:
        raise ValueError("Could not extract enough centreline points")

    slope, intercept = fit_centreline(
        points
    )

    deviations = calculate_deviations(
        points,
        slope,
        intercept
    )

    mean_deviation = float(np.mean(deviations))
    max_deviation = float(np.max(deviations))

    if length_px > 0:
        normalised_mean = mean_deviation / length_px
        normalised_max = max_deviation / length_px
    else:
        normalised_mean = 0.0
        normalised_max = 0.0

    return StraightnessResult(
        mean_deviation_px=mean_deviation,
        max_deviation_px=max_deviation,
        normalised_mean_deviation=normalised_mean,
        normalised_max_deviation=normalised_max,
        slope=slope,
        intercept=intercept,
        centreline_points=points,
        rotated_mask=rotated_mask
    )
