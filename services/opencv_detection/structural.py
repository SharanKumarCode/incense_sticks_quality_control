from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from services.opencv_detection.geometry import GeometryResult
from services.opencv_detection.image_transform_helper import rotate_mask_horizontal, Contour, ImageArray


@dataclass(frozen=True, slots=True)
class HorizontalRun:
    start_x: int
    end_x: int

    @property
    def length_px(self) -> int:
        return self.end_x - self.start_x + 1


@dataclass(frozen=True, slots=True)
class BreakageResult:
    section_count: int
    gap_count: int

    sections: tuple[HorizontalRun, ...]
    gaps: tuple[HorizontalRun, ...]

    maximum_gap_px: int
    total_gap_px:  int

    normalized_maximum_gap: float

    rotated_mask: ImageArray
    corridor_mask: ImageArray

    corridor_top: int
    corridor_bottom: int


@dataclass(frozen=True, slots=True)
class WidthProfileResult:
    x_positions: NDArray[np.int32]
    widths_px: NDArray[np.float64]

    median_width_px: float
    mean_width_px: float
    min_width_px: float
    max_width_px: float

    valid_coulmns_count: int

    normalised_widths: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WidthAnomalyResult:
    minimum_normalised_width: float
    maxmimum_normalised_width: float

    maximum_width_drop: float
    maximum_width_raise: float

    x_positions: NDArray[np.int32]
    smoothed_normalised_widths: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ShapeIrregularityResult:
    solidity: float
    rectangularity: float

    contour_area_px: float
    convex_hull_area_px: float
    bounding_rectangle_area_px: float



def transform_point(x: float, y: float, transformation: NDArray[np.float64]) -> tuple[float, float]:
    point = np.array(
        [
            x,
            y,
            1.0
        ],
        dtype=np.float64
    )

    transformed = transformation @ point

    return (
        float(transformed[0]),
        float(transformed[1])
    )

def extract_analysis_corridor(
        rotated_mask: ImageArray,
        centre_y: float,
        estimated_width_px: float,
        width_multiplier: float = 3.0) -> tuple[ImageArray, int, int]:
    image_height, _ = rotated_mask.shape

    half_height = max(
        5,
        round(
            estimated_width_px * width_multiplier / 2.0
        )
    )

    centre_y_int = round(centre_y)

    top = max(0, centre_y_int - half_height)
    bottom = min(image_height, centre_y_int + half_height + 1)
    corridor = rotated_mask[top:bottom:].copy()

    return corridor, top, bottom

def calculate_column_occupancy(corridor: ImageArray, minimum_foreground_pixels: int = 3) -> NDArray[np.bool]:
    foreground_count = np.count_nonzero(corridor > 0, axis=0)
    return (
        foreground_count >= minimum_foreground_pixels
    )

def find_boolean_runs(values: NDArray[np.bool_], target_value: bool) -> list[HorizontalRun]:
    runs: list[HorizontalRun] = []

    run_start: int | None = None

    for index in range(len(values)):
        matches  = bool(values[index]) == target_value
        if matches:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            runs.append(
                HorizontalRun(
                    start_x=run_start,
                    end_x=index - 1
                )
            )

            run_start = None

    if run_start is not None:
        runs.append(HorizontalRun(
            start_x=run_start,
            end_x=len(values) - 1
        ))

    return runs

def filter_short_sections(sections: list[HorizontalRun], minimum_length_px: int) -> list[HorizontalRun]:
    return [
        section
        for section in sections
        if section.length_px >= minimum_length_px
    ]

def calculate_internal_gaps(sections: list[HorizontalRun]) ->  list[HorizontalRun]:
    if len(sections) < 2:
        return []

    gaps: list[HorizontalRun] = []

    for previous, current in zip(sections, sections[1:], strict=True):
        gap_start = previous.end_x + 1
        gap_end = current.start_x - 1

        if gap_start <= gap_end:
            gaps.append(
                HorizontalRun(
                    start_x=gap_start,
                    end_x=gap_end
                )
            )

    return gaps

def analyse_breakage(
        binary_mask: ImageArray,
        geometry: GeometryResult,
        *,
        corridor_width_multiplier: float = 3.0,
        minimum_section_length_px: int = 10
) -> BreakageResult:
    rotated_mask, rotation_matrix = (
        rotate_mask_horizontal(
            binary_mask,
            geometry.orientation_deg
        )
    )

    _, rotated_centre_y = transform_point(
        geometry.centre_x,
        geometry.centre_y,
        rotation_matrix
    )

    corridor, top, bottom = extract_analysis_corridor(
        rotated_mask=rotated_mask,
        centre_y=rotated_centre_y,
        estimated_width_px=geometry.width_px,
        width_multiplier=corridor_width_multiplier
    )

    cv2.imwrite(
        "/home/sharan/Documents/python/incense_stick_quality_control/runs/opencv_detection/geometry/check/debug_rotated.png",
        rotated_mask
    )

    cv2.imwrite(
            "/home/sharan/Documents/python/incense_stick_quality_control/runs/opencv_detection/geometry/check/debug_corridor.png",
            corridor
        )


    occupancy = calculate_column_occupancy(corridor)

    sections = find_boolean_runs(occupancy, target_value=True)
    sections = filter_short_sections(sections, minimum_length_px=minimum_section_length_px)
    gaps = calculate_internal_gaps(sections)

    maximum_gap= max(
        (
            gap.length_px
            for gap in gaps
        ),
        default=0
    )

    total_gap = sum(
        (
                    gap.length_px
                    for gap in gaps
                )
    )

    normalised_maximum_gap = maximum_gap / geometry.length_px if geometry.length_px > 0 else 0.0

    corridor_mask = np.zeros_like(rotated_mask)

    corridor_mask[top:bottom, :] = rotated_mask[top:bottom,  :]

    return BreakageResult(
        section_count=len(sections),
        gap_count=len(gaps),

        sections=tuple(sections),
        gaps=tuple(gaps),

        maximum_gap_px=maximum_gap,
        total_gap_px=total_gap,

        normalized_maximum_gap=normalised_maximum_gap,

        rotated_mask=rotated_mask,
        corridor_mask=corridor_mask,

        corridor_top=top,
        corridor_bottom=bottom
    )


def calculate_width_profile(rotated_stick_mask: ImageArray) -> WidthProfileResult:
    x_positions: list[int] =[]
    widths: list[float] = []

    _, image_width =  rotated_stick_mask.shape

    for x in range(image_width):

        y_values = np.flatnonzero(rotated_stick_mask[:, x] > 0)
        if len(y_values) == 0:
            continue

        top_y = int(y_values[0])
        bottom_y = int(y_values[-1])

        width = bottom_y - top_y + 1

        x_positions.append(x)
        widths.append(float(width))

    if not widths:
        raise ValueError("Could not calculate stick width profile")

    width_array = np.asarray(widths, dtype=np.float64)
    x_array = np.asarray(x_positions, dtype=np.int32)

    median_width = float(np.median(width_array))

    normalised_widths = width_array / median_width if median_width > 0 else np.zeros_like(width_array)

    return WidthProfileResult(
        x_positions=x_array,
        widths_px=width_array,
        median_width_px=median_width,
        mean_width_px=float(np.mean(width_array)),
        min_width_px=float(np.min(width_array)),
        max_width_px=float(np.max(width_array)),
        valid_coulmns_count=len(width_array),
        normalised_widths=normalised_widths
    )

def get_interior_profile(profile: WidthProfileResult, edge_fraction: float = 0.05) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    count = len(profile.widths_px)

    edge_count = round(count * edge_fraction)

    if edge_count <= 0:
        return profile.x_positions, profile.normalised_widths

    if edge_count * 2 >= count:
        raise ValueError("edg_fraction removes the entire width profile")

    return profile.x_positions[edge_count:-edge_count],  profile.normalised_widths[edge_count:-edge_count]


def smooth_width_profile(widths: NDArray[np.float64], window_size: int = 7) -> NDArray[np.float64]:
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    if window_size == 1:
        return widths.copy()

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    if window_size > len(widths):
        raise ValueError("window_size cannot exceed profile length")

    padding = window_size // 2
    padded = np.pad(widths, (padding, padding), mode="edge")

    kernel = np.ones(window_size, dtype=np.float64) / window_size

    smoothed = np.convolve(padded, kernel, mode="valid")

    if len(smoothed) != len(widths):
        raise RuntimeError(f"Smoothed width profile length mismatch: Input= {len(widths)} ; Output={len(smoothed)}")

    return smoothed


def analyse_width_anomalies(profile: WidthProfileResult) -> WidthAnomalyResult:
    x_positions, normalised_widths = get_interior_profile(profile)


    smoothed = smooth_width_profile(normalised_widths)

    differences = np.diff(smoothed)

    maximum_drop = float(abs(min(np.min(differences), 0.0)))

    maximum_rise = float(abs(max(np.max(differences), 0.0)))

    return WidthAnomalyResult(
        minimum_normalised_width=float(np.min(smoothed)),
        maxmimum_normalised_width=float(np.max(smoothed)),

        maximum_width_drop=maximum_drop,
        maximum_width_raise=maximum_rise,

        x_positions=x_positions,

        smoothed_normalised_widths=smoothed
    )


def calculate_solidity(contour: Contour) -> tuple[float, float, float]:
    contour_area = float(cv2.contourArea(contour))

    hull= cv2.convexHull(contour)

    hull_area = float(cv2.contourArea(hull))

    solidity = contour_area / hull_area if hull_area > 0 else  0.0

    return solidity, contour_area, hull_area


def calculate_rectangularity(contour: Contour) -> tuple[float, float]:
    rectangle = cv2.minAreaRect(contour)

    _, size, _ = rectangle

    rectangle_width, rectangle_height= size

    rectangle_area = float(rectangle_width * rectangle_height)

    contour_area = float(cv2.contourArea(contour))

    rectangularity = contour_area / rectangle_area if rectangle_area > 0 else 0.0

    return rectangularity, rectangle_area


def analyse_shape_irregularity(contour: Contour) -> ShapeIrregularityResult:

    solidity, contour_area, hull_area = calculate_solidity(contour=contour)

    rectangularity, rectangle_area = calculate_rectangularity(contour=contour)

    return ShapeIrregularityResult(
        solidity=solidity,
        rectangularity=rectangularity,
        contour_area_px=contour_area,
        convex_hull_area_px=hull_area,
        bounding_rectangle_area_px=rectangle_area
    )