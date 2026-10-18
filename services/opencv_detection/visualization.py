import cv2
import numpy as np

from services.opencv_detection.geometry import GeometryResult, get_rotated_box_points
from services.opencv_detection.straightness import StraightnessResult
from services.opencv_detection.structural import BreakageResult, WidthProfileResult, WidthAnomalyResult, ShapeIrregularityResult
from services.opencv_detection.image_transform_helper import Contour, ImageArray


def draw_geometry(
        image: ImageArray,
        contour: Contour,
        geometry: GeometryResult
) -> ImageArray:
    output = image.copy()

    cv2.drawContours(
        output,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    box = get_rotated_box_points(contour)

    cv2.polylines(
        output,
        [box],
        isClosed=True,
        color=(255, 0, 0),
        thickness=2
    )

    centre = np.array(
        [
            geometry.centre_x,
            geometry.centre_y
        ],
        dtype=np.float64
    )

    centre_point = (
        round(geometry.centre_x),
        round(geometry.centre_y)
    )

    # Draw centre point
    cv2.circle(
        output,
        centre_point,
        5,
        (0, 0, 255),
        -1
    )

    # Convert orientation to a unit direction vector
    angle_rad = np.deg2rad(
        geometry.orientation_deg
    )

    direction = np.array(
        [
            np.cos(angle_rad),
            np.sin(angle_rad)
        ]
    )

    # Perpendicular direction represents width
    perpendicular =  np.array(
        [
            -direction[1],
            direction[0]
        ]
    )

    # Major axis extends half the measured length from the centre in both directions
    half_length = geometry.length_px / 2.0

    major_start = centre - (direction * half_length)
    major_end = centre + (direction * half_length)

    # Minor axis extends half the measured width
    half_width = geometry.width_px / 2.0

    minor_start = centre - (perpendicular * half_width)
    minor_end = centre  + (perpendicular * half_width)

    major_start_pt = tuple(np.round(major_start).astype(int))
    major_end_pt = tuple(np.round(major_end).astype(int))

    minor_start_pt = tuple(np.round(minor_start).astype(int))
    minor_end_pt = tuple(np.round(minor_end).astype(int))

    # Major axis: yellow
    cv2.line(
        output,
        major_start_pt,
        major_end_pt,
        (0, 255, 255),
        2
    )

    # Minor axis: magenta
    cv2.line(
        output,
        minor_start_pt,
        minor_end_pt,
        (255, 0, 255),
        2
    )

    # Display measurements
    text_lines = [
        f"Length : {geometry.length_px:.1f} px",
        f"Width : {geometry.width_px:.1f} px",
        f"AR : {geometry.aspect_ratio:.2f}",
        f"Angle : {geometry.orientation_deg:.1f} deg"
    ]

    x  =  20
    y = 30

    for line in text_lines:
        cv2.putText(
            output,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 30

    return output


def draw_straightness(result: StraightnessResult) -> ImageArray:

    # Convert the binary mask into BGR so we can overlay coloured diagnostic information

    output = cv2.cvtColor(
        result.rotated_mask,
        cv2.COLOR_GRAY2BGR
    )

    points = result.centreline_points

    if len(points) < 2:
        return output

    # Draw extracted centreline
    for x, y in points:
        cv2.circle(
            output,
            (
                round(x),
                round(y)
            ),
            1,
            (0, 255, 0),
            -1
        )

    # Draw fitted straight line
    x_start = float(points[0, 0])
    x_end = float(points[-1, 0])
        
    y_start = result.slope * x_start + result.intercept
    y_end = result.slope * x_end + result.intercept

    cv2.line(
        output,
        (
            round(x_start),
            round(y_start)
        ),
        (
            round(x_end),
            round(y_end)
        ),
        (255, 0, 0),
        2
    )

    # Find maximum deviation
    x_values = points[:, 0]
    y_values = points[:, 1]

    fitted_y  = (result.slope * x_values + result.intercept)

    deviations = np.abs(y_values - fitted_y)

    max_index = int(np.argmax(deviations))

    max_x = float(x_values[max_index])

    actual_y = float(y_values[max_index])

    ideal_y = float(fitted_y[max_index])

    # Actual Centraline point
    cv2.circle(
        output,
        (
            round(max_x),
            round(actual_y)
        ),
        6,
        (0, 0, 255),
        -1
    )

    # Draw deviation line from ideal line
    cv2.line(
        output,
        (
            round(max_x),
            round(actual_y)
        ),
        (
            round(max_x),
            round(ideal_y)
        ),
        (0, 0, 255),
        2
    )

    #  Diagnostic values
    text_lines = [
        f"Mean deviation: {result.mean_deviation_px:.2f} px",
        f"Max deviation: {result.max_deviation_px:.2f} px",
        f"Normalised mean: {result.normalised_mean_deviation:.5f}",
        f"Normalised max: {result.normalised_max_deviation:.5f}",
    ]

    y = 30

    for text in text_lines:
        cv2.putText(
            output,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 20

    return output


def draw_breakage(result: BreakageResult)-> ImageArray:
    output = cv2.cvtColor(
        result.rotated_mask,
        cv2.COLOR_GRAY2BGR
    )

    image_height,_ = output.shape[:2]

    # Analyse corridor boundaries
    cv2.line(
        output,
        (0, result.corridor_top),
        (
            output.shape[1] - 1,
            result.corridor_top
        ),
        (255, 0, 255),
        1
    )

    cv2.line(
        output,
        (0, result.corridor_bottom),
                (
                    output.shape[1] - 1,
                    result.corridor_bottom - 1
                ),
                (255, 0, 255),
                1
    )

    # Valid occupied sections
    for section in result.sections:
        cv2.rectangle(
            output,
            (
                section.start_x,
                result.corridor_top
            ),
            (
                section.end_x,
                result.corridor_bottom - 1
            ),
            (0, 255, 0),
            2
        )

    # Internal gaps
    for gap in result.gaps:
        cv2.rectangle(
            output,
            (
                gap.start_x,
                result.corridor_top
            ),
            (
                gap.end_x,
                result.corridor_bottom - 1
            ),
            (0, 0, 255),
            2
        )

    text_lines = [
        f"Sections: {result.section_count}",
        f"Gaps: {result.gap_count}",
        f"Max Gap: {result.maximum_gap_px} px",
        f"Total gap: {result.total_gap_px} px",
        f"Normalised maximum gap : {result.normalized_maximum_gap:.5f}"
    ]

    y = 30

    for text in text_lines:
        cv2.putText(
            output,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 28

    return output


def draw_width_profile(rotated_stick_mask: ImageArray, result: WidthProfileResult) -> ImageArray:
    output = cv2.cvtColor(
            rotated_stick_mask,
            cv2.COLOR_GRAY2BGR
        )

    for x in result.x_positions:
        y_values = np.flatnonzero(rotated_stick_mask[:, x] > 0)

        if len(y_values) == 0:
            continue

        top_y = int(y_values[0])
        bottom_y = int(y_values[-1])

        cv2.line(
            output,
            (int(x), top_y),
            (int(x), bottom_y),
            (0, 255,  0),
            1
        )

    text_lines = [
        f"Median width: {result.median_width_px:.1f} px",
        f"Mean width: {result.mean_width_px:.1f} px",
        f"Minimum width: {result.min_width_px:.1f} px",
        f"Maximum width: {result.max_width_px:.1f} px",
    ]

    y = 30
    for text in text_lines:
        cv2.putText(
            output,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 28

    return output


def draw_width_anomalies(rotated_stick_mask: ImageArray, profile: WidthProfileResult, anomalies: WidthAnomalyResult) -> ImageArray:
    output = cv2.cvtColor(
                rotated_stick_mask,
                cv2.COLOR_GRAY2BGR
            )

    if len(anomalies.x_positions) == 0:
        return output

    # Minimum Width location
    min_index = int(np.argmin(anomalies.smoothed_normalised_widths))

    min_x = int(anomalies.x_positions[min_index])

    min_y_values = np.flatnonzero(rotated_stick_mask[:, min_x] >  0)

    if len(min_y_values) > 0:
        cv2.line(
            output,
            (
                min_x,
                int(min_y_values[0])
            ),
            (
                min_x,
                int(min_y_values[-1])
            ),
            (0, 0, 255),
            3
        )

    # Largest abrupt width drop
    differences = np.diff(anomalies.smoothed_normalised_widths)


    if len(differences) > 0:
        drop_index = int(np.argmin(differences))

        drop_x = int(anomalies.x_positions[drop_index + 1])

        drop_y_values = np.flatnonzero(rotated_stick_mask[:, drop_x] > 0)

        cv2.line(
                output,
                (
                    drop_x,
                    int(drop_y_values[0])
                ),
                (
                    drop_x,
                    int(drop_y_values[-1])
                ),
                (0, 156, 255),
                2
            )
        
    text_lines = [
        f"Min normalised width: {anomalies.minimum_normalised_width:.3f}",
        f"Max normalised width: {anomalies.maxmimum_normalised_width:.3f}",
        f"Max width drop: {anomalies.maximum_width_drop:.3f}",
        f"Max width rise: {anomalies.maximum_width_raise:.3f}"
    ]

    y = 30
    for text in text_lines:
        cv2.putText(
            output,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 28

    return output


def draw_shape_irregularity(
        image: ImageArray,
        contour: Contour,
        result: ShapeIrregularityResult
) -> ImageArray:
    
    output = image.copy()

    # Actual contour: green
    cv2.drawContours(
        output,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    # Convex hull: orange
    hull = cv2.convexHull(
        contour
    )

    cv2.drawContours(
            output,
            [hull],
            -1,
            (0, 165, 255),
            2
        )

    # Minimum-area rectangle:  blue
    rectangle = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rectangle)

    box = np.int32(box)

    cv2.polylines(
        output,
        [box],
        True,
        (255, 0, 0),
        2
    )

    text_lines =  [
        f"Solidity: {result.solidity:.4f}",
        f"Rectangulariy: {result.rectangularity:.4f}"
    ]

    y = 30
    for text in text_lines:
        cv2.putText(
            output,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 28

    return output