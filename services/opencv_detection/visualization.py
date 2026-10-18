import cv2
import numpy as np

from services.opencv_detection.geometry import GeometryResult, get_rotated_box_points
from services.opencv_detection.segmentation import Contour,  ImageArray


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