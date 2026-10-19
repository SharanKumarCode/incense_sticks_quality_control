import logging
from pathlib import Path

from services.opencv_detection.config import (
    DefectCheckConfig,
    OpenCVDetectionConfig,
    QualityThresholds
)

from services.opencv_detection.detector import OpenCVDetector

from services.yolo_detection.config import YOLODetectionConfig

from services.yolo_detection.detector import YOLODetector

from services.quality_inspection.inspector import QualityInspector, QualityInspectionResult

from utils.logger import configure_logging, get_logger

import json


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "runs" / "yolo-detection" / "baseline" / "weights" / "best.pt"

logger = get_logger(__name__)


def _get_all_good_images():
    good_images_path = PROJECT_ROOT / "input" / "good"
    if not good_images_path.exists():
        raise Exception("Path does not exist")

    return list(good_images_path.glob("*.jpg"))


def _get_all_defective_images():
    defective_images_path = PROJECT_ROOT / "input" / "damaged"
    if not defective_images_path.exists():
        raise Exception("Path does not exist")

    return list(defective_images_path.glob("*.jpg"))


def _write_output(image_path: str | Path, actual_status: str, result: QualityInspectionResult):
    output_file_path = PROJECT_ROOT / "output" / "output_parallel.json"
    
    # Make sure the output folder actually exists first, genius
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "image_path": str(image_path),
        "actual_status": actual_status,
        "elapsed_time_s": result.elapsed_time_s,
        "inspection_id": result.inspection_id,
        "yolo_status": result.yolo.class_name,
        "opencv_status": result.opencv.quality.status
    }

    # Load existing or start a fresh structure
    if output_file_path.exists() and output_file_path.is_file():
        with open(output_file_path, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        # Use dictionary syntax, not dot notation!
        json_data["results"].append(result_dict)
    else:
        json_data = {"results": [result_dict]}

    # Write the updated data back out to disk
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4)


def main() -> None:
    logger.info("Application started", extra={
        "event": "application.started"
    })

    yolo_detector = YOLODetector(
        YOLODetectionConfig(
            model_path=MODEL_PATH,
            device="cpu"
        )
    )

    opencv_detector = OpenCVDetector(
        OpenCVDetectionConfig(
            check=DefectCheckConfig(
                straightness=True,
                broken=True,
                missing_material=True,
                abrupt_width_change=True,
                shape_irregularity=True
            ),
            thresholds=QualityThresholds(
                max_normalised_mean_deviation=0.015,
                max_normalised_max_deviation=0.050,

                max_section_count=1,
                max_normalised_gap=0.020,

                min_normalised_width=0.65,

                max_width_drop=0.10,
                max_width_rise=0.15,

                min_solidity=0.85,
                min_rectangularity=0.70
            )
        )
    )

    good_images = _get_all_good_images()
    good_images_count = len(good_images)
    defective_images = _get_all_defective_images()
    defective_images_count = len(defective_images)

    with QualityInspector(
        yolo_detector=yolo_detector,
        opencv_detector=opencv_detector) as inspector:

        for index, image_path in enumerate(good_images):
            logger.info(f"Inspecting good image {str(image_path)} : {index} / {good_images_count}")

            result = inspector.inspect(image_path=image_path)

            _write_output(image_path=image_path, actual_status="pass", result=result)

            logger.info(f"Inspection completed for good image {index} / {good_images_count}")


        for index, image_path in enumerate(defective_images):
            logger.info(f"Inspecting defective image {str(image_path)} : {index} / {defective_images_count}")

            result = inspector.inspect(image_path=image_path)

            _write_output(image_path=image_path, actual_status="reject", result=result)

            logger.info(f"Inspection completed for defective image {index} / {defective_images_count}")


    logger.info("Inspection completed")


if __name__  == "__main__":
    configure_logging(level=logging.INFO)

    main()