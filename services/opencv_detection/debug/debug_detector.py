from pathlib import Path

from services.opencv_detection.config import DefectCheckConfig, OpenCVDetectionConfig, QualityThresholds
from services.opencv_detection.detector import OpenCVDetector

from utils.logger import configure_logging, get_logger, inspection_id
from uuid import uuid4

from time import perf_counter_ns


logger = get_logger(__name__)

IMAGE_PATH = Path("/home/sharan/Documents/python/incense_stick_quality_control/dataset/train/good/20260728_064445.jpg")

# IMAGE_PATH = Path("/home/sharan/Documents/python/incense_stick_quality_control/dataset/train/defective/IMG-20260728-WA0006.jpg")

def main():

    start_time = perf_counter_ns()

    token = inspection_id.set(str(uuid4()))

    try:
        logger.info(f"Starting OpenCV detection of the image at path : {IMAGE_PATH}")

        config = OpenCVDetectionConfig(
            check=DefectCheckConfig(
                straightness=True,
                broken=False,
                missing_material=False,
                abrupt_width_change=False,
                shape_irregularity=False
            ),
            thresholds=QualityThresholds(
                max_normalised_max_deviation=1.0,
                max_section_count=100,
                max_normalised_gap=1.0
            )
        )

        detector = OpenCVDetector(config)

        result = detector.inspect(IMAGE_PATH)

        logger.info("=" * 50)
        logger.info("QUALITY RESULT")
        logger.info("=" * 50)

        logger.info(f"Status : {result.quality.status}")

        for finding in result.quality.findings:
            logger.error(
                f"{finding.defect_type}: "
                f"{finding.observed_value:.5f} "
                f"(threshold="
                f"{finding.threshold_value:.5f}"
            )

    finally:
        end_time = perf_counter_ns() - start_time
        logger.info(f"Time taken to complete OpenCV detection : {end_time / 1_000_000_000} secs")
        inspection_id.reset(token)


if __name__ == "__main__":
    configure_logging()
    main()
