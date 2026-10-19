from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path

from utils.logger import get_logger, inspection_id
from uuid import uuid4
from time import perf_counter_ns

from services.yolo_detection.detector import YOLOClassificationResult, YOLODetector
from services.opencv_detection.detector import OpenCVDetectionResult, OpenCVDetector

logger = get_logger(__name__)

yolo_context = copy_context()
opencv_context = copy_context()


@dataclass(frozen=True, slots=True)
class QualityInspectionResult:
    yolo: YOLOClassificationResult
    opencv: OpenCVDetectionResult

    inspection_id: str
    elapsed_time_s: float


class QualityInspector:

    def __init__(self, yolo_detector: YOLODetector, opencv_detector: OpenCVDetector) -> None:
        self.yolo_detector = yolo_detector
        self.opencv_detector = opencv_detector

        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="quality_inspection"
        )


    def inspect(self, image_path: str | Path) -> QualityInspectionResult:

        current_inspection_id = str(uuid4())
        token = inspection_id.set(current_inspection_id)

        try:
            start_time = perf_counter_ns()

            logger.info("Quality inspection started", extra={
                "event": "quality_inspection.started",
                "fields": {
                    "image_path": image_path
                }
            })

            yolo_start = perf_counter_ns()

            yolo_future = self._executor.submit(
                yolo_context.run,
                self.yolo_detector.predict,
                image_path
            )

            opencv_start = perf_counter_ns()

            opencv_future = self._executor.submit(
                opencv_context.run,
                self.opencv_detector.inspect,
                image_path
            )

            yolo_result = yolo_future.result()

            yolo_elapsed_time_ms = (perf_counter_ns() - yolo_start) / 1_000_000

            opencv_result = opencv_future.result()

            opencv_elapsed_time_ms = (perf_counter_ns() - opencv_start) / 1_000_000

            total_elapsed_time_ms = yolo_elapsed_time_ms + opencv_elapsed_time_ms

            logger.info(f"Total elapsed time ms : {total_elapsed_time_ms}")

            end_time = perf_counter_ns()
            elapsed_time = end_time - start_time

            logger.info("Quality inspection completed", extra={
                "event": "quality_inspection.completed",
                "fields": {
                    "yolo_class": yolo_result.class_name,
                    "yolo_confidence": yolo_result.confidence,
                    "opencv_status": opencv_result.quality.status,
                    "elapsed_time_secs": elapsed_time / 1e9,
                }
            })

            logger.info(f"Inspection completed in {elapsed_time / 1e9} secs")

            return QualityInspectionResult(
                yolo=yolo_result,
                opencv=opencv_result,
                inspection_id=current_inspection_id,
                elapsed_time_s=elapsed_time / 1e9
            )
        finally:
            inspection_id.reset(token)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "QualityInspector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()