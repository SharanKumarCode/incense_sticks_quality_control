from dataclasses import dataclass
from pathlib import Path
from enum import StrEnum

from utils.logger import get_logger

from exceptions.opencv_exceptions import (
    AnalysisError,
    InvalidGeometryError,
    NoStickDetectedError
)
from services.opencv_detection.config import OpenCVDetectionConfig
from services.opencv_detection.preprocessing import ImagePreprocessor
from services.opencv_detection.segmentation import create_stick_mask, find_stick_candidates, select_primary_stick
from services.opencv_detection.image_transform_helper import rotate_mask_horizontal

from services.opencv_detection.geometry import GeometryResult, extract_geometry
from services.opencv_detection.straightness import StraightnessResult, analyse_straightness
from services.opencv_detection.structural import (
    BreakageResult,
    WidthAnomalyResult,
    WidthProfileResult,
    ShapeIrregularityResult,
    analyse_breakage,
    analyse_width_anomalies,
    analyse_shape_irregularity,
    calculate_width_profile,
)
from services.opencv_detection.quality import (
    QualityEvaluation,
    QualityFinding,
    QualityStatus,
    evaluate_straightness,
    evaluate_breakage,
    evaluate_abrupt_width_changes,
    evaluate_missing_material,
    evaluate_shape_irregularity
)

logger = get_logger(__name__)

class InspectionErrorType(StrEnum):
    IMAGE_PROCESSING_FAILED = "image_processing_failed"
    NO_STICK_DETECTED = "no_stick_detected"
    INVALID_GEOMETRY = "invalid_geometry"
    ANALYSIS_FAILED = "analysis_failed"


@dataclass(frozen=True, slots=True)
class InspectionError:
    error_type: InspectionErrorType
    message: str


@dataclass(frozen=True, slots=True)
class DetectionMeasurements:
    geometry: GeometryResult

    straightness: StraightnessResult  | None = None
    breakage: BreakageResult  | None = None

    width_profile: WidthProfileResult  | None = None
    width_anomalies: WidthAnomalyResult  | None = None

    shape_irregularity: ShapeIrregularityResult  | None = None


@dataclass(frozen=True, slots=True)
class OpenCVDetectionResult:
    quality: QualityEvaluation
    measurements: DetectionMeasurements | None = None

    error: InspectionError | None = None


class OpenCVDetector:

    def __init__(self, config: OpenCVDetectionConfig) -> None:
        self.config = config
        self.preprocessor = ImagePreprocessor()


    def _evaluate_quality(self, measurements: DetectionMeasurements) -> QualityEvaluation:
        findings: list[QualityFinding] = []
        requires_review = False

        thresholds = self.config.thresholds
        checks = self.config.check

        if checks.straightness:
            if measurements.straightness is None:
                requires_review = True
            elif thresholds.max_normalised_mean_deviation is None and thresholds.max_normalised_max_deviation is None:
                requires_review = True
            else:
                findings.extend(
                    evaluate_straightness(
                        measurements.straightness,
                        thresholds
                    )
                )

        if checks.broken:
            if measurements.breakage is None:
                requires_review = True
            elif thresholds.max_section_count is None and thresholds.max_normalised_gap is None:
                requires_review = True
            else:
                findings.extend(
                    evaluate_breakage(
                        measurements.breakage,
                        thresholds
                    )
                )

        if checks.missing_material:
            if measurements.width_anomalies is None:
                requires_review = True
            elif thresholds.min_normalised_width is None:
                requires_review = True
            else:
                findings.extend(
                    evaluate_missing_material(
                        measurements.width_anomalies,
                        thresholds
                    )
                )

        if checks.abrupt_width_change:
            if measurements.width_anomalies is None:
                requires_review = True
            elif thresholds.max_width_drop is None and thresholds.max_width_rise is None:
                requires_review = True
            else:
                findings.extend(
                    evaluate_abrupt_width_changes(
                        measurements.width_anomalies,
                        thresholds
                    )
                )

        if checks.shape_irregularity:
            if measurements.shape_irregularity is None:
                requires_review = True
            elif thresholds.min_solidity is None and thresholds.min_rectangularity is None:
                requires_review = True
            else:
                findings.extend(
                    evaluate_shape_irregularity(
                        measurements.shape_irregularity,
                        thresholds
                    )
                )


        if findings:
            status = QualityStatus.REJECT
        elif requires_review:
            status = QualityStatus.REVIEW
        else:
            status = QualityStatus.PASS

        return QualityEvaluation(
            status=status,
            findings=tuple(findings)
        )

    @staticmethod
    def _inspection_error(
        error_type: InspectionErrorType,
        message: str
    ) -> OpenCVDetectionResult:
        return OpenCVDetectionResult(
            quality=QualityEvaluation(
                status=QualityStatus.INSPECTION_ERROR,
                findings=()
            ),
            measurements=None,
            error=InspectionError(
                error_type=error_type,
                message=message
            )
        )


    def measure(self, image_path:  str | Path) -> DetectionMeasurements:

        results  = self.preprocessor.preprocess(image_path)
        
        mask = results["adaptive_inverted"]

        candidates = find_stick_candidates(mask)

        primary = select_primary_stick(candidates)

        if primary is None:
            raise NoStickDetectedError("No incense stick candidate detected in the image")

        geometry = extract_geometry(primary.contour)

        if geometry.length_px <= 0 or geometry.width_px <= 0:
            raise InvalidGeometryError("Detected stick produced invalid geometry")

        straightness: StraightnessResult | None = None
        breakage: BreakageResult | None = None
        width_profile: WidthProfileResult | None = None
        width_anomalies: WidthAnomalyResult | None = None
        shape_irregularity: ShapeIrregularityResult | None = None


        if self.config.check.straightness:
            try:
                straightness = analyse_straightness(
                    contour=primary.contour,
                    image_shape=results["resized"].shape,
                    orientation_deg=geometry.orientation_deg,
                    length_px=geometry.length_px
                )
            except ValueError as error:
                raise AnalysisError(
                    analysis_name="Straightness",
                    message=error
                )

        if self.config.check.broken:
            breakage = analyse_breakage(
                binary_mask=mask,
                geometry=geometry
            )

        needs_width_analysis = self.config.check.missing_material or self.config.check.abrupt_width_change

        if needs_width_analysis:
            stick_mask = create_stick_mask(
                results["resized"].shape,
                primary.contour
            )

            rotated_stick_mask, _ = rotate_mask_horizontal(
                stick_mask, geometry.orientation_deg
            )

            try:

                width_profile = calculate_width_profile(rotated_stick_mask)
                width_anomalies = analyse_width_anomalies(width_profile)

            except ValueError as error:
                raise AnalysisError(
                    analysis_name="Width",
                    message=error
                )

        if self.config.check.shape_irregularity:
            shape_irregularity = analyse_shape_irregularity(primary.contour)


        measurements = DetectionMeasurements(
            geometry=geometry,
            straightness=straightness,
            breakage=breakage,
            width_profile=width_profile,
            width_anomalies=width_anomalies,
            shape_irregularity=shape_irregularity
        )

        return measurements
    

    def inspect(self, image_path: str | Path) -> OpenCVDetectionResult:

        try:

            measurements = self.measure(image_path)

        except NoStickDetectedError as error:
            return self._inspection_error(
                error_type=InspectionErrorType.NO_STICK_DETECTED,
                message=str(error)
            )
        except InvalidGeometryError as error:
            return self._inspection_error(
                error_type=InspectionErrorType.INVALID_GEOMETRY,
                message=str(error)
            )
        except AnalysisError as error:
            return self._inspection_error(
                error_type=InspectionErrorType.ANALYSIS_FAILED,
                message=str(error)
            )

        quality = self._evaluate_quality(measurements)

        return OpenCVDetectionResult(
            quality=quality,
            measurements=measurements
        )