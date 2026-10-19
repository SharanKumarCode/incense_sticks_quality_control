from dataclasses import dataclass
from enum import StrEnum

from services.opencv_detection.config import QualityThresholds
from services.opencv_detection.straightness import StraightnessResult
from services.opencv_detection.structural import BreakageResult, WidthAnomalyResult, ShapeIrregularityResult


class QualityStatus(StrEnum):
    PASS = "pass"
    REJECT = "reject"
    REVIEW = "review"
    INSPECTION_ERROR = "inspection_error"


class DefectType(StrEnum):
    STRAIGTHNESS = "straightness"
    BROKEN = "broken"
    MISSING_MATERIAL = "missing_material"
    ABRUPT_WIDTH_CHANGE = "abrupt_width_change"
    SHAPE_IRREGULARITY = "shape_irregularity"


@dataclass(frozen=True, slots=True)
class QualityFinding:
    defect_type: DefectType

    observed_value: float
    threshold_value: float

    message: str


@dataclass(frozen=True, slots=True)
class QualityEvaluation:
    status: QualityStatus
    findings: tuple[QualityFinding, ...]


def evaluate_straightness(result: StraightnessResult, thresholds: QualityThresholds) -> list[QualityFinding]:
    findings: list[QualityFinding] = []

    max_deviation_threshold = thresholds.max_normalised_max_deviation

    if max_deviation_threshold is not None and result.normalised_max_deviation > max_deviation_threshold:
        findings.append(
            QualityFinding(
                defect_type=DefectType.STRAIGTHNESS,
                observed_value=result.normalised_max_deviation,
                threshold_value=max_deviation_threshold,
                message="Maximum normalised straightness deviation exceeded the configured limit"
            )
        )

    mean_deviation_threshold = thresholds.max_normalised_mean_deviation
    
    if mean_deviation_threshold is not None and result.normalised_mean_deviation > mean_deviation_threshold:
        findings.append(
            QualityFinding(
                defect_type=DefectType.STRAIGTHNESS,
                observed_value=result.normalised_mean_deviation,
                threshold_value=mean_deviation_threshold,
                message="Mean normalised straightness deviation exceeded the configured limit"
            )
        )

    return findings


def evaluate_breakage(result: BreakageResult, thresholds: QualityThresholds) -> list[QualityFinding]:
    findings: list[QualityFinding] = []

    section_threshold = thresholds.max_section_count

    if section_threshold is not None and result.section_count > section_threshold:
        findings.append(
            QualityFinding(
                defect_type=DefectType.BROKEN,
                observed_value=result.section_count,
                threshold_value=section_threshold,
                message="Detected stick section count exceeded the configured limit"
            )
        )

    gap_threshold = thresholds.max_normalised_gap

    if gap_threshold is not None and result.normalized_maximum_gap > gap_threshold:
        findings.append(
            QualityFinding(
                defect_type=DefectType.BROKEN,
                observed_value=result.normalized_maximum_gap,
                threshold_value=gap_threshold,
                message="Maximum normalised gap exceeded the configured limit"
            )
        )

    return findings


def evaluate_missing_material(result: WidthAnomalyResult, thresholds: QualityThresholds) -> list[QualityFinding]:

    threshold = thresholds.min_normalised_width

    if threshold is None:
        return []

    if result.minimum_normalised_width >= threshold:
        return  []

    return [
        QualityFinding(
            defect_type=DefectType.MISSING_MATERIAL,
            observed_value=result.minimum_normalised_width,
            threshold_value=threshold,
            message="Minimum normalised width fell below the configured limit"
        )
    ]


def evaluate_abrupt_width_changes(result: WidthAnomalyResult, thresholds: QualityThresholds) -> list[QualityFinding]:
    findings: list[QualityFinding] = []

    drop_threshold = thresholds.max_width_drop

    if drop_threshold is not None and result.maximum_width_drop > drop_threshold:
        findings.append(
            QualityFinding(
                defect_type=DefectType.ABRUPT_WIDTH_CHANGE,
                observed_value=result.maximum_width_drop,
                threshold_value=drop_threshold,
                message="Maximum width drop exceeded the configured limit"
            )
        )

    rise_threshold = thresholds.max_width_rise

    if rise_threshold is not None and result.maximum_width_raise > rise_threshold:
        findings.append(
            QualityFinding(
                defect_type=DefectType.BROKEN,
                observed_value=result.maximum_width_raise,
                threshold_value=rise_threshold,
                message="Maximum width rise exceeded the configured limit"
            )
        )

    return findings


def evaluate_shape_irregularity(result: ShapeIrregularityResult, thresholds: QualityThresholds) -> list[QualityFinding]:
    findings: list[QualityFinding] = []

    solidity_threshold = thresholds.min_solidity

    if solidity_threshold is not None and result.solidity < solidity_threshold:
        findings.append(
            QualityFinding(
                defect_type=DefectType.SHAPE_IRREGULARITY,
                observed_value=result.solidity,
                threshold_value=solidity_threshold,
                message="Solidity fell below the configured limit"
            )
        )

    rectangularity_threshold = thresholds.min_rectangularity

    if rectangularity_threshold is not None and result.rectangularity < rectangularity_threshold:
        findings.append(
            QualityFinding(
                defect_type=DefectType.BROKEN,
                observed_value=result.rectangularity,
                threshold_value=rectangularity_threshold,
                message="Rectangularity fell below the configured limit"
            )
        )

    return findings