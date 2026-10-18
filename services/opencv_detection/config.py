from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class PreprocessingConfig:

    max_width: int = 1280

    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)

    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 0.0

    adaptive_block_size: int = 31
    adaptive_constant: float = 5.0


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    min_contour_area: float = 500.0
    max_contour_area_ratio: float = 0.80
    min_aspect_ratio: float = 4.0


@dataclass(frozen=True, slots=True)
class DefectCheckConfig:
    straightness: bool  = True
    broken: bool  = True
    missing_material: bool  = True
    abrupt_width_change: bool  = True
    shape_irregularity: bool  = True


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    # Straightness
    max_normalised_mean_deviation: float | None = None
    max_normalised_max_deviation: float | None = None

    # Broken / discontinued
    max_section_count: int | None  = None
    max_normalised_gap: float | None  = None

    # Missing material
    min_normalised_width: float | None  = None

    # Abrupt Width Changes
    max_width_drop: float | None  = None
    max_width_rise: float | None  = None

    # General Shape
    min_solidity: float | None  = None
    min_rectangularity: float | None  = None



@dataclass(frozen=True, slots=True)
class OpenCVDetectionConfig:
    check: DefectCheckConfig = field(default_factory=DefectCheckConfig)
    thresholds: QualityThresholds =field(default_factory=QualityThresholds)