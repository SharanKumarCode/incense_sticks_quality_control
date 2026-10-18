from dataclasses import dataclass

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