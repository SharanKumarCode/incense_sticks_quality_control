from pydantic_settings import BaseSettings
from pathlib import Path

class DetectorConfig(BaseSettings):
    # Image processing parameters
    gaussian_blur_ksize: tuple[int, int] = (5, 5)
    gaussian_blur_sigma: float = 0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    
    # Thresholding
    adaptive_thresh_block_size: int = 11
    adaptive_thresh_c: int = 2
    
    # Morphological operations
    morph_kernel_size: int = 3
    
    # Defect detection thresholds
    length_min_threshold: float = 0.8  # Relative to mean of good sticks
    length_max_threshold: float = 1.2
    width_variance_threshold: float = 0.15
    straightness_threshold: float = 0.92
    curvature_max_threshold: float = 0.08
    missing_segment_threshold: float = 0.05
    shape_anomaly_threshold: float = 0.1
    
    # Paths
    data_dir: Path = Path("data")
    good_dir: str = "good"
    defective_dir: str = "defective"

    class Config:
        env_prefix = "OPENCV_DETECTOR_"

config = DetectorConfig()
