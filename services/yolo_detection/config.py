from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class YOLODetectionConfig:
    model_path: Path
    device: str = "cpu"

