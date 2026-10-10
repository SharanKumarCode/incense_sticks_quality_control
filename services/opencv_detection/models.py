from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class DetectionResult:
    is_defective: bool
    confidence: float
    detected_defects: List[str]
    measurements: Dict[str, float]
    processed_image_path: Optional[str] = None
