from pathlib import Path
from dataclasses import dataclass

from ultralytics import YOLO

from utils.logger import get_logger
from services.yolo_detection.config import YOLODetectionConfig

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class YOLOClassificationResult:
    class_id: int
    class_name: str
    confidence: float


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "runs" / "yolo-detection" / "baseline" / "weights" / "best.pt"

class YOLODetector:

    def __init__(
            self,
            config: YOLODetectionConfig
    ) ->  None:
        self.config = config
        self.model  = YOLO(config.model_path)

    def predict(self, image_path: str | Path) -> YOLOClassificationResult:
        results  = self.model.predict(
            source=str(image_path),
            device=self.config.device,
            verbose=False
        )

        result = results[0]

        if result.probs is None:
            raise ValueError(
                "YOLO model did not return classification probabilites"
            )

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        class_name = result.names.get(class_id, "unknown")
        logger.info("YOLO classification completed",
                    extra={
                        "event": "yolo.classification.completed",
                        "fields": {
                            "class_id": class_id,
                            "confidence": confidence,
                            "class_name": class_name
                        }
                    })

        return YOLOClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence
        )
