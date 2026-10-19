from contextvars import ContextVar
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import UTC, datetime
import sys


inspection_id: ContextVar[str | None] = ContextVar(
    "inspection_id",
    default=None
)

class JSONFormatter(logging.Formatter):

    def format(
            self,
            record: logging.LogRecord
    ) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "inspection_id": inspection_id.get()
        }

        event = getattr(
            record,
            "event",
            None
        )

        if event is not None:
            payload["event"] = event

        fields = getattr(
            record,
            "fields",
            None
        )

        if fields:
            payload.update(fields)

        return json.dumps(payload, default=str)


def configure_logging(
        level: int = logging.INFO,
        log_file: str | None = None
) -> None:
    root = logging.getLogger()

    root.setLevel(level)
    root.handlers.clear()

    formatter = JSONFormatter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,
            backupCount=5
        )

        file_handler.setFormatter(formatter)

        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
