"""
Centralized logging configuration for the PDF Redactor service.

Designed for multi-worker uvicorn deployments running as a Windows service (WinSW).

- Daily rotating file logs in ``logs/`` directory
- Separate error log (WARNING+) for quick triage
- PID in format string to distinguish worker processes
- Console output retained for local development
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] pid=%(process)d %(name)s: %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "100"))


def setup_logging() -> None:
    """Configure root logger with file + console handlers."""
    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Avoid duplicate handlers on reload / multiple calls
    if root.handlers:
        return

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Daily rotating application log ────────────────────────────────
    app_handler = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, "app.log"),
        when="midnight",
        interval=1,
        backupCount=LOG_RETENTION_DAYS,
        encoding="utf-8",
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(formatter)
    app_handler.suffix = "%Y-%m-%d"

    # ── Daily rotating error log (WARNING+) ───────────────────────────
    error_handler = TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, "error.log"),
        when="midnight",
        interval=1,
        backupCount=LOG_RETENTION_DAYS,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    error_handler.suffix = "%Y-%m-%d"

    # ── Console handler ───────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    root.addHandler(app_handler)
    root.addHandler(error_handler)
    root.addHandler(console_handler)

    # ── Route uvicorn loggers through the same handlers ───────────────
    for uv_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(uv_name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
