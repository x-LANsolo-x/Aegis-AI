from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime, timezone
from typing import Optional


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(
    *,
    level: int = logging.INFO,
    fmt: str = DEFAULT_LOG_FORMAT,
) -> None:
    """Configure Python logging for training runs.

    Keep this centralized so every training/eval script emits consistent logs.
    """

    logging.basicConfig(level=level, format=fmt)


def log_run_header(
    *,
    seed: int,
    dataset_version: str,
    run_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Emit a consistent run header (seed + dataset version + environment).

    Call this at the start of every training run.
    """

    lg = logger or logging.getLogger("ml.run")

    now = datetime.now(timezone.utc).isoformat()
    lg.info("=" * 88)
    lg.info("AEGISAI TRAINING RUN")
    if run_name:
        lg.info("run_name: %s", run_name)
    lg.info("timestamp_utc: %s", now)
    lg.info("seed: %s", seed)
    lg.info("dataset_version: %s", dataset_version)
    lg.info("python: %s", sys.version.replace("\n", " "))
    lg.info("platform: %s", platform.platform())
    lg.info("=" * 88)
