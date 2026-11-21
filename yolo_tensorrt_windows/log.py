"""Logging: to the console when there is one, and to a rotating file per process.

The built app has no console window, so the log files are where errors end up.
"""

import os
import sys
from pathlib import Path

from loguru import logger


def configure_logging(name: str, log_dir: str | os.PathLike) -> Path:
    """Send this process's logs to ``<log_dir>/<name>.log``; return that path."""
    logger.remove()
    if sys.stderr is not None:  # None in the windowed (console-less) build
        logger.add(sys.stderr, level="INFO")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    slug = "-".join(name.lower().split())
    path = log_dir / f"{slug}.log"
    logger.add(path, level="DEBUG", rotation="10 MB", retention=3, encoding="utf-8")
    return path
