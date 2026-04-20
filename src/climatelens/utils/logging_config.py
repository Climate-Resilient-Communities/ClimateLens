"""
Project-wide logging setup.

Gives every script the same timestamped, level-aware log format instead of
ad-hoc ``print()`` calls. Logs are safe to pipe to AzureML's run logs.

Usage::

    from utils.logging_config import configure_logging, get_logger

    configure_logging()
    log = get_logger(__name__)
    log.info("starting pipeline")
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: Optional[str] = None) -> None:
    """
    Configure the root logger once per process.

    Subsequent calls are no-ops so library code can safely call this without
    double-attaching handlers. Level falls back to ``LOG_LEVEL`` env var, then
    ``INFO``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, resolved, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))

    root = logging.getLogger()
    # Remove any handlers a stray basicConfig may have installed.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(numeric_level)

    # Tame noisy third-party loggers by default.
    for noisy in ("urllib3", "matplotlib", "PIL", "transformers"):
        logging.getLogger(noisy).setLevel(max(numeric_level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Shorthand for :func:`logging.getLogger` with auto-configuration."""
    configure_logging()
    return logging.getLogger(name)
