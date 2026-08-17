"""Structured logging configuration.

Uses loguru; configured once at process startup via `configure_logging()`.
Every module then does `from loguru import logger` directly.
"""

from __future__ import annotations

import sys

from loguru import logger

from app.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent logging setup. Safe to call multiple times (e.g. in tests)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings().settings

    logger.remove()  # drop loguru's default handler so we control format/level
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=False,
        # Never log full exception variables: could leak API keys or
        # retrieved corpus text.
        diagnose=False,
    )

    _CONFIGURED = True
    logger.info("Logging configured (level={})", settings.log_level)
