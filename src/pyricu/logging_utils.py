"""Utility helpers for configuring pyricu logging output."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: str | int = "INFO", *, force: bool = False) -> None:
    """Configure root logging for pyricu workloads.

    Parameters
    ----------
    level:
        Logging level accepted by :func:`logging.basicConfig`.  Can be an int or
        case-insensitive string such as ``"INFO"``.
    force:
        When ``True`` the root handler configuration is reset via
        ``logging.basicConfig(force=True)`` (Python 3.8+).  Use with care in
        interactive notebooks because it affects global logging.
    """

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    root = logging.getLogger()
    if root.handlers and not force:
        # Respect existing configuration to avoid duplicating handlers.
        root.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=force,
    )


__all__ = ["configure_logging"]
