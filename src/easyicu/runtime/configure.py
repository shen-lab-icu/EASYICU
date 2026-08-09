"""Explicit process-level runtime configuration for EasyICU applications."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional


def _configure_stdio_encoding() -> bool:
    """Configure UTF-8 Windows stdio and report whether a stream changed."""
    if os.name != "nt":
        return False

    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    changed = False
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        reconfigure(encoding="utf-8", errors="replace")
        changed = True
    return changed


def configure_runtime(
    *,
    stdio_encoding: bool = True,
    initialize_cache: bool = True,
    clear_cache: Optional[bool] = None,
) -> Dict[str, Any]:
    """Apply process-level settings explicitly.

    Ordinary ``import easyicu`` does not alter stdio or initialize/clear the
    cache manager. CLI and Web entry points may call this function during
    application startup.

    Args:
        stdio_encoding: Configure UTF-8 stdio on Windows.
        initialize_cache: Construct the cache manager.
        clear_cache: Whether to clear registered caches. ``None`` follows the
            legacy ``EASYICU_AUTO_CLEAR_CACHE`` setting, but only after this
            explicit function is called.
    """
    status: Dict[str, Any] = {
        "stdio_configured": False,
        "cache_initialized": False,
        "cache_cleared": False,
    }
    if stdio_encoding:
        status["stdio_configured"] = _configure_stdio_encoding()

    if initialize_cache:
        from .cache_manager import get_cache_manager
        from .project_config import AUTO_CLEAR_CACHE

        manager = get_cache_manager()
        status["cache_initialized"] = True
        should_clear = AUTO_CLEAR_CACHE if clear_cache is None else clear_cache
        if should_clear:
            status["cache_result"] = manager.clear_all_cache()
            status["cache_cleared"] = True

    return status
