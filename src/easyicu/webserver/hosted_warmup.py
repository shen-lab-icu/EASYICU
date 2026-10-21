"""Summarize a hosted instance's prepared data once, right after it starts.

Each clinician's instance starts with empty in-memory caches, and the first
Cohort statistics or Cross-database comparison of a full database reads every
row: minutes of waiting. When the administrator's launcher asks for it
(``EASYICU_HOSTED_WARMUP=1``, set by ``tools/hosted_instances.py start``), one
background thread computes both summaries for every registered export, so a
clinician who opens the page later finds them ready. A failure is logged and
changes nothing: the page then computes, and reports, exactly as before.
"""

from __future__ import annotations

import logging
import os
import threading

from easyicu.webserver import cohort_review, crossdb_review, hosted
from easyicu.webserver import sources as source_store

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_thread: threading.Thread | None = None


def requested() -> bool:
    return hosted.is_hosted() and os.environ.get("EASYICU_HOSTED_WARMUP") == "1"


def start() -> threading.Thread | None:
    """Start the warm-up once per process when the launcher asked for it."""
    global _thread
    if not requested():
        return None
    with _lock:
        if _thread is None:
            _thread = threading.Thread(target=warm, name="easyicu-hosted-warmup", daemon=True)
            _thread.start()
        return _thread


def warm() -> list[tuple[str, str]]:
    """Summarize each registered export, the active one first; return what finished."""
    registry = source_store.load_registry()
    active = registry.get("active_path")
    exports = sorted(
        (source for source in registry.get("sources") or [] if source.get("ok")),
        key=lambda source: source.get("path") != active,
    )
    finished: list[tuple[str, str]] = []
    for source in exports:
        path = str(source.get("path") or "")
        for name, summarize in (
            ("cohort", lambda: cohort_review.cohort_review_summary({"source_path": path})),
            ("crossdb", lambda: crossdb_review.precompute_feature_distributions(path)),
        ):
            try:
                summarize()
            except Exception as exc:  # The page recomputes and reports it.
                _log.warning("hosted warm-up: %s summary of source %s failed: %s",
                             name, source.get("id"), type(exc).__name__)
            else:
                finished.append((str(source.get("id")), name))
    return finished
