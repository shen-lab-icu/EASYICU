"""Process-level owner of stay-origin table reads for one patient batch.

A streamed batch can run several resolvers (a module's own load and its
dependency loads), each with its own data source over the same files. The
stay-origin table (eICU ``patient``, MIMIC ``icustays``, AUMC ``admissions``,
SICdb ``cases``) is read once per batch, restricted to the batch's stays,
with every column; readers take row/column subsets. Entries are keyed by
the source identity (database, files), the table and the batch scope, and
only the latest scope is held.
"""

from __future__ import annotations

import threading
from typing import Callable

import pandas as pd

_LOCK = threading.RLock()
_STORE: dict[tuple, pd.DataFrame] = {}
_READS = {"count": 0}


def stay_table_frame(key: tuple, scope: object, read: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Return the cached frame for ``key`` (read once per scope)."""
    with _LOCK:
        stale = [k for k in _STORE if k[-1] != scope]
        for k in stale:
            del _STORE[k]
        frame = _STORE.get(key)
        if frame is None:
            _READS["count"] += 1
            frame = read().reset_index(drop=True)
            _STORE[key] = frame
        return frame


def clear_stay_tables() -> None:
    """Release every held stay table (a batch or call boundary)."""
    with _LOCK:
        _STORE.clear()


def stay_table_reads() -> int:
    """Number of physical stay-table reads since the process started."""
    return _READS["count"]


def source_identity(data_source) -> tuple:
    """Sources over the same files share; an in-memory source is its own."""
    config = getattr(data_source, "config", None)
    base = str(getattr(data_source, "base_path", "") or "")
    return (
        str(getattr(config, "name", "")),
        base if base else ("object", id(data_source)),
    )
