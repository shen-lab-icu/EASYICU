"""Stay-origin table reads: the run binding, verified on every read; payload-only cache.

The stay-origin table (eICU ``patient``, MIMIC ``icustays``, AUMC
``admissions``, SICdb ``cases``) is read once per patient batch, restricted
to the batch's stays, with every column; readers take row/column subsets.

Binding (Codex delta8c). It is the run's one source binding
(``easyicu.storage_binding``), owned by the outermost unit: an extraction
run, or one public ``load_concepts`` call. Every stay-table read, whether
served from the cache or from disk, resolves the table's validated files
again and verifies them against the binding:
- another file set, or another (size, mtime_ns, ino), raises
  ``stay_table_source_changed``;
- a vanished file raises ``stay_table_source_missing``;
- a read outside any unit raises ``stay_table_unbound``.
All three are ``StorageIntegrityError``. A source without validated files
(an in-memory or CSV source) binds nothing and shares nothing.

Cache. It holds payload only, keyed by (unit, source table, stays). Each
thread keeps one stay set per source table, so a loader's chunk threads in
one unit don't evict each other. The payload may be dropped at any time
without touching the binding: at a streamed batch boundary, or between base
and derived vitals (``drop_stay_table_frames``). It goes with its unit.
"""

from __future__ import annotations

import threading
from typing import Callable

import pandas as pd

from ..storage_binding import (
    StorageSourceChanged,
    active_unit,
    bind_files,
    live_units,
    source_key,
)
from ..storage_layout import StorageIntegrityError

_CACHE = "stay_tables"


class StayTableSourceChanged(StorageSourceChanged):
    """A stay-origin table's validated files differ from the run binding."""


class StayTableUnbound(StorageIntegrityError):
    """A stay-table read (or a nested load) without a live unit binding."""

    reason = "stay_table_unbound"

    def __init__(self, table: str):
        self.table = table
        super().__init__(
            f"stay_table_unbound: {table}: read outside any load_concepts call or "
            "extraction run, so there is no source binding to verify against"
        )


_LOCK = threading.RLock()
_READS = {"count": 0}


def _bound(data_source, table: str) -> tuple:
    """Resolve and verify ``table``; the payload key."""
    base = getattr(data_source, "base_path", None)
    resolve = getattr(data_source, "resolve_target_files", None)
    paths = tuple(resolve(table)) if base and resolve is not None else ()
    if not paths:
        return ("object", id(data_source), table)
    key = source_key(data_source, table)
    bind_files(table, paths, key=key)
    return key


def _as_stay_table(error: StorageSourceChanged) -> StayTableSourceChanged:
    reason = error.reason.replace("storage_source_", "stay_table_source_")
    return StayTableSourceChanged(reason, error.table, error.path)


def stay_table_frame(
    data_source, table: str, stays, read: Callable[[], pd.DataFrame], *, cache: bool = True
) -> pd.DataFrame:
    """``table`` for ``stays`` (None: unscoped), verified against the run binding.

    ``cache=False`` is an exact read that is verified but not kept.
    """
    unit = active_unit()
    if unit is None:
        raise StayTableUnbound(table)
    try:
        key = _bound(data_source, table)
        if not cache:
            frame = read()
            _bound(data_source, table)  # unchanged while it was read
            return frame
        frames = unit.caches.setdefault(_CACHE, {})
        thread = threading.get_ident()
        with _LOCK:
            held = frames.get((key, stays))
            if held is not None:
                return held[1]
            _READS["count"] += 1
            frame = read().reset_index(drop=True)
            _bound(data_source, table)
            for other, (owner, _) in list(frames.items()):
                if other[0] == key and owner == thread:
                    del frames[other]
            frames[(key, stays)] = (thread, frame)
            return frame
    except StayTableSourceChanged:
        raise
    except StorageSourceChanged as error:
        raise _as_stay_table(error) from error


def drop_stay_table_frames() -> None:
    """Drop the active unit's stay-table payload; its binding is kept."""
    unit = active_unit()
    if unit is not None:
        with _LOCK:
            unit.caches.pop(_CACHE, None)


def held_stay_tables() -> int:
    """Number of stay-table frames held now, across every live unit."""
    with _LOCK:
        return sum(len(unit.caches.get(_CACHE, ())) for unit in live_units())


def stay_table_reads() -> int:
    """Number of physical stay-table reads since the process started."""
    return _READS["count"]
