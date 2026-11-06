"""Stay-origin table reads: one source binding per unit, a payload-only cache.

The stay-origin table (eICU ``patient``, MIMIC ``icustays``, AUMC
``admissions``, SICdb ``cases``) is read once per patient batch, restricted
to the batch's stays, with every column; readers take row/column subsets.

Binding (Codex delta8c). The outermost unit owns the run binding. A unit is
an extraction run, or one public ``load_concepts`` call. For each stay-table
source it resolves, the binding records the storage owner's validated file
set (``resolve_target_files``) and each file's (size, mtime_ns, ino). It
records them at first resolution, or inherits them from the parent's
snapshot in a child process. Every read, whether served from the cache or
from disk, resolves and stats the files again and verifies them against the
binding:
- another file set, or another version of a file, raises
  ``stay_table_source_changed``;
- a vanished file raises ``stay_table_source_missing``;
- a read outside any unit raises ``stay_table_unbound``.
All three are ``StorageIntegrityError``. A source without validated files
(an in-memory or CSV source) binds nothing and shares nothing.

Cache. It holds payload only, keyed by (unit, source table, stays). Each
thread keeps one stay set per source table, so a loader's chunk threads
in one unit don't evict each other. It may be dropped at any time without
touching the binding: at a streamed batch boundary, or between base and
derived vitals (``drop_stay_table_frames``). A later read therefore still
verifies against the versions the unit started with.

Lifetime. ``stay_table_unit`` opens the binding when none is active, and
drops it together with its payload in the same ``finally``. A nested unit,
or a thread run in a copied context, joins the active unit. A child process
receives ``child_stay_binding(...)`` and opens its unit from it.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import functools
import os
import threading
from typing import Callable, Iterator, Optional

import pandas as pd

from ..storage_layout import StorageIntegrityError


class StayTableSourceChanged(StorageIntegrityError):
    """A stay-origin table's validated files differ from the unit's binding."""

    def __init__(self, reason: str, table: str, path: str):
        self.reason = reason
        self.table = table
        super().__init__(
            f"{reason}: {table}: {os.path.basename(path)} changed while this run "
            "was bound to it; rerun on a stable installation"
        )


class StayTableUnbound(StorageIntegrityError):
    """A stay-table read (or a nested load) without a live unit binding."""

    reason = "stay_table_unbound"

    def __init__(self, table: str):
        self.table = table
        super().__init__(
            f"stay_table_unbound: {table}: read outside any load_concepts call or "
            "extraction run, so there is no source binding to verify against"
        )


class _Unit:
    def __init__(self, snapshot: Optional[dict]):
        snapshot = snapshot or {}
        self.files: dict = dict(snapshot.get("files", {}))  # source key -> paths
        self.versions: dict = dict(snapshot.get("versions", {}))  # path -> version
        self.frames: dict = {}  # (source key, stays) -> (thread id, frame)


_LOCK = threading.RLock()
_LIVE: set = set()
_READS = {"count": 0}
_UNIT: contextvars.ContextVar = contextvars.ContextVar("easyicu_stay_table_unit", default=None)


@contextmanager
def stay_table_unit(snapshot: Optional[dict] = None) -> Iterator[_Unit]:
    """Join the active unit, or open one (from a parent's ``snapshot``)."""
    active = _UNIT.get()
    if active is not None:
        yield active
        return
    unit = _Unit(snapshot)
    reset = _UNIT.set(unit)
    with _LOCK:
        _LIVE.add(unit)
    try:
        yield unit
    finally:
        _UNIT.reset(reset)
        with _LOCK:
            _LIVE.discard(unit)
            unit.frames.clear()


def in_stay_table_unit(function):
    """Run ``function`` inside the active unit, or as its own unit."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with stay_table_unit():
            return function(*args, **kwargs)

    return wrapper


def stay_table_unit_active() -> bool:
    return _UNIT.get() is not None


def _stat(table: str, path: str) -> tuple:
    try:
        stat = os.stat(path)
    except FileNotFoundError as exc:
        raise StayTableSourceChanged("stay_table_source_missing", table, path) from exc
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino)


def _resolve(data_source, table: str) -> tuple[tuple, tuple]:
    base = str(getattr(data_source, "base_path", "") or "")
    resolve = getattr(data_source, "resolve_target_files", None)
    paths = tuple(sorted(str(p) for p in resolve(table))) if base and resolve is not None else ()
    if not paths:
        return ("object", id(data_source), table), ()
    database = str(getattr(getattr(data_source, "config", None), "name", ""))
    return (database, base, table), paths


def _verify(unit: _Unit, table: str, key: tuple, paths: tuple) -> None:
    if not paths:
        return
    now = {path: _stat(table, path) for path in paths}
    with _LOCK:
        held = unit.files.setdefault(key, paths)
        if held != paths:
            changed = sorted(set(held) ^ set(paths))[0]
            raise StayTableSourceChanged("stay_table_source_changed", table, changed)
        for path, version in now.items():
            if unit.versions.setdefault(path, version) != version:
                raise StayTableSourceChanged("stay_table_source_changed", table, path)


def stay_table_frame(
    data_source, table: str, stays, read: Callable[[], pd.DataFrame], *, cache: bool = True
) -> pd.DataFrame:
    """``table`` for ``stays`` (None: unscoped), verified against the unit's binding.

    ``cache=False`` is an exact read that is verified but not kept.
    """
    unit = _UNIT.get()
    if unit is None:
        raise StayTableUnbound(table)
    key, paths = _resolve(data_source, table)
    _verify(unit, table, key, paths)
    if not cache:
        frame = read()
        _verify(unit, table, key, paths)  # unchanged while it was read
        return frame
    thread = threading.get_ident()
    with _LOCK:
        held = unit.frames.get((key, stays))
        if held is not None:
            return held[1]
        _READS["count"] += 1
        frame = read().reset_index(drop=True)
        _verify(unit, table, key, paths)
        for other, (owner, _) in list(unit.frames.items()):
            if other[0] == key and owner == thread:
                del unit.frames[other]
        unit.frames[(key, stays)] = (thread, frame)
        return frame


def drop_stay_table_frames() -> None:
    """Drop the active unit's payload; its binding is kept."""
    unit = _UNIT.get()
    if unit is not None:
        with _LOCK:
            unit.frames.clear()


def child_stay_binding(data_source) -> dict:
    """The binding a child process verifies against.

    It is the active unit's binding (or, outside any unit, a new run
    binding), with ``data_source``'s stay-origin table pinned now, so that
    every child verifies against the same versions.
    """
    unit = _UNIT.get() or _Unit(None)
    id_configs = getattr(getattr(data_source, "config", None), "id_configs", None) or {}
    table = getattr(id_configs.get("icustay"), "table", None)
    if table:
        try:
            key, paths = _resolve(data_source, table)
        except Exception:  # noqa: BLE001 - not resolvable now: the child's own
            key, paths = None, ()  # first read binds it, or fails typed there
        _verify(unit, table, key, paths)
    with _LOCK:
        return {"files": dict(unit.files), "versions": dict(unit.versions)}


def held_stay_tables() -> int:
    """Number of stay-table frames held now, across every live unit."""
    with _LOCK:
        return sum(len(unit.frames) for unit in _LIVE)


def stay_table_reads() -> int:
    """Number of physical stay-table reads since the process started."""
    return _READS["count"]
