"""The run's one source binding for every storage read (Codex delta8c).

A unit is an extraction run, or one public ``load_concepts`` call. The
outermost unit owns the binding; nested units, and threads run in a copied
context, join it. Isolated child processes receive ``child_binding(...)``
and open their unit from it.

What is bound is what the run consumes: the files a reader actually reads,
captured at the point of reading through the one consumed-files source,
``storage_layout.record_consumed`` → :func:`consume`. Integrity prechecks of
other candidates bind nothing (B9P-CSV-BINDING). That covers:
- a flat table's validated file set, plus each file's (size, mtime_ns, ino);
- a CSV file that was read instead of the Parquet beside it;
- the part files a bucket read selected;
- the installation's manifest and build receipt, each with a sha256;
- a stay-origin or clock table that a reader opens by name.
Each consumed file is recorded with its realpath and format, in the unit
(:func:`consumed_files`) and as a ``table_consumed`` event in the run log.
Every later read verifies against the binding. Any other file set, or
any other version of a file, raises ``storage_source_changed``; a vanished
file raises ``storage_source_missing``. Both are ``StorageSourceChanged``,
a ``StorageIntegrityError``. Outside any unit a read is not part of a run,
and nothing is bound. Stay-origin tables are stricter: they refuse an
unbound read (``concept.stay_tables``).

``binding_summary()`` gives a stable digest of the sorted entries, for the
run receipt.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import functools
import hashlib
import json
import os
import threading
from typing import Iterable, Iterator, Optional

from .storage_layout import StorageIntegrityError


class StorageSourceChanged(StorageIntegrityError):
    """A file the run bound differs from the binding (changed, swapped or gone)."""

    def __init__(self, reason: str, table: str, path: str):
        self.reason = reason
        self.table = table
        self.path = path
        super().__init__(
            f"{reason}: {table}: {os.path.basename(path)} changed while this run "
            "was bound to it; rerun on a stable installation"
        )

    def __reduce__(self):  # crosses process boundaries with its fields
        return (type(self), (self.reason, self.table, self.path))


class _Unit:
    def __init__(self, snapshot: Optional[dict]):
        snapshot = snapshot or {}
        self.files: dict = dict(snapshot.get("files", {}))  # (database, base, table) -> paths
        self.versions: dict = dict(snapshot.get("versions", {}))  # path -> (size, mtime_ns, ino)
        self.sha256: dict = dict(snapshot.get("sha256", {}))  # path -> hex (manifests, receipts)
        self.consumed: dict = {}  # (table, path) -> {table, path, realpath, format}
        self.caches: dict = {}  # payload owners' caches, dropped with the unit


@dataclass(frozen=True)
class BindingSummary:
    """The run binding as recorded: a digest of the sorted entries, and their count."""

    digest: str
    files: int


_LOCK = threading.RLock()
_LIVE: set = set()
_UNIT: contextvars.ContextVar = contextvars.ContextVar("easyicu_binding_unit", default=None)


@contextmanager
def binding_unit(snapshot: Optional[dict] = None) -> Iterator[_Unit]:
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
            unit.caches.clear()


def in_binding_unit(function):
    """Run ``function`` inside the active unit, or as its own unit."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with binding_unit():
            return function(*args, **kwargs)

    return wrapper


def active_unit() -> Optional[_Unit]:
    return _UNIT.get()


def live_units() -> tuple:
    with _LOCK:
        return tuple(_LIVE)


def _version(table: str, path: str) -> tuple:
    try:
        stat = os.stat(path)
    except FileNotFoundError as exc:
        raise StorageSourceChanged("storage_source_missing", table, path) from exc
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino)


def bind_files(
    table: str, paths: Iterable, *, key: Optional[tuple] = None, sha256: bool = False
) -> None:
    """Record ``paths`` in the active unit at first sight; verify them afterwards.

    ``key`` also binds the set itself (a flat table's whole file set).
    ``sha256`` adds the content hash at first sight (manifests, receipts).
    Outside a unit this does nothing.
    """
    unit = _UNIT.get()
    if unit is None:
        return
    paths = tuple(sorted(str(p) for p in paths))
    now = {path: _version(table, path) for path in paths}
    with _LOCK:
        if key is not None:
            held = unit.files.setdefault(key, paths)
            if held != paths:
                changed = sorted(set(held) ^ set(paths))[0]
                raise StorageSourceChanged("storage_source_changed", table, changed)
        for path, version in now.items():
            if unit.versions.setdefault(path, version) != version:
                raise StorageSourceChanged("storage_source_changed", table, path)
            if sha256 and path not in unit.sha256:
                with open(path, "rb") as handle:
                    unit.sha256[path] = hashlib.file_digest(handle, "sha256").hexdigest()
                if _version(table, path) != version:  # rewritten while hashed
                    raise StorageSourceChanged("storage_source_changed", table, path)


def bound_set(key: Optional[tuple]) -> tuple:
    """The file set the active unit bound under ``key``; () if none."""
    unit = _UNIT.get()
    if unit is None or key is None:
        return ()
    with _LOCK:
        return tuple(unit.files.get(key, ()))


def is_bound(path) -> bool:
    """Whether the active unit has bound ``path``."""
    unit = _UNIT.get()
    return unit is not None and str(path) in unit.versions


def consume(
    table: str, paths: Iterable, fmt: str, *, key: Optional[tuple] = None, sha256: bool = False
) -> tuple[dict, ...]:
    """Bind files a reader is about to read and record them as consumed.

    Returns the consumed entries (``table``, ``path``, ``realpath``,
    ``format``). Outside a unit nothing is bound, but the entries are still
    returned for the run log.
    """
    paths = tuple(sorted(str(p) for p in paths))
    bind_files(table, paths, key=key, sha256=sha256)
    entries = tuple(
        {"table": table, "path": path, "realpath": os.path.realpath(path), "format": fmt}
        for path in paths
    )
    unit = _UNIT.get()
    if unit is not None:
        with _LOCK:
            for entry in entries:
                unit.consumed[(table, entry["path"])] = entry
    return entries


def consumed_files() -> tuple[dict, ...]:
    """What the active unit has consumed, sorted by table and path."""
    unit = _UNIT.get()
    if unit is None:
        return ()
    with _LOCK:
        return tuple(dict(unit.consumed[k]) for k in sorted(unit.consumed))


def source_key(source, table: str) -> Optional[tuple]:
    """The binding key of ``table`` in ``source``; None without an installation path."""
    base = getattr(source, "base_path", None)
    if not base:
        return None
    database = str(getattr(getattr(source, "config", None), "name", ""))
    return (database, str(base), table)


def child_binding(data_source) -> dict:
    """The binding a child process verifies against.

    It is the active unit's binding, or, outside any unit, a new run
    binding. ``data_source``'s stay-origin table is pinned now, so every
    child verifies against the same versions of it.
    """
    unit = _UNIT.get()
    if unit is None:
        with binding_unit() as unit:
            return _pinned(unit, data_source)
    return _pinned(unit, data_source)


def _pinned(unit: _Unit, data_source) -> dict:
    id_configs = getattr(getattr(data_source, "config", None), "id_configs", None) or {}
    table = getattr(id_configs.get("icustay"), "table", None)
    resolve = getattr(data_source, "resolve_target_files", None)
    if table and resolve is not None:
        try:
            resolve(table)  # the storage owner binds what it resolves
        except StorageSourceChanged:
            raise
        except Exception:  # noqa: BLE001 - not resolvable now: the child's own
            pass  # first read binds it, or fails typed there
    with _LOCK:
        return {
            "files": dict(unit.files),
            "versions": dict(unit.versions),
            "sha256": dict(unit.sha256),
        }


def binding_summary() -> Optional[BindingSummary]:
    """The active unit's binding digest; None outside a unit."""
    unit = _UNIT.get()
    if unit is None:
        return None
    with _LOCK:
        entries = {
            "files": sorted([list(key), list(paths)] for key, paths in unit.files.items()),
            "versions": sorted([path, list(v)] for path, v in unit.versions.items()),
            "sha256": sorted([path, digest] for path, digest in unit.sha256.items()),
            "consumed": sorted(
                [e["table"], e["path"], e["realpath"], e["format"]] for e in unit.consumed.values()
            ),
        }
        count = len(unit.versions)
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return BindingSummary(hashlib.sha256(payload).hexdigest(), count)
