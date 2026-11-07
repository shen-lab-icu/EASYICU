"""The run's one source binding for every storage read (Codex delta8c).

A unit is an extraction run, or one public ``load_concepts`` call. The
outermost unit owns the binding; nested units, and threads run in a copied
context, join it. Isolated child processes receive ``child_binding(...)``
and open their unit from it.

What is bound is what the run consumes: the files a reader actually reads,
captured at the point of reading through the one consumed-files source,
``storage_layout.record_consumed`` → :func:`consume`. Integrity prechecks of
other candidates bind nothing (B9P-CSV-BINDING). That covers:
- a flat table's validated file set, plus each file's stat identity
  (size, mtime_ns, ctime_ns, ino, dev): an mtime restored with utime still
  moves ctime, so a same-size rewrite in place is a new version;
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

A file changed within ``RACY_WINDOW_NS`` of its first read is racily clean:
its stat cannot vouch for its content (a rewrite inside the filesystem's
timestamp granularity keeps every stat field). Such a file is bound by its
content as well: its sha256 is taken at first sight and every later check
of it re-hashes. No payload cache keeps a frame read from it
(:func:`racily_clean`). Deliberate stat forgery (ctime included) is outside
this guard; a formal run's content identity comes from the full-SHA input
snapshot taken before and after the run (``docs/qa/data_provenance_in_real_runs.md``).

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
import time
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
        # path -> (size, mtime_ns, ctime_ns, ino, dev)
        self.versions: dict = dict(snapshot.get("versions", {}))
        self.sha256: dict = dict(snapshot.get("sha256", {}))  # path -> hex (manifests, receipts)
        # Paths first read while racily clean: verified by content on every check.
        self.racy: set = set(snapshot.get("racy", ()))
        # (database, base, table) -> {"layout": "flat"|"bucket", "root", "realpath"}:
        # the layout the storage policy chose for the table (D10D).
        self.layouts: dict = {k: dict(v) for k, v in snapshot.get("layouts", {}).items()}
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
_CAPTURE: contextvars.ContextVar = contextvars.ContextVar("easyicu_consumed_capture", default=None)


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


RACY_WINDOW_NS = 2_000_000_000


def _version(table: str, path: str) -> tuple:
    """A file's stat identity: (size, mtime_ns, ctime_ns, ino, dev)."""
    try:
        stat = os.stat(path)
    except FileNotFoundError as exc:
        raise StorageSourceChanged("storage_source_missing", table, path) from exc
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev)


def stat_identity(table: str, path) -> tuple:
    """The stat identity the binding records for ``path`` (see :func:`_version`)."""
    return _version(table, str(path))


def racily_clean(version: tuple, now_ns: Optional[int] = None) -> bool:
    """Whether a file of this stat identity changed within ``RACY_WINDOW_NS`` of now.

    Its stat then cannot vouch for its content: a rewrite in the same
    timestamp tick keeps every field. A timestamp in the future counts too.
    """
    now = time.time_ns() if now_ns is None else now_ns
    return now - max(version[1], version[2]) < RACY_WINDOW_NS


def _content_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
            first = path not in unit.versions
            if unit.versions.setdefault(path, version) != version:
                raise StorageSourceChanged("storage_source_changed", table, path)
            if first and racily_clean(version):
                unit.racy.add(path)  # its stat cannot vouch for its content
            if (sha256 or path in unit.racy) and path not in unit.sha256:
                unit.sha256[path] = _content_sha256(path)
                if _version(table, path) != version:  # rewritten while hashed
                    raise StorageSourceChanged("storage_source_changed", table, path)
            elif path in unit.racy and _content_sha256(path) != unit.sha256[path]:
                # Same stat, other bytes: a rewrite inside the timestamp tick.
                raise StorageSourceChanged("storage_source_changed", table, path)


def bound_set(key: Optional[tuple]) -> tuple:
    """The file set the active unit bound under ``key``; () if none."""
    unit = _UNIT.get()
    if unit is None or key is None:
        return ()
    with _LOCK:
        return tuple(unit.files.get(key, ()))


def bound_sha256(path) -> Optional[str]:
    """The sha256 the active unit bound for ``path`` (manifests, receipts); None if none."""
    unit = _UNIT.get()
    if unit is None:
        return None
    with _LOCK:
        return unit.sha256.get(str(path))


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
    captured = _CAPTURE.get()
    if captured is not None:
        for path in paths:
            captured.append((table, path, fmt, sha256, _version(table, path)))
    return entries


@contextmanager
def capturing_consumed() -> Iterator[list]:
    """Collect what is consumed in this context: (table, path, format, sha256, version).

    A payload cache stores these with the payload, so a later hit records
    and verifies the same sources (``storage_layout.replay_consumed``).
    """
    outer = _CAPTURE.get()
    captured: list = []
    reset = _CAPTURE.set(captured)
    try:
        yield captured
    finally:
        _CAPTURE.reset(reset)
        if outer is not None:  # a nested read is part of the outer one
            outer.extend(captured)


def current_capture() -> Optional[tuple]:
    """What the innermost :func:`capturing_consumed` has collected so far; None outside one."""
    captured = _CAPTURE.get()
    return None if captured is None else tuple(captured)


def captured_racily_clean(captured) -> bool:
    """Whether any captured file was racily clean when read: no cache may keep its payload."""
    now = time.time_ns()
    return any(racily_clean(entry[-1], now) for entry in captured)


def captured_still_current(captured) -> bool:
    """Whether every captured file still has the version it had when read."""
    for table, path, _fmt, _sha, version in captured:
        try:
            if _version(table, path) != version:
                return False
        except StorageSourceChanged:
            return False
    return True


def layout_key(key: Optional[tuple]) -> Optional[tuple]:
    """The layout-record key of a source key (table names are case-blind on disk)."""
    return None if key is None else (*key[:-1], str(key[-1]).lower())


def bound_layout(key: Optional[tuple]) -> Optional[dict]:
    """The layout this unit chose for the table at ``key``; None if none yet."""
    unit = _UNIT.get()
    if unit is None or key is None:
        return None
    with _LOCK:
        held = unit.layouts.get(layout_key(key))
        return dict(held) if held is not None else None


def bind_layout(key: Optional[tuple], table: str, layout: str, root) -> None:
    """Record the layout the storage policy chose for a table; refuse a replacement.

    The first choice in a unit is kept (``layout`` "flat" or "bucket", the
    bucket ``root`` and its realpath). A later choice of another layout, or
    of a bucket at another root or realpath, for the same table raises
    ``storage_source_changed``. A deliberate flat read beside a bucket
    (``storage_layout.flat_table_scan``) is not a layout choice.
    """
    unit = _UNIT.get()
    if unit is None or key is None:
        return
    root = "" if root is None else str(root)
    now = {"layout": layout, "root": root, "realpath": os.path.realpath(root) if root else ""}
    with _LOCK:
        held = unit.layouts.setdefault(layout_key(key), now)
    if held != now:
        raise StorageSourceChanged("storage_source_changed", table, root or held["root"] or table)


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
            "racy": sorted(unit.racy),
            "layouts": {k: dict(v) for k, v in unit.layouts.items()},
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
            "layouts": sorted(
                [list(key), [v["layout"], v["root"], v["realpath"]]]
                for key, v in unit.layouts.items()
            ),
            "consumed": sorted(
                [e["table"], e["path"], e["realpath"], e["format"]] for e in unit.consumed.values()
            ),
        }
        count = len(unit.versions)
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return BindingSummary(hashlib.sha256(payload).hexdigest(), count)
