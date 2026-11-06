"""Process-level owner of stay-origin table reads for one patient batch.

A streamed batch can run several resolvers (a module's own load and its
dependency loads), each with its own data source over the same files. The
stay-origin table (eICU ``patient``, MIMIC ``icustays``, AUMC ``admissions``,
SICdb ``cases``) is read once per batch, restricted to the batch's stays,
with every column; readers take row/column subsets.

Identity (Codex delta3 D3-STAY-CACHE): an entry is keyed by the files the
storage owner validates for the table (``resolve_target_files``), not by
the installation path, and holds each file's size and mtime from the read.
Every lookup resolves and stats the files again (no data is re-read):
- another file set is another entry (a source over the same path whose
  storage owner names other files never reuses a frame);
- the same files with another size or mtime raise
  ``stay_table_source_changed``, and a validated file that has gone raises
  ``stay_table_source_missing`` (both ``StorageIntegrityError``): the
  batch's earlier readers used the old bytes, so the batch cannot continue
  on the new ones;
- a source without validated files (an in-memory or CSV source) is its own
  identity and shares nothing with other source objects.

Lifetime: one batch scope. An entry lives only while its scope is the
latest one (another scope evicts it) and is released at the streamed batch
boundary by ``clear_stay_tables`` (``api.extraction`` calls it with the
batch's loaders). A resolver's unscoped (whole-cohort) memo uses the same
binding check through ``checked_memo_frame``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
from typing import Callable

import pandas as pd

from ..storage_layout import StorageIntegrityError


class StayTableSourceChanged(StorageIntegrityError):
    """A stay-origin table's validated files changed while its frame was held."""

    def __init__(self, reason: str, table: str, path: str):
        self.reason = reason
        self.table = table
        super().__init__(
            f"{reason}: {table}: {os.path.basename(path)} changed while this batch "
            "held the table; rerun the batch on a stable installation"
        )


@dataclass(frozen=True)
class SourceBinding:
    """What a stay-table frame was read from: identity plus file versions."""

    identity: tuple
    table: str
    versions: tuple  # ((path, size, mtime_ns), ...) in path order; () if no files


@dataclass(frozen=True)
class _Entry:
    versions: tuple
    frame: pd.DataFrame


_LOCK = threading.RLock()
_STORE: dict[tuple, _Entry] = {}
_READS = {"count": 0}


def _versions(table: str, paths) -> tuple:
    out = []
    for path in sorted(str(p) for p in paths):
        try:
            stat = os.stat(path)
        except FileNotFoundError as exc:
            raise StayTableSourceChanged("stay_table_source_missing", table, path) from exc
        out.append((path, stat.st_size, stat.st_mtime_ns))
    return tuple(out)


def source_binding(data_source, table: str) -> SourceBinding:
    """Bind ``table`` of ``data_source`` to the storage owner's validated files."""
    config = getattr(data_source, "config", None)
    database = str(getattr(config, "name", ""))
    base = str(getattr(data_source, "base_path", "") or "")
    resolve = getattr(data_source, "resolve_target_files", None)
    paths = tuple(resolve(table)) if base and resolve is not None else ()
    if not paths:
        return SourceBinding((database, "object", id(data_source)), table, ())
    versions = _versions(table, paths)
    return SourceBinding((database, base, tuple(p for p, _, _ in versions)), table, versions)


def _check(binding: SourceBinding, held: tuple) -> None:
    if held != binding.versions:
        changed = next(
            (now[0] for now, old in zip(binding.versions, held) if now != old),
            binding.versions[0][0] if binding.versions else "",
        )
        raise StayTableSourceChanged("stay_table_source_changed", binding.table, changed)


def stay_table_frame(
    binding: SourceBinding, scope: object, read: Callable[[], pd.DataFrame]
) -> pd.DataFrame:
    """Return the batch's frame for ``binding`` (read once per scope)."""
    key = (binding.identity, binding.table, scope)
    with _LOCK:
        stale = [k for k in _STORE if k[-1] != scope]
        for k in stale:
            del _STORE[k]
        entry = _STORE.get(key)
        if entry is not None:
            _check(binding, entry.versions)
            return entry.frame
        _READS["count"] += 1
        frame = read().reset_index(drop=True)
        _STORE[key] = _Entry(binding.versions, frame)
        return frame


def checked_memo_frame(
    memo: dict, binding: SourceBinding, read: Callable[[], pd.DataFrame]
) -> pd.DataFrame:
    """A resolver's own unscoped memo, under the same binding check."""
    key = (binding.identity, binding.table)
    entry = memo.get(key)
    if entry is not None:
        _check(binding, entry.versions)
        return entry.frame
    frame = read().reset_index(drop=True)
    memo[key] = _Entry(binding.versions, frame)
    return frame


def clear_stay_tables() -> None:
    """Release every held stay table (a batch or call boundary)."""
    with _LOCK:
        _STORE.clear()


def stay_table_reads() -> int:
    """Number of physical stay-table reads since the process started."""
    return _READS["count"]

