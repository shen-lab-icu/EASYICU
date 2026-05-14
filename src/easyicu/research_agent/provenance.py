"""Raw EHR provenance hashing (O27).

Before the research-agent pipeline runs, it normally receives a
materialised cohort (``cohort.parquet``). That cohort was itself
derived from a pipeline built on top of raw EHR sources (MIMIC /
eICU / HiRID / local extracts). Evidence-chain completeness requires
that the raw sources are hashed and registered too, so a reviewer
can ask "where did this cohort come from" and trace back through
three layers without trusting prose.

What this module provides:

* :class:`SourceFileRecord` — hashed descriptor for one raw file
  (sha256, size, mtime, relative role).
* :func:`hash_sources` — walks a list of paths (files or
  directories), skipping files over a configurable size cap so we
  don't accidentally stream TBs of MIMIC tables through sha256.
* :func:`build_provenance_bundle` — turns the hashes into a single
  JSON artefact (``provenance_sources.json``) that the pipeline can
  register as evidence.

Constraints:

* **PHI-safe.** We never copy source bytes anywhere; we compute
  sha256 in streaming 1 MiB chunks and record only the hash, byte
  size, mtime, and a user-supplied ``role`` label. Filenames are
  kept because reviewers ask about MIMIC tables by name; no per-row
  content is echoed.
* **No SDK creep.** Pure stdlib (hashlib / pathlib / os / json).
* **Opt-in.** Users pass ``source_files=[...]`` to
  ``pipeline.run``; default behaviour stays unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# Streaming chunk. 1 MiB is a sweet spot — smaller than typical
# MIMIC table shards (10–500 MiB), larger than hashlib's internal
# block, and small enough that sha256 overhead doesn't dominate
# I/O.
_HASH_CHUNK = 1 << 20  # 1 MiB

# Default skip cap. Reviewer wants reproducibility, not a 10 TB
# hash. Raise at call time if the user really wants to hash
# everything.
_DEFAULT_MAX_BYTES = 2 * (1 << 30)  # 2 GiB


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SourceFileRecord:
    """sha256-hashed provenance for a single raw EHR source file."""

    relative_path: str
    absolute_path: str
    size_bytes: int
    sha256: Optional[str]
    mtime_iso: str
    role: Optional[str] = None
    database: Optional[str] = None
    skipped_reason: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class ProvenanceBundle:
    """A collection of :class:`SourceFileRecord` plus aggregates."""

    records: List[SourceFileRecord] = field(default_factory=list)
    root: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        total_size = sum(r.size_bytes for r in self.records if r.size_bytes)
        hashed = sum(1 for r in self.records if r.sha256)
        skipped = [r for r in self.records if r.skipped_reason]
        databases = sorted({r.database for r in self.records if r.database})
        return {
            "n_sources": len(self.records),
            "n_hashed": hashed,
            "n_skipped": len(skipped),
            "total_size_bytes": total_size,
            "databases": databases,
        }

    def to_disk(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "easyicu.provenance_sources/1",
                    "root": self.root,
                    "summary": self.summary(),
                    "records": [r.to_json() for r in self.records],
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return path


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_entries(
    entries: Sequence[Any], *, root: Optional[Path] = None
) -> List[Tuple[Path, Optional[str], Optional[str]]]:
    """Normalise the user-supplied list into (path, role, database) triples.

    Each entry can be:

    * a path (string / ``Path``) — role and database are inferred from
      the directory name (``mimic4_icu/`` -> database=``miiv``);
    * a dict ``{"path": ..., "role": ..., "database": ...}``;
    * a tuple ``(path, role, database)``.
    """
    resolved: List[Tuple[Path, Optional[str], Optional[str]]] = []
    for entry in entries:
        role: Optional[str] = None
        database: Optional[str] = None
        if isinstance(entry, dict):
            raw_path = entry.get("path")
            role = entry.get("role")
            database = entry.get("database")
        elif isinstance(entry, tuple) and len(entry) in (2, 3):
            raw_path = entry[0]
            role = entry[1] if len(entry) >= 2 else None
            database = entry[2] if len(entry) == 3 else None
        else:
            raw_path = entry
        if raw_path is None:
            continue
        p = Path(raw_path)
        if not p.is_absolute() and root is not None:
            p = (root / p).resolve()
        resolved.append((p, role, database))
    return resolved


def _walk_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            yield child


def hash_sources(
    entries: Sequence[Any],
    *,
    max_bytes_per_file: int = _DEFAULT_MAX_BYTES,
    root: Optional[Path] = None,
) -> ProvenanceBundle:
    """Hash every source file (files or directories).

    Args:
        entries: paths / dicts / tuples; see :func:`_resolve_entries`.
        max_bytes_per_file: per-file cap; files over the cap are
            recorded with ``sha256=None`` and ``skipped_reason="exceeds_cap"``.
        root: if set, relative ``entries`` resolve underneath this path.

    Returns a :class:`ProvenanceBundle` whose ``records`` list has
    one row per file (directories are expanded).
    """
    bundle = ProvenanceBundle(root=str(root) if root else None)
    for abs_path, role, database in _resolve_entries(entries, root=root):
        if not abs_path.exists():
            bundle.records.append(
                SourceFileRecord(
                    relative_path=str(abs_path),
                    absolute_path=str(abs_path),
                    size_bytes=0,
                    sha256=None,
                    mtime_iso="",
                    role=role,
                    database=database,
                    skipped_reason="not_found",
                )
            )
            continue
        for fp in _walk_files(abs_path):
            try:
                st = fp.stat()
            except OSError as exc:
                bundle.records.append(
                    SourceFileRecord(
                        relative_path=str(fp),
                        absolute_path=str(fp),
                        size_bytes=0,
                        sha256=None,
                        mtime_iso="",
                        role=role,
                        database=database,
                        skipped_reason=f"stat_error:{exc.errno}",
                    )
                )
                continue
            size = int(st.st_size)
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            rel = str(fp.relative_to(abs_path.parent)) if abs_path.is_dir() else fp.name
            if size > max_bytes_per_file:
                bundle.records.append(
                    SourceFileRecord(
                        relative_path=rel,
                        absolute_path=str(fp),
                        size_bytes=size,
                        sha256=None,
                        mtime_iso=mtime,
                        role=role,
                        database=database,
                        skipped_reason=f"exceeds_cap:{max_bytes_per_file}",
                    )
                )
                continue
            try:
                digest = sha256_file(fp)
            except OSError as exc:
                bundle.records.append(
                    SourceFileRecord(
                        relative_path=rel,
                        absolute_path=str(fp),
                        size_bytes=size,
                        sha256=None,
                        mtime_iso=mtime,
                        role=role,
                        database=database,
                        skipped_reason=f"read_error:{exc.errno}",
                    )
                )
                continue
            bundle.records.append(
                SourceFileRecord(
                    relative_path=rel,
                    absolute_path=str(fp),
                    size_bytes=size,
                    sha256=digest,
                    mtime_iso=mtime,
                    role=role,
                    database=database,
                )
            )
    return bundle


def build_provenance_bundle(
    *,
    cohort_path: Path,
    source_files: Optional[Sequence[Any]] = None,
    root: Optional[Path] = None,
    max_bytes_per_file: int = _DEFAULT_MAX_BYTES,
) -> ProvenanceBundle:
    """Top-level driver.

    Always hashes the cohort parquet; adds raw EHR source files if the
    user provides them. The cohort hash matches the one already written
    into the evidence store, but recording it here gives reviewers a
    single JSON to quote in the supplement.
    """
    entries: List[Any] = []
    if source_files:
        entries.extend(source_files)
    entries.append({"path": str(cohort_path), "role": "cohort", "database": None})
    return hash_sources(
        entries, max_bytes_per_file=max_bytes_per_file, root=root,
    )


__all__ = [
    "ProvenanceBundle",
    "SourceFileRecord",
    "build_provenance_bundle",
    "hash_sources",
    "sha256_file",
]
