"""Schema-first prepared-database identity detection.

This dependency-neutral module owns database identity inference.  Callers may
map the canonical keys to surface-specific aliases, but must not reimplement
the evidence ordering.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from .profiles import normalize_database_key


class DatabaseDetectionError(ValueError):
    """Prepared database identity or path could not be established safely."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        data_path: Optional[Union[str, Path]] = None,
        candidates: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.data_path = Path(data_path) if data_path is not None else None
        self.candidates = tuple(sorted(set(candidates)))


def _casefold_children(root: Path) -> dict[str, tuple[Path, ...]]:
    """Index direct children without inheriting host case-sensitivity rules."""

    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError:
        return {}
    indexed: dict[str, list[Path]] = {}
    for child in children:
        indexed.setdefault(child.name.casefold(), []).append(child)
    return {name: tuple(values) for name, values in indexed.items()}


def peek_table_columns(path: Path, table: str) -> set[str]:
    """Read a prepared table schema without loading its rows."""

    path_children = _casefold_children(path)
    roots = (
        path,
        *(path_children.get("icu", ())),
        *(path_children.get("hosp", ())),
    )
    columns: set[str] = set()
    attempted = 0
    readable = 0
    for root in roots:
        children = _casefold_children(root)
        candidates = (
            *children.get(f"{table}.parquet".casefold(), ()),
            *children.get(table.casefold(), ()),
        )
        for candidate in candidates:
            attempted += 1
            try:
                import pyarrow.parquet as pq

                if candidate.is_file():
                    columns.update(
                        str(name).lower() for name in pq.read_schema(candidate).names
                    )
                    readable += 1
                if candidate.is_dir():
                    shards = tuple(
                        child
                        for child in candidate.iterdir()
                        if child.is_file() and child.suffix.casefold() == ".parquet"
                    )
                    for shard in shards:
                        columns.update(
                            str(name).lower()
                            for name in pq.read_schema(shard).names
                        )
                    if shards:
                        readable += 1
            except Exception:
                continue
        for suffix in (".csv", ".csv.gz"):
            for candidate in children.get(f"{table}{suffix}".casefold(), ()):
                attempted += 1
                try:
                    if candidate.is_file():
                        columns.update(
                            str(name).lower()
                            for name in pd.read_csv(candidate, nrows=0).columns
                        )
                        readable += 1
                except Exception:
                    continue
    if attempted and not readable:
        raise DatabaseDetectionError(
            "database_detection_schema_unreadable",
            f"Prepared table {table!r} exists in {path}, but no schema could be read.",
            data_path=path,
        )
    return columns


def _schema_candidates(path: Path) -> set[str]:
    """Collect every database identity asserted by prepared table schemas.

    Returning on the first recognized table made a mixed root containing, for
    example, MIMIC ``stay_id`` and eICU ``patientunitstayid`` look like a valid
    single database.  The schema tier is authoritative only after all of its
    independent table identities agree.
    """

    icustays_columns = peek_table_columns(path, "icustays")
    candidates = {
        database_name
        for column, database_name in (
            ("stay_id", "miiv"),
            ("icustay_id", "mimic"),
        )
        if column in icustays_columns
    }
    patient_columns = peek_table_columns(path, "patient")
    if "patientunitstayid" in patient_columns:
        candidates.add("eicu")
    return candidates


def _schema_identity(path: Path) -> Optional[str]:
    candidates = _schema_candidates(path)
    if len(candidates) > 1:
        raise DatabaseDetectionError(
            "database_detection_ambiguous",
            f"Conflicting prepared-table schema identities in {path}: "
            f"{sorted(candidates)}.",
            data_path=path,
            candidates=candidates,
        )
    return next(iter(candidates), None)


def _path_candidate(path: Path) -> Optional[str]:
    """Use only the selected directory name as supporting evidence.

    Searching the full absolute path lets an unrelated ancestor such as an
    archive or test folder override the schema of the selected dataset.
    """

    name = path.name.casefold()
    compact = name.replace("-", "").replace("_", "")
    if "miiv" in compact or "mimiciv" in compact or "mimic4" in compact:
        return "miiv"
    if "mimiciii" in compact or "mimic3" in compact or "miii" in compact:
        return "mimic"
    for database, patterns in {
        "eicu": ("eicu", "eicu-crd"),
        "aumc": ("aumc", "amsterdam"),
        "hirid": ("hirid",),
        "sic": ("sicdb", "sic-db"),
    }.items():
        if any(pattern in name for pattern in patterns):
            return database
    return None


def _content_candidates(path: Path) -> set[str]:
    try:
        entries = {entry.name.casefold() for entry in path.iterdir()}
    except OSError:
        return set()
    marker_files = {
        "eicu": ("vitalperiodic.parquet", "vitalperiodic.csv"),
        "aumc": ("numericitems",),
        "hirid": ("general_table.csv", "general_table.parquet", "observations"),
        "sic": ("cases.parquet", "data_float_h_bucket"),
    }
    return {
        database
        for database, markers in marker_files.items()
        if any(marker in entries for marker in markers)
    }


def detect_database_identity(
    data_path: Optional[Union[str, Path]] = None,
    *,
    database: Optional[str] = None,
    use_environment: bool = False,
    strict: bool = True,
) -> str:
    """Return one canonical database key from ordered, attributable evidence."""

    if database:
        try:
            return normalize_database_key(database)
        except KeyError as exc:
            raise ValueError(f"Unsupported database: {database!r}") from exc

    path = Path(data_path) if data_path is not None else None
    if path is not None:
        if path.is_dir():
            schema_identity = _schema_identity(path)
            candidates = _content_candidates(path)
            if schema_identity is not None:
                candidates.add(schema_identity)
        else:
            candidates = set()
        if len(candidates) == 1:
            return next(iter(candidates))
        if len(candidates) > 1:
            raise DatabaseDetectionError(
                "database_detection_ambiguous",
                f"Conflicting database evidence in {path}: {sorted(candidates)}.",
                data_path=path,
                candidates=candidates,
            )
        # A directory label is only a fallback when prepared schemas and
        # database-specific content markers are both silent. It must never
        # override or create ambiguity against stronger in-folder evidence.
        path_candidate = _path_candidate(path)
        if path_candidate is not None:
            return path_candidate

    if use_environment:
        environment_keys = {
            "miiv": ("MIIV_PATH", "MIMICIV_PATH"),
            "mimic": ("MIMIC_PATH", "MIMICIII_PATH"),
            "eicu": ("EICU_PATH",),
            "hirid": ("HIRID_PATH",),
            "aumc": ("AUMC_PATH",),
            "sic": ("SIC_PATH", "SICDB_PATH"),
        }
        candidates = {
            name
            for name, variables in environment_keys.items()
            if any(os.getenv(variable) for variable in variables)
        }
        if len(candidates) == 1:
            return next(iter(candidates))
        if len(candidates) > 1:
            raise DatabaseDetectionError(
                "database_detection_ambiguous",
                "Multiple database-specific environment paths are configured; "
                "supply database explicitly.",
                data_path=data_path,
                candidates=candidates,
            )

    if not strict:
        return "unknown"
    raise DatabaseDetectionError(
        "database_detection_unavailable",
        "Database could not be detected from prepared data evidence; supply "
        "database explicitly.",
        data_path=data_path,
    )


__all__ = [
    "DatabaseDetectionError",
    "detect_database_identity",
    "peek_table_columns",
]
