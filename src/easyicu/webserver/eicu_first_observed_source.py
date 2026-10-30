"""Bind the conservative eICU candidate to an exact registered source."""

import gzip
import hashlib
import os
import stat
import zipfile

from pathlib import Path

import pandas as pd

from easyicu.research_agent.acquisition.eicu_first_observed_candidate import (
    REQUIRED_COLUMNS,
    FirstObservedCandidate,
    FirstObservedCandidateError,
    derive_first_observed_candidate,
)
from easyicu.research_agent.acquisition.patient_grouping import open_nofollow, read_digest_bound_parquet

from .demo_source_storage import source_paths, verified_public_demo_raw
from .raw_source_authority import resolve_raw_source_identity_binding


def _stable_stat(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _compare_patient_columns(official: pd.DataFrame, bound: pd.DataFrame) -> None:
    """Compare all source rows without retaining or projecting their values."""
    if len(official) != len(bound) or set(official.columns) != set(REQUIRED_COLUMNS):
        raise FirstObservedCandidateError("official eICU patient table is incomplete or incompatible")
    numeric = {"patientunitstayid", "patienthealthsystemstayid", "unitvisitnumber"}
    try:
        for column in REQUIRED_COLUMNS:
            left, right = official[column], bound[column]
            if column in numeric:
                left = pd.to_numeric(left, errors="raise").astype("Int64")
                right = pd.to_numeric(right, errors="raise").astype("Int64")
            else:
                left, right = left.astype("string"), right.astype("string")
            equal = left.eq(right).fillna(False) | (left.isna() & right.isna())
            if not bool(equal.all()):
                raise FirstObservedCandidateError("official and bound eICU patient tables differ")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, FirstObservedCandidateError):
            raise
        raise FirstObservedCandidateError("official patient table cannot be compared exactly") from exc


def _verify_complete_official_patient_source(binding, table: pd.DataFrame) -> dict:
    """Bind all identity rows to the pinned archive, using only private memory.

    A converted-cache marker alone does not establish a complete patient table.
    Recheck the pinned ZIP bytes and compare all six decision fields, never an
    outcome or identifier value, before deriving a candidate from the bound
    Parquet table. This proof grants neither cohort nor research execution.
    """
    patient_path = Path(binding.table_paths["patient"])
    source = verified_public_demo_raw(patient_path.parent)
    if source is None or source.database != "eicu_demo" or not source.archive_sha256:
        raise FirstObservedCandidateError("official eICU Demo source is not verified")
    archive = source_paths(source).archive
    descriptor = None
    try:
        descriptor = open_nofollow(
            archive, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FirstObservedCandidateError("official archive is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            def digest_archive() -> str:
                handle.seek(0)
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                return digest.hexdigest()

            if digest_archive() != source.archive_sha256:
                raise FirstObservedCandidateError("official eICU archive digest mismatch")
            handle.seek(0)
            with zipfile.ZipFile(handle) as zipped:
                matches = [item for item in zipped.infolist()
                           if item.filename.lower().endswith("/patient.csv.gz")]
                if len(matches) != 1 or matches[0].file_size > source.max_uncompressed_bytes:
                    raise FirstObservedCandidateError("official eICU patient table is ambiguous")
                with zipped.open(matches[0]) as member, gzip.open(member, "rt", encoding="utf-8") as csv:
                    official = pd.read_csv(
                        csv, dtype="string", usecols=lambda c: c.lower() in REQUIRED_COLUMNS,
                    )
            official.columns = official.columns.str.lower()
            _compare_patient_columns(official, table)
            if digest_archive() != source.archive_sha256 or _stable_stat(before) != _stable_stat(os.fstat(descriptor)):
                raise FirstObservedCandidateError("official eICU archive changed while being read")
    except FirstObservedCandidateError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
        raise FirstObservedCandidateError("cannot verify the official eICU patient table") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return {
        "schema_version": "easyicu.eicu_demo_complete_patient_source/1",
        "archive_sha256": source.archive_sha256,
        "patient_table_sha256": binding.table_sha256["patient"],
        "compared_columns": list(REQUIRED_COLUMNS),
        "all_source_rows_compared": True,
        "other_patient_columns_verified": False,
        "outcome_columns_read": False,
        "research_execution_authorized": False,
    }


def derive_from_verified_eicu_demo(export_path: str | Path) -> FirstObservedCandidate:
    """Produce host-private coordinates and counts after source verification.

    Both the coordinates and their detailed exclusion counts are private.
    Publication, model access, or UI projection requires a separate reviewed
    disclosure policy, including suppression of small and inferable cells.
    """
    binding = resolve_raw_source_identity_binding(export_path=export_path, database="eicu_demo")
    if binding is None or binding.family != "eicu" or "patient" not in binding.table_paths:
        raise FirstObservedCandidateError("registered eICU Demo raw patient source is unavailable")
    table, _ = read_digest_bound_parquet(
        binding.table_paths["patient"], expected_sha256=binding.table_sha256["patient"],
        columns=REQUIRED_COLUMNS, label="eICU identity and age table", error=FirstObservedCandidateError,
    )
    proof = _verify_complete_official_patient_source(binding, table)
    result = derive_first_observed_candidate(table)
    return FirstObservedCandidate(result.frame, {
        **result.receipt,
        "source_binding": {
            "database": binding.database,
            "export_manifest_sha256": binding.export_manifest_sha256,
            "patient_table_sha256": binding.table_sha256["patient"],
            "official_release_independently_verified": True,
            "complete_patient_source": proof,
            "raw_path_returned": False,
        },
        "public_projection_authorized": False,
    })


def derive_from_registered_eicu_export(export_path: str | Path) -> FirstObservedCandidate:
    """Read only identity and age columns from an exact digest-bound table.

    A registered export alone does not prove that its raw source is a complete
    official Demo release. The receipt therefore claims only the bound source;
    callers must independently verify release identity and source scope.
    """
    binding = resolve_raw_source_identity_binding(export_path=export_path, database="eicu_demo")
    if binding is None or binding.family != "eicu" or "patient" not in binding.table_paths:
        raise FirstObservedCandidateError("registered eICU Demo raw patient source is unavailable")
    table, _ = read_digest_bound_parquet(
        binding.table_paths["patient"], expected_sha256=binding.table_sha256["patient"],
        columns=REQUIRED_COLUMNS, label="eICU identity and age table", error=FirstObservedCandidateError,
    )
    result = derive_first_observed_candidate(table)
    return FirstObservedCandidate(result.frame, {
        **result.receipt,
        "source_binding": {
            "database": binding.database,
            "export_manifest_sha256": binding.export_manifest_sha256,
            "patient_table_sha256": binding.table_sha256["patient"],
            "official_release_independently_verified": False,
            "raw_path_returned": False,
        },
    })
