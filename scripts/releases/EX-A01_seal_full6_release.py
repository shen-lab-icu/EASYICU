#!/usr/bin/env python3
"""EX-A01: validate and atomically seal one six-database native-v2 release.

The sealer is deliberately read-only until every database manifest and all
114 Parquet files have passed the publication contract.  In particular, a
failed validation cannot replace a pre-existing ``run_metadata.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq


DATABASES = ("aumc", "eicu", "hirid", "mimic", "miiv", "sic")
MODULES = (
    "vitals",
    "demographics",
    "outcome",
    "blood_gas",
    "chemistry",
    "hematology",
    "ventilator",
    "respiratory",
    "vasopressors",
    "medications",
    "neurological",
    "renal",
    "circulatory",
    "other_scores",
    "sepsis_shared",
    "sofa1_score",
    "sofa2_score",
    "sepsis3_sofa1",
    "sepsis3_sofa2",
)
NATIVE_SCHEMA_VERSION = "easyicu_native_export_v2"
CONTRACT_REVISION = "native_v2_row_grain_sha256_size_20260803"
RELEASE_SCHEMA_VERSION = "easyicu_full6_release_v1"
EXPECTED_PARQUET_COUNT = len(DATABASES) * len(MODULES)
CORRECTIONS = (
    "demographics_one_row_per_stay_with_nearest_static_values_and_recomputed_bmi",
    "longitudinal_null_equal_primary_key_consolidation",
    "partial_wide_exact_floating_charttime_key_alignment",
    "streamed_non_demographics_charttime_schema_stabilization",
    "eicu_samp_represents_sampling_event_not_culture_positivity",
    "per_parquet_sha256_and_byte_size_receipts",
)


class ReleaseValidationError(ValueError):
    """Raised when an export cannot be sealed without weakening its contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReleaseValidationError(f"Cannot read {label}: {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"Invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseValidationError(f"{label} must contain one JSON object: {path}")
    return value, raw


def _expected_grain(module: str) -> tuple[list[str], str, str]:
    if module == "demographics":
        return ["stay_id"], "one_row_per_icu_stay", "not_applicable"
    return (
        ["stay_id", "charttime"],
        "one_row_per_icu_stay_relative_hour",
        "nulls_equal",
    )


def _require_plain_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReleaseValidationError(f"{label} must be an integer, got {value!r}")
    return value


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _actual_key_audit(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    primary_key: list[str],
) -> dict[str, int]:
    key_sql = ", ".join(_quote_identifier(column) for column in primary_key)
    select_parts = [
        "count(*) FILTER (WHERE stay_id IS NULL) AS null_stay_id_rows"
    ]
    if "charttime" in primary_key:
        select_parts.append(
            "count(*) FILTER (WHERE charttime IS NULL) AS null_charttime_rows"
        )
    null_counts = connection.execute(
        f"SELECT {', '.join(select_parts)} FROM read_parquet(?)",
        [os.fspath(parquet_path)],
    ).fetchone()
    assert null_counts is not None

    duplicate = connection.execute(
        f"""
        SELECT 1
        FROM (
            SELECT {key_sql}
            FROM read_parquet(?)
            GROUP BY {key_sql}
            HAVING count(*) > 1
            LIMIT 1
        ) AS duplicate_keys
        """,
        [os.fspath(parquet_path)],
    ).fetchone()
    result = {"null_stay_id_rows": int(null_counts[0])}
    if "charttime" in primary_key:
        result["null_charttime_rows"] = int(null_counts[1])
    result["duplicate_key_groups"] = int(duplicate is not None)
    return result


def _validate_sidecar(database_root: Path, manifest: dict[str, Any]) -> None:
    receipt = manifest.get("column_metadata")
    if not isinstance(receipt, dict):
        raise ReleaseValidationError(
            f"{database_root.name}: manifest.column_metadata receipt is missing"
        )
    relative = receipt.get("file")
    expected_hash = receipt.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ReleaseValidationError(
            f"{database_root.name}: column_metadata.file must be non-empty"
        )
    sidecar_candidate = database_root / relative
    if sidecar_candidate.is_symlink():
        raise ReleaseValidationError(
            f"{database_root.name}: column metadata must not be a symlink: {relative}"
        )
    sidecar = sidecar_candidate.resolve()
    try:
        sidecar.relative_to(database_root.resolve())
    except ValueError as exc:
        raise ReleaseValidationError(
            f"{database_root.name}: column metadata escapes its export root"
        ) from exc
    if not sidecar.is_file():
        raise ReleaseValidationError(
            f"{database_root.name}: missing regular column metadata file: {relative}"
        )
    actual_hash = _sha256_file(sidecar)
    if expected_hash != actual_hash:
        raise ReleaseValidationError(
            f"{database_root.name}: column metadata SHA-256 mismatch: "
            f"manifest={expected_hash!r}, actual={actual_hash}"
        )


def _validate_file_entry(
    *,
    connection: duckdb.DuckDBPyConnection,
    database: str,
    database_root: Path,
    entry: dict[str, Any],
) -> dict[str, Any]:
    module = entry.get("module")
    if module not in MODULES:
        raise ReleaseValidationError(
            f"{database}: unknown or missing manifest module: {module!r}"
        )
    label = f"{database}/{module}"
    expected_file = f"{module}.parquet"
    if entry.get("file") != expected_file:
        raise ReleaseValidationError(
            f"{label}: manifest.file must be {expected_file!r}, "
            f"got {entry.get('file')!r}"
        )
    parquet_path = database_root / expected_file
    if not parquet_path.is_file() or parquet_path.is_symlink():
        raise ReleaseValidationError(f"{label}: missing regular Parquet file")

    primary_key, row_grain, null_key_equality = _expected_grain(module)
    if entry.get("primary_key") != primary_key:
        raise ReleaseValidationError(
            f"{label}: primary_key must be {primary_key!r}, "
            f"got {entry.get('primary_key')!r}"
        )
    if entry.get("row_grain") != row_grain:
        raise ReleaseValidationError(
            f"{label}: row_grain must be {row_grain!r}, "
            f"got {entry.get('row_grain')!r}"
        )

    audit = entry.get("row_grain_audit")
    if not isinstance(audit, dict):
        raise ReleaseValidationError(f"{label}: row_grain_audit is missing")
    if audit.get("primary_key") != primary_key:
        raise ReleaseValidationError(f"{label}: row_grain_audit primary key mismatch")
    if audit.get("row_grain") != row_grain:
        raise ReleaseValidationError(f"{label}: row_grain_audit row grain mismatch")
    if audit.get("null_key_equality") != null_key_equality:
        raise ReleaseValidationError(
            f"{label}: row_grain_audit null-key policy mismatch"
        )
    if audit.get("duplicate_excess_rows_after") != 0:
        raise ReleaseValidationError(
            f"{label}: row_grain_audit does not report a unique published key"
        )

    actual_bytes = parquet_path.stat().st_size
    manifest_bytes = _require_plain_int(
        entry.get("parquet_bytes"), label=f"{label}.parquet_bytes"
    )
    if manifest_bytes != actual_bytes:
        raise ReleaseValidationError(
            f"{label}: Parquet size mismatch: manifest={manifest_bytes}, "
            f"actual={actual_bytes}"
        )
    actual_sha256 = _sha256_file(parquet_path)
    if entry.get("parquet_sha256") != actual_sha256:
        raise ReleaseValidationError(
            f"{label}: Parquet SHA-256 mismatch: "
            f"manifest={entry.get('parquet_sha256')!r}, actual={actual_sha256}"
        )

    try:
        parquet_file = pq.ParquetFile(parquet_path)
        arrow_schema = parquet_file.schema_arrow
        actual_rows = parquet_file.metadata.num_rows
    except Exception as exc:
        raise ReleaseValidationError(f"{label}: invalid Parquet footer: {exc}") from exc
    actual_schema = {field.name: str(field.type) for field in arrow_schema}
    if entry.get("physical_schema") != actual_schema:
        raise ReleaseValidationError(
            f"{label}: manifest physical_schema does not match the Parquet schema"
        )
    for key in primary_key:
        if key not in actual_schema:
            raise ReleaseValidationError(
                f"{label}: Parquet file is missing primary-key column {key!r}"
            )
    manifest_rows = _require_plain_int(entry.get("rows"), label=f"{label}.rows")
    if manifest_rows != actual_rows:
        raise ReleaseValidationError(
            f"{label}: row count mismatch: manifest={manifest_rows}, "
            f"footer={actual_rows}"
        )
    if audit.get("published_rows") != actual_rows:
        raise ReleaseValidationError(
            f"{label}: row_grain_audit.published_rows does not match Parquet"
        )

    key_audit = _actual_key_audit(connection, parquet_path, primary_key)
    if key_audit["null_stay_id_rows"]:
        raise ReleaseValidationError(f"{label}: stay_id contains NULL values")
    if key_audit["duplicate_key_groups"]:
        raise ReleaseValidationError(
            f"{label}: Parquet primary key is not unique under NULL-equal semantics"
        )
    if "charttime" in primary_key and audit.get("null_charttime_rows_after") != (
        key_audit["null_charttime_rows"]
    ):
        raise ReleaseValidationError(
            f"{label}: audited NULL charttime count does not match Parquet"
        )

    physical_concepts = entry.get("physical_concept_ids")
    if not isinstance(physical_concepts, list) or not all(
        isinstance(value, str) and value for value in physical_concepts
    ):
        raise ReleaseValidationError(
            f"{label}: physical_concept_ids must be a non-empty string list"
        )
    expected_columns = [*primary_key, *physical_concepts]
    if list(actual_schema) != expected_columns:
        raise ReleaseValidationError(
            f"{label}: Parquet columns do not follow primary-key + concept order"
        )
    return {
        "module": module,
        "rows": actual_rows,
        "bytes": actual_bytes,
        "schema": tuple(actual_schema.items()),
        "physical_concepts": tuple(physical_concepts),
    }


def validate_release(run_root: Path) -> dict[str, Any]:
    """Validate ``run_root/exports`` without writing inside ``run_root``."""

    run_root = run_root.resolve()
    export_root = run_root / "exports"
    if not export_root.is_dir():
        raise ReleaseValidationError(f"Missing export root: {export_root}")
    actual_database_dirs = {
        path.name for path in export_root.iterdir() if path.is_dir()
    }
    if actual_database_dirs != set(DATABASES):
        raise ReleaseValidationError(
            "Export database directories do not match the six-database contract: "
            f"expected={sorted(DATABASES)}, actual={sorted(actual_database_dirs)}"
        )

    source_manifest_sha256: dict[str, str] = {}
    database_commits: dict[str, str] = {}
    runtime_provenance: dict[str, dict[str, Any]] = {}
    module_receipts: dict[str, list[dict[str, Any]]] = {
        module: [] for module in MODULES
    }
    database_totals: dict[str, dict[str, int]] = {}
    total_rows = 0
    total_bytes = 0

    with tempfile.TemporaryDirectory(prefix="easyicu-release-seal-") as temp_dir:
        connection = duckdb.connect(
            ":memory:",
            config={
                "threads": "1",
                "memory_limit": "1GB",
                "temp_directory": temp_dir,
            },
        )
        try:
            for database in DATABASES:
                database_root = export_root / database
                manifest_path = database_root / "_manifest.json"
                if manifest_path.is_symlink():
                    raise ReleaseValidationError(
                        f"{database}: _manifest.json must be a regular file"
                    )
                manifest, raw_manifest = _read_json_object(
                    manifest_path, label=f"{database} manifest"
                )
                if manifest.get("schema_version") != NATIVE_SCHEMA_VERSION:
                    raise ReleaseValidationError(
                        f"{database}: unsupported schema_version "
                        f"{manifest.get('schema_version')!r}"
                    )
                if manifest.get("contract_revision") != CONTRACT_REVISION:
                    raise ReleaseValidationError(
                        f"{database}: unsupported contract_revision "
                        f"{manifest.get('contract_revision')!r}"
                    )
                if manifest.get("database") != database:
                    raise ReleaseValidationError(
                        f"{database}: manifest database identity mismatch"
                    )
                provenance = manifest.get("runtime_provenance")
                if not isinstance(provenance, dict):
                    raise ReleaseValidationError(
                        f"{database}: runtime_provenance is missing"
                    )
                commit = provenance.get("easyicu_git_commit")
                if not isinstance(commit, str) or len(commit) != 40:
                    raise ReleaseValidationError(
                        f"{database}: runtime EasyICU commit is not a full Git SHA"
                    )
                if provenance.get("easyicu_git_dirty") is not False:
                    raise ReleaseValidationError(
                        f"{database}: extraction runtime was not a clean Git checkout"
                    )
                database_commits[database] = commit
                runtime_provenance[database] = provenance

                entries = manifest.get("files")
                if not isinstance(entries, list) or len(entries) != len(MODULES):
                    raise ReleaseValidationError(
                        f"{database}: expected {len(MODULES)} manifest file entries"
                    )
                if not all(isinstance(entry, dict) for entry in entries):
                    raise ReleaseValidationError(
                        f"{database}: every manifest file entry must be an object"
                    )
                entry_modules = [entry.get("module") for entry in entries]
                if len(set(entry_modules)) != len(entry_modules):
                    raise ReleaseValidationError(
                        f"{database}: duplicate module entries in manifest"
                    )
                if set(entry_modules) != set(MODULES):
                    raise ReleaseValidationError(
                        f"{database}: manifest module set does not match the 19-module contract"
                    )
                actual_parquets = {
                    path.relative_to(database_root).as_posix()
                    for path in database_root.rglob("*.parquet")
                    if path.is_file()
                }
                expected_parquets = {f"{module}.parquet" for module in MODULES}
                if actual_parquets != expected_parquets:
                    raise ReleaseValidationError(
                        f"{database}: Parquet file set mismatch: "
                        f"missing={sorted(expected_parquets - actual_parquets)}, "
                        f"extra={sorted(actual_parquets - expected_parquets)}"
                    )
                _validate_sidecar(database_root, manifest)
                by_module = {entry["module"]: entry for entry in entries}
                database_rows = 0
                database_bytes = 0
                for module in MODULES:
                    receipt = _validate_file_entry(
                        connection=connection,
                        database=database,
                        database_root=database_root,
                        entry=by_module[module],
                    )
                    module_receipts[module].append(receipt)
                    database_rows += receipt["rows"]
                    database_bytes += receipt["bytes"]
                    total_rows += receipt["rows"]
                    total_bytes += receipt["bytes"]
                database_totals[database] = {
                    "rows": database_rows,
                    "parquet_bytes": database_bytes,
                }
                source_manifest_sha256[database] = hashlib.sha256(
                    raw_manifest
                ).hexdigest()
        finally:
            connection.close()

    module_concepts: dict[str, list[str]] = {}
    for module, receipts in module_receipts.items():
        reference_schema = receipts[0]["schema"]
        reference_concepts = receipts[0]["physical_concepts"]
        for receipt in receipts[1:]:
            if receipt["schema"] != reference_schema:
                raise ReleaseValidationError(
                    f"{module}: physical Parquet schema differs across databases"
                )
            if receipt["physical_concepts"] != reference_concepts:
                raise ReleaseValidationError(
                    f"{module}: physical concept order differs across databases"
                )
        module_concepts[module] = list(reference_concepts)

    return {
        "database_commits": database_commits,
        "runtime_provenance": runtime_provenance,
        "source_manifest_sha256": source_manifest_sha256,
        "module_concepts": module_concepts,
        "database_totals": database_totals,
        "total_rows": total_rows,
        "total_parquet_bytes": total_bytes,
    }


def validate_extraction_timing(
    *, run_root: Path, validation: dict[str, Any]
) -> dict[str, Any]:
    """Bind the six extraction durations and process-tree peaks to the release."""

    path = run_root / "database_extraction_timing.csv"
    if not path.is_file() or path.is_symlink():
        raise ReleaseValidationError(
            f"Missing regular extraction timing evidence: {path}"
        )
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ReleaseValidationError(f"Cannot parse extraction timing CSV: {exc}") from exc
    by_database = {row.get("database"): row for row in rows}
    if len(rows) != len(DATABASES) or set(by_database) != set(DATABASES):
        raise ReleaseValidationError(
            "Extraction timing CSV must contain exactly one row for each database"
        )

    records: dict[str, dict[str, Any]] = {}
    for database in DATABASES:
        row = by_database[database]
        label = f"timing[{database}]"
        try:
            module_count = int(row.get("module_count") or "")
            valid_parquet_count = int(row.get("valid_parquet_count") or "")
            process_exit_code = int(row.get("process_exit_code") or "")
            total_rows = int(row.get("total_rows") or "")
            total_bytes = int(row.get("total_parquet_bytes") or "")
            elapsed_seconds = float(row.get("elapsed_seconds") or "")
            peak_rss_mb = float(row.get("peak_process_tree_rss_mb") or "")
            peak_pss_mb = float(row.get("peak_process_tree_pss_mb") or "")
        except (TypeError, ValueError) as exc:
            raise ReleaseValidationError(f"{label} has invalid numeric fields") from exc
        expected = validation["database_totals"][database]
        if (
            row.get("status") != "complete"
            or row.get("error") not in {None, ""}
            or process_exit_code != 0
            or module_count != len(MODULES)
            or valid_parquet_count != len(MODULES)
            or total_rows != expected["rows"]
            or total_bytes != expected["parquet_bytes"]
            or elapsed_seconds <= 0
            or peak_rss_mb <= 0
            or peak_pss_mb <= 0
        ):
            raise ReleaseValidationError(
                f"{label} is incomplete or disagrees with the sealed Parquet package"
            )
        records[database] = {
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": round(elapsed_seconds / 60.0, 3),
            "batch_strategy": row.get("batch_strategy"),
            "peak_process_tree_rss_mb": peak_rss_mb,
            "peak_process_tree_pss_mb": peak_pss_mb,
        }
    return {
        "file": path.name,
        "sha256": _sha256_file(path),
        "databases": records,
    }


def build_run_metadata(
    *,
    run_id: str,
    validation: dict[str, Any],
    extraction_timing: dict[str, Any],
    execution_profile: str,
) -> dict[str, Any]:
    if not run_id.strip():
        raise ReleaseValidationError("run_id must be non-empty")
    commits = set(validation["database_commits"].values())
    metadata: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "verified",
        "package_kind": "easyicu_six_database_native_v2_release",
        "contract_revision": CONTRACT_REVISION,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "database_order": list(DATABASES),
        "database_commits": validation["database_commits"],
        "source_manifest_sha256": validation["source_manifest_sha256"],
        "output_layout": "exports/{database}/{module}.parquet",
        "module_order": list(MODULES),
        "module_concepts": validation["module_concepts"],
        "module_count": len(MODULES),
        "database_count": len(DATABASES),
        "expected_parquet_count": EXPECTED_PARQUET_COUNT,
        "corrections": list(CORRECTIONS),
        "extraction_execution": {
            "profile": execution_profile,
            "portable_16gb_validated": execution_profile == "16gb",
            "timing": extraction_timing,
        },
        "validation": {
            "audited_parquet_count": EXPECTED_PARQUET_COUNT,
            "valid_footer_count": EXPECTED_PARQUET_COUNT,
            "primary_key_contract_verified_count": EXPECTED_PARQUET_COUNT,
            "primary_key_uniqueness_verified_count": EXPECTED_PARQUET_COUNT,
            "row_grain_contract_verified_count": EXPECTED_PARQUET_COUNT,
            "parquet_sha256_verified_count": EXPECTED_PARQUET_COUNT,
            "parquet_bytes_verified_count": EXPECTED_PARQUET_COUNT,
            "exact_schema_module_count": len(MODULES),
            "total_rows": validation["total_rows"],
            "total_parquet_bytes": validation["total_parquet_bytes"],
            "failures": [],
        },
        "runtime_provenance": validation["runtime_provenance"],
        "provenance": {
            "database_native_manifests": "exports/{database}/_manifest.json",
            "sealer": "scripts/releases/EX-A01_seal_full6_release.py",
            "sealer_sha256": _sha256_file(Path(__file__)),
        },
    }
    if len(commits) == 1:
        metadata["easyicu_commit"] = next(iter(commits))
    return metadata


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def seal_release(
    *,
    run_root: Path,
    run_id: str | None = None,
    execution_profile: str,
) -> Path:
    resolved_run_root = run_root.resolve()
    validation = validate_release(resolved_run_root)
    extraction_timing = validate_extraction_timing(
        run_root=resolved_run_root, validation=validation
    )
    metadata = build_run_metadata(
        run_id=run_id if run_id is not None else resolved_run_root.name,
        validation=validation,
        extraction_timing=extraction_timing,
        execution_profile=execution_profile,
    )
    destination = resolved_run_root / "run_metadata.json"
    _atomic_write_json(destination, metadata)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="Run directory containing exports/{database}/.",
    )
    parser.add_argument(
        "--run-id",
        help="Immutable release ID; defaults to the run directory name.",
    )
    parser.add_argument(
        "--execution-profile",
        required=True,
        choices=("server-adaptive", "16gb", "portable-low-memory"),
        help="Resource profile actually enforced for this extraction run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = seal_release(
        run_root=args.run_root,
        run_id=args.run_id,
        execution_profile=args.execution_profile,
    )
    print(f"Sealed verified release metadata: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
