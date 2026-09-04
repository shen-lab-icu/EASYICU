#!/usr/bin/env python3
"""EX-A03: refresh selected raw-derived modules in a sealed six-database run.

This entry point is deliberately narrower than a full re-extraction.  It
copies a verified six-database package by hard link, re-reads raw data only for
the named modules and selected databases. It republishes all six native packages
from one clean EasyICU checkout because the release sealer requires one common
runtime commit. Unselected databases are never reread from raw data, and a
bounded multiset audit must prove their table contents unchanged. Thus the source
run is immutable, while the derived candidate records an explicit per-database
refresh scope and can be sealed by ``EX-A01_seal_full6_release.py``.

Only correctness modules and their declared downstream closure are allowlisted.
``renal`` has ascertainment-aware KDIGO outputs. ``outcome`` preserves the
owner-issued death-event time companion required by landmark analyses.
``respiratory`` removes
implicit room-air FiO2 imputation and therefore expands to ``sofa1_score`` and
``sofa2_score``, the shared infection evidence required at execution time, and
the two Sepsis-SOFA labels that consume those scores. This is not a generic way
to bypass the full extraction controller.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from easyicu.api import extract_database  # noqa: E402
from easyicu.api.extraction import EXTRACT_MODULES  # noqa: E402


def _load_republisher():
    """Load EX-A02 helpers without making the hyphenated filename importable."""

    path = REPOSITORY_ROOT / "scripts/releases/EX-A02_republish_full6_candidate.py"
    spec = importlib.util.spec_from_file_location("easyicu_republisher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load republication helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPUBLICATION = _load_republisher()
DATABASES: tuple[str, ...] = tuple(REPUBLICATION.DATABASES)
MODULES: tuple[str, ...] = tuple(REPUBLICATION.MODULES)
DIRECT_REFRESHABLE_MODULES = frozenset(
    {"outcome", "renal", "respiratory", "sofa2_score"}
)
MODULE_DEPENDENCY_CLOSURE: dict[str, tuple[str, ...]] = {
    "outcome": ("outcome",),
    "renal": ("renal",),
    "respiratory": (
        "respiratory",
        "sepsis_shared",
        "sofa1_score",
        "sofa2_score",
        "sepsis3_sofa1",
        "sepsis3_sofa2",
    ),
    # Targeted repair path for owner-issued SOFA-2 receipt companions. The
    # parent candidate must already contain a raw-refreshed respiratory input;
    # downstream M1 validates that parent provenance before using the child.
    "sofa2_score": (
        "sepsis_shared",
        "sofa2_score",
        "sepsis3_sofa2",
    ),
}
SCHEMA_VERSION = "easyicu_full6_selected_module_refresh_v2"
LEGACY_SCHEMA_VERSION = "easyicu_full6_selected_module_refresh_v1"


class ModuleRefreshError(ValueError):
    """Raised before a refresh could make a safe, sealable candidate."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModuleRefreshError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModuleRefreshError(f"{label} must be one JSON object: {path}")
    return value


def _parse_data_path_overrides(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        database, separator, path = raw.partition("=")
        if not separator or database not in DATABASES or not path.strip():
            raise ModuleRefreshError(
                "--data-path must be DATABASE=PATH for one of "
                f"{', '.join(DATABASES)}; got {raw!r}"
            )
        if database in parsed:
            raise ModuleRefreshError(f"Duplicate --data-path override: {database}")
        parsed[database] = str(Path(path.strip()).expanduser().resolve())
    return parsed


def _validate_databases(databases: Sequence[str]) -> tuple[str, ...]:
    """Return a canonical non-empty subset of the six release databases."""

    requested = tuple(dict.fromkeys(str(database) for database in databases))
    if not requested:
        raise ModuleRefreshError("At least one --database is required")
    unknown = set(requested) - set(DATABASES)
    if unknown:
        raise ModuleRefreshError(f"Unknown refresh databases: {sorted(unknown)}")
    return tuple(database for database in DATABASES if database in requested)


def _resolve_data_paths(
    source_manifest: Mapping[str, Any],
    overrides: Mapping[str, str],
    databases: Sequence[str] = DATABASES,
) -> dict[str, str]:
    recorded = source_manifest.get("data_paths")
    if not isinstance(recorded, dict):
        raise ModuleRefreshError("Source run manifest lacks data_paths")
    paths: dict[str, str] = {}
    for database in databases:
        raw = overrides.get(database, recorded.get(database))
        if not isinstance(raw, str) or not raw:
            raise ModuleRefreshError(f"Missing recorded raw path for {database}")
        resolved = Path(raw).expanduser().resolve()
        if not resolved.is_dir():
            raise ModuleRefreshError(
                f"Raw source path is unavailable for {database}: {resolved}"
            )
        paths[database] = str(resolved)
    return paths


def _validate_modules(modules: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(module) for module in modules))
    if not selected:
        raise ModuleRefreshError("At least one --module is required")
    unknown = set(selected) - set(MODULES)
    if unknown:
        raise ModuleRefreshError(f"Unknown extraction modules: {sorted(unknown)}")
    disallowed = set(selected) - DIRECT_REFRESHABLE_MODULES
    if disallowed:
        raise ModuleRefreshError(
            "This audited refresh entry point currently allows only outcome, "
            "renal, respiratory and sofa2_score; "
            f"got disallowed modules: {sorted(disallowed)}"
        )
    return selected


def _expand_module_dependency_closure(modules: Sequence[str]) -> tuple[str, ...]:
    """Return requested correctness modules with every derived consumer."""

    requested = _validate_modules(modules)
    required = {
        module
        for requested_module in requested
        for module in MODULE_DEPENDENCY_CLOSURE[requested_module]
    }
    return tuple(module for module in MODULES if module in required)


def _canonical_provenance_modules(
    values: Any, *, label: str
) -> tuple[str, ...]:
    """Validate and canonically order a module list stored in provenance."""

    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ModuleRefreshError(f"{label} must be a JSON string array")
    if len(values) != len(set(values)):
        raise ModuleRefreshError(f"{label} contains duplicate modules")
    unknown = set(values) - set(MODULES)
    if unknown:
        raise ModuleRefreshError(f"{label} contains unknown modules: {sorted(unknown)}")
    return tuple(module for module in MODULES if module in set(values))


def _database_refresh_scope(
    provenance: Mapping[str, Any], *, label: str
) -> dict[str, list[str]]:
    """Return cumulative per-database lineage after fail-closed validation.

    Version 1 predated database subsets and always re-read every database.  We
    infer its scope only when the six raw paths and six per-database module
    receipts prove that legacy invariant.  Version 2 records the scope
    directly and must agree exactly with the top-level cumulative module list.
    """

    schema_version = provenance.get("schema_version")
    refreshed_modules = _canonical_provenance_modules(
        provenance.get("refreshed_modules"),
        label=f"{label}.refreshed_modules",
    )
    if not refreshed_modules:
        raise ModuleRefreshError(f"{label}.refreshed_modules must be non-empty")
    if provenance.get("raw_database_reread") is not True:
        raise ModuleRefreshError(f"{label} does not prove a raw-database reread")

    if schema_version == LEGACY_SCHEMA_VERSION:
        if provenance.get("publication_easyicu_git_dirty") is not False:
            raise ModuleRefreshError(
                f"{label} legacy publication checkout was not recorded clean"
            )
        raw_paths = provenance.get("raw_data_paths")
        runtime = provenance.get("per_database_runtime")
        if not isinstance(raw_paths, dict) or set(raw_paths) != set(DATABASES):
            raise ModuleRefreshError(
                f"{label} legacy raw_data_paths must cover exactly all six databases"
            )
        if not isinstance(runtime, dict) or set(runtime) != set(DATABASES):
            raise ModuleRefreshError(
                f"{label} legacy runtime must cover exactly all six databases"
            )
        for database in DATABASES:
            database_runtime = runtime.get(database)
            module_runtime = (
                database_runtime.get("modules")
                if isinstance(database_runtime, dict)
                else None
            )
            if not isinstance(module_runtime, dict) or set(module_runtime) != set(
                refreshed_modules
            ):
                raise ModuleRefreshError(
                    f"{label} legacy runtime lacks refreshed-module receipts "
                    f"for {database}"
                )
            if not isinstance(raw_paths.get(database), str) or not raw_paths[database]:
                raise ModuleRefreshError(
                    f"{label} legacy raw path is invalid for {database}"
                )
        return {
            database: list(refreshed_modules) for database in DATABASES
        }

    if schema_version == SCHEMA_VERSION:
        if provenance.get("publication_easyicu_git_dirty") is not False:
            raise ModuleRefreshError(
                f"{label} publication checkout was not recorded clean"
            )
        recorded_scope = provenance.get("per_database_refreshed_modules")
        if not isinstance(recorded_scope, dict) or set(recorded_scope) != set(
            DATABASES
        ):
            raise ModuleRefreshError(
                f"{label}.per_database_refreshed_modules must cover all six databases"
            )
        scope = {
            database: list(
                _canonical_provenance_modules(
                    recorded_scope[database],
                    label=(
                        f"{label}.per_database_refreshed_modules[{database!r}]"
                    ),
                )
            )
            for database in DATABASES
        }
        scope_union = {
            module
            for database_modules in scope.values()
            for module in database_modules
        }
        if scope_union != set(refreshed_modules):
            raise ModuleRefreshError(
                f"{label} top-level refreshed_modules disagrees with its "
                "per-database cumulative scope"
            )
        raw_paths = provenance.get("raw_data_paths")
        runtime = provenance.get("per_database_runtime")
        if not isinstance(raw_paths, dict) or not isinstance(runtime, dict):
            raise ModuleRefreshError(
                f"{label} lacks cumulative raw paths or runtime receipts"
            )
        for database in DATABASES:
            if not scope[database]:
                continue
            database_runtime = runtime.get(database)
            module_runtime = (
                database_runtime.get("modules")
                if isinstance(database_runtime, dict)
                else None
            )
            if not isinstance(module_runtime, dict) or set(module_runtime) != set(
                scope[database]
            ):
                raise ModuleRefreshError(
                    f"{label} runtime lacks cumulative receipts for {database}"
                )
            if not isinstance(raw_paths.get(database), str) or not raw_paths[database]:
                raise ModuleRefreshError(
                    f"{label} raw path is invalid for {database}"
                )
        return scope

    raise ModuleRefreshError(
        f"{label} has unsupported schema_version {schema_version!r}"
    )


def _require_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ModuleRefreshError(f"Missing regular {label}: {path}")


def _validate_sealed_source_receipts(
    source: Path, source_receipts: Mapping[str, Mapping[str, Any]]
) -> str:
    """Prove the refresh parent is sealed and still matches its native bytes."""

    metadata_path = source / "run_metadata.json"
    _require_regular_file(metadata_path, label="source release metadata")
    metadata = _read_json(metadata_path, label="source release metadata")
    if metadata.get("status") != "verified":
        raise ModuleRefreshError("Source release metadata is not verified")
    sealed_manifests = metadata.get("source_manifest_sha256")
    if not isinstance(sealed_manifests, dict) or set(sealed_manifests) != set(
        DATABASES
    ):
        raise ModuleRefreshError(
            "Source release metadata lacks six native-manifest receipts"
        )
    commits: set[str] = set()
    for database in DATABASES:
        receipt = source_receipts.get(database)
        if not isinstance(receipt, Mapping):
            raise ModuleRefreshError(f"Source receipt is invalid for {database}")
        if receipt.get("native_manifest_sha256") != sealed_manifests[database]:
            raise ModuleRefreshError(
                f"Source native manifest changed after sealing for {database}"
            )
        commit = receipt.get("easyicu_git_commit")
        if not isinstance(commit, str) or len(commit) != 40:
            raise ModuleRefreshError(
                f"Source runtime commit is invalid for {database}"
            )
        if receipt.get("easyicu_git_dirty") is not False:
            raise ModuleRefreshError(
                f"Source runtime checkout was dirty for {database}"
            )
        commits.add(commit)
    if len(commits) != 1:
        raise ModuleRefreshError(
            "Source release does not have one harmonized EasyICU commit"
        )
    source_commit = next(iter(commits))
    if metadata.get("easyicu_commit") != source_commit:
        raise ModuleRefreshError(
            "Source release commit disagrees with its six native manifests"
        )
    return source_commit


def _replace_selected_module_files(
    *, staging_root: Path, destination_database_root: Path, modules: Sequence[str]
) -> None:
    """Atomically replace only selected candidate files, never source bytes."""

    for module in modules:
        source_parquet = staging_root / f"{module}.parquet"
        source_manifest = staging_root / f"{module}.manifest.json"
        _require_regular_file(source_parquet, label=f"{module} staged Parquet")
        _require_regular_file(source_manifest, label=f"{module} staged manifest")
        os.replace(source_parquet, destination_database_root / source_parquet.name)
        os.replace(source_manifest, destination_database_root / source_manifest.name)


def _module_is_canonical_refresh(database_root: Path, modules: Sequence[str]) -> bool:
    """Check whether a candidate already contains the selected new columns."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - package is release-required
        raise ModuleRefreshError(
            "pyarrow is required to resume a module refresh"
        ) from exc
    for module in modules:
        parquet = database_root / f"{module}.parquet"
        manifest = database_root / f"{module}.manifest.json"
        if (
            parquet.is_symlink()
            or manifest.is_symlink()
            or not parquet.is_file()
            or not manifest.is_file()
        ):
            return False
        try:
            columns = set(pq.read_schema(parquet).names)
        except Exception:
            return False
        if "stay_id" not in columns or not set(EXTRACT_MODULES[module]).issubset(
            columns
        ):
            return False
    return True


def _validate_refreshed_score_content(
    database_root: Path, modules: Sequence[str]
) -> None:
    """Refuse a structurally valid refresh whose primary score is all null.

    Schema checks alone did not catch the 2026-09 IMV regression: both SOFA
    files had the expected columns and millions of rows, but interval handling
    had displaced their components so every total score was missing. Scan only
    the primary score column in bounded Arrow batches before any candidate file
    is promoted.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - package is release-required
        raise ModuleRefreshError(
            "pyarrow is required to validate refreshed score content"
        ) from exc

    for module, score_column in (
        ("sofa1_score", "sofa"),
        ("sofa2_score", "sofa2"),
    ):
        if module not in modules:
            continue
        path = database_root / f"{module}.parquet"
        _require_regular_file(path, label=f"{module} staged Parquet")
        parquet = pq.ParquetFile(path)
        if score_column not in parquet.schema_arrow.names:
            raise ModuleRefreshError(
                f"{module} staged Parquet lacks primary score {score_column!r}"
            )
        rows = 0
        non_null = 0
        for batch in parquet.iter_batches(
            batch_size=262_144, columns=[score_column], use_threads=False
        ):
            column = batch.column(0)
            rows += len(column)
            non_null += len(column) - column.null_count
        if rows == 0 or non_null == 0:
            raise ModuleRefreshError(
                f"{module} refresh is unusable: {score_column} has "
                f"{non_null} non-null values across {rows} rows"
            )


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _parquet_multiset_receipt(connection: Any, path: Path) -> dict[str, Any]:
    """Return an order-independent logical-content receipt for one Parquet."""

    escaped_path = str(path.resolve()).replace("'", "''")
    relation = f"read_parquet('{escaped_path}')"
    described = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    schema = [(str(row[0]), str(row[1])) for row in described]
    if not schema:
        raise ModuleRefreshError(f"Cannot audit a zero-column Parquet: {path}")
    columns = ", ".join(_quote_identifier(name) for name, _dtype in schema)
    row_hash = f"hash({columns})"
    row_count, hash_xor, hash_sum, hash_min, hash_max = connection.execute(
        "SELECT count(*)::BIGINT, "
        f"bit_xor({row_hash})::UBIGINT, "
        f"sum({row_hash}::HUGEINT)::VARCHAR, "
        f"min({row_hash})::UBIGINT, max({row_hash})::UBIGINT "
        f"FROM {relation}"
    ).fetchone()
    schema_sha256 = hashlib.sha256(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "rows": int(row_count),
        "schema_sha256": schema_sha256,
        "row_hash_xor": str(hash_xor if hash_xor is not None else 0),
        "row_hash_sum": str(hash_sum if hash_sum is not None else 0),
        "row_hash_min": str(hash_min if hash_min is not None else 0),
        "row_hash_max": str(hash_max if hash_max is not None else 0),
    }


def _validate_publication_only_database_semantics(
    source_database_root: Path,
    candidate_database_root: Path,
    *,
    modules: Sequence[str] = MODULES,
) -> dict[str, Any]:
    """Prove publication-only repackaging did not alter table contents.

    The dual commutative 64-bit row-hash aggregates, row count, extrema and
    schema digest make this insensitive to row ordering and Parquet encoding
    while still failing closed on any observed logical-content difference.
    DuckDB is bounded to one thread and 1 GiB with spill beside the candidate.
    """

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - release dependency
        raise ModuleRefreshError(
            "duckdb is required for publication-only semantic validation"
        ) from exc

    receipts: dict[str, dict[str, Any]] = {}
    audit_parent = candidate_database_root.parents[1]
    with tempfile.TemporaryDirectory(
        prefix=f".semantic-audit-{candidate_database_root.name}-",
        dir=audit_parent,
    ) as spill_dir:
        connection = duckdb.connect(
            ":memory:",
            config={
                "threads": "1",
                "memory_limit": "1GB",
                "temp_directory": spill_dir,
            },
        )
        try:
            for module in modules:
                source_path = source_database_root / f"{module}.parquet"
                candidate_path = candidate_database_root / f"{module}.parquet"
                _require_regular_file(source_path, label=f"{module} source Parquet")
                _require_regular_file(
                    candidate_path, label=f"{module} publication-only Parquet"
                )
                source_receipt = _parquet_multiset_receipt(connection, source_path)
                candidate_receipt = _parquet_multiset_receipt(
                    connection, candidate_path
                )
                if source_receipt != candidate_receipt:
                    raise ModuleRefreshError(
                        f"{candidate_database_root.name}/{module}: publication-only "
                        "repackaging changed logical table content; "
                        f"source={source_receipt}, candidate={candidate_receipt}"
                    )
                receipts[module] = source_receipt
        finally:
            connection.close()
    return {
        "status": "PASS",
        "algorithm": (
            "duckdb_order_independent_schema_count_dual_hash_sum_xor_min_max_v1"
        ),
        "modules": receipts,
    }


def _module_files_are_detached_from_source(
    source_database_root: Path,
    candidate_database_root: Path,
    modules: Sequence[str],
) -> bool:
    """Prove atomic replacement, not merely a schema-matching source clone."""

    for module in modules:
        for suffix in (".parquet", ".manifest.json"):
            source = source_database_root / f"{module}{suffix}"
            candidate = candidate_database_root / f"{module}{suffix}"
            if (
                source.is_symlink()
                or candidate.is_symlink()
                or not source.is_file()
                or not candidate.is_file()
            ):
                return False
            try:
                if os.path.samefile(source, candidate):
                    return False
            except OSError:
                return False
    return True


def _metrics_from_module_manifests(
    database_root: Path, modules: Sequence[str]
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for module in modules:
        manifest = _read_json(
            database_root / f"{module}.manifest.json",
            label=f"{module} refresh manifest",
        )
        try:
            metrics[module] = {
                "elapsed_seconds": float(manifest.get("elapsed_sec") or 0.0),
                "peak_rss_mb": float(manifest.get("peak_rss_mb") or 0.0),
                "peak_working_set_mb": float(
                    manifest.get("peak_working_set_mb") or 0.0
                ),
            }
        except (TypeError, ValueError) as exc:
            raise ModuleRefreshError(
                f"{module} refresh manifest has invalid runtime metrics"
            ) from exc
    return metrics


def _module_runtime_metrics(
    extraction: Mapping[str, Any], modules: Sequence[str]
) -> dict[str, dict[str, float]]:
    results = extraction.get("modules")
    if not isinstance(results, Mapping):
        raise ModuleRefreshError("Extraction result lacks per-module receipts")
    metrics: dict[str, dict[str, float]] = {}
    for module in modules:
        receipt = results.get(module)
        if not isinstance(receipt, Mapping):
            raise ModuleRefreshError(f"Extraction result lacks {module} receipt")
        errors = receipt.get("errors") or []
        if errors:
            raise ModuleRefreshError(f"{module} extraction failed: {list(errors)}")
        try:
            metrics[module] = {
                "elapsed_seconds": float(receipt.get("elapsed") or 0.0),
                "peak_rss_mb": float(receipt.get("peak_rss_mb") or 0.0),
                "peak_working_set_mb": float(receipt.get("peak_working_set_mb") or 0.0),
            }
        except (TypeError, ValueError) as exc:
            raise ModuleRefreshError(
                f"{module} extraction receipt has invalid runtime metrics"
            ) from exc
    return metrics


def _reconstruct_database_runtime_metrics(
    source_run_manifest: Mapping[str, Any],
    *,
    database: str,
    cumulative_refresh_runtime: Mapping[str, Any] | None,
) -> dict[str, dict[str, float]]:
    """Rebuild 19 module timings from original and cumulative refresh evidence."""

    sources = source_run_manifest.get("sources")
    source_record = sources.get(database) if isinstance(sources, Mapping) else None
    source_metrics = (
        source_record.get("module_metrics")
        if isinstance(source_record, Mapping)
        else None
    )
    if not isinstance(source_metrics, Mapping) or set(source_metrics) != set(MODULES):
        raise ModuleRefreshError(
            f"Source run manifest lacks complete module metrics for {database}"
        )
    metrics: dict[str, dict[str, float]] = {}
    for module in MODULES:
        receipt = source_metrics.get(module)
        if not isinstance(receipt, Mapping):
            raise ModuleRefreshError(
                f"Source runtime receipt is invalid for {database}/{module}"
            )
        try:
            metrics[module] = {
                "elapsed_seconds": float(receipt["elapsed_seconds"]),
                "peak_rss_mb": float(receipt["peak_rss_mb"]),
                "peak_working_set_mb": float(receipt["peak_working_set_mb"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ModuleRefreshError(
                f"Source runtime receipt is incomplete for {database}/{module}"
            ) from exc

    refreshed_metrics = (
        cumulative_refresh_runtime.get("modules")
        if isinstance(cumulative_refresh_runtime, Mapping)
        else None
    )
    if refreshed_metrics is not None:
        if not isinstance(refreshed_metrics, Mapping):
            raise ModuleRefreshError(
                f"Refresh runtime module receipts are invalid for {database}"
            )
        for module, receipt in refreshed_metrics.items():
            if module not in MODULES or not isinstance(receipt, Mapping):
                raise ModuleRefreshError(
                    f"Refresh runtime receipt is invalid for {database}/{module}"
                )
            try:
                metrics[module] = {
                    "elapsed_seconds": float(receipt["elapsed_seconds"]),
                    "peak_rss_mb": float(receipt["peak_rss_mb"]),
                    "peak_working_set_mb": float(receipt["peak_working_set_mb"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ModuleRefreshError(
                    f"Refresh runtime receipt is incomplete for {database}/{module}"
                ) from exc
    return metrics


def _refresh_one_database(
    *,
    database: str,
    data_path: str,
    source_database_root: Path,
    candidate_root: Path,
    modules: Sequence[str],
    batch_size: int | None,
    reuse_completed_export: bool,
) -> dict[str, Any]:
    staging_root = candidate_root / ".module_refresh_staging" / database
    destination_database_root = candidate_root / "exports" / database
    # A cloned candidate deliberately starts with canonical source files, so
    # schema alone can never prove that raw data were re-read. Explicit resume
    # may reuse only a complete package whose selected Parquet and producer
    # manifests have all been atomically detached from their source hard links.
    if (
        reuse_completed_export
        and _module_is_canonical_refresh(destination_database_root, modules)
        and _module_files_are_detached_from_source(
            source_database_root, destination_database_root, modules
        )
    ):
        _validate_refreshed_score_content(destination_database_root, modules)
        return {
            "database": database,
            "data_path": data_path,
            "num_patients": None,
            "batch_size": None,
            "total_elapsed_seconds": None,
            "modules": _metrics_from_module_manifests(
                destination_database_root, modules
            ),
            "recovery_mode": (
                "explicit_resume_of_complete_files_detached_from_source_clone"
            ),
        }
    if staging_root.exists() or staging_root.is_symlink():
        if _module_is_canonical_refresh(staging_root, modules):
            _validate_refreshed_score_content(staging_root, modules)
            _replace_selected_module_files(
                staging_root=staging_root,
                destination_database_root=destination_database_root,
                modules=modules,
            )
            metrics = _metrics_from_module_manifests(destination_database_root, modules)
            shutil.rmtree(staging_root)
            return {
                "database": database,
                "data_path": data_path,
                "num_patients": None,
                "batch_size": None,
                "total_elapsed_seconds": None,
                "modules": metrics,
                "recovery_mode": "completed_staging_promoted",
            }
        raise ModuleRefreshError(
            f"Existing refresh staging is incomplete or not canonical: {staging_root}"
        )
    staging_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    extraction = extract_database(
        database,
        data_path=data_path,
        output_dir=staging_root,
        modules=list(modules),
        batch_size=batch_size,
        native_export_v2=True,
        stream_output_batches=True,
        adaptive_stream_batches=batch_size is None,
        verbose=True,
    )
    metrics = _module_runtime_metrics(extraction, modules)
    _validate_refreshed_score_content(staging_root, modules)
    _replace_selected_module_files(
        staging_root=staging_root,
        destination_database_root=destination_database_root,
        modules=modules,
    )
    # Successful files have been atomically moved into ``exports``.  Remove
    # the one-use staging area (including DuckDB spill files) from the
    # candidate, while retaining it on any exception for diagnosis.
    shutil.rmtree(staging_root)
    return {
        "database": database,
        "data_path": data_path,
        "num_patients": extraction.get("num_patients"),
        "batch_size": extraction.get("batch_size"),
        "total_elapsed_seconds": extraction.get("total_elapsed"),
        "modules": metrics,
    }


def _rebind_manifest_metrics(
    manifest: dict[str, Any], metrics: Mapping[str, Mapping[str, float]]
) -> None:
    for field, metric_name in (
        ("module_timings_seconds", "elapsed_seconds"),
        ("module_peak_rss_mb", "peak_rss_mb"),
        ("module_peak_working_set_mb", "peak_working_set_mb"),
    ):
        values = manifest.get(field)
        if not isinstance(values, dict):
            raise ModuleRefreshError(f"Republished native manifest lacks {field}")
        for module, receipt in metrics.items():
            values[module] = float(receipt[metric_name])


def _rebind_run_manifest_sources(
    run_manifest: dict[str, Any],
    native_manifests: Mapping[str, Mapping[str, Any]],
    *,
    candidate_root: Path,
    publication_commit: str,
) -> None:
    """Bind root-level module receipts to the newly published native files."""

    sources = run_manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(DATABASES):
        raise ModuleRefreshError(
            "Run manifest sources must cover exactly all six databases"
        )
    for database in DATABASES:
        source_record = sources.get(database)
        native_manifest = native_manifests.get(database)
        if not isinstance(source_record, dict) or not isinstance(
            native_manifest, Mapping
        ):
            raise ModuleRefreshError(
                f"Cannot rebind run-manifest receipts for {database}"
            )
        metrics = source_record.get("module_metrics")
        files = native_manifest.get("files")
        if not isinstance(metrics, dict) or set(metrics) != set(MODULES):
            raise ModuleRefreshError(
                f"Run manifest module_metrics is incomplete for {database}"
            )
        if not isinstance(files, list) or not all(
            isinstance(entry, Mapping) for entry in files
        ):
            raise ModuleRefreshError(
                f"Native manifest file receipts are invalid for {database}"
            )
        files_by_module = {str(entry.get("module")): entry for entry in files}
        if set(files_by_module) != set(MODULES):
            raise ModuleRefreshError(
                f"Native manifest file receipts are incomplete for {database}"
            )
        timing_fields = {
            "elapsed_seconds": native_manifest.get("module_timings_seconds"),
            "peak_rss_mb": native_manifest.get("module_peak_rss_mb"),
            "peak_working_set_mb": native_manifest.get(
                "module_peak_working_set_mb"
            ),
        }
        if any(
            not isinstance(values, Mapping) or set(values) != set(MODULES)
            for values in timing_fields.values()
        ):
            raise ModuleRefreshError(
                f"Native manifest runtime receipts are incomplete for {database}"
            )
        for module in MODULES:
            metric = metrics[module]
            if not isinstance(metric, dict):
                raise ModuleRefreshError(
                    f"Run manifest metric is invalid for {database}/{module}"
                )
            entry = files_by_module[module]
            metric.update(
                {
                    "rows": entry.get("rows"),
                    "parquet_bytes": entry.get("parquet_bytes"),
                    "parquet_sha256": entry.get("parquet_sha256"),
                    **{
                        field: values[module]
                        for field, values in timing_fields.items()
                    },
                }
            )
        source_record["native_publication_easyicu_git_commit"] = publication_commit
        source_record["native_manifest_sha256"] = REPUBLICATION._sha256_file(
            candidate_root / "exports" / database / "_manifest.json"
        )
        source_record["total_rows"] = sum(
            int(files_by_module[module].get("rows") or 0) for module in MODULES
        )
        source_record["total_parquet_bytes"] = sum(
            int(files_by_module[module].get("parquet_bytes") or 0)
            for module in MODULES
        )


def refresh_candidate(
    *,
    source_run_root: Path,
    output_root: Path,
    modules: Sequence[str],
    data_path_overrides: Mapping[str, str],
    batch_size: int | None,
    databases: Sequence[str] = DATABASES,
    resume: bool = False,
    repair_finalized: bool = False,
) -> Path:
    source = source_run_root.resolve()
    destination = output_root.expanduser().absolute()
    if source == destination:
        raise ModuleRefreshError("Source and destination run roots must differ")
    selected_databases = _validate_databases(databases)
    if resume and set(selected_databases) != set(DATABASES):
        raise ModuleRefreshError(
            "Database-subset refreshes must use a fresh candidate; their "
            "interrupted state is not yet transaction-bound for safe resume"
        )
    requested_modules = _validate_modules(modules)
    selected_modules = _expand_module_dependency_closure(requested_modules)
    publication_commit = REPUBLICATION._require_clean_checkout()
    source_run_manifest = REPUBLICATION._validate_source(source)
    data_paths = _resolve_data_paths(
        source_run_manifest, data_path_overrides, selected_databases
    )
    source_run_manifest_sha256 = REPUBLICATION._sha256_file(
        source / "run_manifest.json"
    )
    source_receipts = {
        database: REPUBLICATION._source_database_receipt(source / "exports" / database)
        for database in DATABASES
    }
    source_easyicu_commit = _validate_sealed_source_receipts(
        source, source_receipts
    )
    source_refresh_path = source / "module_refresh_provenance.json"
    embedded_source_refresh = source_run_manifest.get("module_refresh")
    source_refresh_provenance: dict[str, Any] | None = None
    source_refresh_sha256: str | None = None
    source_refresh_scope = {database: [] for database in DATABASES}
    if source_refresh_path.exists() or source_refresh_path.is_symlink():
        _require_regular_file(
            source_refresh_path, label="source module-refresh provenance"
        )
        source_refresh_provenance = _read_json(
            source_refresh_path, label="source module-refresh provenance"
        )
        if embedded_source_refresh != source_refresh_provenance:
            raise ModuleRefreshError(
                "Source run manifest does not embed the exact module-refresh "
                "provenance file"
            )
        source_refresh_scope = _database_refresh_scope(
            source_refresh_provenance, label="source module-refresh provenance"
        )
        if (
            source_refresh_provenance.get("publication_easyicu_git_commit")
            != source_easyicu_commit
        ):
            raise ModuleRefreshError(
                "Source module-refresh commit disagrees with the sealed six-"
                "database native manifests"
            )
        source_refresh_sha256 = REPUBLICATION._sha256_file(source_refresh_path)
    elif embedded_source_refresh is not None:
        raise ModuleRefreshError(
            "Source run manifest embeds module-refresh provenance but the "
            "regular provenance file is missing"
        )

    prior_provenance: dict[str, Any] | None = None
    if repair_finalized and not resume:
        raise ModuleRefreshError("--repair-finalized requires --resume")
    if resume:
        if destination.is_symlink() or not destination.is_dir():
            raise ModuleRefreshError(
                f"--resume requires an existing regular candidate directory: {destination}"
            )
        provenance_path = destination / "module_refresh_provenance.json"
        sealed_path = destination / "run_metadata.json"
        if repair_finalized:
            if sealed_path.exists() or sealed_path.is_symlink():
                raise ModuleRefreshError(
                    "--repair-finalized refuses a sealed candidate"
                )
            prior_provenance = _read_json(
                provenance_path,
                label="existing module-refresh provenance",
            )
            if prior_provenance.get("schema_version") != SCHEMA_VERSION:
                raise ModuleRefreshError(
                    "--repair-finalized requires the current selected-module "
                    "refresh provenance schema"
                )
            if not prior_provenance.get("raw_database_reread"):
                raise ModuleRefreshError(
                    "--repair-finalized requires a prior raw-data refresh"
                )
            if (
                Path(str(prior_provenance.get("source_run_root", ""))).resolve()
                != source
            ):
                raise ModuleRefreshError(
                    "--repair-finalized source run differs from the prior refresh"
                )
            if (
                prior_provenance.get("source_run_manifest_sha256")
                != source_run_manifest_sha256
            ):
                raise ModuleRefreshError(
                    "--repair-finalized source manifest hash changed"
                )
        elif provenance_path.exists() or sealed_path.exists():
            raise ModuleRefreshError(
                "--resume refuses a sealed or already-finalized refresh candidate"
            )
    else:
        REPUBLICATION._clone_source(source, destination)
        (destination / "run_metadata.json").unlink(missing_ok=True)
        (destination / "module_extraction_timing.csv").unlink(missing_ok=True)
        (destination / "republication_provenance.json").unlink(missing_ok=True)
        (destination / "module_refresh_provenance.json").unlink(missing_ok=True)
        shutil.rmtree(destination / ".module_refresh_staging", ignore_errors=True)
        shutil.rmtree(destination / "publication_qc", ignore_errors=True)

    refreshed: dict[str, dict[str, Any]] = {}
    try:
        for database in selected_databases:
            refreshed[database] = _refresh_one_database(
                database=database,
                data_path=data_paths[database],
                source_database_root=source / "exports" / database,
                candidate_root=destination,
                modules=selected_modules,
                batch_size=batch_size,
                reuse_completed_export=resume and not repair_finalized,
            )

        lineage_base = prior_provenance or source_refresh_provenance
        base_scope = source_refresh_scope
        if prior_provenance is not None:
            base_scope = _database_refresh_scope(
                prior_provenance, label="prior module-refresh provenance"
            )
        inherited_refreshed_modules = [
            module
            for module in MODULES
            if any(module in base_scope[database] for database in DATABASES)
        ]
        inherited_requested_modules: list[str] = []
        if lineage_base is not None:
            inherited_requested_modules = list(
                _canonical_provenance_modules(
                    lineage_base.get("requested_modules"),
                    label="inherited module-refresh requested_modules",
                )
            )
        per_database_refreshed_modules = {}
        for database in DATABASES:
            current_modules = (
                set(selected_modules) if database in selected_databases else set()
            )
            inherited_modules = set(base_scope[database])
            per_database_refreshed_modules[database] = [
                module
                for module in MODULES
                if module in inherited_modules or module in current_modules
            ]
        all_refreshed_modules = [
            module
            for module in MODULES
            if any(
                module in per_database_refreshed_modules[database]
                for database in DATABASES
            )
        ]
        all_requested_modules = [
            module
            for module in MODULES
            if module in set(inherited_requested_modules)
            or module in set(requested_modules)
        ]
        prior_runtime = (
            lineage_base.get("per_database_runtime")
            if lineage_base is not None
            else {}
        )
        if not isinstance(prior_runtime, dict):
            raise ModuleRefreshError(
                "Inherited module-refresh runtime must be a JSON object"
            )
        combined_runtime = copy.deepcopy(prior_runtime)
        for database in DATABASES:
            if not base_scope[database]:
                continue
            database_runtime = combined_runtime.get(database)
            runtime_modules = (
                database_runtime.get("modules")
                if isinstance(database_runtime, dict)
                else None
            )
            if not isinstance(runtime_modules, dict) or not set(
                base_scope[database]
            ).issubset(runtime_modules):
                raise ModuleRefreshError(
                    "Inherited module-refresh runtime is incomplete for "
                    f"{database}"
                )
        for database in selected_databases:
            previous = copy.deepcopy(combined_runtime.get(database) or {})
            current = refreshed[database]
            previous_modules = dict(previous.get("modules") or {})
            previous_modules.update(current.get("modules") or {})
            previous.update(
                {
                    "database": database,
                    "data_path": current.get("data_path"),
                    "num_patients": (
                        current.get("num_patients")
                        if current.get("num_patients") is not None
                        else previous.get("num_patients")
                    ),
                    "batch_size": (
                        current.get("batch_size")
                        if current.get("batch_size") is not None
                        else previous.get("batch_size")
                    ),
                    "modules": previous_modules,
                    "latest_refresh_elapsed_seconds": current.get(
                        "total_elapsed_seconds"
                    ),
                }
            )
            combined_runtime[database] = previous
        prior_paths = (
            lineage_base.get("raw_data_paths")
            if lineage_base is not None
            else {}
        )
        if not isinstance(prior_paths, dict):
            raise ModuleRefreshError(
                "Inherited module-refresh raw_data_paths must be a JSON object"
            )
        combined_data_paths = {**prior_paths, **data_paths}
        cumulative_raw_refreshed_databases = [
            database
            for database in DATABASES
            if per_database_refreshed_modules[database]
        ]
        parent_refresh_receipt: dict[str, Any] | None = None
        if source_refresh_provenance is not None:
            parent_refresh_receipt = {
                "path": str(source_refresh_path),
                "sha256": source_refresh_sha256,
                "schema_version": source_refresh_provenance.get("schema_version"),
                "publication_easyicu_git_commit": source_refresh_provenance.get(
                    "publication_easyicu_git_commit"
                ),
            }
        elif prior_provenance is not None:
            inherited_parent = prior_provenance.get(
                "parent_module_refresh_provenance"
            )
            if isinstance(inherited_parent, dict):
                parent_refresh_receipt = copy.deepcopy(inherited_parent)
        repair_history: list[dict[str, Any]] = []
        if prior_provenance is not None:
            repair_history = list(prior_provenance.get("repair_history") or [])
            repair_history.append(
                {
                    "repaired_at": _utc_now(),
                    "prior_publication_easyicu_git_commit": prior_provenance.get(
                        "publication_easyicu_git_commit"
                    ),
                    "publication_easyicu_git_commit": publication_commit,
                    "selected_databases": list(selected_databases),
                    "requested_modules": list(requested_modules),
                    "dependency_closure_applied": list(selected_modules),
                    "reason": "explicit_unsealed_finalized_candidate_repair",
                }
            )

        native_manifests: dict[str, dict[str, Any]] = {}
        for database in DATABASES:
            manifest = REPUBLICATION._republish_database(
                destination / "exports" / database,
                database=database,
                source_receipt=source_receipts[database],
                publication_commit=publication_commit,
            )
            reconstructed_metrics = _reconstruct_database_runtime_metrics(
                source_run_manifest,
                database=database,
                cumulative_refresh_runtime=combined_runtime.get(database),
            )
            _rebind_manifest_metrics(manifest, reconstructed_metrics)
            if database in selected_databases:
                manifest["source_extraction_provenance"] = {
                    **(manifest.get("source_extraction_provenance") or {}),
                    "publication_only": False,
                    "raw_database_reread": True,
                    "refreshed_modules": list(selected_modules),
                    "current_refreshed_modules": list(selected_modules),
                    "inherited_refreshed_modules": list(base_scope[database]),
                    "cumulative_refreshed_modules": list(
                        per_database_refreshed_modules[database]
                    ),
                    "reused_modules": [
                        module
                        for module in MODULES
                        if module not in selected_modules
                    ],
                    "latest_module_refresh_runtime": refreshed[database],
                    "cumulative_module_refresh_runtime": combined_runtime[database],
                    "transformation": (
                        "raw re-extraction of selected modules plus canonical "
                        "native-v2 republication of the complete package"
                    ),
                }
            else:
                manifest["source_extraction_provenance"] = {
                    **(manifest.get("source_extraction_provenance") or {}),
                    "publication_only": True,
                    "raw_database_reread": False,
                    "refreshed_modules": [],
                    "current_refreshed_modules": [],
                    "inherited_refreshed_modules": list(base_scope[database]),
                    "cumulative_refreshed_modules": list(
                        per_database_refreshed_modules[database]
                    ),
                    "reused_modules": list(MODULES),
                    "selected_module_refresh_scope": (
                        "publication_only_for_six_database_commit_harmonization"
                    ),
                    "transformation": (
                        "canonical native-v2 republication without raw-data reread; "
                        "logical contents must match the source release"
                    ),
                }
            REPUBLICATION._atomic_write_json(
                destination / "exports" / database / "_manifest.json", manifest
            )
            native_manifests[database] = manifest

        publication_only_semantic_audit = {
            database: _validate_publication_only_database_semantics(
                source / "exports" / database,
                destination / "exports" / database,
            )
            for database in DATABASES
            if database not in selected_databases
        }
        REPUBLICATION._rebind_extraction_timing_receipts(
            destination / "database_extraction_timing.csv", native_manifests
        )
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "source_run_root": str(source),
            "source_run_manifest_sha256": source_run_manifest_sha256,
            "source_database_receipts": source_receipts,
            "publication_easyicu_git_commit": publication_commit,
            "publication_easyicu_git_dirty": False,
            "refresher": str(Path(__file__).relative_to(REPOSITORY_ROOT)),
            "refreshed_modules": list(all_refreshed_modules),
            "requested_modules": list(all_requested_modules),
            "dependency_closure_applied": list(all_refreshed_modules),
            "latest_requested_modules": list(requested_modules),
            "latest_dependency_closure_applied": list(selected_modules),
            "inherited_requested_modules": inherited_requested_modules,
            "inherited_refreshed_modules": inherited_refreshed_modules,
            "selected_databases": list(selected_databases),
            "latest_refresh_databases": list(selected_databases),
            "cumulative_raw_refreshed_databases": (
                cumulative_raw_refreshed_databases
            ),
            "per_database_refreshed_modules": per_database_refreshed_modules,
            "raw_database_reread": True,
            "raw_database_reread_scope": (
                "all_six_databases"
                if set(selected_databases) == set(DATABASES)
                else "selected_databases_only"
            ),
            "cumulative_raw_database_reread_scope": (
                "all_six_databases"
                if set(cumulative_raw_refreshed_databases) == set(DATABASES)
                else "database_subset_across_lineage"
            ),
            "raw_data_paths": combined_data_paths,
            "per_database_runtime": combined_runtime,
            "publication_only_semantic_audit": publication_only_semantic_audit,
            "reused_module_count_per_database": {
                database: len(MODULES)
                - (len(selected_modules) if database in selected_databases else 0)
                for database in DATABASES
            },
            "cumulative_reused_module_count_per_database": {
                database: len(MODULES)
                - len(per_database_refreshed_modules[database])
                for database in DATABASES
            },
            "seal_required_after_refresh": True,
        }
        if parent_refresh_receipt is not None:
            provenance["parent_module_refresh_provenance"] = parent_refresh_receipt
        if repair_history:
            provenance["repair_history"] = repair_history
        REPUBLICATION._atomic_write_json(
            destination / "module_refresh_provenance.json", provenance
        )

        updated_run_manifest = copy.deepcopy(source_run_manifest)
        updated_run_manifest["module_refresh"] = provenance
        updated_run_manifest["updated_at"] = provenance["created_at"]
        updated_run_manifest["publication_checkout"] = {
            "easyicu_git_commit": publication_commit,
            "easyicu_git_dirty": False,
            "scope": "selected_database_module_refresh",
            "selected_databases": list(selected_databases),
        }
        _rebind_run_manifest_sources(
            updated_run_manifest,
            native_manifests,
            candidate_root=destination,
            publication_commit=publication_commit,
        )
        REPUBLICATION._atomic_write_json(
            destination / "run_manifest.json", updated_run_manifest
        )
    except Exception:
        # The unsealed candidate is intentionally retained for diagnosis.  It
        # can neither replace a source package nor be promoted without EX-A01.
        raise
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an unsealed candidate after recovering canonical staged modules.",
    )
    parser.add_argument(
        "--repair-finalized",
        action="store_true",
        help=(
            "With --resume, force the requested modules to be re-read in an "
            "unsealed finalized candidate and merge their receipts into its "
            "existing refresh provenance. Sealed candidates remain immutable."
        ),
    )
    parser.add_argument(
        "--database",
        action="append",
        default=[],
        choices=DATABASES,
        help=(
            "Database whose raw modules will be refreshed; repeatable. "
            "Defaults to all six databases for backward compatibility."
        ),
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help=(
            "Raw-derived module to refresh (outcome, renal, respiratory or "
            "sofa2_score); repeatable."
        ),
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        metavar="DATABASE=PATH",
        help="Optional raw-data override; otherwise reuses the source run receipt.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Fixed per-database stay batch; default uses adaptive stream planning.",
    )
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        candidate = refresh_candidate(
            source_run_root=args.source_run_root,
            output_root=args.output_root,
            modules=args.module,
            data_path_overrides=_parse_data_path_overrides(args.data_path),
            batch_size=args.batch_size,
            databases=args.database or DATABASES,
            resume=args.resume,
            repair_finalized=args.repair_finalized,
        )
    except (ModuleRefreshError, OSError, ValueError) as exc:
        print(f"selected-module refresh failed: {exc}", file=sys.stderr)
        return 1
    print(f"Refreshed selected-module candidate: {candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
