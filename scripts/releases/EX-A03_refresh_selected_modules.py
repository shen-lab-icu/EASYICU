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
``sofa2_score`` and the two Sepsis-SOFA labels that consume those scores. The
sealed shared-infection timeline is staged as a hash-verified read-only
dependency rather than reread from raw data. This is not a generic way to
bypass the full extraction controller.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
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
from easyicu.api.extraction import (  # noqa: E402
    EXTRACT_MODULES,
    _publish_native_export_v2,
    _resource_budget_execution_limits,
    plan_extraction_resources,
    plan_module_extraction_resources,
)


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
    {"outcome", "renal", "respiratory", "sofa1_score", "sofa2_score"}
)
MODULE_DEPENDENCY_CLOSURE: dict[str, tuple[str, ...]] = {
    "outcome": ("outcome",),
    "renal": ("renal",),
    "respiratory": (
        "respiratory",
        "sofa1_score",
        "sofa2_score",
        "sepsis3_sofa1",
        "sepsis3_sofa2",
    ),
    "sofa1_score": (
        "sofa1_score",
        "sepsis3_sofa1",
    ),
    # Targeted repair path for owner-issued SOFA-2 receipt companions. The
    # parent candidate must already contain a raw-refreshed respiratory input;
    # downstream M1 validates that parent provenance before using the child.
    "sofa2_score": (
        "sofa2_score",
        "sepsis3_sofa2",
    ),
}
SCHEMA_VERSION = "easyicu_full6_selected_module_refresh_v2"
LEGACY_SCHEMA_VERSION = "easyicu_full6_selected_module_refresh_v1"
RESOURCE_PLAN_SCHEMA_VERSION = "easyicu_selected_module_resource_plan_v1"
RESOURCE_BENCHMARK_SCHEMA_VERSION = "easyicu_selected_module_resource_benchmark_v1"
RESOURCE_BENCHMARK_FILENAME = "resource_benchmark_provenance.json"
DEFAULT_RELEASE_MEMORY_BUDGET_MB = 8 * 1024
FORMALLY_MEASURED_RESOURCE_REASONS = frozenset(
    {
        "measured_profile_fast_path",
        "measured_profile_fastest_safe_batch",
    }
)


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


def _parse_database_module_scopes(
    values: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Parse repeatable ``DATABASE=MODULE[,MODULE]`` refresh scopes."""

    parsed: dict[str, tuple[str, ...]] = {}
    for raw in values:
        database, separator, module_text = raw.partition("=")
        database = database.strip()
        modules = tuple(
            module.strip() for module in module_text.split(",") if module.strip()
        )
        if not separator or database not in DATABASES or not modules:
            raise ModuleRefreshError(
                "--database-module must be DATABASE=MODULE[,MODULE] for one of "
                f"{', '.join(DATABASES)}; got {raw!r}"
            )
        if database in parsed:
            raise ModuleRefreshError(
                f"Duplicate --database-module scope: {database}"
            )
        parsed[database] = _validate_modules(modules)
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
    databases: Sequence[str] | None = None,
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
            "renal, respiratory, sofa1_score and sofa2_score; "
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


def _resolve_database_module_scope(
    *,
    modules: Sequence[str],
    databases: Sequence[str],
    database_module_scope: Mapping[str, Sequence[str]] | None = None,
) -> tuple[
    tuple[str, ...],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    """Resolve requested and dependency-closed modules for each database."""

    if database_module_scope:
        if modules or databases:
            raise ModuleRefreshError(
                "Per-database module scope cannot be combined with global "
                "modules/databases"
            )
        selected_databases = _validate_databases(tuple(database_module_scope))
        requested_by_database = {
            database: _validate_modules(database_module_scope[database])
            for database in selected_databases
        }
    else:
        selected_databases = _validate_databases(databases)
        requested = _validate_modules(modules)
        requested_by_database = {
            database: requested for database in selected_databases
        }
    closed_by_database = {
        database: _expand_module_dependency_closure(requested_by_database[database])
        for database in selected_databases
    }
    return selected_databases, requested_by_database, closed_by_database


def _source_cohort_size(
    source_run_manifest: Mapping[str, Any], database: str
) -> int:
    """Read the sealed stay count without opening any raw database table."""

    sources = source_run_manifest.get("sources")
    source = sources.get(database) if isinstance(sources, Mapping) else None
    metrics = source.get("module_metrics") if isinstance(source, Mapping) else None
    outcome = metrics.get("outcome") if isinstance(metrics, Mapping) else None
    rows = outcome.get("rows") if isinstance(outcome, Mapping) else None
    try:
        cohort_size = int(rows)
    except (TypeError, ValueError) as exc:
        raise ModuleRefreshError(
            f"Source run lacks a valid outcome stay count for {database}"
        ) from exc
    if cohort_size <= 0:
        raise ModuleRefreshError(
            f"Source run has a non-positive outcome stay count for {database}"
        )
    return cohort_size


def _build_refresh_resource_plan(
    source_run_manifest: Mapping[str, Any],
    *,
    requested_modules: Sequence[str],
    databases: Sequence[str],
    memory_budget_mb: float,
    requested_batch_size: int | None = None,
    database_module_scope: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Build a read-only database-by-module release execution plan."""

    selected_databases, requested_by_database, closed_by_database = (
        _resolve_database_module_scope(
            modules=requested_modules,
            databases=databases,
            database_module_scope=database_module_scope,
        )
    )
    requested = tuple(
        module
        for module in MODULES
        if any(
            module in requested_by_database[database]
            for database in selected_databases
        )
    )
    closure = tuple(
        module
        for module in MODULES
        if any(
            module in closed_by_database[database]
            for database in selected_databases
        )
    )
    budget = float(memory_budget_mb)
    if budget <= 0:
        raise ModuleRefreshError("memory budget must be positive")

    database_plans: dict[str, Any] = {}
    for database in selected_databases:
        database_closure = closed_by_database[database]
        cohort_size = _source_cohort_size(source_run_manifest, database)
        aggregate = plan_extraction_resources(
            database,
            database_closure,
            cohort_size,
            requested_batch_size,
            available_memory_mb=budget,
        )
        module_plans = plan_module_extraction_resources(
            database,
            database_closure,
            cohort_size,
            requested_batch_size,
            available_memory_mb=budget,
        )
        database_plans[database] = {
            "cohort_stays": cohort_size,
            "requested_modules": list(requested_by_database[database]),
            "dependency_closure": list(database_closure),
            "aggregate_request_plan": aggregate.to_dict(),
            "modules": {
                module: {
                    **plan.to_dict(),
                    "planned_batches": math.ceil(cohort_size / plan.batch_size),
                }
                for module, plan in module_plans.items()
            },
        }
    unmeasured_modules = {
        database: [
            module
            for module, module_plan in database_plans[database]["modules"].items()
            if module_plan["reason_code"] not in FORMALLY_MEASURED_RESOURCE_REASONS
        ]
        for database in selected_databases
    }
    unmeasured_modules = {
        database: modules
        for database, modules in unmeasured_modules.items()
        if modules
    }
    return {
        "schema_version": RESOURCE_PLAN_SCHEMA_VERSION,
        "read_only": True,
        "raw_database_reread": False,
        "memory_budget_mb": budget,
        "resource_execution_limits": _resource_budget_execution_limits(budget),
        "requested_modules": list(requested),
        "dependency_closure": list(closure),
        "per_database_requested_modules": {
            database: list(requested_by_database[database])
            for database in selected_databases
        },
        "per_database_dependency_closure": {
            database: list(closed_by_database[database])
            for database in selected_databases
        },
        "selected_databases": list(selected_databases),
        "explicit_batch_override": requested_batch_size,
        "formal_release_admissible": not unmeasured_modules,
        "unmeasured_or_overridden_modules": unmeasured_modules,
        "databases": database_plans,
    }


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


def _module_is_complete_producer_staging(
    database_root: Path,
    modules: Sequence[str],
) -> bool:
    """Recognise complete pre-native module outputs without assuming native IDs."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - package is release-required
        raise ModuleRefreshError(
            "pyarrow is required to inspect producer staging"
        ) from exc

    for module in modules:
        manifest_path = database_root / f"{module}.manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return False
        try:
            manifest = _read_json(
                manifest_path,
                label=f"{module} staged producer manifest",
            )
        except ModuleRefreshError:
            return False
        if manifest.get("module") != module or manifest.get("errors"):
            return False
        saved = manifest.get("saved")
        if not isinstance(saved, Mapping):
            return False
        # An explicitly empty saved mapping is a complete structural absence;
        # native-v2 owns creation of its typed zero-row placeholder.
        if not saved:
            continue
        parquet = database_root / f"{module}.parquet"
        if parquet.is_symlink() or not parquet.is_file():
            return False
        try:
            columns = set(pq.read_schema(parquet).names)
        except Exception:
            return False
        produced: set[str] = set()
        for saved_name, record in saved.items():
            if isinstance(saved_name, str):
                produced.add(saved_name)
            if isinstance(record, Mapping):
                produced.update(
                    str(concept)
                    for concept in (record.get("concepts") or [])
                    if isinstance(concept, str)
                )
        declared = set(EXTRACT_MODULES[module])
        if not declared.intersection(produced) or not declared.intersection(columns):
            return False
    return True


def _requires_outcome_time_bounds(modules: Sequence[str]) -> bool:
    """Return whether native publication needs the sealed ICU-stay bounds."""

    return any(module not in {"demographics", "outcome"} for module in modules)


def _stage_outcome_time_bound_dependency(
    *,
    source_database_root: Path,
    staging_root: Path,
    modules: Sequence[str],
) -> Path | None:
    """Expose sealed ``outcome.los_icu`` to a selected-module publisher.

    ``outcome`` is an input authority here, not a refreshed output. Copy the
    already-published Parquet rather than rereading raw outcome data or sharing
    its inode with writable staging. The native publisher only reads this file
    because ``outcome`` is absent from ``modules``; SHA-256 is checked before
    use.
    """

    if not _requires_outcome_time_bounds(modules) or "outcome" in modules:
        return None

    source = source_database_root / "outcome.parquet"
    destination = staging_root / "outcome.parquet"
    _require_regular_file(source, label="sealed outcome time-bound dependency")
    staging_root.mkdir(mode=0o700, parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        _require_regular_file(
            destination,
            label="staged outcome time-bound dependency",
        )
        if REPUBLICATION._sha256_file(destination) != REPUBLICATION._sha256_file(
            source
        ):
            raise ModuleRefreshError(
                "Staged outcome time-bound dependency differs from the sealed source"
            )
        return destination

    shutil.copy2(source, destination)
    if REPUBLICATION._sha256_file(destination) != REPUBLICATION._sha256_file(source):
        destination.unlink(missing_ok=True)
        raise ModuleRefreshError(
            "Failed to stage an exact outcome time-bound dependency"
        )
    return destination


def _stage_sepsis_shared_dependency(
    *,
    source_database_root: Path,
    staging_root: Path,
    modules: Sequence[str],
) -> Path | None:
    """Stage sealed suspected-infection evidence without refreshing it.

    Sepsis-3 is a derived consumer of both a refreshed SOFA trajectory and the
    existing `sepsis_shared` timeline. The latter is independent of IMV and of
    the native SOFA row-grain repair, so rereading its raw sources would expand
    the mutation scope without adding information. Copy and hash-check only its
    Parquet dependency; it is excluded from the publisher's module list and is
    removed with staging after the derived labels are published.
    """

    needs_sepsis = any(
        module in {"sepsis3_sofa1", "sepsis3_sofa2"} for module in modules
    )
    if not needs_sepsis or "sepsis_shared" in modules:
        return None

    source = source_database_root / "sepsis_shared.parquet"
    destination = staging_root / "sepsis_shared.parquet"
    _require_regular_file(source, label="sealed sepsis_shared dependency")
    staging_root.mkdir(mode=0o700, parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        _require_regular_file(destination, label="staged sepsis_shared dependency")
        if REPUBLICATION._sha256_file(destination) != REPUBLICATION._sha256_file(
            source
        ):
            raise ModuleRefreshError(
                "Staged sepsis_shared dependency differs from the sealed source"
            )
        return destination

    shutil.copy2(source, destination)
    if REPUBLICATION._sha256_file(destination) != REPUBLICATION._sha256_file(source):
        destination.unlink(missing_ok=True)
        raise ModuleRefreshError(
            "Failed to stage an exact sepsis_shared dependency"
        )
    return destination


def _stage_refresh_read_dependencies(
    *,
    source_database_root: Path,
    staging_root: Path,
    modules: Sequence[str],
) -> dict[str, str]:
    """Stage and receipt every sealed module used only as a read dependency."""

    staged = {}
    for name, path in (
        (
            "outcome",
            _stage_outcome_time_bound_dependency(
                source_database_root=source_database_root,
                staging_root=staging_root,
                modules=modules,
            ),
        ),
        (
            "sepsis_shared",
            _stage_sepsis_shared_dependency(
                source_database_root=source_database_root,
                staging_root=staging_root,
                modules=modules,
            ),
        ),
    ):
        if path is not None:
            staged[name] = REPUBLICATION._sha256_file(path)
    return staged


def _recover_native_staging_publication(
    *,
    database: str,
    data_path: str,
    staging_root: Path,
    modules: Sequence[str],
    resource_budget_mb: float,
) -> None:
    """Finish native-v2 publication from complete producer artifacts.

    A process can finish every expensive module and then fail in the publisher.
    Reconstruct only the small in-memory receipt needed by native-v2; never
    reopen the raw database or recompute a module during recovery.
    """

    module_results: dict[str, dict[str, Any]] = {}
    module_resource_plans: dict[str, Any] = {}
    for module in modules:
        manifest = _read_json(
            staging_root / f"{module}.manifest.json",
            label=f"{module} staged producer manifest",
        )
        errors = list(manifest.get("errors") or [])
        if errors:
            raise ModuleRefreshError(
                f"Cannot recover {module} native publication: {errors}"
            )
        module_results[module] = {
            "errors": [],
            "elapsed": float(manifest.get("elapsed_sec") or 0.0),
            "peak_rss_mb": float(manifest.get("peak_rss_mb") or 0.0),
            "peak_working_set_mb": float(
                manifest.get("peak_working_set_mb") or 0.0
            ),
        }
        if manifest.get("resource_plan") is not None:
            module_resource_plans[module] = manifest["resource_plan"]

    _publish_native_export_v2(
        database=database,
        data_path=data_path,
        output_dir=str(staging_root),
        modules=list(modules),
        max_patients=None,
        result={
            "modules": module_results,
            "stream_retry_history": [],
            "resource_plan": None,
            "module_resource_plans": module_resource_plans,
            "resource_budget_mb": float(resource_budget_mb),
            "resource_execution_limits": _resource_budget_execution_limits(
                resource_budget_mb
            ),
        },
        require_stay_time_bounds=True,
    )


def _validate_existing_staging_native_manifest(
    *,
    database: str,
    staging_root: Path,
    modules: Sequence[str],
) -> None:
    """Reject a stale or unrelated root manifest before staged promotion."""

    manifest_path = staging_root / "_manifest.json"
    _require_regular_file(manifest_path, label="staged native-v2 manifest")
    manifest = _read_json(manifest_path, label="staged native-v2 manifest")
    if manifest.get("schema_version") != "easyicu_native_export_v2":
        raise ModuleRefreshError("Staged refresh lacks a native-v2 root manifest")
    if manifest.get("database") != database:
        raise ModuleRefreshError(
            "Staged native-v2 manifest belongs to a different database"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not all(
        isinstance(entry, Mapping) for entry in files
    ):
        raise ModuleRefreshError("Staged native-v2 file receipts are invalid")
    files_by_module = {str(entry.get("module")): entry for entry in files}
    if set(files_by_module) != set(modules):
        raise ModuleRefreshError(
            "Staged native-v2 module scope differs from the requested refresh"
        )
    if _requires_outcome_time_bounds(modules):
        authority = manifest.get("time_window_authority")
        try:
            bounded_stays = int(authority.get("bounded_stays") or 0)
        except (AttributeError, TypeError, ValueError):
            bounded_stays = 0
        if not (
            isinstance(authority, Mapping)
            and authority.get("required") is True
            and authority.get("source") == "outcome.los_icu"
            and bounded_stays > 0
        ):
            raise ModuleRefreshError(
                "Staged longitudinal modules lack outcome.los_icu time authority"
            )
    for module in modules:
        parquet = staging_root / f"{module}.parquet"
        _require_regular_file(parquet, label=f"{module} staged native Parquet")
        expected_sha256 = files_by_module[module].get("parquet_sha256")
        if not isinstance(expected_sha256, str) or (
            REPUBLICATION._sha256_file(parquet) != expected_sha256
        ):
            raise ModuleRefreshError(
                f"{module} staged Parquet differs from its native-v2 receipt"
            )


def _validate_refreshed_score_content(
    database_root: Path,
    modules: Sequence[str],
    *,
    database: str,
) -> None:
    """Refuse null or internally incoherent refreshed SOFA trajectories.

    Schema checks alone did not catch the 2026-09 IMV regression: both SOFA
    files had the expected columns and millions of rows, but interval handling
    had displaced their components so every total score was missing. A later
    audit also found totals aggregated independently from duplicate-hour organ
    components. Scan bounded Arrow batches before any candidate file is
    promoted and require the public score/receipt identities exactly.
    """

    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - package is release-required
        raise ModuleRefreshError(
            "pyarrow is required to validate refreshed score content"
        ) from exc

    score_specs = {
        "sofa1_score": {
            "score": "sofa",
            "components": (
                "sofa_resp",
                "sofa_coag",
                "sofa_liver",
                "sofa_cardio",
                "sofa_cns",
                "sofa_renal",
            ),
        },
        "sofa2_score": {
            "score": "sofa2",
            "components": (
                "sofa2_resp",
                "sofa2_coag",
                "sofa2_liver",
                "sofa2_cardio",
                "sofa2_cns",
                "sofa2_renal",
            ),
        },
    }
    for module, spec in score_specs.items():
        if module not in modules:
            continue
        score_column = str(spec["score"])
        components = tuple(spec["components"])
        path = database_root / f"{module}.parquet"
        _require_regular_file(path, label=f"{module} staged Parquet")
        parquet = pq.ParquetFile(path)
        if score_column not in parquet.schema_arrow.names:
            raise ModuleRefreshError(
                f"{module} staged Parquet lacks primary score {score_column!r}"
            )
        required = {score_column, *components}
        if module == "sofa2_score":
            required.update(
                f"{component}_{suffix}"
                for component in components
                for suffix in ("observed", "available")
            )
            required.update({"sofa2_observed", "sofa2_available"})
        missing = required.difference(parquet.schema_arrow.names)
        if missing:
            raise ModuleRefreshError(
                f"{module} refresh lacks score-consistency columns: {sorted(missing)}"
            )

        rows = 0
        non_null = 0
        inconsistent = 0
        invalid_component = 0
        component_non_null = {component: 0 for component in components}
        availability_true = {component: 0 for component in components}
        consistency_columns = [score_column, *components]
        if module == "sofa2_score":
            consistency_columns.extend(
                f"{component}_{suffix}"
                for component in components
                for suffix in ("observed", "available")
            )
            consistency_columns.extend(["sofa2_observed", "sofa2_available"])
        for batch in parquet.iter_batches(
            batch_size=262_144,
            columns=consistency_columns,
            use_threads=False,
        ):
            frame = batch.to_pandas()
            rows += len(frame)
            component_frame = frame[list(components)].apply(
                lambda column: pd.to_numeric(column, errors="coerce")
            )
            for component in components:
                component_non_null[component] += int(
                    component_frame[component].notna().sum()
                )
            invalid_component += int(
                ((component_frame < 0) | (component_frame > 4)).any(axis=1).sum()
            )
            score = pd.to_numeric(frame[score_column], errors="coerce")
            non_null += int(score.notna().sum())
            if module == "sofa1_score":
                expected = component_frame.sum(axis=1, skipna=True)
                coherent = score.eq(expected)
            else:
                available_columns = [
                    f"{component}_available" for component in components
                ]
                observed_columns = [
                    f"{component}_observed" for component in components
                ]
                available = (
                    frame[available_columns]
                    .astype("boolean")
                    .fillna(False)
                )
                for component, receipt in zip(components, available_columns):
                    availability_true[component] += int(available[receipt].sum())
                observed = (
                    frame[observed_columns]
                    .astype("boolean")
                    .fillna(False)
                )
                complete = component_frame.notna().all(axis=1) & available.all(
                    axis=1
                )
                expected = component_frame.sum(
                    axis=1, min_count=len(components)
                ).where(complete)
                coherent = score.eq(expected) | (score.isna() & expected.isna())
                coherent &= (
                    frame["sofa2_available"]
                    .astype("boolean")
                    .fillna(False)
                    .eq(complete)
                )
                coherent &= (
                    frame["sofa2_observed"]
                    .astype("boolean")
                    .fillna(False)
                    .eq(complete & observed.all(axis=1))
                )
            inconsistent += int((~coherent.fillna(False)).sum())

        structurally_unavailable_sic_sofa2 = (
            module == "sofa2_score"
            and database == "sic"
            and rows > 0
            and non_null == 0
            and all(count > 0 for count in component_non_null.values())
            and availability_true["sofa2_cns"] == 0
            and all(
                availability_true[component] > 0
                for component in components
                if component != "sofa2_cns"
            )
        )
        if rows == 0 or (non_null == 0 and not structurally_unavailable_sic_sofa2):
            raise ModuleRefreshError(
                f"{module} refresh is unusable: {score_column} has "
                f"{non_null} non-null values across {rows} rows"
            )
        if invalid_component:
            raise ModuleRefreshError(
                f"{module} refresh has {invalid_component} rows with organ "
                "components outside the valid 0-4 range"
            )
        if inconsistent:
            raise ModuleRefreshError(
                f"{module} refresh has {inconsistent} rows whose total/receipts "
                "do not match the post-consolidation organ components"
            )


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _parquet_schema(connection: Any, path: Path) -> list[tuple[str, str]]:
    """Return the DuckDB-visible physical schema for one Parquet."""

    escaped_path = str(path.resolve()).replace("'", "''")
    relation = f"read_parquet('{escaped_path}')"
    described = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    schema = [(str(row[0]), str(row[1])) for row in described]
    if not schema:
        raise ModuleRefreshError(f"Cannot audit a zero-column Parquet: {path}")
    return schema


def _parquet_multiset_receipt(
    connection: Any,
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return an order-independent logical-content receipt for one Parquet."""

    escaped_path = str(path.resolve()).replace("'", "''")
    relation = f"read_parquet('{escaped_path}')"
    full_schema = _parquet_schema(connection, path)
    schema_by_name = dict(full_schema)
    if columns is None:
        schema = full_schema
    else:
        missing = [column for column in columns if column not in schema_by_name]
        if missing:
            raise ModuleRefreshError(
                f"Cannot audit missing Parquet columns {missing}: {path}"
            )
        schema = [(column, schema_by_name[column]) for column in columns]
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
    A newer publisher may materialize a catalog-declared unavailable concept
    as a typed all-null column. Such schema completion is accepted only when
    every source column retains its type and multiset of values, the added
    column is declared unavailable in the candidate manifest, and it contains
    no non-null value. DuckDB is bounded to one thread and 1 GiB with spill
    beside the candidate.
    """

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - release dependency
        raise ModuleRefreshError(
            "duckdb is required for publication-only semantic validation"
        ) from exc

    receipts: dict[str, dict[str, Any]] = {}
    candidate_manifest = _read_json(
        candidate_database_root / "_manifest.json",
        label=f"{candidate_database_root.name} candidate native manifest",
    )
    raw_unavailable = candidate_manifest.get("unavailable_concepts") or []
    if not isinstance(raw_unavailable, list):
        raise ModuleRefreshError(
            f"{candidate_database_root.name}: unavailable_concepts must be a list"
        )
    declared_unavailable = {
        (str(entry.get("module")), str(entry.get("concept")))
        for entry in raw_unavailable
        if isinstance(entry, Mapping)
    }
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
                source_schema = _parquet_schema(connection, source_path)
                candidate_schema = _parquet_schema(connection, candidate_path)
                candidate_schema_by_name = dict(candidate_schema)
                missing_or_retyped = [
                    (name, dtype, candidate_schema_by_name.get(name))
                    for name, dtype in source_schema
                    if candidate_schema_by_name.get(name) != dtype
                ]
                if missing_or_retyped:
                    raise ModuleRefreshError(
                        f"{candidate_database_root.name}/{module}: publication-only "
                        "repackaging removed or retyped source columns; "
                        f"differences={missing_or_retyped}"
                    )
                source_names = [name for name, _dtype in source_schema]
                added_columns = [
                    name for name, _dtype in candidate_schema if name not in source_names
                ]
                undeclared_additions = [
                    name
                    for name in added_columns
                    if (module, name) not in declared_unavailable
                ]
                if undeclared_additions:
                    raise ModuleRefreshError(
                        f"{candidate_database_root.name}/{module}: publication-only "
                        "repackaging added columns not declared unavailable; "
                        f"columns={undeclared_additions}"
                    )
                if added_columns:
                    quoted_added = ", ".join(
                        f"count({_quote_identifier(name)})" for name in added_columns
                    )
                    escaped_candidate_path = str(candidate_path.resolve()).replace(
                        "'", "''"
                    )
                    non_null_counts = connection.execute(
                        f"SELECT {quoted_added} FROM "
                        f"read_parquet('{escaped_candidate_path}')"
                    ).fetchone()
                    if non_null_counts is None or any(
                        int(value) != 0 for value in non_null_counts
                    ):
                        raise ModuleRefreshError(
                            f"{candidate_database_root.name}/{module}: publication-only "
                            "schema completion added a column with data; "
                            f"columns={added_columns}, counts={non_null_counts}"
                        )
                source_receipt = _parquet_multiset_receipt(
                    connection,
                    source_path,
                    columns=source_names,
                )
                candidate_receipt = _parquet_multiset_receipt(
                    connection,
                    candidate_path,
                    columns=source_names,
                )
                if source_receipt != candidate_receipt:
                    raise ModuleRefreshError(
                        f"{candidate_database_root.name}/{module}: publication-only "
                        "repackaging changed logical table content; "
                        f"source={source_receipt}, candidate={candidate_receipt}"
                    )
                receipts[module] = {
                    **source_receipt,
                    "candidate_added_declared_all_null_columns": added_columns,
                }
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
    resource_budget_mb: float = DEFAULT_RELEASE_MEMORY_BUDGET_MB,
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
        _validate_refreshed_score_content(
            destination_database_root,
            modules,
            database=database,
        )
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
        native_manifest = staging_root / "_manifest.json"
        canonical_staging = _module_is_canonical_refresh(staging_root, modules)
        producer_staging = _module_is_complete_producer_staging(
            staging_root,
            modules,
        )
        if canonical_staging or producer_staging:
            read_dependencies = _stage_refresh_read_dependencies(
                source_database_root=source_database_root,
                staging_root=staging_root,
                modules=modules,
            )
            if native_manifest.exists() or native_manifest.is_symlink():
                if not canonical_staging:
                    raise ModuleRefreshError(
                        "Staged root manifest exists before module files satisfy "
                        "the native schema"
                    )
                _validate_existing_staging_native_manifest(
                    database=database,
                    staging_root=staging_root,
                    modules=modules,
                )
            else:
                _recover_native_staging_publication(
                    database=database,
                    data_path=data_path,
                    staging_root=staging_root,
                    modules=modules,
                    resource_budget_mb=resource_budget_mb,
                )
                if not _module_is_canonical_refresh(staging_root, modules):
                    raise ModuleRefreshError(
                        "Recovered native publication did not produce canonical modules"
                    )
            _validate_refreshed_score_content(
                staging_root,
                modules,
                database=database,
            )
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
                "read_only_dependencies": read_dependencies,
            }
        raise ModuleRefreshError(
            f"Existing refresh staging is incomplete or not canonical: {staging_root}"
        )
    staging_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    read_dependencies = _stage_refresh_read_dependencies(
        source_database_root=source_database_root,
        staging_root=staging_root,
        modules=modules,
    )
    extraction = extract_database(
        database,
        data_path=data_path,
        output_dir=staging_root,
        modules=list(modules),
        batch_size=batch_size,
        native_export_v2=True,
        stream_output_batches=True,
        # Formal releases execute the reviewed plan exactly. Runtime growth
        # would turn a measured 50k boundary into an unreviewed 67k batch;
        # failures remain diagnosable in the immutable candidate staging area.
        adaptive_stream_batches=False,
        resource_budget_mb=resource_budget_mb,
        verbose=True,
    )
    metrics = _module_runtime_metrics(extraction, modules)
    _validate_refreshed_score_content(
        staging_root,
        modules,
        database=database,
    )
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
        "resource_budget_mb": extraction.get("resource_budget_mb"),
        "resource_execution_limits": extraction.get(
            "resource_execution_limits"
        ),
        "resource_plan": extraction.get("resource_plan"),
        "module_resource_plans": extraction.get("module_resource_plans"),
        "total_elapsed_seconds": extraction.get("total_elapsed"),
        "modules": metrics,
        "read_only_dependencies": read_dependencies,
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
    resource_budget_mb: float = DEFAULT_RELEASE_MEMORY_BUDGET_MB,
    resource_policy_override_reason: str | None = None,
    databases: Sequence[str] = DATABASES,
    database_module_scope: Mapping[str, Sequence[str]] | None = None,
    resume: bool = False,
    repair_finalized: bool = False,
    benchmark_only: bool = False,
) -> Path:
    source = source_run_root.resolve()
    destination = output_root.expanduser().absolute()
    if source == destination:
        raise ModuleRefreshError("Source and destination run roots must differ")
    if batch_size is not None and not str(
        resource_policy_override_reason or ""
    ).strip():
        raise ModuleRefreshError(
            "An explicit batch size is benchmark-only and requires a recorded "
            "resource-policy override reason"
        )
    resolved_databases: Sequence[str] = (
        ()
        if database_module_scope
        else DATABASES if databases is None else databases
    )
    selected_databases, requested_by_database, selected_by_database = (
        _resolve_database_module_scope(
            modules=modules,
            databases=resolved_databases,
            database_module_scope=database_module_scope,
        )
    )
    if resume and set(selected_databases) != set(DATABASES):
        raise ModuleRefreshError(
            "Database-subset refreshes must use a fresh candidate; their "
            "interrupted state is not yet transaction-bound for safe resume"
        )
    requested_modules = tuple(
        module
        for module in MODULES
        if any(
            module in requested_by_database[database]
            for database in selected_databases
        )
    )
    selected_modules = tuple(
        module
        for module in MODULES
        if any(
            module in selected_by_database[database]
            for database in selected_databases
        )
    )
    if benchmark_only and len(selected_databases) != 1:
        raise ModuleRefreshError(
            "--benchmark-only requires exactly one selected database"
        )
    if benchmark_only and batch_size is None:
        raise ModuleRefreshError(
            "--benchmark-only requires an explicit measured --batch-size"
        )
    if benchmark_only and (resume or repair_finalized):
        raise ModuleRefreshError(
            "--benchmark-only cannot resume or repair a release candidate"
        )
    publication_commit = REPUBLICATION._require_clean_checkout()
    source_run_manifest = REPUBLICATION._validate_source(source)
    execution_resource_plan = _build_refresh_resource_plan(
        source_run_manifest,
        requested_modules=modules,
        databases=resolved_databases,
        memory_budget_mb=resource_budget_mb,
        requested_batch_size=batch_size,
        database_module_scope=database_module_scope,
    )
    if (
        batch_size is None
        and not execution_resource_plan["formal_release_admissible"]
    ):
        raise ModuleRefreshError(
            "Formal selected-module refresh refuses unmeasured fallback "
            "batches; profile and register these database/modules first: "
            f"{execution_resource_plan['unmeasured_or_overridden_modules']}"
        )
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
                modules=selected_by_database[database],
                batch_size=batch_size,
                resource_budget_mb=resource_budget_mb,
                reuse_completed_export=resume and not repair_finalized,
            )

        if benchmark_only:
            database = selected_databases[0]
            output_receipts = {}
            for module in selected_by_database[database]:
                parquet = destination / "exports" / database / f"{module}.parquet"
                manifest_path = (
                    destination / "exports" / database / f"{module}.manifest.json"
                )
                _require_regular_file(parquet, label=f"{database}/{module} benchmark")
                _require_regular_file(
                    manifest_path,
                    label=f"{database}/{module} benchmark producer manifest",
                )
                manifest = _read_json(
                    manifest_path,
                    label=f"{database}/{module} benchmark producer manifest",
                )
                saved = manifest.get("saved") or {}
                output_receipts[module] = {
                    "parquet_sha256": REPUBLICATION._sha256_file(parquet),
                    "parquet_bytes": parquet.stat().st_size,
                    "producer_manifest_sha256": REPUBLICATION._sha256_file(
                        manifest_path
                    ),
                    "producer_rows": sum(
                        int(receipt.get("rows") or 0)
                        for receipt in saved.values()
                        if isinstance(receipt, Mapping)
                    ),
                }
            benchmark_provenance = {
                "schema_version": RESOURCE_BENCHMARK_SCHEMA_VERSION,
                "created_at": _utc_now(),
                "benchmark_only": True,
                "sealable": False,
                "source_run_root": str(source),
                "source_run_manifest_sha256": source_run_manifest_sha256,
                "source_database_receipt": source_receipts[database],
                "easyicu_git_commit": publication_commit,
                "easyicu_git_dirty": False,
                "database": database,
                "requested_modules": list(requested_by_database[database]),
                "dependency_closure": list(selected_by_database[database]),
                "resource_policy_override_reason": (
                    resource_policy_override_reason
                ),
                "resource_plan": execution_resource_plan,
                "runtime": refreshed[database],
                "output_receipts": output_receipts,
                "formal_release_admissible": False,
                "next_step": (
                    "independently review memory evidence and register an exact "
                    "measured profile before any formal release refresh"
                ),
            }
            REPUBLICATION._atomic_write_json(
                destination / RESOURCE_BENCHMARK_FILENAME,
                benchmark_provenance,
            )
            return destination

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
                set(selected_by_database[database])
                if database in selected_databases
                else set()
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
                    "resource_budget_mb": (
                        current.get("resource_budget_mb")
                        if current.get("resource_budget_mb") is not None
                        else previous.get("resource_budget_mb")
                    ),
                    "resource_plan": (
                        current.get("resource_plan")
                        if current.get("resource_plan") is not None
                        else previous.get("resource_plan")
                    ),
                    "module_resource_plans": (
                        current.get("module_resource_plans")
                        if current.get("module_resource_plans") is not None
                        else previous.get("module_resource_plans")
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
                database_selected_modules = selected_by_database[database]
                manifest["source_extraction_provenance"] = {
                    **(manifest.get("source_extraction_provenance") or {}),
                    "publication_only": False,
                    "raw_database_reread": True,
                    "refreshed_modules": list(database_selected_modules),
                    "current_refreshed_modules": list(database_selected_modules),
                    "inherited_refreshed_modules": list(base_scope[database]),
                    "cumulative_refreshed_modules": list(
                        per_database_refreshed_modules[database]
                    ),
                    "reused_modules": [
                        module
                        for module in MODULES
                        if module not in database_selected_modules
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
        reused_module_semantic_audit = {}
        for database in selected_databases:
            reused_modules = tuple(
                module
                for module in MODULES
                if module not in selected_by_database[database]
            )
            if reused_modules:
                reused_module_semantic_audit[database] = (
                    _validate_publication_only_database_semantics(
                        source / "exports" / database,
                        destination / "exports" / database,
                        modules=reused_modules,
                    )
                )
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
            "latest_per_database_requested_modules": {
                database: list(requested_by_database[database])
                for database in selected_databases
            },
            "latest_per_database_dependency_closure": {
                database: list(selected_by_database[database])
                for database in selected_databases
            },
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
            "resource_policy": {
                "owner": "easyicu.api.extraction.plan_module_extraction_resources",
                "memory_budget_mb": float(resource_budget_mb),
                "explicit_batch_override": batch_size,
                "override_reason": resource_policy_override_reason,
                "execution_grain": "database_by_module",
                "adaptive_batch_growth": False,
                "formal_release_admissible": execution_resource_plan[
                    "formal_release_admissible"
                ],
            },
            "latest_resource_plan": execution_resource_plan,
            "per_database_runtime": combined_runtime,
            "latest_read_only_dependencies": {
                database: dict(
                    refreshed[database].get("read_only_dependencies") or {}
                )
                for database in selected_databases
            },
            "publication_only_semantic_audit": publication_only_semantic_audit,
            "reused_module_semantic_audit": reused_module_semantic_audit,
            "reused_module_count_per_database": {
                database: len(MODULES)
                - (
                    len(selected_by_database[database])
                    if database in selected_databases
                    else 0
                )
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
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Validate scope and print the database-by-module resource plan "
            "without cloning a candidate or reading raw database tables."
        ),
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        help="Optional JSON destination for --plan-only output.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an unsealed candidate after recovering canonical staged modules.",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help=(
            "Extract and validate one database's closure, write a non-sealable "
            "resource benchmark receipt, and skip six-database republication."
        ),
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
            "sofa1_score/sofa2_score); repeatable."
        ),
    )
    parser.add_argument(
        "--database-module",
        action="append",
        default=[],
        metavar="DATABASE=MODULE[,MODULE]",
        help=(
            "Audited per-database module scope; repeat once per database. "
            "Cannot be combined with --database or --module."
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
        help=(
            "Expert-only fixed stay batch for every module; normal releases use "
            "the measured database-by-module policy."
        ),
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=float,
        default=DEFAULT_RELEASE_MEMORY_BUDGET_MB,
        help=(
            "Reproducible release memory contract in MiB "
            f"(default: {DEFAULT_RELEASE_MEMORY_BUDGET_MB})."
        ),
    )
    parser.add_argument(
        "--allow-resource-policy-override",
        action="store_true",
        help="Acknowledge that --batch-size bypasses the measured policy.",
    )
    parser.add_argument(
        "--resource-policy-override-reason",
        help="Required audit reason when --batch-size is supplied.",
    )
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.memory_budget_mb <= 0:
        parser.error("--memory-budget-mb must be positive")
    if args.plan_output is not None and not args.plan_only:
        parser.error("--plan-output requires --plan-only")
    if args.database_module and (args.database or args.module):
        parser.error(
            "--database-module cannot be combined with --database or --module"
        )
    if not args.plan_only and args.output_root is None:
        parser.error("--output-root is required unless --plan-only is used")
    if args.batch_size is not None and not args.allow_resource_policy_override:
        parser.error(
            "--batch-size is blocked for formal release execution; add "
            "--allow-resource-policy-override and an audit reason only for an "
            "intentional expert override"
        )
    if args.batch_size is not None and not str(
        args.resource_policy_override_reason or ""
    ).strip():
        parser.error("--batch-size requires --resource-policy-override-reason")
    if args.benchmark_only and args.batch_size is None:
        parser.error("--benchmark-only requires --batch-size")
    if args.benchmark_only and (args.resume or args.repair_finalized):
        parser.error("--benchmark-only cannot be combined with resume/repair")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        database_module_scope = _parse_database_module_scopes(
            args.database_module
        )
        if args.plan_only:
            source_manifest = REPUBLICATION._validate_source(
                args.source_run_root.resolve()
            )
            plan = _build_refresh_resource_plan(
                source_manifest,
                requested_modules=args.module,
                databases=(
                    () if database_module_scope else args.database or DATABASES
                ),
                memory_budget_mb=args.memory_budget_mb,
                requested_batch_size=args.batch_size,
                database_module_scope=database_module_scope or None,
            )
            plan["source_run_root"] = str(args.source_run_root.resolve())
            plan["resource_policy_override_reason"] = (
                args.resource_policy_override_reason
            )
            if args.plan_output is not None:
                REPUBLICATION._atomic_write_json(args.plan_output, plan)
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        candidate = refresh_candidate(
            source_run_root=args.source_run_root,
            output_root=args.output_root,
            modules=args.module,
            data_path_overrides=_parse_data_path_overrides(args.data_path),
            batch_size=args.batch_size,
            resource_budget_mb=args.memory_budget_mb,
            resource_policy_override_reason=(
                args.resource_policy_override_reason
            ),
            databases=(
                () if database_module_scope else args.database or DATABASES
            ),
            database_module_scope=database_module_scope or None,
            resume=args.resume,
            repair_finalized=args.repair_finalized,
            benchmark_only=args.benchmark_only,
        )
    except (ModuleRefreshError, OSError, ValueError) as exc:
        print(f"selected-module refresh failed: {exc}", file=sys.stderr)
        return 1
    print(f"Refreshed selected-module candidate: {candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
