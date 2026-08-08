#!/usr/bin/env python3
"""EX-A03: refresh selected raw-derived modules in a sealed six-database run.

This entry point is deliberately narrower than a full re-extraction.  It
copies a verified six-database package by hard link, re-reads raw data only for
the named modules, then republishes every native manifest from one clean
EasyICU checkout.  Thus unchanged module bytes remain immutable in the source
run, while the derived candidate has one consistent runtime provenance and can
be sealed by ``EX-A01_seal_full6_release.py``.

At present only ``renal`` is allowlisted.  This keeps the command's claim
precise: it is the audited KDIGO/urine/RRT refresh path, not a generic way to
bypass the full extraction controller.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from easyicu.api import extract_database  # noqa: E402


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
REFRESHABLE_MODULES = frozenset({"renal"})
SCHEMA_VERSION = "easyicu_full6_selected_module_refresh_v1"


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


def _resolve_data_paths(
    source_manifest: Mapping[str, Any], overrides: Mapping[str, str]
) -> dict[str, str]:
    recorded = source_manifest.get("data_paths")
    if not isinstance(recorded, dict):
        raise ModuleRefreshError("Source run manifest lacks data_paths")
    paths: dict[str, str] = {}
    for database in DATABASES:
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
    disallowed = set(selected) - REFRESHABLE_MODULES
    if disallowed:
        raise ModuleRefreshError(
            "This audited refresh entry point currently allows only renal; "
            f"got disallowed modules: {sorted(disallowed)}"
        )
    return selected


def _require_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ModuleRefreshError(f"Missing regular {label}: {path}")


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
                "peak_working_set_mb": float(
                    receipt.get("peak_working_set_mb") or 0.0
                ),
            }
        except (TypeError, ValueError) as exc:
            raise ModuleRefreshError(
                f"{module} extraction receipt has invalid runtime metrics"
            ) from exc
    return metrics


def _refresh_one_database(
    *,
    database: str,
    data_path: str,
    candidate_root: Path,
    modules: Sequence[str],
    batch_size: int | None,
) -> dict[str, Any]:
    staging_root = candidate_root / ".module_refresh_staging" / database
    if staging_root.exists() or staging_root.is_symlink():
        raise ModuleRefreshError(f"Refusing existing refresh staging root: {staging_root}")
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
    _replace_selected_module_files(
        staging_root=staging_root,
        destination_database_root=candidate_root / "exports" / database,
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
    manifest: dict[str, Any], metrics: Mapping[str, Mapping[str, float]]) -> None:
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


def refresh_candidate(
    *,
    source_run_root: Path,
    output_root: Path,
    modules: Sequence[str],
    data_path_overrides: Mapping[str, str],
    batch_size: int | None,
) -> Path:
    source = source_run_root.resolve()
    destination = output_root.expanduser().absolute()
    if source == destination:
        raise ModuleRefreshError("Source and destination run roots must differ")
    selected_modules = _validate_modules(modules)
    publication_commit = REPUBLICATION._require_clean_checkout()
    source_run_manifest = REPUBLICATION._validate_source(source)
    data_paths = _resolve_data_paths(source_run_manifest, data_path_overrides)
    source_run_manifest_sha256 = REPUBLICATION._sha256_file(
        source / "run_manifest.json"
    )
    source_receipts = {
        database: REPUBLICATION._source_database_receipt(source / "exports" / database)
        for database in DATABASES
    }

    REPUBLICATION._clone_source(source, destination)
    (destination / "run_metadata.json").unlink(missing_ok=True)
    (destination / "module_extraction_timing.csv").unlink(missing_ok=True)
    (destination / "republication_provenance.json").unlink(missing_ok=True)
    shutil.rmtree(destination / "publication_qc", ignore_errors=True)

    refreshed: dict[str, dict[str, Any]] = {}
    try:
        for database in DATABASES:
            refreshed[database] = _refresh_one_database(
                database=database,
                data_path=data_paths[database],
                candidate_root=destination,
                modules=selected_modules,
                batch_size=batch_size,
            )

        native_manifests: dict[str, dict[str, Any]] = {}
        for database in DATABASES:
            manifest = REPUBLICATION._republish_database(
                destination / "exports" / database,
                database=database,
                source_receipt=source_receipts[database],
                publication_commit=publication_commit,
            )
            _rebind_manifest_metrics(manifest, refreshed[database]["modules"])
            manifest["source_extraction_provenance"] = {
                **(manifest.get("source_extraction_provenance") or {}),
                "raw_database_reread": True,
                "refreshed_modules": list(selected_modules),
                "reused_modules": [
                    module for module in MODULES if module not in selected_modules
                ],
                "module_refresh_runtime": refreshed[database],
                "transformation": (
                    "raw re-extraction of selected modules plus canonical native-v2 "
                    "republication of the complete package"
                ),
            }
            REPUBLICATION._atomic_write_json(
                destination / "exports" / database / "_manifest.json", manifest
            )
            native_manifests[database] = manifest

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
            "refreshed_modules": list(selected_modules),
            "raw_database_reread": True,
            "raw_data_paths": data_paths,
            "per_database_runtime": refreshed,
            "reused_module_count_per_database": len(MODULES) - len(selected_modules),
            "seal_required_after_refresh": True,
        }
        REPUBLICATION._atomic_write_json(
            destination / "module_refresh_provenance.json", provenance
        )

        updated_run_manifest = dict(source_run_manifest)
        updated_run_manifest["module_refresh"] = provenance
        updated_run_manifest["publication_checkout"] = {
            "easyicu_git_commit": publication_commit,
            "easyicu_git_dirty": False,
        }
        REPUBLICATION._atomic_write_json(destination / "run_manifest.json", updated_run_manifest)
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
        "--module",
        action="append",
        default=[],
        help="Raw-derived module to refresh (currently only renal); repeatable.",
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
        )
    except (ModuleRefreshError, OSError, ValueError) as exc:
        print(f"selected-module refresh failed: {exc}", file=sys.stderr)
        return 1
    print(f"Refreshed selected-module candidate: {candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
