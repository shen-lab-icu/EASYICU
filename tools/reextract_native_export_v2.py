#!/usr/bin/env python3
"""Create one new, private six-source native-v2 EasyICU export.

This is intentionally a sequential launcher: it preserves the grouped
``extract_database`` performance path within each source while keeping peak
memory bounded across sources. It never modifies a historical export root and
publishes each source package only after its typed native manifest is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Sequence

from easyicu.api import extract_database
from easyicu.research_agent.intake.export_package import open_export_package

DEFAULT_DATABASE_ORDER = ("miiv", "mimic", "eicu", "aumc", "hirid", "sic")
DEFAULT_DATA_PATHS = {
    "miiv": "/Volumes/外置硬盘/databases/mimiciv",
    "mimic": "/Volumes/外置硬盘/databases/mimiciii",
    "eicu": "/Volumes/外置硬盘/databases/eicu",
    "aumc": "/Volumes/外置硬盘/databases/aumc",
    "hirid": "/Volumes/外置硬盘/databases/hirid",
    "sic": "/Volumes/外置硬盘/databases/sic",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_private_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _remove_private_directory(path: Path) -> bool:
    """Remove one exact, completed private runtime directory."""

    if not path.exists():
        return False
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"unexpected runtime path; refusing removal: {path}")
    shutil.rmtree(path)
    return True


def _remove_worker_spill_directory(source_output: Path) -> bool:
    """Remove only the extractor's completed private DuckDB spill directory."""

    return _remove_private_directory(source_output / ".easyicu_spill")


def _adaptive_oneshot_budget_mb(
    available_memory_mb: float | None = None,
) -> int:
    """Size nested concept worksets from memory that is available *now*.

    The streamed patient batch is already the primary peak-RAM boundary.  A
    fixed 512 MB budget below 12 GB available was too conservative: on an
    8 GB-available laptop it split an already-bounded outer batch into tiny
    nested fragments and repeatedly scanned the same source tables.

    Use one third of memory available at launch, rounded down to 512 MiB, with
    a 512 MiB floor and 8 GiB ceiling.  This gives 2.5 GiB at 8 GiB available,
    while retaining the established 8 GiB server cap.
    """

    if available_memory_mb is None:
        try:
            import psutil

            available_memory_mb = psutil.virtual_memory().available / (1024**2)
        except Exception:
            # Fail conservatively when system memory cannot be inspected.
            available_memory_mb = 0.0
    available = max(0.0, float(available_memory_mb))
    quantum_mb = 512
    budget = int(available / 3.0 / quantum_mb) * quantum_mb
    return max(quantum_mb, min(8 * 1024, budget))


def _one_shot_runtime_limits(
    available_memory_mb: float | None = None,
    cpu_count: int | None = None,
) -> Dict[str, str]:
    """Scale an explicitly requested one-shot run to the current host.

    The streamed/default path stays at the cross-platform 1-thread safety
    baseline. ``--one-shot`` is an explicit request to process a whole module
    at once, so a large server should not be silently throttled to laptop
    resources. Keep the existing conservative profile below 24 GiB available,
    then scale in bounded tiers up to 8 workers and 8 GiB.
    """

    if available_memory_mb is None:
        try:
            import psutil

            available_memory_mb = psutil.virtual_memory().available / (1024**2)
        except Exception:
            available_memory_mb = 0.0
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1

    available = max(0.0, float(available_memory_mb))
    cpus = max(1, int(cpu_count))
    if available >= 64 * 1024:
        workers, memory_gb, cache_mb = min(8, cpus), 8, 8 * 1024
    elif available >= 32 * 1024:
        workers, memory_gb, cache_mb = min(4, cpus), 4, 6 * 1024
    elif available >= 24 * 1024:
        workers, memory_gb, cache_mb = min(2, cpus), 2, 2 * 1024
    else:
        workers, memory_gb, cache_mb = 1, 1, 256

    return {
        "duckdb_threads": str(workers),
        "duckdb_memory_limit": f"{memory_gb}GB",
        "parallel_max_workers": str(workers),
        "cache_budget_mb": str(cache_mb),
    }


def _configure_external_runtime(
    root: Path, *, one_shot: bool
) -> tuple[Dict[str, str | None], str | None]:
    """Force every temporary/spill mechanism onto the external run root.

    ``one_shot`` changes only the extraction policy.  Temporary DuckDB state
    remains external in both modes.
    """

    runtime_tmp = root / ".runtime_tmp"
    runtime_spill = root / ".runtime_spill"
    runtime_tmp.mkdir(mode=0o700)
    runtime_spill.mkdir(mode=0o700)
    runtime_keys = (
        "TMPDIR",
        "TMP",
        "TEMP",
        "EASYICU_DUCKDB_TEMP_DIR",
        "EASYICU_DUCKDB_THREADS",
        "EASYICU_DUCKDB_MEMORY_LIMIT",
        "EASYICU_PARALLEL_MAX_WORKERS",
        "EASYICU_CACHE_BUDGET_MB",
        "EASYICU_ONESHOT_BUDGET_MB",
    )
    prior = {key: os.environ.get(key) for key in runtime_keys}
    prior_tempdir = tempfile.tempdir
    os.environ["TMPDIR"] = str(runtime_tmp)
    os.environ["TMP"] = str(runtime_tmp)
    os.environ["TEMP"] = str(runtime_tmp)
    os.environ["EASYICU_DUCKDB_TEMP_DIR"] = str(runtime_spill)
    # Keep the default streamed path at the portable safety baseline. An
    # explicit one-shot run may scale within bounded tiers on a large server.
    runtime_limits = (
        _one_shot_runtime_limits()
        if one_shot
        else {
            "duckdb_threads": "1",
            "duckdb_memory_limit": "1GB",
            "parallel_max_workers": "1",
            "cache_budget_mb": "256",
        }
    )
    os.environ["EASYICU_DUCKDB_THREADS"] = runtime_limits["duckdb_threads"]
    os.environ["EASYICU_DUCKDB_MEMORY_LIMIT"] = runtime_limits[
        "duckdb_memory_limit"
    ]
    os.environ["EASYICU_PARALLEL_MAX_WORKERS"] = runtime_limits[
        "parallel_max_workers"
    ]
    os.environ["EASYICU_CACHE_BUDGET_MB"] = runtime_limits["cache_budget_mb"]
    if one_shot:
        # Do not let the export launcher silently turn an explicitly requested
        # all-patient module into auto-batches because of its safety profile.
        os.environ.pop("EASYICU_ONESHOT_BUDGET_MB", None)
    else:
        os.environ["EASYICU_ONESHOT_BUDGET_MB"] = str(
            _adaptive_oneshot_budget_mb()
        )
    tempfile.tempdir = str(runtime_tmp)
    return prior, prior_tempdir


def _restore_runtime(prior: Dict[str, str | None], prior_tempdir: str | None) -> None:
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    tempfile.tempdir = prior_tempdir


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="new non-existent root; historical exports are never overwritten",
    )
    parser.add_argument(
        "--databases",
        nargs="+",
        choices=DEFAULT_DATABASE_ORDER,
        default=list(DEFAULT_DATABASE_ORDER),
        help="sequential source order (default: all six)",
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        metavar="DATABASE=PATH",
        help="override one source path; may be repeated",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "bounded external-export batch size; default auto-detects current "
            "available memory (explicit values always win)"
        ),
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help=(
            "process every source module with its full patient set once; "
            "temporary and DuckDB spill files remain under the external output root"
        ),
    )
    return parser.parse_args(argv)


def _data_paths(overrides: Sequence[str]) -> Dict[str, str]:
    paths = dict(DEFAULT_DATA_PATHS)
    for raw in overrides:
        database, separator, path = str(raw).partition("=")
        if not separator or database not in DEFAULT_DATA_PATHS or not path.strip():
            raise ValueError(f"invalid --data-path override: {raw!r}")
        paths[database] = path.strip()
    return paths


def run(args: argparse.Namespace) -> Dict[str, Any]:
    os.umask(0o077)
    output_root = Path(args.output_root).expanduser()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"output root must be new and non-symlink: {output_root}")
    paths = _data_paths(args.data_path)
    databases = list(args.databases)
    for database in databases:
        if not Path(paths[database]).is_dir():
            raise FileNotFoundError(
                f"source data directory missing for {database}: {paths[database]}"
            )

    output_root.mkdir(mode=0o700, parents=True)
    runtime_prior, runtime_tempdir = _configure_external_runtime(
        output_root, one_shot=args.one_shot
    )
    run_manifest_path = output_root / "run_manifest.json"
    run_manifest: Dict[str, Any] = {
        "schema_version": "easyicu_grouped_native_reexport_run_v1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "database_order": databases,
        "batch_size": None if args.one_shot else args.batch_size,
        "extraction_mode": (
            "one_shot_all_patients"
            if args.one_shot
            else "memory_adaptive_streamed_patient_batches"
        ),
        "runtime_limits": {
            "duckdb_threads": int(os.environ["EASYICU_DUCKDB_THREADS"]),
            "duckdb_memory_limit": os.environ["EASYICU_DUCKDB_MEMORY_LIMIT"],
            "parallel_max_workers": int(
                os.environ["EASYICU_PARALLEL_MAX_WORKERS"]
            ),
            "cache_budget_mb": int(os.environ["EASYICU_CACHE_BUDGET_MB"]),
            "nested_workset_budget_mb": (
                None
                if args.one_shot
                else int(os.environ["EASYICU_ONESHOT_BUDGET_MB"])
            ),
        },
        "sources": {},
        "status": "running",
    }
    _write_private_json(run_manifest_path, run_manifest)

    for database in databases:
        started = time.monotonic()
        source_output = output_root / database
        try:
            result = extract_database(
                database,
                data_path=paths[database],
                output_dir=source_output,
                batch_size=None if args.one_shot else args.batch_size,
                native_export_v2=True,
                stream_output_batches=not args.one_shot,
                verbose=True,
            )
            spill_removed = _remove_worker_spill_directory(source_output)
            with open_export_package(source_output) as package:
                native = result["native_export_v2"]
                run_manifest["sources"][database] = {
                    "status": "verified",
                    "elapsed_sec": round(time.monotonic() - started, 1),
                    "num_patients": result["num_patients"],
                    "effective_batch_size": result.get("batch_size"),
                    "native_manifest_sha256": _sha256(source_output / "_manifest.json"),
                    "column_metadata_sha256": package.column_metadata_sha256,
                    "typed_columns": len(package.concept_index),
                    "missing_selected_concepts": list(
                        package.missing_selected_concepts
                    ),
                    "unavailable_modules": json.loads(
                        (source_output / "_manifest.json").read_text(encoding="utf-8")
                    ).get("unavailable_modules", []),
                    "output_validation_reads": native["output_validation_reads"],
                    "spill_directory_removed": spill_removed,
                }
        except BaseException as exc:
            run_manifest["sources"][database] = {
                "status": "failed",
                "elapsed_sec": round(time.monotonic() - started, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }
            run_manifest["status"] = "failed"
            _write_private_json(run_manifest_path, run_manifest)
            raise
        _write_private_json(run_manifest_path, run_manifest)

    run_manifest["status"] = "verified"
    _write_private_json(run_manifest_path, run_manifest)
    _restore_runtime(runtime_prior, runtime_tempdir)
    _remove_private_directory(output_root / ".runtime_tmp")
    _remove_private_directory(output_root / ".runtime_spill")
    return run_manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = run(args)
    except (OSError, ValueError) as exc:
        print(f"native re-export failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "sources": list(manifest["sources"]),
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
