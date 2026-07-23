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


def _configure_external_runtime(root: Path) -> tuple[Dict[str, str | None], str | None]:
    """Force every temporary/spill mechanism onto the external run root."""

    runtime_tmp = root / ".runtime_tmp"
    runtime_spill = root / ".runtime_spill"
    runtime_tmp.mkdir(mode=0o700)
    runtime_spill.mkdir(mode=0o700)
    prior = {
        key: os.environ.get(key)
        for key in ("TMPDIR", "TMP", "TEMP", "EASYICU_DUCKDB_TEMP_DIR")
    }
    prior_tempdir = tempfile.tempdir
    os.environ["TMPDIR"] = str(runtime_tmp)
    os.environ["TMP"] = str(runtime_tmp)
    os.environ["TEMP"] = str(runtime_tmp)
    os.environ["EASYICU_DUCKDB_TEMP_DIR"] = str(runtime_spill)
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
        help="emergency low-memory override; omit to preserve one-shot extraction",
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
    runtime_prior, runtime_tempdir = _configure_external_runtime(output_root)
    run_manifest_path = output_root / "run_manifest.json"
    run_manifest: Dict[str, Any] = {
        "schema_version": "easyicu_grouped_native_reexport_run_v1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "database_order": databases,
        "batch_size": args.batch_size,
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
                batch_size=args.batch_size,
                native_export_v2=True,
                verbose=True,
            )
            spill_removed = _remove_worker_spill_directory(source_output)
            with open_export_package(source_output) as package:
                native = result["native_export_v2"]
                run_manifest["sources"][database] = {
                    "status": "verified",
                    "elapsed_sec": round(time.monotonic() - started, 1),
                    "num_patients": result["num_patients"],
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
