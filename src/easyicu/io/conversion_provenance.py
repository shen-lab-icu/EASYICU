"""Conversion provenance records observed branches without choosing them."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import duckdb
import pyarrow


def environment(converter) -> dict:
    reader = os.environ.get("EASYICU_CSV_READER", "").lower()
    return {
        "EASYICU_USE_ID_PARTITIONING": os.environ.get(
            "EASYICU_USE_ID_PARTITIONING", "1"
        )
        == "1",
        "EASYICU_CSV_READER": {
            "override": reader or None,
            "chunked": "pyarrow_if_supported" if reader == "pyarrow" else "pandas",
            "end_to_end": "pandas" if reader == "pandas" else "pyarrow_if_supported",
        },
        "EASYICU_CLUSTER_BY_PATIENT": getattr(
            converter,
            "_effective_cluster_by_patient",
            os.environ.get("EASYICU_CLUSTER_BY_PATIENT", "").strip().lower()
            in {"1", "true", "yes", "on"},
        ),
        "EASYICU_CONV_WORKERS": converter.parallel_workers,
        "EASYICU_PARQUET_COMPRESSION": converter.parquet_compression,
    }


def software() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    return dict(
        git_sha=sha,
        pyarrow_version=pyarrow.__version__,
        duckdb_version=duckdb.__version__,
    )


def record_branch(converter, result: dict, branch: str) -> None:
    result["storage_provenance"] = dict(
        branch=branch, effective_environment=environment(converter), **software()
    )


def manifest_provenance(converter, file_name, result, outputs) -> dict:
    recorded = result.get("storage_provenance") or converter._status.get(
        file_name, {}
    ).get("storage_provenance")
    provenance = dict(
        recorded
        or dict(
            branch="unknown",
            effective_environment=None,
            git_sha=None,
            pyarrow_version=None,
            duckdb_version=None,
        )
    )
    if any(p.parent.name.startswith("bucket_id=") for p in outputs):
        provenance["branch"] = "bucket"
    return provenance
