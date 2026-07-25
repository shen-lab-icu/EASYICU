"""Emit a PHI-free, auditable record of a SOFA partition-invariance run.

A docstring saying "we measured this" is a claim; this writes the machine
record behind it. Nothing patient-identifying is emitted: cohorts appear only
as a SHA-256 over their sorted stay ids, and frames only as a SHA-256 over the
canonicalised (sorted rows and columns) CSV rendering.

Usage::

    python tools/record_partition_invariance_evidence.py \\
        --data-path /Volumes/外置硬盘/databases/mimiciv \\
        --database miiv --cohort 3000 \\
        --out research_output/partition_invariance/miiv_3000.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import pandas as pd

import easyicu

DEFAULT_CHUNK_SIZES = [None, 250, 500, 1000, 2000, 4000]
DEFAULT_WORKER_COUNTS = [1, 4]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash the canonicalised frame: sorted columns, sorted rows, fixed CSV."""

    ordered = frame.reindex(sorted(frame.columns), axis=1)
    ordered = ordered.sort_values(list(ordered.columns), kind="mergesort").reset_index(
        drop=True
    )
    return _sha256_text(ordered.to_csv(index=False))


def _dtype_fingerprint(frame: pd.DataFrame) -> dict:
    return {str(col): str(dtype) for col, dtype in sorted(frame.dtypes.items())}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
    except Exception:
        return "unknown"


def _git_is_clean() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return not result.stdout.strip()
    except Exception:
        return False


def _prepared_manifest_sha256(data_path: Path) -> str | None:
    manifest = data_path / "conversion_manifest.json"
    if not manifest.is_file():
        return None
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _cohort_fingerprint(frame: pd.DataFrame) -> tuple[str, int]:
    """Hash the cohort's stay ids — never emit the ids themselves."""

    for candidate in ("stay_id", "icustay_id", "patientunitstayid", "admissionid"):
        if candidate in frame.columns:
            ids = sorted(str(value) for value in frame[candidate].dropna().unique())
            return _sha256_text("\n".join(ids)), len(ids)
    return "no-id-column", 0


def _load(concept: str, *, database: str, data_path: str, cohort: int, **extra):
    kwargs = {
        "database": database,
        "data_path": data_path,
        "max_patients": cohort,
        "sample_strategy": "sorted",
        "verbose": False,
    }
    kwargs.update(extra)
    started = time.monotonic()
    frame = easyicu.load_concepts(concept, **kwargs)
    return frame, time.monotonic() - started


def measure(
    *,
    database: str,
    data_path: str,
    cohort: int,
    concepts: list[str],
    chunk_sizes: list,
    worker_counts: list,
) -> dict:
    records = []
    for concept in concepts:
        reference, reference_seconds = _load(
            concept, database=database, data_path=data_path, cohort=cohort
        )
        reference_sha = _canonical_frame_sha256(reference)
        cohort_sha, cohort_size = _cohort_fingerprint(reference)

        configurations = []
        for chunk_size in chunk_sizes:
            if chunk_size is None:
                continue
            frame, seconds = _load(
                concept,
                database=database,
                data_path=data_path,
                cohort=cohort,
                chunk_size=chunk_size,
            )
            configurations.append(
                {
                    "parameter": "chunk_size",
                    "value": chunk_size,
                    "rows": int(len(frame)),
                    "canonical_frame_sha256": _canonical_frame_sha256(frame),
                    "matches_reference": _canonical_frame_sha256(frame)
                    == reference_sha,
                    "seconds": round(seconds, 2),
                }
            )
        for workers in worker_counts:
            frame, seconds = _load(
                concept,
                database=database,
                data_path=data_path,
                cohort=cohort,
                parallel_workers=workers,
            )
            configurations.append(
                {
                    "parameter": "parallel_workers",
                    "value": workers,
                    "rows": int(len(frame)),
                    "canonical_frame_sha256": _canonical_frame_sha256(frame),
                    "matches_reference": _canonical_frame_sha256(frame)
                    == reference_sha,
                    "seconds": round(seconds, 2),
                }
            )

        records.append(
            {
                "concept": concept,
                "reference_rows": int(len(reference)),
                "reference_seconds": round(reference_seconds, 2),
                "reference_canonical_frame_sha256": reference_sha,
                "reference_dtypes": _dtype_fingerprint(reference),
                "cohort_id_sha256": cohort_sha,
                "cohort_size": cohort_size,
                "configurations": configurations,
                "passed": all(item["matches_reference"] for item in configurations),
            }
        )

    return {
        "kind": "sofa_partition_invariance_v1",
        "comparison": (
            "sha256 over the canonicalised frame (columns sorted, rows sorted, "
            "CSV rendering); equivalent to assert_frame_equal(check_exact=True) "
            "on the canonical ordering. NOT a byte comparison of stored files."
        ),
        "code_commit": _git_commit(),
        "code_tree_clean": _git_is_clean(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "database": database,
        "prepared_manifest_sha256": _prepared_manifest_sha256(Path(data_path)),
        "requested_cohort": cohort,
        "results": records,
        "passed": all(record["passed"] for record in records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--database", default="miiv")
    parser.add_argument("--cohort", type=int, default=3000)
    parser.add_argument("--concepts", nargs="+", default=["sofa", "sofa2"])
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[c for c in DEFAULT_CHUNK_SIZES if c is not None],
    )
    parser.add_argument(
        "--worker-counts", nargs="+", type=int, default=DEFAULT_WORKER_COUNTS
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    report = measure(
        database=args.database,
        data_path=args.data_path,
        cohort=args.cohort,
        concepts=args.concepts,
        chunk_sizes=args.chunk_sizes,
        worker_counts=args.worker_counts,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"{'PASS' if report['passed'] else 'FAIL'} -> {out_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
