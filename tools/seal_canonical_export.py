#!/usr/bin/env python
"""Thin CLI over benchmarks.figure2_canonical9.typed_export_seal.

Retrofits an UNTYPED EasyICU module export into a native TYPED export in place
(parquet bytes untouched; extraction_bounds omitted — see the module docstring).
All logic lives in the benchmark package; this file only parses arguments.

    python tools/seal_canonical_export.py --export /path/to/full6/miiv \
        [--database miiv] [--value-vintage 20260717] [--json-report out.json]

``--export`` is required and has NO fallback. The seal refuses to overwrite an
existing ``_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo root (containing ``benchmarks/``) is importable when run as a
# script from an arbitrary CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.figure2_canonical9.typed_export_seal import (  # noqa: E402
    TypedRetrofitSealError,
    seal_export_structural_typed,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Structural typed retrofit seal for an untyped EasyICU export."
    )
    parser.add_argument(
        "--export",
        required=True,
        help="Path to the untyped module export directory (required, no fallback).",
    )
    parser.add_argument("--database", default="miiv")
    parser.add_argument(
        "--value-vintage",
        default="20260717",
        help="Provenance tag for the value vintage of the sealed data.",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        help="Optional path to write the full SealResult JSON (compat report etc.).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan the export and produce the full report WITHOUT writing the "
        "sidecar or _manifest.json (real no-write preflight over the actual data).",
    )
    args = parser.parse_args(argv)

    try:
        result = seal_export_structural_typed(
            args.export,
            database=args.database,
            value_vintage=args.value_vintage,
            dry_run=args.dry_run,
        )
    except TypedRetrofitSealError as exc:
        print(f"seal failed: {exc}", file=sys.stderr)
        return 2

    payload = result.to_dict()
    if args.json_report:
        Path(args.json_report).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    # Human-readable summary to stdout.
    print(
        json.dumps(
            {
                "export_dir": result.export_dir,
                "seal_kind": result.seal_kind,
                "dry_run": result.dry_run,
                "value_vintage": result.value_vintage,
                "value_vintage_basis": result.value_vintage_basis,
                "bounds_authority": result.bounds_authority,
                "metadata_provenance": result.metadata_provenance,
                "dict_fingerprint": result.dict_fingerprint,
                "patient_identity": result.patient_identity,
                "paper_authorized": result.semantic_review["paper_authorized"],
                "sidecar_file": result.sidecar_file,
                "manifest_path": result.manifest_path,
                "parquet_immutability_verified": result.parquet_immutability_verified,
                "compat_summary": result.compat_summary(),
                "sealed_files": [f["file"] for f in result.files],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    # Success = a completed dry-run, or a real seal with verified immutable parquets.
    if result.dry_run:
        return 0
    return 0 if result.parquet_immutability_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
