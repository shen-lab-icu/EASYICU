#!/usr/bin/env python3
"""Build or verify an immutable snapshot receipt using the runtime's owner.

The pipeline preserves terminal results automatically outside its scratch tree.
This CLI publishes an additional retained copy or verifies one against source:

    build   python tools/preserve_run_receipt.py RUN_DIR --out PATH
    verify  python tools/preserve_run_receipt.py RUN_DIR --verify PATH

A receipt records the source artifacts and their governance status. It is not a
human signature, scientific approval, backup of artifact contents, or a way to
reconstruct a deleted run. Verification requires the original source tree.
Existing different receipt bytes are never overwritten.

Exit codes: 0 success, 1 fail-closed finding with a stable reason code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Compatibility exports for existing receipt tooling and verifiers.
from easyicu.research_agent.authority.run_receipt import (  # noqa: E402, F401
    RECEIPT_SCHEMA_VERSION,
    RUN_RECEIPT_RUN_DIR_MISSING,
    RUN_RECEIPT_MANIFEST_MISSING,
    RUN_RECEIPT_STATUS_MISSING,
    RUN_RECEIPT_UNREADABLE_JSON,
    RUN_RECEIPT_ARTIFACT_MISSING,
    RUN_RECEIPT_ARTIFACT_UNRECORDED,
    RUN_RECEIPT_ARTIFACT_PATH_UNSAFE,
    RUN_RECEIPT_DIGEST_MISMATCH,
    RUN_RECEIPT_SELF_DIGEST_MISMATCH,
    RUN_RECEIPT_SCHEMA_INVALID,
    RUN_RECEIPT_SCHEMA_UNSUPPORTED,
    MANIFEST_NAME,
    STATUS_NAME,
    AUTHORITY_HEAD_NAME,
    EVIDENCE_DIR_NAME,
    STEPS_DIR_NAME,
    STEP_SUMMARY_NAME,
    ReceiptError,
    _canonical_json_bytes,
    _receipt_sha256,
    _safe_relative_artifact_path,
    _sha256_file,
    _read_json,
    _optional_json,
    _inventory,
    _step_outcomes,
    build_receipt,
    verify_receipt,
    write_receipt,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify a durable receipt for a research-agent run."
    )
    parser.add_argument("run_dir", type=Path, help="the run directory to preserve")
    parser.add_argument(
        "--out", type=Path, help="write the receipt here (default: stdout)"
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="re-check this saved receipt against RUN_DIR instead of building one",
    )
    args = parser.parse_args(argv)

    try:
        if args.verify is not None:
            saved = _read_json(args.verify, missing_code=RUN_RECEIPT_ARTIFACT_MISSING)
            if not isinstance(saved, dict):
                raise ReceiptError(
                    RUN_RECEIPT_UNREADABLE_JSON, f"{args.verify} is not a JSON object"
                )
            findings = verify_receipt(args.run_dir, saved)
            if findings:
                for finding in findings:
                    print(finding, file=sys.stderr)
                print(f"FAIL: {len(findings)} finding(s).", file=sys.stderr)
                return 1
            print(
                f"OK: {len(saved.get('artifacts', []))} artifact(s) match "
                f"receipt {saved.get('run_id')}."
            )
            return 0

        receipt = build_receipt(args.run_dir)
    except ReceiptError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = _canonical_json_bytes(receipt)
    if args.out is None:
        sys.stdout.write(payload.decode("utf-8"))
        return 0

    try:
        write_receipt(receipt, args.out, run_dir=args.run_dir)
    except ReceiptError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"wrote {args.out} ({len(payload)} bytes) for run {receipt['run_id']}: "
        f"status={receipt['status']}, "
        f"{receipt['artifact_count']} artifacts / {receipt['artifact_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
