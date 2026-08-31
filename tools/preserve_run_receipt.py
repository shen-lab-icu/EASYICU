#!/usr/bin/env python3
"""Preserve a small, verifiable receipt for one research-agent run directory.

Why this exists (2026-08-30 retention audit): a run tree lives under ``output/``
or ``research_output/``, both of which are gitignored regenerable scratch that
gets pruned. The submission plan nevertheless cites those runs as evidence — the
active WebApp row cites ``run_20260829T024326_bfcbf6`` ("11/11 analysis steps, 12
tables, 3 figures, evidence-bound article draft") — and that directory no longer
exists anywhere on disk. What survived is a prose paragraph in an unversioned
task log, which is exactly the unauditable claim the evidence machinery exists to
prevent.

The gap is retention, not mechanism. A run already emits every fact a receipt
needs. The source facts are compact; the complete digest inventory is typically
tens of kilobytes:

    manifest.json                          run id, question, timestamps, plan authority
    run_status.json                        status, gates, code_version
    .easyicu_evidence_authority_head.json  generation + head_sha256

This tool copies those decisions into ONE committable JSON and adds a SHA-256
inventory of the artifacts the run produced. It introduces no new gate, no new
runtime branch, and no new policy: it only preserves what the pipeline already
decided, so a pruned run stays provable.

    build   python tools/preserve_run_receipt.py RUN_DIR --out PATH
    verify  python tools/preserve_run_receipt.py RUN_DIR --verify PATH

``--verify`` re-reads the run tree and fails closed when a recorded digest moved.
It is meaningful only while the run tree still exists; a receipt for a pruned run
remains readable evidence but can no longer be re-verified against its source.

Exit codes: 0 success, 1 fail-closed finding (each names a stable reason code).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

RECEIPT_SCHEMA_VERSION = "easyicu.run_receipt/1"

# Stable reason codes, so a failure names its own cause the way the research
# agent's own validators do.
RUN_RECEIPT_RUN_DIR_MISSING = "RUN_RECEIPT_RUN_DIR_MISSING"
RUN_RECEIPT_MANIFEST_MISSING = "RUN_RECEIPT_MANIFEST_MISSING"
RUN_RECEIPT_STATUS_MISSING = "RUN_RECEIPT_STATUS_MISSING"
RUN_RECEIPT_UNREADABLE_JSON = "RUN_RECEIPT_UNREADABLE_JSON"
RUN_RECEIPT_ARTIFACT_MISSING = "RUN_RECEIPT_ARTIFACT_MISSING"
RUN_RECEIPT_ARTIFACT_UNRECORDED = "RUN_RECEIPT_ARTIFACT_UNRECORDED"
RUN_RECEIPT_ARTIFACT_PATH_UNSAFE = "RUN_RECEIPT_ARTIFACT_PATH_UNSAFE"
RUN_RECEIPT_DIGEST_MISMATCH = "RUN_RECEIPT_DIGEST_MISMATCH"
RUN_RECEIPT_SELF_DIGEST_MISMATCH = "RUN_RECEIPT_SELF_DIGEST_MISMATCH"
RUN_RECEIPT_SCHEMA_INVALID = "RUN_RECEIPT_SCHEMA_INVALID"
RUN_RECEIPT_SCHEMA_UNSUPPORTED = "RUN_RECEIPT_SCHEMA_UNSUPPORTED"

MANIFEST_NAME = "manifest.json"
STATUS_NAME = "run_status.json"
AUTHORITY_HEAD_NAME = ".easyicu_evidence_authority_head.json"
EVIDENCE_DIR_NAME = "evidence"
STEPS_DIR_NAME = "steps"
STEP_SUMMARY_NAME = "step_summary.json"


class ReceiptError(RuntimeError):
    """A fail-closed receipt finding carrying a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _canonical_json_bytes(value: Any) -> bytes:
    """Byte-stable JSON so a receipt digest is reproducible across machines."""

    text = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)
    return (text + "\n").encode("utf-8")


def _receipt_sha256(receipt: Mapping[str, Any]) -> str:
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _safe_relative_artifact_path(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        return None
    return candidate.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, missing_code: str) -> Any:
    if not path.is_file():
        raise ReceiptError(missing_code, f"{path} is not a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReceiptError(RUN_RECEIPT_UNREADABLE_JSON, f"{path}: {exc}") from exc


def _optional_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _inventory(run_dir: Path) -> List[Dict[str, Any]]:
    """SHA-256 every regular file in the run tree, as sorted relative paths.

    Directory walking is deterministic so two builds of the same tree produce
    byte-identical receipts.
    """

    rows: List[Dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _step_outcomes(run_dir: Path) -> List[Dict[str, Any]]:
    steps_dir = run_dir / STEPS_DIR_NAME
    if not steps_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for step_dir in sorted(p for p in steps_dir.iterdir() if p.is_dir()):
        summary = _optional_json(step_dir / STEP_SUMMARY_NAME)
        row: Dict[str, Any] = {"step": step_dir.name}
        if isinstance(summary, dict):
            for key in ("status", "step_id", "kind", "evidence_id"):
                if key in summary:
                    row[key] = summary[key]
        rows.append(row)
    return rows


def build_receipt(run_dir: Path) -> Dict[str, Any]:
    """Compile the run's own decisions into one committable receipt."""

    if not run_dir.is_dir():
        raise ReceiptError(
            RUN_RECEIPT_RUN_DIR_MISSING, f"{run_dir} is not a directory"
        )

    manifest = _read_json(
        run_dir / MANIFEST_NAME, missing_code=RUN_RECEIPT_MANIFEST_MISSING
    )
    status = _read_json(run_dir / STATUS_NAME, missing_code=RUN_RECEIPT_STATUS_MISSING)
    if not isinstance(manifest, dict) or not isinstance(status, dict):
        raise ReceiptError(
            RUN_RECEIPT_UNREADABLE_JSON,
            f"{MANIFEST_NAME} and {STATUS_NAME} must both be JSON objects",
        )

    inventory = _inventory(run_dir)
    receipt: Dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "run_id": manifest.get("run_id"),
        "research_question": manifest.get("research_question"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        # The governance verdict is copied verbatim: this tool never upgrades a
        # run's status and never claims publication readiness on its behalf.
        "status": status.get("status"),
        "strict_fail_closed": status.get("strict_fail_closed"),
        "gates": status.get("gates"),
        "code_version": status.get("code_version"),
        "current_plan_authority": manifest.get("current_plan_authority"),
        "evidence_authority_head": _optional_json(run_dir / AUTHORITY_HEAD_NAME),
        "steps": _step_outcomes(run_dir),
        "artifact_count": len(inventory),
        "artifact_bytes": sum(row["bytes"] for row in inventory),
        "evidence_file_count": sum(
            1
            for row in inventory
            if row["path"].startswith(f"{EVIDENCE_DIR_NAME}/")
        ),
        "artifacts": inventory,
    }
    # Digest of everything above, so a committed receipt can be checked for
    # tampering even after the run tree it describes is gone.
    receipt["receipt_sha256"] = _receipt_sha256(receipt)
    return receipt


def verify_receipt(run_dir: Path, receipt: Dict[str, Any]) -> List[str]:
    """Re-check a saved receipt against a run tree that is still on disk."""

    schema = receipt.get("schema_version")
    if schema != RECEIPT_SCHEMA_VERSION:
        return [
            f"{RUN_RECEIPT_SCHEMA_UNSUPPORTED}: receipt schema {schema!r} "
            f"is not {RECEIPT_SCHEMA_VERSION!r}"
        ]

    findings: List[str] = []
    expected_self_digest = receipt.get("receipt_sha256")
    actual_self_digest = _receipt_sha256(receipt)
    if expected_self_digest != actual_self_digest:
        findings.append(
            f"{RUN_RECEIPT_SELF_DIGEST_MISMATCH}: recorded "
            f"{expected_self_digest!r} but found {actual_self_digest}"
        )

    artifact_rows = receipt.get("artifacts")
    if not isinstance(artifact_rows, list):
        findings.append(f"{RUN_RECEIPT_SCHEMA_INVALID}: artifacts must be a list")
        return findings
    if not run_dir.is_dir():
        findings.append(f"{RUN_RECEIPT_RUN_DIR_MISSING}: {run_dir} is not a directory")
        return findings

    recorded: Dict[str, Mapping[str, Any]] = {}
    for row in artifact_rows:
        if not isinstance(row, Mapping):
            findings.append(
                f"{RUN_RECEIPT_SCHEMA_INVALID}: artifact row must be an object"
            )
            continue
        relative_path = _safe_relative_artifact_path(row.get("path"))
        if relative_path is None:
            findings.append(
                f"{RUN_RECEIPT_ARTIFACT_PATH_UNSAFE}: {row.get('path')!r}"
            )
            continue
        if relative_path in recorded:
            findings.append(
                f"{RUN_RECEIPT_SCHEMA_INVALID}: duplicate artifact {relative_path}"
            )
            continue
        if not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64:
            findings.append(
                f"{RUN_RECEIPT_SCHEMA_INVALID}: invalid digest for {relative_path}"
            )
            continue
        recorded[relative_path] = row

    current = {row["path"]: row for row in _inventory(run_dir)}
    for relative_path, row in recorded.items():
        current_row = current.get(relative_path)
        if current_row is None:
            findings.append(f"{RUN_RECEIPT_ARTIFACT_MISSING}: {relative_path}")
            continue
        if current_row["sha256"] != row["sha256"]:
            findings.append(
                f"{RUN_RECEIPT_DIGEST_MISMATCH}: {relative_path} "
                f"recorded {row['sha256']} but found {current_row['sha256']}"
            )
    for relative_path in sorted(current.keys() - recorded.keys()):
        findings.append(f"{RUN_RECEIPT_ARTIFACT_UNRECORDED}: {relative_path}")
    return findings


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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    print(
        f"wrote {args.out} ({len(payload)} bytes) for run {receipt['run_id']}: "
        f"status={receipt['status']}, "
        f"{receipt['artifact_count']} artifacts / {receipt['artifact_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
