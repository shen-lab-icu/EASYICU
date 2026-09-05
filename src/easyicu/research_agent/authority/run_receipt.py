"""Run-owned facts and artifact digests preserved independently of run scratch.

Receipts are integrity evidence, not a signature or scientific authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

from easyicu.state_paths import state_root

from ..canonical_json import sha256_file as _sha256_file
from .filesystem import (
    AnchoredDirectory,
    AuthorityFilesystemError,
    publish_write_once_bytes,
)

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
RUN_RECEIPT_SOURCE_CHANGED = "RUN_RECEIPT_SOURCE_CHANGED"
RUN_RECEIPT_FACTS_MISMATCH = "RUN_RECEIPT_FACTS_MISMATCH"

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


def _read_json(path: Path, *, missing_code: str) -> Any:
    if path.is_symlink():
        raise ReceiptError(
            RUN_RECEIPT_ARTIFACT_PATH_UNSAFE, f"{path} is a symbolic link"
        )
    if not path.is_file():
        raise ReceiptError(missing_code, f"{path} is not a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReceiptError(RUN_RECEIPT_UNREADABLE_JSON, f"{path}: {exc}") from exc


def _optional_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    return _read_json(path, missing_code=RUN_RECEIPT_ARTIFACT_MISSING)


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
        summary_path = step_dir / "outputs" / STEP_SUMMARY_NAME
        if not summary_path.is_file():
            summary_path = step_dir / STEP_SUMMARY_NAME
        summary = _optional_json(summary_path)
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
        raise ReceiptError(RUN_RECEIPT_RUN_DIR_MISSING, f"{run_dir} is not a directory")

    before = _inventory(run_dir)
    manifest = _read_json(
        run_dir / MANIFEST_NAME, missing_code=RUN_RECEIPT_MANIFEST_MISSING
    )
    status = _read_json(run_dir / STATUS_NAME, missing_code=RUN_RECEIPT_STATUS_MISSING)
    if not isinstance(manifest, dict) or not isinstance(status, dict):
        raise ReceiptError(
            RUN_RECEIPT_UNREADABLE_JSON,
            f"{MANIFEST_NAME} and {STATUS_NAME} must both be JSON objects",
        )

    inventory = before
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
            1 for row in inventory if row["path"].startswith(f"{EVIDENCE_DIR_NAME}/")
        ),
        "artifacts": inventory,
    }
    if inventory != _inventory(run_dir):
        raise ReceiptError(
            RUN_RECEIPT_SOURCE_CHANGED, "run changed while its receipt was read"
        )
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
            findings.append(f"{RUN_RECEIPT_ARTIFACT_PATH_UNSAFE}: {row.get('path')!r}")
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
    if not findings and build_receipt(run_dir) != receipt:
        findings.append(
            f"{RUN_RECEIPT_FACTS_MISMATCH}: receipt facts differ from source artifacts"
        )
    return findings


def write_receipt(receipt: Mapping[str, Any], path: Path, *, run_dir: Path) -> Path:
    """Publish once outside the inventoried tree; identical retries are idempotent."""
    source = run_dir.resolve()
    destination = path.expanduser().absolute()
    if destination.resolve().is_relative_to(source):
        raise ReceiptError(
            RUN_RECEIPT_ARTIFACT_PATH_UNSAFE, "receipt must be outside its run tree"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "posix":
            with AnchoredDirectory.open(destination.parent.resolve()) as directory:
                directory.publish_immutable_bytes(
                    destination.name, _canonical_json_bytes(receipt)
                )
        else:  # Windows has no openat directory descriptors; retain write-once publication.
            if destination.is_symlink():
                raise AuthorityFilesystemError("receipt destination is a symbolic link")
            publish_write_once_bytes(
                destination,
                _canonical_json_bytes(receipt),
                temp_prefix=".run-receipt-",
                conflict_error=AuthorityFilesystemError,
                conflict_message="existing immutable run receipt conflicts with payload",
            )
    except AuthorityFilesystemError as exc:
        raise ReceiptError("RUN_RECEIPT_IMMUTABLE_CONFLICT", str(exc)) from exc
    return destination


def preserve_terminal_run_receipt(
    run_dir: Path, *, destination_root: Optional[Path] = None
) -> Path:
    """Retain every completed/aborted pipeline result without promoting its status.

    A later signoff or package finalization produces a new content-addressed
    receipt. Old versions remain readable. Human-review pauses are not terminal.
    """
    receipt = build_receipt(run_dir)
    run_id = receipt.get("run_id")
    if (
        not isinstance(run_id, str)
        or _safe_relative_artifact_path(run_id) != run_id
        or "/" in run_id
    ):
        raise ReceiptError(
            RUN_RECEIPT_SCHEMA_INVALID, "terminal run requires a safe run_id"
        )
    if not receipt.get("finished_at"):
        raise ReceiptError(
            RUN_RECEIPT_SCHEMA_INVALID, "run has no terminal manifest timestamp"
        )
    if destination_root is None:
        configured = str(os.environ.get("EASYICU_RUN_RECEIPT_ROOT") or "").strip()
        destination_root = (
            Path(configured).expanduser() if configured else state_root() / "run_receipts"
        )
    return write_receipt(
        receipt,
        destination_root / run_id / f"{receipt['receipt_sha256']}.json",
        run_dir=run_dir,
    )
