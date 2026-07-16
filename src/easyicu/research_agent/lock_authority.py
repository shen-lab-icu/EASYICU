"""Authority checks for plan-time lock files.

Plan locks have two copies in a run: a live file used by execution and an
immutable, digest-registered evidence copy.  Modern runs require byte identity
between them.  A narrow compatibility repair is retained for legacy resume
code that rewrote only the volatile ``locked_at`` timestamp while preserving
the complete scientific payload.

The repair deliberately lives at the plan/resume write boundary.  Ordinary
readers remain fail-closed and never self-heal a changed lock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .evidence_authority import (
    EVIDENCE_AUTHORITY_FILENAME,
    EVIDENCE_AUTHORITY_HEAD_FILENAME,
    EVIDENCE_AUTHORITY_MARKER_FILENAME,
    EVIDENCE_AUTHORITY_PREVIOUS_FILENAME,
    EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME,
    EVIDENCE_AUTHORITY_TRANSACTION_FILENAME,
    EvidenceAuthorityIntegrityError,
    load_current_evidence_snapshot,
)


class LockAuthorityError(ValueError):
    """Raised when a plan-time lock cannot be verified against its anchor."""


@dataclass(frozen=True)
class VerifiedLockAnchor:
    path: Path
    sha256: str
    record: Dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_unique_lock_anchor(
    *,
    run_dir: Path,
    evidence_id: str,
    label: str,
) -> Optional[VerifiedLockAnchor]:
    """Return one digest-valid evidence anchor, or ``None`` for legacy runs."""

    run_root = Path(run_dir).resolve()
    try:
        snapshot = load_current_evidence_snapshot(run_root)
    except EvidenceAuthorityIntegrityError as exc:
        raise LockAuthorityError(
            f"{label} evidence authority is invalid: {exc}"
        ) from exc
    raw_records = list(snapshot.records)
    if not raw_records:
        evidence_dir = run_root / "evidence"
        ledger_paths = [
            run_root / EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME,
            run_root / EVIDENCE_AUTHORITY_HEAD_FILENAME,
            run_root / EVIDENCE_AUTHORITY_TRANSACTION_FILENAME,
            evidence_dir / EVIDENCE_AUTHORITY_FILENAME,
            evidence_dir / EVIDENCE_AUTHORITY_PREVIOUS_FILENAME,
            evidence_dir / EVIDENCE_AUTHORITY_MARKER_FILENAME,
            evidence_dir / "evidence_index.json",
            evidence_dir / "evidence_aliases.json",
            evidence_dir / "numeric_claims.json",
        ]
        if not any(path.exists() or path.is_symlink() for path in ledger_paths):
            return None
        raise LockAuthorityError(f"{label} has no unique plan-time evidence anchor")
    anchors = [
        record
        for record in raw_records
        if isinstance(record, dict)
        and str(record.get("evidence_id") or "") == evidence_id
    ]
    if len(anchors) != 1:
        raise LockAuthorityError(f"{label} has no unique plan-time evidence anchor")

    # Lazy import avoids schema/lock module cycles during model initialisation.
    from .runtime_artifacts import verified_run_evidence_path

    record = dict(anchors[0])
    anchor_path = verified_run_evidence_path(run_root, record)
    if anchor_path is None:
        raise LockAuthorityError(f"{label} evidence anchor is missing or stale")
    expected_sha = str(record.get("sha256") or "").strip().lower()
    return VerifiedLockAnchor(
        path=anchor_path,
        sha256=expected_sha,
        record=record,
    )


def assert_lock_matches_evidence_anchor(
    *,
    run_dir: Path,
    lock_path: Path,
    evidence_id: str,
    label: str,
) -> bool:
    """Require byte identity with a modern plan-time evidence anchor."""

    anchor = verified_unique_lock_anchor(
        run_dir=run_dir,
        evidence_id=evidence_id,
        label=label,
    )
    if anchor is None:
        return False
    if lock_path.is_symlink() or not lock_path.is_file():
        raise LockAuthorityError(f"{label} must be a regular file")
    if sha256_file(lock_path) != anchor.sha256:
        raise LockAuthorityError(f"{label} differs from its plan-time evidence anchor")
    return True


def rehydrate_timestamp_only_legacy_lock(
    *,
    run_dir: Path,
    lock_path: Path,
    evidence_id: str,
    label: str,
) -> Optional[Dict[str, Any]]:
    """Restore an old resume-time timestamp rewrite from verified evidence.

    The live and anchored JSON objects must be identical after removing exactly
    their top-level ``locked_at`` values.  Any scientific, schema, digest, or
    additional metadata difference is left untouched so the normal authority
    check rejects it.
    """

    anchor = verified_unique_lock_anchor(
        run_dir=run_dir,
        evidence_id=evidence_id,
        label=label,
    )
    if anchor is None:
        return None
    if lock_path.is_symlink() or not lock_path.is_file():
        raise LockAuthorityError(f"{label} must be a regular file")

    before_sha = sha256_file(lock_path)
    if before_sha == anchor.sha256:
        return None
    try:
        live_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        anchor_payload = json.loads(anchor.path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LockAuthorityError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(live_payload, dict) or not isinstance(anchor_payload, dict):
        raise LockAuthorityError(f"{label} has an invalid payload")

    live_locked_at = live_payload.get("locked_at")
    anchor_locked_at = anchor_payload.get("locked_at")
    if not isinstance(live_locked_at, str) or not isinstance(anchor_locked_at, str):
        return None
    live_without_time = dict(live_payload)
    anchor_without_time = dict(anchor_payload)
    live_without_time.pop("locked_at", None)
    anchor_without_time.pop("locked_at", None)
    if live_without_time != anchor_without_time:
        return None

    from .evidence import _atomic_write_bytes

    _atomic_write_bytes(
        lock_path,
        anchor.path.read_bytes(),
        expected_root=Path(run_dir).resolve(),
    )
    return {
        "repair": "rehydrated_verified_anchor_after_legacy_timestamp_rewrite",
        "before_sha256": before_sha,
        "anchor_sha256": anchor.sha256,
        "live_locked_at": live_locked_at,
        "anchor_locked_at": anchor_locked_at,
    }


__all__ = [
    "LockAuthorityError",
    "assert_lock_matches_evidence_anchor",
    "rehydrate_timestamp_only_legacy_lock",
    "sha256_file",
    "verified_unique_lock_anchor",
]
