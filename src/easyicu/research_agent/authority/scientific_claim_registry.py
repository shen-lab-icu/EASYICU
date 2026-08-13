"""Digest-bound persistence contract for host-derived scientific claims.

``EvidenceStore`` remains the sole owner of evidence transactions.  This
module validates one registered summary record and reproduces claim metadata
from immutable evidence bytes; it never inspects or saves store-private state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

from ..schema import EvidenceRecord
from .scientific_claims import (
    ScientificClaim,
    ScientificClaimDraft,
    bind_scientific_claim_drafts,
    derive_scientific_claim_drafts,
)


class ScientificClaimRegistryError(ValueError):
    """Raised when claim metadata cannot be bound to registered evidence."""


@dataclass(frozen=True)
class ScientificClaimRegistration:
    """Validated claims plus whether EvidenceStore must attach their metadata."""

    claims: tuple[ScientificClaim, ...]
    attach_metadata: bool


def _sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while payload := handle.read(chunk):
            digest.update(payload)
    return digest.hexdigest()


def _claim_payload(claims: Sequence[ScientificClaim]) -> list[dict]:
    return [claim.model_dump(mode="json") for claim in claims]


def validate_scientific_claim_registration(
    *,
    root: Path,
    record: EvidenceRecord,
    step_id: str,
    summary: dict,
    drafts: Sequence[ScientificClaimDraft],
) -> ScientificClaimRegistration:
    """Validate claim drafts against the exact registered summary bytes."""

    normalized_step_id = str(step_id).strip()
    if str(record.produced_by_step or "").strip() != normalized_step_id:
        raise ScientificClaimRegistryError(
            "scientific_claims evidence owner does not match the step"
        )
    if str(record.generation_mode or "").strip() != "deterministic_standard":
        raise ScientificClaimRegistryError(
            "scientific_claims require a deterministic standard executor"
        )
    target = Path(root) / record.relative_path
    if _sha256_of_file(target) != record.sha256:
        raise ScientificClaimRegistryError(
            "scientific_claims summary evidence digest drifted"
        )
    try:
        registered_summary = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ScientificClaimRegistryError(
            "scientific_claims summary evidence is not valid JSON"
        ) from exc
    if registered_summary != summary:
        raise ScientificClaimRegistryError(
            "scientific_claims must come from the registered summary bytes"
        )
    claims = tuple(
        bind_scientific_claim_drafts(
            [draft.model_dump(mode="json") for draft in drafts],
            step_id=normalized_step_id,
            evidence_id=record.evidence_id,
        )
    )
    existing = dict(record.metadata or {}).get("scientific_claims")
    payload = _claim_payload(claims)
    if existing is not None and existing != payload:
        raise ScientificClaimRegistryError(
            "scientific_claims cannot change for registered evidence"
        )
    return ScientificClaimRegistration(
        claims=claims,
        attach_metadata=bool(claims and existing is None),
    )


def load_registered_scientific_claims(
    *,
    root: Path,
    records: Sequence[EvidenceRecord],
) -> tuple[ScientificClaim, ...]:
    """Re-derive and validate claim metadata in evidence registration order."""

    claims: list[ScientificClaim] = []
    seen: set[str] = set()
    for record in records:
        raw_claims = dict(record.metadata or {}).get("scientific_claims")
        if raw_claims is None:
            continue
        if not isinstance(raw_claims, list):
            raise ScientificClaimRegistryError(
                "scientific claim authority is not a list"
            )
        target = Path(root) / record.relative_path
        try:
            if _sha256_of_file(target) != record.sha256:
                raise ValueError("evidence digest drifted")
            registered_summary = json.loads(target.read_text(encoding="utf-8"))
            drafts = derive_scientific_claim_drafts(registered_summary)
            expected_claims = bind_scientific_claim_drafts(
                [draft.model_dump(mode="json") for draft in drafts],
                step_id=str(record.produced_by_step or "").strip(),
                evidence_id=record.evidence_id,
            )
        except Exception as exc:
            raise ScientificClaimRegistryError(
                "scientific claim authority cannot be reproduced from evidence"
            ) from exc
        if raw_claims != _claim_payload(expected_claims):
            raise ScientificClaimRegistryError(
                "scientific claim authority differs from host derivation"
            )
        for claim in expected_claims:
            if claim.evidence_id != record.evidence_id or (
                claim.step_id != str(record.produced_by_step or "").strip()
            ):
                raise ScientificClaimRegistryError(
                    "scientific claim authority coordinates do not match evidence"
                )
            if claim.claim_ref in seen:
                raise ScientificClaimRegistryError(
                    "scientific claim authority contains duplicate references"
                )
            seen.add(claim.claim_ref)
            claims.append(claim)
    return tuple(claims)


__all__ = [
    "ScientificClaimRegistration",
    "ScientificClaimRegistryError",
    "load_registered_scientific_claims",
    "validate_scientific_claim_registration",
]
