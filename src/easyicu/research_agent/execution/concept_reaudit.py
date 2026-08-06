"""Typed authority for one audit-only call after deterministic code repair.

The ordinary provider ceiling remains the owner of generation and LLM repair
spend.  This module proves the narrower exceptional case: an exact registered
deterministic repair changed a candidate, every post-repair error is only the
exhausted final-audit budget, and the candidate digest is unchanged on resume.
It grants no code-repair or scientific-selection authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)


def _exact_automatic_repair_names(values: Sequence[object]) -> tuple[str, ...]:
    names: list[str] = []
    for value in values:
        name = str(value or "").strip()
        if not name or name in names:
            continue
        metadata = repair_metadata_for(name)
        if (
            metadata.classification_source != "exact"
            or metadata.repair_class is RepairClass.METHOD_SUBSTITUTION
            or not automatic_repair_allowed(name)
        ):
            return ()
        names.append(name)
    return tuple(names)


def _prior_exhausted_reaudit_proof(
    prior_step_record: Mapping[str, Any],
    *,
    code_sha256: str,
) -> tuple[str, ...]:
    block = prior_step_record.get("post_repair_concept_audit_block")
    if (
        not isinstance(block, Mapping)
        or str(block.get("code_sha256") or "").strip().lower() != code_sha256
        or prior_step_record.get("quarantined_repair_succeeded") is not True
    ):
        return ()
    errors = block.get("errors")
    if not isinstance(errors, Sequence) or isinstance(errors, (str, bytes)) or not errors:
        return ()
    for error in errors:
        if not isinstance(error, Mapping) or error.get("validator") != "provider_call_budget":
            return ()
        detail = error.get("detail")
        if not isinstance(detail, Mapping):
            return ()
        used = detail.get("used")
        limit = detail.get("limit")
        if (
            detail.get("category") != "concept_audit"
            or isinstance(used, bool)
            or not isinstance(used, int)
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or used < limit
        ):
            return ()
    return _exact_automatic_repair_names(
        prior_step_record.get("applied_concept_repair_names") or ()
    )


def deterministic_concept_reaudit_authority(
    *,
    code_sha256: str,
    current_repair_count: int,
    current_repair_names: Sequence[object],
    current_repair_code_sha256: object = None,
    prior_step_record: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Return exact repair ids that authorize one final-category extension.

    A repair applied in the current process is authoritative immediately.  A
    resumed repair must additionally carry the prior digest-bound
    ``post_repair_concept_audit_block`` proving that the only remaining error
    was lack of a provider slot for the mandatory final audit.
    """

    normalized_digest = str(code_sha256 or "").strip().lower()
    if len(normalized_digest) != 64 or any(
        char not in "0123456789abcdef" for char in normalized_digest
    ):
        return ()
    if (
        not isinstance(current_repair_count, bool)
        and current_repair_count > 0
        and str(current_repair_code_sha256 or "").strip().lower()
        == normalized_digest
    ):
        return _exact_automatic_repair_names(current_repair_names)
    if isinstance(prior_step_record, Mapping):
        return _prior_exhausted_reaudit_proof(
            prior_step_record,
            code_sha256=normalized_digest,
        )
    return ()


__all__ = ["deterministic_concept_reaudit_authority"]
