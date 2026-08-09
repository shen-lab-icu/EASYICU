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


DETERMINISTIC_CONCEPT_REAUDIT_BUDGET_ISSUE_CODE = (
    "deterministic_repair_final_audit_budget_exhausted"
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
    provider_used: int,
    provider_limit: int,
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
            or used != provider_used
            or limit != provider_limit
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
    prior_step_records: Sequence[Mapping[str, Any]] = (),
    provider_used: object = None,
    provider_limit: object = None,
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
    if (
        isinstance(provider_used, bool)
        or not isinstance(provider_used, int)
        or provider_used < 0
        or isinstance(provider_limit, bool)
        or not isinstance(provider_limit, int)
        or provider_limit < 0
        or provider_used < provider_limit
    ):
        return ()
    candidates = [
        record for record in prior_step_records if isinstance(record, Mapping)
    ]
    if isinstance(prior_step_record, Mapping) and all(
        record is not prior_step_record for record in candidates
    ):
        candidates.append(prior_step_record)
    for record in reversed(candidates):
        block = record.get("post_repair_concept_audit_block")
        if not isinstance(block, Mapping) or (
            str(block.get("code_sha256") or "").strip().lower()
            != normalized_digest
        ):
            continue
        # The newest exact-digest block is authoritative.  A newer semantic
        # rejection must not be bypassed by searching farther back for an old
        # budget-only checkpoint.
        return _prior_exhausted_reaudit_proof(
            record,
            code_sha256=normalized_digest,
            provider_used=provider_used,
            provider_limit=provider_limit,
        )
    return ()


def deterministic_concept_reaudit_pending_errors(
    findings: Sequence[Mapping[str, Any]],
    *,
    provider_used: int,
    provider_limit: int,
) -> tuple[dict[str, Any], ...]:
    """Select the exact provider-only continuation error from quarantine.

    Historical semantic findings remain useful regression constraints, but an
    earlier exact-digest checkpoint may prove they were already repaired.  In
    that narrow case only the matching exhausted final-audit error remains an
    active quarantine blocker.  The caller must separately prove the repair
    checkpoint with :func:`deterministic_concept_reaudit_authority`.
    """

    selected: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        detail = finding.get("detail")
        if (
            finding.get("validator") != "provider_call_budget"
            or finding.get("severity") != "error"
            or not isinstance(detail, Mapping)
            or detail.get("category") != "concept_audit"
            or detail.get("used") != provider_used
            or detail.get("limit") != provider_limit
        ):
            continue
        marked_finding = dict(finding)
        marked_detail = dict(detail)
        marked_detail["issue_code"] = (
            DETERMINISTIC_CONCEPT_REAUDIT_BUDGET_ISSUE_CODE
        )
        marked_finding["detail"] = marked_detail
        selected.append(marked_finding)
    return tuple(selected)


__all__ = [
    "DETERMINISTIC_CONCEPT_REAUDIT_BUDGET_ISSUE_CODE",
    "deterministic_concept_reaudit_authority",
    "deterministic_concept_reaudit_pending_errors",
]
