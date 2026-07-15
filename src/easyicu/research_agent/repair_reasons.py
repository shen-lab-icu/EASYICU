"""Typed, case-neutral repair reasons emitted by validation gates.

The classifier intentionally reads validator identity and structured ``detail``
fields only.  Human-facing messages remain free to change without silently
changing repair routing.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

from .schema import ValidationFinding


class RepairReason(str, Enum):
    INVALID_HELPER_SIGNATURE = "INVALID_HELPER_SIGNATURE"
    UNDEFINED_HELPER = "UNDEFINED_HELPER"
    UNBOUND_LOCAL = "UNBOUND_LOCAL"
    LOSSY_NUMERIC_COERCION = "LOSSY_NUMERIC_COERCION"
    ARBITRARY_COLUMN_FALLBACK = "ARBITRARY_COLUMN_FALLBACK"
    TYPED_PRODUCT_BINDING_INVALID = "TYPED_PRODUCT_BINDING_INVALID"
    ROW_ALIGNMENT_UNVERIFIED = "ROW_ALIGNMENT_UNVERIFIED"
    STRUCTURAL_ACCOUNTING_INVALID = "STRUCTURAL_ACCOUNTING_INVALID"
    PROVENANCE_NOT_FAIL_CLOSED = "PROVENANCE_NOT_FAIL_CLOSED"
    DIAGNOSTIC_NOT_COMPLETED = "DIAGNOSTIC_NOT_COMPLETED"
    SCIENTIFIC_SEMANTICS_VIOLATION = "SCIENTIFIC_SEMANTICS_VIOLATION"
    OUTPUT_CONTRACT_INVALID = "OUTPUT_CONTRACT_INVALID"


_DETAIL_REASON_CODES = {
    "invalid_local_helper_call": RepairReason.INVALID_HELPER_SIGNATURE,
    "undefined_helper_call": RepairReason.UNDEFINED_HELPER,
    "branch_local_unbound": RepairReason.UNBOUND_LOCAL,
    "lossy_ordinal_rounding": RepairReason.LOSSY_NUMERIC_COERCION,
    "arbitrary_column_fallback": RepairReason.ARBITRARY_COLUMN_FALLBACK,
    "typed binding unavailable": RepairReason.TYPED_PRODUCT_BINDING_INVALID,
    "unpersisted_binding_metadata": RepairReason.TYPED_PRODUCT_BINDING_INVALID,
    "typed_dataframe_artifact_erased": RepairReason.TYPED_PRODUCT_BINDING_INVALID,
    "authoritative_primary_exposure_unused": (
        RepairReason.TYPED_PRODUCT_BINDING_INVALID
    ),
    "authoritative_primary_exposure_fallback": (
        RepairReason.TYPED_PRODUCT_BINDING_INVALID
    ),
    "finalized_exposure_reconciliation_fallback": (
        RepairReason.TYPED_PRODUCT_BINDING_INVALID
    ),
    "row_alignment_unverified": RepairReason.ROW_ALIGNMENT_UNVERIFIED,
    "structural_accounting_filter": RepairReason.STRUCTURAL_ACCOUNTING_INVALID,
    "structural_accounting_integer_validation": (
        RepairReason.STRUCTURAL_ACCOUNTING_INVALID
    ),
    "provenance_audit_not_fail_closed": RepairReason.PROVENANCE_NOT_FAIL_CLOSED,
    "provenance_helper_error_swallowed": RepairReason.PROVENANCE_NOT_FAIL_CLOSED,
    "provenance_pair_scan_not_bidirectional": (
        RepairReason.PROVENANCE_NOT_FAIL_CLOSED
    ),
    "declared_diagnostic_not_completed": RepairReason.DIAGNOSTIC_NOT_COMPLETED,
}


def repair_reason_for_finding(finding: ValidationFinding) -> RepairReason:
    detail: Mapping[str, Any] = finding.detail or {}
    structured_reason = str(detail.get("reason") or detail.get("kind") or "").strip()
    if structured_reason in _DETAIL_REASON_CODES:
        return _DETAIL_REASON_CODES[structured_reason]
    if finding.validator == "llm_concept_auditor":
        return RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION
    if finding.validator in {
        "declared_product_contract",
        "typed_artifact_evidence_lineage",
    }:
        return RepairReason.TYPED_PRODUCT_BINDING_INVALID
    return RepairReason.OUTPUT_CONTRACT_INVALID


def typed_repair_ticket(
    findings: Sequence[ValidationFinding],
) -> list[dict[str, Any]]:
    """Return a stable, deduplicated ticket for the repair coordinator."""

    ticket: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for finding in findings:
        reason = repair_reason_for_finding(finding)
        detail = dict(finding.detail or {})
        structured_reason = str(
            detail.get("reason") or detail.get("kind") or ""
        ).strip()
        key = (reason.value, finding.validator, structured_reason)
        if key in seen:
            continue
        seen.add(key)
        ticket.append(
            {
                "reason": reason.value,
                "validator": finding.validator,
                "structured_reason": structured_reason or None,
                "detail": detail,
            }
        )
    return ticket


__all__ = ["RepairReason", "repair_reason_for_finding", "typed_repair_ticket"]
