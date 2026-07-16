"""Typed, case-neutral repair reasons emitted by validation gates.

The classifier intentionally reads validator identity and structured ``detail``
fields only.  Human-facing messages remain free to change without silently
changing repair routing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .schema import ValidationFinding


class RepairReason(str, Enum):
    INVALID_HELPER_SIGNATURE = "INVALID_HELPER_SIGNATURE"
    UNDEFINED_HELPER = "UNDEFINED_HELPER"
    UNBOUND_LOCAL = "UNBOUND_LOCAL"
    LOSSY_NUMERIC_COERCION = "LOSSY_NUMERIC_COERCION"
    LOSSY_ORDINAL_ROUNDING = "LOSSY_ORDINAL_ROUNDING"
    INVALID_NUMERIC_REDUCTION = "INVALID_NUMERIC_REDUCTION"
    ARBITRARY_COLUMN_FALLBACK = "ARBITRARY_COLUMN_FALLBACK"
    TYPED_PRODUCT_BINDING_INVALID = "TYPED_PRODUCT_BINDING_INVALID"
    ROW_ALIGNMENT_UNVERIFIED = "ROW_ALIGNMENT_UNVERIFIED"
    STRUCTURAL_ACCOUNTING_INVALID = "STRUCTURAL_ACCOUNTING_INVALID"
    PROVENANCE_NOT_FAIL_CLOSED = "PROVENANCE_NOT_FAIL_CLOSED"
    DIAGNOSTIC_NOT_COMPLETED = "DIAGNOSTIC_NOT_COMPLETED"
    SCIENTIFIC_SEMANTICS_VIOLATION = "SCIENTIFIC_SEMANTICS_VIOLATION"
    OUTPUT_CONTRACT_INVALID = "OUTPUT_CONTRACT_INVALID"


@dataclass(frozen=True)
class StructuredRepairMetadata:
    """Stable routing coordinates recovered from host-owned repair payloads."""

    reasons: frozenset[str]
    helper_names: frozenset[str]
    failure_modes: frozenset[str]
    line_anchors: frozenset[int]


def structured_repair_metadata(run_log: str) -> StructuredRepairMetadata:
    """Parse typed repair tickets and finding details without reading prose.

    Repair prompts contain human-readable explanations as well as host-owned
    JSON payloads.  Only the latter are stable enough to drive context
    selection or repair specialization.
    """

    reasons: set[str] = set()
    helper_names: set[str] = set()
    failure_modes: set[str] = set()
    line_anchors: set[int] = set()

    def _collect(payload: Any) -> None:
        if isinstance(payload, Mapping):
            for key in ("reason", "structured_reason"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    reasons.add(value.strip())
            helper_name = payload.get("helper_name")
            if isinstance(helper_name, str) and helper_name.strip():
                helper_names.add(helper_name.strip())
            helpers = payload.get("helper_names")
            if isinstance(helpers, list):
                helper_names.update(
                    str(item).strip()
                    for item in helpers
                    if isinstance(item, str) and item.strip()
                )
            failure_mode = payload.get("failure_mode")
            if isinstance(failure_mode, str) and failure_mode.strip():
                failure_modes.add(failure_mode.strip())
            for key, value in payload.items():
                if (
                    isinstance(key, str)
                    and (key == "line" or key.endswith("_line"))
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                ):
                    line_anchors.add(value)
            for value in payload.values():
                _collect(value)
        elif isinstance(payload, list):
            for item in payload:
                _collect(item)

    text = str(run_log or "")
    decoder = json.JSONDecoder()
    for marker in (
        "TYPED REPAIR TICKET (authoritative routing):",
        "DETAIL:",
    ):
        cursor = 0
        while True:
            marker_index = text.find(marker, cursor)
            if marker_index < 0:
                break
            fragment = text[marker_index + len(marker) :].lstrip()
            try:
                payload, _ = decoder.raw_decode(fragment)
            except (json.JSONDecodeError, TypeError):
                pass
            else:
                _collect(payload)
            cursor = marker_index + len(marker)

    return StructuredRepairMetadata(
        reasons=frozenset(reasons),
        helper_names=frozenset(helper_names),
        failure_modes=frozenset(failure_modes),
        line_anchors=frozenset(line_anchors),
    )


_DETAIL_REASON_CODES = {
    "invalid_local_helper_call": RepairReason.INVALID_HELPER_SIGNATURE,
    "undefined_helper_call": RepairReason.UNDEFINED_HELPER,
    "branch_local_unbound": RepairReason.UNBOUND_LOCAL,
    "lossy_ordinal_rounding": RepairReason.LOSSY_ORDINAL_ROUNDING,
    "lossy_numeric_coercion": RepairReason.LOSSY_NUMERIC_COERCION,
    "scalar_cast_before_reduction": RepairReason.INVALID_NUMERIC_REDUCTION,
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
    "host_validation_helper_error_swallowed": (
        RepairReason.STRUCTURAL_ACCOUNTING_INVALID
    ),
    "source_variable_missing_from_authoritative_cohort": (
        RepairReason.TYPED_PRODUCT_BINDING_INVALID
    ),
    "source_variable_not_unique_in_authoritative_cohort": (
        RepairReason.TYPED_PRODUCT_BINDING_INVALID
    ),
    "required_raw_source_missing_from_authoritative_cohort": (
        RepairReason.TYPED_PRODUCT_BINDING_INVALID
    ),
    "required_raw_source_not_unique_in_authoritative_cohort": (
        RepairReason.TYPED_PRODUCT_BINDING_INVALID
    ),
    "provenance_audit_not_fail_closed": RepairReason.PROVENANCE_NOT_FAIL_CLOSED,
    "provenance_helper_error_swallowed": RepairReason.PROVENANCE_NOT_FAIL_CLOSED,
    "provenance_pair_scan_not_bidirectional": (RepairReason.PROVENANCE_NOT_FAIL_CLOSED),
    "declared_diagnostic_not_completed": RepairReason.DIAGNOSTIC_NOT_COMPLETED,
}


# llm_concept_auditor findings carry a strict-schema ``issue_code`` enum
# (validated upstream against _LLM_CONCEPT_ISSUE_CODES).  Routing is explicit
# per code so a future mechanical code can be re-routed without prose
# guessing; every current code keeps the historical semantic route.
# Structural/mechanical gaps (e.g. lossy numeric coercion) are identified by
# the deterministic AST preflight BEFORE any LLM audit, so ``issue_code=other``
# no longer needs to absorb them.
_LLM_CONCEPT_ISSUE_CODE_REASONS = {
    "audit_only_companion_row_gating_required": (
        RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION
    ),
    "finalized_exposure_missing_reconciliation": (
        RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION
    ),
    "finalized_exposure_overridden": RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION,
    "finalized_exposure_forced_raw_reconciliation": (
        RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION
    ),
    "plausibility_range_exclusion_required": (
        RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION
    ),
}


def repair_reason_for_finding(finding: ValidationFinding) -> RepairReason:
    detail: Mapping[str, Any] = finding.detail or {}
    structured_reason = str(detail.get("reason") or detail.get("kind") or "").strip()
    if structured_reason in _DETAIL_REASON_CODES:
        return _DETAIL_REASON_CODES[structured_reason]
    if finding.validator == "llm_concept_auditor":
        issue_code = str(detail.get("issue_code") or "").strip()
        return _LLM_CONCEPT_ISSUE_CODE_REASONS.get(
            issue_code, RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION
        )
    if finding.validator in {
        "declared_product_contract",
        "typed_artifact_evidence_lineage",
    }:
        return RepairReason.TYPED_PRODUCT_BINDING_INVALID
    return RepairReason.OUTPUT_CONTRACT_INVALID


def typed_repair_ticket(
    findings: Sequence[ValidationFinding],
) -> list[dict[str, Any]]:
    """Return one typed ticket per reason while retaining every occurrence.

    Findings at two code locations may share a reason but require two edits.
    Only byte-equivalent occurrence payloads are folded; different line/path/
    evidence details remain visible to the Coder in one aggregated repair pass.
    """

    ticket: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    seen_occurrences: set[str] = set()
    for finding in findings:
        detail = dict(finding.detail or {})
        nested_issues = detail.get("issues")
        if (
            isinstance(nested_issues, list)
            and nested_issues
            and all(isinstance(issue, Mapping) for issue in nested_issues)
        ):
            shared_detail = {
                key: value for key, value in detail.items() if key != "issues"
            }
            occurrence_details = [
                {**shared_detail, **dict(issue)} for issue in nested_issues
            ]
        else:
            occurrence_details = [detail]
        for occurrence_detail in occurrence_details:
            occurrence_finding = finding.model_copy(
                update={"detail": occurrence_detail}
            )
            reason = repair_reason_for_finding(occurrence_finding)
            structured_reason = str(
                occurrence_detail.get("reason") or occurrence_detail.get("kind") or ""
            ).strip()
            key = (reason.value, finding.validator, structured_reason)
            occurrence = {
                "message": str(finding.message or ""),
                "detail": occurrence_detail,
                "evidence_ids": list(finding.evidence_ids or []),
            }
            occurrence_key = json.dumps(
                {
                    "group": key,
                    "occurrence": occurrence,
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            if occurrence_key in seen_occurrences:
                continue
            seen_occurrences.add(occurrence_key)
            item = grouped.get(key)
            if item is None:
                item = {
                    "reason": reason.value,
                    "validator": finding.validator,
                    "structured_reason": structured_reason or None,
                    "detail": occurrence_detail,
                    "occurrences": [],
                }
                grouped[key] = item
                ticket.append(item)
            item["occurrences"].append(occurrence)
    for item in ticket:
        item["occurrence_count"] = len(item["occurrences"])
    return ticket


__all__ = [
    "RepairReason",
    "StructuredRepairMetadata",
    "repair_reason_for_finding",
    "structured_repair_metadata",
    "typed_repair_ticket",
]
