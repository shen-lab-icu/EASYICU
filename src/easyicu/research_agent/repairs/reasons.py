"""Typed, case-neutral repair reasons emitted by validation gates.

The classifier intentionally reads validator identity and structured ``detail``
fields only.  Human-facing messages remain free to change without silently
changing repair routing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from ..schema import ValidationFinding


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


class RepairRoute(str, Enum):
    """Closed, case-neutral specialization routes chosen by the host."""

    SPARSE_EVENT = "binary_event_reconciliation"
    PROVENANCE_VALUE_SELECTION = "provenance_audit_not_fail_closed"
    PRIMARY_EXPOSURE_BINDING = "authoritative_primary_exposure_unused"
    TABULAR_EXPOSURE_BINDING = "finalized_exposure_reconciliation_fallback"
    ASSIGNMENT_COMPLETION = "assignment_model_unfitted"
    ASSIGNMENT_BINDING = "registered_propensity_score_column_unavailable"
    UNDEFINED_HELPER = "undefined_helper_call"
    FIGURE_SOURCE_TRACE = "no_verifiable_values"
    STRUCTURAL_ACCOUNTING = "partial_cohort_flow"
    ARBITRARY_COLUMN = "arbitrary_frame_order"
    INTEGER_ACCOUNTING = "fractional_count_values"
    BINDING_METADATA = "unpersisted_binding_metadata"
    ORDINAL_COVARIATE = "ordinal_covariate"


@dataclass(frozen=True)
class StructuredRepairMetadata:
    """Stable routing coordinates recovered from host-owned repair payloads."""

    reasons: frozenset[str]
    helper_names: frozenset[str]
    failure_modes: frozenset[str]
    line_anchors: frozenset[int]


_REPAIR_PROMPT_AUTHORITY_SCHEMA = "easyicu.repair_prompt_authority/1"
_SAFE_AUTHORITY_TOKEN = re.compile(r"^[A-Za-z0-9_.:/-]{1,192}$")
_SAFE_TICKET_KEYS = frozenset(
    {
        "analysis_set_id",
        "allowed_values",
        "branch_line",
        "call_line",
        "column",
        "columns",
        "detail",
        "evidence_ids",
        "expected",
        "expected_count",
        "failure_mode",
        "field",
        "first_use_line",
        "following_guard_line",
        "handler_line",
        "helper_name",
        "helper_names",
        "issue_code",
        "issues",
        "kind",
        "line",
        "matched_patterns",
        "model_id",
        "name",
        "occurrence_count",
        "occurrences",
        "path",
        "preferred",
        "reason",
        "reported",
        "role",
        "spec_id",
        "structured_reason",
        "validator",
        "variable",
        "violations",
    }
)
_DROP_AUTHORITY_VALUE = object()

# Coordinates from a finding may steer both prompt specialization and the code
# excerpt shown to the Coder. Keep that authority deny-by-default: only
# validators implemented by the host and deliberately registered here may
# retain arbitrary structured detail. Unknown/future validators (including
# plugins or model-backed auditors with an unexpected name) receive the same
# conservative projection as known model-origin findings.
_TRUSTED_HOST_REPAIR_VALIDATORS = frozenset(
    {
        "analysis_pattern_auditor",
        "code_preflight",  # legacy deterministic validator identity
        "concept_usage_auditor",
        "declared_product_contract",
        "figure_source_data",
        "mechanical_code_preflight",
        "method_compatibility",
        "ordered_stratified_contract",
        "primary_model_contract",
        "step_summary_integrity",
        "typed_artifact_evidence_lineage",
        "typed_input_authority_flow",
    }
)


def _safe_authority_value(value: Any, *, ticket: bool) -> Any:
    """Project arbitrary payloads onto non-instructional JSON coordinates."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return (
            stripped
            if _SAFE_AUTHORITY_TOKEN.fullmatch(stripped)
            else _DROP_AUTHORITY_VALUE
        )
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            safe = _safe_authority_value(item, ticket=ticket)
            if safe is not _DROP_AUTHORITY_VALUE:
                items.append(safe)
        return items
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or "").strip()
            if not _SAFE_AUTHORITY_TOKEN.fullmatch(key):
                continue
            if (
                ticket
                and key not in _SAFE_TICKET_KEYS
                and not (key == "line" or key.endswith("_line"))
            ):
                continue
            safe = _safe_authority_value(raw_value, ticket=ticket)
            if safe is not _DROP_AUTHORITY_VALUE:
                projected[key] = safe
        return projected
    return _DROP_AUTHORITY_VALUE


def _safe_typed_ticket(
    typed_ticket: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for item in typed_ticket:
        validator = str(item.get("validator") or "").strip().lower()
        if validator not in _TRUSTED_HOST_REPAIR_VALIDATORS:
            safe_item: dict[str, Any] = {}
            for key in ("reason", "validator", "occurrence_count"):
                safe = _safe_authority_value(item.get(key), ticket=True)
                if safe is not _DROP_AUTHORITY_VALUE:
                    safe_item[key] = safe
            issue_codes = sorted(_ticket_issue_codes(item))
            if issue_codes:
                safe_item["detail"] = {"issue_code": issue_codes}
            if safe_item:
                projected.append(safe_item)
            continue
        safe = _safe_authority_value(dict(item), ticket=True)
        if isinstance(safe, dict) and safe:
            projected.append(safe)
    return projected


def _ticket_issue_codes(payload: Any) -> set[str]:
    codes: set[str] = set()
    allowed = {*_LLM_CONCEPT_ISSUE_CODE_REASONS, "other"}
    if isinstance(payload, Mapping):
        value = payload.get("issue_code")
        candidates = value if isinstance(value, list) else [value]
        codes.update(
            item for item in candidates if isinstance(item, str) and item in allowed
        )
        for nested in payload.values():
            codes.update(_ticket_issue_codes(nested))
    elif isinstance(payload, list):
        for item in payload:
            codes.update(_ticket_issue_codes(item))
    return codes


def _ticket_tokens(payload: Any) -> set[str]:
    tokens: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in {
                "helper_name",
                "reason",
                "structured_reason",
                "kind",
                "issue_code",
                "validator",
            } and isinstance(value, str):
                tokens.add(value)
            if key == "helper_names" and isinstance(value, list):
                tokens.update(item for item in value if isinstance(item, str))
            tokens.update(_ticket_tokens(value))
    elif isinstance(payload, list):
        for item in payload:
            tokens.update(_ticket_tokens(item))
    return tokens


def _derived_repair_routes(ticket: Sequence[Mapping[str, Any]]) -> set[str]:
    tokens = _ticket_tokens(ticket)
    routes: set[str] = set()
    if "reconcile_binary_event_presence" in tokens:
        routes.add(RepairRoute.SPARSE_EVENT.value)
    if tokens & {
        RepairReason.PROVENANCE_NOT_FAIL_CLOSED.value,
        "provenance_audit_not_fail_closed",
        "provenance_pair_scan_not_bidirectional",
    }:
        routes.add(RepairRoute.PROVENANCE_VALUE_SELECTION.value)
    if tokens & {
        "authoritative_primary_exposure_unused",
        "authoritative_primary_exposure_fallback",
    }:
        routes.add(RepairRoute.PRIMARY_EXPOSURE_BINDING.value)
    if tokens & {
        "authoritative_primary_exposure_fallback",
        "finalized_exposure_reconciliation_fallback",
        "typed_dataframe_artifact_erased",
    }:
        routes.add(RepairRoute.TABULAR_EXPOSURE_BINDING.value)
    if "assignment_model_unfitted" in tokens:
        routes.update(
            {
                RepairRoute.ASSIGNMENT_COMPLETION.value,
                RepairRoute.TABULAR_EXPOSURE_BINDING.value,
            }
        )
    if tokens & {RepairReason.UNDEFINED_HELPER.value, "undefined_helper_call"}:
        routes.add(RepairRoute.UNDEFINED_HELPER.value)
    if "structural_accounting_filter" in tokens:
        routes.add(RepairRoute.STRUCTURAL_ACCOUNTING.value)
    if "structural_accounting_integer_validation" in tokens:
        routes.add(RepairRoute.INTEGER_ACCOUNTING.value)
    if tokens & {
        RepairReason.ARBITRARY_COLUMN_FALLBACK.value,
        "arbitrary_column_fallback",
    }:
        routes.add(RepairRoute.ARBITRARY_COLUMN.value)
    if "unpersisted_binding_metadata" in tokens:
        routes.add(RepairRoute.BINDING_METADATA.value)
    if tokens & {"figure_source_data", "figure_source_trace"}:
        routes.add(RepairRoute.FIGURE_SOURCE_TRACE.value)
    return routes


def _canonical_json(value: Any) -> str:
    """Return a stable JSON representation of host-owned prompt authority."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


@dataclass(frozen=True)
class RepairPromptAuthority:
    """Immutable host-owned repair ticket transported outside runtime logs.

    Candidate stdout/stderr is intentionally absent from this object.  Only a
    host caller that already owns validator findings or deterministic policy
    may construct it, and the canonical JSON form makes the exact authority
    suitable for receipt binding and system-role prompt transport.
    """

    canonical_json: str = (
        '{"host_guidance":{},"route_codes":[],"schema_version":"'
        + _REPAIR_PROMPT_AUTHORITY_SCHEMA
        + '","typed_ticket":[]}'
    )

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.canonical_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("repair prompt authority is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "typed_ticket",
            "host_guidance",
            "route_codes",
        }:
            raise ValueError("repair prompt authority has an invalid schema")
        if payload.get("schema_version") != _REPAIR_PROMPT_AUTHORITY_SCHEMA:
            raise ValueError("repair prompt authority has an unsupported version")
        if not isinstance(payload.get("typed_ticket"), list):
            raise ValueError("repair prompt authority typed_ticket must be a list")
        if not isinstance(payload.get("host_guidance"), dict):
            raise ValueError("repair prompt authority host_guidance must be a mapping")
        if payload["typed_ticket"] != _safe_typed_ticket(payload["typed_ticket"]):
            raise ValueError("repair prompt authority ticket contains unsafe fields")
        safe_guidance = _safe_authority_value(payload["host_guidance"], ticket=False)
        if payload["host_guidance"] != safe_guidance:
            raise ValueError("repair prompt authority guidance contains unsafe fields")
        route_codes = payload.get("route_codes")
        if not isinstance(route_codes, list) or any(
            not isinstance(item, str)
            or item not in {route.value for route in RepairRoute}
            for item in route_codes
        ):
            raise ValueError("repair prompt authority route_codes are invalid")
        if route_codes != sorted(set(route_codes)):
            raise ValueError(
                "repair prompt authority route_codes must be sorted unique"
            )
        if not _derived_repair_routes(payload["typed_ticket"]) <= set(route_codes):
            raise ValueError("repair prompt authority omits a derived route code")
        if self.canonical_json != _canonical_json(payload):
            raise ValueError("repair prompt authority JSON must be canonical")

    @classmethod
    def create(
        cls,
        *,
        findings: Sequence[ValidationFinding] = (),
        typed_ticket: Sequence[Mapping[str, Any]] | None = None,
        host_guidance: Mapping[str, Any] | None = None,
        route_codes: Sequence[RepairRoute | str] = (),
    ) -> "RepairPromptAuthority":
        """Build authority from host findings or an already typed ticket."""

        if findings and typed_ticket is not None:
            raise ValueError("provide findings or typed_ticket, not both")
        raw_ticket = (
            typed_repair_ticket(findings)
            if typed_ticket is None
            else [dict(item) for item in typed_ticket]
        )
        ticket_payload = _safe_typed_ticket(raw_ticket)
        explicit_routes = {
            item.value if isinstance(item, RepairRoute) else str(item)
            for item in route_codes
        }
        valid_routes = {route.value for route in RepairRoute}
        if not explicit_routes <= valid_routes:
            raise ValueError("repair prompt authority contains an unknown route")
        safe_guidance = _safe_authority_value(dict(host_guidance or {}), ticket=False)
        if not isinstance(safe_guidance, dict):
            safe_guidance = {}
        payload = {
            "schema_version": _REPAIR_PROMPT_AUTHORITY_SCHEMA,
            "typed_ticket": ticket_payload,
            "host_guidance": safe_guidance,
            "route_codes": sorted(
                explicit_routes | _derived_repair_routes(ticket_payload)
            ),
        }
        return cls(canonical_json=_canonical_json(payload))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RepairPromptAuthority":
        """Validate a serialized authority payload without trusting its source."""

        authority = cls(canonical_json=_canonical_json(dict(payload)))
        verified = authority.payload()
        if set(verified["route_codes"]) != _derived_repair_routes(
            verified["typed_ticket"]
        ):
            raise ValueError(
                "serialized repair authority contains routes not derived from "
                "its typed ticket"
            )
        return authority

    def payload(self) -> dict[str, Any]:
        """Return a detached JSON-safe payload for hashing or persistence."""

        payload = json.loads(self.canonical_json)
        if not isinstance(payload, dict):  # pragma: no cover - guarded in __post_init__
            raise AssertionError("validated repair authority must be a mapping")
        return payload

    @property
    def is_empty(self) -> bool:
        payload = self.payload()
        return (
            not payload["typed_ticket"]
            and not payload["host_guidance"]
            and not payload["route_codes"]
        )

    def metadata(self) -> StructuredRepairMetadata:
        return structured_repair_metadata(self)

    def routing_text(self) -> str:
        """Return only closed host route codes for specialization matching."""

        payload = self.payload()
        return " ".join(payload["route_codes"])

    def render(self) -> str:
        """Render the exact typed authority for a dedicated system message."""

        return json.dumps(
            self.payload(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )


def repair_prompt_binding_sha256(
    *,
    untrusted_diagnostic: str,
    repair_authority: RepairPromptAuthority,
    current_repair_authority: RepairPromptAuthority | None = None,
) -> str:
    """Hash the exact diagnostic and typed authority shown to one repair."""

    payload: dict[str, Any] = {
        "schema": "easyicu.repair_prompt_binding/1",
        "untrusted_diagnostic": str(untrusted_diagnostic or ""),
        "repair_authority": repair_authority.payload(),
    }
    current = current_repair_authority or repair_authority
    if current.canonical_json != repair_authority.canonical_json:
        payload["current_repair_authority"] = current.payload()
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def structured_repair_metadata(
    authority: RepairPromptAuthority | None,
) -> StructuredRepairMetadata:
    """Read exact coordinates from a typed host side-channel only.

    Raw runtime text is deliberately not accepted.  This prevents candidate
    stdout/stderr from forging marker strings that alter repair routing,
    scientific context selection, or code excerpt selection.
    """

    if authority is None:
        authority = RepairPromptAuthority()
    if not isinstance(authority, RepairPromptAuthority):
        raise TypeError("structured repair metadata requires RepairPromptAuthority")

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

    _collect(authority.payload()["typed_ticket"])

    return StructuredRepairMetadata(
        reasons=frozenset(reasons),
        helper_names=frozenset(helper_names),
        failure_modes=frozenset(failure_modes),
        line_anchors=frozenset(line_anchors),
    )


_DETAIL_REASON_CODES = {
    "invalid_local_helper_call": RepairReason.INVALID_HELPER_SIGNATURE,
    "host_helper_call_signature_invalid": RepairReason.INVALID_HELPER_SIGNATURE,
    "host_helper_runtime_introspection": RepairReason.INVALID_HELPER_SIGNATURE,
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
    "RepairPromptAuthority",
    "RepairReason",
    "RepairRoute",
    "StructuredRepairMetadata",
    "repair_reason_for_finding",
    "repair_prompt_binding_sha256",
    "structured_repair_metadata",
    "typed_repair_ticket",
]
