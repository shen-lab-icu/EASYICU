"""Pure deterministic concept gates and quarantine-policy revalidation.

The functions in this module inspect code and structured findings only.  They
never call a provider, execute candidate code, write evidence, or select a
scientific design.  Operational LLM/cache/receipt work belongs to
``concept_audit_execution``.  In particular, a new optional LLM audit that does
not repeat a stored error is not evidence that the old quarantine is stale;
retirement requires one of the explicit deterministic proofs below.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..audits.patterns import AnalysisPatternAuditor
from ..audits.validators import (
    ConceptUsageAuditor,
    _reclassify_llm_concept_findings,
    _verified_authoritative_exposure_flow,
)
from ..code_preflight import audit_mechanical_code_contracts
from ..contracts import ValidationFinding
from ..method_compatibility import (
    detect_forbidden_pattern_usage,
    format_violation_message,
)
from ..run_input_capsule import canonical_sha256, engine_code_sha256
from ..schema import AnalysisStep, ResearchContext

DETERMINISTIC_CODE_GATE_VALIDATORS = frozenset(
    {
        "analysis_pattern_auditor",
        "concept_usage_auditor",
        "mechanical_code_preflight",
        "method_compatibility",
        "typed_input_authority_flow",
    }
)
_DETERMINISTIC_GATE_SCHEMA_VERSION = "easyicu.deterministic_step_gate/1"
_POSITIONAL_FINDING_KEYS = {
    "line",
    "lines",
    "lineno",
    "col_offset",
    "end_lineno",
    "end_col_offset",
    "offset",
    "offsets",
}


def deterministic_gate_stamp() -> Dict[str, str]:
    """Return the current host-owned deterministic gate identity."""

    engine_digest = engine_code_sha256()
    fingerprint = canonical_sha256(
        {
            "schema_version": _DETERMINISTIC_GATE_SCHEMA_VERSION,
            "engine_code_sha256": engine_digest,
        }
    )
    return {
        "deterministic_gate_schema_version": _DETERMINISTIC_GATE_SCHEMA_VERSION,
        "deterministic_gate_engine_code_sha256": engine_digest,
        "deterministic_gate_fingerprint": fingerprint,
    }


def deterministic_code_gate_findings(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    script_text: str,
    usage_auditor: Optional[ConceptUsageAuditor] = None,
    pattern_auditor: Optional[AnalysisPatternAuditor] = None,
) -> List[ValidationFinding]:
    """Run the shared deterministic pre-execution code gate."""

    usage = usage_auditor if usage_auditor is not None else ConceptUsageAuditor()
    patterns = (
        pattern_auditor if pattern_auditor is not None else AnalysisPatternAuditor()
    )
    findings = usage.audit(
        context=context,
        script_text=script_text,
        step=step,
    )
    findings.extend(
        patterns.audit(
            context=context,
            script_text=script_text,
            step=step,
        )
    )
    compatibility_violations = detect_forbidden_pattern_usage(
        script_text,
        context,
        step,
    )
    if compatibility_violations:
        findings.append(
            ValidationFinding(
                validator="method_compatibility",
                severity="error",
                message=format_violation_message(compatibility_violations),
                detail={
                    "step_id": step.step_id,
                    "violations": compatibility_violations,
                },
            )
        )
    findings.extend(audit_mechanical_code_contracts(script_text, step))
    requires_primary_exposure_artifact = any(
        str(value).strip().casefold() == "artifact:primary_exposure_definition"
        for value in step.inputs or []
    )
    if (
        requires_primary_exposure_artifact
        and str(context.primary_exposure or "").strip()
        and not _verified_authoritative_exposure_flow(
            script_text,
            primary_exposure=str(context.primary_exposure),
        )
    ):
        findings.append(
            ValidationFinding(
                validator="typed_input_authority_flow",
                severity="error",
                message=(
                    f"Step {step.step_id} does not prove that the exact host-bound "
                    "primary exposure reaches its result-bearing model or figure "
                    "after finite/domain validation."
                ),
                detail={
                    "step_id": step.step_id,
                    "input_key": "artifact:primary_exposure_definition",
                    "issue": "typed_primary_exposure_not_consumed",
                },
            )
        )
    return findings


def finding_detail_without_source_positions(value: Any) -> Any:
    """Remove transient code coordinates from a persisted repair constraint."""

    if isinstance(value, Mapping):
        cleaned: Dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            normalized = key.casefold()
            if (
                normalized in _POSITIONAL_FINDING_KEYS
                or normalized.endswith("_line")
                or normalized.endswith("_lines")
                or normalized.endswith("_lineno")
                or normalized.endswith("_offset")
                or normalized.endswith("_offsets")
            ):
                continue
            cleaned[key] = finding_detail_without_source_positions(raw_value)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [finding_detail_without_source_positions(item) for item in value]
    return value


def finding_occurrence_identity(finding: ValidationFinding) -> str:
    """Return a stable identity for one structured validation occurrence."""

    detail = dict(finding.detail or {})
    structured_reason = str(
        detail.get("reason") or detail.get("kind") or detail.get("issue") or ""
    ).strip()
    stable_detail = finding_detail_without_source_positions(detail)
    explicit_occurrence = stable_detail.get("occurrence_id")
    locator_keys = (
        "scope",
        "name",
        "model_id",
        "requirement_id",
        "check_id",
        "term",
        "term_role",
        "input",
        "input_name",
        "column",
        "column_name",
        "source_variable",
        "field",
        "variable",
        "path",
        "artifact",
        "product",
    )
    if explicit_occurrence not in (None, ""):
        stable_locator = {"occurrence_id": explicit_occurrence}
    else:
        stable_locator = {
            key: stable_detail[key]
            for key in locator_keys
            if stable_detail.get(key) not in (None, "", [], {})
        }
    payload: Dict[str, Any] = {
        "validator": finding.validator,
        "structured_reason": structured_reason,
        "locator": stable_locator if stable_locator else detail,
    }
    if not structured_reason or not stable_locator:
        payload["message"] = str(finding.message or "")
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def quarantined_deterministic_errors_resolved_by_current_gate(
    *,
    prior_errors: Sequence[ValidationFinding],
    current_findings: Sequence[ValidationFinding],
    script_text: str,
    quarantined_script_sha256: str,
) -> Optional[List[Dict[str, Any]]]:
    """Prove that exact-digest, host-owned deterministic errors are stale.

    Retirement is allowed only by replaying the current host-owned deterministic
    gate over the exact quarantined script digest.  Silence from a fresh optional
    LLM audit cannot discharge errors recorded by these deterministic validators.
    """

    digest = hashlib.sha256(script_text.encode("utf-8")).hexdigest()
    if digest != str(quarantined_script_sha256 or ""):
        return None
    if not prior_errors or any(
        finding.severity != "error"
        or finding.validator not in DETERMINISTIC_CODE_GATE_VALIDATORS
        for finding in prior_errors
    ):
        return None
    if any(finding.severity == "error" for finding in current_findings):
        return None

    gate_stamp = deterministic_gate_stamp()
    return [
        {
            "validator": finding.validator,
            "message": finding.message,
            "prior_severity": finding.severity,
            "quarantined_script_sha256": digest,
            "revalidated_by": "current_deterministic_code_gate",
            **gate_stamp,
        }
        for finding in prior_errors
    ]


def quarantined_errors_superseded_by_current_policy(
    *,
    prior_errors: Sequence[ValidationFinding],
    current_findings: Sequence[ValidationFinding],
    context: ResearchContext,
    script_text: str,
    quarantined_script_sha256: str,
) -> Optional[Tuple[List[ValidationFinding], List[Dict[str, Any]]]]:
    """Prove that stored errors were retired by a deterministic policy change.

    Absence of a finding from a new optional LLM audit is not evidence that an
    old quarantine is stale.  The only no-code-change exit is to replay the
    current metadata-supported reclassifier over every stored error while the
    complete current deterministic audit independently has no errors.
    """

    if hashlib.sha256(script_text.encode("utf-8")).hexdigest() != str(
        quarantined_script_sha256 or ""
    ):
        return None
    if not prior_errors or any(
        finding.severity == "error" for finding in current_findings
    ):
        return None
    if any(finding.severity != "error" for finding in prior_errors):
        return None
    reclassified = _reclassify_llm_concept_findings(
        findings=prior_errors,
        context=context,
        script_text=script_text,
    )
    if len(reclassified) != len(prior_errors):
        return None

    provenance: List[Dict[str, Any]] = []
    for prior, current in zip(prior_errors, reclassified):
        prior_detail = dict(prior.detail or {})
        current_detail = dict(current.detail or {})
        reason = current_detail.get("downgraded_reason")
        same_finding = (
            current.validator == prior.validator
            and current.message == prior.message
            and current.evidence_ids == prior.evidence_ids
            and all(
                current_detail.get(key) == value for key, value in prior_detail.items()
            )
        )
        if (
            not same_finding
            or "downgraded_reason" in prior_detail
            or current.severity != "warning"
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            return None
        provenance.append(
            {
                "validator": prior.validator,
                "message": prior.message,
                "prior_severity": prior.severity,
                "reclassified_severity": current.severity,
                "downgraded_reason": reason.strip(),
            }
        )
    return reclassified, provenance


__all__ = [
    "DETERMINISTIC_CODE_GATE_VALIDATORS",
    "deterministic_code_gate_findings",
    "deterministic_gate_stamp",
    "finding_detail_without_source_positions",
    "finding_occurrence_identity",
    "quarantined_deterministic_errors_resolved_by_current_gate",
    "quarantined_errors_superseded_by_current_policy",
]
