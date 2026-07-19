"""Step-scoped concept-audit execution and quarantine coordination.

This module owns the operational boundary around deterministic concept gates,
the optional digest-bound LLM audit, provider-budget accounting, and retirement
of stored quarantine findings.  It does not choose a cohort, exposure, outcome,
method, or estimand.  Policy functions remain host-owned dependencies supplied
by the execute layer, which keeps this module independent from
``pipeline_execute`` and prevents a reverse import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Sequence

from ..audits.patterns import AnalysisPatternAuditor
from ..audits.validators import (
    ConceptUsageAuditor,
    LLMConceptAuditor,
    _reclassify_llm_concept_findings,
)
from .concept_audit_cache import LLMConceptAuditCache
from ..gates.concept import (
    DETERMINISTIC_CODE_GATE_VALIDATORS,
    deterministic_code_gate_findings,
    finding_occurrence_identity,
    quarantined_deterministic_errors_resolved_by_current_gate,
    quarantined_errors_superseded_by_current_policy,
)
from ..contracts import ValidationFinding
from ..authority.evidence_store import sha256_of_bytes
from ..authority.provider_budget import (
    ProviderCallBudgetError,
    ProviderCallBudgetReceiptError,
    StepProviderCallBudget,
)
from ..authority.run_input import canonical_sha256
from ..schema import AnalysisStep, ResearchContext
from ..authority.step_attempt import StepAttemptState
from ..step_worker_state import StepWorkerProgress


class ConceptQuarantineState:
    """Mutable per-step concept-audit quarantine state.

    The object is deliberately step-local.  Concurrent workers create separate
    instances and therefore cannot share quarantined findings or retirement
    flags.
    """

    __slots__ = (
        "draft_active",
        "policy_superseded",
        "deterministic_revalidated",
        "pending_errors",
        "resumed_draft_used",
        "repair_materially_changed",
        "repair_succeeded",
        "superseded_by_fallback",
    )

    def __init__(self) -> None:
        self.draft_active: bool = False
        self.policy_superseded: bool = False
        self.deterministic_revalidated: bool = False
        self.pending_errors: List[ValidationFinding] = []
        self.resumed_draft_used: bool = False
        self.repair_materially_changed: bool = False
        self.repair_succeeded: bool = False
        self.superseded_by_fallback: bool = False


@dataclass(frozen=True, slots=True)
class ConceptAuditAuthority:
    """Immutable scientific and audit identity for one planned step."""

    context: ResearchContext
    step: AnalysisStep
    resolved_input_bindings: Mapping[str, Any]
    environment_sha256: str
    auditor_implementation_sha256: str
    auditor_identity: Callable[[], str]
    enable_llm_audit: bool


@dataclass(slots=True)
class ConceptAuditRuntime:
    """Injected mutable collaborators for one step-scoped audit coordinator."""

    usage_auditor: ConceptUsageAuditor
    pattern_auditor: AnalysisPatternAuditor
    cache: LLMConceptAuditCache
    client: Any
    provider_budget: StepProviderCallBudget
    step_attempt_state: StepAttemptState
    worker_progress: StepWorkerProgress
    quarantine_state: ConceptQuarantineState
    step_record: MutableMapping[str, Any]
    run_dir: Path
    run_id: str
    step_current: int
    total_steps: int
    sync_provider_budget: Callable[[], None]
    emit_progress: Callable[..., None]
    quarantine_error_payloads: Callable[
        [Sequence[ValidationFinding]], List[dict[str, Any]]
    ]
    store_quarantined_draft: Callable[..., Any]


@dataclass(slots=True)
class ConceptAuditCoordinator:
    """Evaluate one candidate digest without leaking orchestration into gates."""

    authority: ConceptAuditAuthority
    runtime: ConceptAuditRuntime
    completed_digests: set[str] = field(default_factory=set)
    tokens_by_digest: dict[str, str] = field(default_factory=dict)

    def findings_for_code(
        self,
        script_text: str,
        *,
        include_llm: bool,
    ) -> List[ValidationFinding]:
        """Run deterministic code gates and, when requested, the LLM audit.

        Deterministic semantic/mechanical checks always run before execution.
        The comparatively expensive LLM audit is reserved for an exact code
        digest that has already executed successfully and passed the early
        host-owned output contracts.  Stored quarantine errors remain part of
        the deterministic pre-execution decision and therefore can never be
        bypassed by deferring a fresh LLM call.
        """

        authority = self.authority
        runtime = self.runtime
        step = authority.step
        quarantine = runtime.quarantine_state
        code_findings = deterministic_code_gate_findings(
            context=authority.context,
            step=step,
            script_text=script_text,
            usage_auditor=runtime.usage_auditor,
            pattern_auditor=runtime.pattern_auditor,
        )
        deterministic_errors = [
            finding
            for finding in code_findings
            if finding.severity == "error"
            and finding.validator != "llm_concept_auditor"
        ]
        if quarantine.pending_errors:
            deterministic_revalidation = (
                quarantined_deterministic_errors_resolved_by_current_gate(
                    prior_errors=quarantine.pending_errors,
                    current_findings=code_findings,
                    script_text=script_text,
                    quarantined_script_sha256=str(
                        runtime.step_record.get("quarantined_draft_sha256") or ""
                    ),
                )
            )
            if deterministic_revalidation is not None:
                quarantine.deterministic_revalidated = True
                quarantine.draft_active = False
                quarantine.pending_errors = []
                runtime.step_record[
                    "quarantine_deterministic_revalidation_succeeded"
                ] = True
                runtime.step_record["quarantine_deterministic_revalidated_findings"] = (
                    deterministic_revalidation
                )
                runtime.emit_progress(
                    "audit",
                    (
                        "Retiring stored deterministic concept errors after "
                        f"exact-digest revalidation for {step.step_id}."
                    ),
                    status="warning",
                    run_id=runtime.run_id,
                    step_id=step.step_id,
                    current_step=runtime.step_current,
                    total_steps=runtime.total_steps,
                )
            # Policy supersession is a deterministic decision about the exact
            # quarantined digest. Make it before the optional LLM audit so a
            # retired historical error cannot trigger an unnecessary call.
            supersession = None
            if quarantine.pending_errors:
                supersession = quarantined_errors_superseded_by_current_policy(
                    prior_errors=quarantine.pending_errors,
                    current_findings=code_findings,
                    context=authority.context,
                    script_text=script_text,
                    quarantined_script_sha256=str(
                        runtime.step_record.get("quarantined_draft_sha256") or ""
                    ),
                )
            if supersession is not None:
                reclassified_findings, provenance = supersession
                existing_keys = {
                    (finding.severity, finding_occurrence_identity(finding))
                    for finding in code_findings
                }
                code_findings.extend(
                    finding
                    for finding in reclassified_findings
                    if (
                        finding.severity,
                        finding_occurrence_identity(finding),
                    )
                    not in existing_keys
                )
                quarantine.policy_superseded = True
                quarantine.draft_active = False
                quarantine.pending_errors = []
                runtime.step_record["quarantine_policy_superseded"] = True
                runtime.step_record["quarantine_policy_superseded_findings"] = (
                    provenance
                )
                runtime.emit_progress(
                    "audit",
                    (
                        "Retiring stored concept errors under the current "
                        f"deterministic validator policy for {step.step_id}."
                    ),
                    status="warning",
                    run_id=runtime.run_id,
                    step_id=step.step_id,
                    current_step=runtime.step_current,
                    total_steps=runtime.total_steps,
                )
        try:
            audited_code_digest = sha256_of_bytes(script_text.encode("utf-8"))
            sealed_capsule_audit = (
                runtime.step_attempt_state.capsule_audit_findings_by_digest.get(
                    audited_code_digest
                )
            )
            if include_llm and sealed_capsule_audit is not None:
                sealed_findings, audit_key = sealed_capsule_audit
                if (
                    runtime.provider_budget.snapshot().get("reserved_final_category")
                    == "concept_audit"
                ):
                    runtime.provider_budget.bind_reserved_category(
                        "concept_audit",
                        token=audit_key,
                    )
                    self.tokens_by_digest[audited_code_digest] = audit_key
                existing = {
                    (
                        finding.validator,
                        finding.severity,
                        finding.message,
                        canonical_sha256(finding.detail or {}),
                    )
                    for finding in code_findings
                }
                code_findings.extend(
                    finding
                    for finding in sealed_findings
                    if (
                        finding.validator,
                        finding.severity,
                        finding.message,
                        canonical_sha256(finding.detail or {}),
                    )
                    not in existing
                )
                self.completed_digests.add(audited_code_digest)
                runtime.step_record["capsule_concept_audit_replayed"] = True
            elif (
                include_llm
                and authority.enable_llm_audit
                and (
                    runtime.worker_progress.deterministic_fallback_used
                    or runtime.worker_progress.deterministic_standard_executor_used
                )
            ):
                generation_mode = (
                    "deterministic_standard"
                    if runtime.worker_progress.deterministic_standard_executor_used
                    else "deterministic_fallback"
                )
                code_findings.append(
                    ValidationFinding(
                        validator="llm_concept_auditor",
                        severity="info",
                        message=(
                            "Skipped optional LLM concept audit for trusted "
                            f"{generation_mode} code in step {step.step_id}; "
                            "deterministic audits still ran."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "generation_mode": generation_mode,
                        },
                    )
                )
            elif include_llm and authority.enable_llm_audit and deterministic_errors:
                code_findings.append(
                    ValidationFinding(
                        validator="llm_concept_auditor",
                        severity="info",
                        message=(
                            "Deferred optional LLM concept audit because the "
                            "deterministic mechanical/concept preflight already "
                            f"blocked step {step.step_id}. The repaired digest "
                            "will be audited after deterministic checks pass."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "deterministic_error_validators": sorted(
                                {finding.validator for finding in deterministic_errors}
                            ),
                        },
                    )
                )
            elif include_llm and authority.enable_llm_audit:
                llm_audit_client = runtime.client
                if llm_audit_client is not None:
                    llm_concept_auditor = LLMConceptAuditor(llm_audit_client)
                    audit_prompt = llm_concept_auditor._prompt(
                        context=authority.context,
                        script_text=script_text,
                        step=step,
                    )
                    audit_key = runtime.cache.key(
                        context=authority.context,
                        step=step,
                        script_text=script_text,
                        audit_prompt=audit_prompt,
                        environment_sha256=authority.environment_sha256,
                        auditor_identity=authority.auditor_identity(),
                        authority_bindings=authority.resolved_input_bindings,
                        validator_implementation_sha256=(
                            authority.auditor_implementation_sha256
                        ),
                    )
                    runtime.provider_budget.bind_reserved_category(
                        "concept_audit",
                        token=audit_key,
                    )
                    self.tokens_by_digest[audited_code_digest] = audit_key
                    cached_findings = runtime.cache.get(audit_key)
                    reservation_status = runtime.provider_budget.reservation_status(
                        "concept_audit",
                        token=audit_key,
                    )
                    if cached_findings is None and reservation_status in {
                        "attempted_incomplete",
                        "completed",
                        "released",
                    }:
                        raise ProviderCallBudgetReceiptError(
                            "Final concept audit has a durable paid/completed "
                            "reservation but no matching digest-bound cache; "
                            "refusing a duplicate provider call."
                        )
                    if cached_findings is not None:
                        cached_findings = _reclassify_llm_concept_findings(
                            findings=cached_findings,
                            context=authority.context,
                            script_text=script_text,
                        )
                        code_findings.extend(cached_findings)
                        self.completed_digests.add(audited_code_digest)
                        runtime.step_record["llm_concept_audit_cache_hits"] = (
                            int(
                                runtime.step_record.get("llm_concept_audit_cache_hits")
                                or 0
                            )
                            + 1
                        )
                    else:
                        llm_findings = llm_concept_auditor.audit(
                            context=authority.context,
                            script_text=script_text,
                            step=step,
                            provider_budget=runtime.provider_budget,
                        )
                        runtime.sync_provider_budget()
                        runtime.cache.put(audit_key, llm_findings)
                        code_findings.extend(llm_findings)
                        self.completed_digests.add(audited_code_digest)
        except ProviderCallBudgetError as exc:
            runtime.sync_provider_budget()
            receipt_error = isinstance(exc, ProviderCallBudgetReceiptError)
            code_findings.append(
                ValidationFinding(
                    validator=(
                        "provider_call_budget_receipt"
                        if receipt_error
                        else "provider_call_budget"
                    ),
                    severity="error",
                    message=(
                        f"Step {step.step_id} could not durably record its "
                        "provider call before concept approval."
                        if receipt_error
                        else f"Step {step.step_id} exhausted its shared LLM "
                        "provider-call budget before concept approval."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "category": getattr(exc, "category", None),
                        "limit": getattr(exc, "limit", runtime.provider_budget.limit),
                        "used": getattr(exc, "used", runtime.provider_budget.used),
                        "reason": str(exc),
                    },
                )
            )
        except BaseException:
            # Operator interrupts propagate, while a draft already rejected by
            # deterministic findings remains resumable.
            error_payloads = runtime.quarantine_error_payloads(code_findings)
            if error_payloads:
                try:
                    runtime.store_quarantined_draft(
                        run_dir=runtime.run_dir,
                        step_id=step.step_id,
                        code=script_text,
                        findings=error_payloads,
                    )
                except Exception:
                    pass
            raise
        if quarantine.pending_errors:
            existing_keys = {
                (finding.severity, finding_occurrence_identity(finding))
                for finding in code_findings
            }
            code_findings.extend(
                finding
                for finding in quarantine.pending_errors
                if finding.validator not in DETERMINISTIC_CODE_GATE_VALIDATORS
                and (
                    finding.severity,
                    finding_occurrence_identity(finding),
                )
                not in existing_keys
            )
        return code_findings


__all__ = [
    "ConceptAuditAuthority",
    "ConceptAuditCoordinator",
    "ConceptAuditRuntime",
    "ConceptQuarantineState",
]
