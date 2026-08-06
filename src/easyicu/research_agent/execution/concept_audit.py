"""Step-scoped concept-audit execution and quarantine coordination.

This module owns the operational boundary around deterministic concept gates,
the optional digest-bound LLM audit, provider-budget accounting, and retirement
of stored quarantine findings.  It does not choose a cohort, exposure, outcome,
method, or estimand.  Policy functions remain host-owned dependencies supplied
by the execute layer, which keeps this module independent from
``execution.phase`` and prevents a reverse import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Sequence

from ..authority.plausibility import FlagOnlyPlausibilityScope
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
from ..contracts.runtime import ValidationFinding
from ..authority.evidence_store import sha256_of_bytes
from ..authority.provider_budget import (
    ProviderCallBudgetError,
    ProviderCallBudgetReceiptError,
    StepProviderCallBudget,
)
from ..authority.run_input import canonical_sha256
from ..schema import AnalysisStep, ResearchContext
from ..authority.step_attempt import StepAttemptState
from ..authority.step_runtime import read_concept_audit_findings
from .concept_reaudit import DETERMINISTIC_CONCEPT_REAUDIT_BUDGET_ISSUE_CODE
from .step_worker_state import StepWorkerProgress

_RETRYABLE_FINAL_AUDIT_ISSUE_CODE = "llm_concept_audit_provider_failure"
_NON_SEMANTIC_AUDIT_VALIDATORS = frozenset(
    {"provider_call_budget", "provider_call_budget_receipt"}
)


def verified_capsule_concept_audit_replay(
    verified: Any,
    *,
    run_dir: Path,
    auditor_identity_sha256: str,
    environment_sha256: str,
    validator_implementation_sha256: str,
) -> Optional[tuple[list[ValidationFinding], str]]:
    """Return exact sealed findings only under current audit dependencies.

    Provider-budget and receipt failures describe control-plane availability,
    not the candidate's scientific semantics. They must never be replayed as
    a completed concept review, including for capsules written by older code.
    """

    audit = verified.capsule.concept_audit
    if (
        audit is None
        or audit.auditor_identity_sha256 != auditor_identity_sha256
        or audit.environment_sha256 != environment_sha256
        or audit.validator_implementation_sha256 != validator_implementation_sha256
    ):
        return None
    findings = read_concept_audit_findings(verified, run_dir=run_dir)
    if any(finding.validator in _NON_SEMANTIC_AUDIT_VALIDATORS for finding in findings):
        return None
    return findings, audit.audit_key


def _retryable_final_audit_provider_failure(
    findings: Sequence[ValidationFinding],
    *,
    step_id: str,
) -> bool:
    """Return whether one exact quarantine contains only transport failure.

    The quarantine draft itself binds these findings to the candidate digest.
    A semantic finding, invalid response, or finding from another step must
    never authorize another paid final-audit call.
    """

    def _is_retryable(finding: ValidationFinding) -> bool:
        detail = finding.detail or {}
        issue_code = str(detail.get("issue_code") or "")
        finding_step_id = str(detail.get("step_id") or "")
        if finding.severity != "error" or finding_step_id != step_id:
            return False
        if finding.validator == "llm_concept_auditor":
            return issue_code == _RETRYABLE_FINAL_AUDIT_ISSUE_CODE
        if (
            finding.validator != "provider_call_budget"
            or issue_code
            != DETERMINISTIC_CONCEPT_REAUDIT_BUDGET_ISSUE_CODE
            or detail.get("category") != "concept_audit"
        ):
            return False
        used = detail.get("used")
        limit = detail.get("limit")
        return (
            not isinstance(used, bool)
            and isinstance(used, int)
            and not isinstance(limit, bool)
            and isinstance(limit, int)
            and used >= limit
        )

    return bool(findings) and all(_is_retryable(finding) for finding in findings)


def _final_audit_continuation_allowed(
    *,
    reservation_status: str,
    quarantine_findings: Sequence[ValidationFinding],
    step_id: str,
) -> bool:
    """Authorize a paid continuation without erasing its prior attempt."""

    return reservation_status == "attempted_incomplete" and (
        _retryable_final_audit_provider_failure(
            quarantine_findings,
            step_id=step_id,
        )
    )


def _defer_provider_failure_until_final_audit(
    *,
    include_llm: bool,
    reserved_final_category: object,
    quarantine_findings: Sequence[ValidationFinding],
    step_id: str,
) -> bool:
    """Keep a proven transport failure from blocking pre-execution gates."""

    return (
        not include_llm
        and reserved_final_category == "concept_audit"
        and _retryable_final_audit_provider_failure(
            quarantine_findings,
            step_id=step_id,
        )
    )


def _retire_completed_provider_failure_continuation(
    quarantine: ConceptQuarantineState,
    *,
    step_id: str,
    fresh_findings: Sequence[ValidationFinding],
) -> bool:
    """Retire an old transport failure only after a later audit returned."""

    if not _retryable_final_audit_provider_failure(
        quarantine.pending_errors,
        step_id=step_id,
    ) or any(
        str((finding.detail or {}).get("issue_code") or "")
        == _RETRYABLE_FINAL_AUDIT_ISSUE_CODE
        for finding in fresh_findings
    ):
        return False
    quarantine.pending_errors = []
    quarantine.draft_active = False
    return True


class ConceptQuarantineState:
    """Mutable per-step concept-audit quarantine state.

    The object is deliberately step-local.  Concurrent workers create separate
    instances and therefore cannot share quarantined findings or retirement
    flags.
    """

    __slots__ = (
        "draft_active",
        "policy_superseded",
        "policy_reclassified_findings",
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
        self.policy_reclassified_findings: List[ValidationFinding] = []
        self.deterministic_revalidated: bool = False
        self.pending_errors: List[ValidationFinding] = []
        self.resumed_draft_used: bool = False
        self.repair_materially_changed: bool = False
        self.repair_succeeded: bool = False
        self.superseded_by_fallback: bool = False


@dataclass(frozen=True, slots=True)
class QuarantineRetirementDecision:
    """Exact-digest retirement result for heterogeneous stored findings."""

    remaining_errors: tuple[ValidationFinding, ...]
    deterministic_provenance: tuple[Mapping[str, Any], ...] = ()
    policy_reclassified_findings: tuple[ValidationFinding, ...] = ()
    policy_provenance: tuple[Mapping[str, Any], ...] = ()


def _quarantine_retirement_decision(
    *,
    prior_errors: Sequence[ValidationFinding],
    current_findings: Sequence[ValidationFinding],
    context: ResearchContext,
    script_text: str,
    quarantined_script_sha256: str,
) -> QuarantineRetirementDecision:
    """Retire independently provable subsets without discarding the rest.

    A quarantine can contain findings from several validator generations.  A
    current deterministic gate can retire its own stale findings while the
    host policy reclassifier independently retires an LLM finding.  Requiring
    every stored finding to belong to one family makes that mixed checkpoint
    impossible to retire and can incorrectly send already-approved code back
    into an exhausted repair loop.

    Each proof remains exact-digest and fail-closed.  An unproved subset stays
    in ``remaining_errors`` and therefore continues to block execution.
    """

    deterministic_errors = tuple(
        finding
        for finding in prior_errors
        if finding.validator in DETERMINISTIC_CODE_GATE_VALIDATORS
    )
    policy_errors = tuple(
        finding
        for finding in prior_errors
        if finding.validator not in DETERMINISTIC_CODE_GATE_VALIDATORS
    )

    deterministic_provenance: tuple[Mapping[str, Any], ...] = ()
    if deterministic_errors:
        resolved = quarantined_deterministic_errors_resolved_by_current_gate(
            prior_errors=deterministic_errors,
            current_findings=current_findings,
            script_text=script_text,
            quarantined_script_sha256=quarantined_script_sha256,
        )
        if resolved is not None:
            deterministic_provenance = tuple(resolved)

    policy_reclassified: tuple[ValidationFinding, ...] = ()
    policy_provenance: tuple[Mapping[str, Any], ...] = ()
    if policy_errors:
        supersession = quarantined_errors_superseded_by_current_policy(
            prior_errors=policy_errors,
            current_findings=current_findings,
            context=context,
            script_text=script_text,
            quarantined_script_sha256=quarantined_script_sha256,
        )
        if supersession is not None:
            reclassified, provenance = supersession
            policy_reclassified = tuple(reclassified)
            policy_provenance = tuple(provenance)

    remaining: list[ValidationFinding] = []
    if deterministic_errors and not deterministic_provenance:
        remaining.extend(deterministic_errors)
    if policy_errors and not policy_provenance:
        remaining.extend(policy_errors)
    return QuarantineRetirementDecision(
        remaining_errors=tuple(remaining),
        deterministic_provenance=deterministic_provenance,
        policy_reclassified_findings=policy_reclassified,
        policy_provenance=policy_provenance,
    )


@dataclass(frozen=True, slots=True)
class ConceptAuditAuthority:
    """Immutable scientific and audit identity for one planned step."""

    context: ResearchContext
    step: AnalysisStep
    resolved_input_bindings: Mapping[str, Any]
    plausibility_scope: FlagOnlyPlausibilityScope
    environment_sha256: str
    auditor_implementation_sha256: str
    auditor_identity: Callable[[], str]
    enable_llm_audit: bool
    # The study's declared endpoint, from the locked plan. Part of the step's
    # scientific identity: without it the auditor judged the script against its
    # own reading of the research question and blocked steps for contradicting a
    # "planner-required" censoring column that appears in no plan.
    study_endpoint: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        self.plausibility_scope.require_step(self.step.step_id)


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
    authorize_deterministic_reaudit: Callable[..., bool]


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
            resolved_input_bindings=authority.resolved_input_bindings,
            plausibility_scope=authority.plausibility_scope,
        )
        deterministic_errors = [
            finding
            for finding in code_findings
            if finding.severity == "error"
            and finding.validator != "llm_concept_auditor"
        ]
        if quarantine.pending_errors:
            retirement = _quarantine_retirement_decision(
                prior_errors=quarantine.pending_errors,
                current_findings=code_findings,
                context=authority.context,
                script_text=script_text,
                quarantined_script_sha256=str(
                    runtime.step_record.get("quarantined_draft_sha256") or ""
                ),
            )
            if retirement.deterministic_provenance:
                quarantine.deterministic_revalidated = True
                runtime.step_record[
                    "quarantine_deterministic_revalidation_succeeded"
                ] = True
                runtime.step_record["quarantine_deterministic_revalidated_findings"] = (
                    list(retirement.deterministic_provenance)
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
            if retirement.policy_provenance:
                existing_keys = {
                    (finding.severity, finding_occurrence_identity(finding))
                    for finding in code_findings
                }
                code_findings.extend(
                    finding
                    for finding in retirement.policy_reclassified_findings
                    if (
                        finding.severity,
                        finding_occurrence_identity(finding),
                    )
                    not in existing_keys
                )
                quarantine.policy_superseded = True
                quarantine.policy_reclassified_findings = list(
                    retirement.policy_reclassified_findings
                )
                runtime.step_record["quarantine_policy_superseded"] = True
                runtime.step_record["quarantine_policy_superseded_findings"] = list(
                    retirement.policy_provenance
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
            quarantine.pending_errors = list(retirement.remaining_errors)
            if not quarantine.pending_errors and (
                retirement.deterministic_provenance or retirement.policy_provenance
            ):
                quarantine.draft_active = False
        try:
            audited_code_digest = sha256_of_bytes(script_text.encode("utf-8"))
            sealed_capsule_audit = (
                runtime.step_attempt_state.capsule_audit_findings_by_digest.get(
                    audited_code_digest
                )
            )
            if include_llm and sealed_capsule_audit is not None:
                sealed_findings, audit_key = sealed_capsule_audit
                if quarantine.policy_reclassified_findings:
                    reclassified_by_identity = {
                        (finding.validator, finding.message): finding
                        for finding in quarantine.policy_reclassified_findings
                    }
                    sealed_findings = [
                        reclassified_by_identity.get(
                            (finding.validator, finding.message),
                            finding,
                        )
                        for finding in sealed_findings
                    ]
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
                        study_endpoint=authority.study_endpoint,
                    )
                    audit_key = runtime.cache.key(
                        context=authority.context,
                        step=step,
                        script_text=script_text,
                        audit_prompt=audit_prompt,
                        environment_sha256=authority.environment_sha256,
                        auditor_identity=authority.auditor_identity(),
                        authority_bindings={
                            "resolved_input_bindings": (
                                authority.resolved_input_bindings
                            ),
                            "flag_only_plausibility_scope": (
                                authority.plausibility_scope.to_dict()
                            ),
                        },
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
                    if (
                        cached_findings is None
                        and not runtime.provider_budget.can_consume("concept_audit")
                        and runtime.authorize_deterministic_reaudit(
                            token=audit_key,
                            code_sha256=audited_code_digest,
                        )
                    ):
                        runtime.sync_provider_budget()
                    reservation_status = runtime.provider_budget.reservation_status(
                        "concept_audit",
                        token=audit_key,
                    )
                    continuation_allowed = _final_audit_continuation_allowed(
                        reservation_status=reservation_status,
                        quarantine_findings=quarantine.pending_errors,
                        step_id=step.step_id,
                    )
                    if (
                        cached_findings is None
                        and reservation_status
                        in {
                            "attempted_incomplete",
                            "completed",
                            "released",
                        }
                        and not continuation_allowed
                    ):
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
                        if _retire_completed_provider_failure_continuation(
                            quarantine,
                            step_id=step.step_id,
                            fresh_findings=cached_findings,
                        ):
                            runtime.step_record[
                                "concept_audit_provider_failure_continued"
                            ] = True
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
                            study_endpoint=authority.study_endpoint,
                        )
                        runtime.sync_provider_budget()
                        runtime.cache.put(audit_key, llm_findings)
                        code_findings.extend(llm_findings)
                        self.completed_digests.add(audited_code_digest)
                        if _retire_completed_provider_failure_continuation(
                            quarantine,
                            step_id=step.step_id,
                            fresh_findings=llm_findings,
                        ):
                            runtime.step_record[
                                "concept_audit_provider_failure_continued"
                            ] = True
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
        defer_provider_failure = _defer_provider_failure_until_final_audit(
            include_llm=include_llm,
            reserved_final_category=(
                runtime.provider_budget.snapshot().get("reserved_final_category")
            ),
            quarantine_findings=quarantine.pending_errors,
            step_id=step.step_id,
        )
        if quarantine.pending_errors and not defer_provider_failure:
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
        elif defer_provider_failure:
            runtime.step_record["concept_audit_provider_failure_deferred"] = True
        return code_findings


__all__ = [
    "ConceptAuditAuthority",
    "ConceptAuditCoordinator",
    "ConceptAuditRuntime",
    "ConceptQuarantineState",
]
