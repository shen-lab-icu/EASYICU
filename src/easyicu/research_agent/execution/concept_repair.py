"""Pre-execution concept repair loop with an explicit step boundary.

The concept audit owns the scientific findings for one candidate digest.  This
module owns the bounded orchestration that responds to those findings:
deterministic mechanical repair first, then a budgeted Coder repair, then an
honest fail-closed record or an authorized deterministic fallback.

All scientific choices remain injected by the execute host.  The loop returns
either an approved exact code digest or the already-persisted terminal record;
it cannot silently fall through with unaudited code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, List, MutableSequence, Optional, Sequence

from ..authority.evidence_store import sha256_of_bytes
from ..authority.provider_budget import ProviderCallBudgetReceiptError
from ..authority.step_attempt import CheckpointAuthority
from ..authority.step_capsule import StepAuthorityCapsuleError
from ..authority.step_runtime import StepAuthorityRuntimeError
from ..contracts.runtime import ValidationFinding
from ..gates.concept import finding_occurrence_identity
from ..gates.plausibility_obligation import (
    flag_only_plausibility_obligation_findings,
)
from ..gates.semantics import blocking_validator_findings
from ..repairs.coordination import StepRepairBudget
from ..repairs.reasons import (
    RepairPromptAuthority,
    RepairReason,
    repair_reason_for_finding,
    typed_repair_ticket,
)
from .code_hygiene import reorder_forward_references
from .concept_audit import ConceptAuditCoordinator
from .publication_figure import (
    SealedRendererState,
    sealed_renderer_code_seal_required,
)
from .runners.plausibility_receipt import host_plausibility_receipt_injected
from .standard_executor_diagnostics import standard_executor_failure_finding

MAX_DETERMINISTIC_CONCEPT_REPAIRS = 3


@dataclass(frozen=True, slots=True)
class ConceptRepairServices:
    """Host actions whose policy remains outside the repair-loop owner."""

    authorized_deterministic_repair: Callable[..., tuple[str, List[str]]]
    record_repair: Callable[..., None]
    deterministic_fallback_code: Callable[[str], Optional[str]]
    logical_budget_available: Callable[[Optional[str]], bool]
    repair_budget_available: Callable[[Optional[str]], bool]
    consume_llm_repair_budget: Callable[..., bool]
    remember_concept_constraints: Callable[[Sequence[ValidationFinding]], None]
    monotonic_constraint_ticket: Callable[[], List[Dict[str, Any]]]
    repair_with_capsule: Callable[..., str]
    python_repair_is_materially_changed: Callable[[str, str], bool]
    append_terminal_record: Callable[[List[Dict[str, Any]], Dict[str, Any]], None]
    flush_partial_manifest: Callable[..., None]


@dataclass(slots=True)
class ConceptRepairRequest:
    """One step's explicit mutable and immutable repair coordinates."""

    initial_code: str
    concept_audit: ConceptAuditCoordinator
    step_repair_budget: StepRepairBudget
    checkpoint_authority: CheckpointAuthority
    sealed_renderer_state: SealedRendererState
    standard_executor: Any
    coder_context: Any
    findings: MutableSequence[ValidationFinding]
    per_step_records: List[Dict[str, Any]]
    shared_lock: Any
    max_code_repair_attempts: int
    max_step_llm_repair_attempts: int
    services: ConceptRepairServices


@dataclass(frozen=True, slots=True)
class ConceptRepairResult:
    """Approved code or a terminal record already committed by this owner."""

    code: str
    concept_approved_code_sha256: Optional[str]
    sealed_renderer_authorized_code_sha256: Optional[str]
    terminal_record: Optional[Dict[str, Any]] = None

    @property
    def is_terminal(self) -> bool:
        return self.terminal_record is not None


def _commit_terminal(
    request: ConceptRepairRequest,
    *,
    new_findings: Sequence[ValidationFinding],
    progress_kind: str,
    progress_message: str,
    progress_status: str = "error",
) -> ConceptRepairResult:
    runtime = request.concept_audit.runtime
    step = request.concept_audit.authority.step
    with request.shared_lock:
        request.findings.extend(new_findings)
        request.services.append_terminal_record(
            request.per_step_records,
            runtime.step_record,
        )
        request.services.flush_partial_manifest()
    runtime.emit_progress(
        progress_kind,
        progress_message,
        status=progress_status,
        run_id=runtime.run_id,
        step_id=step.step_id,
        current_step=runtime.step_current,
        total_steps=runtime.total_steps,
    )
    return ConceptRepairResult(
        code=request.initial_code,
        concept_approved_code_sha256=None,
        sealed_renderer_authorized_code_sha256=None,
        terminal_record=dict(runtime.step_record),
    )


def run_concept_repair_loop(request: ConceptRepairRequest) -> ConceptRepairResult:
    """Return exact concept-approved code or persist one terminal outcome."""

    audit = request.concept_audit
    authority = audit.authority
    runtime = audit.runtime
    step = authority.step
    worker = runtime.worker_progress
    quarantine = runtime.quarantine_state
    record = runtime.step_record
    provider_budget = runtime.provider_budget
    services = request.services
    code = request.initial_code

    worker.concept_repair_attempts = 0
    worker.llm_repair_used = worker.critic_resume_repair_used
    worker.concept_audit_error_count = 0
    worker.deterministic_concept_repairs = 0
    worker.applied_concept_repair_names = []
    max_deterministic_repairs = MAX_DETERMINISTIC_CONCEPT_REPAIRS

    while True:
        if not quarantine.draft_active:
            code = reorder_forward_references(code)
            code = host_plausibility_receipt_injected(
                code,
                scope=authority.plausibility_scope,
                already_satisfied=not flag_only_plausibility_obligation_findings(
                    None,
                    script_text=code,
                    step=step,
                    scope=authority.plausibility_scope,
                ),
            )
        usage_findings = audit.findings_for_code(code, include_llm=False)
        record["usage_findings"] = [finding.model_dump() for finding in usage_findings]
        worker.concept_audit_error_count += sum(
            1
            for finding in usage_findings
            if finding.validator == runtime.usage_auditor.name
            and finding.severity == "error"
        )
        record["concept_audit_error_count"] = worker.concept_audit_error_count
        record["concept_repair_attempts"] = worker.concept_repair_attempts

        if not any(finding.severity == "error" for finding in usage_findings):
            approved_digest = sha256_of_bytes(code.encode("utf-8"))
            record["concept_approved_code_sha256"] = approved_digest
            sealed_digest: Optional[str] = None
            if sealed_renderer_code_seal_required(
                state=request.sealed_renderer_state,
                host_sealed_renderer=bool(
                    request.standard_executor is not None
                    and request.standard_executor.host_sealed_renderer
                ),
            ):
                sealed_digest = approved_digest
                record["sealed_renderer_authorized_code_sha256"] = sealed_digest
            if (
                quarantine.resumed_draft_used
                and quarantine.repair_materially_changed
                and not quarantine.superseded_by_fallback
            ):
                quarantine.repair_succeeded = True
                record["quarantined_repair_succeeded"] = True
            with request.shared_lock:
                request.findings.extend(usage_findings)
            return ConceptRepairResult(
                code=code,
                concept_approved_code_sha256=approved_digest,
                sealed_renderer_authorized_code_sha256=sealed_digest,
            )

        if request.sealed_renderer_state.repair_id is not None:
            terminal_finding = ValidationFinding(
                validator="sealed_renderer_authority",
                severity="error",
                message=(
                    "The authorized rendering-only adapter failed the "
                    "pre-execution deterministic concept gate; execution was "
                    "blocked without coder repair."
                ),
                detail={
                    "step_id": step.step_id,
                    "repair_id": request.sealed_renderer_state.repair_id,
                    "reason": "preexecution_concept_gate_failed",
                },
            )
            terminal_findings = [terminal_finding, *usage_findings]
            record.update(
                {
                    "status": "blocked_by_concept_audit",
                    "diagnostic_only": True,
                    "sealed_renderer_terminal_reason": (
                        "preexecution_concept_gate_failed"
                    ),
                    "contract_findings": [
                        finding.model_dump() for finding in terminal_findings
                    ],
                    "llm_repair_used": False,
                    "generation_mode": "fallback",
                }
            )
            return _commit_terminal(
                request,
                new_findings=terminal_findings,
                progress_kind="audit",
                progress_message=f"Sealed renderer blocked for {step.step_id}.",
            )

        if worker.deterministic_standard_executor_used:
            terminal_finding = standard_executor_failure_finding(
                step_record=record,
                step_id=step.step_id,
                reason="preexecution_concept_gate_failed",
                failure_phase="preexecution_concept_gate",
            )
            terminal_findings = [terminal_finding, *usage_findings]
            record.update(
                {
                    "status": "deterministic_standard_blocked",
                    "diagnostic_only": True,
                    "standard_executor_terminal_reason": (
                        "preexecution_concept_gate_failed"
                    ),
                    "contract_findings": [
                        finding.model_dump() for finding in terminal_findings
                    ],
                    "llm_repair_used": False,
                    "generation_mode": "deterministic_standard",
                }
            )
            return _commit_terminal(
                request,
                new_findings=terminal_findings,
                progress_kind="audit",
                progress_message=f"Trusted standard adapter blocked for {step.step_id}.",
            )

        if worker.deterministic_concept_repairs < max_deterministic_repairs:
            audit_error_messages = [
                value
                for finding in usage_findings
                if finding.severity == "error"
                for value in (
                    finding.message,
                    str((finding.detail or {}).get("reason") or ""),
                )
                if value
            ]
            audit_repair_reasons: List[RepairReason] = [
                repair_reason_for_finding(finding)
                for finding in usage_findings
                if finding.severity == "error"
            ]
            deterministic_code, repair_names = services.authorized_deterministic_repair(
                script_text=code,
                error_messages=audit_error_messages,
                repair_reasons=audit_repair_reasons,
                repair_findings=usage_findings,
                source="deterministic_concept_audit_repair",
            )
            if repair_names and deterministic_code != code:
                before_code = code
                worker.deterministic_concept_repairs += 1
                worker.applied_concept_repair_names.extend(repair_names)
                record["deterministic_concept_repairs"] = (
                    worker.deterministic_concept_repairs
                )
                record["applied_concept_repair_names"] = list(
                    worker.applied_concept_repair_names
                )
                record["deterministic_concept_repair_code_sha256"] = sha256_of_bytes(
                    deterministic_code.encode("utf-8")
                )
                for repair_name in repair_names:
                    services.record_repair(
                        repair_id=repair_name,
                        step_id=step.step_id,
                        trigger={
                            "gate": "concept_audit",
                            "audit_errors": audit_error_messages,
                        },
                        transformation=(
                            "deterministic_concept_audit_repair: rewrote a "
                            "mechanical ICU anti-pattern flagged as an error "
                            "by the static concept-audit gate"
                        ),
                        before_code=code,
                        after_code=deterministic_code,
                        selection_rule=(
                            "applied only because an error finding "
                            "objectively named the anti-pattern"
                        ),
                    )
                runtime.emit_progress(
                    "coder",
                    "Auto-repaired concept-audit anti-pattern "
                    f"({', '.join(repair_names)}) for {step.step_id}.",
                    run_id=runtime.run_id,
                    step_id=step.step_id,
                    current_step=runtime.step_current,
                    total_steps=runtime.total_steps,
                )
                code = deterministic_code
                if (
                    quarantine.draft_active
                    and services.python_repair_is_materially_changed(
                        before_code,
                        code,
                    )
                ):
                    quarantine.draft_active = False
                    quarantine.repair_materially_changed = True
                    quarantine.pending_errors = []
                    record["quarantined_repair_materially_changed"] = True
                continue

        if (
            worker.concept_repair_attempts >= request.max_code_repair_attempts
            or not services.repair_budget_available("concept")
            or provider_budget.exhausted
        ):
            if not services.logical_budget_available("concept"):
                record["step_llm_repair_budget_exhausted"] = True
                record["step_llm_repair_budget"] = request.max_step_llm_repair_attempts
            runtime.sync_provider_budget()
            fallback_code = services.deterministic_fallback_code("concept_audit")
            if fallback_code is not None:
                fallback_checkpoint_error: Optional[Exception] = None
                if quarantine.resumed_draft_used:
                    try:
                        checkpoint = runtime.store_quarantined_draft(
                            run_dir=runtime.run_dir,
                            step_id=step.step_id,
                            code=code,
                            findings=runtime.quarantine_error_payloads(usage_findings),
                        )
                        record["quarantined_draft_sha256"] = checkpoint.sha256
                        record["quarantined_draft_relative_path"] = (
                            checkpoint.relative_path
                        )
                        record["quarantine_checkpoint_is_latest_candidate"] = True
                    except Exception as checkpoint_exc:
                        fallback_checkpoint_error = checkpoint_exc
                with request.shared_lock:
                    if fallback_checkpoint_error is not None:
                        request.findings.append(
                            ValidationFinding(
                                validator="resume",
                                severity="warning",
                                message=(
                                    "Could not update the concept-draft checkpoint "
                                    "before deterministic fallback for step "
                                    f"{step.step_id}: {fallback_checkpoint_error}"
                                ),
                                detail={"step_id": step.step_id},
                            )
                        )
                    seen_occurrences = {
                        finding_occurrence_identity(finding)
                        for finding in request.findings
                    }
                    for finding in usage_findings:
                        if finding_occurrence_identity(finding) in seen_occurrences:
                            continue
                        if finding.severity == "error":
                            finding = finding.model_copy(
                                update={
                                    "severity": "warning",
                                    "message": (
                                        "[surfaced after fallback] " + finding.message
                                    ),
                                }
                            )
                        request.findings.append(finding)
                if quarantine.resumed_draft_used:
                    quarantine.draft_active = False
                    quarantine.pending_errors = []
                    quarantine.repair_succeeded = False
                    quarantine.superseded_by_fallback = True
                    record["quarantined_repair_succeeded"] = False
                    record["quarantine_superseded_by_fallback"] = True
                code = fallback_code
                continue

            record["status"] = "blocked_by_concept_audit"
            checkpoint_error: Optional[Exception] = None
            if not quarantine.superseded_by_fallback:
                try:
                    checkpoint = runtime.store_quarantined_draft(
                        run_dir=runtime.run_dir,
                        step_id=step.step_id,
                        code=code,
                        findings=runtime.quarantine_error_payloads(usage_findings),
                    )
                    record["quarantined_draft_sha256"] = checkpoint.sha256
                    record["quarantined_draft_relative_path"] = checkpoint.relative_path
                    record["quarantined_requires_repair"] = True
                    record["quarantine_checkpoint_is_latest_candidate"] = True
                except Exception as checkpoint_exc:
                    checkpoint_error = checkpoint_exc
            block_errors = [
                {"validator": finding.validator, "message": finding.message}
                for finding in usage_findings
                if finding.severity == "error"
            ]
            offending_lines = [
                line.strip()
                for line in code.splitlines()
                if any(
                    token in line
                    for token in ("fillna(0)", "fillna(0.0)", ".mean()", "dropna(")
                )
            ][:12]
            remedies = [
                "Add the violated ICU rule as an explicit coder/planner constraint "
                "and re-run this question (e.g. 'do not impute a lab with 0; "
                "handle missingness with complete-case or a declared imputation + "
                "missingness indicator').",
                "Use a stronger model for this question — the block was triggered "
                "by generated code, not by the cohort or the question itself.",
                "Accept the withheld result: diagnostic_only is a valid outcome. "
                "The fail-closed gate declined to report an analysis it judged "
                "unsafe; nothing wrong was published.",
            ]
            record["concept_audit_block"] = {
                "step_id": step.step_id,
                "errors": block_errors,
                "deterministic_repairs_applied": list(
                    worker.applied_concept_repair_names
                ),
                "llm_repair_attempts": worker.concept_repair_attempts,
                "offending_code_lines": offending_lines,
                "candidate_remedies": remedies,
            }
            try:
                ticket = [
                    f"# Concept-audit block — step `{step.step_id}`",
                    "",
                    "The static ICU concept-audit gate blocked this step before "
                    "execution and auto-repair could not clear it, so the run "
                    "withheld this analysis (`diagnostic_only`). This is the "
                    "fail-closed safety system working — but here is how to move "
                    "it forward.",
                    "",
                    "## What was flagged (objective errors)",
                    *[
                        f"- **{error['validator']}**: {error['message']}"
                        for error in block_errors
                    ],
                    "",
                    "## Repair already attempted",
                    "- deterministic: "
                    f"{worker.applied_concept_repair_names or 'none matched'}",
                    f"- LLM coder repair attempts: {worker.concept_repair_attempts}",
                    "",
                    "## Offending code lines",
                    "```python",
                    *(offending_lines or ["(no obvious anti-pattern line)"]),
                    "```",
                    "",
                    "## How to resolve (pick one — your analytical choice)",
                    *[
                        f"{index + 1}. {remedy}"
                        for index, remedy in enumerate(remedies)
                    ],
                    "",
                ]
                (runtime.run_dir / f"concept_audit_block_{step.step_id}.md").write_text(
                    "\n".join(ticket),
                    encoding="utf-8",
                )
            except Exception:
                pass
            terminal_findings = list(usage_findings)
            if checkpoint_error is not None:
                terminal_findings.append(
                    ValidationFinding(
                        validator="resume",
                        severity="warning",
                        message=(
                            "Could not update the blocked concept-draft checkpoint "
                            f"for step {step.step_id}: {checkpoint_error}"
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
            return _commit_terminal(
                request,
                new_findings=terminal_findings,
                progress_kind="audit",
                progress_message=(
                    f"Concept audit blocked {step.step_id}; repair ticket written."
                ),
            )

        blocking_findings = blocking_validator_findings(usage_findings)
        audit_log = "\n".join(
            (
                f"{finding.severity.upper()}: {finding.message}"
                + (
                    "\nDETAIL (diagnostic mirror only): "
                    + json.dumps(finding.detail, ensure_ascii=False, sort_keys=True)
                    if finding.detail
                    else ""
                )
            )
            for finding in blocking_findings
        )
        structured_ticket = typed_repair_ticket(blocking_findings)
        current_authority = RepairPromptAuthority.create(typed_ticket=structured_ticket)
        repair_authority = RepairPromptAuthority.create(
            typed_ticket=[
                *structured_ticket,
                *services.monotonic_constraint_ticket(),
            ]
        )
        repair_log = (
            "Static concept audit blocked this script before execution. Fix all "
            "ICU-rule violations.\n\nHUMAN-READABLE FINDINGS (diagnostic mirror "
            "only):\n" + audit_log
        )
        worker.concept_repair_attempts += 1
        if not services.consume_llm_repair_budget(
            "concept",
            before_code=code,
            repair_ticket=repair_log,
            repair_authority=repair_authority,
            current_repair_authority=current_authority,
            provider_category="concept_repair",
            failure_status="concept_failed",
        ):
            raise AssertionError("LLM repair budget changed without mutation")
        record["concept_repair_attempts"] = worker.concept_repair_attempts
        runtime.emit_progress(
            "coder",
            f"Repairing concept-audit violation for {step.step_id}.",
            run_id=runtime.run_id,
            step_id=step.step_id,
            current_step=runtime.step_current,
            total_steps=runtime.total_steps,
            repair_attempts=worker.concept_repair_attempts,
        )
        services.remember_concept_constraints(blocking_findings)
        try:
            repaired_code = services.repair_with_capsule(
                failure_status="concept_failed",
                context=request.coder_context,
                step=step,
                code=code,
                run_log=repair_log,
                repair_authority=repair_authority,
                current_repair_authority=current_authority,
                attempt=worker.concept_repair_attempts,
                provider_budget=provider_budget,
                provider_category="concept_repair",
                logical_repair_attempt_id=request.step_repair_budget.llm_repair_attempts,
            )
            runtime.sync_provider_budget()
            if (
                quarantine.draft_active
                and not services.python_repair_is_materially_changed(
                    code,
                    repaired_code,
                )
            ):
                request.checkpoint_authority.reject_completed_repair_candidate(
                    repaired_code,
                    reason="quarantined_repair_semantic_noop",
                )
                no_op_finding = ValidationFinding(
                    validator="resume",
                    severity="error",
                    message=(
                        "Quarantined concept-draft repair returned no material "
                        f"Python change for step {step.step_id}; the pending "
                        "concept errors remain binding."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "quarantined_draft_sha256": record.get(
                            "quarantined_draft_sha256"
                        ),
                        "repair_attempt": worker.concept_repair_attempts,
                        "semantic_noop": True,
                    },
                )
                if not any(
                    finding.message == no_op_finding.message
                    for finding in quarantine.pending_errors
                ):
                    quarantine.pending_errors.append(no_op_finding)
                record["quarantined_repair_noop_count"] = (
                    int(record.get("quarantined_repair_noop_count") or 0) + 1
                )
                quarantine.repair_succeeded = False
                record["quarantined_repair_succeeded"] = False
                continue
            code = repaired_code
            worker.llm_repair_used = True
            if quarantine.draft_active:
                quarantine.draft_active = False
                quarantine.repair_materially_changed = True
                quarantine.pending_errors = []
                record["quarantined_repair_materially_changed"] = True
        except (
            ProviderCallBudgetReceiptError,
            StepAuthorityRuntimeError,
            StepAuthorityCapsuleError,
        ):
            raise
        except BaseException as exc:
            runtime.sync_provider_budget()
            checkpoint_error: Optional[Exception] = None
            try:
                checkpoint = runtime.store_quarantined_draft(
                    run_dir=runtime.run_dir,
                    step_id=step.step_id,
                    code=code,
                    findings=runtime.quarantine_error_payloads(usage_findings),
                )
                record["quarantined_draft_sha256"] = checkpoint.sha256
                record["quarantined_draft_relative_path"] = checkpoint.relative_path
                record["quarantined_requires_repair"] = True
            except Exception as checkpoint_exc:
                checkpoint_error = checkpoint_exc
            if not isinstance(exc, Exception):
                raise
            fallback_code = services.deterministic_fallback_code(
                "concept_repair_failed"
            )
            if fallback_code is not None:
                quarantine.draft_active = False
                quarantine.pending_errors = []
                quarantine.repair_succeeded = False
                if quarantine.resumed_draft_used:
                    quarantine.superseded_by_fallback = True
                    record["quarantined_repair_succeeded"] = False
                    record["quarantine_superseded_by_fallback"] = True
                code = fallback_code
                continue
            terminal_findings = list(usage_findings)
            if checkpoint_error is not None:
                terminal_findings.append(
                    ValidationFinding(
                        validator="resume",
                        severity="warning",
                        message=(
                            "Could not preserve the rejected concept-audit draft "
                            f"for step {step.step_id}: {checkpoint_error}"
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
            terminal_findings.append(
                ValidationFinding(
                    validator="coder",
                    severity="error",
                    message=(
                        "Coder repair failed after concept audit for step "
                        f"{step.step_id}: {exc}"
                    ),
                )
            )
            record["status"] = "repair_failed"
            return _commit_terminal(
                request,
                new_findings=terminal_findings,
                progress_kind="coder",
                progress_message=f"Concept-audit repair failed for {step.step_id}.",
            )
