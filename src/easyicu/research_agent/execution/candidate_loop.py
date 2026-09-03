"""Candidate execution and repair state machine.

This module owns the digest-audit, execution, visual, contract, and runtime
repair transitions for one analysis-step candidate. The execute-phase host
injects run-scoped collaborators through immutable frames; stages exchange only
the small mutable state object defined here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from ..repairs.source import (
    _deterministic_runner_repair,
    _deterministic_summary_repair,
    deterministic_contract_repair,
)
from ..repairs.attempt_record import record_deterministic_runner_repair_attempt
from .code_hygiene import reorder_forward_references
from .failure_classification import classify_runtime_failure
from .concept_audit import ConceptQuarantineState
from .concept_repair import MAX_DETERMINISTIC_CONCEPT_REPAIRS
from ..authority.plausibility import StepPlausibilityAuthority
from ..authority.execution_input import ExecutionInputAuthorityState
from ..contracts.runtime import ValidationFinding
from ..gates.plausibility_obligation import (
    flag_only_plausibility_obligation_findings as _flag_only_plausibility_obligation_findings,
)
from .runners.plausibility_receipt import (
    host_plausibility_receipt_injected as _host_plausibility_receipt_injected,
)
from ..authority.evidence_store import sha256_of_bytes, sha256_of_file
from ..authority.typed_binding import (
    _write_host_input_binding_receipts,
    host_authored_generation_mode,
    host_owns_input_binding_receipts,
)
from ..gates.contract import (
    _post_canonicalization_figure_findings,
    _step_deterministic_contract_findings,
)
from .figure_preparation import (
    _figure_contract_source_data_canonicalization_candidate,
    _install_figure_contract_source_data_canonicalization,
)
from .figure_plan_binding import validate_step_planned_figure_contract_binding
from .final_validation import _demote_step_contract_for_primary_runner
from .publication_figure import validate_and_record_sealed_renderer_receipt
from .host_services import ExecutePhaseHost
from .output_files import (
    _clear_output_dir,
    bind_primary_output,
    normalize_typed_statistic_sidecars,
)
from ..gates.visual import (
    VisualRepairAction,
    collect_visual_gate_result,
    decide_visual_repair,
)
from ..repairs.reasons import (
    RepairPromptAuthority,
    repair_reason_for_finding,
    typed_repair_ticket,
)
from ..contracts.step_families import effect_output_authorized
from ..gates.step_repair import _step_contract_repair_guidance
from ..orchestration.resume import store_quarantined_concept_draft
from ..repair_registry import is_sealed_renderer_repair
from ..authority.provider_budget import ProviderCallBudgetReceiptError
from .step_attempt_bootstrap import RAW_UNIVERSE_EXECUTION_ROLE
from .cohort_routing import (
    PreselectionUniverseOwnerCapability,
    preselection_universe_capability,
)
from .phase_support import _robustness_sensitivity_runner_owns_step
from ..authority.run_input import canonical_sha256
from ..authority.step_capsule import (
    StepAuthorityCapsuleError,
    load_verified_step_authority_capsule,
)
from ..authority.step_runtime import (
    StepAuthorityRuntimeError,
    current_execution_runtime_sha256,
    execution_context_sha256,
    materialize_sealed_run_result,
    persist_candidate_code,
    seal_concept_audit_capsule,
    seal_deterministic_candidate,
    seal_execution_capsule,
)
from ..authority.step_attempt import StepAttemptState
from .step_execution import LockedStepExecutionRequest, StepExecutor
from ..repairs.summary import salvage_step_summary

_FIGURE_CONTRACT_SOURCE_DATA_SCHEMA_REPAIR_ID = "figure_contract_source_data_schema_v1"


class _CandidateLoopAction(str, Enum):
    """Control transfer requested by one candidate-loop stage."""

    PROCEED = "proceed"
    CONTINUE = "continue"
    BREAK = "break"
    RETURN = "return"


@dataclass(frozen=True, slots=True)
class _CandidateLoopHost:
    """Run-scoped collaborators shared by all candidate-loop stages.

    The loop used to close over these names implicitly.  Keeping them in one
    immutable frame makes the dependency surface inspectable while preserving
    the existing mutable owners (findings, evidence and step records).
    """

    _authorize_automatic_repair: Callable[..., Any]
    _append_terminal_step_record: Callable[..., Any]
    _automatic_repair_authorized: Callable[..., Any]
    _contract_repair_log: Callable[..., Any]
    _execution_input_authority_integrity_finding: Callable[..., Any]
    _flush_partial_manifest: Callable[..., Any]
    _fresh_plausibility_receipt_findings: Callable[..., Any]
    _locked_measurement_data_quality_issues: Callable[..., Any]
    _python_repair_is_materially_changed: Callable[..., Any]
    _record_repair: Callable[..., Any]
    _remove_standard_executor_pending_artifacts: Callable[..., Any]
    _unowned_sealed_authority_markers: Callable[..., Any]
    cohort_path: Path
    concept_audit_environment_sha256: str
    context: Any
    cross_step_cohort_lock_validator: Any
    cross_step_reconciliation_trace_validator: Any
    cross_step_registered_output_validator: Any
    cross_step_source_status_validator: Any
    emit_progress: Callable[..., Any]
    evidence: Any
    figure_contract_validator: Any
    figure_source_validator: Any
    findings: List[ValidationFinding]
    llm_concept_auditor_identity_sha256: str
    llm_concept_auditor_implementation_sha256: str
    llm_signature: str
    per_step_records: List[Dict[str, Any]]
    pipeline: ExecutePhaseHost
    plan: Any
    primary_model_contract_validator: Any
    prompt_version: str
    run_dir: Path
    run_id: str
    run_input_authority_state: ExecutionInputAuthorityState
    runner: Any
    shared_lock: Any
    step_executor: StepExecutor
    step_summary_fraction_validator: Any
    step_summary_integrity_validator: Any
    total_steps: int
    universe_path: Path


@dataclass(frozen=True, slots=True)
class _CandidateLoopAttempt:
    """Step-attempt authority, budgets and callbacks consumed by the loop."""

    _authorized_deterministic_concept_repair: Callable[..., Any]
    _consume_llm_repair_budget: Callable[..., Any]
    _deterministic_fallback_code: Callable[..., Any]
    _llm_repair_budget_available: Callable[..., Any]
    _logical_llm_repair_budget_available: Callable[..., Any]
    _monotonic_concept_constraint_log: Callable[..., Any]
    _monotonic_concept_constraint_ticket: Callable[..., Any]
    _quarantine_error_payloads: Callable[..., Any]
    _remember_concept_constraints: Callable[..., Any]
    _repair_with_capsule: Callable[..., Any]
    _sync_provider_budget: Callable[..., Any]
    checkpoint_authority: Any
    coder_context: Any
    concept_audit: Any
    is_trajectory_stability_standard: bool
    local_runtime_state: Any
    plausibility_authority: StepPlausibilityAuthority
    provider_budget: Any
    quarantine_state: ConceptQuarantineState
    resolved_input_bindings: Mapping[str, Any]
    resolved_input_evidence_ids: Sequence[str]
    resolved_inputs_path: Path
    resolved_inputs_sha256: str
    sealed_renderer_authorized_code_sha256: Optional[str]
    sealed_renderer_state: Any
    standard_executor: Any
    step: Any
    step_attempt_state: StepAttemptState
    step_current: int
    step_execution_cohort_path: Path
    step_record: Dict[str, Any]
    step_repair_budget: Any
    worker_progress: Any


@dataclass(slots=True)
class _CandidateLoopState:
    """Mutable values that legitimately cross candidate-loop stage boundaries."""

    code: str
    concept_approved_code_digest: Optional[str]
    deterministic_contract_approved_code_digest: Optional[str]
    final_concept_gate_approved_code_digest: Optional[str]
    standard_executor_terminal_block: bool = False
    standard_executor_terminal_reason: Optional[str] = None
    standard_executor_terminal_summary: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )
    standard_executor_terminal_findings: List[ValidationFinding] = dataclass_field(
        default_factory=list
    )
    candidate_code_digest: str = ""
    final_llm_audit_due: bool = False
    usage_findings: List[ValidationFinding] = dataclass_field(default_factory=list)
    current_generation_mode: str = ""
    execution_timeout_seconds: int = 0
    run_result: Any = None
    executed_code_digest: str = ""
    script_record: Any = None
    log_path: Optional[Path] = None
    visual_step_summary: Dict[str, Any] = dataclass_field(default_factory=dict)
    visual_gate: Any = None
    early_contract_findings: List[ValidationFinding] = dataclass_field(
        default_factory=list
    )
    early_contract_errors: List[ValidationFinding] = dataclass_field(
        default_factory=list
    )


# Candidate-loop stage implementations are generated below from the previously
# characterized contiguous blocks.  Each stage owns one responsibility and
# communicates only through the typed frames above.
def _candidate_concept_audit_transition(
    host: _CandidateLoopHost, attempt: _CandidateLoopAttempt, state: _CandidateLoopState
) -> _CandidateLoopAction:
    if (
        state.candidate_code_digest != state.concept_approved_code_digest
        or state.final_llm_audit_due
    ):
        # Every mutation still returns through deterministic semantic and
        # mechanical gates before execution.  The LLM concept auditor is
        # invoked only for the exact digest whose local run and early
        # deterministic contracts already passed, preventing runtime- or
        # contract-broken drafts from consuming repeated audit calls.
        state.usage_findings = attempt.concept_audit.findings_for_code(
            state.code,
            include_llm=state.final_llm_audit_due,
        )
        attempt.step_record["usage_findings"] = [
            finding.model_dump() for finding in state.usage_findings
        ]
        post_mutation_errors = [
            finding for finding in state.usage_findings if finding.severity == "error"
        ]
        if post_mutation_errors:
            if (
                state.final_llm_audit_due
                and attempt.step_attempt_state.coordinates is not None
                and attempt.step_attempt_state.current_capsule_ref is not None
                and not any(
                    str((finding.detail or {}).get("issue_code") or "")
                    in {
                        "llm_concept_audit_provider_failure",
                        "llm_concept_audit_response_invalid",
                    }
                    for finding in state.usage_findings
                )
            ):
                current_authority = load_verified_step_authority_capsule(
                    host.run_dir,
                    ref=attempt.step_attempt_state.current_capsule_ref,
                    expected_step_id=attempt.step.step_id,
                )
                if (
                    current_authority.capsule.concept_audit is None
                    or attempt.step_record.get("step_authority_audit_cache_miss")
                    == "audit_identity_drift"
                ):
                    blocked_audit_key = attempt.concept_audit.tokens_by_digest.get(
                        state.candidate_code_digest
                    ) or canonical_sha256(
                        {
                            "schema": ("easyicu.capsule_blocked_concept_audit/1"),
                            "step_id": attempt.step.step_id,
                            "code_sha256": state.candidate_code_digest,
                            "findings": [
                                finding.model_dump(mode="json")
                                for finding in state.usage_findings
                            ],
                        }
                    )
                    blocked_ref = seal_concept_audit_capsule(
                        attempt.step_attempt_state.coordinates,
                        parent_ref=attempt.step_attempt_state.current_capsule_ref,
                        findings=state.usage_findings,
                        audit_key=blocked_audit_key,
                        auditor_identity_sha256=(
                            host.llm_concept_auditor_identity_sha256
                        ),
                        environment_sha256=(host.concept_audit_environment_sha256),
                        validator_implementation_sha256=(
                            host.llm_concept_auditor_implementation_sha256
                            or canonical_sha256("llm_concept_auditor_unavailable")
                        ),
                    )
                    attempt.checkpoint_authority.checkpoint_capsule(
                        blocked_ref,
                        status="concept_audited_pending_review",
                    )
            if state.final_llm_audit_due:
                # These outputs came from a digest rejected by the final
                # semantic audit.  They are never eligible for later
                # sealing/current authority, and a repaired digest must
                # execute afresh before it can regain contract approval.
                state.deterministic_contract_approved_code_digest = None
                _clear_output_dir(
                    host.run_dir / "steps" / attempt.step.step_id / "outputs"
                )
            post_mutation_messages = [
                value
                for finding in post_mutation_errors
                for value in (
                    finding.message,
                    str((finding.detail or {}).get("reason") or ""),
                )
                if value
            ]
            post_mutation_reasons = [
                repair_reason_for_finding(finding) for finding in post_mutation_errors
            ]
            if (
                attempt.worker_progress.deterministic_concept_repairs
                < MAX_DETERMINISTIC_CONCEPT_REPAIRS
            ):
                deterministic_code, deterministic_names = (
                    attempt._authorized_deterministic_concept_repair(
                        script_text=state.code,
                        error_messages=post_mutation_messages,
                        repair_reasons=post_mutation_reasons,
                        repair_findings=post_mutation_errors,
                        source=("post_mutation_deterministic_concept_repair"),
                    )
                )
                if deterministic_names and deterministic_code != state.code:
                    before_code = state.code
                    state.code = deterministic_code
                    attempt.worker_progress.deterministic_concept_repairs += 1
                    attempt.worker_progress.applied_concept_repair_names.extend(
                        deterministic_names
                    )
                    attempt.step_record["deterministic_concept_repairs"] = (
                        attempt.worker_progress.deterministic_concept_repairs
                    )
                    attempt.step_record["applied_concept_repair_names"] = list(
                        attempt.worker_progress.applied_concept_repair_names
                    )
                    attempt.step_record["deterministic_concept_repair_code_sha256"] = (
                        sha256_of_bytes(state.code.encode("utf-8"))
                    )
                    for repair_name in deterministic_names:
                        host._record_repair(
                            repair_id=repair_name,
                            step_id=attempt.step.step_id,
                            trigger={
                                "gate": "post_mutation_concept_audit",
                                "audit_errors": post_mutation_messages,
                            },
                            transformation=(
                                "deterministic concept repair after a "
                                "contract/runtime mutation"
                            ),
                            before_code=before_code,
                            after_code=state.code,
                            selection_rule=(
                                "applied only because a typed mechanical "
                                "error named the anti-pattern"
                            ),
                        )
                    return _CandidateLoopAction.CONTINUE

            if attempt._llm_repair_budget_available("post_mutation_concept"):
                post_mutation_ticket = typed_repair_ticket(post_mutation_errors)
                current_post_mutation_repair_authority = RepairPromptAuthority.create(
                    typed_ticket=post_mutation_ticket,
                )
                post_mutation_repair_authority = RepairPromptAuthority.create(
                    typed_ticket=[
                        *post_mutation_ticket,
                        *attempt._monotonic_concept_constraint_ticket(),
                    ],
                )
                post_mutation_log = "\n".join(
                    (
                        f"{finding.severity.upper()}: {finding.message}"
                        + (
                            "\nDETAIL (diagnostic mirror only): "
                            + json.dumps(
                                finding.detail,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            if finding.detail
                            else ""
                        )
                    )
                    for finding in post_mutation_errors
                )
                post_mutation_repair_log = (
                    "A contract or runtime repair produced a new "
                    "code digest that failed pre-execution audit. "
                    "Fix every typed error with the smallest change; "
                    "preserve the earlier contract repair and all "
                    "Planner-owned science.\n\n"
                    "FINDINGS (diagnostic mirror only):\n" + post_mutation_log
                )
                attempt.worker_progress.concept_repair_attempts += 1
                if not attempt._consume_llm_repair_budget(
                    "post_mutation_concept",
                    before_code=state.code,
                    repair_ticket=post_mutation_repair_log,
                    repair_authority=post_mutation_repair_authority,
                    current_repair_authority=(current_post_mutation_repair_authority),
                    provider_category="post_mutation_concept_repair",
                    failure_status="concept_failed",
                ):
                    raise AssertionError("LLM repair budget changed without mutation")
                attempt.step_record["concept_repair_attempts"] = (
                    attempt.worker_progress.concept_repair_attempts
                )
                host.emit_progress(
                    "coder",
                    (
                        "Repairing post-mutation concept violation for "
                        f"{attempt.step.step_id}."
                    ),
                    run_id=host.run_id,
                    step_id=attempt.step.step_id,
                    current_step=attempt.step_current,
                    total_steps=host.total_steps,
                    repair_attempts=attempt.step_repair_budget.llm_repair_attempts,
                )
                attempt._remember_concept_constraints(post_mutation_errors)
                try:
                    state.code = attempt._repair_with_capsule(
                        failure_status="concept_failed",
                        context=attempt.coder_context,
                        step=attempt.step,
                        code=state.code,
                        run_log=post_mutation_repair_log,
                        repair_authority=post_mutation_repair_authority,
                        current_repair_authority=(
                            current_post_mutation_repair_authority
                        ),
                        attempt=attempt.worker_progress.concept_repair_attempts,
                        provider_budget=attempt.provider_budget,
                        provider_category="post_mutation_concept_repair",
                        logical_repair_attempt_id=(
                            attempt.step_repair_budget.llm_repair_attempts
                        ),
                    )
                    attempt._sync_provider_budget()
                    attempt.worker_progress.llm_repair_used = True
                    return _CandidateLoopAction.CONTINUE
                except (
                    ProviderCallBudgetReceiptError,
                    StepAuthorityRuntimeError,
                    StepAuthorityCapsuleError,
                ):
                    raise
                except Exception as exc:
                    attempt._sync_provider_budget()
                    checkpoint_error: Optional[Exception] = None
                    try:
                        checkpoint = store_quarantined_concept_draft(
                            run_dir=host.run_dir,
                            step_id=attempt.step.step_id,
                            code=state.code,
                            findings=attempt._quarantine_error_payloads(
                                post_mutation_errors
                            ),
                        )
                        attempt.step_record["quarantined_draft_sha256"] = (
                            checkpoint.sha256
                        )
                        attempt.step_record["quarantined_draft_relative_path"] = (
                            checkpoint.relative_path
                        )
                        attempt.step_record["quarantined_requires_repair"] = True
                    except Exception as checkpoint_exc:
                        checkpoint_error = checkpoint_exc
                    fallback_code = attempt._deterministic_fallback_code(
                        "concept_repair_failed"
                    )
                    if fallback_code is not None:
                        state.code = fallback_code
                        return _CandidateLoopAction.CONTINUE
                    with host.shared_lock:
                        host.findings.extend(state.usage_findings)
                        if checkpoint_error is not None:
                            host.findings.append(
                                ValidationFinding(
                                    validator="resume",
                                    severity="warning",
                                    message=(
                                        "Could not preserve the rejected final "
                                        "concept-audit draft for step "
                                        f"{attempt.step.step_id}: {checkpoint_error}"
                                    ),
                                    detail={"step_id": attempt.step.step_id},
                                )
                            )
                        host.findings.append(
                            ValidationFinding(
                                validator="coder",
                                severity="error",
                                message=(
                                    "Coder repair failed after post-mutation "
                                    "concept audit for step "
                                    f"{attempt.step.step_id}: {exc}"
                                ),
                                detail={"step_id": attempt.step.step_id},
                            )
                        )
                        attempt.step_record["status"] = "repair_failed"
                        host._append_terminal_step_record(
                            host.per_step_records, attempt.step_record
                        )
                        host._flush_partial_manifest()
                    host.emit_progress(
                        "coder",
                        f"Concept-audit repair failed for {attempt.step.step_id}.",
                        status="error",
                        run_id=host.run_id,
                        step_id=attempt.step.step_id,
                        current_step=attempt.step_current,
                        total_steps=host.total_steps,
                    )
                    return _CandidateLoopAction.RETURN

            if not attempt._logical_llm_repair_budget_available(
                "post_mutation_concept"
            ):
                attempt.step_record["step_llm_repair_budget_exhausted"] = True
                attempt.step_record["step_llm_repair_budget"] = (
                    host.pipeline._max_step_llm_repair_attempts
                )
            checkpoint_error: Optional[Exception] = None
            try:
                checkpoint = store_quarantined_concept_draft(
                    run_dir=host.run_dir,
                    step_id=attempt.step.step_id,
                    code=state.code,
                    findings=attempt._quarantine_error_payloads(post_mutation_errors),
                )
                attempt.step_record["quarantined_draft_sha256"] = checkpoint.sha256
                attempt.step_record["quarantined_draft_relative_path"] = (
                    checkpoint.relative_path
                )
                attempt.step_record["quarantined_requires_repair"] = True
            except Exception as checkpoint_exc:
                checkpoint_error = checkpoint_exc
            attempt.step_record["status"] = "blocked_by_concept_audit"
            attempt.step_record["post_repair_concept_audit_block"] = {
                "code_sha256": state.candidate_code_digest,
                "errors": [
                    finding.model_dump(mode="json") for finding in post_mutation_errors
                ],
            }
            with host.shared_lock:
                host.findings.extend(state.usage_findings)
                if checkpoint_error is not None:
                    host.findings.append(
                        ValidationFinding(
                            validator="resume",
                            severity="warning",
                            message=(
                                "Could not preserve post-repair code rejected "
                                f"by concept audit for step {attempt.step.step_id}: "
                                f"{checkpoint_error}"
                            ),
                            detail={"step_id": attempt.step.step_id},
                        )
                    )
                host._append_terminal_step_record(
                    host.per_step_records, attempt.step_record
                )
                host._flush_partial_manifest()
            host.emit_progress(
                "audit",
                f"Concept audit blocked mutated code for {attempt.step.step_id}.",
                status="error",
                run_id=host.run_id,
                step_id=attempt.step.step_id,
                current_step=attempt.step_current,
                total_steps=host.total_steps,
            )
            return _CandidateLoopAction.RETURN
        with host.shared_lock:
            host.findings.extend(state.usage_findings)
        state.concept_approved_code_digest = state.candidate_code_digest
        attempt.step_record["concept_approved_code_sha256"] = (
            state.concept_approved_code_digest
        )
        attempt.step_record["deterministic_preflight_approved_code_sha256"] = (
            state.concept_approved_code_digest
        )
        if state.final_llm_audit_due:
            state.final_concept_gate_approved_code_digest = state.candidate_code_digest
            attempt.step_record["final_concept_gate_approved_code_sha256"] = (
                state.final_concept_gate_approved_code_digest
            )
            if state.candidate_code_digest in attempt.concept_audit.completed_digests:
                final_audit_token = attempt.concept_audit.tokens_by_digest.get(
                    state.candidate_code_digest
                )
                if final_audit_token is not None:
                    attempt.provider_budget.complete_reserved_category(
                        "concept_audit",
                        token=final_audit_token,
                    )
                attempt.step_record["llm_concept_audit_status"] = "completed"
                attempt.step_record["llm_concept_approved_code_sha256"] = (
                    state.candidate_code_digest
                )
            elif not host.pipeline._enable_llm_concept_audit:
                attempt.step_record["llm_concept_audit_status"] = "disabled"
            elif (
                attempt.worker_progress.deterministic_fallback_used
                or attempt.worker_progress.deterministic_standard_executor_used
                or host_authored_generation_mode(
                    attempt.step_record.get("resumed_from_generation_mode")
                )
            ):
                attempt.step_record["llm_concept_audit_status"] = (
                    "skipped_trusted_deterministic_code"
                )
            else:
                attempt.step_record["llm_concept_audit_status"] = (
                    "skipped_no_auditor_client"
                )
            audit_authority_complete = bool(
                state.candidate_code_digest in attempt.concept_audit.completed_digests
                or not host.pipeline._enable_llm_concept_audit
                or attempt.worker_progress.deterministic_fallback_used
                or attempt.worker_progress.deterministic_standard_executor_used
                or host_authored_generation_mode(
                    attempt.step_record.get("resumed_from_generation_mode")
                )
            )
            if (
                audit_authority_complete
                and attempt.step_attempt_state.coordinates is not None
                and attempt.step_attempt_state.current_capsule_ref is not None
            ):
                current_authority = load_verified_step_authority_capsule(
                    host.run_dir,
                    ref=attempt.step_attempt_state.current_capsule_ref,
                    expected_step_id=attempt.step.step_id,
                )
                if (
                    current_authority.capsule.concept_audit is None
                    or attempt.step_record.get("step_authority_audit_cache_miss")
                    == "audit_identity_drift"
                ):
                    audit_key = attempt.concept_audit.tokens_by_digest.get(
                        state.candidate_code_digest
                    ) or canonical_sha256(
                        {
                            "schema": ("easyicu.capsule_deterministic_concept_audit/1"),
                            "step_id": attempt.step.step_id,
                            "code_sha256": state.candidate_code_digest,
                            "findings": [
                                finding.model_dump(mode="json")
                                for finding in state.usage_findings
                            ],
                        }
                    )
                    audited_ref = seal_concept_audit_capsule(
                        attempt.step_attempt_state.coordinates,
                        parent_ref=attempt.step_attempt_state.current_capsule_ref,
                        findings=state.usage_findings,
                        audit_key=audit_key,
                        auditor_identity_sha256=(
                            host.llm_concept_auditor_identity_sha256
                        ),
                        environment_sha256=(host.concept_audit_environment_sha256),
                        validator_implementation_sha256=(
                            host.llm_concept_auditor_implementation_sha256
                            or canonical_sha256("llm_concept_auditor_unavailable")
                        ),
                    )
                    attempt.checkpoint_authority.checkpoint_capsule(
                        audited_ref,
                        status="concept_audited_pending_review",
                    )
            # Reuse the already validated outputs.  No second execution
            # of unchanged code is needed after the digest-bound audit.
            return _CandidateLoopAction.BREAK
    return _CandidateLoopAction.PROCEED


def _candidate_execute_transition(
    host: _CandidateLoopHost, attempt: _CandidateLoopAttempt, state: _CandidateLoopState
) -> _CandidateLoopAction:
    state.current_generation_mode = attempt.worker_progress.generation_mode()
    run_label = {
        "llm": "generated script",
        "resumed_code_reuse": "resumed script",
        "fallback": "fallback script",
        "deterministic_standard": "standard executor script",
    }.get(state.current_generation_mode, "repaired script")
    execution_runner = host.runner
    owner_capability = (
        PreselectionUniverseOwnerCapability.DETERMINISTIC_ROBUSTNESS_REPLAY
        if state.current_generation_mode in {"fallback", "deterministic_standard"}
        and _robustness_sensitivity_runner_owns_step(
            attempt.step.method,
            attempt.step.step_id,
            attempt.step.expected_outputs,
            step=attempt.step,
        )
        else None
    )
    universe_capability = preselection_universe_capability(
        step=attempt.step,
        plan=host.plan,
        owner_capability=owner_capability,
    )
    state.execution_timeout_seconds = host.pipeline._timeout_seconds
    if attempt.worker_progress.deterministic_standard_executor_used:
        # Give a registered standard's exact typed workload its dedicated
        # timeout even with a staged primary-cohort universe.
        state.execution_timeout_seconds = (
            host.pipeline._standard_executor_timeout_seconds
        )
    if (
        attempt.step_execution_cohort_path != host.cohort_path
        or attempt.worker_progress.deterministic_standard_executor_used
        or universe_capability is not None
    ):
        execution_runner = host.pipeline._build_runner(
            run_dir=host.run_dir,
            cohort_path=attempt.step_execution_cohort_path,
            target_outcome=host.context.target_outcome,
            universe_path=host.universe_path,
            preselection_universe_capability=universe_capability,
            **host.run_input_authority_state.runner_bindings(),
            timeout_seconds=state.execution_timeout_seconds,
        )
    state.execution_timeout_seconds = host.step_executor.runner_timeout(
        execution_runner, state.execution_timeout_seconds
    )
    host.emit_progress(
        "runner",
        f"Running {run_label} for {attempt.step.step_id}.",
        run_id=host.run_id,
        step_id=attempt.step.step_id,
        current_step=attempt.step_current,
        total_steps=host.total_steps,
        repair_attempts=attempt.worker_progress.repair_attempts,
        phase_timeout_seconds=state.execution_timeout_seconds,
    )
    host.run_input_authority_state.require_trajectory_integrity(
        step_id=attempt.step.step_id,
    )
    runner_identity = (
        f"{type(execution_runner).__module__}.{type(execution_runner).__qualname__}"
    )
    runner_network_identity = str(
        getattr(
            execution_runner,
            "network_policy",
            getattr(execution_runner, "network", "none"),
        )
    )
    runner_authority_identity = getattr(
        execution_runner,
        "authority_identity_sha256",
        None,
    )
    custom_runner_replay_allowed = not (
        host.pipeline._runner_kind == "custom"
        and not (
            isinstance(runner_authority_identity, str)
            and re.fullmatch(r"[0-9a-f]{64}", runner_authority_identity) is not None
        )
    )
    execution_context_digest = execution_context_sha256(
        code_sha256=state.candidate_code_digest,
        resolved_inputs_sha256=attempt.resolved_inputs_sha256,
        cohort_sha256=sha256_of_file(attempt.step_execution_cohort_path),
        universe_sha256=sha256_of_file(host.universe_path),
        runner_identity=runner_identity,
        timeout_seconds=state.execution_timeout_seconds,
        requested_network_policy=runner_network_identity,
        runtime_environment_sha256=(current_execution_runtime_sha256()),
        runner_configuration_sha256=canonical_sha256(
            {
                "schema": "easyicu.runner_configuration/1",
                "runner_identity": runner_identity,
                "configured_kind": str(host.pipeline._runner_kind),
                "configured_image": str(host.pipeline._runner_image or ""),
                "configured_network": str(host.pipeline._runner_network),
                "effective_network": runner_network_identity,
                "preselection_universe_capability": (
                    universe_capability.value
                    if universe_capability is not None
                    else None
                ),
                "runner_authority_identity_sha256": (
                    runner_authority_identity
                    if isinstance(runner_authority_identity, str)
                    else None
                ),
                "manages_output_cleanup": bool(
                    getattr(
                        execution_runner,
                        "manages_output_cleanup",
                        False,
                    )
                ),
            }
        ),
        trajectory_sha256=host.run_input_authority_state.trajectory_sha256,
        trajectory_authority_sha256=(
            host.run_input_authority_state.trajectory_authority_sha256
        ),
    )
    replay_execution = (
        attempt.step_attempt_state.selected_resume_capsule
        if (
            custom_runner_replay_allowed
            and not attempt.step_attempt_state.capsule_execution_replay_consumed
            and attempt.step_attempt_state.selected_resume_capsule is not None
            and attempt.step_attempt_state.selected_resume_capsule.capsule.execution
            is not None
            and attempt.step_attempt_state.selected_resume_capsule.capsule.candidate_code.sha256
            == state.candidate_code_digest
        )
        else None
    )
    if not custom_runner_replay_allowed:
        attempt.step_record["step_authority_execution_cache_miss"] = (
            "custom_runner_authority_unbound"
        )
    # DockerRunner must first prove any previous timed-out container
    # is quiescent before its bind-mounted output directory is reused;
    # it therefore owns cleanup inside ``run``. Other backends retain
    # the pipeline's established pre-execution clearing behaviour.
    attempt.step_record["execution_timeout_seconds"] = state.execution_timeout_seconds
    if replay_execution is not None:
        if (
            replay_execution.capsule.execution.execution_context_sha256
            != execution_context_digest
        ):
            attempt.step_record["step_authority_execution_cache_miss"] = (
                "execution_context_drift"
            )
            replay_execution = None
        else:
            try:
                state.run_result = materialize_sealed_run_result(
                    host.run_dir,
                    replay_execution,
                    expected_execution_context_sha256=(execution_context_digest),
                )
            except StepAuthorityRuntimeError as exc:
                replay_finding = ValidationFinding(
                    validator="step_authority_capsule",
                    severity="error",
                    message=(
                        "Checkpoint-selected execution could not be "
                        f"replayed safely for step {attempt.step.step_id}."
                    ),
                    detail={"step_id": attempt.step.step_id, "reason": str(exc)},
                )
                attempt.step_record["status"] = "contract_failed"
                with host.shared_lock:
                    host.findings.append(replay_finding)
                    host._append_terminal_step_record(
                        host.per_step_records, attempt.step_record
                    )
                    host._flush_partial_manifest()
                return _CandidateLoopAction.RETURN
            attempt.step_attempt_state.capsule_execution_replay_consumed = True
            attempt.step_record["capsule_execution_replayed"] = True
    if replay_execution is None:
        if (
            attempt.step_attempt_state.coordinates is not None
            and attempt.step_attempt_state.current_capsule_ref is not None
        ):
            current_before_execution = load_verified_step_authority_capsule(
                host.run_dir,
                ref=attempt.step_attempt_state.current_capsule_ref,
                expected_step_id=attempt.step.step_id,
            )
            if current_before_execution.capsule.stage not in {
                "candidate",
                "concept_audited",
            }:
                ref = seal_deterministic_candidate(
                    attempt.step_attempt_state.coordinates,
                    parent_ref=attempt.step_attempt_state.current_capsule_ref,
                    code_ref=persist_candidate_code(
                        attempt.step_attempt_state.coordinates, state.code
                    ),
                    reason="execution_context_changed_or_retry_requested",
                )
                attempt.checkpoint_authority.checkpoint_capsule(
                    ref,
                    status="candidate_checkpointed",
                )
        state.run_result = host.step_executor.execute(
            runner=execution_runner,
            request=LockedStepExecutionRequest(
                step_id=attempt.step.step_id,
                code=state.code,
                resolved_inputs_path=attempt.resolved_inputs_path,
                output_dir=(host.run_dir / "steps" / attempt.step.step_id / "outputs"),
            ),
        )

    def _seal_actual_execution_result() -> None:
        if (
            replay_execution is not None
            or attempt.step_attempt_state.coordinates is None
            or attempt.step_attempt_state.current_capsule_ref is None
        ):
            return
        executed_ref = seal_execution_capsule(
            attempt.step_attempt_state.coordinates,
            parent_ref=attempt.step_attempt_state.current_capsule_ref,
            run_result=state.run_result,
            execution_context_digest=execution_context_digest,
        )
        attempt.checkpoint_authority.checkpoint_capsule(
            executed_ref,
            status="executed_pending_review",
        )

    attempt.step_record["outputs_safe_to_collect"] = bool(
        state.run_result.outputs_safe_to_collect
    )
    authority_findings: List[ValidationFinding] = []
    if attempt.step_record.get("execution_cohort_role") == RAW_UNIVERSE_EXECUTION_ROLE:
        cohort_authority_finding = host._execution_input_authority_integrity_finding(
            step_id=attempt.step.step_id,
            universe_path=host.universe_path,
            cohort_path=host.cohort_path,
            expected_universe_sha256=attempt.step_record.get("execution_cohort_sha256"),
            expected_analysis_cohort_sha256=attempt.step_record.get(
                "authoritative_analysis_cohort_sha256"
            ),
        )
        if cohort_authority_finding is not None:
            authority_findings.append(cohort_authority_finding)
    trajectory_authority_finding = (
        host.run_input_authority_state.trajectory_integrity_finding(
            step_id=attempt.step.step_id
        )
    )
    if trajectory_authority_finding is not None:
        authority_findings.append(trajectory_authority_finding)
    if authority_findings:
        if state.run_result.outputs_safe_to_collect:
            _clear_output_dir(state.run_result.out_dir)
        attempt.step_record.update(
            {
                "status": "blocked_input_authority_mutation",
                "input_authority_findings": [
                    item.model_dump() for item in authority_findings
                ],
            }
        )
        with host.shared_lock:
            host.run_input_authority_state.mark_corrupted(step_id=attempt.step.step_id)
            host.findings.extend(authority_findings)
            host._append_terminal_step_record(
                host.per_step_records, attempt.step_record
            )
            host._flush_partial_manifest()
        host.emit_progress(
            "audit",
            f"Rejected mutated execution authority for {attempt.step.step_id}.",
            status="error",
            run_id=host.run_id,
            step_id=attempt.step.step_id,
            current_step=attempt.step_current,
            total_steps=host.total_steps,
        )
        return _CandidateLoopAction.RETURN
    if not state.run_result.outputs_safe_to_collect:
        # The backend could not prove that a process/container with a
        # writable output mount was stopped.  Those outputs remain
        # mutable and are therefore ineligible for inspection,
        # hashing, repair, cleanup, or evidence registration. Docker
        # keeps host-owned script/log control copies, but this step is
        # still terminal until a later explicit retry resolves the
        # teardown sentinel first.
        unsafe_reason = "runner_output_teardown_unconfirmed"
        attempt.step_record.update(
            {
                "status": (
                    "deterministic_standard_blocked"
                    if attempt.is_trajectory_stability_standard
                    else "execution_failed"
                ),
                "diagnostic_only": True,
                "runner_output_safety_reason": unsafe_reason,
            }
        )
        if attempt.is_trajectory_stability_standard:
            attempt.step_record["standard_executor_terminal_reason"] = (
                "executor_runtime_failure"
            )
        elif attempt.sealed_renderer_authorized_code_sha256 is not None:
            attempt.step_record.update(
                {
                    "sealed_renderer_runtime_repair_suppressed": True,
                    "sealed_renderer_terminal_reason": unsafe_reason,
                    "llm_repair_used": False,
                    "generation_mode": "fallback",
                }
            )
        unsafe_finding = ValidationFinding(
            validator="runner_output_safety",
            severity="error",
            message=(
                f"Step {attempt.step.step_id} was stopped because the execution "
                "backend could not confirm teardown of its writable "
                "mount; no files from that mount were inspected or "
                "registered."
            ),
            detail={
                "step_id": attempt.step.step_id,
                "reason": unsafe_reason,
                "timed_out": bool(state.run_result.timed_out),
                "returncode": int(state.run_result.returncode),
            },
        )
        _seal_actual_execution_result()
        with host.shared_lock:
            host.findings.append(unsafe_finding)
            host._append_terminal_step_record(
                host.per_step_records, attempt.step_record
            )
            host._flush_partial_manifest()
        host.emit_progress(
            "runner",
            f"Execution mount teardown was not confirmed for {attempt.step.step_id}.",
            status="error",
            run_id=host.run_id,
            step_id=attempt.step.step_id,
            current_step=attempt.step_current,
            total_steps=host.total_steps,
        )
        return _CandidateLoopAction.RETURN
    state.executed_code_digest = sha256_of_file(state.run_result.script_path)
    attempt.step_record["executed_code_sha256"] = state.executed_code_digest
    if attempt.sealed_renderer_authorized_code_sha256 is not None:
        attempt.step_record["sealed_renderer_executed_code_matches_authority"] = (
            state.executed_code_digest == attempt.sealed_renderer_authorized_code_sha256
        )
    if (
        state.concept_approved_code_digest is None
        or state.executed_code_digest != state.concept_approved_code_digest
    ):
        integrity_finding = ValidationFinding(
            validator="post_repair_concept_gate",
            severity="error",
            message=(
                "The executed analysis script did not match the exact "
                f"concept-approved code digest for step {attempt.step.step_id}; "
                "outputs were rejected before evidence registration."
            ),
            detail={
                "step_id": attempt.step.step_id,
                "concept_approved_code_sha256": state.concept_approved_code_digest,
                "executed_code_sha256": state.executed_code_digest,
                "script_path": str(state.run_result.script_path),
            },
        )
        _clear_output_dir(state.run_result.out_dir)
        attempt.step_record["status"] = "blocked_script_integrity"
        attempt.step_record["script_integrity_findings"] = [
            integrity_finding.model_dump()
        ]
        if attempt.sealed_renderer_authorized_code_sha256 is not None:
            attempt.step_record.update(
                {
                    "sealed_renderer_terminal_reason": (
                        "executed_code_digest_mismatch"
                    ),
                    "llm_repair_used": False,
                    "generation_mode": "fallback",
                }
            )
        with host.shared_lock:
            host.findings.append(integrity_finding)
            host._append_terminal_step_record(
                host.per_step_records, attempt.step_record
            )
            host._flush_partial_manifest()
        host.emit_progress(
            "audit",
            f"Rejected script-integrity mismatch for {attempt.step.step_id}.",
            status="error",
            run_id=host.run_id,
            step_id=attempt.step.step_id,
            current_step=attempt.step_current,
            total_steps=host.total_steps,
        )
        return _CandidateLoopAction.RETURN
    _seal_actual_execution_result()
    attempt.step_record["returncode"] = state.run_result.returncode
    attempt.step_record["timed_out"] = state.run_result.timed_out
    attempt.step_record["requested_network_policy"] = (
        state.run_result.requested_network_policy
    )
    attempt.step_record["effective_isolation"] = state.run_result.effective_isolation
    attempt.step_record["isolation_degraded"] = state.run_result.isolation_degraded
    if state.run_result.isolation_degradation_reason:
        attempt.step_record["isolation_degradation_reason"] = (
            state.run_result.isolation_degradation_reason
        )
    attempt.step_record["code_repair_attempts"] = (
        attempt.worker_progress.repair_attempts
    )

    if state.current_generation_mode == "llm":
        script_description = (
            f"Generated analysis script for step {attempt.step.step_id}."
        )
    elif state.current_generation_mode == "resumed_code_reuse":
        script_description = (
            f"Reused prior agent-generated analysis script for step "
            f"{attempt.step.step_id}."
        )
    elif state.current_generation_mode == "fallback":
        script_description = (
            f"Deterministic fallback analysis script for step {attempt.step.step_id}."
        )
    elif state.current_generation_mode == "deterministic_standard":
        script_description = (
            "Planner-selected deterministic standard executor adapter for "
            f"step {attempt.step.step_id}."
        )
    else:
        total_repair_attempts = (
            attempt.worker_progress.repair_attempts
            + attempt.worker_progress.concept_repair_attempts
        )
        script_description = (
            f"Repaired analysis script for step {attempt.step.step_id} "
            f"(attempt {total_repair_attempts})."
        )
    script_digest = sha256_of_file(state.run_result.script_path)
    script_authority = "\0".join(
        (attempt.step.step_id, script_digest, state.current_generation_mode)
    )
    script_evidence_id = (
        "code_analysis_"
        + hashlib.sha256(script_authority.encode("utf-8")).hexdigest()[:16]
    )
    state.script_record = host.evidence.register_file(
        kind="code",
        description=script_description,
        source_path=state.run_result.script_path,
        produced_by_step=attempt.step.step_id,
        inputs=attempt.resolved_input_evidence_ids or None,
        evidence_id=script_evidence_id,
        producer=(
            "standard_executor"
            if state.current_generation_mode == "deterministic_standard"
            else "coder"
        ),
        generation_mode=state.current_generation_mode,
        prompt_pack_version=host.prompt_version,
        metadata={
            "repair_attempts": attempt.worker_progress.repair_attempts,
            "concept_repair_attempts": attempt.worker_progress.concept_repair_attempts,
            "deterministic_concept_repairs": attempt.worker_progress.deterministic_concept_repairs,
            "llm_repair_used": attempt.worker_progress.llm_repair_used,
            "fallback_reason": attempt.step_record.get("deterministic_code_fallback"),
            "runner_repair": attempt.worker_progress.runner_repair_name,
            "resumed_code_evidence_id": attempt.step_record.get(
                "resumed_code_evidence_id"
            ),
            "resumed_code_relative_path": attempt.step_record.get(
                "resumed_code_relative_path"
            ),
            "resumed_from_generation_mode": attempt.step_record.get(
                "resumed_from_generation_mode"
            ),
            "resumed_code_evidence_generation_mode": attempt.step_record.get(
                "resumed_code_evidence_generation_mode"
            ),
            "resumed_quarantined_draft": attempt.quarantine_state.resumed_draft_used,
            "quarantined_draft_sha256": attempt.step_record.get(
                "quarantined_draft_sha256"
            ),
            "quarantined_repair_succeeded": attempt.quarantine_state.repair_succeeded,
            "quarantine_policy_superseded": attempt.quarantine_state.policy_superseded,
            "quarantine_policy_superseded_findings": attempt.step_record.get(
                "quarantine_policy_superseded_findings"
            ),
            "llm_signature": host.llm_signature,
        },
    )
    attempt.step_record["script_evidence_id"] = state.script_record.evidence_id
    state.log_path = state.run_result.runner_log_path or (
        state.run_result.cwd / "run.log"
    )
    if state.log_path.exists():
        host.evidence.register_file(
            kind="log",
            description=f"stdout/stderr log for step {attempt.step.step_id}.",
            source_path=state.log_path,
            produced_by_step=attempt.step.step_id,
            script_evidence_id=state.script_record.evidence_id,
            producer="runner",
            generation_mode=state.current_generation_mode,
            metadata={
                "repair_attempts": attempt.worker_progress.repair_attempts,
                "concept_repair_attempts": attempt.worker_progress.concept_repair_attempts,
                "deterministic_concept_repairs": (
                    attempt.worker_progress.deterministic_concept_repairs
                ),
                "llm_repair_used": attempt.worker_progress.llm_repair_used,
                "fallback_reason": attempt.step_record.get(
                    "deterministic_code_fallback"
                ),
                "runner_repair": attempt.worker_progress.runner_repair_name,
                "resumed_from_generation_mode": attempt.step_record.get(
                    "resumed_from_generation_mode"
                ),
            },
        )
    return _CandidateLoopAction.PROCEED


def _candidate_success_prepare_transition(
    host: _CandidateLoopHost, attempt: _CandidateLoopAttempt, state: _CandidateLoopState
) -> _CandidateLoopAction:
    salvage_outcome = salvage_step_summary(state.run_result, step=attempt.step)
    if salvage_outcome is not None:
        if salvage_outcome.reset_artefacts:
            state.run_result.artefacts = sorted(
                p for p in state.run_result.out_dir.iterdir() if p.is_file()
            )
        host._record_repair(
            repair_id=salvage_outcome.repair_id,
            step_id=attempt.step.step_id,
            trigger={
                "source": "summary_salvage",
                "reason": salvage_outcome.trigger_reason,
            },
            transformation=salvage_outcome.transformation,
            selection_rule=salvage_outcome.selection_rule,
        )
    if not state.run_result.artefacts:
        if attempt.is_trajectory_stability_standard:
            state.standard_executor_terminal_block = True
            state.standard_executor_terminal_reason = "missing_executor_outputs"
            return _CandidateLoopAction.BREAK
        fallback_code = attempt._deterministic_fallback_code("no_artefacts")
        if fallback_code is not None:
            state.code = fallback_code
            return _CandidateLoopAction.CONTINUE
    state.visual_step_summary: Dict[str, Any] = {}
    visual_summary_path = state.run_result.out_dir / "step_summary.json"
    if visual_summary_path.exists():
        try:
            vloaded = json.loads(visual_summary_path.read_text(encoding="utf-8"))
        except Exception:
            vloaded = None
        if isinstance(vloaded, dict):
            state.visual_step_summary = vloaded
        else:
            state.visual_step_summary = {"raw": vloaded}
    # The early repair gate must evaluate the same host-bound
    # canonical scalars as the final gate.  Otherwise a valid,
    # Planner-declared statistic sidecar is invisible here and
    # triggers pointless LLM repairs before evidence registration.
    normalized_statistic_outputs = normalize_typed_statistic_sidecars(
        state.visual_step_summary,
        state.run_result.out_dir,
    )
    if normalized_statistic_outputs:
        host._record_repair(
            repair_id="typed_output_normalization_v1",
            step_id=attempt.step.step_id,
            trigger={
                "source": "host_output_normalizer",
                "normalized_outputs": normalized_statistic_outputs,
            },
            transformation=(
                "Added the exact Planner-declared statistic product "
                "identity to host-bound JSON sidecars without changing "
                "their numeric values."
            ),
        )
    state.visual_step_summary = bind_primary_output(
        state.visual_step_summary,
        state.run_result.out_dir,
    )
    # Same rule as the post-execution site below, from the single
    # owner -- this one runs BEFORE the contract gate, so a producer
    # missing here is refused for a receipt the host never wrote.
    if host_owns_input_binding_receipts(
        deterministic_standard_executor_used=(
            attempt.worker_progress.deterministic_standard_executor_used
        ),
        deterministic_fallback_used=(
            attempt.worker_progress.deterministic_fallback_used
        ),
        sealed_renderer_repair=bool(
            attempt.worker_progress.runner_repair_name
            and is_sealed_renderer_repair(attempt.worker_progress.runner_repair_name)
        ),
        resumed_from_generation_mode=attempt.step_record.get(
            "resumed_from_generation_mode"
        ),
    ):
        state.visual_step_summary = _write_host_input_binding_receipts(
            out_dir=state.run_result.out_dir,
            step_summary=state.visual_step_summary,
            resolved_input_bindings=attempt.resolved_input_bindings,
            consumed_input_keys=(
                attempt.standard_executor.consumed_input_keys
                if attempt.worker_progress.deterministic_standard_executor_used
                and attempt.standard_executor is not None
                else tuple(attempt.resolved_input_bindings)
            ),
        )
    if attempt.is_trajectory_stability_standard:
        terminal_status = (
            str(state.visual_step_summary.get("status") or "").strip().lower()
        )
        if terminal_status != "ok":
            state.standard_executor_terminal_block = True
            state.standard_executor_terminal_reason = "executor_reported_" + (
                terminal_status or "missing_status"
            )
            state.standard_executor_terminal_summary = dict(state.visual_step_summary)
            return _CandidateLoopAction.BREAK
    step_figures = [
        art
        for art in state.run_result.artefacts
        if art.suffix.lower() in {".png", ".svg", ".tiff", ".tif"}
    ]
    state.visual_gate = collect_visual_gate_result(
        enabled=host.pipeline._enable_visual_qa,
        step_figures=step_figures,
        step=attempt.step,
        step_summary=state.visual_step_summary,
    )
    return _CandidateLoopAction.PROCEED


def _candidate_visual_transition(
    host: _CandidateLoopHost, attempt: _CandidateLoopAttempt, state: _CandidateLoopState
) -> _CandidateLoopAction:
    if state.visual_gate.ran:
        visual_findings = list(state.visual_gate.findings)
        attempt.step_record["visual_findings"] = [
            f.model_dump() for f in visual_findings
        ]
        if state.visual_gate.has_errors:
            visual_repair_decision = decide_visual_repair(
                state.visual_gate,
                sealed=attempt.sealed_renderer_authorized_code_sha256 is not None,
                attempts_exhausted=(
                    attempt.worker_progress.visual_repair_attempts
                    >= host.pipeline._max_code_repair_attempts
                ),
                budget_available=attempt._llm_repair_budget_available(),
            )
            if visual_repair_decision.action is VisualRepairAction.SEALED_SUPPRESS:
                demoted_findings = list(state.visual_gate.demoted_findings)
                blocking_visual_errors = list(state.visual_gate.blocking_errors)
                attempt.step_record["visual_findings"] = [
                    finding.model_dump() for finding in demoted_findings
                ]
                attempt.step_record["sealed_renderer_visual_repair_suppressed"] = True
                attempt.step_record["visual_qa_demoted"] = state.visual_gate.was_demoted
                with host.shared_lock:
                    host.findings.extend(demoted_findings)
                if blocking_visual_errors:
                    attempt.step_record.update(
                        {
                            "status": "execution_failed",
                            "diagnostic_only": True,
                            "sealed_renderer_terminal_reason": ("visual_qa_failed"),
                            "llm_repair_used": False,
                            "generation_mode": "fallback",
                        }
                    )
                    with host.shared_lock:
                        host._append_terminal_step_record(
                            host.per_step_records, attempt.step_record
                        )
                        host._flush_partial_manifest()
                    host.emit_progress(
                        "visual_qa",
                        (
                            "Visual QA blocked sealed renderer "
                            f"{attempt.step.step_id}; coder repair was not "
                            "authorized."
                        ),
                        status="error",
                        run_id=host.run_id,
                        step_id=attempt.step.step_id,
                        current_step=attempt.step_current,
                        total_steps=host.total_steps,
                    )
                    return _CandidateLoopAction.RETURN
                host.emit_progress(
                    "visual_qa",
                    (
                        "Cosmetic visual QA findings were retained as "
                        "warnings for sealed renderer "
                        f"{attempt.step.step_id}; its verified code and outputs "
                        "were not rewritten."
                    ),
                    status="warning",
                    run_id=host.run_id,
                    step_id=attempt.step.step_id,
                    current_step=attempt.step_current,
                    total_steps=host.total_steps,
                )
            elif visual_repair_decision.action is VisualRepairAction.EXHAUSTED:
                fallback_code = attempt._deterministic_fallback_code("visual_qa")
                if fallback_code is not None:
                    state.code = fallback_code
                    return _CandidateLoopAction.CONTINUE
                demoted_findings = list(state.visual_gate.demoted_findings)
                blocking_visual_errors = list(state.visual_gate.blocking_errors)
                attempt.step_record["visual_findings"] = [
                    finding.model_dump() for finding in demoted_findings
                ]
                with host.shared_lock:
                    host.findings.extend(demoted_findings)
                attempt.step_record["visual_qa_demoted"] = state.visual_gate.was_demoted
                if blocking_visual_errors:
                    attempt.step_record["status"] = "execution_failed"
                    with host.shared_lock:
                        host._append_terminal_step_record(
                            host.per_step_records, attempt.step_record
                        )
                        host._flush_partial_manifest()
                    host.emit_progress(
                        "visual_qa",
                        (
                            f"Visual QA blocked {attempt.step.step_id} after "
                            f"{attempt.worker_progress.visual_repair_attempts} layout repair "
                            "attempts."
                        ),
                        status="error",
                        run_id=host.run_id,
                        step_id=attempt.step.step_id,
                        current_step=attempt.step_current,
                        total_steps=host.total_steps,
                    )
                    return _CandidateLoopAction.RETURN
                host.emit_progress(
                    "visual_qa",
                    (
                        f"Cosmetic visual QA findings demoted to warning "
                        f"for {attempt.step.step_id} after "
                        f"{attempt.worker_progress.visual_repair_attempts} layout repair attempts."
                    ),
                    status="warning",
                    run_id=host.run_id,
                    step_id=attempt.step.step_id,
                    current_step=attempt.step_current,
                    total_steps=host.total_steps,
                )
                # Fall through to contract checks and evidence
                # registration only when every remaining visual
                # error was a deterministic layout/cosmetic issue.
            else:
                attempt.worker_progress.visual_repair_attempts += 1
                visual_host_guidance = visual_repair_decision.host_guidance
                current_visual_repair_authority = RepairPromptAuthority.create(
                    typed_ticket=list(visual_repair_decision.repair_ticket),
                    host_guidance=visual_host_guidance,
                )
                visual_repair_authority = RepairPromptAuthority.create(
                    typed_ticket=[
                        *visual_repair_decision.repair_ticket,
                        *attempt._monotonic_concept_constraint_ticket(),
                    ],
                    host_guidance=visual_host_guidance,
                )
                visual_repair_log = visual_repair_decision.repair_log
                if not attempt._consume_llm_repair_budget(
                    "visual",
                    before_code=state.code,
                    repair_ticket=visual_repair_log,
                    repair_authority=visual_repair_authority,
                    current_repair_authority=(current_visual_repair_authority),
                    provider_category="visual_repair",
                    failure_status="visual_failed",
                ):
                    raise AssertionError("LLM repair budget changed without mutation")
                attempt.worker_progress.repair_attempts += 1
                attempt.step_record["code_repair_attempts"] = (
                    attempt.worker_progress.repair_attempts
                )
                attempt.step_record["visual_repair_attempts"] = (
                    attempt.worker_progress.visual_repair_attempts
                )
                host.emit_progress(
                    "visual_qa",
                    f"Repairing figure layout for {attempt.step.step_id}.",
                    run_id=host.run_id,
                    step_id=attempt.step.step_id,
                    current_step=attempt.step_current,
                    total_steps=host.total_steps,
                    repair_attempts=attempt.worker_progress.repair_attempts,
                    visual_repair_attempts=attempt.worker_progress.visual_repair_attempts,
                )
                try:
                    state.code = attempt._repair_with_capsule(
                        failure_status="visual_failed",
                        context=attempt.coder_context,
                        step=attempt.step,
                        code=state.code,
                        run_log=visual_repair_log,
                        repair_authority=visual_repair_authority,
                        current_repair_authority=(current_visual_repair_authority),
                        attempt=attempt.worker_progress.visual_repair_attempts,
                        provider_budget=attempt.provider_budget,
                        provider_category="visual_repair",
                        logical_repair_attempt_id=(
                            attempt.step_repair_budget.llm_repair_attempts
                        ),
                    )
                    attempt._sync_provider_budget()
                    attempt.worker_progress.llm_repair_used = True
                    return _CandidateLoopAction.CONTINUE
                except (
                    ProviderCallBudgetReceiptError,
                    StepAuthorityRuntimeError,
                    StepAuthorityCapsuleError,
                ):
                    raise
                except Exception as exc:
                    attempt._sync_provider_budget()
                    demoted_findings = list(state.visual_gate.demoted_findings)
                    blocking_visual_errors = list(state.visual_gate.blocking_errors)
                    if not blocking_visual_errors:
                        provider_finding = ValidationFinding(
                            validator="coder",
                            severity="warning",
                            message=(
                                "Cosmetic visual-layout repair was "
                                f"unavailable for step {attempt.step.step_id}; "
                                "the current data-valid artifacts were "
                                f"retained: {exc}"
                            ),
                            detail={
                                "step_id": attempt.step.step_id,
                                "error_type": type(exc).__name__,
                                "visual_repair_attempts": (
                                    attempt.worker_progress.visual_repair_attempts
                                ),
                            },
                        )
                        attempt.step_record["visual_findings"] = [
                            finding.model_dump() for finding in demoted_findings
                        ]
                        attempt.step_record["visual_qa_demoted"] = True
                        attempt.step_record["visual_repair_provider_failed"] = True
                        with host.shared_lock:
                            host.findings.extend(demoted_findings)
                            host.findings.append(provider_finding)
                        host.emit_progress(
                            "visual_qa",
                            (
                                "Cosmetic visual repair unavailable; "
                                f"retained current artifacts for {attempt.step.step_id}."
                            ),
                            status="warning",
                            run_id=host.run_id,
                            step_id=attempt.step.step_id,
                            current_step=attempt.step_current,
                            total_steps=host.total_steps,
                        )
                    else:
                        fallback_code = attempt._deterministic_fallback_code(
                            "visual_qa_repair_failed"
                        )
                        if fallback_code is not None:
                            state.code = fallback_code
                            return _CandidateLoopAction.CONTINUE
                        with host.shared_lock:
                            host.findings.extend(visual_findings)
                            host.findings.append(
                                ValidationFinding(
                                    validator="coder",
                                    severity="error",
                                    message=(
                                        "Coder repair failed after visual QA "
                                        f"for step {attempt.step.step_id}: {exc}"
                                    ),
                                )
                            )
                            attempt.step_record["status"] = "repair_failed"
                            host._append_terminal_step_record(
                                host.per_step_records, attempt.step_record
                            )
                            host._flush_partial_manifest()
                        host.emit_progress(
                            "visual_qa",
                            f"Visual QA repair failed for {attempt.step.step_id}.",
                            status="error",
                            run_id=host.run_id,
                            step_id=attempt.step.step_id,
                            current_step=attempt.step_current,
                            total_steps=host.total_steps,
                        )
                        return _CandidateLoopAction.RETURN
    return _CandidateLoopAction.PROCEED


def _candidate_contract_setup_transition(
    host: _CandidateLoopHost, attempt: _CandidateLoopAttempt, state: _CandidateLoopState
) -> _CandidateLoopAction:
    with host.shared_lock:
        completed_records_snapshot = list(host.per_step_records)
    # Early pre-registration deterministic contract gate: the SAME
    # 15-validator sequence the final gate runs
    # (_evaluate_final_deterministic_gates), evaluated here before
    # evidence registration so contract errors enter the in-run repair
    # loop instead of becoming a terminal record. The figure-contract
    # canonicalization repair and the figure-contract / figure-source /
    # ordered-stratified validators stay below because the early gate
    # interleaves the canonicalization repair between them.
    state.early_contract_findings = _step_deterministic_contract_findings(
        step=attempt.step,
        plan=host.plan,
        context=host.context,
        step_summary=state.visual_step_summary,
        completed_step_records=completed_records_snapshot,
        resolved_input_bindings=attempt.resolved_input_bindings,
        effect_output_is_authorized=effect_output_authorized(
            attempt.step,
            step_record=attempt.step_record,
        ),
        out_dir=state.run_result.out_dir,
        run_dir=host.run_dir,
        universe_path=host.universe_path,
        cohort_path=host.cohort_path,
        execution_cohort_path=attempt.step_execution_cohort_path,
        cross_step_cohort_lock_validator=host.cross_step_cohort_lock_validator,
        cross_step_registered_output_validator=(
            host.cross_step_registered_output_validator
        ),
        cross_step_reconciliation_trace_validator=(
            host.cross_step_reconciliation_trace_validator
        ),
        step_summary_integrity_validator=host.step_summary_integrity_validator,
        step_summary_fraction_validator=host.step_summary_fraction_validator,
        cross_step_source_status_validator=(host.cross_step_source_status_validator),
        primary_model_contract_validator=host.primary_model_contract_validator,
    )
    # Figure quality and source-data errors must enter the same
    # in-run repair loop as table/model contract errors. Checking
    # them only after evidence registration produces a terminal
    # contract_failed record with no opportunity to repair the
    # generated rendering script.
    for contract_path in sorted(
        state.run_result.out_dir.glob("*.figure_contract.json")
    ):
        schema_candidate = _figure_contract_source_data_canonicalization_candidate(
            contract_path=contract_path,
            out_dir=state.run_result.out_dir,
        )
        if schema_candidate is None:
            continue
        before_contract, after_contract, source_names = schema_candidate
        repair_id = _FIGURE_CONTRACT_SOURCE_DATA_SCHEMA_REPAIR_ID
        if not host._automatic_repair_authorized(
            repair_id,
            step=attempt.step,
            source="figure_contract_schema_canonicalization",
            before_code=before_contract,
            after_code=after_contract,
        ):
            continue
        _install_figure_contract_source_data_canonicalization(
            contract_path=contract_path,
            expected_before=before_contract,
            canonical_text=after_contract,
        )
        attempt.step_record.setdefault(
            "figure_contract_schema_canonicalizations", []
        ).append(
            {
                "contract": contract_path.name,
                "source_data": list(source_names),
                "repair_id": repair_id,
            }
        )
        host._record_repair(
            repair_id=repair_id,
            step_id=str(attempt.step.step_id),
            trigger={
                "source": "figure_contract_schema_canonicalization",
                "contract": contract_path.name,
            },
            transformation=(
                "Canonicalized an exact local source-data descriptor "
                "to the persistent flat FigureContract basename schema."
            ),
            before_code=before_contract,
            after_code=after_contract,
        )
    # Figure-contract canonicalization repair (above) must run before
    # these figure audits; keep the ordering by calling the post-canon
    # figure findings helper here, after the repair.
    state.early_contract_findings += _post_canonicalization_figure_findings(
        step=attempt.step,
        out_dir=state.run_result.out_dir,
        run_dir=host.run_dir,
        step_summary=state.visual_step_summary,
        completed_step_records=completed_records_snapshot,
        resolved_input_bindings=attempt.resolved_input_bindings,
        execution_cohort_path=attempt.step_execution_cohort_path,
        figure_contract_validator=host.figure_contract_validator,
        figure_source_validator=host.figure_source_validator,
    )
    state.early_contract_findings += validate_step_planned_figure_contract_binding(
        step=attempt.step,
        out_dir=state.run_result.out_dir,
        step_summary=state.visual_step_summary,
    )
    unowned_sealed_markers = host._unowned_sealed_authority_markers(
        state.visual_step_summary,
        authorized_code_sha256=(attempt.sealed_renderer_authorized_code_sha256),
    )
    if unowned_sealed_markers:
        state.early_contract_findings.append(
            ValidationFinding(
                validator="sealed_renderer_authority",
                severity="error",
                message=(
                    "Generated code reported sealed-renderer authority "
                    "that the host did not authorize before execution."
                ),
                detail={
                    "step_id": attempt.step.step_id,
                    "unowned_authority_markers": unowned_sealed_markers,
                },
            )
        )
    if (
        attempt.sealed_renderer_authorized_code_sha256 is not None
        and state.visual_step_summary.get("rendering_only") is not True
    ):
        state.early_contract_findings.append(
            ValidationFinding(
                validator="sealed_renderer_authority",
                severity="error",
                message=(
                    "The authorized figure adapter did not report its "
                    "required rendering-only execution scope."
                ),
                detail={
                    "step_id": attempt.step.step_id,
                    "repair_id": attempt.sealed_renderer_state.repair_id,
                    "reported_rendering_only": state.visual_step_summary.get(
                        "rendering_only"
                    ),
                },
            )
        )
    state.early_contract_findings.extend(
        validate_and_record_sealed_renderer_receipt(
            state=attempt.sealed_renderer_state,
            authorized_code_sha256=(attempt.sealed_renderer_authorized_code_sha256),
            visual_step_summary=state.visual_step_summary,
            run_dir=host.run_dir,
            step_id=attempt.step.step_id,
            step_record=attempt.step_record,
        )
    )
    # PRIMARY runners keep their trustworthy core estimate.
    state.early_contract_findings = _demote_step_contract_for_primary_runner(
        attempt.step_record, state.visual_step_summary, state.early_contract_findings
    )
    state.early_contract_findings += host._fresh_plausibility_receipt_findings(
        state.visual_step_summary,
        attempt.step,
        state.code,
        attempt.plausibility_authority,
    )
    state.early_contract_errors = [
        f for f in state.early_contract_findings if f.severity == "error"
    ]
    return _CandidateLoopAction.PROCEED


def _candidate_contract_repair_transition(
    host: _CandidateLoopHost, attempt: _CandidateLoopAttempt, state: _CandidateLoopState
) -> _CandidateLoopAction:
    if state.early_contract_errors:
        # Record the reason here, where the host decides the
        # contract failed -- not on whichever terminal branch
        # happens to fire later.  A repaired draft can be
        # quarantined before it ever executes, and that branch
        # writes its own status; without this the only surviving
        # trace is a bare ``contract_repair_attempts`` counter and
        # the record reads as if the quarantine were the cause.
        attempt.step_record.setdefault("contract_repair_triggers", []).append(
            [finding.model_dump() for finding in state.early_contract_errors]
        )
        locked_data_quality_issues = host._locked_measurement_data_quality_issues(
            state.early_contract_errors
        )
        if locked_data_quality_issues:
            attempt.step_record.update(
                {
                    "status": "contract_failed",
                    "diagnostic_only": True,
                    "measurement_provenance_repair_suppressed": True,
                    "measurement_provenance_terminal_reason": (
                        "locked_cohort_data_quality_failed"
                    ),
                    "measurement_provenance_terminal_issues": (
                        locked_data_quality_issues
                    ),
                    "contract_findings": [
                        finding.model_dump()
                        for finding in state.early_contract_findings
                    ],
                    "step_summary": state.visual_step_summary,
                    "llm_repair_used": attempt.worker_progress.llm_repair_used,
                    "generation_mode": state.current_generation_mode,
                    "code_repair_attempts": attempt.worker_progress.repair_attempts,
                    "contract_repair_attempts": (
                        attempt.worker_progress.contract_repair_attempts
                    ),
                }
            )
            with host.shared_lock:
                host.findings.extend(state.early_contract_findings)
                host._append_terminal_step_record(
                    host.per_step_records, attempt.step_record
                )
                host._flush_partial_manifest()
            host.emit_progress(
                "contract",
                (
                    "Locked-cohort measurement provenance failed for "
                    f"{attempt.step.step_id}; retained diagnostics without "
                    "attempting a code repair."
                ),
                status="error",
                run_id=host.run_id,
                step_id=attempt.step.step_id,
                current_step=attempt.step_current,
                total_steps=host.total_steps,
            )
            return _CandidateLoopAction.RETURN
        if attempt.sealed_renderer_authorized_code_sha256 is not None:
            attempt.step_record.update(
                {
                    "status": "contract_failed",
                    "diagnostic_only": True,
                    "sealed_renderer_contract_repair_suppressed": True,
                    "sealed_renderer_terminal_reason": ("output_contract_failed"),
                    "contract_findings": [
                        finding.model_dump()
                        for finding in state.early_contract_findings
                    ],
                    "step_summary": state.visual_step_summary,
                    "llm_repair_used": False,
                    "generation_mode": "fallback",
                }
            )
            with host.shared_lock:
                host.findings.extend(state.early_contract_findings)
                host._append_terminal_step_record(
                    host.per_step_records, attempt.step_record
                )
                host._flush_partial_manifest()
            host.emit_progress(
                "contract",
                (
                    "Contract validation blocked sealed renderer "
                    f"{attempt.step.step_id}; its code and outputs were retained "
                    "without coder repair."
                ),
                status="error",
                run_id=host.run_id,
                step_id=attempt.step.step_id,
                current_step=attempt.step_current,
                total_steps=host.total_steps,
            )
            return _CandidateLoopAction.RETURN
        if attempt.is_trajectory_stability_standard:
            state.standard_executor_terminal_block = True
            state.standard_executor_terminal_reason = "executor_output_contract_failed"
            state.standard_executor_terminal_summary = dict(state.visual_step_summary)
            state.standard_executor_terminal_findings = list(
                state.early_contract_findings
            )
            return _CandidateLoopAction.BREAK
        if host.pipeline._enable_deterministic_runner_repair:
            before_repair_code = state.code
            summary_repair = _deterministic_summary_repair(
                code=state.code,
                step_summary=state.visual_step_summary,
                previous_repair=attempt.worker_progress.runner_repair_name,
                analysis_family=attempt.local_runtime_state.analysis_family,
                on_semantic_escalation=(
                    attempt.step_repair_budget.record_semantic_escalation
                ),
            )
            summary_repair = host._authorize_automatic_repair(
                summary_repair,
                step=attempt.step,
                source="deterministic_summary_repair_before_contract",
                before_code=before_repair_code,
            )
        else:
            summary_repair = None
        if summary_repair is not None:
            attempt.worker_progress.contract_repair_attempts += 1
            attempt.worker_progress.repair_attempts += 1
            attempt.worker_progress.runner_repair_name, state.code = summary_repair
            attempt.step_record["runner_repair"] = (
                attempt.worker_progress.runner_repair_name
            )
            attempt.step_record["code_repair_attempts"] = (
                attempt.worker_progress.repair_attempts
            )
            attempt.step_record["contract_repair_attempts"] = (
                attempt.worker_progress.contract_repair_attempts
            )
            host._record_repair(
                repair_id=attempt.worker_progress.runner_repair_name,
                step_id=attempt.step.step_id,
                trigger={
                    "source": "deterministic_summary_repair",
                    "step_summary_keys": sorted(
                        str(key) for key in state.visual_step_summary.keys()
                    ),
                    "contract_findings": [
                        f.message for f in state.early_contract_errors
                    ],
                },
                transformation=("Deterministic repair before LLM contract repair."),
                before_code=before_repair_code,
                after_code=state.code,
            )
            host.emit_progress(
                "runner_repair",
                (
                    f"Applied deterministic summary repair for "
                    f"{attempt.step.step_id}: {attempt.worker_progress.runner_repair_name}."
                ),
                run_id=host.run_id,
                step_id=attempt.step.step_id,
                current_step=attempt.step_current,
                total_steps=host.total_steps,
            )
            return _CandidateLoopAction.CONTINUE
        if host.pipeline._enable_deterministic_runner_repair:
            before_repair_code = state.code
            contract_repair = deterministic_contract_repair(
                code=state.code,
                findings=state.early_contract_errors,
                previous_repair=attempt.worker_progress.runner_repair_name,
                on_semantic_escalation=(
                    attempt.step_repair_budget.record_semantic_escalation
                ),
            )
            contract_repair = host._authorize_automatic_repair(
                contract_repair,
                step=attempt.step,
                source="deterministic_contract_repair",
                before_code=before_repair_code,
            )
        else:
            contract_repair = None
        if contract_repair is not None:
            attempt.worker_progress.contract_repair_attempts += 1
            attempt.worker_progress.repair_attempts += 1
            attempt.worker_progress.runner_repair_name, state.code = contract_repair
            attempt.step_record["runner_repair"] = (
                attempt.worker_progress.runner_repair_name
            )
            attempt.step_record["code_repair_attempts"] = (
                attempt.worker_progress.repair_attempts
            )
            attempt.step_record["contract_repair_attempts"] = (
                attempt.worker_progress.contract_repair_attempts
            )
            host._record_repair(
                repair_id=attempt.worker_progress.runner_repair_name,
                step_id=attempt.step.step_id,
                trigger={
                    "source": "deterministic_contract_repair",
                    "contract_findings": [
                        f.message for f in state.early_contract_errors
                    ],
                },
                transformation=(
                    "Applied a centrally authorized deterministic source "
                    "transformation for objective contract findings."
                ),
                before_code=before_repair_code,
                after_code=state.code,
            )
            host.emit_progress(
                "runner_repair",
                (
                    f"Applied deterministic contract repair for "
                    f"{attempt.step.step_id}: {attempt.worker_progress.runner_repair_name}."
                ),
                run_id=host.run_id,
                step_id=attempt.step.step_id,
                current_step=attempt.step_current,
                total_steps=host.total_steps,
            )
            return _CandidateLoopAction.CONTINUE
        if (
            attempt.worker_progress.llm_contract_repair_attempts
            >= host.pipeline._max_code_repair_attempts
            or not attempt._llm_repair_budget_available()
        ):
            with host.shared_lock:
                host.findings.extend(state.early_contract_findings)
                attempt.step_record["status"] = "contract_failed"
                attempt.step_record["contract_findings"] = [
                    f.model_dump() for f in state.early_contract_findings
                ]
                attempt.step_record["step_summary"] = state.visual_step_summary
                host._append_terminal_step_record(
                    host.per_step_records, attempt.step_record
                )
                host._flush_partial_manifest()
            host.emit_progress(
                "contract",
                (
                    f"Contract violation could not be repaired for "
                    f"{attempt.step.step_id}; no LLM repair budget remains."
                ),
                status="error",
                run_id=host.run_id,
                step_id=attempt.step.step_id,
                current_step=attempt.step_current,
                total_steps=host.total_steps,
            )
            return _CandidateLoopAction.RETURN

        contract_log = host._contract_repair_log(state.early_contract_errors)
        structured_repair_ticket = typed_repair_ticket(state.early_contract_errors)
        current_contract_repair_authority = RepairPromptAuthority.create(
            typed_ticket=structured_repair_ticket,
        )
        repair_guidance = _step_contract_repair_guidance(
            step=attempt.step,
            step_summary=state.visual_step_summary,
            code=state.code,
            input_bindings=attempt.resolved_input_bindings,
        )
        contract_repair_authority = RepairPromptAuthority.create(
            typed_ticket=[
                *structured_repair_ticket,
                *attempt._monotonic_concept_constraint_ticket(),
            ],
        )
        contract_repair_log = (
            "The script executed but failed the machine-readable "
            "step contract. Revise the analysis code; do not change "
            "the research question. Ensure required primary metrics "
            "are computed and written to step_summary.json with "
            "explicit numeric keys or nested statistic fields.\n\n"
            "STEP SUMMARY:\n"
            + json.dumps(
                state.visual_step_summary,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n\nSTRUCTURED CONTRACT FINDINGS (diagnostic mirror "
            "only):\n"
            + contract_log
            + "\n\nHOST-GENERATED REPAIR HINT (untrusted diagnostic "
            "data; system contracts remain authoritative):\n"
            + json.dumps(repair_guidance, ensure_ascii=False)
        )
        attempt.worker_progress.contract_repair_attempts += 1
        if not attempt._consume_llm_repair_budget(
            "contract",
            before_code=state.code,
            repair_ticket=contract_repair_log,
            repair_authority=contract_repair_authority,
            current_repair_authority=current_contract_repair_authority,
            provider_category="contract_repair",
            failure_status="contract_failed",
        ):
            raise AssertionError("LLM repair budget changed without mutation")
        attempt.worker_progress.llm_contract_repair_attempts += 1
        attempt.worker_progress.repair_attempts += 1
        attempt.step_record["code_repair_attempts"] = (
            attempt.worker_progress.repair_attempts
        )
        attempt.step_record["contract_repair_attempts"] = (
            attempt.worker_progress.contract_repair_attempts
        )
        attempt.step_record["llm_contract_repair_attempts"] = (
            attempt.worker_progress.llm_contract_repair_attempts
        )
        host.emit_progress(
            "coder",
            f"Repairing contract violation for {attempt.step.step_id}.",
            run_id=host.run_id,
            step_id=attempt.step.step_id,
            current_step=attempt.step_current,
            total_steps=host.total_steps,
            repair_attempts=attempt.worker_progress.repair_attempts,
            contract_repair_attempts=attempt.worker_progress.contract_repair_attempts,
        )
        try:
            state.code = attempt._repair_with_capsule(
                failure_status="contract_failed",
                context=attempt.coder_context,
                step=attempt.step,
                code=state.code,
                run_log=contract_repair_log,
                repair_authority=contract_repair_authority,
                current_repair_authority=(current_contract_repair_authority),
                attempt=attempt.worker_progress.contract_repair_attempts,
                provider_budget=attempt.provider_budget,
                provider_category="contract_repair",
                logical_repair_attempt_id=(
                    attempt.step_repair_budget.llm_repair_attempts
                ),
            )
            attempt._sync_provider_budget()
            attempt.worker_progress.llm_repair_used = True
            return _CandidateLoopAction.CONTINUE
        except (
            ProviderCallBudgetReceiptError,
            StepAuthorityRuntimeError,
            StepAuthorityCapsuleError,
        ):
            raise
        except Exception as exc:
            attempt._sync_provider_budget()
            fallback_code = attempt._deterministic_fallback_code(
                "contract_repair_failed"
            )
            if fallback_code is not None:
                state.code = fallback_code
                return _CandidateLoopAction.CONTINUE
            with host.shared_lock:
                host.findings.extend(state.early_contract_findings)
                host.findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="error",
                        message=(
                            f"Coder repair failed after contract check "
                            f"for step {attempt.step.step_id}: {exc}"
                        ),
                    )
                )
                attempt.step_record["status"] = "repair_failed"
                attempt.step_record["contract_findings"] = [
                    f.model_dump() for f in state.early_contract_findings
                ]
                attempt.step_record["step_summary"] = state.visual_step_summary
                host._append_terminal_step_record(
                    host.per_step_records, attempt.step_record
                )
                host._flush_partial_manifest()
            host.emit_progress(
                "coder",
                f"Contract repair failed for {attempt.step.step_id}.",
                status="error",
                run_id=host.run_id,
                step_id=attempt.step.step_id,
                current_step=attempt.step_current,
                total_steps=host.total_steps,
            )
            return _CandidateLoopAction.RETURN
    return _CandidateLoopAction.PROCEED


def _candidate_summary_transition(
    host: _CandidateLoopHost, attempt: _CandidateLoopAttempt, state: _CandidateLoopState
) -> _CandidateLoopAction:
    if (
        host.pipeline._enable_deterministic_runner_repair
        and attempt.sealed_renderer_authorized_code_sha256 is None
    ):
        before_repair_code = state.code
        summary_repair = _deterministic_summary_repair(
            code=state.code,
            step_summary=state.visual_step_summary,
            previous_repair=attempt.worker_progress.runner_repair_name,
            analysis_family=attempt.local_runtime_state.analysis_family,
            on_semantic_escalation=(
                attempt.step_repair_budget.record_semantic_escalation
            ),
        )
        summary_repair = host._authorize_automatic_repair(
            summary_repair,
            step=attempt.step,
            source="deterministic_summary_repair_after_contract",
            before_code=before_repair_code,
        )
    else:
        summary_repair = None
    if summary_repair is not None:
        attempt.worker_progress.runner_repair_name, state.code = summary_repair
        attempt.step_record["runner_repair"] = (
            attempt.worker_progress.runner_repair_name
        )
        host._record_repair(
            repair_id=attempt.worker_progress.runner_repair_name,
            step_id=attempt.step.step_id,
            trigger={
                "source": "deterministic_summary_repair",
                "step_summary_keys": sorted(
                    str(key) for key in state.visual_step_summary.keys()
                ),
            },
            transformation="Deterministic repair after step_summary contract inspection.",
            before_code=before_repair_code,
            after_code=state.code,
        )
        host.emit_progress(
            "runner_repair",
            f"Applied deterministic summary repair for {attempt.step.step_id}: {attempt.worker_progress.runner_repair_name}.",
            run_id=host.run_id,
            step_id=attempt.step.step_id,
            current_step=attempt.step_current,
            total_steps=host.total_steps,
        )
        return _CandidateLoopAction.CONTINUE
    state.deterministic_contract_approved_code_digest = state.candidate_code_digest
    attempt.step_record["deterministic_contract_approved_code_sha256"] = (
        state.deterministic_contract_approved_code_digest
    )
    # Return once through the digest gate for the single final LLM
    # concept audit.  The output directory is intentionally retained;
    # on approval it proceeds without re-executing unchanged code.
    #
    # An unreachable `return _CandidateLoopAction.PROCEED` sat below this line
    # until 2026-08-22 -- a leftover from before the digest gate took over.
    # It changed nothing at runtime but read as a second live branch, which is
    # the wrong thing for a reader auditing this state machine to believe.
    return _CandidateLoopAction.CONTINUE


def _candidate_failure_transition(
    host: _CandidateLoopHost, attempt: _CandidateLoopAttempt, state: _CandidateLoopState
) -> _CandidateLoopAction:
    if state.log_path.exists():
        run_log = state.log_path.read_text(encoding="utf-8", errors="replace")
    else:
        run_log = (
            (state.run_result.stdout or "") + "\n" + (state.run_result.stderr or "")
        )
    if attempt.is_trajectory_stability_standard:
        # A timeout can interrupt the standard executor between its
        # private streaming write and atomic rename.  That file is an
        # implementation detail, not a diagnostic product, and must
        # be gone before the generic output-directory scan below can
        # register it as evidence.
        host._remove_standard_executor_pending_artifacts(state.run_result.out_dir)
        state.standard_executor_terminal_block = True
        state.standard_executor_terminal_reason = "executor_runtime_failure"
        # This branch is already terminal and already spends no repair,
        # so it is safe. It is not diagnosable: the executor with the
        # largest wall clock in the pipeline reports the same generic
        # reason whether it raised in its first second or was killed an
        # hour in. Name the timeout in the vocabulary generated code
        # already uses, and leave the terminal decision above untouched.
        #
        # Only the timeout class is adopted. The classifier also reads a
        # plan/data contract failure out of the log text, and this
        # executor's log is not the log that vocabulary was written for.
        timeout_decision = (
            classify_runtime_failure(
                run_log=run_log,
                timed_out=True,
                step_id=attempt.step.step_id,
                returncode=state.run_result.returncode,
                timeout_seconds=state.execution_timeout_seconds,
                deterministic_executor_used=True,
            )
            if state.run_result.timed_out
            else None
        )
        if timeout_decision is not None:
            attempt.step_record["runtime_failure_class"] = (
                timeout_decision.step_updates["runtime_failure_class"]
            )
            with host.shared_lock:
                host.findings.append(timeout_decision.finding)
        return _CandidateLoopAction.BREAK
    runtime_failure = classify_runtime_failure(
        run_log=run_log,
        timed_out=bool(state.run_result.timed_out),
        step_id=attempt.step.step_id,
        returncode=state.run_result.returncode,
        timeout_seconds=state.execution_timeout_seconds,
        deterministic_executor_used=bool(
            attempt.worker_progress.deterministic_standard_executor_used
        ),
        runner_failure_code=state.run_result.runner_failure_code,
    )
    if runtime_failure is not None:
        attempt.step_record.update(runtime_failure.step_updates)
        with host.shared_lock:
            host.findings.append(runtime_failure.finding)
            host._append_terminal_step_record(
                host.per_step_records, attempt.step_record
            )
            host._flush_partial_manifest()
        host.emit_progress(
            "runner",
            runtime_failure.progress_message,
            status="error",
            run_id=host.run_id,
            step_id=attempt.step.step_id,
            current_step=attempt.step_current,
            total_steps=host.total_steps,
        )
        return _CandidateLoopAction.RETURN
    if attempt.sealed_renderer_authorized_code_sha256 is not None:
        runtime_finding = ValidationFinding(
            validator="sealed_renderer_authority",
            severity="error",
            message=(
                "The authorized rendering-only adapter failed at runtime; "
                "its diagnostics were retained and no deterministic or LLM "
                "code repair was allowed."
            ),
            detail={
                "step_id": attempt.step.step_id,
                "repair_id": attempt.sealed_renderer_state.repair_id,
                "returncode": state.run_result.returncode,
                "timed_out": state.run_result.timed_out,
            },
        )
        attempt.step_record.update(
            {
                "status": "execution_failed",
                "diagnostic_only": True,
                "sealed_renderer_runtime_repair_suppressed": True,
                "sealed_renderer_terminal_reason": "runtime_failure",
                "llm_repair_used": False,
                "generation_mode": "fallback",
            }
        )
        with host.shared_lock:
            host.findings.append(runtime_finding)
            host._append_terminal_step_record(
                host.per_step_records, attempt.step_record
            )
            host._flush_partial_manifest()
        host.emit_progress(
            "runner",
            (
                f"Sealed renderer failed for {attempt.step.step_id}; coder repair "
                "was not authorized."
            ),
            status="error",
            run_id=host.run_id,
            step_id=attempt.step.step_id,
            current_step=attempt.step_current,
            total_steps=host.total_steps,
        )
        return _CandidateLoopAction.RETURN
    if host.pipeline._enable_deterministic_runner_repair:
        before_repair_code = state.code
        plugin_repair = host.pipeline._case_plugin_registry.repair_code(
            context=host.context,
            step=attempt.step,
            code=state.code,
            run_log=(run_log + attempt._monotonic_concept_constraint_log()),
        )
        if (
            plugin_repair is not None
            and plugin_repair[0] != attempt.worker_progress.runner_repair_name
        ):
            runner_repair = plugin_repair
        else:
            runner_repair = _deterministic_runner_repair(
                code=state.code,
                run_log=run_log,
                previous_repair=attempt.worker_progress.runner_repair_name,
                analysis_family=attempt.local_runtime_state.analysis_family,
                resolved_input_bindings=attempt.resolved_input_bindings,
                on_semantic_escalation=(
                    attempt.step_repair_budget.record_semantic_escalation
                ),
            )
        runner_repair = host._authorize_automatic_repair(
            runner_repair,
            step=attempt.step,
            source=(
                "case_plugin_repair"
                if plugin_repair is not None and runner_repair is plugin_repair
                else "deterministic_runner_repair"
            ),
            before_code=before_repair_code,
        )
        record_deterministic_runner_repair_attempt(
            attempt.step_record,
            code=before_repair_code,
            run_log=run_log,
            previous_repair=attempt.worker_progress.runner_repair_name,
            outcome=("applied" if runner_repair is not None else "declined"),
            repair_id=(runner_repair[0] if runner_repair is not None else None),
        )
    else:
        runner_repair = None
        record_deterministic_runner_repair_attempt(
            attempt.step_record,
            code=state.code,
            run_log=run_log,
            previous_repair=attempt.worker_progress.runner_repair_name,
            outcome="disabled",
            repair_id=None,
        )
    if runner_repair is not None:
        attempt.worker_progress.runner_repair_name, state.code = runner_repair
        attempt.step_record["runner_repair"] = (
            attempt.worker_progress.runner_repair_name
        )
        host._record_repair(
            repair_id=attempt.worker_progress.runner_repair_name,
            step_id=attempt.step.step_id,
            trigger={
                "source": "deterministic_runner_repair",
                "run_log_tail": run_log[-1200:],
            },
            transformation="Deterministic repair after runner failure.",
            before_code=before_repair_code,
            after_code=state.code,
        )
        host.emit_progress(
            "runner_repair",
            f"Applied deterministic runner repair for {attempt.step.step_id}: {attempt.worker_progress.runner_repair_name}.",
            run_id=host.run_id,
            step_id=attempt.step.step_id,
            current_step=attempt.step_current,
            total_steps=host.total_steps,
        )
        return _CandidateLoopAction.CONTINUE
    if (
        attempt.worker_progress.runtime_repair_attempts
        >= host.pipeline._max_code_repair_attempts
        or not attempt._llm_repair_budget_available("runtime")
    ):
        fallback_code = attempt._deterministic_fallback_code("execution_failure")
        if fallback_code is not None:
            state.code = fallback_code
            return _CandidateLoopAction.CONTINUE
        with host.shared_lock:
            host.findings.append(
                ValidationFinding(
                    validator="runner",
                    severity="error",
                    message=(
                        f"Step {attempt.step.step_id} "
                        f"{'timed out' if state.run_result.timed_out else 'failed'} "
                        f"with returncode {state.run_result.returncode}."
                    ),
                )
            )
            attempt.step_record["status"] = "execution_failed"
            host._append_terminal_step_record(
                host.per_step_records, attempt.step_record
            )
            host._flush_partial_manifest()
        host.emit_progress(
            "runner",
            f"Execution failed for {attempt.step.step_id}.",
            status="error",
            run_id=host.run_id,
            step_id=attempt.step.step_id,
            current_step=attempt.step_current,
            total_steps=host.total_steps,
        )
        return _CandidateLoopAction.RETURN
    runtime_repair_applied = False
    runtime_repair_fallback_applied = False
    runtime_repair_authority = RepairPromptAuthority()
    while (
        attempt.worker_progress.runtime_repair_attempts
        < host.pipeline._max_code_repair_attempts
        and attempt._llm_repair_budget_available("runtime")
    ):
        attempt.worker_progress.repair_attempts += 1
        attempt.worker_progress.runtime_repair_attempts += 1
        if not attempt._consume_llm_repair_budget(
            "runtime",
            before_code=state.code,
            repair_ticket=run_log,
            repair_authority=runtime_repair_authority,
            provider_category="runtime_repair",
            failure_status="runtime_failed",
        ):
            raise AssertionError("LLM repair budget changed without mutation")
        attempt.step_record["code_repair_attempts"] = (
            attempt.worker_progress.repair_attempts
        )
        attempt.step_record["runtime_repair_attempts"] = (
            attempt.worker_progress.runtime_repair_attempts
        )
        host.emit_progress(
            "coder",
            f"Repairing failed script for {attempt.step.step_id}.",
            run_id=host.run_id,
            step_id=attempt.step.step_id,
            current_step=attempt.step_current,
            total_steps=host.total_steps,
            repair_attempts=attempt.worker_progress.repair_attempts,
        )
        try:
            repaired_code = attempt._repair_with_capsule(
                failure_status="runtime_failed",
                context=attempt.coder_context,
                step=attempt.step,
                code=state.code,
                run_log=run_log,
                repair_authority=runtime_repair_authority,
                attempt=attempt.worker_progress.repair_attempts,
                provider_budget=attempt.provider_budget,
                provider_category="runtime_repair",
                logical_repair_attempt_id=(
                    attempt.step_repair_budget.llm_repair_attempts
                ),
            )
            attempt._sync_provider_budget()
            if not host._python_repair_is_materially_changed(state.code, repaired_code):
                attempt.checkpoint_authority.reject_completed_repair_candidate(
                    repaired_code,
                    reason="runtime_repair_semantic_noop",
                )
                raise RuntimeError("Runtime repair returned no material Python change.")
            state.code = repaired_code
            attempt.worker_progress.llm_repair_used = True
            runtime_repair_applied = True
            _clear_output_dir(state.run_result.out_dir)
            break
        except (
            ProviderCallBudgetReceiptError,
            StepAuthorityRuntimeError,
            StepAuthorityCapsuleError,
        ):
            raise
        except Exception as exc:
            attempt._sync_provider_budget()
            # Transport/parse failure did not change the candidate. Retry
            # the repair request itself with the same code and traceback;
            # never pay to execute a digest whose failure is already known.
            message = str(exc).lower()
            is_noop_repair = "no material python change" in message
            is_transient = (
                isinstance(exc, json.JSONDecodeError)
                or "expecting value" in message
                or ("json" in message and "decode" in message)
                or "503" in message
                or "rate" in message
            )
            can_retry_repair = bool(
                (is_transient or is_noop_repair)
                and attempt.worker_progress.runtime_repair_attempts
                < host.pipeline._max_code_repair_attempts
                and attempt._llm_repair_budget_available("runtime")
                and not attempt.provider_budget.exhausted
            )
            if can_retry_repair:
                host.emit_progress(
                    "coder",
                    (
                        f"Repair attempt did not yield usable code for "
                        f"{attempt.step.step_id} "
                        f"(attempt {attempt.worker_progress.repair_attempts}): {type(exc).__name__}; "
                        "retrying the repair without re-executing unchanged code."
                    ),
                    run_id=host.run_id,
                    step_id=attempt.step.step_id,
                    current_step=attempt.step_current,
                    total_steps=host.total_steps,
                    repair_attempts=attempt.worker_progress.repair_attempts,
                )
                continue

            # The causal failure is the unavailable repair, not a new
            # runner failure. Preserve that reason even when the logical
            # or provider-call budget became exhausted on this attempt.
            fallback_code = attempt._deterministic_fallback_code("repair_failed")
            if fallback_code is not None:
                state.code = fallback_code
                runtime_repair_fallback_applied = True
                _clear_output_dir(state.run_result.out_dir)
                break
            with host.shared_lock:
                host.findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="error",
                        message=(
                            f"Coder repair failed for step {attempt.step.step_id}: {exc}"
                        ),
                    )
                )
                attempt.step_record["status"] = "repair_failed"
                host._append_terminal_step_record(
                    host.per_step_records, attempt.step_record
                )
                host._flush_partial_manifest()
            host.emit_progress(
                "coder",
                f"Repair failed for {attempt.step.step_id}.",
                status="error",
                run_id=host.run_id,
                step_id=attempt.step.step_id,
                current_step=attempt.step_current,
                total_steps=host.total_steps,
            )
            return _CandidateLoopAction.RETURN
    if runtime_repair_applied or runtime_repair_fallback_applied:
        return _CandidateLoopAction.CONTINUE
    return _CandidateLoopAction.PROCEED


def _run_candidate_loop(
    host: _CandidateLoopHost,
    attempt: _CandidateLoopAttempt,
    state: _CandidateLoopState,
) -> bool:
    """Run audited candidates until terminal return or accepted-loop exit.

    Returns ``True`` when a stage already committed a terminal step record;
    ``False`` means the caller must continue with post-execution validation.
    """

    while True:
        state.code = reorder_forward_references(state.code)
        # Every candidate, including a repaired rewrite, receives the same
        # host-owned receipt before its authority digest is calculated.
        state.code = _host_plausibility_receipt_injected(
            state.code,
            scope=attempt.plausibility_authority.scope,
            already_satisfied=not _flag_only_plausibility_obligation_findings(
                None,
                script_text=state.code,
                step=attempt.step,
                scope=attempt.plausibility_authority.scope,
            ),
        )
        attempt.checkpoint_authority.ensure_candidate(
            state.code,
            reason="host_code_normalization_or_deterministic_mutation",
        )
        state.candidate_code_digest = sha256_of_bytes(state.code.encode("utf-8"))
        if (
            attempt.sealed_renderer_authorized_code_sha256 is not None
            and state.candidate_code_digest
            != attempt.sealed_renderer_authorized_code_sha256
        ):
            authority_finding = ValidationFinding(
                validator="sealed_renderer_authority",
                severity="error",
                message=(
                    "The active rendering-only adapter no longer matches its "
                    "authorized code digest; execution was blocked without "
                    "running or repairing the mutated code."
                ),
                detail={
                    "step_id": attempt.step.step_id,
                    "repair_id": attempt.sealed_renderer_state.repair_id,
                    "authorized_code_sha256": (
                        attempt.sealed_renderer_authorized_code_sha256
                    ),
                    "candidate_code_sha256": state.candidate_code_digest,
                },
            )
            attempt.step_record.update(
                {
                    "status": "execution_failed",
                    "diagnostic_only": True,
                    "sealed_renderer_terminal_reason": "code_digest_changed",
                    "llm_repair_used": False,
                    "generation_mode": "fallback",
                }
            )
            with host.shared_lock:
                host.findings.append(authority_finding)
                host._append_terminal_step_record(
                    host.per_step_records,
                    attempt.step_record,
                )
                host._flush_partial_manifest()
            return True
        state.final_llm_audit_due = bool(
            state.candidate_code_digest
            == state.deterministic_contract_approved_code_digest
            and state.candidate_code_digest
            != state.final_concept_gate_approved_code_digest
        )

        action = _candidate_concept_audit_transition(host, attempt, state)
        if action is _CandidateLoopAction.RETURN:
            return True
        if action is _CandidateLoopAction.BREAK:
            break
        if action is _CandidateLoopAction.CONTINUE:
            continue

        action = _candidate_execute_transition(host, attempt, state)
        if action is _CandidateLoopAction.RETURN:
            return True
        if action is _CandidateLoopAction.BREAK:
            break
        if action is _CandidateLoopAction.CONTINUE:
            continue

        if state.run_result.succeeded:
            stages = (
                _candidate_success_prepare_transition,
                _candidate_visual_transition,
                _candidate_contract_setup_transition,
                _candidate_contract_repair_transition,
                _candidate_summary_transition,
            )
            for stage in stages:
                action = stage(host, attempt, state)
                if action is not _CandidateLoopAction.PROCEED:
                    break
            if action is _CandidateLoopAction.RETURN:
                return True
            if action is _CandidateLoopAction.BREAK:
                break
            continue

        action = _candidate_failure_transition(host, attempt, state)
        if action is _CandidateLoopAction.RETURN:
            return True
        if action is _CandidateLoopAction.BREAK:
            break
        # Both CONTINUE and an ordinary fall-through retry the current candidate.
        continue
    return False
