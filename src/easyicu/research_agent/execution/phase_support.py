"""Execution-phase helpers extracted from phase.py (byte-preserved).

The execute loop (``run_execute_phase``) and every test-patched seam stay
in :mod:`phase`; this module owns the self-contained pre-loop helpers the
loop consumes through the phase-module facade import.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from contextvars import copy_context
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from ..audits.step_summary_integrity import StepSummaryIntegrityValidator
from .article_audit import RunArticleAuditResult, collect_run_article_audits
from ..gates.concept import (
    deterministic_gate_stamp as _deterministic_gate_stamp,
    finding_detail_without_source_positions as _finding_detail_without_source_positions,
    finding_occurrence_identity as _finding_occurrence_identity,
)
from ..gates.plausibility_receipt import plausibility_audit_receipt_findings
from ..authority.coder_authority import HostCoderAuthority
from ..authority.plausibility import (
    StepPlausibilityAuthority,
)
from ..cohort.repair import extract_cohort_definition_from_prose
from ..cohort.schema import (
    CohortDefinition,
)
from ..contracts.runtime import ValidationFinding

if TYPE_CHECKING:
    from ..orchestration.resume import QuarantinedConceptDraft
    from .step_candidate_recovery import StepCandidateRecovery
from .runners.deterministic_robustness import (
    robustness_replay_spec_has_kind_mismatch,
    robustness_replay_spec_is_emittable,
)
from ..authority.plan_authority import (
    NormalizedPlanCandidate as NormalizedPlanCandidate,
)
from ..authority.plan_scope import (
    _serializable_plan_scientific_scope_signature,
    _step_scientific_signature,
)
from ..gates.contract import (  # execute-layer collaborators use the canonical gate API
    _authoritative_primary_robustness_contract,
    _closed_auxiliary_output_products,
    _is_cohort_definition_sensitivity_result_step,
    _method_head,
    _read_locked_robustness_spec_dicts,
)
from .figure_preparation import (
    _step_has_figure_only_output_contract,
)
from .output_files import (
    bind_primary_output,
)
from ..gates.semantics import (
    blocking_validator_findings as _blocking_validator_findings,
)
from ..planning.method_vocabulary import (
    MISSINGNESS_SOURCE_AVAILABILITY_AUDIT,
)
from ..agents.coder import CoderAgent
from ..providers.mocks import MockLLMClient
from ..agents.core import (
    ReplannerAgent,
)
from . import replan_review
from .owner_declaration import (
    owner_declaration_plan_findings,
)
from ..authority.plan_authority import (
    normalize_replan_candidate,
)
from ..planning.replan_gate import (
    partition_replan_candidate_findings,
    replan_candidate_contract_findings,
    replan_candidate_rejection_finding,
)
from ..robustness.panel import (
    robustness_specs_for_execution,
)
from ..authority.evidence_store import (
    sha256_of_file,
)
from ..authority.registration import (
    step_owned_artifact_evidence_id,
)
from ..authority.typed_binding import (
    _typed_input_product,
)
from ..agents.agentic_coder import AgenticCoderAgent
from ..authority.step_capsule import StepAuthorityCapsuleError
from ..authority.step_runtime import (
    StepAuthorityRuntimeError,
    persist_candidate_code,
)
from .step_authority_resume import (
    StepAuthorityResumeRequest,
    prepare_step_authority_resume,
)
from .publication_figure import (
    _deterministic_publication_figure_code,
)
from ..authority.execution_identity import (
    execution_identity_for_pipeline as _execution_identity,
)
from ..authority.plan_input_closure import (
    plan_manifest_fields,
)
from ..repairs.coordination import (
    authorized_deterministic_concept_repair,
)
from .concept_reaudit import (
    deterministic_concept_reaudit_authority,
)
from ..authority.plausibility import (
    FlagOnlyPlausibilityScope,
)
from .runners.deterministic_descriptive import absolute_risk_context_code
from .runners.deterministic_missingness import (
    missingness_audit_input_scope_supported,
    missingness_measurement_audit_code,
)
from .runners.deterministic_robustness import (
    robustness_sensitivity_preflight_code,
)
from ..cohort.schema import (
    materialize_locked_analysis_cohort,
    write_locked_cohort_definition,
)
from ..intake.materialized_metadata import (
    MaterializedMetadataError,
)
from ..providers.prompt_budget import (
    budgeted_role_client,
)
from .cohort_adoption import (
    commit_staged_cohort_plan,
    record_planned_host_cohort_checkpoint,
    stage_candidate_cohort_plan,
)
from .development_sample import (
    materialize_development_execution_sample,
    record_development_sample_authority,
)
from ..repairs.reasons import (
    RepairPromptAuthority,
    RepairReason,
    repair_prompt_binding_sha256,
    typed_repair_ticket,
)
from ..plan_utils import (
    _clustering_contract_applies,
    _cohort_definition_prose,
    _cohort_definition_contract_findings,
    _cohort_definition_is_empty,
    _plan_expects_analysis_cohort,
    _normalised_expected_output_names,
    _normalised_structured_output_names,
    _output_declares_figure,
    _parent_step_id_for_figure_step,
    _step_expects_figure,
)
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext
from ..contracts.robustness_execution import (
    ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE,
)
from ..repair_registry import (
    InvariantStatus,
    RepairClass,
    RepairObservedState,
    automatic_repair_allowed,
    is_sealed_renderer_repair,
    repair_metadata_for,
)
from ..authority.provider_budget import (
    PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
    ProviderCallBudgetError,
    ProviderCallBudgetReceiptError,
    StepProviderCallBudget,
    complete_with_provider_budget,
    load_provider_call_budget_state,
    provider_call_budget_receipt_path,
)
from .step_attempt_bootstrap import (
    RAW_UNIVERSE_EXECUTION_ROLE,
)
from ..authority.run_input import (
    RUN_INPUT_CAPSULE_EVIDENCE_ID,
    RUN_INPUT_CAPSULE_FILENAME,
    RunInputIdentityError,
    _declares_host_cohort_products as _declares_host_cohort_only_product,
)
from ..authority.runtime_artifacts import (
    current_step_records,
    verified_run_evidence_path,
    write_run_checkpoint,
)
from ..viability import (
    CohortViability,
    step_requires_model_performance,
    step_summary_block_signal,
)

logger = logging.getLogger(__name__)


def _persist_run_article_audit_result(
    *,
    result: RunArticleAuditResult,
    evidence_store: Any,
    flush_partial_manifest: Callable[[Dict[str, Any]], None],
) -> Tuple[ValidationFinding, ...]:
    """Apply one read-only audit result through execute-phase host services."""

    try:
        artifact = result.artifact
        if artifact is not None and evidence_store.get(artifact.evidence_id) is None:
            evidence_store.register_file(
                kind=artifact.kind,
                description=artifact.description,
                source_path=artifact.source_path,
                evidence_id=artifact.evidence_id,
                producer=artifact.producer,
                generation_mode=artifact.generation_mode,
            )
        if result.manifest_items:
            flush_partial_manifest(dict(result.manifest_items))
    except Exception as exc:
        return (
            ValidationFinding(
                validator="article_analysis_contract",
                severity="warning",
                message=(
                    "Run-level article analysis contract audit failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ),
        )
    return ()
def _collect_and_persist_run_article_audits(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    evidence_store: Any,
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
    flush_partial_manifest: Callable[[Dict[str, Any]], None],
) -> Tuple[ValidationFinding, ...]:
    """Join the read-only article owner to execute-phase persistence."""

    result = collect_run_article_audits(
        context=context,
        plan=plan,
        evidence_records=evidence_store.records(),
        per_step_records=per_step_records,
        run_dir=run_dir,
    )
    return result.findings + _persist_run_article_audit_result(
        result=result,
        evidence_store=evidence_store,
        flush_partial_manifest=flush_partial_manifest,
    )
def _repair_prompt_binding_sha256(
    *,
    untrusted_diagnostic: str,
    repair_authority: RepairPromptAuthority,
    current_repair_authority: RepairPromptAuthority | None = None,
) -> str:
    """Bind one provider reservation to diagnostics and typed host authority."""

    return repair_prompt_binding_sha256(
        untrusted_diagnostic=untrusted_diagnostic,
        repair_authority=repair_authority,
        current_repair_authority=current_repair_authority,
    )
def _untrusted_runtime_repair_allowed(*, repair_id: str, source: str) -> bool:
    """Allow raw runtime diagnostics to authorize syntactic transforms only."""
    if source == "case_plugin_repair":
        return False
    if source != "deterministic_runner_repair":
        return True
    metadata = repair_metadata_for(repair_id)
    return (
        repair_id
        in {
            "all_rows_outcome_coordinate_filter_v1",
            "nonfinite_missing_mask_conflation_v1",
            "non_tabular_companion_row_gate_v1",
        }
        or metadata.repair_class is RepairClass.SYNTACTIC
    )
_STANDARD_EXECUTOR_INTERNAL_PENDING_ARTIFACTS = frozenset(
    {".cluster_stability_assignments.pending.csv"}
)
_FIGURE_CONTRACT_SOURCE_DATA_SCHEMA_REPAIR_ID = "figure_contract_source_data_schema_v1"
_COHORT_TRANSLATION_PROVIDER_CATEGORY = "cohort_definition_translation"
_HOST_COHORT_TRANSLATION_BUDGET_STEP_ID = "host_cohort_definition_translation"
_RAW_UNIVERSE_EXECUTION_ROLE = RAW_UNIVERSE_EXECUTION_ROLE
def _submit_in_current_context(executor: Any, callback: Any, *args: Any) -> Any:
    """Submit one step with an independent copy of runner capability context."""

    return executor.submit(copy_context().run, callback, *args)
def _verified_run_input_capsule_digest(
    *,
    run_dir: Path,
    evidence_store: Any,
) -> str:
    """Return the working capsule digest only when sealed evidence agrees."""

    record = evidence_store.get(RUN_INPUT_CAPSULE_EVIDENCE_ID)
    if record is None:
        raise RunInputIdentityError("run input capsule evidence is missing")
    sealed_path = verified_run_evidence_path(run_dir, record)
    working_path = run_dir / RUN_INPUT_CAPSULE_FILENAME
    if sealed_path is None:
        raise RunInputIdentityError("run input capsule evidence failed verification")
    if not working_path.is_file() or working_path.is_symlink():
        raise RunInputIdentityError("run input capsule working copy is missing")
    # Read each copy exactly once and derive the digest, the record check, and
    # the sealed byte-equality check from those same in-memory buffers. The
    # earlier digest-then-reread form returned a digest computed on read #1
    # while comparing read #2 against the sealed copy, so a swap between the two
    # reads could return a digest that never matched the compared bytes; an
    # OSError from the reread also escaped the RunInputIdentityError boundary.
    try:
        working_bytes = working_path.read_bytes()
        sealed_bytes = sealed_path.read_bytes()
    except OSError as exc:
        raise RunInputIdentityError(
            "run input capsule copies could not be read"
        ) from exc
    digest = hashlib.sha256(working_bytes).hexdigest()
    if digest != str(record.sha256):
        raise RunInputIdentityError("run input capsule working digest changed")
    if working_bytes != sealed_bytes:
        raise RunInputIdentityError(
            "run input capsule working copy differs from sealed evidence"
        )
    return digest
def _cohort_translation_budget_owner_step_id(plan: AnalysisPlan) -> str:
    """Return one stable budget owner without making a cohort decision.

    A single cohort-only product step is the natural owner because successful
    host materialisation completes exactly that planned step.  Ambiguous or
    mixed-product plans use a host pseudo-step instead of charging an arbitrary
    analysis step.  This helper only assigns provider-call accounting; the
    Planner's prose remains the sole source of inclusion/exclusion criteria.
    """

    cohort_only_step_ids = [
        str(step.step_id)
        for step in plan.steps
        if _declares_host_cohort_only_product(step)
    ]
    if len(cohort_only_step_ids) == 1:
        return cohort_only_step_ids[0]
    return _HOST_COHORT_TRANSLATION_BUDGET_STEP_ID
def _extract_cohort_definition_with_provider_budget(
    *,
    run_dir: Path,
    budget_owner_step_id: str,
    configured_limit: int,
    cohort_prose: str,
    universe_columns: Sequence[str],
    llm: Any,
    name: str,
    reserved_final_category: Optional[str] = None,
) -> Tuple[Optional[CohortDefinition], Dict[str, Any]]:
    """Run cohort-prose translation under a crash-safe provider receipt.

    This call happens before ``_execute_one_step`` creates its ordinary
    per-step budget.  Reusing the same receipt namespace makes a cohort-only
    planned step inherit this paid call if translation fails and the Coder must
    later execute it.  Transport retries are charged by the active provider
    scope just like coder/auditor retries.
    """

    receipt_path = provider_call_budget_receipt_path(
        run_dir,
        step_id=budget_owner_step_id,
    )
    effective_limit = max(0, int(configured_limit))
    consumed_categories: Tuple[str, ...] = ()
    logical_repair_entries: tuple[Dict[str, object], ...] = ()
    initial_generation_entries: tuple[Dict[str, object], ...] = ()
    required_reservation_token: Optional[str] = None
    reservation_bound_provider_history_len: Optional[int] = None
    completed_reservation_token: Optional[str] = None
    reservation_released = False
    reserved_category_extensions: tuple[Dict[str, object], ...] = ()
    if receipt_path.exists():
        receipt_state = load_provider_call_budget_state(
            receipt_path,
            step_id=budget_owner_step_id,
            expected_reserved_final_category=reserved_final_category,
        )
        effective_limit = min(effective_limit, receipt_state.limit)
        consumed_categories = receipt_state.categories
        logical_repair_entries = receipt_state.logical_repairs
        initial_generation_entries = receipt_state.initial_generations
        required_reservation_token = receipt_state.required_reservation_token
        reservation_bound_provider_history_len = (
            receipt_state.reservation_bound_provider_history_len
        )
        completed_reservation_token = receipt_state.completed_reservation_token
        reservation_released = receipt_state.reservation_released
        reserved_category_extensions = receipt_state.reserved_category_extensions
    budget = StepProviderCallBudget(
        effective_limit,
        step_id=budget_owner_step_id,
        consumed_categories=consumed_categories,
        logical_repair_entries=logical_repair_entries,
        initial_generation_entries=initial_generation_entries,
        receipt_path=receipt_path,
        reserved_final_category=reserved_final_category,
        required_reservation_token=required_reservation_token,
        reservation_bound_provider_history_len=(reservation_bound_provider_history_len),
        completed_reservation_token=completed_reservation_token,
        reservation_released=reservation_released,
        reserved_category_extensions=reserved_category_extensions,
    )
    definition = complete_with_provider_budget(
        budget=budget,
        category=_COHORT_TRANSLATION_PROVIDER_CATEGORY,
        call=lambda: extract_cohort_definition_from_prose(
            cohort_prose=cohort_prose,
            universe_columns=universe_columns,
            llm=llm,
            name=name,
        ),
    )
    snapshot = budget.snapshot()
    return definition, {
        "budget_owner_step_id": budget_owner_step_id,
        "step_provider_call_budget_scope": _COHORT_TRANSLATION_PROVIDER_CATEGORY,
        "step_provider_call_budget": snapshot["limit"],
        "step_provider_call_attempts": snapshot["used"],
        "step_provider_call_remaining": snapshot["remaining"],
        "step_provider_call_budget_exhausted": snapshot["exhausted"],
        "step_provider_call_categories": snapshot["categories"],
        "step_provider_call_receipt_version": (
            PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION
        ),
        "step_provider_call_receipt": str(receipt_path.relative_to(run_dir)),
    }
def _merge_monotonic_concept_constraints(
    existing: Sequence[ValidationFinding],
    candidates: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Merge binding concept errors without losing earlier repair constraints.

    A later repair may introduce a different error after an earlier error has
    already been removed from the current script.  Quarantine checkpoints must
    retain both constraints so a resumed repair cannot regress the earlier fix.
    """

    merged: List[ValidationFinding] = []
    index_by_occurrence: Dict[str, int] = {}

    def _latest_with_evidence(
        prior: ValidationFinding,
        latest: ValidationFinding,
    ) -> ValidationFinding:
        evidence_ids = list(
            dict.fromkeys(
                [
                    *(str(item) for item in prior.evidence_ids or []),
                    *(str(item) for item in latest.evidence_ids or []),
                ]
            )
        )
        return latest.model_copy(update={"evidence_ids": evidence_ids})

    for finding in existing:
        if finding.severity != "error":
            # Preserve pre-existing nonblocking audit history exactly as the
            # prior implementation did. New warnings are not monotonic repair
            # constraints and therefore are not added below.
            merged.append(finding)
            continue
        key = _finding_occurrence_identity(finding)
        prior_index = index_by_occurrence.get(key)
        if prior_index is None:
            index_by_occurrence[key] = len(merged)
            merged.append(finding)
        else:
            # Keep the latest wording and source coordinates for the same
            # durable occurrence while preserving every distinct locator.
            merged[prior_index] = _latest_with_evidence(merged[prior_index], finding)
    for finding in candidates:
        if finding.severity != "error":
            continue
        key = _finding_occurrence_identity(finding)
        prior_index = index_by_occurrence.get(key)
        if prior_index is None:
            index_by_occurrence[key] = len(merged)
            merged.append(finding)
        else:
            merged[prior_index] = _latest_with_evidence(merged[prior_index], finding)
    return merged
def _persisted_monotonic_concept_constraints(
    record: Mapping[str, Any] | None,
) -> List[ValidationFinding]:
    """Load binding constraints from the latest unfinished step record."""

    if not isinstance(record, Mapping) or str(record.get("status") or "") == "ok":
        return []
    raw_constraints = record.get("monotonic_concept_constraints")
    if not isinstance(raw_constraints, list):
        return []
    parsed: List[ValidationFinding] = []
    for payload in raw_constraints:
        if not isinstance(payload, Mapping):
            continue
        try:
            finding = ValidationFinding.model_validate(payload)
        except (TypeError, ValueError):
            continue
        parsed = _merge_monotonic_concept_constraints(parsed, [finding])
    return parsed
def _step_remember_concept_constraints(
    monotonic_concept_constraints: List[ValidationFinding],
    step_record: Dict[str, Any],
    candidates: Sequence[ValidationFinding],
) -> None:
    """Keep repaired scientific defects binding across later repairs."""

    monotonic_concept_constraints[:] = _merge_monotonic_concept_constraints(
        monotonic_concept_constraints,
        candidates,
    )
    if monotonic_concept_constraints:
        step_record["monotonic_concept_constraints"] = [
            finding.model_dump(mode="json")
            for finding in monotonic_concept_constraints
        ]


def _step_quarantine_error_payloads(
    monotonic_concept_constraints: List[ValidationFinding],
    step_record: Dict[str, Any],
    candidates: Sequence[ValidationFinding],
) -> List[Dict[str, Any]]:
    """Serialize the complete cross-repair constraint set for resume."""

    _step_remember_concept_constraints(
        monotonic_concept_constraints,
        step_record,
        candidates,
    )
    return [
        finding.model_dump(mode="json")
        for finding in monotonic_concept_constraints
    ]


def _step_monotonic_concept_constraint_payload(
    monotonic_concept_constraints: Sequence[ValidationFinding],
) -> List[Dict[str, Any]]:
    return [
        {
            "validator": finding.validator,
            "message": finding.message,
            "detail": _finding_detail_without_source_positions(
                dict(finding.detail or {})
            ),
        }
        for finding in monotonic_concept_constraints
    ]


def _step_monotonic_concept_constraint_ticket(
    monotonic_concept_constraints: Sequence[ValidationFinding],
) -> List[Dict[str, Any]]:
    """Return durable repair constraints without stale code positions."""

    return typed_repair_ticket(
        [
            finding.model_copy(
                update={
                    "detail": _finding_detail_without_source_positions(
                        dict(finding.detail or {})
                    )
                }
            )
            for finding in monotonic_concept_constraints
        ]
    )


def _step_monotonic_concept_constraint_log(
    monotonic_concept_constraints: Sequence[ValidationFinding],
) -> str:
    if not monotonic_concept_constraints:
        return ""
    payload = _step_monotonic_concept_constraint_payload(
        monotonic_concept_constraints
    )
    return (
        "\n\nPREVIOUSLY REPAIRED CONCEPT FINDINGS (binding regression "
        "constraints; do not reintroduce them):\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    )


def _step_use_quarantined_draft(
    candidate_recovery: "StepCandidateRecovery",
    draft: "QuarantinedConceptDraft",
) -> str:
    return candidate_recovery.use_quarantined_draft(draft)


def _step_use_resumed_code(
    candidate_recovery: "StepCandidateRecovery",
    resumed_code: Tuple[str, Dict[str, Any]],
    *,
    error: Optional[BaseException] = None,
) -> str:
    return candidate_recovery.use_resumed_code(resumed_code, error=error)


def _step_repair_with_capsule(
    candidate_recovery: "StepCandidateRecovery",
    *,
    failure_status: str,
    context: ResearchContext,
    step: AnalysisStep,
    code: str,
    run_log: str,
    repair_authority: RepairPromptAuthority,
    current_repair_authority: Optional[RepairPromptAuthority] = None,
    attempt: int,
    provider_budget: StepProviderCallBudget,
    provider_category: str,
    logical_repair_attempt_id: int,
) -> str:
    return candidate_recovery.repair_with_capsule(
        failure_status=failure_status,
        context=context,
        step=step,
        code=code,
        run_log=run_log,
        repair_authority=repair_authority,
        current_repair_authority=current_repair_authority,
        attempt=attempt,
        provider_budget=provider_budget,
        provider_category=provider_category,
        logical_repair_attempt_id=logical_repair_attempt_id,
    )


def _step_reserve_compatibility_repair(
    candidate_recovery: "StepCandidateRecovery",
    before_code: str,
    repair_ticket: str,
    repair_authority: RepairPromptAuthority,
) -> Optional[int]:
    return candidate_recovery.reserve_compatibility_repair(
        before_code,
        repair_ticket,
        repair_authority,
    )


def _step_resume_deterministic_repair_code(
    candidate_recovery: "StepCandidateRecovery",
) -> Optional[str]:
    return candidate_recovery.resume_deterministic_repair_code()


def _step_resume_critic_repair_code(
    candidate_recovery: "StepCandidateRecovery",
) -> Optional[str]:
    return candidate_recovery.resume_critic_repair_code()


def _step_deterministic_fallback_code(
    reason: str,
    *,
    worker_progress: Any,
    pipeline: Any,
    plan_result: Any,
    step_record: Dict[str, Any],
    emit_progress: Any,
    step: AnalysisStep,
    run_id: str,
    step_current: int,
    total_steps: int,
    coder_context: ResearchContext,
    coder_authority: Any,
) -> Optional[str]:
    if (
        worker_progress.deterministic_fallback_used
        or not pipeline._enable_deterministic_code_fallback
    ):
        return None
    worker_progress.deterministic_fallback_used = True
    plan_result.used_mock_llm = True
    step_record["deterministic_code_fallback"] = reason
    emit_progress(
        "coder",
        f"Using deterministic fallback script for {step.step_id}.",
        run_id=run_id,
        step_id=step.step_id,
        current_step=step_current,
        total_steps=total_steps,
        fallback_reason=reason,
    )
    fallback_coder = CoderAgent(MockLLMClient(context=coder_context))
    return fallback_coder.run(
        context=coder_context,
        step=step,
        host_authority=coder_authority,
    )


def _step_authorize_deterministic_concept_reaudit(
    *,
    token: str,
    code_sha256: str,
    provider_budget: StepProviderCallBudget,
    worker_progress: Any,
    step_record: Dict[str, Any],
    prior_step_record: Any,
    prior_attempt_records: Any,
) -> bool:
    budget_snapshot = provider_budget.snapshot()
    repair_names = deterministic_concept_reaudit_authority(
        code_sha256=code_sha256,
        current_repair_count=worker_progress.deterministic_concept_repairs,
        current_repair_names=worker_progress.applied_concept_repair_names,
        current_repair_code_sha256=step_record.get(
            "deterministic_concept_repair_code_sha256"
        ),
        prior_step_record=prior_step_record,
        prior_step_records=prior_attempt_records,
        provider_used=budget_snapshot["used"],
        provider_limit=budget_snapshot["limit"],
    )
    if not repair_names:
        return False
    granted = provider_budget.authorize_deterministic_reserved_category_extension(
        "concept_audit",
        token=token,
    )
    if granted:
        step_record["deterministic_concept_reaudit_extension"] = {
            "code_sha256": code_sha256,
            "repair_names": list(repair_names),
            "diagnostic_code": (
                "deterministic_repair_final_audit_extension_v1"
            ),
        }
    return granted


def _step_authorized_deterministic_concept_repair(
    *,
    authorize: Any,
    step: AnalysisStep,
    context: ResearchContext,
    step_repair_budget: Any,
    script_text: str,
    error_messages: Sequence[str],
    repair_reasons: Sequence[RepairReason] = (),
    repair_findings: Sequence[ValidationFinding] = (),
    source: str,
) -> Tuple[str, List[str]]:
    return authorized_deterministic_concept_repair(
        script_text,
        error_messages,
        repair_reasons=repair_reasons,
        repair_findings=repair_findings,
        authorize=authorize,
        step=step,
        source=source,
        context=context,
        on_semantic_escalation=step_repair_budget.record_semantic_escalation,
    )


def _step_register_current_step_numeric_claims(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    step_summary_record_id: Optional[str],
    standard_executor_terminal_block: bool,
    evidence: Any,
    findings: List[ValidationFinding],
    pipeline: Any,
) -> None:
    """Stage numeric authority before exposing current result aliases."""

    if (
        not step_summary
        or step_summary_record_id is None
        or standard_executor_terminal_block
    ):
        return
    cap = pipeline._max_numeric_claims_per_step
    evidence.register_step_summary_numerics(
        step_id=step.step_id,
        evidence_id=step_summary_record_id,
        summary=step_summary,
        max_leaves=cap if cap > 0 else None,
    )
    _, derived_errors = evidence.register_step_derived_claims(
        step_id=step.step_id,
        evidence_id=step_summary_record_id,
        summary=step_summary,
    )
    for err in derived_errors:
        findings.append(
            ValidationFinding(
                validator="derived_claim",
                severity="warning",
                message=(
                    f"derived_claims entry {err['name']!r} for step "
                    f"{step.step_id} was rejected: {err['message']}"
                ),
                detail={
                    "step_id": step.step_id,
                    "claim_name": err["name"],
                    "reason": err["message"],
                },
            )
        )


def _step_publication_figure_preflight_supported(
    *,
    step: AnalysisStep,
    run_dir: Path,
    services: Any,
) -> bool:
    if not _step_has_figure_only_output_contract(step):
        return False
    return services.deterministic_figure_family_supported_for_upstream(
        run_dir, step.step_id
    )


def _step_absolute_risk_context_preflight_supported(
    *,
    step: AnalysisStep,
) -> bool:
    if _step_expects_figure(step):
        return False
    return _absolute_risk_context_runner_owns_step(
        str(step.method or ""),
        str(step.step_id or ""),
        step.expected_outputs or [],
    )


def _step_deterministic_absolute_risk_context_code(
    reason: str,
    *,
    preflight_supported: Any,
    worker_progress: Any,
    pipeline: Any,
    step: AnalysisStep,
    step_record: Dict[str, Any],
    plausibility_authority: Any,
    emit_progress: Any,
    run_id: str,
    step_current: int,
    total_steps: int,
    preflight: bool = False,
) -> Optional[str]:
    if (
        worker_progress.deterministic_fallback_used
        or not pipeline._enable_deterministic_runner_repair
        or (preflight and not preflight_supported())
    ):
        return None
    if not preflight_supported():
        return None
    worker_progress.deterministic_fallback_used = True
    step_record["deterministic_code_fallback"] = reason
    step_record["deterministic_standard_analysis"] = "absolute_risk_context"
    emit_progress(
        "coder",
        f"Using deterministic absolute-risk context runner for {step.step_id}.",
        run_id=run_id,
        step_id=step.step_id,
        current_step=step_current,
        total_steps=total_steps,
        fallback_reason=reason,
    )
    return absolute_risk_context_code(
        plausibility_scope=plausibility_authority.scope
    )


def _step_robustness_sensitivity_preflight_supported(
    *,
    step: AnalysisStep,
) -> bool:
    if _step_expects_figure(step):
        return False
    return _robustness_sensitivity_runner_owns_step(
        str(step.method or ""),
        str(step.step_id or ""),
        step.expected_outputs or [],
        step=step,
    )


def _step_deterministic_robustness_sensitivity_code(
    reason: str,
    *,
    preflight_supported: Any,
    worker_progress: Any,
    pipeline: Any,
    step: AnalysisStep,
    step_record: Dict[str, Any],
    emit_progress: Any,
    run_id: str,
    step_current: int,
    total_steps: int,
    plausibility_scope: Optional[FlagOnlyPlausibilityScope],
    preflight: bool = False,
) -> Optional[str]:
    if (
        worker_progress.deterministic_fallback_used
        or not pipeline._enable_deterministic_runner_repair
        or (preflight and not preflight_supported())
    ):
        return None
    if not preflight_supported():
        return None
    worker_progress.deterministic_fallback_used = True
    step_record["deterministic_code_fallback"] = reason
    step_record["deterministic_standard_analysis"] = "robustness_sensitivity"
    emit_progress(
        "coder",
        f"Using deterministic robustness runner for {step.step_id}.",
        run_id=run_id,
        step_id=step.step_id,
        current_step=step_current,
        total_steps=total_steps,
        fallback_reason=reason,
    )
    return robustness_sensitivity_preflight_code(
        step,
        plausibility_scope=plausibility_scope,
    )


def _step_missingness_audit_preflight_supported(
    *,
    step: AnalysisStep,
) -> bool:
    if _step_expects_figure(step):
        return False
    if not missingness_audit_input_scope_supported(step):
        return False
    return _simple_missingness_audit_runner_owns_step(
        str(step.method or ""),
        str(step.step_id or ""),
        str(step.intent or ""),
        step.expected_outputs or [],
    )


def _step_deterministic_missingness_audit_code(
    reason: str,
    *,
    preflight_supported: Any,
    worker_progress: Any,
    pipeline: Any,
    step: AnalysisStep,
    step_record: Dict[str, Any],
    emit_progress: Any,
    run_id: str,
    step_current: int,
    total_steps: int,
    preflight: bool = False,
) -> Optional[str]:
    if (
        worker_progress.deterministic_fallback_used
        or not pipeline._enable_deterministic_runner_repair
        or (preflight and not preflight_supported())
    ):
        return None
    if not preflight_supported():
        return None
    worker_progress.deterministic_fallback_used = True
    step_record["deterministic_code_fallback"] = reason
    step_record["deterministic_standard_analysis"] = (
        "missingness_measurement_audit"
    )
    emit_progress(
        "coder",
        f"Using deterministic missingness/measurement audit runner for {step.step_id}.",
        run_id=run_id,
        step_id=step.step_id,
        current_step=step_current,
        total_steps=total_steps,
        fallback_reason=reason,
    )
    return missingness_measurement_audit_code(step)


def _step_flush_partial_manifest(
    extra: Optional[Dict[str, Any]] = None,
    *,
    per_step_records: List[Dict[str, Any]],
    step_attempt_history: List[Dict[str, Any]],
    run_id: str,
    run_dir: Path,
    evidence: Any,
    plan: AnalysisPlan,
    plan_path: Path,
    context: ResearchContext,
    plan_result: Any,
    findings: List[ValidationFinding],
    llm_signature: str,
    prompt_version: str,
    prompt_files: List[str],
    pipeline: Any,
    notes: Optional[str],
    runtime_state: Any,
    repair_ledger: Any,
    _replan_state: Dict[str, Any],
) -> None:
    for record in per_step_records:
        snapshot = dict(record)
        if snapshot not in step_attempt_history:
            step_attempt_history.append(snapshot)
    try:
        plan_fields: Dict[str, Any] = dict(
            plan_manifest_fields(run_dir, evidence, plan, plan_path)
        )
    except ValueError as authority_error:
        plan_fields = {"current_plan_authority_error": str(authority_error)}
    payload: Dict[str, Any] = {
        "schema_version": "easyicu.research_manifest_partial/1",
        "run_id": run_id,
        "research_question": context.research_question,
        "started_at": plan_result.started_at.isoformat(),
        "context_path": str(plan_result.context_path.relative_to(run_dir)),
        **plan_fields,
        "evidence": [r.model_dump(mode="json") for r in evidence.records()],
        "findings": [f.model_dump(mode="json") for f in findings],
        "per_step_records": per_step_records,
        "step_attempt_history": step_attempt_history,
        "llm_signature": llm_signature,
        "used_mock_llm": plan_result.used_mock_llm,
        "prompt_pack_version": prompt_version,
        "prompt_pack_files": prompt_files,
        "execution_identity": _execution_identity(pipeline).model_dump(mode="json"),
        "notes": notes,
        "runtime_state": runtime_state.model_dump(mode="json"),
        "repair_ledger_path": str(repair_ledger.path.relative_to(run_dir)),
        "repairs_applied": [record.__dict__ for record in repair_ledger.records],
        "cohort_translation_provider_budget": _replan_state.get(
            "cohort_translation_provider_budget"
        ),
    }
    if extra:
        payload.update(extra)
    write_run_checkpoint(run_dir / "manifest_partial.json", payload)


def _step_register_plan_revision(
    revised_plan: AnalysisPlan,
    *,
    reason: str,
    producer: str,
    run_dir: Path,
    context: ResearchContext,
    evidence: Any,
    llm_signature: str,
    prompt_version: str,
) -> Path:
    from ..authority.declared_levels import bind_step_declared_levels
    from ..authority.table_one_binding import (
        bind_table_one_execution_spec,
        write_table_one_private_checkpoint,
    )

    for revised_step in revised_plan.steps:
        bind_table_one_execution_spec(revised_step, context)
        bind_step_declared_levels(revised_step, context)
    write_table_one_private_checkpoint(run_dir=run_dir, plan=revised_plan)
    revision_path = run_dir / f"analysis_plan_revision_{revised_plan.revision}.json"
    revision_path.write_text(
        revised_plan.model_dump_json(indent=2),
        encoding="utf-8",
    )
    base_id = f"analysis_plan_revision_{revised_plan.revision}"
    try:
        evidence.register_file(
            kind="log",
            description=f"Revised analysis plan (reason={reason}).",
            source_path=revision_path,
            evidence_id=base_id,
            producer=producer,
            generation_mode="llm",
            prompt_pack_version=prompt_version,
            metadata={"reason": reason, "llm_signature": llm_signature},
        )
    except ValueError:
        import hashlib

        digest = hashlib.sha256(revision_path.read_bytes()).hexdigest()[:8]
        evidence.register_file(
            kind="log",
            description=(
                f"Revised analysis plan (reason={reason}; resume re-revision)."
            ),
            source_path=revision_path,
            evidence_id=f"{base_id}_{digest}",
            producer=producer,
            generation_mode="llm",
            prompt_pack_version=prompt_version,
            metadata={
                "reason": reason,
                "llm_signature": llm_signature,
                "resume_reregistration": True,
            },
        )
    return revision_path


def _step_no_analysis_step_has_run(
    per_step_records: Sequence[Dict[str, Any]],
) -> bool:
    return not any(
        (rec.get("step_id") or "") != "00_probe" for rec in per_step_records
    )


def _step_universe_columns(
    run_input_authority_state: Any,
    universe_path: Path,
) -> list:
    typed_columns = run_input_authority_state.cohort_authority.universe_columns
    if typed_columns is not None:
        return list(typed_columns)
    try:
        import pyarrow.parquet as pq  # type: ignore

        return list(pq.read_schema(universe_path).names)
    except Exception:
        try:
            import pandas as pd  # type: ignore

            return list(pd.read_parquet(universe_path).columns)
        except Exception:
            return []


def _step_enforce_cohort_contract_on_executing_plan(
    candidate_plan: AnalysisPlan,
    *,
    reason: str,
    _replan_state: Dict[str, Any],
    findings: List[ValidationFinding],
    run_input_authority_state: Any,
) -> None:
    if _replan_state["cohort_contract_emitted"]:
        return
    if run_input_authority_state.analysis_path.exists():
        return
    if not (
        _plan_expects_analysis_cohort(candidate_plan)
        and _cohort_definition_is_empty(candidate_plan)
    ):
        return
    for finding in _cohort_definition_contract_findings(candidate_plan):
        findings.append(
            finding.model_copy(
                update={
                    "detail": {
                        **(finding.detail or {}),
                        "stage": "execute",
                        "reason": reason,
                    }
                }
            )
        )
    _replan_state["cohort_contract_emitted"] = True


def _step_resolve_cohort_definition(
    candidate_plan: AnalysisPlan,
    *,
    reason: str,
    try_materialize: Any,
    enforce_contract: Any,
) -> bool:
    if not (
        _plan_expects_analysis_cohort(candidate_plan)
        and _cohort_definition_is_empty(candidate_plan)
    ):
        return False
    before = candidate_plan.model_dump(mode="json").get("cohort")
    if not try_materialize(candidate_plan, reason=reason):
        enforce_contract(candidate_plan, reason=reason)
    return candidate_plan.model_dump(mode="json").get("cohort") != before


def _step_record_repair(
    *,
    repair_id: str,
    step_id: str,
    trigger: Dict[str, Any],
    transformation: str,
    repair_ledger: Any,
    repair_ledger_lock: Any,
    llm_signature: str,
    findings: List[ValidationFinding],
    before_code: Optional[str] = None,
    after_code: Optional[str] = None,
    selection_rule: Optional[str] = None,
    before_state: Optional[RepairObservedState] = None,
    after_state: Optional[RepairObservedState] = None,
    outcome: str = "applied",
) -> None:
    try:
        with repair_ledger_lock:
            provenance = repair_ledger.append_application(
                repair_id=repair_id,
                step_id=step_id,
                trigger=trigger,
                transformation=transformation,
                outcome=outcome,
                model_id=llm_signature,
                before_text=before_code,
                after_text=after_code,
                selection_rule=selection_rule,
                before_state=before_state,
                after_state=after_state,
            )
        if provenance.invariant_status == InvariantStatus.VERIFIED_FAIL.value:
            findings.append(
                ValidationFinding(
                    validator="repair_invariant",
                    severity="warning",
                    message=(
                        f"Repair {repair_id} violated declared invariant(s) "
                        f"{list(provenance.invariant_failures)} on step {step_id}."
                    ),
                    detail={
                        "repair_id": repair_id,
                        "step_id": step_id,
                        "repair_class": provenance.repair_class,
                        "invariant_failures": list(provenance.invariant_failures),
                    },
                )
            )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="repair_ledger",
                severity="warning",
                message=(
                    f"Could not record repair provenance for {repair_id}: {exc}"
                ),
                detail={"repair_id": repair_id, "step_id": step_id},
            )
        )


def _step_automatic_repair_authorized(
    repair_id: str,
    *,
    step: AnalysisStep,
    source: str,
    findings: List[ValidationFinding],
    record_repair: Any,
    before_code: Optional[str] = None,
    after_code: Optional[str] = None,
    sealed_renderer_wrapper: bool = False,
) -> bool:
    step_id = str(step.step_id)
    untrusted_runtime_policy_denied = not _untrusted_runtime_repair_allowed(
        repair_id=repair_id,
        source=source,
    )
    if not untrusted_runtime_policy_denied and automatic_repair_allowed(
        repair_id,
        step=step,
        sealed_renderer_wrapper=sealed_renderer_wrapper,
    ):
        return True
    if untrusted_runtime_policy_denied:
        policy_reason = (
            "case_plugin_requires_typed_repair_contract"
            if source == "case_plugin_repair"
            else "untrusted_runtime_diagnostic_allows_syntactic_only"
        )
    else:
        sealed_context_denied = is_sealed_renderer_repair(repair_id)
        policy_reason = (
            "sealed_renderer_requires_preexecution_wrapper"
            if sealed_context_denied
            else "method_substitution_default_deny"
        )
    record_repair(
        repair_id=repair_id,
        step_id=step_id,
        trigger={
            "source": source,
            "automatic_repair_policy": policy_reason,
        },
        transformation=(
            "Candidate repair was not applied because its execution context "
            "did not satisfy the central automatic-repair policy."
        ),
        before_code=before_code,
        after_code=after_code,
        outcome="blocked_by_automatic_repair_policy",
    )
    findings.append(
        ValidationFinding(
            validator="automatic_repair_policy",
            severity="info",
            message=(
                f"Blocked automatic repair {repair_id} for step {step_id}; "
                f"policy={policy_reason}."
            ),
            detail={
                "repair_id": repair_id,
                "step_id": step_id,
                "source": source,
                "policy": policy_reason,
                "outcome": "blocked_by_automatic_repair_policy",
            },
        )
    )
    return False


def _step_authorize_automatic_repair(
    repair: Optional[Tuple[str, str]],
    *,
    step: AnalysisStep,
    source: str,
    before_code: str,
    automatic_repair_authorized: Any,
    sealed_renderer_wrapper: bool = False,
) -> Optional[Tuple[str, str]]:
    if repair is None:
        return None
    repair_id, candidate_code = repair
    if not automatic_repair_authorized(
        repair_id,
        step=step,
        source=source,
        before_code=before_code,
        after_code=candidate_code,
        sealed_renderer_wrapper=sealed_renderer_wrapper,
    ):
        return None
    return repair


def _step_propagate_findings_to_evidence(
    evidence_ids: Sequence[str],
    findings_for_step: Sequence[ValidationFinding],
    *,
    evidence: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    scoped = scope_findings_to_records(evidence_ids, findings_for_step)
    for evidence_id in evidence_ids:
        severity, messages = scoped[str(evidence_id)]
        evidence.update_record(
            evidence_id,
            finding_severity=severity,
            finding_messages=messages,
            metadata=metadata,
        )


def _step_validator_messages(
    *finding_groups: Sequence[ValidationFinding],
) -> List[str]:
    return _actionable_validator_messages(*finding_groups)


def _step_failed_dependency_record(
    step: AnalysisStep,
    *,
    per_step_records: List[Dict[str, Any]],
    shared_lock: Any,
) -> Optional[Dict[str, Any]]:
    parent_step_id = _parent_step_id_for_figure_step(step)
    if parent_step_id is None:
        return None
    with shared_lock:
        records = list(per_step_records)
    latest = {
        str(record.get("step_id") or ""): record
        for record in current_step_records(records)
    }
    record = latest.get(parent_step_id)
    if record is not None:
        if str(record.get("status") or "").lower() == "ok":
            return None
        return dict(record)
    return None


def _step_try_materialize_cohort_from_prose(
    candidate_plan: AnalysisPlan,
    *,
    reason: str,
    _replan_state: Dict[str, Any],
    context: ResearchContext,
    emit_progress: Any,
    evidence: Any,
    findings: List[ValidationFinding],
    llm_signature: str,
    per_step_records: List[Dict[str, Any]],
    pipeline: Any,
    preexecuted_step_ids: Any,
    prompt_version: str,
    role_resolver: Any,
    run_dir: Path,
    run_id: str,
    run_input_authority_state: Any,
    universe_path: Path,
    no_analysis_step_has_run: Any,
    universe_columns: Any,
    cohort_path: Path,
    runner: Any,
    cohort_concept_ids: Sequence[str] = (),
) -> Tuple[bool, Path, Any]:
    """Extract the agent's prose 纳排 into typed predicates, materialise the
    filtered analysis cohort, and re-point the runner at it.
    """

    if _replan_state["cohort_materialized"]:
        return True, cohort_path, runner
    if run_input_authority_state.analysis_path.exists():
        return True, cohort_path, runner
    if not no_analysis_step_has_run():
        return False, cohort_path, runner
    columns = universe_columns()
    if not columns:
        return False, cohort_path, runner
    budget_owner_step_id = _replan_state.get(
        "cohort_translation_budget_owner_step_id"
    )
    if not budget_owner_step_id:
        budget_owner_step_id = _cohort_translation_budget_owner_step_id(
            candidate_plan
        )
        _replan_state["cohort_translation_budget_owner_step_id"] = (
            budget_owner_step_id
        )
    try:
        definition, budget_snapshot = (
            _extract_cohort_definition_with_provider_budget(
                run_dir=run_dir,
                budget_owner_step_id=str(budget_owner_step_id),
                configured_limit=pipeline._max_step_provider_calls,
                cohort_prose=_cohort_definition_prose(candidate_plan),
                universe_columns=columns,
                llm=budgeted_role_client(
                    role_resolver,
                    "planner",
                    "cohort_extraction",
                    limit_tokens=pipeline._max_prompt_tokens_per_call,
                ),
                name=getattr(
                    getattr(candidate_plan, "cohort", None),
                    "name",
                    "primary",
                )
                or "primary",
                reserved_final_category=(
                    "concept_audit" if pipeline._enable_llm_concept_audit else None
                ),
            )
        )
    except ProviderCallBudgetError as exc:
        error_detail = f"{type(exc).__name__}: {exc}"
        _replan_state["cohort_translation_provider_budget"] = {
            "budget_owner_step_id": str(budget_owner_step_id),
            "error": error_detail,
        }
        if not _replan_state.get(
            "cohort_translation_provider_budget_error_emitted"
        ):
            findings.append(
                ValidationFinding(
                    validator="cohort_translation_provider_budget",
                    severity="error",
                    message=(
                        "Cohort-definition translation could not obtain a "
                        "trusted provider-call reservation; the host did not "
                        "infer or apply cohort criteria."
                    ),
                    detail={
                        "stage": "execute_repair",
                        "reason": reason,
                        "budget_owner_step_id": str(budget_owner_step_id),
                        "error": error_detail,
                    },
                )
            )
            _replan_state["cohort_translation_provider_budget_error_emitted"] = True
        return False, cohort_path, runner
    _replan_state["cohort_translation_provider_budget"] = budget_snapshot
    if definition is None:
        return False, cohort_path, runner
    materialization_plan = stage_candidate_cohort_plan(
        candidate_plan,
        definition,
    )
    try:
        write_locked_cohort_definition(
            run_dir=run_dir,
            plan=materialization_plan,
            evidence=evidence,
            prompt_pack_version=prompt_version,
            llm_signature=llm_signature,
            allow_empty_promotion=True,
            cohort_concept_ids=cohort_concept_ids,
        )
        result = materialize_locked_analysis_cohort(
            run_dir=run_dir,
            plan=materialization_plan,
            universe_path=universe_path,
            context=context,
            cohort_concept_ids=cohort_concept_ids,
        )
    except MaterializedMetadataError:
        raise
    except Exception as exc:  # legacy translation errors remain recoverable
        findings.append(
            ValidationFinding(
                validator="cohort_materializer",
                severity="warning",
                message=(
                    "Extracted a cohort definition from step prose but could "
                    f"not materialise it: {type(exc).__name__}: {exc}"
                ),
                detail={"stage": "execute_repair", "reason": reason},
            )
        )
        return False, cohort_path, runner
    committed = commit_staged_cohort_plan(
        candidate_plan,
        materialization_plan,
        materialization_status=result.get("status"),
        authority_state=run_input_authority_state,
        context=context,
    )
    if not committed:
        findings.append(
            ValidationFinding(
                validator="cohort_materializer",
                severity="error",
                message=(
                    "The extracted cohort definition was not applied; the "
                    "executing plan remains unchanged and cannot claim those "
                    "criteria."
                ),
                detail={
                    "stage": "execute_repair",
                    "reason": reason,
                    "materialization_status": result.get("status"),
                },
            )
        )
        return False, cohort_path, runner
    cohort_path = run_input_authority_state.selected_path
    record_planned_host_cohort_checkpoint(
        plan=candidate_plan,
        result=result,
        cohort_path=cohort_path,
        evidence=evidence,
        prompt_pack_version=prompt_version,
        llm_signature=llm_signature,
        run_dir=run_dir,
        reason=reason,
        gate_stamp=_deterministic_gate_stamp(),
        per_step_records=per_step_records,
        preexecuted_step_ids=preexecuted_step_ids,
        findings=findings,
        budget_snapshot=(
            budget_snapshot
            if budget_owner_step_id
            == _cohort_translation_budget_owner_step_id(candidate_plan)
            else None
        ),
    )
    if pipeline._development_sample_size is not None:
        run_input_authority_state.apply_development_sample(
            materialize_development_execution_sample(
                run_dir=run_dir,
                target_rows=pipeline._development_sample_size,
                seed=pipeline._development_sample_seed,
                declared_id_columns=tuple(
                    getattr(context.cohort, "id_columns", ()) or ()
                ),
                trajectory_binding=run_input_authority_state.trajectory_binding,
            )
        )
        record_development_sample_authority(
            binding=run_input_authority_state.development_sample,
            evidence=evidence,
            findings=findings,
            emit_progress=emit_progress,
            run_id=run_id,
        )
    cohort_path = run_input_authority_state.selected_path
    runner = pipeline._build_runner(
        run_dir=run_dir,
        cohort_path=cohort_path,
        target_outcome=context.target_outcome,
        universe_path=universe_path,
        **run_input_authority_state.runner_bindings(),
    )
    findings.append(
        ValidationFinding(
            validator="cohort_materializer",
            severity="info",
            message=(
                "Translated the cohort-definition step's prose into typed "
                "predicates and applied them: analysis cohort "
                f"n={result['n_cohort']} of universe n={result['n_universe']}. "
                "Downstream steps now read the filtered cohort "
                "(COHORT_PARQUET); the full universe is exposed only to "
                "explicitly authorized typed steps."
            ),
            detail={
                "stage": "execute_repair",
                "reason": reason,
                "n_universe": result["n_universe"],
                "n_analysis_cohort": result["n_cohort"],
            },
        )
    )
    _replan_state["cohort_materialized"] = True
    return True, cohort_path, runner


def _step_maybe_replan(
    *,
    current_plan: AnalysisPlan,
    reason: str,
    plan_path: Path,
    _replan_state: Dict[str, Any],
    context: ResearchContext,
    findings: List[ValidationFinding],
    pipeline: Any,
    plan_result: Any,
    role_resolver: Any,
    run_dir: Path,
    skill_obj: Any,
    flush_partial_manifest: Any,
    resolve_cohort_definition: Any,
    register_plan_revision: Any,
    probe_summary_payload: Optional[Dict[str, Any]] = None,
    completed_records: Optional[Sequence[Dict[str, Any]]] = None,
    directive: Optional[str] = None,
    force: bool = False,
) -> Tuple[AnalysisPlan, Path]:
    if not pipeline._enable_replanning or skill_obj is not None:
        return current_plan, plan_path
    if _replan_state["disabled"] and not force:
        return current_plan, plan_path
    terminal_repair_skip = _terminal_publication_repair_replan_skip_detail(
        plan=current_plan,
        completed_records=completed_records,
        run_dir=run_dir,
    )
    if terminal_repair_skip is not None and not force:
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="info",
                message=(
                    "Skipped replanner because only terminal rendering-only "
                    "publication-figure repair steps remain, and a completed "
                    "step already produced a primary-result publication bundle."
                ),
                detail={
                    "reason": reason,
                    **terminal_repair_skip,
                },
            )
        )
        return current_plan, plan_path
    replanner = ReplannerAgent(role_resolver("planner"))
    try:
        revised = replanner.run(
            context=plan_result.agent_context,
            current_plan=current_plan,
            probe_summary=probe_summary_payload,
            completed_step_records=completed_records,
            directive=directive,
            allowed_literature_citation_keys=(
                plan_result.allowed_literature_citation_keys
            ),
            direct_comparator_literature_keys=(
                plan_result.direct_comparator_literature_keys
            ),
            suffix_only=(pipeline._planner_strategy == "progressive_v2"),
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="warning",
                message=f"Replanner failed; keeping existing plan: {exc}",
                detail={"reason": reason},
            )
        )
        return current_plan, plan_path
    locked_robustness_specs = robustness_specs_for_execution(
        run_dir=run_dir,
        plan=current_plan,
    )
    normalized_candidate = normalize_replan_candidate(
        current_plan=current_plan,
        candidate_plan=revised,
        completed_records=completed_records or [],
        context=context,
        max_total_steps=pipeline._max_total_steps,
        locked_robustness_specs=locked_robustness_specs,
    )
    revised = normalized_candidate.plan

    candidate_contract_findings = replan_candidate_contract_findings(
        plan=revised,
        context=context,
        allowed_literature_citation_keys=(
            plan_result.allowed_literature_citation_keys
        ),
        direct_comparator_literature_keys=(
            plan_result.direct_comparator_literature_keys
        ),
        owner_declaration_findings=owner_declaration_plan_findings(plan=revised),
    )
    active_candidate_findings, candidate_contract_errors = (
        partition_replan_candidate_findings(
            normalization_findings=list(normalized_candidate.findings),
            contract_findings=candidate_contract_findings,
        )
    )
    findings.extend(active_candidate_findings)
    if candidate_contract_errors:
        findings.append(
            replan_candidate_rejection_finding(
                contract_errors=candidate_contract_errors,
                trigger=reason,
                candidate_revision=revised.revision,
            )
        )
        return current_plan, plan_path

    if not normalized_candidate.substantive:
        _replan_state["noop_streak"] += 1
        cap_noop = pipeline._max_consecutive_noop_replans
        if cap_noop and _replan_state["noop_streak"] >= cap_noop:
            _replan_state["disabled"] = True
            findings.append(
                ValidationFinding(
                    validator="replanner",
                    severity="info",
                    message=(
                        f"Replanning disabled after {_replan_state['noop_streak']} "
                        "consecutive no-op revisions (unchanged step plan)."
                    ),
                    detail={"reason": reason},
                )
            )
        return current_plan, plan_path

    if replan_review.record_runtime_replan_review_pause(
        bool(pipeline._config.require_human_plan_review),
        current_plan, revised, reason,
        _replan_state, findings, flush_partial_manifest,
    ):
        return current_plan, plan_path

    _replan_state["noop_streak"] = 0
    _replan_state["total"] += 1
    resolve_cohort_definition(revised, reason=reason)
    plan_path = register_plan_revision(revised, reason=reason)
    plan_result.plan_path = plan_path
    findings.append(
        ValidationFinding(
            validator="replanner",
            severity="info",
            message=f"Plan revised after {reason}.",
            detail={
                "from_revision": current_plan.revision,
                "to_revision": revised.revision,
            },
        )
    )
    cap_total = pipeline._max_replans
    if cap_total and _replan_state["total"] >= cap_total:
        _replan_state["disabled"] = True
        _replan_state["budget_exhausted"] = True
        findings.append(
            ValidationFinding(
                validator="replan_budget",
                severity="error",
                message=(
                    "Replan budget exhausted: "
                    f"{_replan_state['total']} substantive plan revisions "
                    f"reached the cap of {cap_total} without the plan "
                    "converging. Run demoted to diagnostic_only "
                    "(fail-closed) rather than emitting a manuscript from a "
                    "non-converging replan loop."
                ),
                detail={
                    "replan_budget_exhausted": True,
                    "cap": cap_total,
                    "substantive_revisions": _replan_state["total"],
                    "reason": reason,
                },
            )
        )
    return revised, plan_path


def _step_register_run_artifacts(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    run_result: Any,
    worker_progress: Any,
    services: Any,
    sealed_result_digests: Dict[str, str],
    script_record: Any,
    evidence: Any,
    lineage_input_evidence_ids: List[str],
    figure_role: Optional[str],
    standard_executor_terminal_block: bool,
    repair_evidence_metadata: Dict[str, Any],
    evidence_ids_for_step: List[str],
    pending_success_aliases: Dict[str, List[str]],
) -> Tuple[Optional[str], Dict[str, List[str]], List[str]]:
    """Register a step's output artifacts and return alias/claim bookkeeping."""

    declared_output_kinds: Dict[str, set[str]] = {}
    raw_output_files = step_summary.get("output_files")
    if isinstance(raw_output_files, Mapping):
        for raw_product, raw_path in raw_output_files.items():
            parsed_product = _typed_input_product(raw_product)
            if (
                parsed_product is None
                or parsed_product[0] not in {"table", "statistic", "figure", "log"}
                or not isinstance(raw_path, str)
                or Path(raw_path).name != raw_path
            ):
                continue
            declared_output_kinds.setdefault(raw_path, set()).add(parsed_product[0])
    for art in run_result.artefacts:
        if not run_result.outputs_safe_to_collect:
            continue
        if worker_progress.deterministic_standard_executor_used and (
            _is_standard_executor_internal_artifact(art)
        ):
            continue
        step_aliases = services.semantic_aliases_for(step, art)
        generation_mode = worker_progress.generation_mode()
        registered_kinds = declared_output_kinds.get(art.name, set())
        if len(registered_kinds) == 1:
            artifact_kind = next(iter(registered_kinds))
        elif art.suffix.lower() in {".csv", ".tsv", ".parquet", ".feather"}:
            artifact_kind = "table"
        elif art.suffix.lower() in {
            ".png",
            ".svg",
            ".pdf",
            ".tiff",
            ".tif",
            ".pptx",
        }:
            artifact_kind = "figure"
        else:
            artifact_kind = "log"
        artifact_evidence_id = step_owned_artifact_evidence_id(
            kind=artifact_kind,
            step_id=step.step_id,
            source_name=art.name,
            artifact_sha256=sealed_result_digests.get(
                art.name,
                sha256_of_file(art),
            ),
            script_evidence_id=script_record.evidence_id,
        )
        if art.name == "step_summary.json":
            summary_authority = "\0".join(
                (
                    step.step_id,
                    sealed_result_digests.get(
                        art.name,
                        sha256_of_file(art),
                    ),
                    script_record.evidence_id,
                )
            )
            summary_evidence_id = (
                "statistic_step_summary_"
                + hashlib.sha256(summary_authority.encode("utf-8")).hexdigest()[:16]
            )
            rec = evidence.register_file(
                kind="statistic",
                description=f"Machine-readable summary for step {step.step_id}.",
                source_path=art,
                produced_by_step=step.step_id,
                inputs=lineage_input_evidence_ids or None,
                script_evidence_id=script_record.evidence_id,
                aliases=step_aliases,
                producer="runner",
                generation_mode=generation_mode,
                evidence_id=summary_evidence_id,
                publish_aliases=False,
                metadata={
                    "script_evidence_id": script_record.evidence_id,
                    "figure_role": figure_role or "analysis_figure",
                    "diagnostic_only": standard_executor_terminal_block,
                    **repair_evidence_metadata,
                },
            )
            step_summary_record_id = rec.evidence_id
        elif artifact_kind == "table":
            rec = evidence.register_file(
                kind="table",
                description=f"Table {art.stem} from step {step.step_id}.",
                source_path=art,
                produced_by_step=step.step_id,
                inputs=lineage_input_evidence_ids or None,
                script_evidence_id=script_record.evidence_id,
                aliases=step_aliases,
                producer="runner",
                generation_mode=generation_mode,
                evidence_id=artifact_evidence_id,
                publish_aliases=False,
                metadata={
                    "script_evidence_id": script_record.evidence_id,
                    "diagnostic_only": standard_executor_terminal_block,
                    **repair_evidence_metadata,
                },
            )
        elif artifact_kind == "statistic":
            rec = evidence.register_file(
                kind="statistic",
                description=f"Statistic {art.stem} from step {step.step_id}.",
                source_path=art,
                produced_by_step=step.step_id,
                inputs=lineage_input_evidence_ids or None,
                script_evidence_id=script_record.evidence_id,
                aliases=step_aliases,
                producer="runner",
                generation_mode=generation_mode,
                evidence_id=artifact_evidence_id,
                publish_aliases=False,
                metadata={
                    "script_evidence_id": script_record.evidence_id,
                    "diagnostic_only": standard_executor_terminal_block,
                    **repair_evidence_metadata,
                },
            )
        elif artifact_kind == "figure":
            rec = evidence.register_file(
                kind="figure",
                description=f"Figure {art.stem} from step {step.step_id}.",
                source_path=art,
                produced_by_step=step.step_id,
                inputs=lineage_input_evidence_ids or None,
                script_evidence_id=script_record.evidence_id,
                aliases=step_aliases,
                producer="runner",
                generation_mode=generation_mode,
                evidence_id=artifact_evidence_id,
                publish_aliases=False,
                metadata={
                    "script_evidence_id": script_record.evidence_id,
                    "figure_role": figure_role or "analysis_figure",
                    "diagnostic_only": standard_executor_terminal_block,
                    **repair_evidence_metadata,
                },
            )
        else:
            rec = evidence.register_file(
                kind="log",
                description=f"Auxiliary artefact {art.name}.",
                source_path=art,
                produced_by_step=step.step_id,
                inputs=lineage_input_evidence_ids or None,
                script_evidence_id=script_record.evidence_id,
                aliases=step_aliases,
                producer="runner",
                generation_mode=generation_mode,
                evidence_id=artifact_evidence_id,
                publish_aliases=False,
                metadata={
                    "script_evidence_id": script_record.evidence_id,
                    "diagnostic_only": standard_executor_terminal_block,
                    **repair_evidence_metadata,
                },
            )
        pending_success_aliases[rec.evidence_id] = list(step_aliases)
        evidence_ids_for_step.append(rec.evidence_id)
    return step_summary_record_id, pending_success_aliases, evidence_ids_for_step


def _step_resolve_initial_code(
    *,
    preflight_standard_code: Optional[str],
    preflight_figure_code: Optional[str],
    resume_deterministic_repair_code: Optional[str],
    resume_critic_repair_code: Optional[str],
    preflight_resumed_code: Optional[Tuple[str, Dict[str, Any]]],
    failed_contract_code_preflight_reuse: bool,
    quarantined_resume_draft: Any,
    step_attempt_state: Any,
    step: AnalysisStep,
    step_record: Dict[str, Any],
    worker_progress: Any,
    findings: List[ValidationFinding],
    shared_lock: Any,
    standard_executor: Any,
    _use_quarantined_draft: Any,
    _use_resumed_code: Any,
    _deterministic_absolute_risk_context_code: Any,
    _deterministic_robustness_sensitivity_code: Any,
    _deterministic_missingness_audit_code: Any,
    run_dir: Path,
    pipeline: Any,
    services: Any,
    plan_result: Any,
    sealed_renderer_state: Any,
    _authorize_automatic_repair: Any,
    _record_repair: Any,
    per_step_records: List[Dict[str, Any]],
    _append_terminal_step_record: Any,
    _flush_partial_manifest: Any,
    emit_progress: Any,
    run_id: str,
    step_current: int,
    total_steps: int,
    coder_context: ResearchContext,
    coder_authority: Any,
    provider_budget: Any,
    checkpoint_authority: Any,
    _reserve_compatibility_repair: Any,
    coder: Any,
    _sync_provider_budget: Any,
    resume_controller: Any,
    prior_step_record: Any,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Resolve the code the step executes; returns ``(code, terminal_record)``.

    A non-``None`` terminal record means the step failed before execution and
    the caller must return it immediately.
    """
    if preflight_standard_code is not None:
        code = preflight_standard_code
        with shared_lock:
            findings.append(
                ValidationFinding(
                    validator="coder",
                    severity="info",
                    message=(
                        "Using the deterministic calculator for the complete "
                        "Planner-owned standard-executor specification in "
                        f"step {step.step_id}."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "analysis_kind": standard_executor.analysis_kind,
                    },
                )
            )
    elif preflight_figure_code is not None:
        code = preflight_figure_code
        with shared_lock:
            findings.append(
                ValidationFinding(
                    validator="coder",
                    severity="info",
                    message=(
                        "Using deterministic publication-figure renderer "
                        f"for figure step {step.step_id} before requesting "
                        "new coder code."
                    ),
                    detail={"step_id": step.step_id},
                )
            )
    elif resume_deterministic_repair_code is not None:
        code = resume_deterministic_repair_code
    elif step_attempt_state.selected_resume_capsule is not None:
        code = step_attempt_state.selected_resume_capsule.candidate_code
        worker_progress.resumed_code_reuse_used = True
        step_record["generation_mode"] = "resumed_code_reuse"
        step_record["step_authority_capsule_reused"] = True
        step_record["resumed_from_generation_mode"] = str(
            (prior_step_record or {}).get("generation_mode") or "capsule"
        )
    elif quarantined_resume_draft is not None:
        code = _use_quarantined_draft(quarantined_resume_draft)
    elif resume_critic_repair_code is not None:
        code = resume_critic_repair_code
    elif preflight_resumed_code is not None:
        if failed_contract_code_preflight_reuse:
            step_record["resumed_failed_contract_code_preflight"] = True
        code = _use_resumed_code(preflight_resumed_code)
    # Primary estimands and cohort selection stay agent-owned.  Deterministic
    # preflight below is limited to standard auxiliary products (descriptive
    # context, robustness replay, missingness audit, figures, and overlap
    # rendering); it must never replace a planned Cox/IPTW/ordinal method or
    # choose the analysis cohort before the coder runs.
    elif (
        _preflight_absolute_risk_code := (
            _deterministic_absolute_risk_context_code(
                "absolute_risk_context_preflight", preflight=True
            )
        )
    ) is not None:
        code = _preflight_absolute_risk_code
        with shared_lock:
            findings.append(
                ValidationFinding(
                    validator="coder",
                    severity="info",
                    message=(
                        "Using deterministic absolute-risk context runner "
                        f"before requesting new coder code for step {step.step_id}."
                    ),
                    detail={"step_id": step.step_id},
                )
            )
    elif (
        _preflight_robustness_code := _deterministic_robustness_sensitivity_code(
            "robustness_sensitivity_preflight", preflight=True
        )
    ) is not None:
        code = _preflight_robustness_code
        with shared_lock:
            findings.append(
                ValidationFinding(
                    validator="coder",
                    severity="info",
                    message=(
                        "Using deterministic robustness-sensitivity runner "
                        f"before requesting new coder code for step {step.step_id}."
                    ),
                    detail={"step_id": step.step_id},
                )
            )
    elif (
        _preflight_missingness_code := _deterministic_missingness_audit_code(
            "missingness_audit_preflight", preflight=True
        )
    ) is not None:
        # The missingness/measurement audit is a deterministic per-concept
        # count; the LLM coder reliably timed out on it (~27.6 min then fail,
        # blocking the run). The runner produces the audit table + a
        # data_quality step_summary, so the figure step then renders via the
        # parent-family fallback (data_quality -> missingness renderer).
        code = _preflight_missingness_code
        with shared_lock:
            findings.append(
                ValidationFinding(
                    validator="coder",
                    severity="info",
                    message=(
                        "Using deterministic missingness/measurement audit runner "
                        f"before requesting new coder code for step {step.step_id}."
                    ),
                    detail={"step_id": step.step_id},
                )
            )
    else:
        preflight_figure_code = _deterministic_publication_figure_code(
            "publication_figure_parent_outputs_preflight",
            run_dir=run_dir,
            step=step,
            worker_progress=worker_progress,
            pipeline=pipeline,
            authority_services=services.publication_figure_authority,
            agent_context=plan_result.agent_context,
            step_record=step_record,
            sealed_renderer_state=sealed_renderer_state,
            _authorize_automatic_repair=_authorize_automatic_repair,
            _record_repair=_record_repair,
        )
        if preflight_figure_code is not None:
            code = preflight_figure_code
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            f"Using deterministic publication-figure renderer "
                            f"for figure step {step.step_id} before requesting "
                            "new coder code."
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
        else:

            def _record_initial_coder_failure(exc: Exception) -> Dict[str, Any]:
                """Persist an ordinary provider/candidate failure as terminal.

                Receipt/capsule integrity exceptions are handled by the
                dedicated hard-raise branch below. This path only prevents
                an already failed paid generation from falling back to
                prior or untracked code.
                """

                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="coder",
                            severity="error",
                            message=(
                                f"Coder agent failed for step {step.step_id}: {exc}"
                            ),
                            detail={
                                "step_id": step.step_id,
                                "error_type": type(exc).__name__,
                            },
                        )
                    )
                    step_record["status"] = "coder_failed"
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "coder",
                    f"Coder failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            try:
                emit_progress(
                    "coder",
                    f"Generating analysis script for {step.step_id}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                code = coder.run(
                    context=coder_context,
                    step=step,
                    host_authority=coder_authority,
                    provider_budget=provider_budget,
                    initial_generation_binding=(
                        step_attempt_state.coordinates.initial_generation_binding()
                        if step_attempt_state.coordinates is not None
                        else None
                    ),
                    persist_candidate=(
                        (
                            lambda candidate: persist_candidate_code(
                                step_attempt_state.coordinates, candidate
                            )
                        )
                        if step_attempt_state.coordinates is not None
                        else None
                    ),
                    on_initial_reserved=(
                        checkpoint_authority.checkpoint_initial_reservation
                        if step_attempt_state.coordinates is not None
                        else None
                    ),
                    on_initial_candidate=(
                        (
                            lambda ref, _transport_id: (
                                checkpoint_authority.seal_initial_candidate(ref)
                            )
                        )
                        if step_attempt_state.coordinates is not None
                        else None
                    ),
                    reserve_compatibility_repair=(
                        _reserve_compatibility_repair
                        if step_attempt_state.coordinates is not None
                        else None
                    ),
                    on_repair_candidate=(
                        (
                            lambda ref, _mode, logical_id: (
                                checkpoint_authority.seal_completed_repair_candidate(
                                    ref,
                                    logical_id,
                                    failure_status="concept_failed",
                                )
                            )
                        )
                        if step_attempt_state.coordinates is not None
                        else None
                    ),
                )
                if isinstance(coder, AgenticCoderAgent):
                    step_record["step_authority_initial_transport"] = (
                        "agentic_cli_untracked"
                        if coder.last_delegation_used
                        else "fallback_provider_receipt"
                    )
                _sync_provider_budget()
            except (
                ProviderCallBudgetReceiptError,
                StepAuthorityRuntimeError,
                StepAuthorityCapsuleError,
            ):
                _sync_provider_budget()
                raise
            except Exception as exc:
                _sync_provider_budget()
                if (
                    step_attempt_state.coordinates is not None
                    and provider_budget.initial_generation_resume_status()
                    == "failed"
                ):
                    return None, _record_initial_coder_failure(exc)
                resumed_code = resume_controller.prior_code_for_step(step.step_id)
                if resumed_code is not None:
                    code = _use_resumed_code(resumed_code, error=exc)
                else:
                    fallback_code = _deterministic_publication_figure_code(
                        "publication_figure_coder_failed",
                        run_dir=run_dir,
                        step=step,
                        worker_progress=worker_progress,
                        pipeline=pipeline,
                        authority_services=services.publication_figure_authority,
                        agent_context=plan_result.agent_context,
                        step_record=step_record,
                        sealed_renderer_state=sealed_renderer_state,
                        _authorize_automatic_repair=_authorize_automatic_repair,
                        _record_repair=_record_repair,
                    )
                    if fallback_code is not None:
                        code = fallback_code
                        with shared_lock:
                            findings.append(
                                ValidationFinding(
                                    validator="coder",
                                    severity="warning",
                                    message=(
                                        f"Coder agent failed for step {step.step_id}; "
                                        "using its explicitly matched auxiliary "
                                        "deterministic fallback."
                                    ),
                                    detail={
                                        "step_id": step.step_id,
                                        "error": str(exc)[:300],
                                    },
                                )
                            )
                    else:
                        return None, _record_initial_coder_failure(exc)

    return code, None

def _step_apply_authority_resume(
    *,
    run_input_capsule_sha256: Optional[str],
    run_dir: Path,
    step: AnalysisStep,
    _deterministic_gate_stamp: Any,
    engine_code_sha256: Any,
    validator_code_sha256: Any,
    seal_repair_candidate_from_receipt: Any,
    coder_context: ResearchContext,
    coder_authority: Any,
    coder_provider_identity_sha256: str,
    resolved_inputs_path: Path,
    resolved_input_bindings: Any,
    resolved_input_evidence_ids: Any,
    cohort_path: Path,
    universe_path: Path,
    plan_result: Any,
    requested_resume_from_step_id: Optional[str],
    prior_step_record: Any,
    prior_attempt_records: Any,
    prompt_version: str,
    prompt_files: List[str],
    provider_budget: Any,
    provider_receipt_path: Path,
    reserved_final_category: Optional[str],
    llm_concept_auditor_identity_sha256: str,
    llm_concept_auditor_implementation_sha256: str,
    concept_audit_environment_sha256: str,
    step_attempt_state: Any,
    checkpoint_authority: Any,
    step_record: Dict[str, Any],
    shared_lock: Any,
    findings: List[ValidationFinding],
    _append_terminal_step_record: Any,
    _flush_partial_manifest: Any,
    per_step_records: List[Dict[str, Any]],
) -> Tuple[ResearchContext, Optional[Dict[str, Any]]]:
    """Apply an explicitly checkpointed run-input capsule, or record the miss."""

    if run_input_capsule_sha256 is None:
        step_record["step_authority_capsule_cache_miss"] = (
            "run_input_capsule_unavailable"
            if run_input_capsule_sha256 is None
            else "agentic_cli_transport_untracked"
        )
        return coder_context, None
    try:
        coder_context = prepare_step_authority_resume(
            StepAuthorityResumeRequest(
                run_dir=run_dir,
                step=step,
                run_input_capsule_sha256=run_input_capsule_sha256,
                deterministic_gate_stamp=_deterministic_gate_stamp(),
                engine_code_sha256=engine_code_sha256(),
                validator_code_sha256=validator_code_sha256(),
                seal_repair_candidate=seal_repair_candidate_from_receipt,
                coder_context=coder_context,
                coder_authority=coder_authority,
                coder_provider_identity_sha256=(coder_provider_identity_sha256),
                resolved_inputs_path=resolved_inputs_path,
                resolved_input_bindings=resolved_input_bindings,
                resolved_input_evidence_ids=resolved_input_evidence_ids,
                cohort_path=cohort_path,
                universe_path=universe_path,
                resume_state=(
                    plan_result.resume_state
                    if isinstance(plan_result.resume_state, Mapping)
                    else None
                ),
                requested_resume_from_step_id=requested_resume_from_step_id,
                prior_step_record=prior_step_record,
                prior_attempt_records=prior_attempt_records,
                prompt_version=prompt_version,
                prompt_files=prompt_files,
                provider_budget=provider_budget,
                provider_receipt_path=provider_receipt_path,
                reserved_final_category=reserved_final_category,
                llm_concept_auditor_identity_sha256=(
                    llm_concept_auditor_identity_sha256
                ),
                llm_concept_auditor_implementation_sha256=(
                    llm_concept_auditor_implementation_sha256
                ),
                concept_audit_environment_sha256=(
                    concept_audit_environment_sha256
                ),
                step_attempt_state=step_attempt_state,
                checkpoint_authority=checkpoint_authority,
                step_record=step_record,
            )
        )
    except StepAuthorityRuntimeError as exc:
        capsule_finding = ValidationFinding(
            validator="step_authority_capsule",
            severity="error",
            message=(
                f"Step {step.step_id} cannot resume because its explicitly "
                "checkpointed capsule is invalid."
            ),
            detail={"step_id": step.step_id, "reason": str(exc)},
        )
        step_record.update(
            {
                "status": "contract_failed",
                "generation_mode": "system",
                "step_authority_capsule_invalid": True,
            }
        )
        with shared_lock:
            findings.append(capsule_finding)
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        return coder_context, step_record
    return coder_context, None


def _remove_standard_executor_pending_artifacts(out_dir: Path) -> None:
    """Remove private partial files before failed-run evidence discovery."""

    for name in _STANDARD_EXECUTOR_INTERNAL_PENDING_ARTIFACTS:
        (out_dir / name).unlink(missing_ok=True)
def _is_standard_executor_internal_artifact(path: Path) -> bool:
    """Return whether *path* is a private, never-evidence work product."""

    return path.name in _STANDARD_EXECUTOR_INTERNAL_PENDING_ARTIFACTS
def _planner_locked_cohort_prompt_payload(plan: AnalysisPlan) -> str:
    """Return only the exact Planner-owned cohort definition for Coder scope."""

    cohort = plan.model_dump(mode="json", include={"cohort"}).get("cohort")
    return json.dumps(
        cohort,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
def _failed_contract_code_can_be_reused_before_coder(
    *,
    prior_step_record: Optional[Mapping[str, Any]],
    resumed_code: Optional[Tuple[str, Mapping[str, Any]]],
    step: AnalysisStep,
    plan: AnalysisPlan,
    resolved_inputs_sha256: Optional[str],
    run_input_capsule_sha256: Optional[str],
) -> bool:
    """Allow a failed deterministic-contract attempt one exact-code replay.

    An explicit step resume normally asks Coder for a fresh script.  That is
    wasteful when the previous script executed successfully and only a
    host-owned output contract failed (for example, after the contract parser
    itself is fixed).  Reuse is safe only when the checkpoint binds the exact
    code digest, step specification, and plan-wide scientific scope.  The code
    still passes every current preflight, execution, contract, concept, and
    Critic gate; this helper merely avoids paying for a replacement draft
    before those gates are rerun.

    Older or incomplete checkpoints deliberately fail closed to the normal
    Coder path rather than gaining implicit reuse authority.
    """

    if not isinstance(prior_step_record, Mapping) or resumed_code is None:
        return False
    if str(prior_step_record.get("status") or "").lower() != "contract_failed":
        return False
    if (
        prior_step_record.get("provider_call_budget_receipt_invalid") is True
        or prior_step_record.get("quarantined_requires_repair") is True
        or prior_step_record.get("resumed_failed_contract_code_preflight") is True
        or prior_step_record.get("returncode") != 0
        or prior_step_record.get("timed_out") is not False
        or prior_step_record.get("outputs_safe_to_collect") is not True
    ):
        return False

    code, evidence_record = resumed_code
    if not isinstance(code, str) or not isinstance(evidence_record, Mapping):
        return False
    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if str(prior_step_record.get("executed_code_sha256") or "") != code_sha256:
        return False
    if str(prior_step_record.get("concept_approved_code_sha256") or "") != code_sha256:
        return False
    if str(evidence_record.get("sha256") or "") != code_sha256:
        return False
    evidence_id = str(evidence_record.get("evidence_id") or "")
    if (
        not evidence_id
        or str(prior_step_record.get("script_evidence_id") or "") != evidence_id
    ):
        return False

    def _valid_sha256(value: Any) -> bool:
        return (
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        )

    for field, current_digest in (
        ("resolved_inputs_sha256", resolved_inputs_sha256),
        ("run_input_capsule_sha256", run_input_capsule_sha256),
    ):
        recorded_digest = prior_step_record.get(field)
        if (
            not _valid_sha256(recorded_digest)
            or not _valid_sha256(current_digest)
            or recorded_digest != current_digest
        ):
            return False

    recorded_scope = prior_step_record.get("plan_scientific_signature")
    if not isinstance(recorded_scope, (list, tuple)) or list(recorded_scope) != (
        _serializable_plan_scientific_scope_signature(plan)
    ):
        return False
    analysis_request = prior_step_record.get("analysis_request")
    executed_step_payload = (
        analysis_request.get("step") if isinstance(analysis_request, Mapping) else None
    )
    if not isinstance(executed_step_payload, Mapping):
        return False
    try:
        executed_step = AnalysisStep.model_validate(executed_step_payload)
    except (TypeError, ValueError):
        return False
    return _step_scientific_signature(executed_step) == _step_scientific_signature(step)
class _InertPythonNodeStripper(ast.NodeTransformer):
    """Remove syntax that cannot repair analytical behavior."""

    def visit_Pass(self, node: ast.Pass) -> None:
        del node
        return None

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.Expr]:
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Constant):
            return None
        return node
def _python_semantic_sha256(code: str) -> Optional[str]:
    """Hash executable Python structure while ignoring comments/whitespace."""

    try:
        tree = _InertPythonNodeStripper().visit(ast.parse(code))
        normalized = ast.dump(
            tree,
            annotate_fields=True,
            include_attributes=False,
        )
    except (SyntaxError, TypeError, ValueError):
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
def _python_repair_is_materially_changed(before: str, after: str) -> bool:
    """Reject exact and AST-equivalent repair responses."""

    if (
        hashlib.sha256(before.encode("utf-8")).digest()
        == hashlib.sha256(after.encode("utf-8")).digest()
    ):
        return False
    before_semantic = _python_semantic_sha256(before)
    after_semantic = _python_semantic_sha256(after)
    if before_semantic is not None and before_semantic == after_semantic:
        return False
    return True
def _actionable_validator_messages(
    *finding_groups: Sequence[ValidationFinding],
) -> List[str]:
    """Return only blocking validator messages that require Critic action.

    Warning and informational audit records remain in the manifest and global
    findings, but the untyped Critic input cannot preserve their severity. If
    forwarded as bare strings they become ``needs_revision`` and incorrectly
    fail an otherwise valid step. Only fail-closed errors are actionable here.
    """

    return [
        finding.message
        for finding in _blocking_validator_findings(*finding_groups)
        if finding.message
    ]
_CAPSULE_TRANSIENT_STEP_STATUSES = {
    "initial_generation_pending",
    "repair_transport_pending",
    "candidate_checkpointed",
    "capsule_revalidation_pending",
    "concept_audited_pending_review",
    "executed_pending_review",
}
def _append_terminal_step_record(
    records: List[Dict[str, Any]],
    record: Dict[str, Any],
) -> None:
    """Replace this attempt's capsule checkpoint instead of retaining both."""

    step_id = record.get("step_id")
    attempt_id = record.get("attempt_id")
    records[:] = [
        existing
        for existing in records
        if not (
            existing.get("step_id") == step_id
            and existing.get("attempt_id") == attempt_id
            and existing.get("status") in _CAPSULE_TRANSIENT_STEP_STATUSES
        )
    ]
    records.append(record)
def _upsert_current_capsule_checkpoint(
    records: List[Dict[str, Any]],
    record: Dict[str, Any],
) -> None:
    """Append a new attempt, replacing only its own latest transient state."""

    step_id = record.get("step_id")
    attempt_id = record.get("attempt_id")
    for index in range(len(records) - 1, -1, -1):
        existing = records[index]
        if existing.get("step_id") != step_id:
            continue
        if (
            existing.get("attempt_id") == attempt_id
            and existing.get("status") in _CAPSULE_TRANSIENT_STEP_STATUSES
        ):
            records[index] = record
        else:
            records.append(record)
        return
    records.append(record)
_SUCCESS_REPLAN_REQUEST_FIELDS = (
    "replan_requested",
    "plan_revision_requested",
)


def _successful_step_requests_replan(
    record: Mapping[str, Any],
    *,
    progressive_observation_loop: bool = False,
) -> bool:
    """Return whether a clean agent step explicitly requests plan adaptation.

    The deterministic probe already receives one automatic replan and failed
    model steps have their own bounded directed-replan path. Calling the LLM
    replanner after every ordinary legacy step adds latency and usually
    produces a no-op, so those paths still require an exact boolean request.
    Progressive v2 explicitly runs an observation loop and therefore opts in
    after each successful step with a remaining suffix. Strings and other
    truthy values are intentionally not accepted as explicit requests.
    """

    if str(record.get("status") or "") != "ok":
        return False
    containers: List[Mapping[str, Any]] = [record]
    summary = record.get("step_summary")
    if isinstance(summary, Mapping):
        containers.append(summary)
    explicit_request = any(
        container.get(field) is True
        for container in containers
        for field in _SUCCESS_REPLAN_REQUEST_FIELDS
    )
    return explicit_request or progressive_observation_loop


def _step_status_from_contract_findings(
    *,
    contract_findings: Sequence[ValidationFinding],
    figure_source_findings: Sequence[ValidationFinding],
    stat_findings: Sequence[ValidationFinding],
    critique_status: Optional[str] = None,
) -> str:
    """Map deterministic review failures to the outer step status.

    A Critic ``needs_revision`` decision is not a successful scientific step.
    Contract validation normally catches objective defects early enough for an
    in-run coder repair, but the Critic is the final independent review layer;
    its negative decision must therefore remain fail-closed rather than being
    stored as a warning on an otherwise ``ok`` record.
    """

    has_contract_error = any(
        finding.severity == "error"
        for finding in (
            list(contract_findings) + list(figure_source_findings) + list(stat_findings)
        )
    )
    if has_contract_error:
        return "contract_failed"
    if str(critique_status or "").strip().lower() in {
        "needs_revision",
        "blocked",
    }:
        return "critic_failed"
    return "ok"
_LOCKED_MEASUREMENT_DATA_QUALITY_ISSUES = frozenset(
    {
        "measurement_provenance_count_column_ambiguous",
        "measurement_provenance_count_flag_discordance",
        "measurement_provenance_host_replay_failed",
        "measurement_provenance_host_source_missing",
        "measurement_provenance_host_source_unreadable",
        "measurement_provenance_invalid_measured_values",
        "measurement_provenance_invalid_pairs",
        "measurement_provenance_measured_column_missing",
    }
)
def _locked_measurement_data_quality_issues(
    contract_findings: Sequence[ValidationFinding],
) -> List[str]:
    """Identify locked-cohort facts that generated code cannot repair."""

    return sorted(
        {
            str(finding.detail.get("issue"))
            for finding in contract_findings
            if finding.severity == "error"
            and finding.validator == StepSummaryIntegrityValidator.name
            and finding.detail.get("issue") in _LOCKED_MEASUREMENT_DATA_QUALITY_ISSUES
        }
    )
def _step_requires_publication_figure_exports(step: AnalysisStep) -> bool:
    """Return whether ``step`` structurally owns a figure export contract.

    Step ids and intents are narrative metadata and may mention a downstream
    publication figure without declaring one as this step's product.  The
    mandatory export gate therefore accepts only an exact publication-renderer
    method or the closed method/output evidence recognised by
    :func:`_step_expects_figure`.
    """

    method = str(step.method or "").strip().lower()
    return method == "publication_figure_generation" or _step_expects_figure(step)
def _coder_authority_with_locked_robustness_specs(
    *,
    authority: HostCoderAuthority,
    context: ResearchContext,
    step: AnalysisStep,
    run_dir: Path,
) -> HostCoderAuthority:
    """Attach the planner-locked variant contract out of band."""

    if not _is_cohort_definition_sensitivity_result_step(step):
        return authority
    try:
        specs = _read_locked_robustness_spec_dicts(run_dir)
    except Exception:
        return authority
    if not specs:
        return authority
    fields = (
        "spec_id",
        "axis",
        "description",
        "cohort_override",
        "missing_override",
        "outcome_override",
    )
    locked_contract = [{field: spec.get(field) for field in fields} for spec in specs]
    primary_contract: Optional[Dict[str, Any]] = None
    for manifest_name in ("manifest_partial.json", "manifest.json"):
        manifest_path = Path(run_dir) / manifest_name
        if not manifest_path.is_file():
            continue
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_records = (
            manifest_payload.get("per_step_records")
            if isinstance(manifest_payload, Mapping)
            else None
        )
        if not isinstance(raw_records, list):
            continue
        primary_contract = _authoritative_primary_robustness_contract(
            completed_step_records=raw_records,
            context=context,
        )
        if primary_contract is not None:
            break
    attachment = (
        "LOCKED ROBUSTNESS SPECIFICATIONS (binding plan-time state):\n"
        + json.dumps(locked_contract, ensure_ascii=False, separators=(",", ":"))
        + "\nExecute every spec_id exactly as declared; do not rename, replace, "
        "or invent specifications. Cohort-axis definitions that can recover "
        "rows outside the locked analysis cohort must be materialised from "
        "os.environ['EASYICU_UNIVERSE_PARQUET']; COHORT_PARQUET is the locked "
        "analysis cohort."
        "\n\n" + ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE
    )
    if primary_contract is not None:
        attachment += (
            "\n\nAUTHORITATIVE PRIMARY MODEL CONTRACT (binding; variants must "
            "re-estimate this model rather than substitute descriptive risks):\n"
            + json.dumps(
                primary_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return authority.append(attachment)
_MAX_DIRECTED_MODEL_REPLANS = 2
def _contract_repair_log(
    findings: Sequence[ValidationFinding],
) -> str:
    """Serialize an untrusted diagnostic mirror of contract failures."""

    return json.dumps(
        [
            {
                "validator": finding.validator,
                "severity": finding.severity,
                "message": finding.message,
                "detail": finding.detail,
            }
            for finding in findings
        ],
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
def _is_terminal_publication_figure_repair_step(step: Any) -> bool:
    """Return true for rendering-only terminal publication figure repair steps."""

    expected_outputs = getattr(step, "expected_outputs", None) or []
    method = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(getattr(step, "method", "") or "").strip().lower(),
    ).strip("_")
    rendering_methods = {
        "publication_figure_generation",
        "publication_figure_repair",
        "rendering_only_repair_from_primary_results",
    }
    if method not in rendering_methods or not expected_outputs:
        return False
    return all(_output_declares_figure(str(output)) for output in expected_outputs)
def _publication_bundle_has_primary_result_roles(outputs_dir: Path) -> bool:
    """Check whether an output directory already has a primary-result figure bundle."""

    contract_path = outputs_dir / "publication_figure.figure_contract.json"
    if not contract_path.exists():
        return False
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    panels = contract.get("panels") if isinstance(contract, Mapping) else None
    if not isinstance(panels, list):
        return False
    roles = {
        str(panel.get("role") or "").strip()
        for panel in panels
        if isinstance(panel, Mapping)
    }
    if not {"descriptive_result", "primary_estimand"}.issubset(roles):
        return False

    export_formats = contract.get("export_formats")
    if not isinstance(export_formats, list) or not export_formats:
        export_formats = ["svg", "png", "pdf", "tiff"]
    if not any(
        (outputs_dir / f"publication_figure.{str(ext).lstrip('.')}").exists()
        for ext in export_formats
    ):
        return False

    source_data = contract.get("source_data")
    if isinstance(source_data, list):
        source_paths = [
            outputs_dir / str(name)
            for name in source_data
            if isinstance(name, str) and Path(name).suffix
        ]
        if source_paths and not all(path.exists() for path in source_paths):
            return False
    return True
def _terminal_publication_repair_replan_skip_detail(
    *,
    plan: Any,
    completed_records: Optional[Sequence[Dict[str, Any]]],
    run_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Return a skip reason when replanning would only delay deterministic repairs."""

    current_records = current_step_records(completed_records or [])
    completed_ok = {
        str(record.get("step_id") or "")
        for record in current_records
        if record.get("status") == "ok" and record.get("step_id")
    }
    remaining_steps = [
        step
        for step in getattr(plan, "steps", []) or []
        if str(getattr(step, "step_id", "") or "") not in completed_ok
    ]
    if not remaining_steps:
        return None
    if not all(
        _is_terminal_publication_figure_repair_step(step) for step in remaining_steps
    ):
        return None

    for record in reversed(current_records):
        if record.get("status") != "ok" or not record.get("step_id"):
            continue
        step_id = str(record["step_id"])
        outputs_dir = run_dir / "steps" / step_id / "outputs"
        if _publication_bundle_has_primary_result_roles(outputs_dir):
            return {
                "remaining_step_ids": [
                    str(getattr(step, "step_id", "") or "") for step in remaining_steps
                ],
                "satisfied_by_step_id": step_id,
                "satisfied_by_outputs_dir": str(outputs_dir),
            }
    return None
def _detached_figure_repair_binding(
    *,
    step: AnalysisStep,
    plan: AnalysisPlan,
    completed_records: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[str, str, List[str]]]:
    """Bind a detached rendering-only repair to one failed figure target.

    The binding is orchestrator-owned: it comes from the current plan and
    latest outer step ledger, never from the renderer's self-reported
    ``parent_step`` text. Ambiguous repairs remain unbound and therefore cannot
    receive execution credit.
    """

    if not _is_terminal_publication_figure_repair_step(step):
        return None
    latest = {
        str(record.get("step_id") or ""): record
        for record in current_step_records(completed_records)
    }
    plan_steps = {
        str(candidate.step_id or ""): candidate for candidate in plan.steps or []
    }
    declared_step_inputs = {
        str(value or "").strip()
        for value in (step.inputs or [])
        if str(value or "").strip() in plan_steps
    }
    candidates: List[Tuple[str, str, List[str]]] = []
    for target_step_id, target_step in plan_steps.items():
        if target_step_id == str(step.step_id or ""):
            continue
        target_record = latest.get(target_step_id)
        target_status = str((target_record or {}).get("status") or "").strip().lower()
        if target_record is None or target_status not in {
            "execution_failed",
            "contract_failed",
            "repair_failed",
        }:
            continue
        if not _step_has_figure_only_output_contract(target_step):
            continue
        source_step_id = _parent_step_id_for_figure_step(target_step)
        if source_step_id is None:
            continue
        source_record = latest.get(source_step_id)
        if (
            source_record is None
            or str(source_record.get("status") or "").strip().lower() != "ok"
        ):
            continue
        if declared_step_inputs and not (
            {target_step_id, source_step_id} & declared_step_inputs
        ):
            continue
        source_evidence_ids = [
            str(evidence_id)
            for evidence_id in (source_record.get("evidence_ids") or [])
            if str(evidence_id).strip()
        ]
        if not source_evidence_ids:
            continue
        candidates.append((target_step_id, source_step_id, source_evidence_ids))
    if len(candidates) != 1:
        return None
    return candidates[0]
_SEALED_AUTHORITY_SUMMARY_MARKERS = (
    "sealed_renderer_repair",
    "sealed_renderer_implementation_sha256",
    "sealed_renderer_parent_digests",
    "planner_bound_figure_products",
    "planner_product_slot_bindings",
    "planner_product_binding",
)
def _unowned_sealed_authority_markers(
    step_summary: Mapping[str, Any],
    *,
    authorized_code_sha256: Optional[str],
) -> List[str]:
    """Reject sealed provenance unless the host authorized it pre-execution."""

    if authorized_code_sha256 is not None:
        return []
    return [
        marker for marker in _SEALED_AUTHORITY_SUMMARY_MARKERS if marker in step_summary
    ]
def _max_finding_severity(
    findings_for_step: Sequence[ValidationFinding],
) -> Optional[str]:
    """Return the strongest severity across findings (error > warning > info)."""
    if any(f.severity == "error" for f in findings_for_step):
        return "error"
    if any(f.severity == "warning" for f in findings_for_step):
        return "warning"
    if any(f.severity == "info" for f in findings_for_step):
        return "info"
    return None
def scope_findings_to_records(
    evidence_ids: Sequence[str],
    findings_for_step: Sequence[ValidationFinding],
) -> Dict[str, tuple[Optional[str], List[str]]]:
    """Map each step output record to the caveat that actually concerns it.

    A finding that names specific records (``finding.evidence_ids``) taints
    ONLY those records. A step-global finding — no evidence_ids, e.g. an
    "immortal-time-bias risk" or "cohort is keyed at the stay level"
    advisory — describes the ANALYSIS DESIGN, not any one artifact.
    Blanket-tainting every output record with a step-global WARNING made the
    primary result table uncitable and the manuscript unwinnable: one design
    advisory flags ``table_one`` / ``adjusted_association``, and the
    manifest-caveat gate then blocks any draft that cites them (which every
    real Results section must). Those advisories still live in the manifest
    findings list and reach the writer as limitations — they simply no longer
    masquerade as per-artifact taint.

    Step-global ERRORS keep the blanket behaviour (fail-closed: a step-level
    error means the step's outputs are not to be trusted).

    Returns ``{evidence_id: (severity_or_None, messages)}``.
    """
    targeted: Dict[str, List[ValidationFinding]] = {}
    for finding in findings_for_step:
        for eid in finding.evidence_ids or []:
            targeted.setdefault(str(eid), []).append(finding)

    global_error_findings = [
        f for f in findings_for_step if f.severity == "error" and not f.evidence_ids
    ]
    global_error_messages = [f.message for f in global_error_findings]

    scoped: Dict[str, tuple[Optional[str], List[str]]] = {}
    for evidence_id in evidence_ids:
        eid = str(evidence_id)
        relevant = targeted.get(eid, [])
        severity = _max_finding_severity(list(relevant) + global_error_findings)
        messages = [
            f.message for f in relevant if f.severity in {"warning", "error"}
        ] + global_error_messages
        scoped[eid] = (severity, messages)
    return scoped
def _load_step_summary_from_outputs(out_dir: Path) -> Dict[str, Any]:
    """Load the current staged summary without granting it evidence authority."""

    summary_path = out_dir / "step_summary.json"
    if not summary_path.exists():
        return {}
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        loaded = None
    return bind_primary_output(loaded, out_dir)
def build_self_block_replan_directive(
    *,
    failed_step: AnalysisStep,
    failed_record: Mapping[str, Any],
    completed_records: Sequence[Mapping[str, Any]],
    viability: "CohortViability",
) -> Optional[str]:
    """Return a viability-conditioned override directive when a model/estimation
    step self-blocked on a task-viable cohort, else ``None``.

    Pure and deterministic so the trigger logic is unit-testable without a run.
    Fires only when ALL hold: the failed step's contract requires model
    performance statistics (``statistic:auroc`` / ``statistic:brier_score``); the
    cohort cleared the viability floor; and a deliberate block signal is present
    on the failed step or an upstream completed step (e.g. a
    ``modeling_block_registration`` step). Stays silent otherwise — a genuinely
    non-viable cohort or a hard crash leaves blocking legitimate.

    Impartiality: the directive is conditioned on viability twice over — the
    trigger requires ``viability.viable`` and the directive text itself reaffirms
    that blocking stays legitimate on genuinely non-viable data. It never
    dictates which model to fit, only that a model must actually be fit.
    """
    if not step_requires_model_performance(failed_step.expected_outputs):
        return None
    if not viability.viable:
        return None
    block_reason = step_summary_block_signal(failed_record.get("step_summary") or {})
    if not block_reason:
        for rec in completed_records:
            if not isinstance(rec, Mapping):
                continue
            block_reason = step_summary_block_signal(rec.get("step_summary") or {})
            if block_reason:
                break
    if not block_reason:
        return None
    return (
        "The locked analysis cohort is task-viable (" + viability.note + "), yet "
        "the modeling step recorded a non-execution/blocked status "
        f'("{block_reason}") and produced no model and no required performance '
        "statistics (AUROC / Brier). On a cohort this populated, declaring the "
        "repaired artifacts unusable, registering a modeling block, or emitting a "
        "non-execution model stub is NOT an acceptable outcome for this task. "
        "Revise the remaining plan so the primary modeling step actually fits a "
        "model on the available predictors and emits the required performance "
        "statistics. Do NOT re-insert any step whose purpose is to gate, block, "
        "or declare the modeling unexecutable on this cohort. (Blocking would be "
        "legitimate only if "
        "the data were genuinely non-viable — too few rows, no outcome variation, "
        "or no usable predictors — which is not the case here.)"
    )
_ORDINAL_PRIMARY_METHODS = frozenset(
    {
        "dose_response",
        "dose_response_analysis",
        "ordinal_regression",
        "ordinal_logistic_regression",
        "trend_analysis",
        "association",
        "association_analysis",
        "stratified_analysis",
        "subgroup_analysis",
        "regression",
        "logistic_regression",
        "glm",
        "modeling",
        "model",
        "estimation",
        "ordinal",
    }
)
_ORDINAL_EXPLICIT_METHODS = frozenset(
    {
        "dose_response",
        "dose_response_analysis",
    }
)
_ORDINAL_OUTPUT_PRODUCTS = frozenset(
    {
        "dose_response",
        "per_stage",
        "per_stage_odds",
        "per_stage_odds_ratio",
        "per_stage_odds_ratios",
        "trend_or",
        "ordinal_trend",
        "ordinal_trend_model",
    }
)
_COHORT_DEF_SENSITIVITY_METHODS = frozenset(
    {
        "cohort_definition_sensitivity",
        "cohort_sensitivity",
        "definition_sensitivity",
    }
)
_COHORT_DEF_SENSITIVITY_OUTPUT_TOKENS = (
    "alternative_cohort_attrition",
    "cohort_overlap",
    "overlap_and_movement_across_cohorts",
    "sensitivity_grid",
    # Not "sensitivity_comparison": it substring-matches within-cohort comparison
    # outputs. Each kept token uniquely signals an across-definition comparison.
    "definition_sensitivity",
    "sensitivity_definition_summary",
    "outcome_by_definition",
    "adjustment_denominator_sensitivity",
)
_PRIMARY_COHORT_FLOW_METHODS = frozenset(
    {
        "cohort_construction",
        "cohort_definition",
        "eligibility_definition",
    }
)
_PRIMARY_COHORT_FLOW_OUTPUTS = frozenset(
    {
        "cohort_attrition",
        "cohort_denominator",
        "cohort_denominators",
        "cohort_flow",
        "attrition_by_rule",
        "eligibility_flow",
    }
)
_EFFECT_ASSOCIATION_METHOD_TOKENS = frozenset(
    {
        "association",
        "causal",
        "cox",
        "effect",
        "estimand",
        "hazard",
        "logistic",
        "logit",
        "mixed",
        "model",
        "prediction",
        "regression",
        "survival",
    }
)
_EFFECT_OUTPUT_FRAGMENTS = (
    "adjusted_effect",
    "association_estimate",
    "coefficient",
    "odds_ratio",
    "hazard_ratio",
    "risk_ratio",
    "risk_difference",
    "primary_estimate",
    "primary_or",
    "primary_hr",
    "c_statistic",
    "c_index",
    "auroc",
    "cox_summary",
)
def _method_is_effect_or_association(method: str) -> bool:
    head = _method_head(method)
    tokens = set(filter(None, re.split(r"[_\-\s]+", head)))
    return bool(tokens & _EFFECT_ASSOCIATION_METHOD_TOKENS)
def _declares_effect_output(expected_outputs: Sequence[str]) -> bool:
    """True for structured primary-effect/model outputs, including OR/HR."""

    for output in expected_outputs or []:
        value = str(output or "").strip().lower()
        if any(fragment in value for fragment in _EFFECT_OUTPUT_FRAGMENTS):
            return True
        tokens = set(re.findall(r"[a-z0-9]+", value))
        if tokens & {"or", "hr", "auc"}:
            return True
    return False
def _primary_cohort_flow_runner_owns_step(
    method: str,
    step_id: str,
    intent: str,
    expected_outputs: Sequence[str],
) -> bool:
    """True for the owner that defines the single locked primary cohort.

    Alternative-definition/overlap/sensitivity steps are deliberately excluded;
    those have separate deterministic runners.  The owner must declare an
    attrition/denominator output, so a generic preparation step is not hijacked.
    """

    del step_id, intent
    method_normalized = str(method or "").lower()
    expected_names = _closed_auxiliary_output_products(
        expected_outputs,
        supported_products=_PRIMARY_COHORT_FLOW_OUTPUTS,
    )
    if expected_names is None:
        return False
    method_head = _method_head(method_normalized)
    if _method_is_effect_or_association(method_head) or _declares_effect_output(
        expected_outputs
    ):
        return False
    return method_head in _PRIMARY_COHORT_FLOW_METHODS
_RICH_EXPOSURE_AUDIT_OUTPUT_TOKENS = (
    "exposure_distribution",
    "joint_availability",
    "complete_case_attrition",
    "score_level_distribution",
    "score_completeness",
    "invalid_range",
    "model_availability",
    "source_reconciliation",
)
_COMPACT_MISSINGNESS_SUPPORTED_OUTPUTS = frozenset(
    {
        "missingness_audit",
        "missingness_measurement_audit",
        "measurement_audit",
        "measurement_source_audit",
        "measurement_process_audit",
        "data_quality_audit",
        "source_coverage",
        "cohort_flow",
        "analytic_denominator",
        "analytic_denominators",
    }
)
_COMPACT_MISSINGNESS_METHODS = frozenset(
    {
        "missingness_audit",
        "missingness",
        "measurement_audit",
        MISSINGNESS_SOURCE_AVAILABILITY_AUDIT,
        "measurement_process_audit",
        "data_quality_audit",
        "data_quality",
    }
)
_ROBUSTNESS_SENSITIVITY_METHODS = frozenset(
    {
        "prespecified_robustness",
        "robustness_sensitivity",
        "sensitivity_comparison",
    }
)
def _is_cohort_definition_sensitivity_step(
    method: str,
    step_id: str,
    intent: str,
    expected_outputs: Sequence[str],
) -> bool:
    """Pure routing test: is this an ACTUAL cohort-definition-sensitivity step?

    Require an exact method head plus a closed comparison product, or a pair of
    closed across-definition products. Step ids and prose never establish the
    role. This keeps ordinary within-cohort sensitivity language from vetoing a
    legitimate primary estimand step.
    """
    del step_id, intent
    head = _method_head(str(method or "").lower())
    expected_names = _normalised_expected_output_names(expected_outputs)
    matched_outputs = expected_names & set(_COHORT_DEF_SENSITIVITY_OUTPUT_TOKENS)
    return head in _COHORT_DEF_SENSITIVITY_METHODS and bool(matched_outputs)
def _cohort_definition_sensitivity_runner_owns_step(
    method: str,
    step_id: str,
    intent: str,
    expected_outputs: Sequence[str],
) -> bool:
    """Legacy comparator code is explicit-only and never a preflight owner.

    The historical script reconstructed cohorts, chose covariates, and refit a
    GLM.  Those are scientific decisions, so no method/output combination may
    automatically replace the coder with that script.
    """

    del method, step_id, intent, expected_outputs
    return False
def _cohort_definition_overlap_runner_owns_step(
    method: str,
    expected_outputs: Sequence[str],
) -> bool:
    """Legacy cohort-construction code is explicit-only, never automatic."""

    del method, expected_outputs
    return False
def _simple_missingness_audit_runner_owns_step(
    method: str,
    step_id: str,
    intent: str,
    expected_outputs: Sequence[str],
) -> bool:
    """True when the compact per-concept missingness runner owns the contract."""

    if _normalised_expected_output_names(expected_outputs) & set(
        _RICH_EXPOSURE_AUDIT_OUTPUT_TOKENS
    ):
        return False
    declared_names = _closed_auxiliary_output_products(
        expected_outputs,
        supported_products=_COMPACT_MISSINGNESS_SUPPORTED_OUTPUTS,
    )
    if declared_names is None:
        # A method label such as ``data_quality_audit`` is not sufficient
        # ownership.  If even one declared artefact belongs to a different
        # contract (e.g. representation reconciliation), leave the step to its
        # coder instead of returning a successful but irrelevant compact audit.
        return False

    method_head = _method_head(method)
    if method_head not in _COMPACT_MISSINGNESS_METHODS:
        return False
    if _declares_effect_output(expected_outputs):
        return False
    return True
def _absolute_risk_context_runner_owns_step(
    method: str,
    step_id: str,
    expected_outputs: Sequence[str],
) -> bool:
    """True for a descriptive exposure-prevalence / absolute-risk owner."""

    del step_id
    # No figure clause. MEASURED over 1,965 recorded plan steps: 597 declare a
    # figure and the closed-product clause below refuses every one of them on
    # its own -- ``figure:absolute_risk_context`` is not a supported PRODUCT, so
    # the set never closes. A guard that has never decided a case reads as
    # protection while protecting nothing; the property it stood for is pinned
    # by a test instead, so a change to the product reader fails loudly rather
    # than being silently absorbed here.
    supported_products = {
        "exposure_outcome_summary",
        "exposure_prevalence_and_absolute_risk",
        "absolute_risk",
        "absolute_risk_context",
    }
    # No method allowlist. It had three spellings -- ``absolute_risk_context``,
    # ``descriptive_context``, ``exposure_outcome_summary`` -- and MEASURED over
    # 89 recorded steps promising ``table:absolute_risk_context``, the Planner
    # wrote none of them: 52 said ``descriptive``, 7
    # ``descriptive_binary_outcome_summary``, 5
    # ``prevalence_and_absolute_risk_summary``, and so on down a tail of
    # synonyms. So this owner -- registered in the capability registry,
    # advertised to the Planner, and wired into the dispatcher -- claimed 0 of
    # 89, the Coder wrote the table every time, and the figures drawn over it
    # died 46 times out of 47. The sibling table written by a real owner ran the
    # other way: 40 of 51 figures over it passed.
    #
    # The promised product IS the claim, and the clauses below already do every
    # bit of the discrimination: a figure output, an effect method or an effect
    # output, and a product set outside this owner's four are each refused.
    # Re-measured over all 1,965 recorded plan steps, dropping the allowlist
    # claims 77 of the 89 and adds exactly ZERO steps that do not promise the
    # product -- so the allowlist only ever subtracted correct claims. This is
    # the same conclusion the exposure/outcome distribution owner reached about
    # its own two-string allowlist.
    if _method_is_effect_or_association(method) or _declares_effect_output(
        expected_outputs
    ):
        return False
    structured_products = _closed_auxiliary_output_products(
        expected_outputs,
        supported_products=supported_products,
    )
    if structured_products is not None:
        return True
    # A reconciliation/audit step may mention absolute-risk context while
    # owning different artefacts (representation reconciliation, gap notes,
    # etc.).  The compact runner must not claim such a step merely because its
    # id contains ``absolute_risk_context``; it only owns the closed output
    # contract above.
    return False
def _robustness_sensitivity_runner_owns_step(
    method: str,
    step_id: str,
    expected_outputs: Sequence[str],
    *,
    step: Optional[AnalysisStep] = None,
) -> bool:
    """True for a separate prespecified robustness-comparison owner.

    A step carrying an EMITTABLE ``robustness_replay_spec`` has stated outright
    that it is the locked-grid replay, so neither the method label below nor the
    product names need decide anything for it.  Both were string sets, and over
    the recorded corpus the method set is the one that bled: 182 robustness
    steps that were neither figures nor claimed by the agent-owned validation
    gate were turned away by it.

    AN INCOMPLETE DECLARATION MUST NOT COST THE STEP ITS OWNER.  Until
    2026-07-31 the spec branch *returned* rather than fell through, so declaring
    a spec replaced the label path outright -- and a spec that did not yet back
    every promised product answered ``False``, permanently.  Measured over the
    recorded plans: 10 steps that declare NO spec are claimed by the label path,
    and 8 that declare a PARTIAL one are refused -- every one of which the label
    path would have claimed had the field simply been left empty.  The host was
    punishing the Planner for answering half its question, which teaches exactly
    the wrong lesson and is invisible from the plan.

    So an unemittable spec now falls through to the same path an absent one
    takes: no better off, and no worse.  The gap is still reported --
    ``robustness_replay_declaration_verdict`` names the unbacked products for
    the plan-time gate -- so the Planner is still asked to close it, but the
    step is not left to the Coder in the meantime for having tried.
    """

    del step_id
    # A promised figure is refused structurally by both paths below, so the
    # explicit `figure:` guard that used to stand here was unreachable and was
    # deleted (verified 2026-07-31): `figure` is not one of the three
    # `_AUXILIARY_OUTPUT_KINDS`, so `_closed_auxiliary_output_products` returns
    # None for any step promising one, and no replay `output` names a figure,
    # so `robustness_replay_spec_is_emittable` is False for one too. The
    # property is locked by a test rather than by a guard that never fires.
    if (
        step is not None
        and step.robustness_replay_spec is not None
        and robustness_replay_spec_is_emittable(step)
    ):
        return True
    if (
        step is not None
        and step.robustness_replay_spec is not None
        and robustness_replay_spec_has_kind_mismatch(step)
    ):
        # Incomplete declarations retain the characterised label fallback;
        # incompatible declarations do not. Otherwise a plan that maps a CSV
        # replay output but promises it as JSON is claimed and only fails after
        # irrelevant code-repair calls.
        return False
    method_head = _method_head(method)
    if method_head not in _ROBUSTNESS_SENSITIVITY_METHODS:
        return False
    supported_products = {
        "robustness_matrix",
        "robustness_summary",
        "complete_case_n",
        "primary_or",
        "missingness_strategy_notes",
    }
    structured_products = _closed_auxiliary_output_products(
        expected_outputs,
        supported_products=supported_products,
    )
    if structured_products is None:
        return False
    has_matrix = "robustness_matrix" in structured_products
    has_summary_contract = {
        "robustness_summary",
        "complete_case_n",
    }.issubset(structured_products)
    return has_matrix or has_summary_contract
def _method_has_ordinal_primary_token(method: str) -> bool:
    """True if ``method`` IS, or is a compound built from, a primary-estimation
    method token (e.g. ``multivariable_association`` -> ``association``,
    ``adjusted_logistic_regression`` -> ``regression``).

    Word-boundary token match (split on ``_`` / ``-``), NOT substring, so
    ``remodeling`` never matches ``model``. This is only ever reached after the
    closed ordinal-product gate in :func:`_ordinal_dose_response_step_matches`;
    a plain association label cannot establish ownership on its own.
    """
    if method in _ORDINAL_PRIMARY_METHODS:
        return True
    tokens = method.replace("-", "_").split("_")
    return any(tok in _ORDINAL_PRIMARY_METHODS for tok in tokens)
def _ordinal_dose_response_step_matches(
    method: str, blob: str, expected_blob: str
) -> bool:
    """Pure routing test: is this the PRIMARY dose-response estimation step?

    This legacy compatibility predicate is unit-testable without a full run. The
    caller supplies lowercased strings and has already excluded figure and
    cohort-definition-sensitivity steps.

    ``blob`` = step_id + intent + research_question + expected_outputs;
    ``expected_blob`` = expected_outputs only.
    """
    del blob
    head = _method_head(method)
    products = _normalised_structured_output_names(expected_blob)
    if not products.intersection(_ORDINAL_OUTPUT_PRODUCTS):
        return False
    return head in _ORDINAL_EXPLICIT_METHODS or _method_has_ordinal_primary_token(head)
def _trajectory_clustering_step_matches(
    method: str,
    blob: str,
    expected_blob: str = "",
) -> bool:
    """Whether a legacy KMeans artifact contract is phenotype-compatible.

    The caller supplies lowercased strings and has already excluded figure steps.
    Compatibility requires an explicit KMeans method head plus at least two
    standard clustering products.  A primary EFFECT step (OR/HR/AUROC) is always
    excluded, and latent-class/GMM/unspecified phenotyping remains agent-owned so
    the auxiliary cannot silently replace the planned scientific method.
    """
    expected_outputs = re.split(r"[\s,]+", str(expected_blob or ""))
    if _declares_effect_output(expected_outputs):
        return False
    return _clustering_contract_applies(
        method=str(method or ""),
        intent=str(blob or ""),
        expected_outputs=str(expected_blob or ""),
        auxiliary_kmeans_only=True,
        minimum_output_signals=2,
    )
def _fresh_plausibility_receipt_findings(
    step_summary: Mapping[str, Any],
    step: AnalysisStep,
    script_text: str,
    authority: StepPlausibilityAuthority,
) -> List[ValidationFinding]:
    return plausibility_audit_receipt_findings(
        step_summary=step_summary,
        step=step,
        script_text=script_text,
        scope=authority.scope,
    )
def _non_llm_interpretation_for_generation(
    *, step_id: str, generation_mode: str
) -> Optional[Tuple[str, str]]:
    mode_labels = {
        "resumed_code_reuse": "resumed agent-generated code",
        "fallback": "deterministic fallback code",
        "deterministic_standard": "host-owned deterministic standard code",
    }
    mode_label = mode_labels.get(generation_mode)
    if mode_label is None:
        return None
    interpretation_mode = {
        "resumed_code_reuse": "resumed_code_reuse",
        "fallback": "deterministic_fallback",
        "deterministic_standard": "deterministic_standard",
    }[generation_mode]
    return (
        f"Step `{step_id}` was executed from {mode_label}. "
        "Review the registered step summary and artefacts for numeric "
        "interpretation; no new LLM interpretation was requested.",
        interpretation_mode,
    )
