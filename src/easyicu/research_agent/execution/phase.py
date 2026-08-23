"""Execute phase of the research-agent pipeline.

Implements the probe → per-step analysis loop with optional replanning
and final figure visual-QA. Extracted from
:class:`ResearchAgentPipeline._run_execute_phase` (which is now a thin
delegate) so:

* the 1500-line execute loop reads as its own module;
* the planning / writing phases in :mod:`pipeline` don't have to scroll
  past it;
* a future graph-style runner (LangGraph or similar) has a single
  free-function entry point to wrap, rather than a method buried in a
  god-object.

This module is the canonical owner of execute-phase orchestration.  The
function is intentionally a free function, not a class. All state
that the execute phase mutates (``runtime_state``, ``per_step_records``,
``probe_summary``, ``findings``, ``plan``) is local to one call; nothing
needs to survive across calls. The pipeline instance is passed in only
as a *read-only collaborator* — execute-phase reads several ``_enable_*``
flags and calls ``pipeline._build_runner(...)``, but never mutates
pipeline state. The audit on 2026-05-15 confirmed zero ``self.* = ...``
writes inside the original method body.
"""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import importlib
import functools
import inspect
import json
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import threading
import traceback
from contextvars import copy_context
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
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
    Set,
    Tuple,
)

from ..agents.core import (
    AnalyzerAgent,
    ClinicalSemanticsAgent,
    CoderAgent,
    CriticAgent,
    DataExtractionAgent,
    ReplannerAgent,
    RuntimeSupervisor,
    StatisticalAnalysisAgent,
    VisualizationAgent,
)
from ..audits.validators import (
    ClinicalConstraintValidator,
    ConceptUsageAuditor,
    CrossStepCohortLockValidator,
    CrossStepRegisteredOutputValidator,
    CrossStepReconciliationTraceValidator,
    CrossStepSourceStatusValidator,
    FigureContractQualityValidator,
    FigureSourceDataValidator,
    LLMConceptAuditor,
    PrimaryModelContractValidator,
    StatisticalGuard,
    StatisticalValidator,
    StepSummaryFractionValidator,
)
from ..audits.patterns import AnalysisPatternAuditor
from ..audits.step_summary_integrity import StepSummaryIntegrityValidator
from ..repairs.source import (
    _deterministic_runner_repair,
    _deterministic_summary_repair,
    deterministic_contract_repair,
)
from ..repairs.attempt_record import record_deterministic_runner_repair_attempt
from .code_hygiene import reorder_forward_references
from .article_audit import RunArticleAuditResult, collect_run_article_audits
from ..repairs.coordination import (
    authorized_deterministic_concept_repair,
    resume_deterministic_repair_candidate,
)
from .concept_audit_cache import LLMConceptAuditCache
from .development_sample import (
    materialize_development_execution_sample,
    record_development_sample_authority,
)
from .envelope_sealing import (
    SealedStepResultEnvelopeSnapshot,
    compile_sealed_step_result_shadow,
)
from ..authority.result_envelope_sidecar import (
    publish_terminal_step_result_envelope_sidecar,
)
from .failure_classification import classify_runtime_failure
from .cohort_routing import (
    bind_step_execution_cohort as _bind_step_execution_cohort,
    bound_step_execution_cohort_path as _bound_step_execution_cohort_path,
)
from . import replan_review
from .concept_audit import (
    ConceptAuditAuthority,
    ConceptAuditCoordinator,
    ConceptAuditRuntime,
    ConceptQuarantineState,
    verified_capsule_concept_audit_replay as _verified_capsule_concept_audit_replay,
)
from .concept_repair import (
    MAX_DETERMINISTIC_CONCEPT_REPAIRS,
    ConceptRepairRequest,
    ConceptRepairServices,
    run_concept_repair_loop,
)
from .candidate_loop import (
    _CandidateLoopAction,
    _CandidateLoopAttempt,
    _CandidateLoopHost,
    _CandidateLoopState,
    _candidate_concept_audit_transition,
    _candidate_contract_repair_transition,
    _candidate_contract_setup_transition,
    _candidate_execute_transition,
    _candidate_failure_transition,
    _candidate_success_prepare_transition,
    _candidate_summary_transition,
    _candidate_visual_transition,
    _run_candidate_loop,
)
from .concept_reaudit import (
    deterministic_concept_reaudit_authority,
    deterministic_concept_reaudit_pending_errors,
)
from ..gates.concept import (
    DETERMINISTIC_CODE_GATE_VALIDATORS as _DETERMINISTIC_CODE_GATE_VALIDATORS,
    deterministic_code_gate_findings as _deterministic_code_gate_findings,
    deterministic_gate_stamp as _deterministic_gate_stamp,
    finding_detail_without_source_positions as _finding_detail_without_source_positions,
    finding_occurrence_identity as _finding_occurrence_identity,
    quarantined_deterministic_errors_resolved_by_current_gate as _quarantined_deterministic_errors_resolved_by_current_gate,
    quarantined_errors_superseded_by_current_policy as _quarantined_errors_superseded_by_current_policy,
)
from ..gates.plausibility_receipt import plausibility_audit_receipt_findings
from .owner_declaration import (
    execution_declaration_refusal,
    owner_declaration_plan_findings,
    owner_declaration_replan_directive,
)
from ..gates.plan_declared_inputs import declared_raw_input_plan_findings
from ..gates.product_promise import (
    product_promise_plan_findings,
    product_promise_replan_directive,
)
from ..authority.coder_authority import HostCoderAuthority
from ..authority.plausibility import (
    FlagOnlyPlausibilityScope,
    StepPlausibilityAuthority,
    compile_resumed_flag_only_plausibility_scope,
    compile_step_plausibility_authority,
    restore_revalidated_resolved_inputs_sha256,
)
from ..cohort.repair import extract_cohort_definition_from_prose
from ..cohort.schema import (
    CohortDefinition,
    materialized_cohort_concept_id_scope,
    materialize_locked_analysis_cohort,
    write_locked_cohort_definition,
)
from ..authority.execution_input import ExecutionInputAuthorityState
from ..authority.execution_identity import (
    execution_identity_for_pipeline as _execution_identity,
)
from ..intake.materialized_metadata import (
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
    materialized_provenance_path,
)
from ..intake.materialized_trajectory import (
    MaterializedTrajectoryError,
    StagedTrajectoryBinding,
)
from ..resources.coder import (
    attach_step_coder_input_authority,
    bind_materialized_coder_authority,
    bind_primary_cohort_role,
)
from ..research_context.typed import resolved_raw_input_contracts_for_step
from ..contracts.runtime import ValidationFinding, _ExecutePhaseResult, _PlanPhaseResult
from ..gates.plausibility_obligation import (
    flag_only_plausibility_obligation_findings as _flag_only_plausibility_obligation_findings,
)
from .runners.plausibility_receipt import (
    host_plausibility_receipt_injected as _host_plausibility_receipt_injected,
)
from .runners.deterministic_descriptive import absolute_risk_context_code
from .runners.deterministic_missingness import (
    missingness_audit_input_scope_supported,
    missingness_measurement_audit_code,
)
from .runners.deterministic_robustness import (
    robustness_replay_spec_has_kind_mismatch,
    robustness_replay_spec_is_emittable,
    robustness_sensitivity_preflight_code,
)
from ..contracts.declared_product import (
    RUNTIME_BINDABLE_TYPED_INPUT_KINDS,
    RUNTIME_TYPED_INPUT_EVIDENCE_KINDS,
    authorize_declared_figure_product_slots,
    typed_product_binding_contract,
    typed_product_schema_receipt,
    typed_product as _canonical_typed_product,
)
from ..contracts.primary_cohort import (
    primary_analysis_cohort_plan_findings,
    primary_analysis_cohort_producer_uses_universe,
)
from ..authority.evidence_store import (
    EvidenceAuthorityIntegrityError,
    sha256_of_bytes,
    sha256_of_file,
)
from ..authority.registration import (
    StepEvidenceCommit,
    filter_success_alias_bindings as _filter_success_alias_bindings,
    step_owned_artifact_evidence_id,
)
from ..authority.parent_artifact import _resolve_upstream_manifest_step
from ..authority.plan_authority import (
    NormalizedPlanCandidate as NormalizedPlanCandidate,
    _preserve_completed_step_snapshots_after_replan,
    _preserve_locked_robustness_specs_after_replan,
    normalize_replan_candidate,
)
from ..authority.plan_input_closure import (
    close_measurement_companion_inputs,
    plan_manifest_fields,
    register_measurement_companion_input_closure,
)
from ..authority.plan_scope import (
    _normalise_scientific_text,
    _plan_scientific_scope_signature,
    _plan_signature,
    _serializable_plan_scientific_scope_signature,
    _step_scientific_signature,
)
from ..authority.typed_binding import (
    TypedBindingResolver,
    _EvidenceLineageResolutionError,
    _assignment_model_authority_context_block,
    _declared_typed_artifact_paths,
    _declared_typed_product_paths,
    _evidence_kind_matches_typed_product,
    _evidence_record_field,
    _current_verified_evidence_record,
    _lineage_failure_product_fields,
    _normalise_typed_product_name,
    _registered_source_name,
    _resolve_typed_artifact_evidence,
    _resolve_typed_input_evidence,
    _resolved_typed_input_binding,
    _resume_typed_input_bindings,
    _resume_typed_input_bindings_fingerprint,
    _step_summary_statistic_values,
    _typed_artifact_name,
    _typed_input_product,
    _typed_parent_schema_context_block,
    _write_host_input_binding_receipts,
    _write_resolved_inputs_manifest,
    host_authorized_ambient_trajectory_entry,
    rank_scale_columns_entry,
    study_endpoint_declaration_entry,
    host_owns_input_binding_receipts,
)
from ..gates.contract import (  # execute-layer collaborators use the canonical gate API
    _AGENT_OWNED_ROBUSTNESS_RESULT_METHODS,
    _AGENT_OWNED_ROBUSTNESS_RESULT_PRODUCTS,
    _AUXILIARY_OUTPUT_KINDS,
    _authoritative_primary_robustness_contract,
    _closed_auxiliary_output_products,
    _cohort_definition_sensitivity_contract_findings,
    _declared_sensitivity_csv_paths,
    _is_cohort_definition_sensitivity_result_step,
    _method_head,
    _nonnegative_integral_value,
    _post_canonicalization_figure_findings,
    _read_locked_robustness_spec_dicts,
    _sensitivity_csv_rows,
    _step_deterministic_contract_findings,
)
from .figure_preparation import (
    _ensure_step_figure_contract,
    _family_has_deterministic_figure_renderer,
    _figure_contract_source_data_canonicalization_candidate,
    _infer_step_figure_panel_role,
    _install_figure_contract_source_data_canonicalization,
    _reader_label_from_stem,
    _step_has_figure_only_output_contract,
    _step_summary_paths,
)
from .final_validation import (
    _FinalDeterministicGateFindings,
    _PRIMARY_DETERMINISTIC_RUNNERS,
    _bind_findings_to_step_attempt,
    _demote_result_figure_shape_for_family_renderer,
    _demote_step_contract_for_primary_runner,
    _evaluate_final_deterministic_gates,
    _is_too_few_panels_figure_finding,
    _primary_runner_core_estimate_present,
)
from .publication_figure import (
    SealedRendererState,
    _deterministic_publication_figure_code,
    _sealed_parent_planner_anchors,
    _sealed_renderer_implementation_digest,
    _sealed_renderer_source_digests,
    _sealed_typed_figure_products,
    sealed_renderer_code_seal_required,
    validate_and_record_sealed_renderer_receipt,
)
from .host_services import ExecutePhaseHost
from .output_files import (
    _clear_output_dir,
    _has_figure_exports,
    bind_primary_output,
    normalize_typed_statistic_sidecars,
)
from ..gates.visual import (
    VisualGateResult,
    VisualRepairAction,
    VisualRepairDecision,
    _demote_cosmetic_visual_findings,
    _is_cosmetic_visual_finding,
    _visual_repair_request_log,
    collect_visual_gate_result,
    decide_visual_repair,
)
from ..gates.semantics import (
    blocking_validator_findings as _blocking_validator_findings,
)
from ..providers.mocks import MockLLMClient
from ..providers.prompt_budget import (
    budgeted_coder_clients,
    budgeted_role_client,
    budgeted_vlm_client,
)
from ..planning.cohort_contract import (
    CohortSchemaError,
    coerce_cohort_definition,
    cohort_definition_has_explicit_selection,
    cohort_definition_sha,
)
from ..planning.method_vocabulary import (
    MISSINGNESS_SOURCE_AVAILABILITY_AUDIT,
)
from ..planning.replan_gate import (
    partition_replan_candidate_findings,
    replan_candidate_contract_findings,
    replan_candidate_rejection_finding,
)
from ..contracts.ordered_stratified import ordered_stratified_numeric_findings
from ..repairs.reasons import (
    RepairPromptAuthority,
    RepairReason,
    repair_prompt_binding_sha256,
    repair_reason_for_finding,
    typed_repair_ticket,
)
from ..plan_utils import (
    _augment_report_typed_product_inputs,
    _cap_plan_preserving_figure_steps,
    _clustering_contract_applies,
    _cohort_definition_contract_findings,
    _cohort_definition_is_empty,
    _cohort_definition_prose,
    endpoint_contract_findings,
    _normalised_expected_output_names,
    _normalised_structured_output_names,
    _output_declares_figure,
    _parent_step_id_for_figure_step,
    _plan_expects_analysis_cohort,
    _preserve_figure_steps_after_replan,
    _step_contract_repair_guidance,
    _step_expects_figure,
    _typed_plan_dag_findings,
)
from ..orchestration.resume import (
    QuarantinedConceptDraft,
    ResumeController,
    clear_quarantined_concept_draft,
    store_quarantined_concept_draft,
    upsert_step_record,
)
from ..orchestration.step_selector import (
    resolve_resume_from_step_selector as _resolve_resume_from_step_selector,
    resolve_stop_after_step_selector as _resolve_stop_after_step_selector,
)
from ..schema import AnalysisPlan, AnalysisStep, EvidenceRef, ResearchContext
from ..contracts.robustness_execution import (
    ROBUSTNESS_COHORT_MEMBERSHIP_ALIASES,
    ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE,
    _executed_robustness_result_issues,
)
from ..robustness.panel import (
    RobustnessSpec,
    robustness_specs_for_execution,
    robustness_specs_sha,
)
from ..robustness.runtime_panel import finalize_run_robustness_panel
from ..trajectory.bundle import trajectory_bundle_findings
from ..trajectory.plan_contract import (
    augment_trajectory_plan_products,
    trajectory_plan_contract_applies,
    trajectory_plan_dag_findings,
)
from .runners.selection import StandardExecutorCandidate, select_standard_executor
from .runners.selection_report import standard_executor_candidate_report
from .standard_executor_diagnostics import standard_executor_failure_finding
from ..repair_registry import (
    InvariantStatus,
    RepairClass,
    RepairLedger,
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
from .provider_budget_runtime import (
    monotonic_step_llm_repair_history as _monotonic_step_llm_repair_history,
    step_snapshot_requires_provider_receipt as _step_snapshot_requires_provider_receipt,
)
from .repair_reservation import StepRepairReservation
from .step_attempt_bootstrap import (
    RAW_UNIVERSE_EXECUTION_ROLE,
    prepare_step_attempt_bootstrap,
)
from .step_authority_resume import (
    StepAuthorityResumeRequest,
    prepare_step_authority_resume,
)
from .step_candidate_recovery import (
    StepCandidateRecovery,
    StepCandidateRecoveryRequest,
)
from .resume_revalidation import (
    ResumeDeterministicRevalidationResult as _ResumeDeterministicRevalidationResult,
    ResumeRevalidationServices,
    _discard_stale_resolved_input_receipts,
    _materialize_verified_step_output_view,
    _project_verified_replay_output_paths,
    _resume_success_dependencies,
    _trusted_resume_success_records,
    _verified_explicit_step_authority,
    _verify_resume_step_script_lineage,
    revalidate_resume_successes,
)
from ..authority.run_input import (
    RUN_INPUT_CAPSULE_EVIDENCE_ID,
    RUN_INPUT_CAPSULE_FILENAME,
    RunInputIdentityError,
    _HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD,
    _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND,
    _HOST_COHORT_MATERIALIZER_GENERATION_MODE,
    _HOST_COHORT_FLOW_AUTHORITY_FIELD,
    _HOST_PROBE_AUTHORITIES,
    _HOST_PROBE_AUTHORITY_KIND,
    _declares_host_cohort_products as _declares_host_cohort_only_product,
    _host_cohort_materializer_authority_error,
    _host_probe_authority_error,
    build_concept_audit_environment_identity as _concept_audit_environment,
    canonical_sha256,
    engine_code_sha256,
    verify_legacy_trajectory_capsule_receipt,
    validator_code_sha256,
)
from .run_coordination import (
    RunCoordinator,
    RunExecutionState,
    RunTransition,
    _successful_run_transition_requests_replan,
    _successful_step_requests_replan,
)
from .cohort_adoption import (
    adopt_existing_host_cohort_materialization,
    commit_staged_cohort_plan,
    record_planned_host_cohort_checkpoint,
    stage_candidate_cohort_plan,
)
from ..authority.runtime_artifacts import (
    current_step_records,
    current_successful_step_records,
    verified_run_evidence_path,
    write_run_checkpoint,
)
from ..scalar_utils import _expected_numeric_annotations_for_step
from ..reporting.side_findings import SideFinding
from ..skills import ClinicalSkill
from ..authority.step_capsule import (
    StepAuthorityCapsuleRef,
    StepAuthorityCapsuleError,
    load_verified_step_authority_capsule,
)
from ..authority.step_runtime import (
    StepAuthorityRuntimeError,
    adopt_candidate_for_control_plane_revalidation,
    adopt_frozen_scoped_coder_context,
    capsule_matches_coordinates,
    current_execution_runtime_sha256,
    dependency_blocked_candidate_metadata,
    coordinates_from_verified_capsule,
    execution_context_sha256,
    initial_generation_code_ref,
    load_checkpoint_selected_step_capsule,
    materialize_sealed_run_result,
    persist_candidate_code,
    prepare_step_authority_coordinates,
    repair_code_ref,
    seal_concept_audit_capsule,
    seal_deterministic_candidate,
    seal_execution_capsule,
    seal_initial_generation_candidate,
    seal_legacy_candidate,
    seal_repair_candidate_from_receipt,
    select_explicit_step_capsule_for_targeted_resume,
)
from ..authority.step_attempt import (
    RESUMABLE_ATTEMPT_COORDINATE_FIELDS,
    CheckpointAuthority,
    StepAttemptState,
    StepAuthorityOperations,
)
from .step_execution import LockedStepExecutionRequest, StepExecutor
from .step_worker_state import StepWorkerProgress
from ..repairs.summary import salvage_step_summary
from ..viability import (
    CohortViability,
    assess_cohort_viability,
    step_requires_model_performance,
    step_summary_block_signal,
)
from ..gates.visual_qa import VLMVisualQAAdapter, VisualQAAuditor

logger = logging.getLogger(__name__)


from .phase_support import (  # noqa: F401 — owner module
    _CAPSULE_TRANSIENT_STEP_STATUSES,
    _COHORT_DEF_SENSITIVITY_METHODS,
    _COHORT_DEF_SENSITIVITY_OUTPUT_TOKENS,
    _COHORT_TRANSLATION_PROVIDER_CATEGORY,
    _COMPACT_MISSINGNESS_METHODS,
    _COMPACT_MISSINGNESS_SUPPORTED_OUTPUTS,
    _EFFECT_ASSOCIATION_METHOD_TOKENS,
    _EFFECT_OUTPUT_FRAGMENTS,
    _FIGURE_CONTRACT_SOURCE_DATA_SCHEMA_REPAIR_ID,
    _HOST_COHORT_TRANSLATION_BUDGET_STEP_ID,
    _InertPythonNodeStripper,
    _LOCKED_MEASUREMENT_DATA_QUALITY_ISSUES,
    _MAX_DIRECTED_MODEL_REPLANS,
    _ORDINAL_EXPLICIT_METHODS,
    _ORDINAL_OUTPUT_PRODUCTS,
    _ORDINAL_PRIMARY_METHODS,
    _PRIMARY_COHORT_FLOW_METHODS,
    _PRIMARY_COHORT_FLOW_OUTPUTS,
    _RAW_UNIVERSE_EXECUTION_ROLE,
    _RICH_EXPOSURE_AUDIT_OUTPUT_TOKENS,
    _ROBUSTNESS_SENSITIVITY_METHODS,
    _SEALED_AUTHORITY_SUMMARY_MARKERS,
    _STANDARD_EXECUTOR_INTERNAL_PENDING_ARTIFACTS,
    _assert_execution_plan_locks,
    _absolute_risk_context_runner_owns_step,
    _actionable_validator_messages,
    _append_terminal_step_record,
    _coder_authority_with_locked_robustness_specs,
    _cohort_definition_overlap_runner_owns_step,
    _cohort_definition_sensitivity_runner_owns_step,
    _cohort_translation_budget_owner_step_id,
    _collect_and_persist_run_article_audits,
    _contract_repair_log,
    _declares_effect_output,
    _detached_figure_repair_binding,
    _extract_cohort_definition_with_provider_budget,
    _failed_contract_code_can_be_reused_before_coder,
    _fresh_plausibility_receipt_findings,
    _is_cohort_definition_sensitivity_step,
    _is_standard_executor_internal_artifact,
    _is_terminal_publication_figure_repair_step,
    _load_step_summary_from_outputs,
    _locked_measurement_data_quality_issues,
    _max_finding_severity,
    _merge_monotonic_concept_constraints,
    _method_has_ordinal_primary_token,
    _method_is_effect_or_association,
    _non_llm_interpretation_for_generation,
    _ordinal_dose_response_step_matches,
    _persist_run_article_audit_result,
    _persisted_monotonic_concept_constraints,
    _planner_locked_cohort_prompt_payload,
    _primary_cohort_flow_runner_owns_step,
    _publication_bundle_has_primary_result_roles,
    _python_repair_is_materially_changed,
    _python_semantic_sha256,
    _remove_standard_executor_pending_artifacts,
    _repair_prompt_binding_sha256,
    _robustness_sensitivity_runner_owns_step,
    _simple_missingness_audit_runner_owns_step,
    _step_monotonic_concept_constraint_log,
    _step_monotonic_concept_constraint_payload,
    _step_monotonic_concept_constraint_ticket,
    _step_repair_with_capsule,
    _step_reserve_compatibility_repair,
    _step_resume_critic_repair_code,
    _step_authorize_deterministic_concept_reaudit,
    _step_authorized_deterministic_concept_repair,
    _step_deterministic_fallback_code,
    _step_absolute_risk_context_preflight_supported,
    _step_deterministic_absolute_risk_context_code,
    _step_deterministic_missingness_audit_code,
    _step_deterministic_robustness_sensitivity_code,
    _step_missingness_audit_preflight_supported,
    _step_publication_figure_preflight_supported,
    _step_register_run_artifacts,
    _step_resolve_initial_code,
    _step_apply_authority_resume,
    _step_robustness_sensitivity_preflight_supported,
    _step_authorize_automatic_repair,
    _step_automatic_repair_authorized,
    _step_enforce_cohort_contract_on_executing_plan,
    _step_failed_dependency_record,
    _step_flush_partial_manifest,
    _step_no_analysis_step_has_run,
    _step_propagate_findings_to_evidence,
    _step_record_repair,
    _step_register_plan_revision,
    _step_maybe_replan,
    _step_try_materialize_cohort_from_prose,
    _step_resolve_cohort_definition,
    _step_universe_columns,
    _step_validator_messages,
    _step_register_current_step_numeric_claims,
    _step_resume_deterministic_repair_code,
    _step_use_quarantined_draft,
    _step_use_resumed_code,
    _step_quarantine_error_payloads,
    _step_remember_concept_constraints,
    _step_requires_publication_figure_exports,
    _step_status_from_contract_findings,
    _submit_in_current_context,
    _terminal_publication_repair_replan_skip_detail,
    _trajectory_clustering_step_matches,
    _unowned_sealed_authority_markers,
    _untrusted_runtime_repair_allowed,
    _upsert_current_capsule_checkpoint,
    _verified_run_input_capsule_digest,
    build_self_block_replan_directive,
    scope_findings_to_records,
)


def _selectively_revalidate_resume_successes(
    *,
    resume_state: Dict[str, Any],
    plan: AnalysisPlan,
    context: ResearchContext,
    evidence: Any,
    run_dir: Path,
    cohort_path: Path,
    universe_path: Path,
    resume_from_step_id: Optional[str],
    development_sample: Optional[Any] = None,
) -> _ResumeDeterministicRevalidationResult:
    """Replay changed deterministic gates against sealed evidence only."""

    return revalidate_resume_successes(
        resume_state=resume_state,
        plan=plan,
        context=context,
        evidence=evidence,
        run_dir=run_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
        resume_from_step_id=resume_from_step_id,
        development_sample=development_sample,
        services=ResumeRevalidationServices(
            deterministic_gate_stamp=_deterministic_gate_stamp,
            evaluate_final_deterministic_gates=(
                lambda **kwargs: _evaluate_final_deterministic_gates(**kwargs)
            ),
            deterministic_code_gate_findings=(
                lambda **kwargs: _deterministic_code_gate_findings(**kwargs)
            ),
            actionable_validator_messages=_actionable_validator_messages,
            write_run_checkpoint=write_run_checkpoint,
        ),
    )

def _execution_input_authority_integrity_finding(
    *,
    step_id: str,
    universe_path: Path,
    cohort_path: Path,
    expected_universe_sha256: Optional[str],
    expected_analysis_cohort_sha256: Optional[str],
) -> Optional[ValidationFinding]:
    """Return a blocking finding when execution mutated host-owned inputs."""
    try:
        current_universe_sha256 = sha256_of_file(universe_path)
        current_cohort_sha256 = sha256_of_file(cohort_path)
    except Exception as exc:
        current_universe_sha256 = None
        current_cohort_sha256 = None
        authority_error = f"{type(exc).__name__}: {exc}"[:300]
    else:
        authority_error = None
    if (
        current_universe_sha256 == expected_universe_sha256
        and current_cohort_sha256 == expected_analysis_cohort_sha256
    ):
        return None
    return ValidationFinding(
        validator="execution_input_authority_integrity",
        severity="error",
        message=(
            "The raw universe or authoritative analysis cohort changed while "
            f"step {step_id} executed; all outputs from this attempt were rejected."
        ),
        detail={
            "step_id": step_id,
            "expected_universe_sha256": expected_universe_sha256,
            "observed_universe_sha256": current_universe_sha256,
            "expected_analysis_cohort_sha256": expected_analysis_cohort_sha256,
            "observed_analysis_cohort_sha256": current_cohort_sha256,
            "error": authority_error,
        },
    )

def _should_attempt_detached_figure_binding(
    *, out_dir: Path, sealed_renderer_authorized_code_sha256: Optional[str]
) -> bool:
    """Detached rescue lineage must never rewrite an authorized sealed summary."""

    return sealed_renderer_authorized_code_sha256 is None and _has_figure_exports(
        out_dir
    )

def _planner_materialized_cohort_prompt_payload(
    *,
    plan: AnalysisPlan,
    universe_path: Path,
    analysis_cohort_path: Path,
) -> str:
    """Serialize the verified cohort receipt for the Coder authority prompt."""

    receipt = _planner_materialized_cohort_execution_receipt(
        plan=plan,
        universe_path=universe_path,
        analysis_cohort_path=analysis_cohort_path,
    )
    return json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

def _planner_materialized_cohort_execution_receipt(
    *,
    plan: AnalysisPlan,
    universe_path: Path,
    analysis_cohort_path: Path,
) -> Dict[str, Any]:
    """Return a verified execution receipt for the Planner-owned predicates.

    The host already applies the locked cohort definition before execution.
    This projection gives both the Coder prompt and its read-only runtime
    manifest the same physical column bindings and row-accounting checks
    without exposing row identities or choosing any new scientific rule.
    """
    analysis_cohort_path = Path(analysis_cohort_path)
    verified = load_verified_materialized_cohort_authority(analysis_cohort_path)
    if verified is not None:
        raw_provenance = verified.authority.to_dict()["semantic_provenance"]
        if not isinstance(raw_provenance, Mapping):
            raise MaterializedMetadataError(
                "typed analysis cohort provenance is not an object"
            )
        provenance = dict(raw_provenance)
        identity_column: Optional[str] = verified.authority.identity_column
        row_identity_sha256: Optional[str] = verified.authority.row_identity_sha256
        authority_sha256: Optional[str] = verified.reference.sha256
        authoritative_rows = verified.authority.cohort_rows
    else:
        provenance_path = materialized_provenance_path(analysis_cohort_path)
        if provenance_path.is_symlink() or not provenance_path.is_file():
            raise MaterializedMetadataError(
                "analysis cohort provenance is unavailable for Coder authority"
            )
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MaterializedMetadataError(
                "analysis cohort provenance is unreadable for Coder authority"
            ) from exc
        if not isinstance(provenance, dict):
            raise MaterializedMetadataError(
                "analysis cohort provenance is not an object"
            )
        if provenance.get("cohort_parquet_sha256") != sha256_of_file(
            analysis_cohort_path
        ):
            raise MaterializedMetadataError(
                "legacy analysis cohort provenance digest changed"
            )
        identity_column = None
        row_identity_sha256 = None
        authority_sha256 = None
        authoritative_rows = provenance.get("n_analysis_cohort")

    # Both sides of this comparison are serializations of one locked
    # definition, so both must come from the same owner.  ``to_dict`` omits a
    # default ``selection_mode`` to keep legacy authority digests stable while
    # pydantic's ``model_dump`` always emits it, so comparing those two
    # spellings reported every predicate-filtered cohort as a mismatch.
    # Re-hashing through the canonical owner proves ``cohort_sha256`` indexes
    # the stored definition rather than merely asserting it.
    with materialized_cohort_concept_id_scope(analysis_cohort_path, verified):
        definition = coerce_cohort_definition(getattr(plan, "cohort", None))
        if definition is None:
            raise MaterializedMetadataError(
                "active plan carries no cohort definition for the execution receipt"
            )
        planner_cohort_sha256 = cohort_definition_sha(definition)
        recorded_cohort = provenance.get("cohort_definition")
        if not isinstance(recorded_cohort, Mapping):
            raise MaterializedMetadataError(
                "analysis cohort execution receipt has no recorded cohort definition"
            )
        try:
            recorded_cohort_sha256 = cohort_definition_sha(
                CohortDefinition.from_dict(dict(recorded_cohort))
            )
        except (CohortSchemaError, KeyError, TypeError, ValueError) as exc:
            raise MaterializedMetadataError(
                "analysis cohort execution receipt cohort definition is unreadable"
            ) from exc
    flow = provenance.get("cohort_flow")
    if (
        recorded_cohort_sha256 != planner_cohort_sha256
        or str(provenance.get("cohort_sha256") or "") != planner_cohort_sha256
        or not isinstance(flow, list)
        or not flow
        or any(not isinstance(row, dict) for row in flow)
    ):
        raise MaterializedMetadataError(
            "analysis cohort execution receipt does not match the active plan"
        )
    try:
        n_universe = int(provenance["n_universe"])
        n_analysis_cohort = int(provenance["n_analysis_cohort"])
        authoritative_rows_int = int(authoritative_rows)
        first_before = int(flow[0]["n_before"])
        first_remaining = int(flow[0]["n_remaining"])
        final_remaining = int(flow[-1]["n_remaining"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MaterializedMetadataError(
            "analysis cohort execution receipt has invalid row accounting"
        ) from exc
    if (
        flow[0].get("predicate_kind") != "universe"
        or first_before != n_universe
        or first_remaining != n_universe
        or final_remaining != n_analysis_cohort
        or authoritative_rows_int != n_analysis_cohort
    ):
        raise MaterializedMetadataError(
            "analysis cohort execution receipt row accounting changed"
        )

    return {
        "schema_version": "easyicu.primary_cohort_execution_prompt/1",
        "cohort_definition_sha256": planner_cohort_sha256,
        "raw_universe": {
            "rows": n_universe,
            "sha256": sha256_of_file(universe_path),
        },
        "authoritative_analysis_cohort": {
            "rows": n_analysis_cohort,
            "sha256": sha256_of_file(analysis_cohort_path),
            "identity_column": identity_column,
            "row_identity_sha256": row_identity_sha256,
            "authority_sha256": authority_sha256,
        },
        "ordered_predicate_flow": flow,
    }

def _repair_publication_figure_in_staging(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    authorizer: Callable[[str], bool],
    renderer: Callable[..., Optional[str]],
    step_text: str = "",
) -> Optional[str]:
    """Render into staging and replace agent exports only after success.

    A routing false-positive or strict renderer guard returning ``None`` must
    leave the agent-produced figure, source data, and contract untouched.  Once
    a staged renderer emits a real figure export, move the old directory into a
    same-filesystem backup, install the staged bundle, and roll back on any move
    failure.
    """

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".publication-figure-repair-", dir=out_dir.parent
    ) as staging_name:
        staging_dir = Path(staging_name)
        try:
            repair_id = renderer(
                run_dir=run_dir,
                current_step_id=current_step_id,
                out_dir=staging_dir,
                step_text=step_text,
            )
        except Exception as exc:
            logger.warning(
                "Staged publication-figure repair failed for %s: %s",
                current_step_id,
                exc,
            )
            return None
        if (
            repair_id is None
            or is_sealed_renderer_repair(repair_id)
            or not _has_figure_exports(staging_dir)
        ):
            return None
        # Rendering into an isolated temporary directory is non-authoritative.
        # Ask the central repair policy before installing any generated bundle
        # into the live step directory.
        if not authorizer(repair_id):
            return None

        backup_dir = Path(
            tempfile.mkdtemp(prefix=".publication-figure-backup-", dir=out_dir.parent)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            for child in list(out_dir.iterdir()):
                shutil.move(str(child), str(backup_dir / child.name))
            for child in list(staging_dir.iterdir()):
                shutil.move(str(child), str(out_dir / child.name))
        except Exception:
            _clear_output_dir(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for child in list(backup_dir.iterdir()):
                shutil.move(str(child), str(out_dir / child.name))
            raise
        finally:
            shutil.rmtree(backup_dir, ignore_errors=True)

        # Renderers may store absolute output paths in JSON summaries/contracts.
        # They were valid in staging; rewrite only that exact directory prefix
        # after the atomic-style move so provenance points to the installed bundle.
        for json_path in out_dir.rglob("*.json"):
            try:
                content = json_path.read_text(encoding="utf-8")
                rewritten = content.replace(str(staging_dir), str(out_dir))
                if rewritten != content:
                    json_path.write_text(rewritten, encoding="utf-8")
            except Exception:
                continue
        return repair_id

@dataclass
class _ExecutePhasePreparation:
    services: Any
    context: Any
    coder_base_context: Any
    evidence: Any
    step_evidence_commit: Any
    findings: Any
    plan: Any
    plan_path: Any
    resume_controller: Any
    requested_resume_from_step_id: Any
    requested_stop_after_step_id: Any
    reuse_selected_step_code_opt_in: Any
    _replan_state: Any
    role_resolver: Any
    llm_signature: Any
    llm_concept_audit_client: Any
    llm_concept_auditor_identity_sha256: Any
    concept_audit_environment_sha256: Any
    prompt_version: Any
    prompt_files: Any


def _prepare_execute_phase_authority(
    pipeline: ExecutePhaseHost,
    *,
    plan_result: _PlanPhaseResult,
    run_dir: Path,
    resume_from_step_id: Optional[str],
    stop_after_step_id: Optional[str],
) -> _ExecutePhasePreparation:
    """Execute probe + per-step analysis loop, with optional replanning."""
    services = pipeline._execute_phase_services()
    context = plan_result.context
    # Planner memory, retrieval narration, hypothesis notes, and article
    # blueprints shaped the typed plan but are not Coder authority.  Preserve
    # the Planner's selected variable projection while restoring the original
    # user/run notes. Host execution attachments travel through a separate
    # typed side channel so user prose can never be elevated to host authority.
    coder_base_context = plan_result.agent_context.model_copy(
        update={"notes": context.notes}
    )
    evidence = plan_result.evidence
    step_evidence_commit = StepEvidenceCommit(evidence)
    findings = plan_result.findings
    plan = plan_result.plan
    plan_path = plan_result.plan_path
    plan, companion_input_findings = close_measurement_companion_inputs(
        plan=plan,
        context=context,
    )
    if companion_input_findings:
        findings.extend(companion_input_findings)
        plan, runtime_compile_findings = (
            pipeline._scientific_runtime_authorities.bind_plan(plan)
        )
        findings.extend(runtime_compile_findings)
        if plan != plan_result.plan:
            plan_path = register_measurement_companion_input_closure(
                run_dir=run_dir,
                evidence=evidence,
                plan=plan,
                prompt_pack_version=plan_result.prompt_version,
            ).evidence_path
            plan_result.plan_path = plan_path
    stop_after_step_id = _resolve_stop_after_step_selector(plan, stop_after_step_id)
    resume_from_step_id = _resolve_resume_from_step_selector(
        plan,
        resume_from_step_id,
    )
    resume_controller = ResumeController(
        plan=plan,
        run_dir=run_dir,
        resume_state=plan_result.resume_state,
        resume_from_step_id=resume_from_step_id,
        stop_after_step_id=stop_after_step_id,
    )
    requested_resume_from_step_id = resume_controller.resume_from_step_id
    requested_stop_after_step_id = resume_controller.stop_after_step_id
    reuse_selected_step_code_opt_in = (
        requested_resume_from_step_id is not None
        and os.environ.get("EASYICU_RESUME_REUSE_STEP_CODE") == "1"
    )
    resumed_cohort_translation_budget: Optional[Dict[str, Any]] = None
    resumed_cohort_translation_budget_owner: Optional[str] = None
    if isinstance(plan_result.resume_state, Mapping):
        raw_cohort_translation_budget = plan_result.resume_state.get(
            "cohort_translation_provider_budget"
        )
        if isinstance(raw_cohort_translation_budget, Mapping):
            candidate_owner = str(
                raw_cohort_translation_budget.get("budget_owner_step_id") or ""
            ).strip()
            if candidate_owner:
                resumed_cohort_translation_budget = dict(raw_cohort_translation_budget)
                resumed_cohort_translation_budget_owner = candidate_owner
    # Replan convergence bookkeeping (see _maybe_replan). ``noop_streak``
    # counts consecutive substantively-identical revisions; ``total`` counts
    # substantive revisions; ``disabled`` latches once a guard trips.
    _replan_state = {
        "noop_streak": 0,
        "total": 0,
        "disabled": False,
        # Latches True when the substantive-revision count reaches
        # ``max_replans``; drives the fail-closed diagnostic_only demotion.
        "budget_exhausted": False,
        "cohort_contract_emitted": False,
        "cohort_materialized": False,
        # The first cohort-prose translation latches one provider-budget owner
        # for the run.  Later replans cannot buy a fresh allowance by renaming
        # or reshaping the cohort-definition step.
        "cohort_translation_budget_owner_step_id": (
            resumed_cohort_translation_budget_owner
        ),
        "cohort_translation_provider_budget": resumed_cohort_translation_budget,
        "cohort_translation_provider_budget_error_emitted": False,
        # Directed replans fired when a model/estimation step self-blocks on a
        # task-viable cohort (see _maybe_directed_model_replan). Bounded so a
        # run that keeps self-blocking falls back to an honest diagnostic_only
        # rather than looping the replanner indefinitely.
        "directed_model_replans": 0,
    }
    role_resolver = plan_result.role_resolver
    llm_signature = plan_result.llm_signature
    llm_concept_audit_client = pipeline._llm_concept_auditor_client or (
        budgeted_role_client(
            role_resolver,
            "analyzer",
            "concept_audit",
            limit_tokens=pipeline._max_prompt_tokens_per_call,
        )
    )
    llm_concept_auditor_signature = (
        pipeline._llm_signature(llm_concept_audit_client)
        if llm_concept_audit_client is not None
        else "llm_concept_auditor_unavailable"
    )
    llm_concept_auditor_identity_sha256 = canonical_sha256(
        llm_concept_auditor_signature
    )
    concept_audit_environment_sha256 = canonical_sha256(
        _concept_audit_environment(llm_signature=llm_concept_auditor_signature)
    )
    prompt_version = plan_result.prompt_version
    prompt_files = plan_result.prompt_files
    _assert_execution_plan_locks(run_dir, plan, plan_result.cohort_concept_ids)
    return _ExecutePhasePreparation(
        services=services,
        context=context,
        coder_base_context=coder_base_context,
        evidence=evidence,
        step_evidence_commit=step_evidence_commit,
        findings=findings,
        plan=plan,
        plan_path=plan_path,
        resume_controller=resume_controller,
        requested_resume_from_step_id=requested_resume_from_step_id,
        requested_stop_after_step_id=requested_stop_after_step_id,
        reuse_selected_step_code_opt_in=reuse_selected_step_code_opt_in,
        _replan_state=_replan_state,
        role_resolver=role_resolver,
        llm_signature=llm_signature,
        llm_concept_audit_client=llm_concept_audit_client,
        llm_concept_auditor_identity_sha256=(
            llm_concept_auditor_identity_sha256
        ),
        concept_audit_environment_sha256=concept_audit_environment_sha256,
        prompt_version=prompt_version,
        prompt_files=prompt_files,
    )


def run_execute_phase(
    pipeline: ExecutePhaseHost,
    *,
    plan_result: _PlanPhaseResult,
    cohort_path: Path,
    trajectory_binding: Optional[StagedTrajectoryBinding],
    run_dir: Path,
    run_id: str,
    skill_obj: Optional[ClinicalSkill],
    notes: Optional[str],
    emit_progress: Callable[..., None],
    resume_from_step_id: Optional[str] = None,
    stop_after_step_id: Optional[str] = None,
) -> _ExecutePhaseResult:
    preparation = _prepare_execute_phase_authority(
        pipeline,
        plan_result=plan_result,
        run_dir=run_dir,
        resume_from_step_id=resume_from_step_id,
        stop_after_step_id=stop_after_step_id,
    )
    services = preparation.services
    context = preparation.context
    coder_base_context = preparation.coder_base_context
    evidence = preparation.evidence
    step_evidence_commit = preparation.step_evidence_commit
    findings = preparation.findings
    plan = preparation.plan
    plan_path = preparation.plan_path
    resume_controller = preparation.resume_controller
    requested_resume_from_step_id = preparation.requested_resume_from_step_id
    requested_stop_after_step_id = preparation.requested_stop_after_step_id
    reuse_selected_step_code_opt_in = preparation.reuse_selected_step_code_opt_in
    _replan_state = preparation._replan_state
    role_resolver = preparation.role_resolver
    llm_signature = preparation.llm_signature
    llm_concept_audit_client = preparation.llm_concept_audit_client
    llm_concept_auditor_identity_sha256 = (
        preparation.llm_concept_auditor_identity_sha256
    )
    concept_audit_environment_sha256 = (
        preparation.concept_audit_environment_sha256
    )
    prompt_version = preparation.prompt_version
    prompt_files = preparation.prompt_files

    # Dual-track cohort. If the plan phase materialised the locked cohort
    # definition into a filtered analysis cohort, every downstream consumer
    # (probe, statistical validators, robustness fitter, and the step runner)
    # reads THAT — so the declared inclusion/exclusion is enforced once,
    # consistently, instead of being silently re-implemented (or skipped) by
    # each generated step. The full universe is injected only into runners for
    # typed steps whose cohort/robustness contract explicitly authorizes it.
    universe_path = cohort_path
    run_input_authority_state = ExecutionInputAuthorityState.bind(
        universe_path=universe_path,
        analysis_path=run_dir / "cohort_analysis.parquet",
        trajectory_binding=trajectory_binding,
        run_dir=run_dir,
        legacy_trajectory_verifier=verify_legacy_trajectory_capsule_receipt,
        plan=plan,
        context=context,
    )
    if (
        pipeline._development_sample_size is not None
        and run_input_authority_state.analysis_path.exists()
    ):
        run_input_authority_state.apply_development_sample(
            materialize_development_execution_sample(
                run_dir=run_dir,
                target_rows=pipeline._development_sample_size,
                seed=pipeline._development_sample_seed,
                declared_id_columns=tuple(
                    getattr(context.cohort, "id_columns", ()) or ()
                ),
                trajectory_binding=trajectory_binding,
            )
        )
    cohort_path = run_input_authority_state.selected_path
    run_input_authority_state.require_trajectory_integrity(
        step_id="execute_phase_initialization",
    )
    # The context can only ever show the wide fixed-window representation; the
    # long tier is a verified typed run input this scope can see and it cannot.
    # Derived here, beside the integrity check, because the trajectory contract
    # is consulted from several points below and every one of them needs it.
    long_trajectory_bound = (
        run_input_authority_state.trajectory_authority_sha256 is not None
    )

    if (
        pipeline._development_sample_size is not None
        and run_input_authority_state.development_sample is None
    ):
        findings.append(
            ValidationFinding(
                validator="development_sample_authority",
                severity="error",
                message=(
                    "This run requested a non-paper post-QC development sample. "
                    "The locked analysis cohort is not materialized yet, so "
                    "sampling is deferred until cohort QC completes; regardless "
                    "of whether that occurs, this run cannot become paper authority."
                ),
                detail={
                    "paper_authority": False,
                    "stage": "awaiting_locked_cohort_materialization_and_qc",
                    "target_rows": pipeline._development_sample_size,
                    "seed": pipeline._development_sample_seed,
                },
            )
        )
    if run_input_authority_state.development_sample is not None:
        record_development_sample_authority(
            binding=run_input_authority_state.development_sample,
            evidence=evidence,
            findings=findings,
            emit_progress=emit_progress,
            run_id=run_id,
        )

    # Validator/code drift is resolved before ResumeController decides which
    # prior successes to skip and before any coder, runner, analyzer, or LLM
    # collaborator is constructed.  The replay reads sealed evidence only.
    if plan_result.resume_state is not None:
        resume_revalidation = _selectively_revalidate_resume_successes(
            resume_state=plan_result.resume_state,
            plan=plan,
            context=context,
            evidence=evidence,
            run_dir=run_dir,
            cohort_path=cohort_path,
            universe_path=universe_path,
            resume_from_step_id=requested_resume_from_step_id,
            development_sample=run_input_authority_state.development_sample,
        )
        plan_result.resume_state = resume_revalidation.resume_state
        resume_controller.resume_state = resume_revalidation.resume_state

    coder_transports = budgeted_coder_clients(
        role_resolver,
        limit_tokens=pipeline._max_prompt_tokens_per_call,
    )
    fallback_coder_provider_identity_sha256 = canonical_sha256(
        pipeline._llm_signature(coder_transports[0])
    )
    coder = CoderAgent(coder_transports[0], repair_llm=coder_transports[1])
    # Opt-in altitude-2a: delegate script authoring + self-repair to a local
    # coding-agent CLI when EASYICU_AGENTIC_CODER_BACKEND is set. Off by default;
    # degrades back to ``coder`` when the CLI is unavailable. The script it
    # returns is still executed + evidence-bound by the instrumented runtime.
    from ..agents.agentic_coder import AgenticCoderAgent, maybe_wrap_coder

    coder = maybe_wrap_coder(coder)
    coder_provider_identity_sha256 = (
        canonical_sha256(
            {
                "schema": "easyicu.agentic_coder_provider/1",
                "backend": coder.backend,
                "fallback_provider_identity_sha256": (
                    fallback_coder_provider_identity_sha256
                ),
            }
        )
        if isinstance(coder, AgenticCoderAgent)
        else fallback_coder_provider_identity_sha256
    )
    analyzer = AnalyzerAgent(
        budgeted_role_client(
            role_resolver,
            "analyzer",
            "analyzer_interpretation",
            limit_tokens=pipeline._max_prompt_tokens_per_call,
        )
    )
    supervisor = RuntimeSupervisor(
        clinical_semantics=ClinicalSemanticsAgent(),
        data_extraction=DataExtractionAgent(),
        statistical_analysis=StatisticalAnalysisAgent(),
        visualization=VisualizationAgent(),
        critic=CriticAgent(role_resolver("analyzer")),
    )
    runner = pipeline._build_runner(
        run_dir=run_dir,
        cohort_path=cohort_path,
        target_outcome=context.target_outcome,
        universe_path=universe_path,
        **run_input_authority_state.runner_bindings(),
    )
    step_executor = StepExecutor(clear_output_dir=_clear_output_dir)
    run_coordinator = RunCoordinator()
    usage_auditor = ConceptUsageAuditor()
    pattern_auditor = AnalysisPatternAuditor()
    stat_validator = StatisticalValidator()
    figure_contract_validator = FigureContractQualityValidator()
    figure_source_validator = FigureSourceDataValidator()
    clinical_validator = ClinicalConstraintValidator()
    cross_step_cohort_lock_validator = CrossStepCohortLockValidator()
    cross_step_registered_output_validator = CrossStepRegisteredOutputValidator()
    cross_step_reconciliation_trace_validator = CrossStepReconciliationTraceValidator()
    cross_step_source_status_validator = CrossStepSourceStatusValidator()
    step_summary_fraction_validator = StepSummaryFractionValidator()
    step_summary_integrity_validator = StepSummaryIntegrityValidator()
    primary_model_contract_validator = PrimaryModelContractValidator()
    statistical_guard = StatisticalGuard()
    llm_concept_audit_cache = LLMConceptAuditCache(run_dir)
    llm_concept_auditor_source = inspect.getsourcefile(LLMConceptAuditor)
    llm_concept_auditor_implementation_sha256 = (
        sha256_of_file(Path(llm_concept_auditor_source))
        if llm_concept_auditor_source and Path(llm_concept_auditor_source).is_file()
        else ""
    )
    runtime_state = supervisor.bootstrap_state(run_id=run_id, context=context)
    repair_ledger = RepairLedger(run_dir / "repairs_applied.json")
    repair_ledger_lock = threading.Lock()

    per_step_records: List[Dict[str, Any]] = []
    step_attempt_history: List[Dict[str, Any]] = []
    probe_summary: Dict[str, Any] = {}
    resumed_step_ids: set = set()
    # Steps can finish before the ordinary plan execution loop.  Keep those
    # ids distinct from resume state: a probe-aware replan is allowed to retain
    # the probe in its returned plan, but that must not schedule a second coder
    # execution for work the host already completed deterministically.
    preexecuted_step_ids: set = set()
    if plan_result.resume_state is not None:
        resume_application = resume_controller.apply()
        step_attempt_history.extend(resume_application.audit_history)
        per_step_records.extend(resume_application.per_step_records)
        resumed_step_ids = set(resume_application.resumed_step_ids)
        preexecuted_step_ids.update(resumed_step_ids)
        findings.extend(resume_application.findings)
        probe_summary = resume_application.probe_summary
        if resumed_step_ids:
            print(
                f"[research_agent] resume: skipping {len(resumed_step_ids)} "
                f"already-completed step(s) — {sorted(resumed_step_ids)}"
            )
    adopt_existing_host_cohort_materialization(
        plan=plan,
        run_dir=run_dir,
        cohort_path=run_input_authority_state.analysis_path,
        evidence=evidence,
        prompt_pack_version=prompt_version,
        llm_signature=llm_signature,
        gate_stamp=_deterministic_gate_stamp(),
        per_step_records=per_step_records,
        preexecuted_step_ids=preexecuted_step_ids,
        findings=findings,
        cohort_concept_ids=plan_result.cohort_concept_ids,
    )

    _flush_partial_manifest = functools.partial(
        _step_flush_partial_manifest,
        per_step_records=per_step_records,
        step_attempt_history=step_attempt_history,
        run_id=run_id,
        run_dir=run_dir,
        evidence=evidence,
        plan=plan,
        plan_path=plan_path,
        context=context,
        plan_result=plan_result,
        findings=findings,
        llm_signature=llm_signature,
        prompt_version=prompt_version,
        prompt_files=prompt_files,
        pipeline=pipeline,
        notes=notes,
        runtime_state=runtime_state,
        repair_ledger=repair_ledger,
        _replan_state=_replan_state,
    )

    runtime_packets = {
        "clinical_semantics_resolution": runtime_state.semantics,
        "data_extraction_request": runtime_state.extraction_request,
        "data_extraction_result": runtime_state.extraction_result,
    }
    for alias, packet in runtime_packets.items():
        if packet is None or evidence.get(alias) is not None:
            continue
        evidence.register_json(
            kind="log",
            description=f"Typed runtime packet: {alias}.",
            payload=packet.model_dump(mode="json"),
            filename=f"{alias}.json",
            evidence_id=alias,
            aliases=[alias],
            producer="runtime_supervisor",
            generation_mode="system",
            prompt_pack_version=prompt_version,
            metadata={"run_id": run_id},
        )

    _flush_partial_manifest()

    _register_plan_revision = functools.partial(
        _step_register_plan_revision,
        producer="replanner",
        run_dir=run_dir,
        context=context,
        evidence=evidence,
        llm_signature=llm_signature,
        prompt_version=prompt_version,
    )

    _no_analysis_step_has_run = functools.partial(
        _step_no_analysis_step_has_run,
        per_step_records=per_step_records,
    )
    _universe_columns = functools.partial(
        _step_universe_columns,
        run_input_authority_state=run_input_authority_state,
        universe_path=universe_path,
    )

    def _try_materialize_cohort_from_prose(
        candidate_plan: AnalysisPlan,
        *,
        reason: str,
    ) -> bool:
        nonlocal cohort_path, runner
        success, cohort_path, runner = _step_try_materialize_cohort_from_prose(
            candidate_plan,
            reason=reason,
            _replan_state=_replan_state,
            context=context,
            emit_progress=emit_progress,
            evidence=evidence,
            findings=findings,
            llm_signature=llm_signature,
            per_step_records=per_step_records,
            pipeline=pipeline,
            preexecuted_step_ids=preexecuted_step_ids,
            prompt_version=prompt_version,
            role_resolver=role_resolver,
            run_dir=run_dir,
            run_id=run_id,
            run_input_authority_state=run_input_authority_state,
            universe_path=universe_path,
            no_analysis_step_has_run=_no_analysis_step_has_run,
            universe_columns=_universe_columns,
            cohort_path=cohort_path,
            runner=runner,
            cohort_concept_ids=plan_result.cohort_concept_ids,
        )
        return success
    _enforce_cohort_contract_on_executing_plan = functools.partial(
        _step_enforce_cohort_contract_on_executing_plan,
        _replan_state=_replan_state,
        findings=findings,
        run_input_authority_state=run_input_authority_state,
    )

    _resolve_cohort_definition = functools.partial(
        _step_resolve_cohort_definition,
        try_materialize=_try_materialize_cohort_from_prose,
        enforce_contract=_enforce_cohort_contract_on_executing_plan,
    )

    if _resolve_cohort_definition(plan, reason="execute_start"):
        # Same obligation as the replan path below, on the plan that entered
        # execution: whatever cohort the host just wrote is the one this run
        # analyses, so an immutable record of it has to exist before any step
        # runs.  Sealed only when the cohort actually changed -- every run
        # reaches this line, and registering an identical revision on each of
        # them would add a record that carries no new authority.
        plan_path = _register_plan_revision(
            plan,
            reason="execute_start_cohort_materialization",
            producer="cohort_materializer",
        )
        plan_result.plan_path = plan_path

    def _maybe_replan(
        *,
        current_plan: AnalysisPlan,
        reason: str,
        probe_summary_payload: Optional[Dict[str, Any]] = None,
        completed_records: Optional[Sequence[Dict[str, Any]]] = None,
        directive: Optional[str] = None,
        force: bool = False,
    ) -> AnalysisPlan:
        nonlocal plan_path
        revised, plan_path = _step_maybe_replan(
            current_plan=current_plan,
            reason=reason,
            probe_summary_payload=probe_summary_payload,
            completed_records=completed_records,
            directive=directive,
            force=force,
            plan_path=plan_path,
            _replan_state=_replan_state,
            context=context,
            findings=findings,
            pipeline=pipeline,
            plan_result=plan_result,
            role_resolver=role_resolver,
            run_dir=run_dir,
            skill_obj=skill_obj,
            flush_partial_manifest=_flush_partial_manifest,
            resolve_cohort_definition=_resolve_cohort_definition,
            register_plan_revision=_register_plan_revision,
        )
        return revised

    trajectory_plan_blocked = False
    typed_plan_dag_blocked = False
    product_promise_blocked = False
    endpoint_contract_blocked = False
    probe_step_id = "00_probe"
    # The plan-time gates below validate the PLAN, not the probe, so they
    # must not live inside the probe branch. canary36 is why: its probe was
    # satisfied by pre-execution, the whole branch was skipped, and with it
    # went the typed-DAG, primary-cohort, trajectory, declared-input,
    # product-promise and owner-declaration gates AND the replan that
    # answers them. A robustness step whose declaration the gate names
    # exactly then reached execution unrepaired and died -- in a run where
    # six other steps were claimed by their deterministic owners.
    probe_summary: Optional[Dict[str, Any]] = None
    probe_record: Optional[Dict[str, Any]] = None
    if pipeline._enable_probe_step and probe_step_id not in resumed_step_ids:
        probe_summary, probe_record = _step_build_probe_summary_and_record(
            services=services,
            context=context,
            cohort_path=cohort_path,
            run_dir=run_dir,
            probe_step_id=probe_step_id,
            evidence=evidence,
            per_step_records=per_step_records,
            preexecuted_step_ids=preexecuted_step_ids,
            _flush_partial_manifest=_flush_partial_manifest,
        )
    typed_plan_preflight = _typed_plan_dag_findings(plan)
    primary_cohort_preflight = primary_analysis_cohort_plan_findings(plan=plan)
    trajectory_preflight = trajectory_plan_dag_findings(
        plan=plan,
        context=context,
        long_trajectory_bound=long_trajectory_bound,
    )
    declared_input_preflight = declared_raw_input_plan_findings(
        plan=plan,
        context=context,
    )
    owner_declaration_preflight = owner_declaration_plan_findings(plan=plan)
    product_promise_preflight = product_promise_plan_findings(plan=plan)
    endpoint_preflight = endpoint_contract_findings(
        plan, context=context, severity="error"
    )
    replan_directives = _step_build_replan_directives(
        typed_plan_preflight=typed_plan_preflight,
        primary_cohort_preflight=primary_cohort_preflight,
        trajectory_preflight=trajectory_preflight,
        declared_input_preflight=declared_input_preflight,
        owner_declaration_preflight=owner_declaration_preflight,
        product_promise_preflight=product_promise_preflight,
        endpoint_preflight=endpoint_preflight,
    )
    typed_plan_directive = replan_directives["typed_plan_directive"]
    primary_cohort_directive = replan_directives["primary_cohort_directive"]
    trajectory_directive = replan_directives["trajectory_directive"]
    declared_input_directive = replan_directives["declared_input_directive"]
    owner_declaration_directive = replan_directives[
        "owner_declaration_directive"
    ]
    product_promise_directive = replan_directives["product_promise_directive"]
    endpoint_directive = replan_directives["endpoint_directive"]
    plan = _step_run_initial_replan(
        _maybe_replan=_maybe_replan,
        plan=plan,
        probe_record=probe_record,
        probe_summary=probe_summary,
        per_step_records=per_step_records,
        endpoint_directive=endpoint_directive,
        typed_plan_directive=typed_plan_directive,
        primary_cohort_directive=primary_cohort_directive,
        trajectory_directive=trajectory_directive,
        declared_input_directive=declared_input_directive,
        product_promise_directive=product_promise_directive,
        owner_declaration_directive=owner_declaration_directive,
        typed_plan_preflight=typed_plan_preflight,
        endpoint_preflight=endpoint_preflight,
        primary_cohort_preflight=primary_cohort_preflight,
        trajectory_preflight=trajectory_preflight,
        declared_input_preflight=declared_input_preflight,
        product_promise_preflight=product_promise_preflight,
        owner_declaration_preflight=owner_declaration_preflight,
    )
    final_typed_plan_findings = [
        *_typed_plan_dag_findings(plan),
        *primary_analysis_cohort_plan_findings(plan=plan),
        # A raw input the sealed context cannot resolve raises inside
        # _execute_one_step, and nothing wraps execute_step -- so without this
        # the run dies mid-flight with no sealed artifacts instead of blocking
        # here with a named, repairable finding.
        *declared_raw_input_plan_findings(plan=plan, context=context),
    ]
    # The first pass feeds a repair directive to the Planner.  A repaired plan
    # is still model output, so verify the same product contract again before
    # any executor is selected.  The E1 run that motivated this check removed
    # the duplicate promise but kept the *wrong* kind; without this second pass
    # the replay tried to parse its CSV summary as a JSON statistic.
    final_product_promise_findings = product_promise_plan_findings(plan=plan)
    final_endpoint_findings = [
        finding.model_copy(
            update={
                "detail": {
                    **dict(finding.detail or {}),
                    "stage": "execute_final",
                    "reason": "endpoint_retry_exhausted",
                }
            }
        )
        for finding in endpoint_contract_findings(
            plan, context=context, severity="error"
        )
    ]
    if final_endpoint_findings:
        endpoint_contract_blocked = True
        findings.extend(final_endpoint_findings)
        _flush_partial_manifest(
            {
                "endpoint_contract_blocked": True,
                "endpoint_contract_error_count": len(final_endpoint_findings),
            }
        )
    if final_typed_plan_findings:
        typed_plan_dag_blocked = True
        findings.extend(final_typed_plan_findings)
        _flush_partial_manifest(
            {
                "typed_plan_dag_blocked": True,
                "typed_plan_dag_error_count": len(final_typed_plan_findings),
            }
        )
    if final_product_promise_findings:
        product_promise_blocked = True
        findings.extend(final_product_promise_findings)
        _flush_partial_manifest(
            {
                "product_promise_blocked": True,
                "product_promise_error_count": len(final_product_promise_findings),
            }
        )

    final_trajectory_plan_findings = trajectory_plan_dag_findings(
        plan=plan,
        context=context,
        long_trajectory_bound=long_trajectory_bound,
    )
    if final_trajectory_plan_findings:
        trajectory_plan_blocked = True
        findings.extend(final_trajectory_plan_findings)
        _flush_partial_manifest(
            {
                "trajectory_plan_contract_blocked": True,
                "trajectory_plan_contract_error_count": len(
                    final_trajectory_plan_findings
                ),
            }
        )

    shared_lock = threading.Lock()
    typed_binding_resolver = TypedBindingResolver(
        evidence_store=evidence,
        per_step_records=per_step_records,
        records_lock=shared_lock,
        run_dir=run_dir,
        authoritative_cohort_path=cohort_path,
        development_sample=run_input_authority_state.development_sample,
    )
    step_order = {s.step_id: i for i, s in enumerate(plan.steps)}
    total_steps = len(plan.steps)

    _record_repair = functools.partial(
        _step_record_repair,
        repair_ledger=repair_ledger,
        repair_ledger_lock=repair_ledger_lock,
        llm_signature=llm_signature,
        findings=findings,
    )

    _automatic_repair_authorized = functools.partial(
        _step_automatic_repair_authorized,
        findings=findings,
        record_repair=_record_repair,
    )

    _authorize_automatic_repair = functools.partial(
        _step_authorize_automatic_repair,
        automatic_repair_authorized=_automatic_repair_authorized,
    )

    _propagate_findings_to_evidence = functools.partial(
        _step_propagate_findings_to_evidence,
        evidence=evidence,
    )

    _validator_messages = _step_validator_messages

    _failed_dependency_record = functools.partial(
        _step_failed_dependency_record,
        per_step_records=per_step_records,
        shared_lock=shared_lock,
    )

    def _execute_one_step(step: AnalysisStep) -> Dict[str, Any]:
        nonlocal runtime_state
        runtime_state, step_record = _execute_step(
            step,
            runtime_state=runtime_state,
            _authorize_automatic_repair=_authorize_automatic_repair,
            _automatic_repair_authorized=_automatic_repair_authorized,
            _failed_dependency_record=_failed_dependency_record,
            _flush_partial_manifest=_flush_partial_manifest,
            _propagate_findings_to_evidence=_propagate_findings_to_evidence,
            _record_repair=_record_repair,
            _validator_messages=_validator_messages,
            analyzer=analyzer,
            clinical_validator=clinical_validator,
            coder=coder,
            coder_base_context=coder_base_context,
            coder_provider_identity_sha256=coder_provider_identity_sha256,
            cohort_path=cohort_path,
            concept_audit_environment_sha256=concept_audit_environment_sha256,
            context=context,
            cross_step_cohort_lock_validator=cross_step_cohort_lock_validator,
            cross_step_reconciliation_trace_validator=cross_step_reconciliation_trace_validator,
            cross_step_registered_output_validator=cross_step_registered_output_validator,
            cross_step_source_status_validator=cross_step_source_status_validator,
            emit_progress=emit_progress,
            evidence=evidence,
            figure_contract_validator=figure_contract_validator,
            figure_source_validator=figure_source_validator,
            findings=findings,
            llm_concept_audit_cache=llm_concept_audit_cache,
            llm_concept_audit_client=llm_concept_audit_client,
            llm_concept_auditor_identity_sha256=llm_concept_auditor_identity_sha256,
            llm_concept_auditor_implementation_sha256=llm_concept_auditor_implementation_sha256,
            llm_signature=llm_signature,
            pattern_auditor=pattern_auditor,
            per_step_records=per_step_records,
            pipeline=pipeline,
            plan=plan,
            plan_result=plan_result,
            primary_model_contract_validator=primary_model_contract_validator,
            prompt_files=prompt_files,
            prompt_version=prompt_version,
            requested_resume_from_step_id=requested_resume_from_step_id,
            resume_controller=resume_controller,
            reuse_selected_step_code_opt_in=reuse_selected_step_code_opt_in,
            run_dir=run_dir,
            run_id=run_id,
            run_input_authority_state=run_input_authority_state,
            runner=runner,
            services=services,
            shared_lock=shared_lock,
            stat_validator=stat_validator,
            statistical_guard=statistical_guard,
            step_attempt_history=step_attempt_history,
            step_evidence_commit=step_evidence_commit,
            step_executor=step_executor,
            step_order=step_order,
            step_summary_fraction_validator=step_summary_fraction_validator,
            step_summary_integrity_validator=step_summary_integrity_validator,
            supervisor=supervisor,
            total_steps=total_steps,
            typed_binding_resolver=typed_binding_resolver,
            universe_path=universe_path,
            usage_auditor=usage_auditor,
        )
        return step_record

    if (
        pipeline._development_sample_size is not None
        and run_input_authority_state.development_sample is None
    ):
        findings.append(
            ValidationFinding(
                validator="development_sample_authority",
                severity="error",
                message=(
                    "Development execution stopped before scientific steps: "
                    "the Agent did not produce a locked, post-QC analysis "
                    "cohort from which the requested non-paper sample could "
                    "be drawn. The host will not silently run those steps on "
                    "the full universe."
                ),
                detail={
                    "paper_authority": False,
                    "stage": "blocked_before_scientific_execution",
                    "target_rows": pipeline._development_sample_size,
                    "seed": pipeline._development_sample_seed,
                    "provider_calls_avoided": "coder_and_step_level_calls",
                },
            )
        )

    development_sample_blocked = (
        pipeline._development_sample_size is not None
        and run_input_authority_state.development_sample is None
    )
    plan_block_reason = replan_review.first_plan_block_reason(
        (
            (_replan_state.get("human_review_pause") is not None, "runtime_replan_human_review_required"),
            (endpoint_contract_blocked, "endpoint_contract_blocked"),
            (trajectory_plan_blocked, "trajectory_plan_contract_blocked"),
            (typed_plan_dag_blocked, "typed_plan_dag_blocked"),
            (product_promise_blocked, "product_promise_blocked"),
            (development_sample_blocked, "development_sample_unauthorized"),
        )
    )
    steps_to_run = (
        []
        if plan_block_reason is not None
        else resume_controller.remaining_steps(
            plan=plan,
            executed_step_ids=set(preexecuted_step_ids),
        )
    )
    has_typed_input_dependencies, has_primary_cohort_universe_producer = (
        _step_emit_execution_preflight(
            plan_block_reason=plan_block_reason,
            plan=plan,
            preexecuted_step_ids=preexecuted_step_ids,
            emit_progress=emit_progress,
            run_id=run_id,
            pipeline=pipeline,
            steps_to_run=steps_to_run,
            findings=findings,
            resumed_step_ids=resumed_step_ids,
        )
    )
    if (
        pipeline._max_concurrent_steps <= 1
        or len(steps_to_run) <= 1
        or pipeline._enable_replanning
        or has_typed_input_dependencies
        or has_primary_cohort_universe_producer
        or requested_stop_after_step_id is not None
        or pipeline._submission_profile_name is not None
    ):

        _maybe_directed_model_replan = functools.partial(
            _step_maybe_directed_model_replan,
            pipeline=pipeline,
            _replan_state=_replan_state,
            cohort_path=cohort_path,
            per_step_records=per_step_records,
            findings=findings,
            _maybe_replan=_maybe_replan,
            plan=plan,
            probe_summary=probe_summary,
        )

        _resolve_run_transition = functools.partial(
            _step_resolve_run_transition,
            run_input_authority_state=run_input_authority_state,
            emit_progress=emit_progress,
            run_id=run_id,
            requested_stop_after_step_id=requested_stop_after_step_id,
            _maybe_directed_model_replan=_maybe_directed_model_replan,
            _replan_state=_replan_state,
            pipeline=pipeline,
            _maybe_replan=_maybe_replan,
            plan=plan,
            probe_summary=probe_summary,
            per_step_records=per_step_records,
        )

        def _apply_revised_plan(
            revised_plan: AnalysisPlan,
            executed_step_ids: set[str],
        ) -> Sequence[AnalysisStep]:
            nonlocal plan, total_steps
            remaining, plan, total_steps = _step_apply_revised_plan(
                revised_plan=revised_plan,
                executed_step_ids=executed_step_ids,
                plan=plan,
                step_order=step_order,
                resume_controller=resume_controller,
                total_steps=total_steps,
            )
            return remaining

        _record_step_exception = functools.partial(
            _step_record_step_exception,
            shared_lock=shared_lock,
            findings=findings,
            _flush_partial_manifest=_flush_partial_manifest,
            per_step_records=per_step_records,
            _append_terminal_step_record=_append_terminal_step_record,
        )

        # ``steps_to_run`` carries the fail-closed preflight decision; recomputing
        # from the full plan here would revive
        # every step after a typed-DAG/trajectory contract ERROR and spend
        # Coder calls on a plan the host has declared non-executable.
        run_coordinator.run_sequential(
            state=RunExecutionState(
                remaining_steps=list(steps_to_run),
                executed_step_ids=set(preexecuted_step_ids),
                stop_on_failure=(pipeline._submission_profile_name is not None),
                stop_failure_roles=frozenset({"primary"}),
            ),
            execute_step=_execute_one_step,
            resolve_transition=_resolve_run_transition,
            apply_revised_plan=_apply_revised_plan,
            on_step_exception=_record_step_exception,
        )
    else:

        _record_parallel_worker_error = functools.partial(
            _step_record_parallel_worker_error,
            shared_lock=shared_lock,
            findings=findings,
        )

        run_coordinator.run_parallel(
            steps=steps_to_run,
            max_workers=pipeline._max_concurrent_steps,
            execute_step=_execute_one_step,
            submit_step=_submit_in_current_context,
            on_worker_error=_record_parallel_worker_error,
        )
    if run_input_authority_state.corrupted:
        _flush_partial_manifest(
            {
                "run_input_authority_corrupted": True,
                "run_input_authority_corrupted_step_id": (
                    run_input_authority_state.step_id
                ),
                "remaining_steps_suppressed": True,
            }
        )
        plan_result.plan = plan
        plan_result.plan_path = plan_path
        return _ExecutePhaseResult(
            plan=plan,
            per_step_records=per_step_records,
            step_attempt_history=step_attempt_history,
            probe_summary=probe_summary,
            runtime_state=runtime_state,
            flush_partial_manifest=_flush_partial_manifest,
        )

    if (
        not endpoint_contract_blocked
        and not trajectory_plan_blocked
        and not typed_plan_dag_blocked
        and not product_promise_blocked
        and trajectory_plan_contract_applies(
            plan=plan,
            context=context,
            long_trajectory_bound=long_trajectory_bound,
        )
    ):
        run_level_trajectory_findings = trajectory_bundle_findings(
            context=context,
            plan=plan,
            per_step_records=per_step_records,
            evidence=evidence,
            run_dir=run_dir,
            cohort_path=cohort_path,
            long_trajectory_bound=long_trajectory_bound,
        )
        findings.extend(run_level_trajectory_findings)
        _flush_partial_manifest(
            {
                "trajectory_bundle_error_count": sum(
                    finding.severity == "error"
                    for finding in run_level_trajectory_findings
                )
            }
        )

    robustness_result = finalize_run_robustness_panel(
        run_dir=run_dir,
        plan=plan,
        per_step_records=per_step_records,
        cohort_path=cohort_path,
        context=context,
        evidence=evidence,
        prompt_pack_version=prompt_version,
    )
    findings.extend(robustness_result.findings)
    robustness_manifest_update = robustness_result.manifest_update()
    if robustness_manifest_update:
        _flush_partial_manifest(robustness_manifest_update)

    findings += _step_audit_final_figures(
        pipeline=pipeline,
        requested_stop_after_step_id=requested_stop_after_step_id,
        emit_progress=emit_progress,
        run_dir=run_dir,
        run_id=run_id,
        evidence=evidence,
        role_resolver=role_resolver,
    )
    findings.extend(
        _collect_and_persist_run_article_audits(
            context=context,
            plan=plan,
            evidence_store=evidence,
            per_step_records=per_step_records,
            run_dir=run_dir,
            flush_partial_manifest=_flush_partial_manifest,
        )
    )

    plan_result.plan = plan
    plan_result.plan_path = plan_path
    return _ExecutePhaseResult(
        plan=plan,
        per_step_records=per_step_records,
        step_attempt_history=step_attempt_history,
        probe_summary=probe_summary,
        runtime_state=runtime_state,
        flush_partial_manifest=_flush_partial_manifest,
    )


@dataclass
class _StepAuthorityPreparation:
    worker_progress: Any
    quarantine_state: Any
    step_attempt_state: Any
    checkpoint_authority: Any
    coder_context: Any
    coder_authority: Any
    step_execution_cohort_path: Any
    plausibility_authority: Any
    resolved_input_bindings: Any
    resolved_input_evidence_ids: Any
    existing_refs: Any
    resolved_inputs_path: Any
    resolved_inputs_sha256: Any
    run_input_capsule_sha256: Any
    sealed_renderer_state: Any
    sealed_renderer_authorized_code_sha256: Any
    monotonic_concept_constraints: Any
    _llm_repair_budget_available: Any
    _logical_llm_repair_budget_available: Any
    step_current: Any


@dataclass
class _StepInitialCodeSettlement:
    code: Any
    local_runtime_state: Any
    standard_executor: Any
    _consume_llm_repair_budget: Any
    _remember_concept_constraints: Any
    _quarantine_error_payloads: Any
    _monotonic_concept_constraint_ticket: Any
    _monotonic_concept_constraint_log: Any
    _repair_with_capsule: Any


@dataclass
class _StepConceptRepairOutcome:
    code: Any
    concept_approved_code_digest: Any
    sealed_renderer_authorized_code_sha256: Any
    concept_audit: Any
    _deterministic_fallback_code: Any
    _authorized_deterministic_concept_repair: Any


@dataclass
class _StepPostCandidateFigurePreparation:
    figure_role: Any
    lineage_input_evidence_ids: Any
    repair_evidence_metadata: Any
    step_summary: Any


def _step_prepare_execution_authority(
    *,
    _failed_dependency_record: Any,
    _flush_partial_manifest: Any,
    _sync_provider_budget: Any,
    coder_base_context: Any,
    coder_provider_identity_sha256: Any,
    cohort_path: Any,
    concept_audit_environment_sha256: Any,
    context: Any,
    emit_progress: Any,
    evidence: Any,
    findings: Any,
    llm_concept_auditor_identity_sha256: Any,
    llm_concept_auditor_implementation_sha256: Any,
    per_step_records: Any,
    pipeline: Any,
    plan: Any,
    plan_result: Any,
    prior_attempt_records: Any,
    prior_step_record: Any,
    prompt_files: Any,
    prompt_version: Any,
    provider_budget: Any,
    provider_receipt_path: Any,
    requested_resume_from_step_id: Any,
    reserved_final_category: Any,
    run_dir: Any,
    run_id: Any,
    run_input_authority_state: Any,
    runtime_state: Any,
    shared_lock: Any,
    step: Any,
    step_attempt_history: Any,
    step_execution_cohort_path: Any,
    step_order: Any,
    step_record: Any,
    step_repair_budget: Any,
    step_summary_integrity_validator: Any,
    total_steps: Any,
    typed_binding_resolver: Any,
    universe_path: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional["_StepAuthorityPreparation"]]:
    primary_cohort_execution_receipt = (
        _planner_materialized_cohort_execution_receipt(
            plan=plan,
            universe_path=universe_path,
            analysis_cohort_path=run_input_authority_state.analysis_path,
        )
        if step_record.get("execution_cohort_role") == _RAW_UNIVERSE_EXECUTION_ROLE
        # One owner decides whether a plan made an explicit locked
        # selection.  An ``all_input_rows`` cohort is such a selection, and
        # it is exactly the case where a host-verified row-conservation
        # receipt is cheapest and most useful: it tells the producer the
        # universe count it must still hold after materialisation.
        and cohort_definition_has_explicit_selection(
            coerce_cohort_definition(getattr(plan, "cohort", None))
        )
        else None
    )
    coder_authority = _coder_authority_with_locked_robustness_specs(
        authority=HostCoderAuthority(),
        context=coder_base_context,
        step=step,
        run_dir=run_dir,
    )
    coder_context, coder_authority = bind_materialized_coder_authority(
        coder_base_context, step, coder_authority
    )
    coder_authority = bind_primary_cohort_role(
        authority=coder_authority,
        locked_cohort_payload=(
            _planner_locked_cohort_prompt_payload(plan)
            if step_record.get("execution_cohort_role")
            == _RAW_UNIVERSE_EXECUTION_ROLE
            else None
        ),
        materialized_execution_payload=(
            json.dumps(
                primary_cohort_execution_receipt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if primary_cohort_execution_receipt is not None
            else None
        ),
    )
    worker_progress = StepWorkerProgress()
    quarantine_state = ConceptQuarantineState()
    step_attempt_state = StepAttemptState()
    checkpoint_authority = CheckpointAuthority(
        run_dir=run_dir,
        step_id=step.step_id,
        state=step_attempt_state,
        step_record=step_record,
        per_step_records=per_step_records,
        step_attempt_history=step_attempt_history,
        shared_lock=shared_lock,
        flush_partial_manifest=_flush_partial_manifest,
        upsert_checkpoint=_upsert_current_capsule_checkpoint,
        provider_receipt_path=provider_receipt_path,
        reserved_final_category=reserved_final_category,
        sync_provider_budget=_sync_provider_budget,
        operations=StepAuthorityOperations(
            load_verified_capsule=load_verified_step_authority_capsule,
            persist_candidate_code=persist_candidate_code,
            seal_deterministic_candidate=seal_deterministic_candidate,
            seal_legacy_candidate=seal_legacy_candidate,
            seal_initial_candidate=seal_initial_generation_candidate,
            seal_repair_candidate=seal_repair_candidate_from_receipt,
            load_provider_receipt=load_provider_call_budget_state,
        ),
    )

    # Batch-1 of the A2 control-plane split: the four budget-accounting
    # closures now live in repair_coordination.StepRepairBudget; the local
    # names below are pure aliases so every call site stays unchanged.
    _logical_llm_repair_budget_available = step_repair_budget.logical_available
    _provider_repair_call_available = step_repair_budget.provider_available
    _llm_repair_budget_available = step_repair_budget.available

    monotonic_concept_constraints = _persisted_monotonic_concept_constraints(
        prior_step_record
    )
    if monotonic_concept_constraints:
        step_record["monotonic_concept_constraints"] = [
            finding.model_dump(mode="json")
            for finding in monotonic_concept_constraints
        ]
    worker_progress.preexecution_runner_repair_name = None
    worker_progress.runner_repair_name = None
    sealed_renderer_state = SealedRendererState()
    sealed_renderer_authorized_code_sha256: Optional[str] = None
    step_current = step_order.get(step.step_id, 0) + 1
    dependency_record = _failed_dependency_record(step)
    if dependency_record is not None:
        parent_step_id = str(dependency_record.get("step_id") or "")
        step_record.update(
            {
                "status": "skipped_dependency_failed",
                "dependency_step_id": parent_step_id,
                "diagnostic_only": True,
                "generation_mode": "system",
            }
        )
        with shared_lock:
            findings.append(
                ValidationFinding(
                    validator="dependency_gate",
                    severity="warning",
                    message=(
                        f"Skipped downstream figure step {step.step_id} because "
                        f"required analysis step {parent_step_id} did not pass."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "dependency_step_id": parent_step_id,
                        "dependency_status": dependency_record.get("status"),
                        "diagnostic_only": True,
                    },
                )
            )
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        emit_progress(
            "step",
            f"Skipped {step.step_id}; required step {parent_step_id} failed.",
            status="skipped",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        return step_record, None
    emit_progress(
        "step",
        f"Step {step_current}/{total_steps} started: {step.step_id}.",
        run_id=run_id,
        step_id=step.step_id,
        current_step=step_current,
        total_steps=total_steps,
    )
    locked_measurement_findings = (
        step_summary_integrity_validator.audit_locked_measurement_data_quality(
            step=step,
            cohort_path=cohort_path,
        )
    )
    locked_measurement_issues = _locked_measurement_data_quality_issues(
        locked_measurement_findings
    )
    if locked_measurement_issues:
        step_record.update(
            {
                "status": "contract_failed",
                "diagnostic_only": True,
                "measurement_provenance_preflight": True,
                "measurement_provenance_repair_suppressed": True,
                "measurement_provenance_terminal_reason": (
                    "locked_cohort_data_quality_failed"
                ),
                "measurement_provenance_terminal_issues": (
                    locked_measurement_issues
                ),
                "contract_findings": [
                    finding.model_dump() for finding in locked_measurement_findings
                ],
                "step_summary": {},
                "llm_repair_used": False,
                "generation_mode": "system",
                "code_repair_attempts": 0,
                "contract_repair_attempts": 0,
            }
        )
        with shared_lock:
            findings.extend(locked_measurement_findings)
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        emit_progress(
            "contract",
            (
                "Locked-cohort measurement provenance failed before code "
                f"generation for {step.step_id}; retained diagnostics "
                "without attempting a repair."
            ),
            status="error",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        return step_record, None
    try:
        (
            existing_refs,
            resolved_input_evidence_ids,
            resolved_input_bindings,
        ) = typed_binding_resolver.resolve_names(
            step.inputs,
            plan=plan,
            consumer_step=step,
        )
    except _EvidenceLineageResolutionError as exc:
        step_record.update(
            {
                "status": "blocked_dependency_evidence",
                "diagnostic_only": True,
                "generation_mode": "system",
                "evidence_lineage_failures": exc.failures,
                **dependency_blocked_candidate_metadata(
                    run_dir,
                    step_id=step.step_id,
                    current_record=prior_step_record,
                    records=prior_attempt_records,
                ),
            }
        )
        lineage_finding = ValidationFinding(
            validator="typed_artifact_evidence_lineage",
            severity="error",
            message=(
                f"Step {step.step_id} was blocked because one or more typed "
                "artifact inputs lack a unique, current, digest-verified "
                "producer output."
            ),
            detail={"step_id": step.step_id, "failures": exc.failures},
        )
        with shared_lock:
            findings.append(lineage_finding)
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        emit_progress(
            "audit",
            f"Blocked {step.step_id}; typed artifact evidence is unresolved.",
            status="error",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        return step_record, None
    step_execution_cohort_path = _bind_step_execution_cohort(
        run_dir, step_execution_cohort_path, resolved_input_bindings, step_record
    )
    plausibility_authority = compile_step_plausibility_authority(
        context=context,
        step=step,
        raw_input_contracts=resolved_raw_input_contracts_for_step(
            coder_base_context=coder_base_context,
            coder_context=coder_context,
            planner_declared_inputs=step.inputs,
            primary_cohort_execution_receipt=primary_cohort_execution_receipt,
        ),
    )
    step_record["flag_only_plausibility_scope"] = (
        plausibility_authority.scope.to_dict()
    )
    resolved_inputs_path = _write_resolved_inputs_manifest(
        run_dir=run_dir,
        step_id=step.step_id,
        planner_declared_inputs=step.inputs,
        bindings=resolved_input_bindings,
        context_path=plan_result.context_path,
        plan_path=plan_result.plan_path,
        raw_input_contracts=plausibility_authority.raw_input_contracts(),
        host_verified_cohort_execution_receipt=(
            primary_cohort_execution_receipt
            if step_execution_cohort_path == universe_path
            else None
        ),
        host_authorized_ambient_trajectory=(
            host_authorized_ambient_trajectory_entry(
                # The unscoped context: this entry describes the staged
                # table, and `coder_context` is the step-scoped projection
                # whose concept list a long-bound run empties out.
                getattr(
                    getattr(context, "materialized_inputs", None),
                    "trajectory",
                    None,
                )
            )
        ),
        # From the sealed context, the study's unique endpoint authority.
        study_endpoint=study_endpoint_declaration_entry(context.endpoint),
        # Which of this step's bound columns are ranks rather than interval
        # measurements. From the unscoped context: the roles are a property
        # of the columns, and the step-scoped projection would drop the ones
        # this step reads through an artifact rather than declaring by name.
        rank_scale_columns=rank_scale_columns_entry(context),
    )
    coder_authority = attach_step_coder_input_authority(
        enabled=pipeline._enable_coder_resources,
        authority=coder_authority,
        run_dir=run_dir,
        profile_ref=pipeline._submission_profile_ref,
        context=coder_context,
        step=step,
        analysis_type=plan.analysis_type,
        resolved_input_bindings=resolved_input_bindings,
        plausibility_scope=plausibility_authority.scope,
        runtime_import_names=pipeline._validated_runtime_capabilities or (),
        step_record=step_record,
        reviewed_memory_runtime=pipeline._reviewed_memory_runtime,
        approved_software_resources=pipeline._approved_capability_resources,
    )
    step_record["resolved_inputs_path"] = str(
        resolved_inputs_path.relative_to(run_dir)
    )
    resolved_inputs_sha256 = sha256_of_file(resolved_inputs_path)
    step_record["resolved_inputs_sha256"] = resolved_inputs_sha256
    try:
        run_input_capsule_sha256 = _verified_run_input_capsule_digest(
            run_dir=run_dir,
            evidence_store=evidence,
        )
    except (OSError, RunInputIdentityError) as exc:
        capsule_finding = ValidationFinding(
            validator="run_input_capsule",
            severity="error",
            message=(
                f"Step {step.step_id} was blocked because its immutable run "
                "input authority is missing or changed."
            ),
            detail={"step_id": step.step_id, "reason": str(exc)},
        )
        step_record.update(
            {
                "status": "contract_failed",
                "generation_mode": "system",
                "run_input_capsule_invalid": True,
            }
        )
        with shared_lock:
            findings.append(capsule_finding)
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        return step_record, None
    step_record["run_input_capsule_sha256"] = run_input_capsule_sha256
    step_record["resolved_input_evidence_ids"] = list(resolved_input_evidence_ids)
    coder_context, terminal_step_record = _step_apply_authority_resume(
        run_input_capsule_sha256=run_input_capsule_sha256,
        run_dir=run_dir,
        step=step,
        _deterministic_gate_stamp=_deterministic_gate_stamp,
        engine_code_sha256=engine_code_sha256,
        validator_code_sha256=validator_code_sha256,
        seal_repair_candidate_from_receipt=seal_repair_candidate_from_receipt,
        coder_context=coder_context,
        coder_authority=coder_authority,
        coder_provider_identity_sha256=coder_provider_identity_sha256,
        resolved_inputs_path=resolved_inputs_path,
        resolved_input_bindings=resolved_input_bindings,
        resolved_input_evidence_ids=resolved_input_evidence_ids,
        cohort_path=cohort_path,
        universe_path=universe_path,
        plan_result=plan_result,
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
        concept_audit_environment_sha256=concept_audit_environment_sha256,
        step_attempt_state=step_attempt_state,
        checkpoint_authority=checkpoint_authority,
        step_record=step_record,
        shared_lock=shared_lock,
        findings=findings,
        _append_terminal_step_record=_append_terminal_step_record,
        _flush_partial_manifest=_flush_partial_manifest,
        per_step_records=per_step_records,
    )
    return None, _StepAuthorityPreparation(
        worker_progress=worker_progress,
        quarantine_state=quarantine_state,
        step_attempt_state=step_attempt_state,
        checkpoint_authority=checkpoint_authority,
        coder_context=coder_context,
        coder_authority=coder_authority,
        step_execution_cohort_path=step_execution_cohort_path,
        plausibility_authority=plausibility_authority,
        resolved_input_bindings=resolved_input_bindings,
        resolved_input_evidence_ids=resolved_input_evidence_ids,
        existing_refs=existing_refs,
        resolved_inputs_path=resolved_inputs_path,
        resolved_inputs_sha256=resolved_inputs_sha256,
        run_input_capsule_sha256=run_input_capsule_sha256,
        sealed_renderer_state=sealed_renderer_state,
        sealed_renderer_authorized_code_sha256=(
            sealed_renderer_authorized_code_sha256
        ),
        monotonic_concept_constraints=monotonic_concept_constraints,
        _llm_repair_budget_available=_llm_repair_budget_available,
        _logical_llm_repair_budget_available=_logical_llm_repair_budget_available,
        step_current=step_current,
    )


def _step_settle_initial_code(
    *,
    _authorize_automatic_repair: Any,
    _flush_partial_manifest: Any,
    _record_repair: Any,
    _sync_provider_budget: Any,
    checkpoint_authority: Any,
    coder: Any,
    coder_authority: Any,
    coder_context: Any,
    coder_provider_identity_sha256: Any,
    context: Any,
    emit_progress: Any,
    existing_refs: Any,
    findings: Any,
    monotonic_concept_constraints: Any,
    per_step_records: Any,
    pipeline: Any,
    plan: Any,
    plan_result: Any,
    plausibility_authority: Any,
    prior_attempt_records: Any,
    prior_step_record: Any,
    prompt_version: Any,
    provider_budget: Any,
    quarantine_state: Any,
    requested_resume_from_step_id: Any,
    resolved_input_bindings: Any,
    resolved_inputs_sha256: Any,
    resume_controller: Any,
    reuse_selected_step_code_opt_in: Any,
    run_dir: Any,
    run_id: Any,
    run_input_capsule_sha256: Any,
    runtime_state: Any,
    sealed_renderer_state: Any,
    services: Any,
    shared_lock: Any,
    step: Any,
    step_attempt_state: Any,
    step_current: Any,
    step_record: Any,
    step_repair_budget: Any,
    supervisor: Any,
    total_steps: Any,
    worker_progress: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional["_StepInitialCodeSettlement"]]:
    repair_reservation = StepRepairReservation(
        step=step,
        repair_budget=step_repair_budget,
        checkpoint_authority=checkpoint_authority,
        attempt_state=step_attempt_state,
        coder_context=coder_context,
        coder_authority=coder_authority,
        resolved_inputs_sha256=resolved_inputs_sha256,
        coder_provider_identity_sha256=coder_provider_identity_sha256,
        prompt_version=prompt_version,
        run_input_capsule_sha256=run_input_capsule_sha256,
        deterministic_gate_stamp=_deterministic_gate_stamp(),
    )
    _consume_llm_repair_budget = repair_reservation.consume
    local_runtime_state = supervisor.prepare_step_state(
        state=runtime_state,
        context=context,
        step=step,
        evidence_refs=existing_refs,
    )
    step_record["analysis_request"] = (
        local_runtime_state.analysis_request.model_dump(mode="json")
        if local_runtime_state.analysis_request is not None
        else None
    )
    step_record["visualization_request"] = (
        local_runtime_state.visualization_request.model_dump(mode="json")
        if local_runtime_state.visualization_request is not None
        else None
    )
    step_record["semantics_family"] = local_runtime_state.analysis_family

    worker_progress.deterministic_fallback_used = False
    worker_progress.deterministic_standard_executor_used = False

    _remember_concept_constraints = functools.partial(
        _step_remember_concept_constraints,
        monotonic_concept_constraints,
        step_record,
    )
    _quarantine_error_payloads = functools.partial(
        _step_quarantine_error_payloads,
        monotonic_concept_constraints,
        step_record,
    )
    _monotonic_concept_constraint_payload = functools.partial(
        _step_monotonic_concept_constraint_payload,
        monotonic_concept_constraints,
    )
    _monotonic_concept_constraint_ticket = functools.partial(
        _step_monotonic_concept_constraint_ticket,
        monotonic_concept_constraints,
    )
    _monotonic_concept_constraint_log = functools.partial(
        _step_monotonic_concept_constraint_log,
        monotonic_concept_constraints,
    )
    candidate_recovery = StepCandidateRecovery(
        StepCandidateRecoveryRequest(
            step=step,
            run_dir=run_dir,
            run_id=run_id,
            step_current=step_current,
            total_steps=total_steps,
            requested_resume_from_step_id=requested_resume_from_step_id,
            prior_step_record=prior_step_record,
            prior_attempt_records=prior_attempt_records,
            provider_budget=provider_budget,
            step_repair_budget=step_repair_budget,
            step_record=step_record,
            findings=findings,
            shared_lock=shared_lock,
            worker_progress=worker_progress,
            quarantine_state=quarantine_state,
            resume_controller=resume_controller,
            analysis_family=local_runtime_state.analysis_family,
            coder=coder,
            coder_context=coder_context,
            coder_authority=coder_authority,
            step_attempt_state=step_attempt_state,
            checkpoint_authority=checkpoint_authority,
            deterministic_runner_repair_enabled=(
                pipeline._enable_deterministic_runner_repair
            ),
            emit_progress=emit_progress,
            remember_concept_constraints=_remember_concept_constraints,
            consume_llm_repair_budget=_consume_llm_repair_budget,
            sync_provider_budget=_sync_provider_budget,
            authorize_automatic_repair=_authorize_automatic_repair,
            record_repair=_record_repair,
        )
    )

    _use_quarantined_draft = functools.partial(
        _step_use_quarantined_draft,
        candidate_recovery,
    )
    _use_resumed_code = functools.partial(
        _step_use_resumed_code,
        candidate_recovery,
    )
    _repair_with_capsule = functools.partial(
        _step_repair_with_capsule,
        candidate_recovery,
    )
    _reserve_compatibility_repair = functools.partial(
        _step_reserve_compatibility_repair,
        candidate_recovery,
    )
    _resume_deterministic_repair_code = functools.partial(
        _step_resume_deterministic_repair_code,
        candidate_recovery,
    )
    _resume_critic_repair_code = functools.partial(
        _step_resume_critic_repair_code,
        candidate_recovery,
    )
    _publication_figure_preflight_supported = functools.partial(
        _step_publication_figure_preflight_supported,
        step=step,
        run_dir=run_dir,
        services=services,
    )
    _absolute_risk_context_preflight_supported = functools.partial(
        _step_absolute_risk_context_preflight_supported,
        step=step,
    )
    _deterministic_absolute_risk_context_code = functools.partial(
        _step_deterministic_absolute_risk_context_code,
        preflight_supported=_absolute_risk_context_preflight_supported,
        worker_progress=worker_progress,
        pipeline=pipeline,
        step=step,
        step_record=step_record,
        plausibility_authority=plausibility_authority,
        emit_progress=emit_progress,
        run_id=run_id,
        step_current=step_current,
        total_steps=total_steps,
    )
    _robustness_sensitivity_preflight_supported = functools.partial(
        _step_robustness_sensitivity_preflight_supported,
        step=step,
    )
    _deterministic_robustness_sensitivity_code = functools.partial(
        _step_deterministic_robustness_sensitivity_code,
        preflight_supported=_robustness_sensitivity_preflight_supported,
        worker_progress=worker_progress,
        pipeline=pipeline,
        step=step,
        step_record=step_record,
        emit_progress=emit_progress,
        run_id=run_id,
        step_current=step_current,
        total_steps=total_steps,
        plausibility_scope=plausibility_authority.scope,
    )
    _missingness_audit_preflight_supported = functools.partial(
        _step_missingness_audit_preflight_supported,
        step=step,
    )
    _deterministic_missingness_audit_code = functools.partial(
        _step_deterministic_missingness_audit_code,
        preflight_supported=_missingness_audit_preflight_supported,
        worker_progress=worker_progress,
        pipeline=pipeline,
        step=step,
        step_record=step_record,
        emit_progress=emit_progress,
        run_id=run_id,
        step_current=step_current,
        total_steps=total_steps,
    )
    standard_executor_trace: list[StandardExecutorCandidate] = []
    standard_executor = select_standard_executor(
        step,
        plan=plan,
        plausibility_scope=plausibility_authority.scope,
        resolved_bindings=resolved_input_bindings,
        trajectory_scientific_runtime_authority=pipeline._scientific_runtime_authorities.trajectory,
        current_case_scientific_runtime_authority=pipeline._scientific_runtime_authorities.current_case,
        scientific_runtime_projection_sha256=getattr(
            pipeline, "_scientific_runtime_projection_sha256", None
        ),
        trace=standard_executor_trace,
    )
    preflight_standard_code = None
    if standard_executor is not None:
        worker_progress.deterministic_standard_executor_used = True
        step_record["deterministic_standard_selection_reason"] = (
            standard_executor.selection_reason
        )
        step_record["deterministic_standard_analysis"] = (
            standard_executor.analysis_kind
        )
        emit_progress(
            "coder",
            f"{standard_executor.progress_message} for {step.step_id}.",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
            fallback_reason=standard_executor.selection_reason,
        )
        preflight_standard_code = standard_executor.code
    # A step no deterministic owner claims falls to the stochastic Coder
    # silently.  Record which owners were consulted and how they answered,
    # so the next reader can tell an unsupported analysis apart from a
    # supported one wearing a name or a product count nobody recognises.
    # The verdicts come from the selector's own trace, never from re-running
    # its predicates here -- a second evaluation cannot see the gates the
    # selector applies after a contract matches.
    step_record["standard_executor_candidates"] = (
        standard_executor_candidate_report(
            step,
            plan=plan,
            trace=standard_executor_trace,
            resolved_bindings=resolved_input_bindings,
            claimed_by=(
                standard_executor.analysis_kind
                if standard_executor is not None
                else None
            ),
        )
    )
    # ...and "silently" is the part that is not acceptable when the owner is
    # only waiting on a field.  The plan-time gate already asked for it and
    # spent a forced replan on the answer; arriving here means the Planner
    # did not fill it in.  Handing the step to the Coder anyway is a
    # fail-open at a declaration boundary: it produces a number for the
    # paper's primary result whose model nobody declared, by the one actor
    # whose accumulated repair guidance records it going wrong.
    #
    # Blocked per step, not per run.  The sibling plan-DAG blocks set
    # ``steps_to_run = []`` and kill everything; a step the host merely
    # cannot claim should not take the table-one and missingness steps down
    # with it.  The manuscript still cannot be authorised without its
    # primary result -- that is a different gate's job, and it already
    # holds.
    #
    # What counts as under-declared is decided in one place, shared with the
    # plan-time gate that already asked for the field.  The verdicts come
    # from the selector's own trace, never from re-running its predicates
    # here -- a second evaluation cannot see the gates the selector applies
    # after a contract matches.
    owner_declaration_gaps = execution_declaration_refusal(
        claimed_by=standard_executor,
        trace=standard_executor_trace,
    )
    if owner_declaration_gaps:
        missing_by_owner = {
            candidate.analysis_kind: list(candidate.missing_declarations)
            for candidate in owner_declaration_gaps
        }
        step_record.update(
            {
                "status": "blocked_owner_declaration_incomplete",
                "diagnostic_only": True,
                "generation_mode": "system",
                "owner_declaration_missing": missing_by_owner,
            }
        )
        declaration_finding = ValidationFinding(
            validator="execution_owner_declaration",
            severity="error",
            message=(
                f"Step {step.step_id} was refused rather than generated: the "
                "host has a deterministic owner for its declared product and "
                "the plan still does not declare "
                + "; ".join(
                    f"{kind}: {', '.join(repr(name) for name in names)}"
                    for kind, names in sorted(missing_by_owner.items())
                )
                + ". The plan-time gate asked for these fields and the replan "
                "did not supply them. Generating this step would answer the "
                "question with a model the plan never specified."
            ),
            detail={
                "reason": "owner_declaration_incomplete_at_execution",
                "step_id": step.step_id,
                "missing_declarations_by_owner": missing_by_owner,
            },
        )
        with shared_lock:
            findings.append(declaration_finding)
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        emit_progress(
            "audit",
            f"Blocked {step.step_id}; its declared model is incomplete.",
            status="error",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        return step_record, None
    preflight_figure_code = (
        None
        if preflight_standard_code is not None
        else _deterministic_publication_figure_code(
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
    )
    quarantined_resume_draft = (
        None
        if (
            preflight_standard_code is not None or preflight_figure_code is not None
        )
        else resume_controller.quarantined_concept_draft_for_step(step.step_id)
    )
    resume_critic_repair_code = (
        None
        if (
            preflight_standard_code is not None
            or preflight_figure_code is not None
            or quarantined_resume_draft is not None
        )
        else _resume_critic_repair_code()
    )
    resume_deterministic_repair_code = (
        None
        if (
            preflight_standard_code is not None
            or preflight_figure_code is not None
            or quarantined_resume_draft is not None
            or resume_critic_repair_code is not None
        )
        else _resume_deterministic_repair_code()
    )
    preflight_resumed_code = None
    failed_contract_code_preflight_reuse = False
    if (
        preflight_standard_code is None
        and preflight_figure_code is None
        and quarantined_resume_draft is None
        and resume_deterministic_repair_code is None
        and resume_critic_repair_code is None
    ):
        resumed_code_candidate = resume_controller.prior_code_for_step(step.step_id)
        failed_contract_code_preflight_reuse = (
            _failed_contract_code_can_be_reused_before_coder(
                prior_step_record=prior_step_record,
                resumed_code=resumed_code_candidate,
                step=step,
                plan=plan,
                resolved_inputs_sha256=resolved_inputs_sha256,
                run_input_capsule_sha256=run_input_capsule_sha256,
            )
        )
        if reuse_selected_step_code_opt_in or failed_contract_code_preflight_reuse:
            preflight_resumed_code = resumed_code_candidate
    code, terminal_step_record = _step_resolve_initial_code(
        preflight_standard_code=preflight_standard_code,
        preflight_figure_code=preflight_figure_code,
        resume_deterministic_repair_code=resume_deterministic_repair_code,
        resume_critic_repair_code=resume_critic_repair_code,
        preflight_resumed_code=preflight_resumed_code,
        failed_contract_code_preflight_reuse=(
            failed_contract_code_preflight_reuse
        ),
        quarantined_resume_draft=quarantined_resume_draft,
        step_attempt_state=step_attempt_state,
        step=step,
        step_record=step_record,
        worker_progress=worker_progress,
        findings=findings,
        shared_lock=shared_lock,
        standard_executor=standard_executor,
        _use_quarantined_draft=_use_quarantined_draft,
        _use_resumed_code=_use_resumed_code,
        _deterministic_absolute_risk_context_code=(
            _deterministic_absolute_risk_context_code
        ),
        _deterministic_robustness_sensitivity_code=(
            _deterministic_robustness_sensitivity_code
        ),
        _deterministic_missingness_audit_code=(
            _deterministic_missingness_audit_code
        ),
        run_dir=run_dir,
        pipeline=pipeline,
        services=services,
        plan_result=plan_result,
        sealed_renderer_state=sealed_renderer_state,
        _authorize_automatic_repair=_authorize_automatic_repair,
        _record_repair=_record_repair,
        per_step_records=per_step_records,
        _append_terminal_step_record=_append_terminal_step_record,
        _flush_partial_manifest=_flush_partial_manifest,
        emit_progress=emit_progress,
        run_id=run_id,
        step_current=step_current,
        total_steps=total_steps,
        coder_context=coder_context,
        coder_authority=coder_authority,
        provider_budget=provider_budget,
        checkpoint_authority=checkpoint_authority,
        _reserve_compatibility_repair=_reserve_compatibility_repair,
        coder=coder,
        _sync_provider_budget=_sync_provider_budget,
        resume_controller=resume_controller,
        prior_step_record=prior_step_record,
        prior_attempt_records=prior_attempt_records,
    )
    return None, _StepInitialCodeSettlement(
        code=code,
        local_runtime_state=local_runtime_state,
        standard_executor=standard_executor,
        _consume_llm_repair_budget=_consume_llm_repair_budget,
        _remember_concept_constraints=_remember_concept_constraints,
        _quarantine_error_payloads=_quarantine_error_payloads,
        _monotonic_concept_constraint_ticket=_monotonic_concept_constraint_ticket,
        _monotonic_concept_constraint_log=_monotonic_concept_constraint_log,
        _repair_with_capsule=_repair_with_capsule,
    )


def _step_run_concept_repair_phase(
    *,
    _authorize_automatic_repair: Any,
    _consume_llm_repair_budget: Any,
    _flush_partial_manifest: Any,
    _llm_repair_budget_available: Any,
    _logical_llm_repair_budget_available: Any,
    _monotonic_concept_constraint_ticket: Any,
    _quarantine_error_payloads: Any,
    _record_repair: Any,
    _remember_concept_constraints: Any,
    _repair_with_capsule: Any,
    _sync_provider_budget: Any,
    checkpoint_authority: Any,
    code: Any,
    coder_authority: Any,
    coder_context: Any,
    concept_audit_environment_sha256: Any,
    context: Any,
    emit_progress: Any,
    findings: Any,
    llm_concept_audit_cache: Any,
    llm_concept_audit_client: Any,
    llm_concept_auditor_implementation_sha256: Any,
    pattern_auditor: Any,
    per_step_records: Any,
    pipeline: Any,
    plan_result: Any,
    plausibility_authority: Any,
    prior_attempt_records: Any,
    prior_step_record: Any,
    provider_budget: Any,
    quarantine_state: Any,
    resolved_input_bindings: Any,
    run_dir: Any,
    run_id: Any,
    runtime_state: Any,
    sealed_renderer_state: Any,
    shared_lock: Any,
    standard_executor: Any,
    step: Any,
    step_attempt_state: Any,
    step_current: Any,
    step_record: Any,
    step_repair_budget: Any,
    total_steps: Any,
    usage_auditor: Any,
    worker_progress: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional["_StepConceptRepairOutcome"]]:
    _deterministic_fallback_code = functools.partial(
        _step_deterministic_fallback_code,
        worker_progress=worker_progress,
        pipeline=pipeline,
        plan_result=plan_result,
        step_record=step_record,
        emit_progress=emit_progress,
        step=step,
        run_id=run_id,
        step_current=step_current,
        total_steps=total_steps,
        coder_context=coder_context,
        coder_authority=coder_authority,
    )

    # Every branch above has now settled `code`. The flag-only plausibility
    # receipt is mechanical and host-owned, and it is the single largest
    # pre-execution blocker on record: 37 findings over 32 distinct steps in
    # 8 of the 9 tasks, 53 % of all mechanical-preflight findings. The
    # deterministic executors get it rendered for them; agent-authored steps
    # had to hand-write it and repeatedly could not -- h2's causal step spent
    # BOTH LLM repairs on this one message with five provider calls unspent.
    #
    # The injection itself lives at the audit loop head (see below), not
    # here: a repair that rewrites the script drops an appended host block,
    # and injecting only at initial settling let exactly that happen.
    _authorize_deterministic_concept_reaudit = functools.partial(
        _step_authorize_deterministic_concept_reaudit,
        provider_budget=provider_budget,
        worker_progress=worker_progress,
        step_record=step_record,
        prior_step_record=prior_step_record,
        prior_attempt_records=prior_attempt_records,
    )

    concept_audit = ConceptAuditCoordinator(
        authority=ConceptAuditAuthority(
            context=context,
            step=step,
            resolved_input_bindings=resolved_input_bindings,
            plausibility_scope=plausibility_authority.scope,
            environment_sha256=concept_audit_environment_sha256,
            auditor_implementation_sha256=(
                llm_concept_auditor_implementation_sha256
            ),
            auditor_identity=(
                lambda: pipeline._llm_signature(llm_concept_audit_client)
            ),
            enable_llm_audit=pipeline._enable_llm_concept_audit,
            study_endpoint=study_endpoint_declaration_entry(context.endpoint),
            # Every step of the locked plan, so a requirement the plan
            # assigned to another step stops looking like this script's
            # omission. Id/role/method only: the other steps' rule prose
            # would be the largest block in that prompt and would invite
            # auditing them instead of this one.
            plan_step_roster=tuple(
                {
                    "step_id": other.step_id,
                    "planned_analysis_role": other.planned_analysis_role,
                    "method": other.method,
                }
                for other in (plan_result.plan.steps or ())
                if other.step_id != step.step_id
            ),
        ),
        runtime=ConceptAuditRuntime(
            usage_auditor=usage_auditor,
            pattern_auditor=pattern_auditor,
            cache=llm_concept_audit_cache,
            client=llm_concept_audit_client,
            provider_budget=provider_budget,
            step_attempt_state=step_attempt_state,
            worker_progress=worker_progress,
            quarantine_state=quarantine_state,
            step_record=step_record,
            run_dir=run_dir,
            run_id=run_id,
            step_current=step_current,
            total_steps=total_steps,
            sync_provider_budget=_sync_provider_budget,
            emit_progress=emit_progress,
            quarantine_error_payloads=_quarantine_error_payloads,
            store_quarantined_draft=store_quarantined_concept_draft,
            authorize_deterministic_reaudit=(
                _authorize_deterministic_concept_reaudit
            ),
        ),
    )

    _authorized_deterministic_concept_repair = functools.partial(
        _step_authorized_deterministic_concept_repair,
        authorize=_authorize_automatic_repair,
        step=step,
        context=context,
        step_repair_budget=step_repair_budget,
    )

    concept_repair_result = run_concept_repair_loop(
        ConceptRepairRequest(
            initial_code=code,
            concept_audit=concept_audit,
            step_repair_budget=step_repair_budget,
            checkpoint_authority=checkpoint_authority,
            sealed_renderer_state=sealed_renderer_state,
            standard_executor=standard_executor,
            coder_context=coder_context,
            findings=findings,
            per_step_records=per_step_records,
            shared_lock=shared_lock,
            max_code_repair_attempts=pipeline._max_code_repair_attempts,
            max_step_llm_repair_attempts=(pipeline._max_step_llm_repair_attempts),
            services=ConceptRepairServices(
                authorized_deterministic_repair=(
                    _authorized_deterministic_concept_repair
                ),
                record_repair=_record_repair,
                deterministic_fallback_code=_deterministic_fallback_code,
                logical_budget_available=(_logical_llm_repair_budget_available),
                repair_budget_available=_llm_repair_budget_available,
                consume_llm_repair_budget=_consume_llm_repair_budget,
                remember_concept_constraints=_remember_concept_constraints,
                monotonic_constraint_ticket=(_monotonic_concept_constraint_ticket),
                repair_with_capsule=_repair_with_capsule,
                python_repair_is_materially_changed=(
                    _python_repair_is_materially_changed
                ),
                append_terminal_record=_append_terminal_step_record,
                flush_partial_manifest=_flush_partial_manifest,
            ),
        )
    )
    if concept_repair_result.terminal_record is not None:
        return concept_repair_result.terminal_record, None
    code = concept_repair_result.code
    concept_approved_code_digest = (
        concept_repair_result.concept_approved_code_sha256
    )
    sealed_renderer_authorized_code_sha256 = (
        concept_repair_result.sealed_renderer_authorized_code_sha256
    )

    provider_failure_deferred = bool(
        step_record.get("concept_audit_provider_failure_deferred")
    )
    if (
        quarantine_state.draft_active
        and not quarantine_state.repair_succeeded
        and not provider_failure_deferred
    ):
        hard_gate_finding = ValidationFinding(
            validator="resume",
            severity="error",
            message=(
                "Quarantined concept-audit draft cannot execute before a "
                f"successful coder repair for step {step.step_id}."
            ),
            detail={
                "step_id": step.step_id,
                "quarantined_draft_sha256": step_record.get(
                    "quarantined_draft_sha256"
                ),
            },
        )
        step_record["status"] = "blocked_quarantined_draft"
        with shared_lock:
            findings.append(hard_gate_finding)
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        emit_progress(
            "audit",
            f"Blocked unrepaired quarantined draft for {step.step_id}.",
            status="error",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        return step_record, None

    if (
        quarantine_state.repair_succeeded
        or quarantine_state.policy_superseded
        or quarantine_state.deterministic_revalidated
    ) and not provider_failure_deferred:
        try:
            clear_quarantined_concept_draft(
                run_dir=run_dir,
                step_id=step.step_id,
            )
            step_record["quarantined_requires_repair"] = False
            step_record["quarantine_retired"] = True
            if quarantine_state.policy_superseded:
                step_record["quarantine_retired_by"] = (
                    "deterministic_validator_policy_supersession"
                )
            elif quarantine_state.deterministic_revalidated:
                step_record["quarantine_retired_by"] = (
                    "deterministic_code_gate_revalidation"
                )
        except ValueError as exc:
            cleanup_finding = ValidationFinding(
                validator="resume",
                severity="error",
                message=(
                    "Concept-approved code could not retire its stale "
                    f"quarantine safely for step {step.step_id}: {exc}"
                ),
                detail={"step_id": step.step_id},
            )
            step_record["status"] = "blocked_quarantine_cleanup"
            with shared_lock:
                findings.append(cleanup_finding)
                _append_terminal_step_record(per_step_records, step_record)
                _flush_partial_manifest()
            return step_record, None
    return None, _StepConceptRepairOutcome(
        code=code,
        concept_approved_code_digest=concept_approved_code_digest,
        sealed_renderer_authorized_code_sha256=(
            sealed_renderer_authorized_code_sha256
        ),
        concept_audit=concept_audit,
        _deterministic_fallback_code=_deterministic_fallback_code,
        _authorized_deterministic_concept_repair=(
            _authorized_deterministic_concept_repair
        ),
    )


def _step_prepare_post_candidate_figures(
    *,
    _automatic_repair_authorized: Any,
    _flush_partial_manifest: Any,
    _record_repair: Any,
    attempt_id: Any,
    emit_progress: Any,
    figure_contract_validator: Any,
    figure_source_validator: Any,
    findings: Any,
    per_step_records: Any,
    plan: Any,
    resolved_input_bindings: Any,
    resolved_input_evidence_ids: Any,
    review_checkpoint_id: Any,
    run_dir: Any,
    run_id: Any,
    run_result: Any,
    runtime_state: Any,
    script_record: Any,
    sealed_renderer_authorized_code_sha256: Any,
    services: Any,
    shared_lock: Any,
    standard_executor: Any,
    standard_executor_terminal_block: Any,
    step: Any,
    step_current: Any,
    step_record: Any,
    total_steps: Any,
    worker_progress: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional["_StepPostCandidateFigurePreparation"]]:
    publication_step = _step_requires_publication_figure_exports(step)
    # A deterministic data-only auxiliary does not declare a figure product,
    # so the structural predicate excludes it without special-casing the
    # executor.  A deterministic rendering-only step *does* own the declared
    # figure bundle and must retain publication-figure identity after the
    # contract/source gates pass; otherwise the downstream skill cannot
    # promote its already verified outputs.
    figure_role = (
        "publication_figure"
        if publication_step
        else "analysis_figure"
        if _step_expects_figure(step)
        else None
    )
    if (
        publication_step
        and not _has_figure_exports(run_result.out_dir)
        and sealed_renderer_authorized_code_sha256 is None
    ):
        sibling_repair_id = "sibling_figure_exports_promote_v1"
        promoted = None
        if _automatic_repair_authorized(
            sibling_repair_id,
            step=step,
            source="publication_figure_sibling_promotion",
        ):
            promoted = services.promote_sibling_figure_exports(
                out_dir=run_result.out_dir
            )
        if promoted is not None:
            worker_progress.runner_repair_name = promoted
            step_record["runner_repair"] = promoted
            _record_repair(
                repair_id=promoted,
                step_id=step.step_id,
                trigger={"source": "publication_figure_sibling_promotion"},
                transformation="Promoted sibling figure exports into canonical outputs directory.",
            )
        else:
            rescued = None
            if _step_has_figure_only_output_contract(
                step
            ) and services.deterministic_figure_family_supported_for_upstream(
                run_dir, step.step_id
            ):
                rescued = _repair_publication_figure_in_staging(
                    run_dir=run_dir,
                    current_step_id=step.step_id,
                    out_dir=run_result.out_dir,
                    renderer=(
                        services.render_publication_bundle_from_prior_outputs_for_step
                    ),
                    step_text=f"{step.intent} {step.method}",
                    authorizer=lambda repair_id: _automatic_repair_authorized(
                        repair_id,
                        step=step,
                        source="typed_publication_bundle_rescue",
                    ),
                )
            if rescued is not None:
                worker_progress.runner_repair_name = rescued
                step_record["runner_repair"] = rescued
                _record_repair(
                    repair_id=rescued,
                    step_id=step.step_id,
                    trigger={"source": "typed_publication_bundle_rescue"},
                    transformation=(
                        "Rendered deterministic publication figure bundle "
                        "from the registered parent outputs for this step type."
                    ),
                )
            else:
                parent_step_id = str(step.step_id or "").removesuffix("_figure")
                direct_parent = run_dir / "steps" / parent_step_id
                promoted = None
                if (
                    parent_step_id != str(step.step_id or "")
                    and direct_parent.is_dir()
                    and _automatic_repair_authorized(
                        "publication_bundle_promote_v1",
                        step=step,
                        source="publication_figure_prior_bundle_promotion",
                    )
                ):
                    promoted = services.promote_prior_publication_bundle(
                        run_dir=run_dir,
                        current_step_id=step.step_id,
                        out_dir=run_result.out_dir,
                        require_declared_sources=True,
                    )
                if promoted is not None:
                    worker_progress.runner_repair_name = promoted
                    step_record["runner_repair"] = promoted
                    _record_repair(
                        repair_id=promoted,
                        step_id=step.step_id,
                        trigger={
                            "source": "publication_figure_prior_bundle_promotion"
                        },
                        transformation="Promoted prior publication figure bundle into current outputs directory.",
                    )

    if _should_attempt_detached_figure_binding(
        out_dir=run_result.out_dir,
        sealed_renderer_authorized_code_sha256=(
            sealed_renderer_authorized_code_sha256
        ),
    ):
        with shared_lock:
            repair_binding_records = list(per_step_records)
        detached_repair_binding = _detached_figure_repair_binding(
            step=step,
            plan=plan,
            completed_records=repair_binding_records,
        )
    else:
        detached_repair_binding = None
    repair_source_evidence_ids: List[str] = []
    repair_evidence_metadata: Dict[str, Any] = {}
    if detached_repair_binding is not None:
        (
            repair_target_step_id,
            repair_source_step_id,
            repair_source_evidence_ids,
        ) = detached_repair_binding
        step_record["repair_target_step_id"] = repair_target_step_id
        step_record["source_evidence_ids"] = list(repair_source_evidence_ids)
        repair_evidence_metadata = {
            "repair_target_step_id": repair_target_step_id,
            "source_step_id": repair_source_step_id,
            "source_evidence_ids": list(repair_source_evidence_ids),
        }
        # Persist the same orchestrator binding in the registered summary.
        # The renderer may suggest a parent, but this exact value comes only
        # from the current plan + latest outer execution ledger above.
        summary_path = run_result.out_dir / "step_summary.json"
        try:
            summary_payload = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists()
                else {}
            )
        except Exception:
            summary_payload = {}
        if not isinstance(summary_payload, dict):
            summary_payload = {"raw": summary_payload}
        figure_exports = sorted(
            path.name
            for path in run_result.out_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}
        )
        summary_payload.update(
            {
                "rendering_only": True,
                "source_step_id": repair_source_step_id,
                "repair_target_step_id": repair_target_step_id,
                "source_evidence_ids": list(repair_source_evidence_ids),
                "figure_files": figure_exports,
            }
        )
        summary_path.write_text(
            json.dumps(
                summary_payload,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    lineage_input_evidence_ids = list(
        dict.fromkeys([*resolved_input_evidence_ids, *repair_source_evidence_ids])
    )

    if standard_executor_terminal_block:
        # Defence in depth for every terminal path: only published
        # diagnostics may reach evidence enumeration.
        _remove_standard_executor_pending_artifacts(run_result.out_dir)
    if run_result.outputs_safe_to_collect:
        run_result.artefacts = sorted(
            p
            for p in run_result.out_dir.iterdir()
            if p.is_file()
            and not (
                worker_progress.deterministic_standard_executor_used
                and _is_standard_executor_internal_artifact(p)
            )
        )
    else:
        # A sandbox backend could not prove that a timed-out writer was
        # stopped. Never enumerate or hash its mutable mount. The script
        # and host-written run log remain available outside this list.
        run_result.artefacts = []

    if publication_step and not _has_figure_exports(run_result.out_dir):
        with shared_lock:
            findings.append(
                ValidationFinding(
                    validator="publication_figure_outputs",
                    severity="error",
                    message=(
                        f"Step {step.step_id} completed without any publication-figure exports."
                    ),
                )
            )
            step_record["status"] = "execution_failed"
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        emit_progress(
            "runner",
            f"Publication figure missing for {step.step_id}.",
            status="error",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        return step_record, None

    # Finalise every result-bearing figure before any output is copied into
    # EvidenceStore or any numeric claim is registered.  The staged repair
    # replaces the entire output directory, so running it after registration
    # would leave evidence digests and claims bound to a retired draft.
    step_summary = _load_step_summary_from_outputs(run_result.out_dir)
    # Any host-authored deterministic code, not only a registered standard
    # executor.  The rule itself lives with the writer it governs; widening
    # it here alone was what left the pre-gate site narrower.
    if host_owns_input_binding_receipts(
        deterministic_standard_executor_used=(
            worker_progress.deterministic_standard_executor_used
        ),
        deterministic_fallback_used=worker_progress.deterministic_fallback_used,
        sealed_renderer_repair=bool(
            worker_progress.runner_repair_name
            and is_sealed_renderer_repair(worker_progress.runner_repair_name)
        ),
        resumed_from_generation_mode=step_record.get(
            "resumed_from_generation_mode"
        ),
    ):
        step_summary = _write_host_input_binding_receipts(
            out_dir=run_result.out_dir,
            step_summary=step_summary,
            resolved_input_bindings=resolved_input_bindings,
            consumed_input_keys=(
                standard_executor.consumed_input_keys
                if worker_progress.deterministic_standard_executor_used
                and standard_executor is not None
                else tuple(resolved_input_bindings)
            ),
        )
    _ensure_step_figure_contract(
        step=step,
        out_dir=run_result.out_dir,
        step_summary=step_summary,
        evidence_ids=[script_record.evidence_id, *lineage_input_evidence_ids],
    )
    with shared_lock:
        preseal_completed_records = list(per_step_records)
    preseal_contract_findings = figure_contract_validator.audit(
        step=step,
        out_dir=run_result.out_dir,
        run_dir=run_dir,
        step_summary=step_summary,
    )
    preseal_source_findings = figure_source_validator.audit(
        step=step,
        out_dir=run_result.out_dir,
        run_dir=run_dir,
        step_summary=step_summary,
        completed_step_records=preseal_completed_records,
        resolved_input_bindings=resolved_input_bindings,
    )
    preseal_figure_errors = [
        finding
        for finding in preseal_contract_findings + preseal_source_findings
        if finding.severity == "error"
    ]
    repairable_publication_step = (
        publication_step
        and sealed_renderer_authorized_code_sha256 is None
        and _step_has_figure_only_output_contract(step)
        and services.deterministic_figure_family_supported_for_upstream(
            run_dir, step.step_id
        )
    )
    if repairable_publication_step and preseal_figure_errors:
        repaired = _repair_publication_figure_in_staging(
            run_dir=run_dir,
            current_step_id=step.step_id,
            out_dir=run_result.out_dir,
            renderer=services.render_publication_bundle_from_prior_outputs_for_step,
            step_text=f"{step.intent} {step.method}",
            authorizer=lambda repair_id: _automatic_repair_authorized(
                repair_id,
                step=step,
                source="publication_figure_quality_repair",
            ),
        )
        if repaired is not None:
            worker_progress.runner_repair_name = repaired
            step_record["runner_repair"] = repaired
            _record_repair(
                repair_id=repaired,
                step_id=step.step_id,
                trigger={
                    "source": "publication_figure_quality_repair",
                    "blocked_by": [
                        finding.message for finding in preseal_figure_errors[:5]
                    ],
                },
                transformation=(
                    "Replaced invalid figure-step exports with a deterministic "
                    "publication figure from the registered parent table for "
                    "this step type before evidence sealing."
                ),
            )
            step_summary = _load_step_summary_from_outputs(run_result.out_dir)
            if is_sealed_renderer_repair(repaired):
                step_summary = _write_host_input_binding_receipts(
                    out_dir=run_result.out_dir,
                    step_summary=step_summary,
                    resolved_input_bindings=resolved_input_bindings,
                    consumed_input_keys=tuple(resolved_input_bindings),
                )
            _ensure_step_figure_contract(
                step=step,
                out_dir=run_result.out_dir,
                step_summary=step_summary,
                evidence_ids=[
                    script_record.evidence_id,
                    *lineage_input_evidence_ids,
                ],
            )
            preseal_contract_findings = figure_contract_validator.audit(
                step=step,
                out_dir=run_result.out_dir,
                run_dir=run_dir,
                step_summary=step_summary,
            )
            preseal_source_findings = figure_source_validator.audit(
                step=step,
                out_dir=run_result.out_dir,
                run_dir=run_dir,
                step_summary=step_summary,
                completed_step_records=preseal_completed_records,
                resolved_input_bindings=resolved_input_bindings,
            )

    preseal_figure_errors = [
        finding
        for finding in preseal_contract_findings + preseal_source_findings
        if finding.severity == "error"
    ]
    if preseal_figure_errors:
        preseal_contract_findings = _bind_findings_to_step_attempt(
            preseal_contract_findings,
            step_id=step.step_id,
            attempt_id=attempt_id,
            checkpoint_id=review_checkpoint_id,
        )
        preseal_source_findings = _bind_findings_to_step_attempt(
            preseal_source_findings,
            step_id=step.step_id,
            attempt_id=attempt_id,
            checkpoint_id=review_checkpoint_id,
        )
        step_record.update(
            {
                "status": "contract_failed",
                "diagnostic_only": True,
                "step_summary": step_summary,
                "contract_findings": [
                    finding.model_dump() for finding in preseal_contract_findings
                ],
                "figure_source_findings": [
                    finding.model_dump() for finding in preseal_source_findings
                ],
                "evidence_ids": [script_record.evidence_id],
                "result_evidence_sealed": False,
            }
        )
        with shared_lock:
            findings.extend(preseal_contract_findings)
            findings.extend(preseal_source_findings)
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        emit_progress(
            "contract",
            f"Figure validation failed before evidence sealing for {step.step_id}.",
            status="error",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        return step_record, None
    return None, _StepPostCandidateFigurePreparation(
        figure_role=figure_role,
        lineage_input_evidence_ids=lineage_input_evidence_ids,
        repair_evidence_metadata=repair_evidence_metadata,
        step_summary=step_summary,
    )



def _execute_step(
    step: AnalysisStep,
    *,
    runtime_state: Any,
    _authorize_automatic_repair: Any,
    _automatic_repair_authorized: Any,
    _failed_dependency_record: Any,
    _flush_partial_manifest: Any,
    _propagate_findings_to_evidence: Any,
    _record_repair: Any,
    _validator_messages: Any,
    analyzer: Any,
    clinical_validator: Any,
    coder: Any,
    coder_base_context: Any,
    coder_provider_identity_sha256: Any,
    cohort_path: Any,
    concept_audit_environment_sha256: Any,
    context: Any,
    cross_step_cohort_lock_validator: Any,
    cross_step_reconciliation_trace_validator: Any,
    cross_step_registered_output_validator: Any,
    cross_step_source_status_validator: Any,
    emit_progress: Any,
    evidence: Any,
    figure_contract_validator: Any,
    figure_source_validator: Any,
    findings: Any,
    llm_concept_audit_cache: Any,
    llm_concept_audit_client: Any,
    llm_concept_auditor_identity_sha256: Any,
    llm_concept_auditor_implementation_sha256: Any,
    llm_signature: Any,
    pattern_auditor: Any,
    per_step_records: Any,
    pipeline: Any,
    plan: Any,
    plan_result: Any,
    primary_model_contract_validator: Any,
    prompt_files: Any,
    prompt_version: Any,
    requested_resume_from_step_id: Any,
    resume_controller: Any,
    reuse_selected_step_code_opt_in: Any,
    run_dir: Any,
    run_id: Any,
    run_input_authority_state: Any,
    runner: Any,
    services: Any,
    shared_lock: Any,
    stat_validator: Any,
    statistical_guard: Any,
    step_attempt_history: Any,
    step_evidence_commit: Any,
    step_executor: Any,
    step_order: Any,
    step_summary_fraction_validator: Any,
    step_summary_integrity_validator: Any,
    supervisor: Any,
    total_steps: Any,
    typed_binding_resolver: Any,
    universe_path: Any,
    usage_auditor: Any,
) -> Tuple[Any, Dict[str, Any]]:
    attempt_bootstrap = prepare_step_attempt_bootstrap(
        resume_state=(
            plan_result.resume_state
            if isinstance(plan_result.resume_state, Mapping)
            else None
        ),
        per_step_records=per_step_records,
        shared_lock=shared_lock,
        step=step,
        plan=plan,
        run_id=run_id,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
        plan_scientific_signature=(
            _serializable_plan_scientific_scope_signature(plan)
        ),
        findings=findings,
        max_provider_calls=pipeline._max_step_provider_calls,
        max_llm_repairs=pipeline._max_step_llm_repair_attempts,
        reserve_concept_audit=pipeline._enable_llm_concept_audit,
        allow_terminal_initial_generation_restart=(
            resume_controller.explicitly_reruns_step(step.step_id)
        ),
    )
    prior_attempt_records = attempt_bootstrap.prior_attempt_records
    prior_step_record = attempt_bootstrap.prior_step_record
    attempt_id = attempt_bootstrap.attempt_id
    review_checkpoint_id = attempt_bootstrap.review_checkpoint_id
    step_record = attempt_bootstrap.step_record
    step_execution_cohort_path = attempt_bootstrap.execution_cohort_path
    budget_runtime = attempt_bootstrap.budget_runtime
    provider_budget = budget_runtime.provider_budget
    step_repair_budget = budget_runtime.repair_budget
    provider_receipt_path = budget_runtime.receipt_path
    provider_receipt_relative_path = budget_runtime.receipt_relative_path
    reserved_final_category = budget_runtime.reserved_final_category
    provider_receipt_integrity_error = budget_runtime.integrity_error
    _sync_provider_budget = step_repair_budget.sync_provider
    if provider_receipt_integrity_error is not None:
        step_record.update(
            {
                "status": "contract_failed",
                "generation_mode": "system",
                "provider_call_budget_receipt_invalid": True,
                "provider_call_budget_receipt_error": (
                    provider_receipt_integrity_error
                ),
            }
        )
        receipt_finding = ValidationFinding(
            validator="provider_call_budget_receipt",
            severity="error",
            message=(
                f"Step {step.step_id} cannot resume because its durable "
                "provider-call receipt is missing, corrupt, or inconsistent."
            ),
            detail={
                "step_id": step.step_id,
                "receipt_path": provider_receipt_relative_path,
                "reason": provider_receipt_integrity_error,
            },
        )
        with shared_lock:
            findings.append(receipt_finding)
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        emit_progress(
            "step",
            f"Step {step.step_id} failed closed: provider-call receipt invalid.",
            status="failed",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_order.get(step.step_id, 0) + 1,
            total_steps=total_steps,
        )
        return runtime_state, step_record
    terminal_step_record, preparation = _step_prepare_execution_authority(
        _failed_dependency_record=_failed_dependency_record,
        _flush_partial_manifest=_flush_partial_manifest,
        _sync_provider_budget=_sync_provider_budget,
        coder_base_context=coder_base_context,
        coder_provider_identity_sha256=coder_provider_identity_sha256,
        cohort_path=cohort_path,
        concept_audit_environment_sha256=concept_audit_environment_sha256,
        context=context,
        emit_progress=emit_progress,
        evidence=evidence,
        findings=findings,
        llm_concept_auditor_identity_sha256=llm_concept_auditor_identity_sha256,
        llm_concept_auditor_implementation_sha256=(
            llm_concept_auditor_implementation_sha256
        ),
        per_step_records=per_step_records,
        pipeline=pipeline,
        plan=plan,
        plan_result=plan_result,
        prior_attempt_records=prior_attempt_records,
        prior_step_record=prior_step_record,
        prompt_files=prompt_files,
        prompt_version=prompt_version,
        provider_budget=provider_budget,
        provider_receipt_path=provider_receipt_path,
        requested_resume_from_step_id=requested_resume_from_step_id,
        reserved_final_category=reserved_final_category,
        run_dir=run_dir,
        run_id=run_id,
        run_input_authority_state=run_input_authority_state,
        runtime_state=runtime_state,
        shared_lock=shared_lock,
        step=step,
        step_attempt_history=step_attempt_history,
        step_execution_cohort_path=step_execution_cohort_path,
        step_order=step_order,
        step_record=step_record,
        step_repair_budget=step_repair_budget,
        step_summary_integrity_validator=step_summary_integrity_validator,
        total_steps=total_steps,
        typed_binding_resolver=typed_binding_resolver,
        universe_path=universe_path,
    )
    if terminal_step_record is not None:
        return runtime_state, terminal_step_record
    worker_progress = preparation.worker_progress
    quarantine_state = preparation.quarantine_state
    step_attempt_state = preparation.step_attempt_state
    checkpoint_authority = preparation.checkpoint_authority
    coder_context = preparation.coder_context
    coder_authority = preparation.coder_authority
    step_execution_cohort_path = preparation.step_execution_cohort_path
    plausibility_authority = preparation.plausibility_authority
    resolved_input_bindings = preparation.resolved_input_bindings
    resolved_input_evidence_ids = preparation.resolved_input_evidence_ids
    existing_refs = preparation.existing_refs
    resolved_inputs_path = preparation.resolved_inputs_path
    resolved_inputs_sha256 = preparation.resolved_inputs_sha256
    run_input_capsule_sha256 = preparation.run_input_capsule_sha256
    sealed_renderer_state = preparation.sealed_renderer_state
    sealed_renderer_authorized_code_sha256 = (
        preparation.sealed_renderer_authorized_code_sha256
    )
    monotonic_concept_constraints = preparation.monotonic_concept_constraints
    _llm_repair_budget_available = preparation._llm_repair_budget_available
    _logical_llm_repair_budget_available = (
        preparation._logical_llm_repair_budget_available
    )
    step_current = preparation.step_current

    terminal_step_record, settlement = _step_settle_initial_code(
        _authorize_automatic_repair=_authorize_automatic_repair,
        _flush_partial_manifest=_flush_partial_manifest,
        _record_repair=_record_repair,
        _sync_provider_budget=_sync_provider_budget,
        checkpoint_authority=checkpoint_authority,
        coder=coder,
        coder_authority=coder_authority,
        coder_context=coder_context,
        coder_provider_identity_sha256=coder_provider_identity_sha256,
        context=context,
        emit_progress=emit_progress,
        existing_refs=existing_refs,
        findings=findings,
        monotonic_concept_constraints=monotonic_concept_constraints,
        per_step_records=per_step_records,
        pipeline=pipeline,
        plan=plan,
        plan_result=plan_result,
        plausibility_authority=plausibility_authority,
        prior_attempt_records=prior_attempt_records,
        prior_step_record=prior_step_record,
        prompt_version=prompt_version,
        provider_budget=provider_budget,
        quarantine_state=quarantine_state,
        requested_resume_from_step_id=requested_resume_from_step_id,
        resolved_input_bindings=resolved_input_bindings,
        resolved_inputs_sha256=resolved_inputs_sha256,
        resume_controller=resume_controller,
        reuse_selected_step_code_opt_in=reuse_selected_step_code_opt_in,
        run_dir=run_dir,
        run_id=run_id,
        run_input_capsule_sha256=run_input_capsule_sha256,
        runtime_state=runtime_state,
        sealed_renderer_state=sealed_renderer_state,
        services=services,
        shared_lock=shared_lock,
        step=step,
        step_attempt_state=step_attempt_state,
        step_current=step_current,
        step_record=step_record,
        step_repair_budget=step_repair_budget,
        supervisor=supervisor,
        total_steps=total_steps,
        worker_progress=worker_progress,
    )
    if terminal_step_record is not None:
        return runtime_state, terminal_step_record
    code = settlement.code
    local_runtime_state = settlement.local_runtime_state
    standard_executor = settlement.standard_executor
    _consume_llm_repair_budget = settlement._consume_llm_repair_budget
    _remember_concept_constraints = settlement._remember_concept_constraints
    _quarantine_error_payloads = settlement._quarantine_error_payloads
    _monotonic_concept_constraint_ticket = (
        settlement._monotonic_concept_constraint_ticket
    )
    _monotonic_concept_constraint_log = settlement._monotonic_concept_constraint_log
    _repair_with_capsule = settlement._repair_with_capsule

    terminal_step_record, concept_outcome = _step_run_concept_repair_phase(
        _authorize_automatic_repair=_authorize_automatic_repair,
        _consume_llm_repair_budget=_consume_llm_repair_budget,
        _flush_partial_manifest=_flush_partial_manifest,
        _llm_repair_budget_available=_llm_repair_budget_available,
        _logical_llm_repair_budget_available=_logical_llm_repair_budget_available,
        _monotonic_concept_constraint_ticket=_monotonic_concept_constraint_ticket,
        _quarantine_error_payloads=_quarantine_error_payloads,
        _record_repair=_record_repair,
        _remember_concept_constraints=_remember_concept_constraints,
        _repair_with_capsule=_repair_with_capsule,
        _sync_provider_budget=_sync_provider_budget,
        checkpoint_authority=checkpoint_authority,
        code=code,
        coder_authority=coder_authority,
        coder_context=coder_context,
        concept_audit_environment_sha256=concept_audit_environment_sha256,
        context=context,
        emit_progress=emit_progress,
        findings=findings,
        llm_concept_audit_cache=llm_concept_audit_cache,
        llm_concept_audit_client=llm_concept_audit_client,
        llm_concept_auditor_implementation_sha256=(
            llm_concept_auditor_implementation_sha256
        ),
        pattern_auditor=pattern_auditor,
        per_step_records=per_step_records,
        pipeline=pipeline,
        plan_result=plan_result,
        plausibility_authority=plausibility_authority,
        prior_attempt_records=prior_attempt_records,
        prior_step_record=prior_step_record,
        provider_budget=provider_budget,
        quarantine_state=quarantine_state,
        resolved_input_bindings=resolved_input_bindings,
        run_dir=run_dir,
        run_id=run_id,
        runtime_state=runtime_state,
        sealed_renderer_state=sealed_renderer_state,
        shared_lock=shared_lock,
        standard_executor=standard_executor,
        step=step,
        step_attempt_state=step_attempt_state,
        step_current=step_current,
        step_record=step_record,
        step_repair_budget=step_repair_budget,
        total_steps=total_steps,
        usage_auditor=usage_auditor,
        worker_progress=worker_progress,
    )
    if terminal_step_record is not None:
        return runtime_state, terminal_step_record
    code = concept_outcome.code
    concept_approved_code_digest = concept_outcome.concept_approved_code_digest
    sealed_renderer_authorized_code_sha256 = (
        concept_outcome.sealed_renderer_authorized_code_sha256
    )
    concept_audit = concept_outcome.concept_audit
    _deterministic_fallback_code = concept_outcome._deterministic_fallback_code
    _authorized_deterministic_concept_repair = (
        concept_outcome._authorized_deterministic_concept_repair
    )

    worker_progress.repair_attempts = 0
    worker_progress.contract_repair_attempts = 0
    worker_progress.llm_contract_repair_attempts = 0
    worker_progress.visual_repair_attempts = 0
    # Contract, visual-layout, and runtime failures have independent repair
    # budgets. ``repair_attempts`` remains the total mutation count used for
    # provenance and generation-mode labels.
    worker_progress.runtime_repair_attempts = 0
    worker_progress.runner_repair_name = (
        worker_progress.preexecution_runner_repair_name
    )
    is_trajectory_stability_standard = bool(
        step.trajectory_stability_spec is not None
        and step_record.get("deterministic_standard_analysis")
        == "trajectory_cluster_stability"
    )
    standard_executor_terminal_block = False
    standard_executor_terminal_reason: Optional[str] = None
    standard_executor_terminal_summary: Dict[str, Any] = {}
    standard_executor_terminal_findings: List[ValidationFinding] = []
    deterministic_contract_approved_code_digest: Optional[str] = None
    final_concept_gate_approved_code_digest: Optional[str] = None
    candidate_loop_state = _CandidateLoopState(
        code=code,
        concept_approved_code_digest=concept_approved_code_digest,
        deterministic_contract_approved_code_digest=(
            deterministic_contract_approved_code_digest
        ),
        final_concept_gate_approved_code_digest=(
            final_concept_gate_approved_code_digest
        ),
        standard_executor_terminal_block=standard_executor_terminal_block,
        standard_executor_terminal_reason=standard_executor_terminal_reason,
        standard_executor_terminal_summary=standard_executor_terminal_summary,
        standard_executor_terminal_findings=standard_executor_terminal_findings,
    )
    candidate_loop_terminal = _run_candidate_loop(
        host=_CandidateLoopHost(
            _authorize_automatic_repair=_authorize_automatic_repair,
            _append_terminal_step_record=_append_terminal_step_record,
            _automatic_repair_authorized=_automatic_repair_authorized,
            _contract_repair_log=_contract_repair_log,
            _execution_input_authority_integrity_finding=(
                _execution_input_authority_integrity_finding
            ),
            _flush_partial_manifest=_flush_partial_manifest,
            _fresh_plausibility_receipt_findings=(
                _fresh_plausibility_receipt_findings
            ),
            _locked_measurement_data_quality_issues=(
                _locked_measurement_data_quality_issues
            ),
            _python_repair_is_materially_changed=(
                _python_repair_is_materially_changed
            ),
            _record_repair=_record_repair,
            _remove_standard_executor_pending_artifacts=(
                _remove_standard_executor_pending_artifacts
            ),
            _unowned_sealed_authority_markers=(_unowned_sealed_authority_markers),
            cohort_path=cohort_path,
            concept_audit_environment_sha256=concept_audit_environment_sha256,
            context=context,
            cross_step_cohort_lock_validator=cross_step_cohort_lock_validator,
            cross_step_reconciliation_trace_validator=(
                cross_step_reconciliation_trace_validator
            ),
            cross_step_registered_output_validator=(
                cross_step_registered_output_validator
            ),
            cross_step_source_status_validator=cross_step_source_status_validator,
            emit_progress=emit_progress,
            evidence=evidence,
            figure_contract_validator=figure_contract_validator,
            figure_source_validator=figure_source_validator,
            findings=findings,
            llm_concept_auditor_identity_sha256=(
                llm_concept_auditor_identity_sha256
            ),
            llm_concept_auditor_implementation_sha256=(
                llm_concept_auditor_implementation_sha256
            ),
            llm_signature=llm_signature,
            per_step_records=per_step_records,
            pipeline=pipeline,
            plan=plan,
            primary_model_contract_validator=primary_model_contract_validator,
            prompt_version=prompt_version,
            run_dir=run_dir,
            run_id=run_id,
            run_input_authority_state=run_input_authority_state,
            runner=runner,
            shared_lock=shared_lock,
            step_executor=step_executor,
            step_summary_fraction_validator=step_summary_fraction_validator,
            step_summary_integrity_validator=step_summary_integrity_validator,
            total_steps=total_steps,
            universe_path=universe_path,
        ),
        attempt=_CandidateLoopAttempt(
            _authorized_deterministic_concept_repair=(
                _authorized_deterministic_concept_repair
            ),
            _consume_llm_repair_budget=_consume_llm_repair_budget,
            _deterministic_fallback_code=_deterministic_fallback_code,
            _llm_repair_budget_available=_llm_repair_budget_available,
            _logical_llm_repair_budget_available=(
                _logical_llm_repair_budget_available
            ),
            _monotonic_concept_constraint_log=_monotonic_concept_constraint_log,
            _monotonic_concept_constraint_ticket=(
                _monotonic_concept_constraint_ticket
            ),
            _quarantine_error_payloads=_quarantine_error_payloads,
            _remember_concept_constraints=_remember_concept_constraints,
            _repair_with_capsule=_repair_with_capsule,
            _sync_provider_budget=_sync_provider_budget,
            checkpoint_authority=checkpoint_authority,
            coder_context=coder_context,
            concept_audit=concept_audit,
            is_trajectory_stability_standard=is_trajectory_stability_standard,
            local_runtime_state=local_runtime_state,
            plausibility_authority=plausibility_authority,
            provider_budget=provider_budget,
            quarantine_state=quarantine_state,
            resolved_input_bindings=resolved_input_bindings,
            resolved_input_evidence_ids=resolved_input_evidence_ids,
            resolved_inputs_path=resolved_inputs_path,
            resolved_inputs_sha256=resolved_inputs_sha256,
            sealed_renderer_authorized_code_sha256=(
                sealed_renderer_authorized_code_sha256
            ),
            sealed_renderer_state=sealed_renderer_state,
            standard_executor=standard_executor,
            step=step,
            step_attempt_state=step_attempt_state,
            step_current=step_current,
            step_execution_cohort_path=step_execution_cohort_path,
            step_record=step_record,
            step_repair_budget=step_repair_budget,
            worker_progress=worker_progress,
        ),
        state=candidate_loop_state,
    )
    if candidate_loop_terminal:
        return runtime_state, step_record
    code = candidate_loop_state.code
    concept_approved_code_digest = candidate_loop_state.concept_approved_code_digest
    deterministic_contract_approved_code_digest = (
        candidate_loop_state.deterministic_contract_approved_code_digest
    )
    final_concept_gate_approved_code_digest = (
        candidate_loop_state.final_concept_gate_approved_code_digest
    )
    usage_findings = candidate_loop_state.usage_findings
    run_result = candidate_loop_state.run_result
    executed_code_digest = candidate_loop_state.executed_code_digest
    script_record = candidate_loop_state.script_record
    standard_executor_terminal_block = (
        candidate_loop_state.standard_executor_terminal_block
    )
    standard_executor_terminal_reason = (
        candidate_loop_state.standard_executor_terminal_reason
    )
    standard_executor_terminal_summary = (
        candidate_loop_state.standard_executor_terminal_summary
    )
    standard_executor_terminal_findings = (
        candidate_loop_state.standard_executor_terminal_findings
    )
    terminal_step_record, figure_preparation = _step_prepare_post_candidate_figures(
        _automatic_repair_authorized=_automatic_repair_authorized,
        _flush_partial_manifest=_flush_partial_manifest,
        _record_repair=_record_repair,
        attempt_id=attempt_id,
        emit_progress=emit_progress,
        figure_contract_validator=figure_contract_validator,
        figure_source_validator=figure_source_validator,
        findings=findings,
        per_step_records=per_step_records,
        plan=plan,
        resolved_input_bindings=resolved_input_bindings,
        resolved_input_evidence_ids=resolved_input_evidence_ids,
        review_checkpoint_id=review_checkpoint_id,
        run_dir=run_dir,
        run_id=run_id,
        run_result=run_result,
        runtime_state=runtime_state,
        script_record=script_record,
        sealed_renderer_authorized_code_sha256=(
            sealed_renderer_authorized_code_sha256
        ),
        services=services,
        shared_lock=shared_lock,
        standard_executor=standard_executor,
        standard_executor_terminal_block=standard_executor_terminal_block,
        step=step,
        step_current=step_current,
        step_record=step_record,
        total_steps=total_steps,
        worker_progress=worker_progress,
    )
    if terminal_step_record is not None:
        return runtime_state, terminal_step_record
    figure_role = figure_preparation.figure_role
    lineage_input_evidence_ids = figure_preparation.lineage_input_evidence_ids
    repair_evidence_metadata = figure_preparation.repair_evidence_metadata
    step_summary = figure_preparation.step_summary

    # This is the seal boundary.  From here onward result artifacts are
    # immutable: validation may fail closed, but no repair can mutate them.
    if run_result.outputs_safe_to_collect:
        run_result.artefacts = sorted(
            path
            for path in run_result.out_dir.iterdir()
            if path.is_file()
            and not (
                worker_progress.deterministic_standard_executor_used
                and _is_standard_executor_internal_artifact(path)
            )
        )
    sealed_result_digests = {
        path.name: sha256_of_file(path) for path in run_result.artefacts
    }
    step_record["result_seal_sha256"] = sha256_of_bytes(
        json.dumps(
            sealed_result_digests,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    step_record["result_evidence_sealed"] = True

    evidence_ids_for_step: List[str] = [script_record.evidence_id]
    pending_success_aliases: Dict[str, List[str]] = {}
    step_summary_record_id: Optional[str] = None
    (
        step_summary_record_id,
        pending_success_aliases,
        evidence_ids_for_step,
    ) = _step_register_run_artifacts(
        step=step,
        step_summary=step_summary,
        run_result=run_result,
        worker_progress=worker_progress,
        services=services,
        sealed_result_digests=sealed_result_digests,
        script_record=script_record,
        evidence=evidence,
        lineage_input_evidence_ids=lineage_input_evidence_ids,
        figure_role=figure_role,
        standard_executor_terminal_block=standard_executor_terminal_block,
        repair_evidence_metadata=repair_evidence_metadata,
        evidence_ids_for_step=evidence_ids_for_step,
        pending_success_aliases=pending_success_aliases,
    )
    if step_summary_record_id is not None:
        step_record["step_summary_evidence_id"] = step_summary_record_id

    return _step_finalize_step(
        _flush_partial_manifest=_flush_partial_manifest,
        _propagate_findings_to_evidence=_propagate_findings_to_evidence,
        _sync_provider_budget=_sync_provider_budget,
        _validator_messages=_validator_messages,
        analyzer=analyzer,
        attempt_id=attempt_id,
        clinical_validator=clinical_validator,
        code=code,
        cohort_path=cohort_path,
        concept_approved_code_digest=concept_approved_code_digest,
        concept_audit=concept_audit,
        context=context,
        cross_step_cohort_lock_validator=cross_step_cohort_lock_validator,
        cross_step_reconciliation_trace_validator=cross_step_reconciliation_trace_validator,
        cross_step_registered_output_validator=cross_step_registered_output_validator,
        cross_step_source_status_validator=cross_step_source_status_validator,
        emit_progress=emit_progress,
        evidence=evidence,
        evidence_ids_for_step=evidence_ids_for_step,
        executed_code_digest=executed_code_digest,
        figure_contract_validator=figure_contract_validator,
        figure_source_validator=figure_source_validator,
        final_concept_gate_approved_code_digest=final_concept_gate_approved_code_digest,
        findings=findings,
        local_runtime_state=local_runtime_state,
        pending_success_aliases=pending_success_aliases,
        per_step_records=per_step_records,
        pipeline=pipeline,
        plan=plan,
        plan_result=plan_result,
        plausibility_authority=plausibility_authority,
        primary_model_contract_validator=primary_model_contract_validator,
        prompt_version=prompt_version,
        provider_budget=provider_budget,
        quarantine_state=quarantine_state,
        resolved_input_bindings=resolved_input_bindings,
        runtime_state=runtime_state,
        review_checkpoint_id=review_checkpoint_id,
        run_dir=run_dir,
        run_id=run_id,
        run_result=run_result,
        script_record=script_record,
        sealed_result_digests=sealed_result_digests,
        shared_lock=shared_lock,
        standard_executor_terminal_block=standard_executor_terminal_block,
        standard_executor_terminal_findings=standard_executor_terminal_findings,
        standard_executor_terminal_reason=standard_executor_terminal_reason,
        standard_executor_terminal_summary=standard_executor_terminal_summary,
        stat_validator=stat_validator,
        statistical_guard=statistical_guard,
        step=step,
        step_current=step_current,
        step_evidence_commit=step_evidence_commit,
        step_record=step_record,
        step_summary=step_summary,
        step_summary_fraction_validator=step_summary_fraction_validator,
        step_summary_integrity_validator=step_summary_integrity_validator,
        step_summary_record_id=step_summary_record_id,
        supervisor=supervisor,
        total_steps=total_steps,
        typed_binding_resolver=typed_binding_resolver,
        universe_path=universe_path,
        usage_findings=usage_findings,
        worker_progress=worker_progress,
    )



def _step_maybe_directed_model_replan(
    *,
    failed_step: AnalysisStep,
    failed_record: Dict[str, Any],
    pipeline: Any,
    _replan_state: Dict[str, Any],
    cohort_path: Path,
    per_step_records: List[Dict[str, Any]],
    findings: List[ValidationFinding],
    _maybe_replan: Any,
    plan: AnalysisPlan,
    probe_summary: Optional[Dict[str, Any]],
) -> Optional[AnalysisPlan]:
    """Fire a forced, directive-carrying replan when a model/estimation
    step self-blocks on a task-viable cohort, else return ``None``.
    """

    if not pipeline._enable_replanning:
        return None
    if failed_record.get("status") == "ok":
        return None
    if _replan_state["directed_model_replans"] >= _MAX_DIRECTED_MODEL_REPLANS:
        return None
    if not step_requires_model_performance(failed_step.expected_outputs):
        return None
    try:
        import pandas as pd  # lazy: only on the rare self-block path

        viability = assess_cohort_viability(
            pd.read_parquet(cohort_path), outcome=None
        )
    except Exception:
        return None
    directive = build_self_block_replan_directive(
        failed_step=failed_step,
        failed_record=failed_record,
        completed_records=per_step_records,
        viability=viability,
    )
    if directive is None:
        return None
    _replan_state["directed_model_replans"] += 1
    findings.append(
        ValidationFinding(
            validator="replanner",
            severity="warning",
            message=(
                "Directed replan: modeling step "
                f"{failed_step.step_id} self-blocked on a task-viable "
                f"cohort ({viability.note}); issued a viability-conditioned "
                "override to fit the model rather than register a block."
            ),
            detail={
                "step_id": failed_step.step_id,
                "directed_model_replans": _replan_state[
                    "directed_model_replans"
                ],
            },
        )
    )
    return _maybe_replan(
        current_plan=plan,
        reason=f"{failed_step.step_id}:self_inflicted_block_on_viable_cohort",
        probe_summary_payload=probe_summary,
        completed_records=per_step_records,
        directive=directive,
        force=True,
    )


def _step_resolve_run_transition(
    step: AnalysisStep,
    record: Dict[str, Any],
    has_remaining: bool,
    *,
    run_input_authority_state: Any,
    emit_progress: Any,
    run_id: str,
    requested_stop_after_step_id: Optional[str],
    _maybe_directed_model_replan: Any,
    _replan_state: Dict[str, Any],
    pipeline: Any,
    _maybe_replan: Any,
    plan: AnalysisPlan,
    probe_summary: Optional[Dict[str, Any]],
    per_step_records: List[Dict[str, Any]],
) -> RunTransition:
    if run_input_authority_state.corrupted:
        emit_progress(
            "audit",
            "Stopped the run after execution input authority corruption.",
            status="error",
            run_id=run_id,
            step_id=str(run_input_authority_state.step_id or ""),
        )
        return RunTransition.stop("input_authority_corrupted")
    if step.step_id == requested_stop_after_step_id:
        emit_progress(
            "pause",
            f"Stopped after requested step: {step.step_id}.",
            status="paused",
            run_id=run_id,
            step_id=step.step_id,
        )
        return RunTransition.stop("requested_stop_after_step")
    directed_plan = _maybe_directed_model_replan(
        failed_step=step, failed_record=record
    )
    if pause_transition := replan_review.runtime_replan_pause_transition(_replan_state):
        return pause_transition
    if directed_plan is not None:
        return RunTransition.replan(
            directed_plan,
            rerun_current_step=True,
        )
    if _successful_run_transition_requests_replan(pipeline, record, has_remaining):
        revised_plan = _maybe_replan(
            current_plan=plan,
            reason=step.step_id,
            probe_summary_payload=probe_summary,
            completed_records=per_step_records,
        )
        if pause_transition := replan_review.runtime_replan_pause_transition(_replan_state):
            return pause_transition
        return RunTransition.replan(revised_plan)
    return RunTransition.continue_run()


def _step_apply_revised_plan(
    *,
    revised_plan: AnalysisPlan,
    executed_step_ids: set[str],
    plan: AnalysisPlan,
    step_order: Dict[str, int],
    resume_controller: Any,
    total_steps: int,
) -> Tuple[Sequence[AnalysisStep], AnalysisPlan, int]:
    plan = revised_plan
    step_order.clear()
    step_order.update({s.step_id: i for i, s in enumerate(plan.steps)})
    remaining = resume_controller.remaining_steps(
        plan=plan,
        executed_step_ids=executed_step_ids,
    )
    total_steps = len(plan.steps)
    return remaining, plan, total_steps


def _step_record_step_exception(
    step: AnalysisStep,
    error: BaseException,
    *,
    shared_lock: Any,
    findings: List[ValidationFinding],
    _flush_partial_manifest: Any,
    per_step_records: List[Dict[str, Any]],
    _append_terminal_step_record: Any,
) -> None:
    """Seal a terminal record for a step that raised instead of returning."""

    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        with shared_lock:
            findings.append(
                ValidationFinding(
                    validator="run_interrupted",
                    severity="error",
                    message=(
                        f"The run was interrupted by "
                        f"{type(error).__name__} while step "
                        f"{step.step_id} was in flight; the step's own "
                        "in-flight record is kept so a resume can pick "
                        "it up, and no later step ran."
                    ),
                    detail={
                        "reason": "operator_interrupt",
                        "step_id": step.step_id,
                        "error_type": type(error).__name__,
                    },
                )
            )
            _flush_partial_manifest({"run_interrupted": step.step_id})
        return

    detail = f"{type(error).__name__}: {error}".strip()
    superseded = next(
        (
            record
            for record in reversed(per_step_records)
            if isinstance(record, Mapping)
            and str(record.get("step_id") or "") == step.step_id
        ),
        None,
    )
    carried_coordinates: Dict[str, Any] = {}
    if superseded is not None:
        for field_name in RESUMABLE_ATTEMPT_COORDINATE_FIELDS:
            value = superseded.get(field_name)
            if value is not None:
                carried_coordinates[field_name] = value
    crash_record: Dict[str, Any] = {
        "step_id": step.step_id,
        "status": "execution_raised",
        "error": detail,
        "error_type": type(error).__name__,
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        **carried_coordinates,
    }
    with shared_lock:
        findings.append(
            ValidationFinding(
                validator="step_execution_exception",
                severity="error",
                message=(
                    f"Step {step.step_id} raised {detail} instead of "
                    "returning a step record, so the run stopped "
                    "fail-closed before any later step."
                ),
                detail={
                    "reason": "step_execution_raised",
                    "step_id": step.step_id,
                    "error_type": type(error).__name__,
                },
            )
        )
        _append_terminal_step_record(per_step_records, crash_record)
        _flush_partial_manifest({"step_execution_raised": step.step_id})


def _step_record_parallel_worker_error(
    exc: BaseException,
    *,
    shared_lock: Any,
    findings: List[ValidationFinding],
) -> None:
    with shared_lock:
        findings.append(
            ValidationFinding(
                validator="step_executor",
                severity="error",
                message=f"Worker raised an unhandled exception: {exc!r}",
            )
        )


def _step_build_probe_summary_and_record(
    *,
    services: Any,
    context: ResearchContext,
    cohort_path: Path,
    run_dir: Path,
    probe_step_id: str,
    evidence: Any,
    per_step_records: List[Dict[str, Any]],
    preexecuted_step_ids: set[str],
    _flush_partial_manifest: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Build the deterministic probe and seal it as the first step record."""

    probe_summary, probe_files = services.build_probe_summary(
        context=context,
        cohort_path=cohort_path,
        out_dir=run_dir / "steps" / probe_step_id / "outputs",
    )
    probe_evidence_ids: List[str] = []
    probe_authority_fields: Dict[str, str] = {}
    for probe_file in probe_files:
        kind = "statistic" if probe_file.name.endswith(".json") else "table"
        aliases = [probe_step_id]
        if probe_file.name == "probe_summary.json":
            aliases.extend(
                [
                    "probe_summary",
                    "cohort_probe",
                ]
            )
        rec = evidence.register_file(
            kind=kind,
            description=f"Probe artefact {probe_file.name}.",
            source_path=probe_file,
            produced_by_step=probe_step_id,
            producer="pipeline",
            generation_mode="deterministic_probe",
            aliases=aliases,
        )
        probe_evidence_ids.append(rec.evidence_id)
        for field, (_kind, source_name) in _HOST_PROBE_AUTHORITIES.items():
            if probe_file.name == source_name:
                probe_authority_fields[field] = rec.evidence_id
    missing_probe_authorities = sorted(
        set(_HOST_PROBE_AUTHORITIES) - set(probe_authority_fields)
    )
    if missing_probe_authorities:
        raise RuntimeError(
            "Host probe did not produce its required authority fields: "
            + ", ".join(missing_probe_authorities)
        )
    probe_record = {
        "step_id": probe_step_id,
        "intent": "Probe distributions, missingness, and obvious anomalies before execution.",
        "planned_analysis_role": "auxiliary",
        "status": "ok",
        "generation_mode": "deterministic_probe",
        "step_summary": probe_summary,
        "evidence_ids": probe_evidence_ids,
        "step_authority_kind": _HOST_PROBE_AUTHORITY_KIND,
        **probe_authority_fields,
        **_deterministic_gate_stamp(),
    }
    per_step_records.append(probe_record)
    preexecuted_step_ids.add(probe_step_id)
    _flush_partial_manifest()
    return probe_summary, probe_record


def _step_build_replan_directives(
    *,
    typed_plan_preflight: List[ValidationFinding],
    primary_cohort_preflight: List[ValidationFinding],
    trajectory_preflight: List[ValidationFinding],
    declared_input_preflight: List[ValidationFinding],
    owner_declaration_preflight: List[ValidationFinding],
    product_promise_preflight: List[ValidationFinding],
    endpoint_preflight: List[ValidationFinding],
) -> Dict[str, Optional[str]]:
    """Build case-neutral replan directives from the plan preflight gates."""

    trajectory_directive = None
    typed_plan_directive = None
    declared_input_directive = None
    if typed_plan_preflight:
        typed_plan_directive = (
            "Repair the plan's declared typed product DAG without changing "
            "its scientific choices. Every typed kind:product input must "
            "have exactly one declared producer, every required producer "
            "must remain in the plan, and producers must precede consumers. "
            "Do not invent an exposure, outcome, cohort, estimator, or "
            "analysis method. Contract findings: "
            + json.dumps(
                [
                    {
                        "message": finding.message,
                        "detail": finding.detail,
                    }
                    for finding in typed_plan_preflight
                ],
                ensure_ascii=False,
                default=str,
            )
        )
    primary_cohort_directive = None
    if primary_cohort_preflight:
        primary_cohort_directive = (
            "Repair the plan's primary-cohort typed-product ownership "
            "without changing any scientific choice. A cohort construction + "
            "attrition step must uniquely own exactly one materialised product: "
            "`artifact|dataset|table:analysis_cohort`, `cohort:analysis_set`, "
            "or `cohort:<exact cohort.name>`. Definition/protocol/status artifacts "
            "are not cohort datasets. Contract findings: "
            + json.dumps(
                [
                    {
                        "message": finding.message,
                        "detail": finding.detail,
                    }
                    for finding in primary_cohort_preflight
                ],
                ensure_ascii=False,
                default=str,
            )
        )
    if trajectory_preflight:
        trajectory_directive = (
            "Repair the agent-declared fixed-window trajectory plan DAG "
            "without changing its scientific choices. Preserve legitimate "
            "representation, candidate-selection, stability/freeze, and "
            "characterization step boundaries; repair only missing/ambiguous "
            "typed artifact edges, role declarations, and silent internal "
            "window-grid omissions. Do not choose a clustering method, k, "
            "eligibility threshold, or deterministic runner. Contract findings: "
            + json.dumps(
                [
                    {
                        "message": finding.message,
                        "detail": finding.detail,
                    }
                    for finding in trajectory_preflight
                ],
                ensure_ascii=False,
                default=str,
            )
        )
    if declared_input_preflight:
        declared_input_directive = (
            "Repair the plan's declared raw column inputs without changing "
            "any scientific choice. Every declared raw input must be a "
            "column the sealed research context carries; a value a previous "
            "step derives must be declared as that step's typed product "
            "instead. Do not invent an exposure, outcome, cohort, covariate, "
            "or method to satisfy this. Contract findings: "
            + json.dumps(
                [
                    {
                        "message": finding.message,
                        "detail": finding.detail,
                    }
                    for finding in declared_input_preflight
                ],
                ensure_ascii=False,
                default=str,
            )
        )
    owner_declaration_directive = owner_declaration_replan_directive(
        owner_declaration_preflight
    )
    product_promise_directive = product_promise_replan_directive(
        product_promise_preflight
    )
    endpoint_directive = None
    if (
        endpoint_preflight
        and (endpoint_preflight[0].detail or {}).get("reason")
        == "endpoint_projection_mismatch"
    ):
        endpoint_directive = (
            "Repair or remove the plan's stale endpoint projection without "
            "changing the sealed ResearchContext endpoint or any other science. "
            "The projection may equal the context exactly; it is not a second "
            "authority. Contract findings: "
            + json.dumps(
                [
                    {"message": finding.message, "detail": finding.detail}
                    for finding in endpoint_preflight
                ],
                ensure_ascii=False,
                default=str,
            )
        )
    return {
        "typed_plan_directive": typed_plan_directive,
        "primary_cohort_directive": primary_cohort_directive,
        "trajectory_directive": trajectory_directive,
        "declared_input_directive": declared_input_directive,
        "owner_declaration_directive": owner_declaration_directive,
        "product_promise_directive": product_promise_directive,
        "endpoint_directive": endpoint_directive,
    }


def _step_run_initial_replan(
    *,
    _maybe_replan: Any,
    plan: AnalysisPlan,
    probe_record: Optional[Dict[str, Any]],
    probe_summary: Optional[Dict[str, Any]],
    per_step_records: List[Dict[str, Any]],
    endpoint_directive: Optional[str],
    typed_plan_directive: Optional[str],
    primary_cohort_directive: Optional[str],
    trajectory_directive: Optional[str],
    declared_input_directive: Optional[str],
    product_promise_directive: Optional[str],
    owner_declaration_directive: Optional[str],
    typed_plan_preflight: List[ValidationFinding],
    endpoint_preflight: List[ValidationFinding],
    primary_cohort_preflight: List[ValidationFinding],
    trajectory_preflight: List[ValidationFinding],
    declared_input_preflight: List[ValidationFinding],
    product_promise_preflight: List[ValidationFinding],
    owner_declaration_preflight: List[ValidationFinding],
) -> AnalysisPlan:
    """Run the probe/contract-preflight replan once and return the plan."""

    return _maybe_replan(
        current_plan=plan,
        reason=(
            "probe_summary" if probe_record is not None else "plan_contract_preflight"
        ),
        probe_summary_payload=probe_summary,
        completed_records=per_step_records,
        directive="\n\n".join(
            directive
            for directive in (
                endpoint_directive,
                typed_plan_directive,
                primary_cohort_directive,
                trajectory_directive,
                declared_input_directive,
                product_promise_directive,
                owner_declaration_directive,
            )
            if directive
        )
        or None,
        force=bool(
            typed_plan_preflight
            or endpoint_directive
            or primary_cohort_preflight
            or trajectory_preflight
            or declared_input_preflight
            or product_promise_preflight
            or owner_declaration_preflight
        ),
    )


def _step_emit_execution_preflight(
    *,
    plan_block_reason: Optional[str],
    plan: AnalysisPlan,
    preexecuted_step_ids: set[str],
    emit_progress: Any,
    run_id: str,
    pipeline: Any,
    steps_to_run: Sequence[AnalysisStep],
    findings: List[ValidationFinding],
    resumed_step_ids: set[str],
) -> Tuple[bool, bool]:
    """Emit run-level preflight progress/findings and return dependency flags."""

    if plan_block_reason is not None:
        dropped = [
            step.step_id
            for step in plan.steps
            if step.step_id not in preexecuted_step_ids
        ]
        emit_progress(
            "step",
            f"Plan blocked before execution ({plan_block_reason}); "
            f"{len(dropped)} planned step(s) will not run.",
            status="blocked",
            run_id=run_id,
            block_reason=plan_block_reason,
            dropped_step_ids=dropped,
            planned_step_count=len(plan.steps),
        )
    has_typed_input_dependencies = any(
        _typed_input_product(input_name) is not None
        for step in steps_to_run
        for input_name in (step.inputs or [])
    )
    has_primary_cohort_universe_producer = any(
        primary_analysis_cohort_producer_uses_universe(step=step, plan=plan)
        for step in steps_to_run
    )
    for skipped_step_id in sorted(resumed_step_ids):
        emit_progress(
            "resume",
            f"Skipped completed step from prior run: {skipped_step_id}.",
            status="complete",
            run_id=run_id,
            step_id=skipped_step_id,
        )
    for skipped_step_id in sorted(preexecuted_step_ids - resumed_step_ids):
        emit_progress(
            "step",
            f"Skipped step already completed by pre-execution: {skipped_step_id}.",
            status="complete",
            run_id=run_id,
            step_id=skipped_step_id,
        )
    if pipeline._enable_replanning and pipeline._max_concurrent_steps > 1:
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="info",
                message=(
                    "Replanning is enabled, so step execution was forced to sequential "
                    "mode to preserve run-internal plan revisions."
                ),
            )
        )
    elif has_typed_input_dependencies and pipeline._max_concurrent_steps > 1:
        findings.append(
            ValidationFinding(
                validator="typed_artifact_evidence_lineage",
                severity="info",
                message=(
                    "Typed product dependencies are present, so step execution "
                    "was forced to plan order before resolving producer evidence."
                ),
            )
        )
    elif has_primary_cohort_universe_producer and pipeline._max_concurrent_steps > 1:
        findings.append(
            ValidationFinding(
                validator="execution_input_authority_integrity",
                severity="info",
                message=(
                    "A primary analysis-cohort producer requires raw-universe "
                    "authority, so step execution was forced to plan order."
                ),
            )
        )
    return has_typed_input_dependencies, has_primary_cohort_universe_producer


def _step_audit_final_figures(
    *,
    pipeline: Any,
    requested_stop_after_step_id: Optional[str],
    emit_progress: Any,
    run_dir: Path,
    run_id: str,
    evidence: Any,
    role_resolver: Any,
) -> List[ValidationFinding]:
    """Run the optional final visual-QA pass and return its findings."""

    if pipeline._enable_visual_qa and requested_stop_after_step_id is None:
        emit_progress(
            "visual_qa",
            "Auditing generated figures.",
            run_id=run_id,
        )
        fig_paths = [
            run_dir / r.relative_path for r in evidence.records() if r.kind == "figure"
        ]
        vlm_adapter = pipeline._visual_qa_adapter
        if vlm_adapter is None and pipeline._enable_vlm_visual_qa:
            client = budgeted_vlm_client(pipeline, role_resolver, "vlm_visual_qa")
            if client is not None:
                vlm_adapter = VLMVisualQAAdapter(
                    client,
                    egress_policy=pipeline._figure_egress_policy(
                        evidence=evidence, run_dir=run_dir
                    ),
                )
        final_visual_findings = VisualQAAuditor(vlm_adapter=vlm_adapter).audit(
            figure_paths=fig_paths
        )
        demoted_final_findings, _ = _demote_cosmetic_visual_findings(
            final_visual_findings
        )
        return demoted_final_findings
    return []


def _step_finalize_step(
    *,
    _flush_partial_manifest: Any,
    _propagate_findings_to_evidence: Any,
    _sync_provider_budget: Any,
    _validator_messages: Any,
    analyzer: Any,
    attempt_id: Any,
    clinical_validator: Any,
    code: Any,
    cohort_path: Any,
    concept_approved_code_digest: Any,
    concept_audit: Any,
    context: Any,
    cross_step_cohort_lock_validator: Any,
    cross_step_reconciliation_trace_validator: Any,
    cross_step_registered_output_validator: Any,
    cross_step_source_status_validator: Any,
    emit_progress: Any,
    evidence: Any,
    evidence_ids_for_step: Any,
    executed_code_digest: Any,
    figure_contract_validator: Any,
    figure_source_validator: Any,
    final_concept_gate_approved_code_digest: Any,
    findings: Any,
    local_runtime_state: Any,
    pending_success_aliases: Any,
    per_step_records: Any,
    pipeline: Any,
    plan: Any,
    plan_result: Any,
    plausibility_authority: Any,
    primary_model_contract_validator: Any,
    prompt_version: Any,
    provider_budget: Any,
    quarantine_state: Any,
    resolved_input_bindings: Any,
    review_checkpoint_id: Any,
    run_dir: Any,
    run_id: Any,
    run_result: Any,
    runtime_state: Any,
    script_record: Any,
    sealed_result_digests: Any,
    shared_lock: Any,
    standard_executor_terminal_block: Any,
    standard_executor_terminal_findings: Any,
    standard_executor_terminal_reason: Any,
    standard_executor_terminal_summary: Any,
    stat_validator: Any,
    statistical_guard: Any,
    step: Any,
    step_current: Any,
    step_evidence_commit: Any,
    step_record: Any,
    step_summary: Any,
    step_summary_fraction_validator: Any,
    step_summary_integrity_validator: Any,
    step_summary_record_id: Any,
    supervisor: Any,
    total_steps: Any,
    typed_binding_resolver: Any,
    universe_path: Any,
    usage_findings: Any,
    worker_progress: Any,
) -> Tuple[Any, Dict[str, Any]]:
    _register_current_step_numeric_claims = functools.partial(
        _step_register_current_step_numeric_claims,
        step=step,
        step_summary=step_summary,
        step_summary_record_id=step_summary_record_id,
        standard_executor_terminal_block=standard_executor_terminal_block,
        evidence=evidence,
        findings=findings,
        pipeline=pipeline,
    )
    if standard_executor_terminal_block:
        terminal_summary = (
            step_summary if step_summary else standard_executor_terminal_summary
        )
        terminal_finding = standard_executor_failure_finding(
            step_record=step_record,
            step_id=step.step_id,
            reason=standard_executor_terminal_reason,
            failure_phase="execution_or_output_validation",
            executor_errors=(
                terminal_summary.get("errors")
                if isinstance(terminal_summary, Mapping)
                else None
            ),
        )
        terminal_findings = [
            terminal_finding,
            *standard_executor_terminal_findings,
        ]
        evidence_ids_for_step = list(dict.fromkeys(evidence_ids_for_step))
        step_record.update(
            {
                "status": "deterministic_standard_blocked",
                "diagnostic_only": True,
                "standard_executor_terminal_reason": (
                    standard_executor_terminal_reason
                ),
                "step_summary": terminal_summary,
                "contract_findings": [
                    finding.model_dump() for finding in terminal_findings
                ],
                "evidence_ids": evidence_ids_for_step,
                "llm_repair_used": False,
                "generation_mode": worker_progress.generation_mode(
                    llm_repair_used=False
                ),
            }
        )
        with shared_lock:
            findings.extend(terminal_findings)
            _append_terminal_step_record(per_step_records, step_record)
            _flush_partial_manifest()
        emit_progress(
            "runner",
            (
                "Deterministic standard executor "
                f"{step_record.get('deterministic_standard_analysis')!r} "
                f"failed closed for {step.step_id}."
            ),
            status="error",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
            reason=standard_executor_terminal_reason,
        )
        return runtime_state, step_record
    with shared_lock:
        completed_records_snapshot = list(per_step_records)
    final_gate_findings = _evaluate_final_deterministic_gates(
        context=context,
        plan=plan,
        cohort_path=cohort_path,
        universe_path=universe_path,
        run_dir=run_dir,
        out_dir=run_result.out_dir,
        step=step,
        step_summary=step_summary,
        step_record=step_record,
        completed_step_records=completed_records_snapshot,
        resolved_input_bindings=resolved_input_bindings,
        plausibility_scope=plausibility_authority.scope,
        script_text=code,
        attempt_id=attempt_id, checkpoint_id=review_checkpoint_id,
        evidence_store=evidence,
        stat_validator=stat_validator,
        clinical_validator=clinical_validator,
        statistical_guard=statistical_guard,
        cross_step_cohort_lock_validator=cross_step_cohort_lock_validator,
        cross_step_registered_output_validator=(
            cross_step_registered_output_validator
        ),
        cross_step_reconciliation_trace_validator=(
            cross_step_reconciliation_trace_validator
        ),
        step_summary_integrity_validator=step_summary_integrity_validator,
        step_summary_fraction_validator=step_summary_fraction_validator,
        cross_step_source_status_validator=cross_step_source_status_validator,
        primary_model_contract_validator=primary_model_contract_validator,
        figure_contract_validator=figure_contract_validator,
        figure_source_validator=figure_source_validator,
    )
    stat_findings = list(final_gate_findings.stat_findings)
    clinical_findings = list(final_gate_findings.clinical_findings)
    guard_findings = list(final_gate_findings.guard_findings)
    contract_findings = list(final_gate_findings.contract_findings)
    figure_source_findings = list(final_gate_findings.figure_source_findings)
    with shared_lock:
        findings.extend(stat_findings)
        findings.extend(clinical_findings)
        findings.extend(guard_findings)
        findings.extend(contract_findings)
        findings.extend(figure_source_findings)
    step_record["stat_findings"] = [f.model_dump() for f in stat_findings]
    step_record["clinical_findings"] = [f.model_dump() for f in clinical_findings]
    step_record["guard_findings"] = [f.model_dump() for f in guard_findings]
    step_record["contract_findings"] = [f.model_dump() for f in contract_findings]
    step_record["figure_source_findings"] = [
        f.model_dump() for f in figure_source_findings
    ]
    step_record["llm_repair_used"] = worker_progress.llm_repair_used
    step_record["generation_mode"] = worker_progress.generation_mode()
    raw_side_findings = step_summary.get("side_findings")
    if isinstance(raw_side_findings, list):
        side_findings = []
        for idx, raw in enumerate(raw_side_findings):
            if not isinstance(raw, dict):
                continue
            payload = dict(raw)
            payload.setdefault("step_id", step.step_id)
            payload.setdefault("finding_id", f"{step.step_id}_side_{idx + 1}")
            side_findings.append(SideFinding.from_dict(payload).to_dict())
        if side_findings:
            step_record["side_findings"] = side_findings
    step_record["step_summary"] = step_summary
    evidence_refs_for_step, _, _ = typed_binding_resolver.resolve_names(
        evidence_ids_for_step,
        plan=plan,
        allow_unpublished_direct_ids=True,
    )
    validator_messages = _validator_messages(
        usage_findings,
        stat_findings,
        clinical_findings,
        guard_findings,
        contract_findings,
        figure_source_findings,
    )
    local_runtime_state = supervisor.critique_step(
        state=local_runtime_state,
        step_summary=step_summary,
        evidence_refs=evidence_refs_for_step,
        findings=validator_messages,
    )
    critique = local_runtime_state.critique
    critique_findings: List[ValidationFinding] = []
    if critique is not None:
        critique_path = run_result.out_dir / "critique_report.json"
        critique_path.write_text(
            critique.model_dump_json(indent=2),
            encoding="utf-8",
        )
        critique_record = evidence.register_file(
            kind="log",
            description=f"Structured critique report for step {step.step_id}.",
            source_path=critique_path,
            produced_by_step=step.step_id,
            script_evidence_id=script_record.evidence_id,
            aliases=[f"{step.step_id}_critique"],
            producer="critic",
            generation_mode="system",
            publish_aliases=False,
            metadata={"script_evidence_id": script_record.evidence_id},
        )
        pending_success_aliases[critique_record.evidence_id] = [
            f"{step.step_id}_critique"
        ]
        evidence_ids_for_step.append(critique_record.evidence_id)
        step_record["critique_report"] = critique.model_dump(mode="json")
        if critique.status in {"needs_revision", "blocked"}:
            critique_finding = ValidationFinding(
                validator="critic_agent",
                severity=(
                    "warning" if critique.status == "needs_revision" else "error"
                ),
                message=(
                    f"CriticAgent marked {step.step_id} as {critique.status}: "
                    + "; ".join(
                        critique.concerns
                        or critique.suggested_repairs
                        or ["review required"]
                    )
                ),
                evidence_ids=[critique_record.evidence_id],
                detail={
                    "step_id": step.step_id,
                    "critic_status": critique.status,
                },
            )
            critique_findings.append(critique_finding)
            with shared_lock:
                findings.append(critique_finding)

    step_record["evidence_ids"] = list(dict.fromkeys(evidence_ids_for_step))
    checkpoint_record = dict(step_record)
    checkpoint_record["status"] = "executed_pending_review"
    checkpoint_record["review_pending"] = True
    with shared_lock:
        _upsert_current_capsule_checkpoint(
            per_step_records,
            checkpoint_record,
        )
        _flush_partial_manifest()

    final_generation_mode = str(step_record.get("generation_mode") or "")
    non_llm_interpretation = _non_llm_interpretation_for_generation(
        step_id=step.step_id,
        generation_mode=final_generation_mode,
    )
    if non_llm_interpretation is not None:
        interpretation, interp_generation_mode = non_llm_interpretation
    else:
        interp_generation_mode = "llm"
        final_code_digest = sha256_of_bytes(code.encode("utf-8"))
        final_audit_token = concept_audit.tokens_by_digest.get(final_code_digest)
        if (
            pipeline._enable_llm_concept_audit
            and final_audit_token is not None
            and final_concept_gate_approved_code_digest == final_code_digest
            and concept_approved_code_digest == final_code_digest
            and executed_code_digest == final_code_digest
            and step_record.get("llm_concept_approved_code_sha256")
            == final_code_digest
        ):
            provider_budget.release_reserved_category(
                "concept_audit",
                token=final_audit_token,
            )
            _sync_provider_budget()
        try:
            interpretation = analyzer.run(
                context=plan_result.agent_context,
                step=step,
                step_summary=step_summary,
                evidence_ids=evidence_ids_for_step,
                provider_budget=provider_budget,
            )
        except Exception as exc:
            interpretation = f"(analyzer failed: {exc})"
            interp_generation_mode = "system"
    _sync_provider_budget()
    # Content-addressing alone is insufficient for step-owned evidence:
    # two steps may legitimately receive identical analyzer text.  Bind
    # the identity to the producing step and exact script so a later
    # resume never reuses another step's first-written authority record.
    interpretation_authority = "\0".join(
        (step.step_id, script_record.evidence_id, interpretation)
    )
    interpretation_evidence_id = (
        "log_interpretation_"
        + hashlib.sha256(interpretation_authority.encode("utf-8")).hexdigest()[:16]
    )
    interp_record = evidence.register_text(
        kind="log",
        description=f"Analyzer interpretation for step {step.step_id}.",
        text=interpretation,
        filename=f"interpretation_{step.step_id}.md",
        produced_by_step=step.step_id,
        script_evidence_id=script_record.evidence_id,
        evidence_id=interpretation_evidence_id,
        producer="analyzer",
        generation_mode=interp_generation_mode,
        prompt_pack_version=prompt_version,
        publish_aliases=False,
    )
    pending_success_aliases[interp_record.evidence_id] = [
        f"interpretation_{step.step_id}"
    ]
    step_record["interpretation_evidence_id"] = interp_record.evidence_id
    evidence_ids_for_step.append(interp_record.evidence_id)
    step_record["evidence_ids"] = list(dict.fromkeys(evidence_ids_for_step))
    step_record.pop("review_pending", None)
    mutated_after_seal = [
        name
        for name, expected_digest in sealed_result_digests.items()
        if not (candidate := run_result.out_dir / name).is_file()
        or sha256_of_file(candidate) != expected_digest
    ]
    if mutated_after_seal:
        seal_finding = ValidationFinding(
            validator="result_evidence_seal",
            severity="error",
            message=(
                f"Result artifacts for step {step.step_id} changed after the "
                "validate-and-seal boundary; registered evidence was retired."
            ),
            detail={
                "step_id": step.step_id,
                "attempt_id": attempt_id,
                "checkpoint_id": review_checkpoint_id,
                "mutated_artifacts": sorted(mutated_after_seal),
            },
        )
        contract_findings.append(seal_finding)
        step_record["contract_findings"] = [
            finding.model_dump() for finding in contract_findings
        ]
        step_record["result_evidence_sealed"] = False
        with shared_lock:
            findings.append(seal_finding)
    _propagate_findings_to_evidence(
        evidence_ids_for_step,
        usage_findings
        + stat_findings
        + clinical_findings
        + guard_findings
        + contract_findings
        + figure_source_findings
        + critique_findings,
        metadata={
            "step_id": step.step_id,
            "generation_mode": step_record["generation_mode"],
        },
    )
    with shared_lock:
        runtime_state = local_runtime_state
    step_record["status"] = _step_status_from_contract_findings(
        contract_findings=contract_findings,
        figure_source_findings=figure_source_findings,
        stat_findings=stat_findings,
        critique_status=(critique.status if critique is not None else None),
    )
    has_contract_error = step_record["status"] == "contract_failed"
    final_cleanup_finding: Optional[ValidationFinding] = None
    if step_record["status"] == "ok":
        step_record.pop("monotonic_concept_constraints", None)
        try:
            clear_quarantined_concept_draft(
                run_dir=run_dir,
                step_id=step.step_id,
            )
            if quarantine_state.resumed_draft_used:
                step_record["quarantined_requires_repair"] = False
                step_record["quarantine_retired"] = True
                if quarantine_state.superseded_by_fallback:
                    step_record["quarantine_retired_by"] = (
                        "successful_deterministic_fallback"
                    )
        except ValueError as exc:
            step_record["status"] = "blocked_quarantine_cleanup"
            final_cleanup_finding = ValidationFinding(
                validator="resume",
                severity="error",
                message=(
                    "Successful step output could not retire its stale "
                    f"quarantine safely for step {step.step_id}: {exc}"
                ),
                detail={"step_id": step.step_id},
            )
    envelope_sidecar_finding: Optional[ValidationFinding] = None
    if step_record["status"] == "ok":
        # Publish the sealed step-result envelope as a sidecar: the compiled
        # final snapshot re-bound to the terminal ``ok`` status, registered
        # outside the raw step outputs (in the evidence directory).  Its
        # alias is added to pending_success_aliases so the SAME success
        # transaction below promotes it; a rolled-back commit therefore
        # leaves an unpublished record that the loader can never recover as
        # current authority. The final validator and writer consume the
        # verified sidecar through RegisteredOutputEnvelopeConsumer.
        sidecar_snapshot = (
            final_gate_findings.result_envelope_snapshot.envelope
            if final_gate_findings.result_envelope_snapshot is not None
            else None
        )
        sidecar_failure_reason: Optional[str] = None
        try:
            published_envelope_sidecar = (
                publish_terminal_step_result_envelope_sidecar(
                    snapshot_envelope=sidecar_snapshot,
                    step_id=step.step_id,
                    attempt_id=attempt_id,
                    checkpoint_id=review_checkpoint_id,
                    script_evidence_id=script_record.evidence_id,
                    terminal_status="ok",
                    evidence_store=evidence,
                )
            )
        except (
            EvidenceAuthorityIntegrityError,
            ValueError,
            OSError,
        ) as exc:
            published_envelope_sidecar = None
            sidecar_failure_reason = f"sidecar_registration_failed: {exc}"
        if published_envelope_sidecar is not None:
            pending_success_aliases[published_envelope_sidecar.evidence_id] = [
                published_envelope_sidecar.alias
            ]
            evidence_ids_for_step.append(published_envelope_sidecar.evidence_id)
            step_record["evidence_ids"] = list(dict.fromkeys(evidence_ids_for_step))
        else:
            # A successful step MUST seal a recoverable envelope sidecar.
            # A missing snapshot, a fail-closed prepare, or a registration
            # error breaks the ``status == "ok"`` invariant; it is a typed
            # contract failure, never a silent commit without recoverable
            # envelope authority.  The step therefore never reaches the
            # StepEvidenceCommit below, so no alias is promoted.
            step_record["status"] = "contract_failed"
            envelope_sidecar_finding = ValidationFinding(
                validator="result_envelope_sidecar",
                severity="error",
                message=(
                    "Successful step could not seal a recoverable "
                    "step-result envelope sidecar for step "
                    f"{step.step_id}; refusing to commit the step as ok "
                    "without recoverable envelope authority."
                ),
                detail={
                    "step_id": step.step_id,
                    "attempt_id": attempt_id,
                    "checkpoint_id": review_checkpoint_id,
                    "reason": (
                        sidecar_failure_reason
                        or "sidecar_unavailable_for_successful_step"
                    ),
                },
            )
            contract_findings.append(envelope_sidecar_finding)
            step_record["contract_findings"] = [
                finding.model_dump() for finding in contract_findings
            ]
            has_contract_error = True
    evidence_publication_finding: Optional[ValidationFinding] = None
    if step_record["status"] == "ok":
        try:
            # Numeric provenance and result aliases must share one durable
            # commit. StepEvidenceCommit opens the store's transaction so both
            # go current as one generation; an alias collision or I/O failure
            # rolls the staged claims back too.
            promotion = step_evidence_commit.commit_validated_step(
                step_id=step.step_id,
                pending_aliases=pending_success_aliases,
                allowed_evidence_ids=evidence_ids_for_step,
                register_numeric_claims=_register_current_step_numeric_claims,
            )
            if promotion.retained_cross_step_aliases:
                step_record["retained_cross_step_aliases"] = (
                    promotion.retained_cross_step_aliases
                )
        except (
            EvidenceAuthorityIntegrityError,
            KeyError,
            ValueError,
            OSError,
        ) as exc:
            store_unavailable = isinstance(
                exc, (EvidenceAuthorityIntegrityError, OSError)
            )
            step_record["status"] = "contract_failed"
            evidence_publication_finding = ValidationFinding(
                validator="result_evidence_authority",
                severity="error",
                message=(
                    "Validated result evidence and numeric provenance could "
                    "not be promoted to current authority for step "
                    f"{step.step_id}."
                ),
                detail={
                    "step_id": step.step_id,
                    "attempt_id": attempt_id,
                    "checkpoint_id": review_checkpoint_id,
                    "reason": str(exc),
                    "evidence_store_write_suppressed": store_unavailable,
                },
            )
            contract_findings.append(evidence_publication_finding)
            step_record["contract_findings"] = [
                finding.model_dump() for finding in contract_findings
            ]
            if not store_unavailable:
                _propagate_findings_to_evidence(
                    evidence_ids_for_step,
                    [evidence_publication_finding],
                    metadata={
                        "step_id": step.step_id,
                        "generation_mode": step_record["generation_mode"],
                    },
                )
            has_contract_error = True
    if step_record["status"] == "ok":
        # This stamp is written only after deterministic artifact gates and
        # Critic review pass, numeric authority, and current evidence aliases
        # are published.
        step_record.update(_deterministic_gate_stamp())
    with shared_lock:
        if final_cleanup_finding is not None:
            findings.append(final_cleanup_finding)
        if envelope_sidecar_finding is not None:
            findings.append(envelope_sidecar_finding)
        if evidence_publication_finding is not None:
            findings.append(evidence_publication_finding)
        _append_terminal_step_record(per_step_records, step_record)
        _flush_partial_manifest()
    emit_progress(
        "step",
        (
            f"Step {step_current}/{total_steps} failed contract checks: "
            f"{step.step_id}."
            if has_contract_error
            else (
                f"Step {step_current}/{total_steps} failed Critic review: "
                f"{step.step_id}."
                if step_record["status"] == "critic_failed"
                else (
                    f"Step {step_current}/{total_steps} could not retire its "
                    f"quarantine: {step.step_id}."
                    if step_record["status"] == "blocked_quarantine_cleanup"
                    else f"Step {step_current}/{total_steps} complete: {step.step_id}."
                )
            )
        ),
        status=("complete" if step_record["status"] == "ok" else "error"),
        run_id=run_id,
        step_id=step.step_id,
        current_step=step_current,
        total_steps=total_steps,
    )
    return runtime_state, step_record
