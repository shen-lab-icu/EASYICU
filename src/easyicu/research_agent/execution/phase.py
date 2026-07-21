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
from contextvars import copy_context
from dataclasses import dataclass
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
from ..reporting.article_contract import (
    article_contract_audit_payload,
    summarize_article_contract_coverage,
    validate_run_against_article_contract,
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
from .code_hygiene import reorder_forward_references
from ..repairs.coordination import (
    RepairAuthorityBinding,
    StepRepairBudget,
    authorized_deterministic_concept_repair,
)
from .concept_audit_cache import LLMConceptAuditCache
from .development_sample import (
    DEVELOPMENT_PRIMARY_COHORT_CONFIRMATION_ROLE,
    materialize_development_execution_sample,
    record_development_sample_authority,
)
from .cohort_routing import (
    bind_step_execution_cohort as _bind_step_execution_cohort,
    bound_step_execution_cohort_path as _bound_step_execution_cohort_path,
    step_execution_cohort_path as _step_execution_cohort_path,
)
from .concept_audit import (
    ConceptAuditAuthority,
    ConceptAuditCoordinator,
    ConceptAuditRuntime,
    ConceptQuarantineState,
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
from ..authority.coder_authority import HostCoderAuthority
from ..research_context.prompt_scope import scoped_coder_context
from ..cohort.repair import extract_cohort_definition_from_prose
from ..cohort.schema import (
    CohortDefinition,
    assert_cohort_definition_locked,
    materialize_locked_analysis_cohort,
    write_locked_cohort_definition,
)
from ..authority.execution_input import ExecutionInputAuthorityState
from ..intake.materialized_metadata import MaterializedMetadataError
from ..intake.materialized_trajectory import (
    MaterializedTrajectoryError,
    StagedTrajectoryBinding,
)
from ..research_context.typed import materialized_input_prompt_attachment
from ..contracts.runtime import ValidationFinding, _ExecutePhaseResult, _PlanPhaseResult
from .runners.deterministic_descriptive import absolute_risk_context_code
from .runners.deterministic_missingness import (
    missingness_measurement_audit_code,
)
from .runners.deterministic_robustness import (
    robustness_sensitivity_preflight_code,
)
from ..contracts.declared_product import (
    RUNTIME_BINDABLE_TYPED_INPUT_KINDS,
    RUNTIME_TYPED_INPUT_EVIDENCE_KINDS,
    authorize_declared_figure_product_slots,
    primary_analysis_cohort_plan_findings,
    primary_analysis_cohort_producer_uses_universe,
    read_digest_bound_artifact_snapshot,
    typed_product_binding_contract,
    typed_product_schema_receipt,
    typed_product as _canonical_typed_product,
)
from ..robustness.estimators import fit_robustness_rows_from_records
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
    _coder_authority_with_typed_parent_schema_receipts,
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
from .publication_figure import (
    SealedRendererState,
    _deterministic_publication_figure_code,
    _sealed_parent_planner_anchors,
    _sealed_renderer_implementation_digest,
    _sealed_renderer_source_digests,
    _sealed_typed_figure_products,
)
from .host_services import ExecutePhaseHost
from .output_files import _clear_output_dir, _has_figure_exports
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
from ..contracts.ordered_stratified import ordered_stratified_numeric_findings
from ..repairs.reasons import (
    RepairPromptAuthority,
    RepairReason,
    repair_prompt_binding_sha256,
    repair_reason_for_finding,
    typed_repair_ticket,
)
from ..plan_utils import (
    _augment_measurement_companion_inputs,
    _augment_report_typed_product_inputs,
    _cap_plan_preserving_figure_steps,
    _clustering_contract_applies,
    _cohort_definition_contract_findings,
    _cohort_definition_is_empty,
    _cohort_definition_prose,
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
from ..schema import AnalysisPlan, AnalysisStep, EvidenceRef, ResearchContext
from ..contracts.robustness_execution import (
    ROBUSTNESS_COHORT_MEMBERSHIP_ALIASES,
    ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE,
    _executed_robustness_result_issues,
)
from ..robustness.panel import (
    RobustnessSpec,
    assert_robustness_specs_locked,
    build_robustness_panel_from_records,
    robustness_specs_for_execution,
    robustness_specs_sha,
    write_robustness_panel,
)
from ..trajectory.bundle import trajectory_bundle_findings
from ..trajectory.plan_contract import (
    augment_trajectory_plan_products,
    trajectory_plan_contract_applies,
    trajectory_plan_dag_findings,
)
from .runners.trajectory_stability_executor import (
    trajectory_stability_executor_code,
    trajectory_stability_executor_owns_step,
)
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
from ..authority.run_input import (
    RUN_INPUT_CAPSULE_EVIDENCE_ID,
    RUN_INPUT_CAPSULE_FILENAME,
    RunInputIdentityError,
    _HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD,
    _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND,
    _HOST_COHORT_MATERIALIZER_GENERATION_MODE,
    _HOST_PROBE_AUTHORITIES,
    _HOST_PROBE_AUTHORITY_KIND,
    _host_cohort_materializer_authority_error,
    _host_probe_authority_error,
    build_environment_identity,
    canonical_sha256,
    engine_code_sha256,
    verify_legacy_trajectory_capsule_receipt,
    validator_code_sha256,
)
from .run_coordination import RunCoordinator, RunExecutionState, RunTransition
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
    read_concept_audit_findings,
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
    return repair_metadata_for(repair_id).repair_class is RepairClass.SYNTACTIC


_STANDARD_EXECUTOR_INTERNAL_PENDING_ARTIFACTS = frozenset(
    {".cluster_stability_assignments.pending.csv"}
)
_FIGURE_CONTRACT_SOURCE_DATA_SCHEMA_REPAIR_ID = "figure_contract_source_data_schema_v1"
_COHORT_TRANSLATION_PROVIDER_CATEGORY = "cohort_definition_translation"
_HOST_COHORT_TRANSLATION_BUDGET_STEP_ID = "host_cohort_definition_translation"


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
    digest = sha256_of_file(working_path)
    if digest != str(record.sha256):
        raise RunInputIdentityError("run input capsule working digest changed")
    if working_path.read_bytes() != sealed_path.read_bytes():
        raise RunInputIdentityError(
            "run input capsule working copy differs from sealed evidence"
        )
    return digest


def _declares_host_cohort_only_product(step: AnalysisStep) -> bool:
    declared = {
        str(value or "").strip().casefold()
        for value in (step.expected_outputs or [])
        if str(value or "").strip()
    }
    return declared == {"table:analysis_cohort"}


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
    initial_generation_entry: Optional[Dict[str, object]] = None
    required_reservation_token: Optional[str] = None
    reservation_bound_provider_history_len: Optional[int] = None
    completed_reservation_token: Optional[str] = None
    reservation_released = False
    if receipt_path.exists():
        receipt_state = load_provider_call_budget_state(
            receipt_path,
            step_id=budget_owner_step_id,
            expected_reserved_final_category=reserved_final_category,
        )
        effective_limit = min(effective_limit, receipt_state.limit)
        consumed_categories = receipt_state.categories
        logical_repair_entries = receipt_state.logical_repairs
        initial_generation_entry = receipt_state.initial_generation
        required_reservation_token = receipt_state.required_reservation_token
        reservation_bound_provider_history_len = (
            receipt_state.reservation_bound_provider_history_len
        )
        completed_reservation_token = receipt_state.completed_reservation_token
        reservation_released = receipt_state.reservation_released
    budget = StepProviderCallBudget(
        effective_limit,
        step_id=budget_owner_step_id,
        consumed_categories=consumed_categories,
        logical_repair_entries=logical_repair_entries,
        initial_generation_entry=initial_generation_entry,
        receipt_path=receipt_path,
        reserved_final_category=reserved_final_category,
        required_reservation_token=required_reservation_token,
        reservation_bound_provider_history_len=(reservation_bound_provider_history_len),
        completed_reservation_token=completed_reservation_token,
        reservation_released=reservation_released,
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


def _monotonic_step_llm_repair_history(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[int, List[str], bool]:
    """Recover the largest durable logical-repair counter for one step.

    Step records are append-only attempts.  The latest attempt may terminate
    before copying the logical counter (for example, on a damaged provider
    receipt), so latest-record-only recovery can incorrectly buy a fresh
    repair budget.  A malformed explicit counter is treated conservatively as
    exhausted instead of being ignored.
    """

    attempts = 0
    classes: List[str] = []
    invalid_snapshot = False
    for record in records:
        if "step_llm_repair_attempts" in record:
            raw_attempts = record.get("step_llm_repair_attempts")
            if (
                isinstance(raw_attempts, bool)
                or not isinstance(raw_attempts, int)
                or raw_attempts < 0
            ):
                invalid_snapshot = True
            else:
                attempts = max(attempts, raw_attempts)
        raw_classes = record.get("step_llm_repair_classes")
        if not isinstance(raw_classes, list):
            continue
        normalized = [str(item).strip() for item in raw_classes]
        if any(not item for item in normalized):
            invalid_snapshot = True
            continue
        if len(normalized) > len(classes):
            classes = normalized
    if invalid_snapshot:
        attempts = max(attempts, max(0, int(limit)))
    return attempts, classes, invalid_snapshot


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


def _resolve_stop_after_step_selector(
    plan: AnalysisPlan,
    requested: Optional[str],
) -> Optional[str]:
    """Resolve a structural checkpoint without guessing Agent-owned step ids."""

    if requested is None:
        return None
    if requested == "@first":
        if not plan.steps:
            raise ValueError("stop_after_step_id='@first' requires a non-empty plan")
        return str(plan.steps[0].step_id)
    index_match = re.fullmatch(r"@index:([1-9][0-9]*)", requested)
    if index_match is None:
        return requested
    one_based_index = int(index_match.group(1))
    if one_based_index > len(plan.steps):
        raise ValueError(
            f"stop_after_step_id={requested!r} exceeds the active plan's "
            f"{len(plan.steps)} step(s)."
        )
    return str(plan.steps[one_based_index - 1].step_id)


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


def _step_snapshot_requires_provider_receipt(
    record: Mapping[str, Any],
    *,
    provider_attempts: int,
    logical_repair_attempts: int,
) -> bool:
    """Whether a checkpoint proves a durable provider ledger must exist."""

    if record.get("step_provider_call_receipt_version") not in {
        1,
        2,
        3,
        4,
        5,
        PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
    }:
        return False
    return bool(
        provider_attempts > 0
        or logical_repair_attempts > 0
        or record.get("capsule_pending_initial_transport_id")
        or record.get("step_provider_call_receipt")
    )


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


def _successful_step_requests_replan(record: Mapping[str, Any]) -> bool:
    """Return whether a clean agent step explicitly requests plan adaptation.

    The deterministic probe already receives one automatic replan and failed
    model steps have their own bounded directed-replan path. Calling the LLM
    replanner after every ordinary successful step adds latency and usually
    produces a no-op. Preserve adaptive agent behavior through exact boolean
    declarations in either the outer record or ``step_summary``; strings and
    other truthy values are intentionally not accepted.
    """

    if str(record.get("status") or "") != "ok":
        return False
    containers: List[Mapping[str, Any]] = [record]
    summary = record.get("step_summary")
    if isinstance(summary, Mapping):
        containers.append(summary)
    return any(
        container.get(field) is True
        for container in containers
        for field in _SUCCESS_REPLAN_REQUEST_FIELDS
    )


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


def _bind_findings_to_step_attempt(
    findings: Sequence[ValidationFinding],
    *,
    step_id: str,
    attempt_id: str,
    checkpoint_id: str,
) -> List[ValidationFinding]:
    """Attach host-owned execution identity to deterministic findings.

    Validator payloads are intentionally reusable and therefore do not know
    which resume attempt invoked them.  Supersession must never infer that
    identity from a message string: the orchestrator binds it at the review
    checkpoint before persisting either the finding or the outer step record.
    """

    bound: List[ValidationFinding] = []
    for finding in findings:
        detail = dict(finding.detail or {})
        detail.update(
            {
                "step_id": step_id,
                "attempt_id": attempt_id,
                "checkpoint_id": checkpoint_id,
            }
        )
        bound.append(finding.model_copy(update={"detail": detail}))
    return bound


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


def _bind_materialized_coder_authority(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    authority: HostCoderAuthority,
) -> tuple[ResearchContext, HostCoderAuthority]:
    """Bind one step-scoped V2 fact attachment to every coder call path."""

    scoped_context = scoped_coder_context(context, step)
    attachment = materialized_input_prompt_attachment(scoped_context)
    if not attachment:
        # Preserve the archived V1 context and authority coordinates exactly.
        return context, authority
    return scoped_context, authority.append(attachment)


# Max directed full-replans fired when a model/estimation step self-blocks on a
# task-viable cohort. Two attempts give the replanner a fair chance to honour
# the override directive; beyond that the run falls back to an honest
# diagnostic_only rather than burning the replanner on a stuck plan.
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


def _should_attempt_detached_figure_binding(
    *, out_dir: Path, sealed_renderer_authorized_code_sha256: Optional[str]
) -> bool:
    """Detached rescue lineage must never rewrite an authorized sealed summary."""

    return sealed_renderer_authorized_code_sha256 is None and _has_figure_exports(
        out_dir
    )


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
    return loaded if isinstance(loaded, dict) else {"raw": loaded}


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


# No deterministic runner owns a primary scientific estimand.  Kept as an
# explicit empty compatibility surface for drift checks and legacy run records.
_PRIMARY_DETERMINISTIC_RUNNERS: set[str] = set()

# Method names the planner uses for a PRIMARY estimation step (not a
# prep/audit/figure step). A dose-response is routed to the ordinal runner only
# when a dose-response signal is ALSO present, so listing broad association
# methods here does not hijack a plain association step.
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
# Methods that are UNAMBIGUOUSLY a dose-response primary on their own.
_ORDINAL_EXPLICIT_METHODS = frozenset(
    {
        "dose_response",
        "dose_response_analysis",
    }
)
# General dose-response / graded-exposure vocabulary (case-neutral: never a
# specific score name). Present in the question, intent, or declared outputs.
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

# --- Cohort-definition-sensitivity routing (precise, not blunt keyword) -------
# A cohort-definition-sensitivity step VARIES the cohort/eligibility definition
# and compares the result across alternative definitions. The authoritative
# signal is the planner's own ``method`` key; the historical blunt test --
# ``"sensitivity" in blob and ("cohort"|"definition" in blob)`` -- false-positives
# on a primary estimand step that merely mentions a pre-specified within-cohort
# sensitivity sub-analysis. Require an alternative-definition signal instead.
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


# The compact missingness runner owns per-concept measurement counts only.  A
# richer exposure/source repair must retain the coder path until a runner that
# actually owns all of these contracts exists.
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
        "measurement_process_audit",
        "data_quality_audit",
        "data_quality",
    }
)
_ABSOLUTE_RISK_CONTEXT_METHODS = frozenset(
    {
        "absolute_risk_context",
        "descriptive_context",
        "exposure_outcome_summary",
    }
)
_ROBUSTNESS_SENSITIVITY_METHODS = frozenset(
    {
        "prespecified_robustness",
        "robustness_sensitivity",
        "sensitivity_comparison",
    }
)


# An ordinal *trend test* can be a purely descriptive result.  The primary
# dose-response runner fits an adjusted model, so it may only claim a broadly
# named ordinal/association step when the declared contract or intent actually
# asks for a model/effect estimate.  This keeps exposure derivation/QC and
# stage-stratified descriptive steps with their own owners.
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
    outputs = {str(item or "").lower() for item in (expected_outputs or [])}
    if any(item.startswith("figure:") for item in outputs):
        return False
    supported_products = {
        "exposure_outcome_summary",
        "exposure_prevalence_and_absolute_risk",
        "absolute_risk",
        "absolute_risk_context",
    }
    if _method_head(method) not in _ABSOLUTE_RISK_CONTEXT_METHODS:
        return False
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
) -> bool:
    """True for a separate prespecified robustness-comparison owner."""

    del step_id
    outputs = {str(item or "").lower() for item in (expected_outputs or [])}
    if any(item.startswith("figure:") for item in outputs):
        return False
    method_head = _method_head(method)
    if method_head not in _ROBUSTNESS_SENSITIVITY_METHODS:
        return False
    supported_products = {
        "robustness_matrix",
        "robustness_summary",
        "complete_case_n",
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


# --- Trajectory-clustering compatibility audit ------------------------------
# Kept as a tested contract helper for legacy/resume inspection. Production has
# no clustering preflight or coder-failure runner: the agent owns feature/method/k
# and deterministic code only renders registered clustering products.
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


def _primary_runner_core_estimate_present(
    kind: Optional[str], step_summary: Mapping[str, Any]
) -> bool:
    """True when a PRIMARY deterministic runner emitted its core estimate.

    The runner's own ``status`` is the authority: it writes ``ok`` only when the
    estimate computed and ``blocked`` on genuinely non-viable data. When ``ok``
    and the effect key is present, the runner has satisfied the scientific
    contract for the step -- any extra planner-requested output tables it does
    not emit are advisory, not a reason to discard a trustworthy estimate.
    """
    if kind not in _PRIMARY_DETERMINISTIC_RUNNERS:
        return False
    if not isinstance(step_summary, Mapping):
        return False
    if str(step_summary.get("status") or "").lower() != "ok":
        return False
    if kind in ("causal_primary_iptw", "ordinal_dose_response"):
        # Both emit the scale-neutral ``adjusted_effect`` as their core estimate
        # (causal: marginal OR; ordinal: trend OR per +1 stage).
        return step_summary.get("adjusted_effect") is not None
    # survival_primary_cox
    if step_summary.get("hazard_ratio") is not None:
        return True
    primary_model = step_summary.get("primary_model")
    return (
        isinstance(primary_model, Mapping)
        and primary_model.get("hazard_ratio") is not None
    )


def _demote_step_contract_for_primary_runner(
    step_record: Mapping[str, Any],
    step_summary: Mapping[str, Any],
    findings: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Apply contract compatibility to legacy deterministic-primary records.

    When such a runner produced its core estimate, demote ``step_contract``
    missing-output ERRORS to advisory warnings. Otherwise a planner that
    over-specifies a step's ``expected_outputs`` (e.g. 17 documentation tables a
    causal step does not need) fail-closes the step and triggers a repair that
    replaces a validated legacy estimate with a repair. Integrity findings from
    other validators (exposure / overadjustment / leakage / figure) remain
    blocking. Live primary science is agent-owned; this is record compatibility.
    """
    kind = step_record.get("deterministic_standard_analysis")
    if not _primary_runner_core_estimate_present(kind, step_summary):
        return list(findings)
    demoted: List[ValidationFinding] = []
    for finding in findings:
        if (
            getattr(finding, "validator", "") == "step_contract"
            and finding.severity == "error"
        ):
            finding = finding.model_copy(
                update={
                    "severity": "warning",
                    "message": (
                        finding.message
                        + f" [advisory: step satisfied by deterministic {kind} "
                        "runner; extra planner-requested outputs are non-blocking]"
                    ),
                }
            )
        demoted.append(finding)
    return demoted


def _is_too_few_panels_figure_finding(finding: ValidationFinding) -> bool:
    """True for the ``figure_contract_quality`` "result figure has <2 panels"
    ERROR specifically.

    Keyed off ``detail['panel_count']`` (which only that finding sets) rather
    than the message text, so it stays robust if the wording changes. Blank-
    title / weak-claim / fallback-term figure errors are deliberately NOT
    matched -- only the panel-count shape rule is demoted below.
    """
    if getattr(finding, "validator", "") != "figure_contract_quality":
        return False
    if getattr(finding, "severity", "") != "error":
        return False
    detail = getattr(finding, "detail", None) or {}
    panel_count = detail.get("panel_count") if isinstance(detail, Mapping) else None
    return isinstance(panel_count, int) and panel_count < 2


def _demote_result_figure_shape_for_family_renderer(
    context: Any,
    findings: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Demote a step-level "result figure has <2 panels" ERROR to a warning when
    the study-design family assembles its primary figure deterministically.

    A family in ``FAMILY_RENDERERS`` can have a deterministic multi-panel publication
    figure renderer, but it only runs in the WRITE phase -- which is gated behind
    ``execution_complete``. When the LLM's step-level figure is single-panel, the
    ``figure_contract_quality`` panel-count ERROR marks the step ``contract_
    failed`` -> ``execution_complete`` stays False -> the write phase is skipped
    -> the deterministic renderer (the very thing that would produce the >=2-panel
    primary) never runs. The step-level figure is NOT the manuscript's primary
    for these families, so its panel count is advisory here. The write-phase
    display-suite gate remains fully fail-closed: if the deterministic renderer
    cannot build a >=2-panel primary from the registered tables, the run still
    fails with "no primary publication result-bearing figure contract". Pure so
    both branches are unit-testable.
    """
    if not any(_is_too_few_panels_figure_finding(f) for f in findings):
        return list(findings)
    if not _family_has_deterministic_figure_renderer(context):
        return list(findings)
    demoted: List[ValidationFinding] = []
    for finding in findings:
        if _is_too_few_panels_figure_finding(finding):
            finding = finding.model_copy(
                update={
                    "severity": "warning",
                    "message": (
                        finding.message
                        + " [advisory: this study-design family builds its "
                        "manuscript-facing primary figure deterministically in "
                        "the write phase; the display-suite gate remains the "
                        "fail-closed backstop for panel count and role diversity]"
                    ),
                }
            )
        demoted.append(finding)
    return demoted


@dataclass(frozen=True)
class _FinalDeterministicGateFindings:
    """Attempt-bound finding groups produced by the final deterministic gate.

    The immutable grouping keeps evaluation separate from orchestration: the
    evaluator below reads sealed outputs and returns findings, while the caller
    remains responsible for publishing them to the run manifest, evidence
    metadata, and outer step status.  Resume revalidation can therefore reuse
    the same evaluator without duplicating its gate composition.
    """

    stat_findings: Tuple[ValidationFinding, ...]
    clinical_findings: Tuple[ValidationFinding, ...]
    guard_findings: Tuple[ValidationFinding, ...]
    contract_findings: Tuple[ValidationFinding, ...]
    figure_source_findings: Tuple[ValidationFinding, ...]

    def all_findings(self) -> Tuple[ValidationFinding, ...]:
        """Return all groups in the historical manifest publication order."""

        return (
            *self.stat_findings,
            *self.clinical_findings,
            *self.guard_findings,
            *self.contract_findings,
            *self.figure_source_findings,
        )


@dataclass(frozen=True)
class _ResumeDeterministicRevalidationResult:
    """Append-only resume ledger after selective deterministic replay."""

    resume_state: Dict[str, Any]
    revalidated_step_ids: Tuple[str, ...]
    invalidated_step_ids: Tuple[str, ...]


def _verified_explicit_step_authority(
    *,
    record: Mapping[str, Any],
    field: str,
    expected_kind: str,
    expected_source_name: Optional[str],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
) -> Tuple[Any, Path]:
    """Resolve one exact checkpoint authority through owner/path/SHA checks."""

    step_id = str(record.get("step_id") or "").strip()
    evidence_id = str(record.get(field) or "").strip()
    listed = {
        str(value).strip()
        for value in (record.get("evidence_ids") or [])
        if str(value).strip()
    }
    if not evidence_id:
        raise ValueError(f"successful checkpoint is missing required {field}")
    if evidence_id not in listed:
        raise ValueError(f"{field} {evidence_id} is absent from evidence_ids")
    authority = evidence_by_id.get(evidence_id)
    if authority is None:
        raise ValueError(f"{field} references missing evidence {evidence_id}")
    if str(_evidence_record_field(authority, "produced_by_step") or "") != step_id:
        raise ValueError(f"{field} is not owned by step {step_id}")
    actual_kind = str(_evidence_record_field(authority, "kind") or "").lower()
    if actual_kind != expected_kind:
        raise ValueError(
            f"{field} has kind {actual_kind or '<missing>'}, expected {expected_kind}"
        )
    verified_path = verified_run_evidence_path(run_dir, authority)
    if verified_path is None:
        raise ValueError(f"{field} failed path/digest verification")
    source_name = _registered_source_name(authority, verified_path)
    if expected_source_name is not None and source_name != expected_source_name:
        raise ValueError(f"{field} does not name {expected_source_name}")
    return authority, verified_path


def _verified_resume_step_summary(
    *,
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """Load a summary only from the record's explicit digest-bound evidence."""

    field = (
        "probe_summary_evidence_id"
        if str(record.get("step_id") or "") == "00_probe"
        else "step_summary_evidence_id"
    )
    _, summary_path = _verified_explicit_step_authority(
        record=record,
        field=field,
        expected_kind="statistic",
        expected_source_name=(
            "probe_summary.json"
            if field == "probe_summary_evidence_id"
            else "step_summary.json"
        ),
        evidence_by_id=evidence_by_id,
        run_dir=run_dir,
    )
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field} payload is not an object")
    return payload


def _verify_resume_step_script_lineage(
    *,
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Any],
) -> None:
    """Require every sealed non-code output to bind the reviewed script.

    Owner and digest checks alone are insufficient: a mutable checkpoint could
    list a second benign script from the same step and point
    ``script_evidence_id`` at it while retaining outputs produced by the real
    script.  Exact lineage closes that decoy-code path before preflight.
    """

    step_id = str(record.get("step_id") or "").strip()
    script_evidence_id = str(record.get("script_evidence_id") or "").strip()
    if not script_evidence_id:
        raise ValueError("successful checkpoint is missing script_evidence_id")
    for raw_id in record.get("evidence_ids") or []:
        evidence_id = str(raw_id).strip()
        authority = evidence_by_id.get(evidence_id)
        if authority is None:
            raise ValueError(f"listed evidence {evidence_id} is missing")
        owner = str(_evidence_record_field(authority, "produced_by_step") or "")
        if owner != step_id:
            raise ValueError(
                f"listed evidence {evidence_id} belongs to {owner or '<run-level>'}"
            )
        if evidence_id == script_evidence_id:
            if str(_evidence_record_field(authority, "kind") or "").lower() != "code":
                raise ValueError("script_evidence_id does not reference code evidence")
            continue
        bound_script_id = str(
            _evidence_record_field(authority, "script_evidence_id") or ""
        ).strip()
        if bound_script_id != script_evidence_id:
            raise ValueError(
                f"listed evidence {evidence_id} is bound to script "
                f"{bound_script_id or '<missing>'}, not {script_evidence_id}"
            )


_STALE_RESOLVED_INPUT_RECEIPT_FIELDS = (
    "resolved_inputs",
    "resolved_input_bindings",
    "resolved_inputs_path",
    "resolved_inputs_sha256",
    "revalidated_input_bindings_fingerprint",
)


def _discard_stale_resolved_input_receipts(record: Dict[str, Any]) -> None:
    """Remove mutable or superseded resolved-input receipts in place."""

    for field in _STALE_RESOLVED_INPUT_RECEIPT_FIELDS:
        record.pop(field, None)


def _trusted_resume_success_records(
    *,
    records: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Replace mutable checkpoint summaries with explicit evidence payloads."""

    trusted: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    for record in records:
        if str(record.get("status") or "").lower() != "ok":
            continue
        step_id = str(record.get("step_id") or "").strip()
        if (
            str(record.get("generation_mode") or "").strip().lower()
            == _HOST_COHORT_MATERIALIZER_GENERATION_MODE
            and record.get("step_authority_kind")
            == _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
        ):
            copy = dict(record)
            _discard_stale_resolved_input_receipts(copy)
            trusted.append(copy)
            continue
        try:
            summary = _verified_resume_step_summary(
                record=record,
                evidence_by_id=evidence_by_id,
                run_dir=run_dir,
            )
        except ValueError as exc:
            errors[step_id] = str(exc)
            continue
        copy = dict(record)
        copy["step_summary"] = summary
        # These mutable convenience receipts are never replay authority.
        _discard_stale_resolved_input_receipts(copy)
        trusted.append(copy)
    return trusted, errors


def _materialize_verified_step_output_view(
    *,
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
    destination: Path,
) -> Dict[str, str]:
    """Copy only listed, verified same-step evidence under source filenames."""

    step_id = str(record.get("step_id") or "").strip()
    listed = [
        str(value).strip()
        for value in (record.get("evidence_ids") or [])
        if str(value).strip()
    ]
    if not listed:
        raise ValueError("successful checkpoint has no evidence_ids")
    destination.mkdir(parents=True, exist_ok=False)
    copied: Dict[str, str] = {}
    for evidence_id in listed:
        authority = evidence_by_id.get(evidence_id)
        if authority is None:
            raise ValueError(f"listed evidence {evidence_id} is missing")
        owner = str(_evidence_record_field(authority, "produced_by_step") or "")
        if owner != step_id:
            raise ValueError(
                f"listed evidence {evidence_id} belongs to {owner or '<run-level>'}"
            )
        verified_path = verified_run_evidence_path(run_dir, authority)
        if verified_path is None:
            raise ValueError(
                f"listed evidence {evidence_id} failed digest verification"
            )
        source_name = _registered_source_name(authority, verified_path)
        if (
            not source_name
            or Path(source_name).name != source_name
            or "/" in source_name
            or "\\" in source_name
        ):
            raise ValueError(
                f"listed evidence {evidence_id} has no safe source filename"
            )
        prior_id = copied.get(source_name)
        if prior_id is not None and prior_id != evidence_id:
            raise ValueError(f"multiple listed evidence records claim {source_name}")
        target = destination / source_name
        shutil.copyfile(verified_path, target)
        target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        copied[source_name] = evidence_id
    return copied


_REPLAY_SUMMARY_OUTPUT_CONTAINER_KEYS = frozenset(
    {"output_files", "output_artifacts", "outputs", "figure_files"}
)
_REPLAY_SUMMARY_DESCRIPTOR_PATH_KEYS = frozenset({"path", "relative_path", "filename"})
_REPLAY_SUMMARY_DIRECT_FIGURE_KEYS = frozenset({"figure_file", "figure_path"})


def _project_verified_replay_output_paths(
    summary: Mapping[str, Any],
    *,
    materialized_evidence_by_source_name: Mapping[str, str],
) -> Dict[str, Any]:
    """Point one in-memory replay summary at its verified temporary view.

    Historical summaries may contain absolute paths into the original step
    output directory.  Resume revalidation deliberately does not trust those
    mutable files: it copies the checkpoint's digest-verified evidence into a
    temporary output view instead.  Project only path values whose basename
    is backed by exactly one materialized, same-step evidence record.  An
    unmatched absolute path is left intact so containment gates continue to
    fail closed.

    The sealed summary bytes and checkpoint record are never modified; this
    projection exists only for deterministic replay against the temporary
    evidence view.
    """

    source_names = {
        str(name)
        for name in materialized_evidence_by_source_name
        if str(name) and Path(str(name)).name == str(name)
    }

    def project_path(value: str) -> str:
        raw = str(value).strip()
        source_name = Path(raw).name
        return source_name if source_name in source_names else value

    def visit(value: Any, *, output_container: bool = False) -> Any:
        if isinstance(value, Mapping):
            projected: Dict[Any, Any] = {}
            for raw_key, child in value.items():
                key = re.sub(r"[^a-z0-9]+", "_", str(raw_key).strip().lower()).strip(
                    "_"
                )
                starts_output_container = key in _REPLAY_SUMMARY_OUTPUT_CONTAINER_KEYS
                child_is_output = output_container or starts_output_container
                if isinstance(child, str) and (
                    key in _REPLAY_SUMMARY_DIRECT_FIGURE_KEYS
                    or (
                        output_container and key in _REPLAY_SUMMARY_DESCRIPTOR_PATH_KEYS
                    )
                    or starts_output_container
                ):
                    projected[raw_key] = project_path(child)
                elif isinstance(child, str):
                    projected[raw_key] = child
                else:
                    projected[raw_key] = visit(
                        child,
                        output_container=child_is_output,
                    )
            return projected
        if isinstance(value, list):
            return [visit(item, output_container=output_container) for item in value]
        if isinstance(value, tuple):
            return tuple(
                visit(item, output_container=output_container) for item in value
            )
        if isinstance(value, str) and output_container:
            return project_path(value)
        return copy.deepcopy(value)

    return visit(summary)


def _resume_success_dependencies(
    *,
    plan: AnalysisPlan,
    current_records: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Any],
) -> Dict[str, Set[str]]:
    """Derive immutable plan/evidence producer edges for invalidation."""

    product_producers: Dict[Tuple[str, str], Set[str]] = {}
    for step in plan.steps:
        for raw_output in step.expected_outputs or []:
            product = _typed_input_product(raw_output)
            if product is not None:
                product_producers.setdefault(product, set()).add(step.step_id)
    dependencies: Dict[str, Set[str]] = {}
    steps_by_id = {step.step_id: step for step in plan.steps}
    for record in current_records:
        step_id = str(record.get("step_id") or "").strip()
        deps = dependencies.setdefault(step_id, set())
        step = steps_by_id.get(step_id)
        if step is not None:
            for raw_input in step.inputs or []:
                product = _typed_input_product(raw_input)
                producers = product_producers.get(product or ("", ""), set())
                if len(producers) == 1:
                    deps.update(producers - {step_id})
        pending = [
            str(value).strip()
            for evidence_id in (record.get("evidence_ids") or [])
            if (authority := evidence_by_id.get(str(evidence_id).strip())) is not None
            for value in (_evidence_record_field(authority, "inputs") or [])
            if str(value).strip()
        ]
        seen: Set[str] = set()
        while pending:
            evidence_id = pending.pop()
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            authority = evidence_by_id.get(evidence_id)
            if authority is None:
                continue
            owner = str(_evidence_record_field(authority, "produced_by_step") or "")
            if owner and owner != step_id:
                deps.add(owner)
                continue
            pending.extend(
                str(value).strip()
                for value in (_evidence_record_field(authority, "inputs") or [])
                if str(value).strip()
            )
    return dependencies


def _evaluate_final_deterministic_gates(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    cohort_path: Path,
    universe_path: Path,
    run_dir: Path,
    out_dir: Path,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    step_record: Mapping[str, Any],
    completed_step_records: Sequence[Mapping[str, Any]],
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
    attempt_id: str,
    checkpoint_id: str,
    stat_validator: StatisticalValidator,
    clinical_validator: ClinicalConstraintValidator,
    statistical_guard: StatisticalGuard,
    cross_step_cohort_lock_validator: CrossStepCohortLockValidator,
    cross_step_registered_output_validator: CrossStepRegisteredOutputValidator,
    cross_step_reconciliation_trace_validator: CrossStepReconciliationTraceValidator,
    step_summary_integrity_validator: StepSummaryIntegrityValidator,
    step_summary_fraction_validator: StepSummaryFractionValidator,
    cross_step_source_status_validator: CrossStepSourceStatusValidator,
    primary_model_contract_validator: PrimaryModelContractValidator,
    figure_contract_validator: FigureContractQualityValidator,
    figure_source_validator: FigureSourceDataValidator,
) -> _FinalDeterministicGateFindings:
    """Evaluate the complete final deterministic review for one step attempt.

    This function deliberately does not append to the run-wide findings list,
    mutate ``step_record``, publish evidence, or decide the outer step status.
    Filesystem-reading validators make it only *pure-ish*, but all gate
    composition, compatibility demotions, and attempt binding now live here as
    one reusable authority.
    """

    execution_cohort_path = _step_execution_cohort_path(
        step=step,
        plan=plan,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
    )
    execution_cohort_path = _bound_step_execution_cohort_path(
        run_dir=run_dir,
        fallback_path=execution_cohort_path,
        resolved_input_bindings=resolved_input_bindings,
    )

    stat_findings = stat_validator.audit(
        context=context,
        cohort_path=execution_cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    clinical_findings = clinical_validator.audit(
        context=context,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    guard_findings = statistical_guard.audit(
        context=context,
        cohort_path=execution_cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    contract_findings = _step_deterministic_contract_findings(
        step=step,
        plan=plan,
        context=context,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
        resolved_input_bindings=resolved_input_bindings,
        out_dir=out_dir,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
        execution_cohort_path=execution_cohort_path,
        cross_step_cohort_lock_validator=cross_step_cohort_lock_validator,
        cross_step_registered_output_validator=cross_step_registered_output_validator,
        cross_step_reconciliation_trace_validator=(
            cross_step_reconciliation_trace_validator
        ),
        step_summary_integrity_validator=step_summary_integrity_validator,
        step_summary_fraction_validator=step_summary_fraction_validator,
        cross_step_source_status_validator=cross_step_source_status_validator,
        primary_model_contract_validator=primary_model_contract_validator,
    )
    contract_findings.extend(
        figure_contract_validator.audit(
            step=step,
            out_dir=out_dir,
            run_dir=run_dir,
            step_summary=step_summary,
        )
    )
    # Legacy deterministic-primary records own their historical core estimate;
    # only their excess step-output declarations are demoted.  All integrity,
    # exposure, leakage, and figure findings remain blocking.
    contract_findings = _demote_step_contract_for_primary_runner(
        step_record,
        step_summary,
        contract_findings,
    )
    # Some study-design families build the manuscript-facing multi-panel figure
    # in the write phase.  Preserve the existing narrow step-figure demotion;
    # the publication display-suite gate remains fail-closed.
    contract_findings = _demote_result_figure_shape_for_family_renderer(
        context,
        contract_findings,
    )
    figure_source_findings = figure_source_validator.audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
        resolved_input_bindings=resolved_input_bindings,
    )

    def _bind(
        group: Sequence[ValidationFinding],
    ) -> Tuple[ValidationFinding, ...]:
        return tuple(
            _bind_findings_to_step_attempt(
                group,
                step_id=step.step_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
            )
        )

    return _FinalDeterministicGateFindings(
        stat_findings=_bind(stat_findings),
        clinical_findings=_bind(clinical_findings),
        guard_findings=_bind(guard_findings),
        contract_findings=_bind(contract_findings),
        figure_source_findings=_bind(figure_source_findings),
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
    """Replay changed deterministic gates against sealed evidence only.

    The function runs before :class:`ResumeController` applies skip decisions.
    It never invokes the coder, runner, analyzer, or LLM concept auditor.
    Successful replay appends a new ``ok`` authority checkpoint; any error
    appends ``resume_validator_invalid`` and makes the step executable again.
    """

    state = dict(resume_state)
    authority_history = [
        dict(record)
        for record in (resume_state.get("per_step_records") or [])
        if isinstance(record, Mapping)
    ]
    saved_attempt_history = [
        dict(record)
        for record in (resume_state.get("step_attempt_history") or [])
        if isinstance(record, Mapping)
    ]
    history = saved_attempt_history or list(authority_history)
    for authority_record in authority_history:
        if authority_record not in history:
            history.append(authority_record)
    # Resume audit history is append-only and may contain a newer invalidation
    # than the compact outer authority view (for example after an interrupted
    # execution attempt rewrote only the latter).  Resolve latest-per-step from
    # the merged monotonic history so validator-invalid checkpoints cannot
    # disappear and fall through to a second initial-generation purchase.
    current_records = [dict(record) for record in current_step_records(history)]
    current_successes = [
        record
        for record in current_records
        if str(record.get("status") or "").strip().lower() == "ok"
    ]
    steps_by_id = {step.step_id: step for step in plan.steps}
    step_order = {"00_probe": -1, **{s.step_id: i for i, s in enumerate(plan.steps)}}
    seeded_invalidated = {
        str(record.get("step_id") or "").strip(): (
            "prior checkpoint already lacks current resume authority "
            f"(status={str(record.get('status') or '').strip().lower()})"
        )
        for record in current_records
        if str(record.get("status") or "").strip().lower()
        in {"resume_evidence_invalid", "resume_validator_invalid"}
    }
    if resume_from_step_id and seeded_invalidated:
        cut = step_order.get(resume_from_step_id)
        earlier_invalid = sorted(
            step_id
            for step_id in seeded_invalidated
            if cut is not None and step_order.get(step_id, cut) < cut
        )
        if earlier_invalid:
            raise RunInputIdentityError(
                "Cannot start resume after an already-invalid upstream "
                "authority; resume at or before: " + ", ".join(earlier_invalid)
            )
    stamp = _deterministic_gate_stamp()
    stale_successes = [
        record
        for record in current_successes
        if record.get("deterministic_gate_fingerprint")
        != stamp["deterministic_gate_fingerprint"]
    ]
    if not stale_successes and not seeded_invalidated:
        return _ResumeDeterministicRevalidationResult(state, (), ())

    evidence_records = list(evidence.records())
    evidence_by_id = {
        str(_evidence_record_field(record, "evidence_id") or ""): record
        for record in evidence_records
    }
    trusted_records, trusted_summary_errors = _trusted_resume_success_records(
        records=current_successes,
        evidence_by_id=evidence_by_id,
        run_dir=run_dir,
    )
    trusted_by_step = {
        str(record.get("step_id") or ""): record for record in trusted_records
    }
    current_by_step = {
        str(record.get("step_id") or ""): record for record in current_successes
    }
    dependencies = _resume_success_dependencies(
        plan=plan,
        current_records=current_records,
        evidence_by_id=evidence_by_id,
    )
    invalidated: Dict[str, str] = dict(seeded_invalidated)
    revalidated: List[str] = []
    invalid_payloads: Dict[str, Dict[str, Any]] = {}
    retirement_records: Dict[str, Mapping[str, Any]] = {}

    def attempt_identity(step_id: str) -> Tuple[str, str]:
        sequence = 1 + sum(
            1
            for record in history
            if str(record.get("step_id") or "") == step_id
            and record.get("revalidated_without_execution") is True
        )
        attempt_id = f"{step_id}:resume_revalidation:{sequence}"
        return attempt_id, f"{attempt_id}:deterministic_review"

    def indexed_alias_evidence_ids(prior_record: Mapping[str, Any]) -> List[str]:
        step_id = str(prior_record.get("step_id") or "").strip()
        indexed_ids: List[str] = []
        for raw_id in prior_record.get("evidence_ids") or []:
            evidence_id = str(raw_id).strip()
            authority = evidence_by_id.get(evidence_id)
            if (
                authority is not None
                and str(_evidence_record_field(authority, "produced_by_step") or "")
                == step_id
            ):
                indexed_ids.append(evidence_id)
        return list(dict.fromkeys(indexed_ids))

    for invalid_step_id in seeded_invalidated:
        prior_success = next(
            (
                record
                for record in reversed(history)
                if str(record.get("step_id") or "").strip() == invalid_step_id
                and str(record.get("status") or "").strip().lower() == "ok"
            ),
            None,
        )
        if prior_success is not None:
            retirement_records[invalid_step_id] = prior_success
            current_invalid = next(
                (
                    record
                    for record in reversed(history)
                    if str(record.get("step_id") or "").strip() == invalid_step_id
                    and str(record.get("status") or "").strip().lower()
                    in {"resume_evidence_invalid", "resume_validator_invalid"}
                ),
                None,
            )
            raw_capsule_ref = prior_success.get("step_authority_capsule_ref")
            prior_code_sha256 = str(
                prior_success.get("executed_code_sha256")
                or prior_success.get("concept_approved_code_sha256")
                or ""
            )
            if (
                isinstance(current_invalid, Mapping)
                and "resume_revalidation_candidate_capsule_ref" not in current_invalid
                and isinstance(raw_capsule_ref, Mapping)
                and re.fullmatch(r"[0-9a-f]{64}", prior_code_sha256)
            ):
                # Append a monotonic continuation for invalid checkpoints
                # written before recovery coordinates existed. Never mutate
                # the historical checkpoint or reset its provider receipt.
                history.append(
                    {
                        **dict(current_invalid),
                        "attempt_id": (
                            f"{str(current_invalid.get('attempt_id') or invalid_step_id)}"
                            ":candidate_recovery"
                        ),
                        "resume_revalidation_candidate_capsule_ref": dict(
                            raw_capsule_ref
                        ),
                        "resume_revalidation_candidate_code_sha256": (
                            prior_code_sha256
                        ),
                        "resume_revalidation_candidate_attempt_id": str(
                            prior_success.get("attempt_id") or ""
                        ),
                    }
                )

    def append_invalid(
        *,
        prior_record: Mapping[str, Any],
        reason: str,
        code_findings: Sequence[ValidationFinding] = (),
        gate_findings: Optional[_FinalDeterministicGateFindings] = None,
    ) -> None:
        step_id = str(prior_record.get("step_id") or "").strip()
        if step_id in invalidated:
            return
        attempt_id, checkpoint_id = attempt_identity(step_id)
        if not code_findings and gate_findings is None:
            code_findings = _bind_findings_to_step_attempt(
                [
                    ValidationFinding(
                        validator="resume_deterministic_revalidation",
                        severity="error",
                        message=(
                            f"Prior success for step {step_id} failed current "
                            "deterministic replay."
                        ),
                        detail={"reason": reason},
                    )
                ],
                step_id=step_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
            )
        payload: Dict[str, Any] = {
            "step_id": step_id,
            "status": "resume_validator_invalid",
            "revalidated_without_execution": True,
            "attempt_id": attempt_id,
            "review_checkpoint_id": checkpoint_id,
            "resume_invalidation_reason": reason,
            "invalidated_evidence_ids": list(prior_record.get("evidence_ids") or []),
            "evidence_ids": [],
            "deterministic_code_findings": [
                finding.model_dump(mode="json") for finding in code_findings
            ],
            "retired_current_aliases": {},
            **stamp,
        }
        raw_capsule_ref = prior_record.get("step_authority_capsule_ref")
        prior_code_sha256 = str(
            prior_record.get("executed_code_sha256")
            or prior_record.get("concept_approved_code_sha256")
            or ""
        )
        if isinstance(raw_capsule_ref, Mapping) and re.fullmatch(
            r"[0-9a-f]{64}", prior_code_sha256
        ):
            # Invalid status retires current authority, but this explicit
            # immutable coordinate lets the next attempt revalidate the exact
            # candidate without purchasing a second initial generation.
            payload.update(
                {
                    "resume_revalidation_candidate_capsule_ref": dict(raw_capsule_ref),
                    "resume_revalidation_candidate_code_sha256": (prior_code_sha256),
                    "resume_revalidation_candidate_attempt_id": str(
                        prior_record.get("attempt_id") or ""
                    ),
                }
            )
        for key, value in prior_record.items():
            if key.startswith("step_provider_call_") or key.startswith(
                "step_llm_repair_"
            ):
                payload[key] = value
        if gate_findings is not None:
            payload.update(
                {
                    "stat_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.stat_findings
                    ],
                    "clinical_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.clinical_findings
                    ],
                    "guard_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.guard_findings
                    ],
                    "contract_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.contract_findings
                    ],
                    "figure_source_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.figure_source_findings
                    ],
                }
            )
        invalidated[step_id] = reason
        invalid_payloads[step_id] = payload
        retirement_records[step_id] = prior_record
        history.append(payload)

    stale_successes.sort(
        key=lambda record: step_order.get(
            str(record.get("step_id") or ""), len(step_order)
        )
    )
    for prior_record in stale_successes:
        step_id = str(prior_record.get("step_id") or "").strip()
        invalid_upstream = sorted(
            dependencies.get(step_id, set()).intersection(invalidated)
        )
        if invalid_upstream:
            append_invalid(
                prior_record=prior_record,
                reason=(
                    "current success depends on invalidated upstream step(s): "
                    + ", ".join(invalid_upstream)
                ),
            )
            continue
        if step_id == "00_probe":
            summary_error = trusted_summary_errors.get(step_id)
            if summary_error is not None or step_id not in trusted_by_step:
                append_invalid(
                    prior_record=prior_record,
                    reason=(summary_error or "probe summary authority is unavailable"),
                )
                continue
            evidence_payloads = {
                evidence_id: (
                    record.model_dump(mode="json")
                    if hasattr(record, "model_dump")
                    else dict(record)
                )
                for evidence_id, record in evidence_by_id.items()
            }
            error = _host_probe_authority_error(
                record=prior_record,
                evidence_ids=list(prior_record.get("evidence_ids") or []),
                step_id=step_id,
                run_dir=run_dir,
                records=evidence_payloads,
            )
            if error is not None:
                append_invalid(prior_record=prior_record, reason=error)
                continue
            attempt_id, checkpoint_id = attempt_identity(step_id)
            summary = trusted_by_step[step_id]["step_summary"]
            replayed = {
                **prior_record,
                "status": "ok",
                "step_summary": dict(summary),
                "revalidated_without_execution": True,
                "attempt_id": attempt_id,
                "review_checkpoint_id": checkpoint_id,
                **stamp,
            }
            _discard_stale_resolved_input_receipts(replayed)
            history.append(replayed)
            trusted_by_step[step_id] = replayed
            revalidated.append(step_id)
            continue

        is_host_cohort_materializer = (
            str(prior_record.get("generation_mode") or "").strip().lower()
            == _HOST_COHORT_MATERIALIZER_GENERATION_MODE
            or prior_record.get("step_authority_kind")
            == _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
        )
        if is_host_cohort_materializer:
            evidence_payloads = {
                evidence_id: (
                    record.model_dump(mode="json")
                    if hasattr(record, "model_dump")
                    else dict(record)
                )
                for evidence_id, record in evidence_by_id.items()
            }
            error = _host_cohort_materializer_authority_error(
                record=prior_record,
                evidence_ids=list(prior_record.get("evidence_ids") or []),
                step_id=step_id,
                run_dir=run_dir,
                records=evidence_payloads,
            )
            if error is not None:
                append_invalid(prior_record=prior_record, reason=error)
                continue
            attempt_id, checkpoint_id = attempt_identity(step_id)
            replayed = {
                **prior_record,
                "status": "ok",
                "step_summary": dict(prior_record["step_summary"]),
                "revalidated_without_execution": True,
                "attempt_id": attempt_id,
                "review_checkpoint_id": checkpoint_id,
                **stamp,
            }
            _discard_stale_resolved_input_receipts(replayed)
            history.append(replayed)
            trusted_by_step[step_id] = replayed
            revalidated.append(step_id)
            continue

        step = steps_by_id.get(step_id)
        summary_error = trusted_summary_errors.get(step_id)
        if step is None or summary_error is not None:
            append_invalid(
                prior_record=prior_record,
                reason=(summary_error or "successful step is absent from active plan"),
            )
            continue
        trusted_record = trusted_by_step[step_id]
        attempt_id, checkpoint_id = attempt_identity(step_id)
        try:
            _verify_resume_step_script_lineage(
                record=prior_record,
                evidence_by_id=evidence_by_id,
            )
            _, script_path = _verified_explicit_step_authority(
                record=prior_record,
                field="script_evidence_id",
                expected_kind="code",
                expected_source_name=None,
                evidence_by_id=evidence_by_id,
                run_dir=run_dir,
            )
            script_text = script_path.read_text(encoding="utf-8")
            code_findings = _bind_findings_to_step_attempt(
                _deterministic_code_gate_findings(
                    context=context,
                    step=step,
                    script_text=script_text,
                ),
                step_id=step_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
            )
            if any(finding.severity == "error" for finding in code_findings):
                append_invalid(
                    prior_record=prior_record,
                    reason="current deterministic code preflight failed",
                    code_findings=code_findings,
                )
                continue
            trusted_current_records = [
                record
                for record in trusted_by_step.values()
                if str(record.get("status") or "").lower() == "ok"
                and str(record.get("step_id") or "") not in invalidated
            ]
            resolved_bindings, resolved_input_evidence_ids = (
                _resume_typed_input_bindings(
                    step=step,
                    plan=plan,
                    evidence_records=evidence_records,
                    trusted_step_records=trusted_current_records,
                    run_dir=run_dir,
                    cohort_path=cohort_path,
                    development_sample=development_sample,
                )
            )
            with tempfile.TemporaryDirectory(
                prefix=f".resume_gate_{step_id}_",
                dir=run_dir,
            ) as temporary_root:
                replay_out_dir = Path(temporary_root) / "outputs"
                materialized_outputs = _materialize_verified_step_output_view(
                    record=prior_record,
                    evidence_by_id=evidence_by_id,
                    run_dir=run_dir,
                    destination=replay_out_dir,
                )
                replay_step_summary = _project_verified_replay_output_paths(
                    trusted_record["step_summary"],
                    materialized_evidence_by_source_name=materialized_outputs,
                )
                completed_records = [
                    record
                    for record in trusted_current_records
                    if str(record.get("step_id") or "") != step_id
                    and step_order.get(str(record.get("step_id") or ""), -1)
                    < step_order.get(step_id, len(step_order))
                ]
                gate_findings = _evaluate_final_deterministic_gates(
                    context=context,
                    plan=plan,
                    cohort_path=cohort_path,
                    universe_path=universe_path,
                    run_dir=run_dir,
                    out_dir=replay_out_dir,
                    step=step,
                    step_summary=replay_step_summary,
                    step_record=prior_record,
                    completed_step_records=completed_records,
                    resolved_input_bindings=resolved_bindings,
                    attempt_id=attempt_id,
                    checkpoint_id=checkpoint_id,
                    stat_validator=StatisticalValidator(),
                    clinical_validator=ClinicalConstraintValidator(),
                    statistical_guard=StatisticalGuard(),
                    cross_step_cohort_lock_validator=CrossStepCohortLockValidator(),
                    cross_step_registered_output_validator=(
                        CrossStepRegisteredOutputValidator()
                    ),
                    cross_step_reconciliation_trace_validator=(
                        CrossStepReconciliationTraceValidator()
                    ),
                    step_summary_integrity_validator=StepSummaryIntegrityValidator(),
                    step_summary_fraction_validator=StepSummaryFractionValidator(),
                    cross_step_source_status_validator=CrossStepSourceStatusValidator(),
                    primary_model_contract_validator=PrimaryModelContractValidator(),
                    figure_contract_validator=FigureContractQualityValidator(),
                    figure_source_validator=FigureSourceDataValidator(),
                )
            if any(
                finding.severity == "error" for finding in gate_findings.all_findings()
            ):
                append_invalid(
                    prior_record=prior_record,
                    reason="current deterministic artifact gates failed",
                    code_findings=code_findings,
                    gate_findings=gate_findings,
                )
                continue
            prior_critique = prior_record.get("critique_report")
            prior_critique_status = (
                str(prior_critique.get("status") or "").strip().lower()
                if isinstance(prior_critique, Mapping)
                else ""
            )
            if prior_critique_status in {"blocked", "needs_revision"}:
                append_invalid(
                    prior_record=prior_record,
                    reason=(
                        "prior deterministic Critic status remains "
                        f"{prior_critique_status}"
                    ),
                    code_findings=code_findings,
                    gate_findings=gate_findings,
                )
                continue
            evidence_refs = [
                EvidenceRef(
                    evidence_id=str(_evidence_record_field(authority, "evidence_id")),
                    kind=_evidence_record_field(authority, "kind"),
                    description=str(
                        _evidence_record_field(authority, "description") or ""
                    ),
                    relative_path=str(
                        _evidence_record_field(authority, "relative_path") or ""
                    ),
                )
                for evidence_id in (prior_record.get("evidence_ids") or [])
                if (authority := evidence_by_id.get(str(evidence_id))) is not None
                and verified_run_evidence_path(run_dir, authority) is not None
            ]
            critique = CriticAgent().review_step(
                step=step,
                step_summary=dict(trusted_record["step_summary"]),
                evidence_refs=evidence_refs,
                findings=_actionable_validator_messages(
                    code_findings,
                    gate_findings.all_findings(),
                ),
            )
            if critique.status != "pass":
                append_invalid(
                    prior_record=prior_record,
                    reason=f"current deterministic Critic status={critique.status}",
                    code_findings=code_findings,
                    gate_findings=gate_findings,
                )
                continue
        except (OSError, TypeError, UnicodeError, ValueError) as exc:
            append_invalid(
                prior_record=prior_record,
                reason=f"{type(exc).__name__}: {exc}",
            )
            continue

        replayed = {
            **prior_record,
            "status": "ok",
            "step_summary": dict(trusted_record["step_summary"]),
            "resolved_input_evidence_ids": resolved_input_evidence_ids,
            "deterministic_code_findings": [
                finding.model_dump(mode="json") for finding in code_findings
            ],
            "stat_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.stat_findings
            ],
            "clinical_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.clinical_findings
            ],
            "guard_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.guard_findings
            ],
            "contract_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.contract_findings
            ],
            "figure_source_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.figure_source_findings
            ],
            "critique_report": critique.model_dump(mode="json"),
            "revalidated_without_execution": True,
            "attempt_id": attempt_id,
            "review_checkpoint_id": checkpoint_id,
            **stamp,
        }
        _discard_stale_resolved_input_receipts(replayed)
        replayed["revalidated_input_bindings_fingerprint"] = (
            _resume_typed_input_bindings_fingerprint(resolved_bindings)
        )
        history.append(replayed)
        trusted_by_step[step_id] = replayed
        revalidated.append(step_id)

    # Propagate invalid authority through immutable plan/evidence edges, even
    # when a downstream success already carries the current fingerprint.
    while True:
        changed = False
        for step_id, prior_record in current_by_step.items():
            if step_id in invalidated:
                continue
            failed_dependencies = sorted(
                dependencies.get(step_id, set()).intersection(invalidated)
            )
            if not failed_dependencies:
                continue
            append_invalid(
                prior_record=prior_record,
                reason=(
                    "current success depends on invalidated upstream step(s): "
                    + ", ".join(failed_dependencies)
                ),
            )
            changed = True
        if not changed:
            break

    # An explicit cut after a newly detected invalid authority must fail
    # before any alias or manifest mutation.  The caller can restart at or
    # before the earliest invalid upstream step.
    if resume_from_step_id and invalidated:
        cut = step_order.get(resume_from_step_id)
        earlier_invalid = sorted(
            step_id
            for step_id in invalidated
            if cut is not None and step_order.get(step_id, cut) < cut
        )
        if earlier_invalid:
            raise RunInputIdentityError(
                "Cannot start resume after deterministic-validator-invalid "
                "upstream evidence; resume at or before: " + ", ".join(earlier_invalid)
            )

    if invalid_payloads:
        state_findings = list(resume_state.get("findings") or [])
        for step_id, payload in invalid_payloads.items():
            reason = str(payload.get("resume_invalidation_reason") or "")
            state_findings.append(
                ValidationFinding(
                    validator="resume_deterministic_revalidation",
                    severity="warning",
                    message=(
                        f"Prior success for step {step_id} was invalidated by "
                        "current deterministic gates and requires re-execution."
                    ),
                    detail={
                        "step_id": step_id,
                        "reason": reason,
                        "requires_reexecution": True,
                    },
                ).model_dump(mode="json")
            )
        state["findings"] = state_findings
    state["step_attempt_history"] = history
    state["per_step_records"] = [
        dict(record) for record in current_step_records(history)
    ]

    retirement_batch = {
        step_id: evidence_ids
        for step_id, prior_record in retirement_records.items()
        if (evidence_ids := indexed_alias_evidence_ids(prior_record))
    }
    current_aliases = evidence.aliases() if retirement_batch else {}
    for step_id, evidence_ids in retirement_batch.items():
        payload = invalid_payloads.get(step_id)
        if payload is not None:
            payload["retired_current_aliases"] = {
                alias: evidence_id
                for alias, evidence_id in current_aliases.items()
                if evidence_id in set(evidence_ids)
            }

    # Persist the append-only checkpoint before revoking aliases.  A failed
    # checkpoint write leaves aliases untouched; the batch retirement itself
    # is atomic across every invalid step.  If retirement fails, restore the
    # prior manifest so the two authority ledgers cannot disagree.
    checkpoint_path = run_dir / "manifest_partial.json"
    write_run_checkpoint(checkpoint_path, state)
    if retirement_batch:
        try:
            evidence.retire_steps_current_aliases(retirement_batch)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            try:
                write_run_checkpoint(checkpoint_path, resume_state)
            except (OSError, TypeError, ValueError) as rollback_exc:
                raise RuntimeError(
                    "resume revalidation alias retirement and manifest rollback "
                    "both failed"
                ) from rollback_exc
            raise RuntimeError(
                "resume revalidation alias retirement failed; manifest was rolled back"
            ) from exc

    return _ResumeDeterministicRevalidationResult(
        resume_state=state,
        revalidated_step_ids=tuple(revalidated),
        invalidated_step_ids=tuple(sorted(invalidated)),
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
    plan, companion_input_findings = _augment_measurement_companion_inputs(
        plan=plan,
        context=context,
    )
    if companion_input_findings:
        findings.extend(companion_input_findings)
        plan_path = run_dir / "analysis_plan_input_closure.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        if evidence.get("analysis_plan_input_closure") is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Analysis plan with structural measurement-provenance "
                    "input closure."
                ),
                source_path=plan_path,
                evidence_id="analysis_plan_input_closure",
                producer="runtime_supervisor",
                generation_mode="system",
                prompt_pack_version=plan_result.prompt_version,
                metadata={"reason": "measurement_companion_input_closure"},
            )
        plan_result.plan_path = plan_path
    stop_after_step_id = _resolve_stop_after_step_selector(plan, stop_after_step_id)
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
    llm_concept_audit_client = pipeline._llm_concept_auditor_client or role_resolver(
        "analyzer"
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
        build_environment_identity(llm_signature=llm_concept_auditor_signature)
    )
    prompt_version = plan_result.prompt_version
    prompt_files = plan_result.prompt_files
    assert_cohort_definition_locked(run_dir=run_dir, plan=plan)
    assert_robustness_specs_locked(run_dir=run_dir, plan=plan)

    # Dual-track cohort. If the plan phase materialised the locked cohort
    # definition into a filtered analysis cohort, every downstream consumer
    # (probe, statistical validators, robustness fitter, and the step runner)
    # reads THAT — so the declared inclusion/exclusion is enforced once,
    # consistently, instead of being silently re-implemented (or skipped) by
    # each generated step. The full universe stays reachable via the runner's
    # EASYICU_UNIVERSE_PARQUET env for explicit robustness steps.
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

    coder_llm_client = role_resolver("coder")
    fallback_coder_provider_identity_sha256 = canonical_sha256(
        pipeline._llm_signature(coder_llm_client)
    )
    coder = CoderAgent(coder_llm_client)
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
    analyzer = AnalyzerAgent(role_resolver("analyzer"))
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

    def _flush_partial_manifest(extra: Optional[Dict[str, Any]] = None) -> None:
        for record in per_step_records:
            snapshot = dict(record)
            if snapshot not in step_attempt_history:
                step_attempt_history.append(snapshot)
        payload: Dict[str, Any] = {
            "schema_version": "easyicu.research_manifest_partial/1",
            "run_id": run_id,
            "research_question": context.research_question,
            "started_at": plan_result.started_at.isoformat(),
            "context_path": str(plan_result.context_path.relative_to(run_dir)),
            "plan_path": str(plan_path.relative_to(run_dir)),
            "evidence": [r.model_dump(mode="json") for r in evidence.records()],
            "findings": [f.model_dump(mode="json") for f in findings],
            "per_step_records": per_step_records,
            "step_attempt_history": step_attempt_history,
            "llm_signature": llm_signature,
            "used_mock_llm": plan_result.used_mock_llm,
            "prompt_pack_version": prompt_version,
            "prompt_pack_files": prompt_files,
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

    def _register_plan_revision(
        revised_plan: AnalysisPlan,
        *,
        reason: str,
    ) -> Path:
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
                producer="replanner",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
                metadata={"reason": reason, "llm_signature": llm_signature},
            )
        except ValueError:
            # Resume + replan can legitimately re-emit the same revision number
            # with different content (the replanner is non-deterministic across
            # runs), which collides with the prior run's
            # ``analysis_plan_revision_N`` id. Keep both by versioning the id
            # with a content digest instead of crashing the resumed run. The
            # global evidence-id collision guard stays intact for every other
            # artefact.
            import hashlib

            digest = hashlib.sha256(revision_path.read_bytes()).hexdigest()[:8]
            evidence.register_file(
                kind="log",
                description=(
                    f"Revised analysis plan (reason={reason}; resume re-revision)."
                ),
                source_path=revision_path,
                evidence_id=f"{base_id}_{digest}",
                producer="replanner",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
                metadata={
                    "reason": reason,
                    "llm_signature": llm_signature,
                    "resume_reregistration": True,
                },
            )
        return revision_path

    def _no_analysis_step_has_run() -> bool:
        """True while only the deterministic probe (00_probe) has executed.

        The cohort may be (re)materialised and the runner re-pointed only at
        this point; switching the cohort after analysis steps already ran on
        the universe would split a single run across two populations.
        """
        return not any(
            (rec.get("step_id") or "") != "00_probe" for rec in per_step_records
        )

    def _universe_columns() -> list:
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

    def _try_materialize_cohort_from_prose(
        candidate_plan: AnalysisPlan,
        *,
        reason: str,
    ) -> bool:
        """Extract the agent's prose 纳排 into typed predicates, materialise the
        filtered analysis cohort, and re-point the runner at it.

        Returns ``True`` when the cohort was materialised (so the caller skips
        the auditable contract error). The locked initial cohort was an empty
        placeholder for the bench's 0-step plan; locking the first real
        definition here is a provisional→real lock, fully provenance-recorded.
        """
        nonlocal cohort_path, runner
        if _replan_state["cohort_materialized"]:
            return True
        if run_input_authority_state.analysis_path.exists():
            return True
        if not _no_analysis_step_has_run():
            return False
        columns = _universe_columns()
        if not columns:
            return False
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
                    llm=role_resolver("planner"),
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
            return False
        _replan_state["cohort_translation_provider_budget"] = budget_snapshot
        if definition is None:
            return False
        candidate_plan.cohort = definition
        try:
            write_locked_cohort_definition(
                run_dir=run_dir,
                plan=candidate_plan,
                evidence=evidence,
                prompt_pack_version=prompt_version,
                llm_signature=llm_signature,
                allow_empty_promotion=True,
            )
            result = materialize_locked_analysis_cohort(
                run_dir=run_dir,
                plan=candidate_plan,
                universe_path=universe_path,
                context=context,
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
            return False
        if result.get("status") != "applied":
            return False
        run_input_authority_state.rebind_cohort(
            plan=candidate_plan,
            context=context,
        )
        cohort_path = run_input_authority_state.selected_path
        cohort_product_steps = [
            step
            for step in candidate_plan.steps
            if _declares_host_cohort_only_product(step)
        ]
        cohort_product_step = (
            cohort_product_steps[0] if len(cohort_product_steps) == 1 else None
        )
        try:
            materialized_authority_ref = result.get("authority_ref")
            cohort_definition_sha256 = result.get("cohort_definition_sha256")
            cohort_metadata = {
                "llm_signature": llm_signature,
                "reason": reason,
            }
            if materialized_authority_ref is not None:
                cohort_metadata.update(
                    {
                        "materialized_cohort_authority_ref": (
                            materialized_authority_ref
                        ),
                        "cohort_definition_sha256": cohort_definition_sha256,
                    }
                )
            cohort_record = evidence.register_file(
                kind="table",
                description=(
                    "Analysis cohort materialised from the agent's prose 纳排, "
                    "translated to typed CTAS predicates during execution."
                ),
                source_path=cohort_path,
                evidence_id="analysis_cohort_execute_repair",
                produced_by_step=(
                    cohort_product_step.step_id if cohort_product_step else None
                ),
                producer="cohort_repair",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
                metadata=cohort_metadata,
            )
        except ValueError:
            cohort_record = evidence.get("analysis_cohort_execute_repair")
        if cohort_product_step is not None and cohort_record is not None:
            # The deterministic materialiser has completely realised this
            # single-product step using the cohort the Agent selected.  Record
            # that product under the planned producer and do not ask the Coder
            # to recreate or reinterpret the cohort scientifically.
            cohort_checkpoint = {
                "step_id": cohort_product_step.step_id,
                "intent": cohort_product_step.intent,
                "planned_analysis_role": cohort_product_step.planned_analysis_role,
                "analysis_request": {
                    "step": cohort_product_step.model_dump(mode="json")
                },
                "status": "ok",
                "generation_mode": _HOST_COHORT_MATERIALIZER_GENERATION_MODE,
                "step_authority_kind": _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND,
                _HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD: (cohort_record.evidence_id),
                "step_summary": {
                    "output_files": {
                        "table:analysis_cohort": str(cohort_path.relative_to(run_dir))
                    },
                    "n_universe": int(result["n_universe"]),
                    "n_analysis_cohort": int(result["n_cohort"]),
                },
                "evidence_ids": [cohort_record.evidence_id],
                **_deterministic_gate_stamp(),
            }
            if materialized_authority_ref is not None:
                cohort_checkpoint["step_summary"].update(
                    {
                        "materialized_cohort_authority_ref": (
                            materialized_authority_ref
                        ),
                        "cohort_definition_sha256": cohort_definition_sha256,
                    }
                )
            if budget_owner_step_id == cohort_product_step.step_id:
                cohort_checkpoint.update(
                    {
                        key: value
                        for key, value in budget_snapshot.items()
                        if key != "budget_owner_step_id"
                    }
                )
            cohort_authority_error = _host_cohort_materializer_authority_error(
                record=cohort_checkpoint,
                evidence_ids=[cohort_record.evidence_id],
                step_id=cohort_product_step.step_id,
                run_dir=run_dir,
                records={
                    record.evidence_id: record.model_dump(mode="json")
                    for record in evidence.records()
                },
            )
            if cohort_authority_error is None:
                per_step_records.append(cohort_checkpoint)
                preexecuted_step_ids.add(cohort_product_step.step_id)
            else:
                findings.append(
                    ValidationFinding(
                        validator="cohort_materializer_authority",
                        severity="error",
                        message=(
                            "Host cohort materializer could not seal its exact "
                            "single-product authority."
                        ),
                        detail={
                            "step_id": cohort_product_step.step_id,
                            "reason": cohort_authority_error,
                        },
                    )
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
                    "(COHORT_PARQUET); the full universe stays available as "
                    "EASYICU_UNIVERSE_PARQUET."
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
        return True

    def _enforce_cohort_contract_on_executing_plan(
        candidate_plan: AnalysisPlan,
        *,
        reason: str,
    ) -> None:
        """Re-check the structured-纳排 contract against the plan that actually
        executes.

        The plan-phase contract (``pipeline._run_plan_phase``) only sees the
        *initial* plan. For non-deterministic providers that initial plan is
        commonly a 0-step shell, and the real plan — which carries a
        cohort-definition step but leaves ``plan.cohort`` structurally empty —
        is grown here by the replanner. Without this re-check the contract is
        bypassed and downstream steps silently run on the unfiltered universe
        while each generated step re-applies 纳排 inconsistently (run12).

        Emitted once, as an auditable error, and only when the locked cohort
        was *not* materialised into a filtered analysis cohort (an applied
        definition already enforces 纳排 on the data).
        """
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

    def _resolve_cohort_definition(
        candidate_plan: AnalysisPlan,
        *,
        reason: str,
    ) -> None:
        """For an executing plan that implies a cohort but left it unstructured:
        first try to materialise it from the step prose (real enforcement); if
        that fails, surface the auditable contract error (visibility)."""
        if not (
            _plan_expects_analysis_cohort(candidate_plan)
            and _cohort_definition_is_empty(candidate_plan)
        ):
            return
        if _try_materialize_cohort_from_prose(candidate_plan, reason=reason):
            return
        _enforce_cohort_contract_on_executing_plan(candidate_plan, reason=reason)

    _resolve_cohort_definition(plan, reason="execute_start")

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
        if not pipeline._enable_replanning or skill_obj is not None:
            return current_plan
        if _replan_state["disabled"] and not force:
            # A convergence guard already tripped earlier in this run; stop
            # paying for replanner calls that cannot change the outcome. A
            # ``force``d directed replan (bounded by its own caller-side budget)
            # bypasses this — it carries a new instruction the replanner has not
            # yet seen, so the prior no-op/budget verdict does not apply.
            return current_plan
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
            return current_plan
        replanner = ReplannerAgent(role_resolver("planner"))
        try:
            revised = replanner.run(
                context=plan_result.agent_context,
                current_plan=current_plan,
                probe_summary=probe_summary_payload,
                completed_step_records=completed_records,
                directive=directive,
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
            return current_plan
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
        findings.extend(normalized_candidate.findings)

        # The typed authority result owns only candidate normalization. The
        # orchestration state transition and any durable registration stay here.
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
            return current_plan

        # Substantive revision: reset the no-op streak and register it.
        _replan_state["noop_streak"] = 0
        _replan_state["total"] += 1
        plan_path = _register_plan_revision(revised, reason=reason)
        plan_result.plan_path = plan_path
        _resolve_cohort_definition(revised, reason=reason)
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
            # Fail closed: reaching the replan budget without the plan
            # converging is a runaway loop, not a clean run. The run is
            # demoted to diagnostic_only so a non-converging replan cascade
            # cannot launder a manuscript. The trigger is kept in ``detail``
            # (never the message) so a step-id-shaped reason cannot make the
            # readiness supersession rule drop this run-level latch.
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
        return revised

    trajectory_plan_blocked = False
    typed_plan_dag_blocked = False
    probe_step_id = "00_probe"
    if pipeline._enable_probe_step and probe_step_id not in resumed_step_ids:
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
        typed_plan_preflight = _typed_plan_dag_findings(plan)
        primary_cohort_preflight = primary_analysis_cohort_plan_findings(plan=plan)
        trajectory_preflight = trajectory_plan_dag_findings(
            plan=plan,
            context=context,
        )
        trajectory_directive = None
        typed_plan_directive = None
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
        plan = _maybe_replan(
            current_plan=plan,
            reason="probe_summary",
            probe_summary_payload=probe_summary,
            completed_records=[probe_record],
            directive="\n\n".join(
                directive
                for directive in (
                    typed_plan_directive,
                    primary_cohort_directive,
                    trajectory_directive,
                )
                if directive
            )
            or None,
            force=bool(
                typed_plan_preflight or primary_cohort_preflight or trajectory_preflight
            ),
        )

    final_typed_plan_findings = [
        *_typed_plan_dag_findings(plan),
        *primary_analysis_cohort_plan_findings(plan=plan),
    ]
    if final_typed_plan_findings:
        typed_plan_dag_blocked = True
        findings.extend(final_typed_plan_findings)
        _flush_partial_manifest(
            {
                "typed_plan_dag_blocked": True,
                "typed_plan_dag_error_count": len(final_typed_plan_findings),
            }
        )

    final_trajectory_plan_findings = trajectory_plan_dag_findings(
        plan=plan,
        context=context,
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

    def _record_repair(
        *,
        repair_id: str,
        step_id: str,
        trigger: Dict[str, Any],
        transformation: str,
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
            # P1: a runtime invariant that was actually checked and failed is a
            # non-blocking warning in soft mode; P2 will escalate this to a
            # fail-closed block for STRUCTURAL / CONTRACT_FILL repairs.
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

    def _automatic_repair_authorized(
        repair_id: str,
        *,
        step: AnalysisStep,
        source: str,
        before_code: Optional[str] = None,
        after_code: Optional[str] = None,
        sealed_renderer_wrapper: bool = False,
    ) -> bool:
        """Apply the central no-auto-method-substitution policy.

        Code rewrites and artifact/rendering transforms share this boundary. A
        staged figure may be built speculatively, but it is not installed into
        the live step unless this policy authorizes its typed repair id.
        """

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
        _record_repair(
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

    def _authorize_automatic_repair(
        repair: Optional[Tuple[str, str]],
        *,
        step: AnalysisStep,
        source: str,
        before_code: str,
        sealed_renderer_wrapper: bool = False,
    ) -> Optional[Tuple[str, str]]:
        """Authorize a generated code repair before assigning live code."""

        if repair is None:
            return None
        repair_id, candidate_code = repair
        if not _automatic_repair_authorized(
            repair_id,
            step=step,
            source=source,
            before_code=before_code,
            after_code=candidate_code,
            sealed_renderer_wrapper=sealed_renderer_wrapper,
        ):
            return None
        return repair

    def _propagate_findings_to_evidence(
        evidence_ids: Sequence[str],
        findings_for_step: Sequence[ValidationFinding],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Delegates to the module-level ``scope_findings_to_records`` so the
        # caveat-scoping rule (targeted taint + step-global-error fail-closed,
        # step-global warnings stay advisory) is unit-testable in isolation.
        scoped = scope_findings_to_records(evidence_ids, findings_for_step)
        for evidence_id in evidence_ids:
            severity, messages = scoped[str(evidence_id)]
            evidence.update_record(
                evidence_id,
                finding_severity=severity,
                finding_messages=messages,
                metadata=metadata,
            )

    def _validator_messages(
        *finding_groups: Sequence[ValidationFinding],
    ) -> List[str]:
        return _actionable_validator_messages(*finding_groups)

    def _failed_dependency_record(step: AnalysisStep) -> Optional[Dict[str, Any]]:
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

    def _execute_one_step(step: AnalysisStep) -> Dict[str, Any]:
        nonlocal runtime_state
        with shared_lock:
            resume_history = (
                list(
                    plan_result.resume_state.get("step_attempt_history")
                    or plan_result.resume_state.get("per_step_records")
                    or []
                )
                if isinstance(plan_result.resume_state, Mapping)
                else []
            )
            candidate_history = resume_history + list(per_step_records)
            prior_attempt_records = [
                record
                for record in candidate_history
                if isinstance(record, Mapping)
                and str(record.get("step_id") or "") == step.step_id
            ]
            prior_step_record = next(
                (
                    record
                    for record in current_step_records(prior_attempt_records)
                    if str(record.get("step_id") or "") == step.step_id
                ),
                None,
            )
        prior_attempt_sequences = [
            int(record.get("attempt_sequence"))
            for record in prior_attempt_records
            if isinstance(record.get("attempt_sequence"), int)
            and int(record.get("attempt_sequence")) >= 1
        ]
        attempt_sequence = (
            max(prior_attempt_sequences, default=len(prior_attempt_records)) + 1
        )
        attempt_id = f"{run_id}:{step.step_id}:{attempt_sequence}"
        review_checkpoint_id = f"{attempt_id}:deterministic_review"
        step_record: Dict[str, Any] = {
            "step_id": step.step_id,
            "intent": step.intent,
            "planned_analysis_role": step.planned_analysis_role,
            "attempt_id": attempt_id,
            "attempt_sequence": attempt_sequence,
            "review_checkpoint_id": review_checkpoint_id,
            "plan_scientific_signature": (
                _serializable_plan_scientific_scope_signature(plan)
            ),
        }
        step_execution_cohort_path = _step_execution_cohort_path(
            step=step,
            plan=plan,
            run_dir=run_dir,
            universe_path=universe_path,
            cohort_path=cohort_path,
        )
        primary_cohort_uses_universe = step_execution_cohort_path == universe_path
        if primary_cohort_uses_universe:
            step_record.update(
                {
                    "execution_cohort_role": (
                        "raw_universe_for_primary_analysis_cohort_producer"
                    ),
                    "execution_cohort_sha256": sha256_of_file(universe_path),
                    "authoritative_analysis_cohort_sha256": sha256_of_file(cohort_path),
                }
            )
        elif primary_analysis_cohort_producer_uses_universe(step=step, plan=plan):
            step_record.update(
                {
                    "execution_cohort_role": (
                        DEVELOPMENT_PRIMARY_COHORT_CONFIRMATION_ROLE
                    ),
                    "execution_cohort_sha256": sha256_of_file(cohort_path),
                    "authoritative_analysis_cohort_sha256": sha256_of_file(cohort_path),
                    "paper_authority": False,
                }
            )
        (
            step_llm_repair_attempts,
            prior_repair_classes,
            repair_history_invalid,
        ) = _monotonic_step_llm_repair_history(
            prior_attempt_records,
            limit=pipeline._max_step_llm_repair_attempts,
        )
        if step_llm_repair_attempts:
            step_record["step_llm_repair_attempts"] = step_llm_repair_attempts
            step_record["step_llm_repair_budget"] = (
                pipeline._max_step_llm_repair_attempts
            )
        if prior_repair_classes:
            step_record["step_llm_repair_classes"] = list(prior_repair_classes)
        if repair_history_invalid:
            step_record["step_llm_repair_history_invalid"] = True
            step_record["step_llm_repair_budget_exhausted"] = True
        configured_provider_limit = pipeline._max_step_provider_calls
        effective_provider_limit = configured_provider_limit
        reserved_final_category = (
            "concept_audit" if pipeline._enable_llm_concept_audit else None
        )
        provider_receipt_path = provider_call_budget_receipt_path(
            run_dir,
            step_id=step.step_id,
        )
        provider_receipt_relative_path = str(provider_receipt_path.relative_to(run_dir))
        prior_provider_categories: tuple[str, ...] = ()
        prior_logical_repair_entries: tuple[Dict[str, object], ...] = ()
        prior_initial_generation_entry: Optional[Dict[str, object]] = None
        prior_required_reservation_token: Optional[str] = None
        prior_reservation_bound_provider_history_len: Optional[int] = None
        prior_completed_reservation_token: Optional[str] = None
        prior_reservation_released = False
        prior_provider_attempts = 0
        provider_receipt_integrity_error: Optional[str] = None
        prior_snapshot_present = False
        if isinstance(prior_step_record, Mapping):
            snapshot_keys = {
                "step_provider_call_budget",
                "step_provider_call_attempts",
                "step_provider_call_categories",
            }
            prior_snapshot_present = any(
                key in prior_step_record for key in snapshot_keys
            )
            if prior_snapshot_present:
                prior_limit = prior_step_record.get("step_provider_call_budget")
                prior_attempts_raw = prior_step_record.get(
                    "step_provider_call_attempts"
                )
                prior_categories_raw = prior_step_record.get(
                    "step_provider_call_categories"
                )
                if (
                    isinstance(prior_limit, bool)
                    or not isinstance(prior_limit, int)
                    or prior_limit < 0
                    or isinstance(prior_attempts_raw, bool)
                    or not isinstance(prior_attempts_raw, int)
                    or prior_attempts_raw < 0
                    or not isinstance(prior_categories_raw, list)
                ):
                    provider_receipt_integrity_error = (
                        "Prior provider-call budget snapshot is incomplete or invalid."
                    )
                else:
                    normalized_categories = tuple(
                        str(item).strip() for item in prior_categories_raw
                    )
                    if any(
                        not item for item in normalized_categories
                    ) or prior_attempts_raw != len(normalized_categories):
                        provider_receipt_integrity_error = "Prior provider-call attempts and category history disagree."
                    else:
                        prior_provider_attempts = prior_attempts_raw
                        prior_provider_categories = normalized_categories
                        effective_provider_limit = min(
                            effective_provider_limit,
                            prior_limit,
                        )

        if provider_receipt_integrity_error is None and provider_receipt_path.exists():
            try:
                receipt_state = load_provider_call_budget_state(
                    provider_receipt_path,
                    step_id=step.step_id,
                    expected_reserved_final_category=reserved_final_category,
                )
                receipt_limit = receipt_state.limit
                receipt_categories = receipt_state.categories
                prior_logical_repair_entries = receipt_state.logical_repairs
                prior_initial_generation_entry = receipt_state.initial_generation
                prior_required_reservation_token = (
                    receipt_state.required_reservation_token
                )
                prior_reservation_bound_provider_history_len = (
                    receipt_state.reservation_bound_provider_history_len
                )
                prior_completed_reservation_token = (
                    receipt_state.completed_reservation_token
                )
                prior_reservation_released = receipt_state.reservation_released
                effective_provider_limit = min(
                    effective_provider_limit,
                    receipt_limit,
                )
                if prior_snapshot_present and (
                    len(receipt_categories) < len(prior_provider_categories)
                    or receipt_categories[: len(prior_provider_categories)]
                    != prior_provider_categories
                ):
                    raise ProviderCallBudgetReceiptError(
                        "Durable provider-call receipt conflicts with the latest "
                        "step snapshot."
                    )
                prior_provider_categories = receipt_categories
                prior_provider_attempts = len(receipt_categories)
            except ProviderCallBudgetReceiptError as exc:
                provider_receipt_integrity_error = str(exc)
        elif (
            provider_receipt_integrity_error is None
            and isinstance(prior_step_record, Mapping)
            and _step_snapshot_requires_provider_receipt(
                prior_step_record,
                provider_attempts=prior_provider_attempts,
                logical_repair_attempts=step_llm_repair_attempts,
            )
        ):
            provider_receipt_integrity_error = (
                "Durable provider/repair receipt is missing for a prior reservation."
            )

        provider_budget = StepProviderCallBudget(
            effective_provider_limit,
            step_id=step.step_id,
            consumed_categories=prior_provider_categories,
            logical_repair_entries=prior_logical_repair_entries,
            initial_generation_entry=prior_initial_generation_entry,
            receipt_path=provider_receipt_path,
            reserved_final_category=reserved_final_category,
            required_reservation_token=prior_required_reservation_token,
            reservation_bound_provider_history_len=(
                prior_reservation_bound_provider_history_len
            ),
            completed_reservation_token=prior_completed_reservation_token,
            reservation_released=prior_reservation_released,
        )

        if provider_receipt_integrity_error is None:
            try:
                # A crash before the first provider call leaves an exact unpaid
                # reservation that can be resumed. A crash after any paid call
                # but before the result digest was sealed is unknowable and must
                # block the step before any other route can ignore or replace it.
                provider_budget.next_logical_repair_attempt_id()
                initial_resume_status = (
                    provider_budget.initial_generation_resume_status()
                )
                if initial_resume_status == "paid_pending":
                    raise ProviderCallBudgetReceiptError(
                        "Initial generation has paid provider calls but no durable "
                        "transport result."
                    )
                if initial_resume_status == "failed":
                    raise ProviderCallBudgetReceiptError(
                        "Initial generation previously reached a terminal provider "
                        "failure."
                    )
            except ProviderCallBudgetReceiptError as exc:
                provider_receipt_integrity_error = str(exc)

        try:
            step_repair_budget = StepRepairBudget(
                provider_budget=provider_budget,
                step_record=step_record,
                max_llm_repairs=pipeline._max_step_llm_repair_attempts,
                initial_llm_repair_attempts=step_llm_repair_attempts,
                initial_repair_classes=(
                    prior_repair_classes
                    if provider_receipt_integrity_error is None
                    else ()
                ),
                provider_receipt_relative_path=provider_receipt_relative_path,
            )
        except (ProviderCallBudgetReceiptError, ValueError) as exc:
            provider_receipt_integrity_error = str(exc)
            step_repair_budget = StepRepairBudget(
                provider_budget=provider_budget,
                step_record=step_record,
                max_llm_repairs=pipeline._max_step_llm_repair_attempts,
                initial_llm_repair_attempts=step_llm_repair_attempts,
                provider_receipt_relative_path=provider_receipt_relative_path,
            )
        _sync_provider_budget = step_repair_budget.sync_provider

        _sync_provider_budget()
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
            return step_record
        coder_context = coder_base_context
        coder_authority = _coder_authority_with_locked_robustness_specs(
            authority=HostCoderAuthority(),
            context=coder_base_context,
            step=step,
            run_dir=run_dir,
        )
        coder_context, coder_authority = _bind_materialized_coder_authority(
            context=coder_base_context,
            step=step,
            authority=coder_authority,
        )
        if primary_cohort_uses_universe:
            locked_cohort_payload = _planner_locked_cohort_prompt_payload(plan)
            role_note = (
                "CURRENT STEP INPUT ROLE (host-owned execution contract): this "
                "is the plan's unique primary analysis_cohort + attrition "
                "producer, so COHORT_PARQUET is the raw study universe for this "
                "step only. Apply exactly the Planner-locked cohort definition, "
                "report truthful universe-to-final attrition, and emit an "
                "analysis_cohort whose ordered row identity matches the locked "
                "host cohort. Downstream steps receive the filtered cohort. "
                "Planner-locked cohort definition JSON: "
                f"{locked_cohort_payload}."
            )
            coder_authority = coder_authority.append(role_note)
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

        def _consume_llm_repair_budget(
            repair_class: str,
            *,
            before_code: str,
            repair_ticket: str,
            repair_authority: RepairPromptAuthority,
            current_repair_authority: Optional[RepairPromptAuthority] = None,
            provider_category: str,
            failure_status: str,
        ) -> bool:
            """Reserve one repair bound to its exact host-owned authority."""

            checkpoint_authority.ensure_candidate(
                before_code,
                reason="pre_repair_authority_binding",
            )
            binding = RepairAuthorityBinding(
                step_id=step.step_id,
                attempt_id=step_repair_budget.next_attempt_id,
                repair_class=str(repair_class),
                provider_category=provider_category,
                before_code_sha256=sha256_of_bytes(before_code.encode("utf-8")),
                step_spec_sha256=canonical_sha256(step.model_dump(mode="json")),
                resolved_inputs_sha256=resolved_inputs_sha256,
                coder_context_sha256=(
                    step_attempt_state.coordinates.scoped_coder_context.sha256
                    if step_attempt_state.coordinates is not None
                    else canonical_sha256(
                        {
                            "research_context": coder_context.model_dump(mode="json"),
                            "host_coder_authority": coder_authority.payload(),
                        }
                    )
                ),
                repair_ticket_sha256=_repair_prompt_binding_sha256(
                    untrusted_diagnostic=repair_ticket,
                    repair_authority=repair_authority,
                    current_repair_authority=current_repair_authority,
                ),
                engine_validator_sha256=(
                    step_attempt_state.coordinates.deterministic_gate_fingerprint
                    if step_attempt_state.coordinates is not None
                    else canonical_sha256(
                        {
                            "schema": "easyicu.step_control_plane_fingerprint/1",
                            "deterministic_gate_fingerprint": (
                                _deterministic_gate_stamp()[
                                    "deterministic_gate_fingerprint"
                                ]
                            ),
                            "coder_provider_identity_sha256": (
                                coder_provider_identity_sha256
                            ),
                        }
                    )
                ),
                prompt_pack_version=prompt_version,
                run_input_capsule_sha256=run_input_capsule_sha256,
            )
            consumed = step_repair_budget.consume(
                repair_class,
                authority_binding=binding,
            )
            if consumed:
                checkpoint_authority.checkpoint_state(
                    "repair_transport_pending",
                    extra={
                        "capsule_pending_repair_attempt_id": (
                            step_repair_budget.llm_repair_attempts
                        ),
                        "capsule_pending_repair_binding_sha256": binding.sha256,
                        "capsule_pending_repair_failure_status": failure_status,
                    },
                )
            return consumed

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
            return step_record
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
            return step_record
        try:
            (
                existing_refs,
                resolved_input_evidence_ids,
                resolved_input_bindings,
            ) = typed_binding_resolver.resolve_names(step.inputs, plan=plan)
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
            return step_record
        step_execution_cohort_path = _bind_step_execution_cohort(
            run_dir, step_execution_cohort_path, resolved_input_bindings, step_record
        )
        primary_cohort_uses_universe = step_execution_cohort_path == universe_path
        resolved_inputs_path = _write_resolved_inputs_manifest(
            run_dir=run_dir,
            step_id=step.step_id,
            planner_declared_inputs=step.inputs,
            bindings=resolved_input_bindings,
            context_path=plan_result.context_path,
        )
        coder_authority = _coder_authority_with_typed_parent_schema_receipts(
            authority=coder_authority,
            bindings=resolved_input_bindings,
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
            return step_record
        step_record["run_input_capsule_sha256"] = run_input_capsule_sha256
        step_record["resolved_input_evidence_ids"] = list(resolved_input_evidence_ids)
        if run_input_capsule_sha256 is not None:
            gate_stamp = _deterministic_gate_stamp()
            step_control_plane_fingerprint = canonical_sha256(
                {
                    "schema": "easyicu.step_control_plane_fingerprint/1",
                    "deterministic_gate_fingerprint": gate_stamp[
                        "deterministic_gate_fingerprint"
                    ],
                    "coder_provider_identity_sha256": (coder_provider_identity_sha256),
                }
            )
            step_attempt_state.coordinates = prepare_step_authority_coordinates(
                run_dir=run_dir,
                step_id=step.step_id,
                run_input_capsule_sha256=run_input_capsule_sha256,
                planner_scope=step.model_dump(mode="json"),
                scoped_coder_context={
                    "research_context": coder_context.model_dump(mode="json"),
                    "host_coder_authority": coder_authority.payload(),
                },
                resolved_inputs_path=resolved_inputs_path,
                typed_bindings=resolved_input_bindings,
                upstream_authority={
                    "resolved_input_evidence_ids": list(resolved_input_evidence_ids),
                    "resolved_input_bindings": resolved_input_bindings,
                    "cohort_sha256": sha256_of_file(cohort_path),
                    "universe_sha256": sha256_of_file(universe_path),
                },
                deterministic_gate_fingerprint=step_control_plane_fingerprint,
                engine_code_sha256=engine_code_sha256(),
                validator_code_sha256=validator_code_sha256(),
                prompt_pack_version=prompt_version,
                prompt_pack=prompt_files,
            )
            try:
                step_attempt_state.selected_resume_capsule = (
                    load_checkpoint_selected_step_capsule(
                        run_dir,
                        step_id=step.step_id,
                        checkpoint=(
                            plan_result.resume_state
                            if isinstance(plan_result.resume_state, Mapping)
                            else None
                        ),
                    )
                )
                if (
                    step_attempt_state.selected_resume_capsule is None
                    and requested_resume_from_step_id == step.step_id
                    and isinstance(prior_step_record, Mapping)
                ):
                    explicit_selection = (
                        select_explicit_step_capsule_for_targeted_resume(
                            run_dir,
                            step_id=step.step_id,
                            current_record=prior_step_record,
                            records=prior_attempt_records,
                            deterministic_gate_fingerprint=gate_stamp[
                                "deterministic_gate_fingerprint"
                            ],
                        )
                    )
                    if explicit_selection is not None:
                        (
                            step_attempt_state.selected_resume_capsule,
                            explicit_metadata,
                        ) = explicit_selection
                        step_record.update(explicit_metadata)
                if (
                    step_attempt_state.selected_resume_capsule is None
                    and isinstance(prior_step_record, Mapping)
                    and str(prior_step_record.get("status") or "").strip().lower()
                    == "resume_validator_invalid"
                    and isinstance(
                        prior_step_record.get(
                            "resume_revalidation_candidate_capsule_ref"
                        ),
                        Mapping,
                    )
                ):
                    try:
                        recovery_ref = StepAuthorityCapsuleRef.model_validate(
                            prior_step_record[
                                "resume_revalidation_candidate_capsule_ref"
                            ]
                        )
                        recovery_capsule = load_verified_step_authority_capsule(
                            run_dir,
                            ref=recovery_ref,
                            expected_step_id=step.step_id,
                        )
                    except (ValueError, StepAuthorityCapsuleError) as exc:
                        raise StepAuthorityRuntimeError(
                            "resume revalidation candidate is invalid"
                        ) from exc
                    expected_recovery_sha256 = str(
                        prior_step_record.get(
                            "resume_revalidation_candidate_code_sha256"
                        )
                        or ""
                    )
                    if (
                        not re.fullmatch(r"[0-9a-f]{64}", expected_recovery_sha256)
                        or recovery_capsule.capsule.candidate_code.sha256
                        != expected_recovery_sha256
                    ):
                        raise StepAuthorityRuntimeError(
                            "resume revalidation candidate digest is inconsistent"
                        )
                    step_attempt_state.selected_resume_capsule = recovery_capsule
                    step_record["resume_validator_invalid_candidate_reused"] = True
                # A paid repair result belongs to the historical parent and
                # coordinates recorded before a crash. Recover that exact
                # candidate first; only then may current engine/validator drift
                # adopt its bytes for revalidation. Reversing this order makes
                # the receipt's before-code and authority binding impossible to
                # satisfy after a control-plane update.
                if (
                    step_attempt_state.selected_resume_capsule is not None
                    and isinstance(prior_step_record, Mapping)
                    and prior_step_record.get("capsule_pending_repair_attempt_id")
                    is not None
                ):
                    step_attempt_state.current_capsule_ref = (
                        step_attempt_state.selected_resume_capsule.ref
                    )
                    pending_attempt = prior_step_record.get(
                        "capsule_pending_repair_attempt_id"
                    )
                    pending_binding = str(
                        prior_step_record.get("capsule_pending_repair_binding_sha256")
                        or ""
                    )
                    pending_failure_status = str(
                        prior_step_record.get("capsule_pending_repair_failure_status")
                        or ""
                    )
                    receipt_state = load_provider_call_budget_state(
                        provider_receipt_path,
                        step_id=step.step_id,
                        expected_reserved_final_category=reserved_final_category,
                    )
                    if (
                        isinstance(pending_attempt, bool)
                        or not isinstance(pending_attempt, int)
                        or not 1
                        <= pending_attempt
                        <= len(receipt_state.logical_repairs)
                        or str(
                            receipt_state.logical_repairs[pending_attempt - 1].get(
                                "binding_sha256"
                            )
                            or ""
                        )
                        != pending_binding
                        or not pending_failure_status
                    ):
                        raise StepAuthorityRuntimeError(
                            "completed repair lacks its exact pending checkpoint"
                        )
                    historical_coordinates = coordinates_from_verified_capsule(
                        run_dir,
                        step_attempt_state.selected_resume_capsule,
                    )
                    recovered_code_ref = repair_code_ref(
                        receipt_state,
                        attempt_id=pending_attempt,
                    )
                    recovered_ref = seal_repair_candidate_from_receipt(
                        historical_coordinates,
                        parent_ref=step_attempt_state.current_capsule_ref,
                        checkpoint_parent_ref=step_attempt_state.current_capsule_ref,
                        code_ref=recovered_code_ref,
                        receipt_state=receipt_state,
                        attempt_id=pending_attempt,
                        failure_status=pending_failure_status,
                    )
                    checkpoint_authority.checkpoint_capsule(
                        recovered_ref,
                        status="candidate_checkpointed",
                    )
                    step_attempt_state.selected_resume_capsule = (
                        load_verified_step_authority_capsule(
                            run_dir,
                            ref=recovered_ref,
                            expected_step_id=step.step_id,
                        )
                    )
                if step_attempt_state.selected_resume_capsule is not None and not (
                    capsule_matches_coordinates(
                        step_attempt_state.selected_resume_capsule,
                        step_attempt_state.coordinates,
                    )
                ):
                    frozen_context = adopt_frozen_scoped_coder_context(
                        step_attempt_state.selected_resume_capsule,
                        step_attempt_state.coordinates,
                    )
                    if frozen_context is None:
                        adopted_candidate = (
                            adopt_candidate_for_control_plane_revalidation(
                                step_attempt_state.selected_resume_capsule,
                                step_attempt_state.coordinates,
                            )
                        )
                        if adopted_candidate is None:
                            step_record["step_authority_capsule_cache_miss"] = (
                                "authority_drift"
                            )
                            step_attempt_state.selected_resume_capsule = None
                        else:
                            (
                                coder_context,
                                step_attempt_state.coordinates,
                                adopted_ref,
                            ) = adopted_candidate
                            step_attempt_state.current_capsule_ref = adopted_ref
                            step_attempt_state.selected_resume_capsule = (
                                load_verified_step_authority_capsule(
                                    run_dir,
                                    ref=adopted_ref,
                                    expected_step_id=step.step_id,
                                )
                            )
                            step_record["step_authority_capsule_cache_miss"] = (
                                "control_plane_drift_revalidation"
                            )
                    else:
                        coder_context, step_attempt_state.coordinates = frozen_context
                        step_record["step_authority_frozen_context_reused"] = True
                if (
                    step_attempt_state.selected_resume_capsule is not None
                    and isinstance(prior_step_record, Mapping)
                    and prior_step_record.get("quarantined_requires_repair") is True
                ):
                    step_record["step_authority_capsule_cache_miss"] = (
                        "quarantine_not_migrated"
                    )
                    step_attempt_state.selected_resume_capsule = None
                if step_attempt_state.selected_resume_capsule is not None:
                    step_attempt_state.current_capsule_ref = (
                        step_attempt_state.selected_resume_capsule.ref
                    )
                    step_record["step_authority_capsule_ref"] = (
                        step_attempt_state.selected_resume_capsule.ref.model_dump(
                            mode="json"
                        )
                    )
                    step_record["step_authority_capsule_stage"] = (
                        step_attempt_state.selected_resume_capsule.capsule.stage
                    )
                    audit = (
                        step_attempt_state.selected_resume_capsule.capsule.concept_audit
                    )
                    if audit is not None:
                        current_auditor_identity = llm_concept_auditor_identity_sha256
                        current_validator_identity = (
                            llm_concept_auditor_implementation_sha256
                            or canonical_sha256("llm_concept_auditor_unavailable")
                        )
                        if (
                            audit.auditor_identity_sha256 == current_auditor_identity
                            and audit.environment_sha256
                            == concept_audit_environment_sha256
                            and audit.validator_implementation_sha256
                            == current_validator_identity
                        ):
                            step_attempt_state.capsule_audit_findings_by_digest[
                                step_attempt_state.selected_resume_capsule.capsule.candidate_code.sha256
                            ] = (
                                read_concept_audit_findings(
                                    step_attempt_state.selected_resume_capsule,
                                    run_dir=run_dir,
                                ),
                                audit.audit_key,
                            )
                        else:
                            step_record["step_authority_audit_cache_miss"] = (
                                "audit_identity_drift"
                            )
                    checkpoint_authority.checkpoint_state(
                        "capsule_revalidation_pending"
                    )
                elif (
                    provider_budget.initial_generation_resume_status() == "completed"
                    and isinstance(prior_step_record, Mapping)
                    and str(prior_step_record.get("status") or "")
                    == "initial_generation_pending"
                ):
                    initial_entry = provider_budget.initial_generation_entry
                    pending_binding = str(
                        prior_step_record.get("capsule_pending_initial_binding_sha256")
                        or ""
                    )
                    pending_transport = str(
                        prior_step_record.get("capsule_pending_initial_transport_id")
                        or ""
                    )
                    if (
                        initial_entry is None
                        or pending_binding
                        != str(initial_entry.get("binding_sha256") or "")
                        or pending_transport
                        != str(initial_entry.get("provider_transport_id") or "")
                    ):
                        raise StepAuthorityRuntimeError(
                            "completed initial generation lacks its exact pending "
                            "checkpoint"
                        )
                    recovered_code_ref = initial_generation_code_ref(
                        step_attempt_state.coordinates,
                        load_provider_call_budget_state(
                            provider_receipt_path,
                            step_id=step.step_id,
                            expected_reserved_final_category=reserved_final_category,
                        ),
                    )
                    recovered_ref = seal_initial_generation_candidate(
                        step_attempt_state.coordinates,
                        code_ref=recovered_code_ref,
                        receipt_state=load_provider_call_budget_state(
                            provider_receipt_path,
                            step_id=step.step_id,
                            expected_reserved_final_category=reserved_final_category,
                        ),
                    )
                    checkpoint_authority.checkpoint_capsule(
                        recovered_ref,
                        status="candidate_checkpointed",
                    )
                    step_attempt_state.selected_resume_capsule = (
                        load_verified_step_authority_capsule(
                            run_dir,
                            ref=recovered_ref,
                            expected_step_id=step.step_id,
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
                return step_record
        else:
            step_record["step_authority_capsule_cache_miss"] = (
                "run_input_capsule_unavailable"
                if run_input_capsule_sha256 is None
                else "agentic_cli_transport_untracked"
            )
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

        def _remember_concept_constraints(
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

        def _quarantine_error_payloads(
            candidates: Sequence[ValidationFinding],
        ) -> List[Dict[str, Any]]:
            """Serialize the complete cross-repair constraint set for resume."""

            _remember_concept_constraints(candidates)
            return [
                finding.model_dump(mode="json")
                for finding in monotonic_concept_constraints
            ]

        def _monotonic_concept_constraint_payload() -> List[Dict[str, Any]]:
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

        def _monotonic_concept_constraint_ticket() -> List[Dict[str, Any]]:
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

        def _monotonic_concept_constraint_log() -> str:
            if not monotonic_concept_constraints:
                return ""
            payload = _monotonic_concept_constraint_payload()
            return (
                "\n\nPREVIOUSLY REPAIRED CONCEPT FINDINGS (binding regression "
                "constraints; do not reintroduce them):\n"
                + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            )

        def _use_quarantined_draft(draft: QuarantinedConceptDraft) -> str:
            quarantine_state.resumed_draft_used = True
            quarantine_state.draft_active = True
            quarantine_state.repair_succeeded = False
            quarantine_state.pending_errors = [
                ValidationFinding.model_validate(payload) for payload in draft.findings
            ]
            # Historical errors remain binding regression constraints, but
            # their old source coordinates are not findings on the current
            # digest and must never enter an exact minimal-patch ticket.
            _remember_concept_constraints(quarantine_state.pending_errors)
            step_record["resumed_quarantined_draft"] = True
            step_record["quarantined_draft_sha256"] = draft.sha256
            step_record["quarantined_draft_relative_path"] = draft.relative_path
            step_record["quarantined_requires_repair"] = True
            step_record["quarantined_repair_succeeded"] = False
            emit_progress(
                "coder",
                f"Resuming rejected draft for mandatory repair: {step.step_id}.",
                status="warning",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return draft.code

        def _use_resumed_code(
            resumed_code: Tuple[str, Dict[str, Any]],
            *,
            error: Optional[BaseException] = None,
        ) -> str:
            worker_progress.resumed_code_reuse_used = True
            prior_code, resumed_record = resumed_code
            step_record["generation_mode"] = "resumed_code_reuse"
            step_record["resumed_code_evidence_id"] = resumed_record.get("evidence_id")
            step_record["resumed_code_relative_path"] = resumed_record.get(
                "relative_path"
            )
            resumed_evidence_generation_mode = str(
                resumed_record.get("generation_mode") or ""
            )
            resumed_from_generation_mode = resumed_evidence_generation_mode
            if resumed_evidence_generation_mode == "resumed_code_reuse":
                resumed_metadata = resumed_record.get("metadata")
                if isinstance(resumed_metadata, dict):
                    resumed_from_generation_mode = str(
                        resumed_metadata.get("resumed_from_generation_mode") or ""
                    )
            step_record["resumed_code_evidence_generation_mode"] = (
                resumed_evidence_generation_mode
            )
            step_record["resumed_from_generation_mode"] = resumed_from_generation_mode
            detail = {
                "step_id": step.step_id,
                "resume_from_step_id": requested_resume_from_step_id,
                "evidence_id": resumed_record.get("evidence_id"),
                "relative_path": resumed_record.get("relative_path"),
                "resumed_from_generation_mode": resumed_from_generation_mode,
            }
            if error is None:
                message = (
                    "Explicit resume reused prior agent-generated code "
                    f"(source mode: {resumed_from_generation_mode}) for step "
                    f"{step.step_id} before requesting a new coder script."
                )
            else:
                detail["error"] = str(error)
                message = (
                    f"Coder agent failed for step {step.step_id}; reused prior "
                    "agent-generated code from resume evidence "
                    f"(source mode: {resumed_from_generation_mode})."
                )
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="warning",
                        message=message,
                        detail=detail,
                    )
                )
            emit_progress(
                "coder",
                f"Reused prior generated analysis script for {step.step_id}.",
                status="warning",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return prior_code

        def _repair_with_capsule(
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
            coordinates = step_attempt_state.coordinates
            try:
                return coder.repair(
                    context=context,
                    step=step,
                    host_authority=coder_authority,
                    code=code,
                    run_log=run_log,
                    repair_authority=repair_authority,
                    current_repair_authority=current_repair_authority,
                    attempt=attempt,
                    provider_budget=provider_budget,
                    provider_category=provider_category,
                    logical_repair_attempt_id=logical_repair_attempt_id,
                    persist_candidate=(
                        (
                            lambda candidate: persist_candidate_code(
                                coordinates, candidate
                            )
                        )
                        if coordinates is not None
                        else None
                    ),
                    on_candidate_completed=(
                        lambda ref, _mode, logical_id: (
                            (
                                checkpoint_authority.seal_completed_repair_candidate(
                                    ref,
                                    logical_id,
                                    failure_status=failure_status,
                                )
                            )
                            if coordinates is not None
                            else None
                        )
                    ),
                )
            except Exception:
                checkpoint_authority.clear_failed_repair_transport(
                    logical_repair_attempt_id
                )
                raise

        def _reserve_compatibility_repair(
            before_code: str,
            repair_ticket: str,
            repair_authority: RepairPromptAuthority,
        ) -> Optional[int]:
            if not _consume_llm_repair_budget(
                "compatibility",
                before_code=before_code,
                repair_ticket=repair_ticket,
                repair_authority=repair_authority,
                provider_category="compatibility_repair",
                failure_status="concept_failed",
            ):
                return None
            return step_repair_budget.llm_repair_attempts

        def _resume_summary_repair_code() -> Optional[str]:
            if (
                requested_resume_from_step_id != step.step_id
                or not pipeline._enable_deterministic_runner_repair
            ):
                return None
            resumed_code = resume_controller.prior_code_for_step(step.step_id)
            if resumed_code is None:
                return None
            prior_code, _resumed_record = resumed_code
            prior_summary_path = (
                run_dir / "steps" / step.step_id / "outputs" / "step_summary.json"
            )
            if not prior_summary_path.exists():
                return None
            try:
                prior_summary = json.loads(
                    prior_summary_path.read_text(encoding="utf-8")
                )
            except Exception:
                return None
            if not isinstance(prior_summary, dict) or not prior_summary:
                return None
            repair = _deterministic_summary_repair(
                code=prior_code,
                step_summary=prior_summary,
                previous_repair=None,
                analysis_family=(
                    local_runtime_state.analysis_family
                    or prior_summary.get("analysis_family")
                ),
            )
            repair = _authorize_automatic_repair(
                repair,
                step=step,
                source="resume_summary_repair_preflight",
                before_code=prior_code,
            )
            if repair is None:
                return None
            repair_name, repaired_code = repair
            _use_resumed_code(resumed_code)
            worker_progress.preexecution_runner_repair_name = repair_name
            step_record["runner_repair"] = repair_name
            step_record["resume_summary_repair"] = repair_name
            _record_repair(
                repair_id=repair_name,
                step_id=step.step_id,
                trigger={
                    "source": "resume_summary_repair_preflight",
                    "step_summary_path": str(prior_summary_path),
                    "step_summary_keys": sorted(str(k) for k in prior_summary),
                },
                transformation=(
                    "Reused the explicitly resumed step's prior generated code "
                    "after deterministic summary repair, before requesting a "
                    "new coder script."
                ),
                before_code=prior_code,
                after_code=repaired_code,
                selection_rule=(
                    "only when the prior step_summary triggers a case-neutral "
                    "deterministic summary repair"
                ),
            )
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            f"Applied deterministic resume-summary repair for "
                            f"step {step.step_id}: {repair_name}."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "repair_id": repair_name,
                            "step_summary_path": str(prior_summary_path),
                        },
                    )
                )
            emit_progress(
                "runner_repair",
                (
                    f"Applied deterministic resume-summary repair for "
                    f"{step.step_id}: {repair_name}."
                ),
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return repaired_code

        def _resume_critic_repair_code() -> Optional[str]:
            """Repair the selected prior script from structured Critic feedback."""

            report = resume_controller.prior_negative_critic_report_for_step(
                step.step_id
            )
            if report is None:
                return None
            resumed_code = resume_controller.prior_code_for_step(step.step_id)
            if resumed_code is None:
                return None
            prior_code = resumed_code[0]
            critique_log = (
                "PRIOR CRITIC REVIEW (binding repair requirements):\n"
                + json.dumps(report, indent=2, ensure_ascii=False, default=str)
            )
            critic_repair_authority = RepairPromptAuthority.create(
                typed_ticket=[
                    {
                        "reason": "OUTPUT_CONTRACT_INVALID",
                        "validator": "critic_resume",
                        "detail": {"critic_report": report},
                    }
                ]
            )
            if not _consume_llm_repair_budget(
                "critic_resume",
                before_code=prior_code,
                repair_ticket=critique_log,
                repair_authority=critic_repair_authority,
                provider_category="critic_resume_repair",
                failure_status="critic_failed",
            ):
                return None
            prior_code = _use_resumed_code(resumed_code)
            emit_progress(
                "coder",
                f"Repairing prior Critic findings for {step.step_id}.",
                status="warning",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            try:
                repaired = _repair_with_capsule(
                    failure_status="critic_failed",
                    context=coder_context,
                    step=step,
                    code=prior_code,
                    run_log=critique_log,
                    repair_authority=critic_repair_authority,
                    attempt=1,
                    provider_budget=provider_budget,
                    provider_category="critic_resume_repair",
                    logical_repair_attempt_id=(step_repair_budget.llm_repair_attempts),
                )
                _sync_provider_budget()
            except (
                ProviderCallBudgetReceiptError,
                StepAuthorityRuntimeError,
                StepAuthorityCapsuleError,
            ):
                raise
            except Exception as exc:
                _sync_provider_budget()
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="critic_resume_repair",
                            severity="warning",
                            message=(
                                "Prior Critic-guided repair was unavailable; "
                                "falling back to ordinary code generation."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:300],
                            },
                        )
                    )
                return None
            worker_progress.critic_resume_repair_used = True
            step_record["critic_resume_repair"] = True
            step_record["critic_resume_repair_status"] = report.get("status")
            return repaired

        def _publication_figure_preflight_supported() -> bool:
            # Preflight may replace the coder, so names/prose are insufficient.
            # Claim only a split figure whose direct parent recorded a controlled
            # figure-data family, exact method, or analysis family.  Legacy name
            # routing remains available only after an agent figure fails QA.
            if not _step_has_figure_only_output_contract(step):
                return False
            return services.deterministic_figure_family_supported_for_upstream(
                run_dir, step.step_id
            )

        def _absolute_risk_context_preflight_supported() -> bool:
            if _step_expects_figure(step):
                return False
            return _absolute_risk_context_runner_owns_step(
                str(step.method or ""),
                str(step.step_id or ""),
                step.expected_outputs or [],
            )

        def _deterministic_absolute_risk_context_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            if (
                worker_progress.deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (preflight and not _absolute_risk_context_preflight_supported())
            ):
                return None
            if not _absolute_risk_context_preflight_supported():
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
            return absolute_risk_context_code()

        def _robustness_sensitivity_preflight_supported() -> bool:
            if _step_expects_figure(step):
                return False
            return _robustness_sensitivity_runner_owns_step(
                str(step.method or ""),
                str(step.step_id or ""),
                step.expected_outputs or [],
            )

        def _deterministic_robustness_sensitivity_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            if (
                worker_progress.deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (preflight and not _robustness_sensitivity_preflight_supported())
            ):
                return None
            if not _robustness_sensitivity_preflight_supported():
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
            return robustness_sensitivity_preflight_code()

        def _missingness_audit_preflight_supported() -> bool:
            """True for a missingness / measurement-process AUDIT step.

            The audit is a pure per-concept count (measured vs missing fraction +
            structural-vs-measurement split); the LLM coder reliably exhausted its
            retry budget on it (~27.6 min then fail). The deterministic runner owns
            it so the audit never blocks the run. It must NOT claim a figure step
            nor a primary result step that merely mentions missingness. Trigger is
            case-neutral (the controlled ``method`` first, then audit vocabulary).
            """
            if _step_expects_figure(step):
                return False
            return _simple_missingness_audit_runner_owns_step(
                str(step.method or ""),
                str(step.step_id or ""),
                str(step.intent or ""),
                step.expected_outputs or [],
            )

        def _deterministic_missingness_audit_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            if (
                worker_progress.deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (preflight and not _missingness_audit_preflight_supported())
            ):
                return None
            if not _missingness_audit_preflight_supported():
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
            return missingness_measurement_audit_code()

        def _trajectory_stability_preflight_supported() -> bool:
            return trajectory_stability_executor_owns_step(step, plan=plan)

        def _deterministic_trajectory_stability_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            if worker_progress.deterministic_standard_executor_used or (
                preflight and not _trajectory_stability_preflight_supported()
            ):
                return None
            if not _trajectory_stability_preflight_supported():
                return None
            worker_progress.deterministic_standard_executor_used = True
            step_record["deterministic_standard_selection_reason"] = reason
            step_record["deterministic_standard_analysis"] = (
                "trajectory_cluster_stability"
            )
            emit_progress(
                "coder",
                f"Using planner-specified trajectory stability executor for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            return trajectory_stability_executor_code(step, plan=plan)

        # ``--resume-from-step-id`` means the selected step is intentionally
        # rerun. Completed predecessors stay checkpointed. A previously
        # successful step gets a fresh Coder draft unless the operator opts in
        # to reuse; a prior deterministic ``contract_failed`` attempt may reuse
        # only its exact evidence-bound code and scientific signature. Reused
        # code still runs through every current execution audit and repair gate.
        preflight_trajectory_stability_code = (
            None
            if step_attempt_state.selected_resume_capsule is not None
            else _deterministic_trajectory_stability_code(
                "trajectory_stability_spec_preflight", preflight=True
            )
        )
        preflight_figure_code = (
            None
            if preflight_trajectory_stability_code is not None
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
                preflight_trajectory_stability_code is not None
                or preflight_figure_code is not None
            )
            else resume_controller.quarantined_concept_draft_for_step(step.step_id)
        )
        resume_critic_repair_code = (
            None
            if (
                preflight_trajectory_stability_code is not None
                or preflight_figure_code is not None
                or quarantined_resume_draft is not None
            )
            else _resume_critic_repair_code()
        )
        resume_summary_repair_code = (
            None
            if (
                preflight_trajectory_stability_code is not None
                or preflight_figure_code is not None
                or quarantined_resume_draft is not None
                or resume_critic_repair_code is not None
            )
            else _resume_summary_repair_code()
        )
        preflight_resumed_code = None
        failed_contract_code_preflight_reuse = False
        if (
            preflight_trajectory_stability_code is None
            and preflight_figure_code is None
            and quarantined_resume_draft is None
            and resume_summary_repair_code is None
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
        if preflight_trajectory_stability_code is not None:
            code = preflight_trajectory_stability_code
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            "Using the deterministic calculator for the complete "
                            "planner-owned trajectory stability specification in "
                            f"step {step.step_id}."
                        ),
                        detail={"step_id": step.step_id},
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
        elif resume_summary_repair_code is not None:
            code = resume_summary_repair_code
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
                                    f"Coder agent failed for step {step.step_id}: "
                                    f"{exc}"
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
                        return _record_initial_coder_failure(exc)
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
                            return _record_initial_coder_failure(exc)

        def _deterministic_fallback_code(reason: str) -> Optional[str]:
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

        concept_audit = ConceptAuditCoordinator(
            authority=ConceptAuditAuthority(
                context=context,
                step=step,
                resolved_input_bindings=resolved_input_bindings,
                environment_sha256=concept_audit_environment_sha256,
                auditor_implementation_sha256=(
                    llm_concept_auditor_implementation_sha256
                ),
                auditor_identity=(
                    lambda: pipeline._llm_signature(llm_concept_audit_client)
                ),
                enable_llm_audit=pipeline._enable_llm_concept_audit,
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
            ),
        )

        def _authorized_deterministic_concept_repair(
            *,
            script_text: str,
            error_messages: Sequence[str],
            repair_reasons: Sequence[RepairReason] = (),
            repair_findings: Sequence[ValidationFinding] = (),
            source: str,
        ) -> Tuple[str, List[str]]:
            # Implementation extracted to repair_coordination (A2 batch-1);
            # the authorization side effects stay with the local callback.
            return authorized_deterministic_concept_repair(
                script_text,
                error_messages,
                repair_reasons=repair_reasons,
                repair_findings=repair_findings,
                authorize=_authorize_automatic_repair,
                step=step,
                source=source,
                context=context,
            )

        worker_progress.concept_repair_attempts = 0
        worker_progress.llm_repair_used = worker_progress.critic_resume_repair_used
        worker_progress.concept_audit_error_count = 0
        worker_progress.deterministic_concept_repairs = 0
        _MAX_DETERMINISTIC_CONCEPT_REPAIRS = 3
        worker_progress.applied_concept_repair_names = []
        concept_approved_code_digest: Optional[str] = None
        while True:
            # A quarantined checkpoint is digest-bound authority.  Do not
            # normalize it before testing deterministic policy supersession;
            # even a semantics-preserving rewrite would break the exact SHA
            # proof and force an otherwise unnecessary LLM repair.
            if not quarantine_state.draft_active:
                code = reorder_forward_references(code)
            usage_findings = concept_audit.findings_for_code(
                code,
                include_llm=False,
            )
            step_record["usage_findings"] = [f.model_dump() for f in usage_findings]
            worker_progress.concept_audit_error_count += sum(
                1
                for f in usage_findings
                if f.validator == usage_auditor.name and f.severity == "error"
            )
            step_record["concept_audit_error_count"] = (
                worker_progress.concept_audit_error_count
            )
            step_record["concept_repair_attempts"] = (
                worker_progress.concept_repair_attempts
            )
            if not any(f.severity == "error" for f in usage_findings):
                concept_approved_code_digest = sha256_of_bytes(code.encode("utf-8"))
                step_record["concept_approved_code_sha256"] = (
                    concept_approved_code_digest
                )
                if sealed_renderer_state.repair_id is not None:
                    sealed_renderer_authorized_code_sha256 = (
                        concept_approved_code_digest
                    )
                    step_record["sealed_renderer_authorized_code_sha256"] = (
                        sealed_renderer_authorized_code_sha256
                    )
                if (
                    quarantine_state.resumed_draft_used
                    and quarantine_state.repair_materially_changed
                    and not quarantine_state.superseded_by_fallback
                ):
                    quarantine_state.repair_succeeded = True
                    step_record["quarantined_repair_succeeded"] = True
                with shared_lock:
                    findings.extend(usage_findings)
                break

            if sealed_renderer_state.repair_id is not None:
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
                        "repair_id": sealed_renderer_state.repair_id,
                        "reason": "preexecution_concept_gate_failed",
                    },
                )
                terminal_findings = [terminal_finding, *usage_findings]
                step_record.update(
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
                with shared_lock:
                    findings.extend(terminal_findings)
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Sealed renderer blocked for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            if worker_progress.deterministic_standard_executor_used:
                terminal_finding = ValidationFinding(
                    validator="trajectory_stability_executor",
                    severity="error",
                    message=(
                        "The trusted trajectory stability adapter failed the "
                        "pre-execution deterministic concept gate; execution was "
                        "blocked without coder repair."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "reason": "preexecution_concept_gate_failed",
                    },
                )
                terminal_findings = [terminal_finding, *usage_findings]
                step_record.update(
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
                with shared_lock:
                    findings.extend(terminal_findings)
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Trusted standard adapter blocked for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            # Tier A — deterministic mechanical repair. For a closed set of
            # objectively-flagged ICU anti-patterns (e.g. silent fillna(0) on
            # a lab) there is a single neutral fix, so we apply it without a
            # model round-trip and re-audit. This does NOT consume the LLM
            # repair budget, and is bounded because each repair removes its
            # own pattern (a re-audit then finds nothing left to change).
            if (
                worker_progress.deterministic_concept_repairs
                < _MAX_DETERMINISTIC_CONCEPT_REPAIRS
            ):
                _audit_error_msgs = [
                    value
                    for finding in usage_findings
                    if finding.severity == "error"
                    for value in (
                        finding.message,
                        str((finding.detail or {}).get("reason") or ""),
                    )
                    if value
                ]
                _audit_repair_reasons = [
                    repair_reason_for_finding(finding)
                    for finding in usage_findings
                    if finding.severity == "error"
                ]
                _det_code, _det_names = _authorized_deterministic_concept_repair(
                    script_text=code,
                    error_messages=_audit_error_msgs,
                    repair_reasons=_audit_repair_reasons,
                    repair_findings=usage_findings,
                    source="deterministic_concept_audit_repair",
                )
                if _det_names and _det_code != code:
                    _det_before_code = code
                    worker_progress.deterministic_concept_repairs += 1
                    worker_progress.applied_concept_repair_names.extend(_det_names)
                    step_record["deterministic_concept_repairs"] = (
                        worker_progress.deterministic_concept_repairs
                    )
                    step_record["applied_concept_repair_names"] = list(
                        worker_progress.applied_concept_repair_names
                    )
                    for _name in _det_names:
                        _record_repair(
                            repair_id=_name,
                            step_id=step.step_id,
                            trigger={
                                "gate": "concept_audit",
                                "audit_errors": _audit_error_msgs,
                            },
                            transformation=(
                                "deterministic_concept_audit_repair: rewrote a "
                                "mechanical ICU anti-pattern flagged as an error "
                                "by the static concept-audit gate"
                            ),
                            before_code=code,
                            after_code=_det_code,
                            selection_rule=(
                                "applied only because an error finding "
                                "objectively named the anti-pattern"
                            ),
                        )
                    emit_progress(
                        "coder",
                        f"Auto-repaired concept-audit anti-pattern "
                        f"({', '.join(_det_names)}) for {step.step_id}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    code = _det_code
                    if (
                        quarantine_state.draft_active
                        and _python_repair_is_materially_changed(
                            _det_before_code,
                            code,
                        )
                    ):
                        # The stored errors remain useful regression constraints,
                        # but they are not findings on a new, materially repaired
                        # digest.  Re-audit that digest from scratch just as the
                        # LLM-repair path below does.
                        quarantine_state.draft_active = False
                        quarantine_state.repair_materially_changed = True
                        quarantine_state.pending_errors = []
                        step_record["quarantined_repair_materially_changed"] = True
                    continue

            if (
                worker_progress.concept_repair_attempts
                >= pipeline._max_code_repair_attempts
                or not _llm_repair_budget_available()
                or provider_budget.exhausted
            ):
                if not _logical_llm_repair_budget_available():
                    step_record["step_llm_repair_budget_exhausted"] = True
                    step_record["step_llm_repair_budget"] = (
                        pipeline._max_step_llm_repair_attempts
                    )
                _sync_provider_budget()
                fallback_code = _deterministic_fallback_code("concept_audit")
                if fallback_code is not None:
                    fallback_checkpoint_error: Optional[Exception] = None
                    if quarantine_state.resumed_draft_used:
                        try:
                            checkpoint = store_quarantined_concept_draft(
                                run_dir=run_dir,
                                step_id=step.step_id,
                                code=code,
                                findings=_quarantine_error_payloads(usage_findings),
                            )
                            step_record["quarantined_draft_sha256"] = checkpoint.sha256
                            step_record["quarantined_draft_relative_path"] = (
                                checkpoint.relative_path
                            )
                            step_record["quarantine_checkpoint_is_latest_candidate"] = (
                                True
                            )
                        except Exception as checkpoint_exc:
                            fallback_checkpoint_error = checkpoint_exc
                    # Surface the pattern/concept findings that
                    # forced the fallback; otherwise the manifest
                    # silently drops the original ICU rule
                    # violations that the LLM emitted. We dedupe by
                    # message so repeated retries don't spam.
                    with shared_lock:
                        if fallback_checkpoint_error is not None:
                            findings.append(
                                ValidationFinding(
                                    validator="resume",
                                    severity="warning",
                                    message=(
                                        "Could not update the concept-draft "
                                        "checkpoint before deterministic fallback "
                                        f"for step {step.step_id}: "
                                        f"{fallback_checkpoint_error}"
                                    ),
                                    detail={"step_id": step.step_id},
                                )
                            )
                        seen_occurrences = {
                            _finding_occurrence_identity(f) for f in findings
                        }
                        for f in usage_findings:
                            if _finding_occurrence_identity(f) in seen_occurrences:
                                continue
                            # Demote ``error`` severity to
                            # ``warning`` because the run is
                            # continuing on the deterministic
                            # fallback; reviewer still sees the
                            # original violation in the manifest.
                            if f.severity == "error":
                                f = f.model_copy(
                                    update={
                                        "severity": "warning",
                                        "message": (
                                            "[surfaced after fallback] " + f.message
                                        ),
                                    }
                                )
                            findings.append(f)
                    if quarantine_state.resumed_draft_used:
                        quarantine_state.draft_active = False
                        quarantine_state.pending_errors = []
                        quarantine_state.repair_succeeded = False
                        quarantine_state.superseded_by_fallback = True
                        step_record["quarantined_repair_succeeded"] = False
                        step_record["quarantine_superseded_by_fallback"] = True
                    code = fallback_code
                    continue
                step_record["status"] = "blocked_by_concept_audit"
                checkpoint_error: Optional[Exception] = None
                if not quarantine_state.superseded_by_fallback:
                    try:
                        checkpoint = store_quarantined_concept_draft(
                            run_dir=run_dir,
                            step_id=step.step_id,
                            code=code,
                            findings=_quarantine_error_payloads(usage_findings),
                        )
                        step_record["quarantined_draft_sha256"] = checkpoint.sha256
                        step_record["quarantined_draft_relative_path"] = (
                            checkpoint.relative_path
                        )
                        step_record["quarantined_requires_repair"] = True
                        step_record["quarantine_checkpoint_is_latest_candidate"] = True
                    except Exception as checkpoint_exc:
                        checkpoint_error = checkpoint_exc
                # Tier C — when auto-repair (deterministic + LLM) could not
                # clear the violation, do NOT just stop with a status code.
                # Emit an actionable repair ticket so a human can either add a
                # constraint and re-run, or knowingly accept the withheld
                # (diagnostic_only) result. We name candidate remedies without
                # mandating one — the analytical choice stays with the user.
                _block_errors = [
                    {"validator": f.validator, "message": f.message}
                    for f in usage_findings
                    if f.severity == "error"
                ]
                _offending_lines = [
                    ln.strip()
                    for ln in code.splitlines()
                    if any(
                        tok in ln
                        for tok in ("fillna(0)", "fillna(0.0)", ".mean()", "dropna(")
                    )
                ][:12]
                _remedies = [
                    "Add the violated ICU rule as an explicit coder/planner "
                    "constraint and re-run this question (e.g. 'do not impute a "
                    "lab with 0; handle missingness with complete-case or a "
                    "declared imputation + missingness indicator').",
                    "Use a stronger model for this question — the block was "
                    "triggered by generated code, not by the cohort or the "
                    "question itself.",
                    "Accept the withheld result: diagnostic_only is a valid "
                    "outcome. The fail-closed gate declined to report an "
                    "analysis it judged unsafe; nothing wrong was published.",
                ]
                step_record["concept_audit_block"] = {
                    "step_id": step.step_id,
                    "errors": _block_errors,
                    "deterministic_repairs_applied": list(
                        worker_progress.applied_concept_repair_names
                    ),
                    "llm_repair_attempts": worker_progress.concept_repair_attempts,
                    "offending_code_lines": _offending_lines,
                    "candidate_remedies": _remedies,
                }
                try:
                    _ticket = [
                        f"# Concept-audit block — step `{step.step_id}`",
                        "",
                        "The static ICU concept-audit gate blocked this step "
                        "before execution and auto-repair could not clear it, "
                        "so the run withheld this analysis (`diagnostic_only`). "
                        "This is the fail-closed safety system working — but "
                        "here is how to move it forward.",
                        "",
                        "## What was flagged (objective errors)",
                        *[
                            f"- **{e['validator']}**: {e['message']}"
                            for e in _block_errors
                        ],
                        "",
                        "## Repair already attempted",
                        f"- deterministic: "
                        f"{worker_progress.applied_concept_repair_names or 'none matched'}",
                        f"- LLM coder repair attempts: {worker_progress.concept_repair_attempts}",
                        "",
                        "## Offending code lines",
                        "```python",
                        *(_offending_lines or ["(no obvious anti-pattern line)"]),
                        "```",
                        "",
                        "## How to resolve (pick one — your analytical choice)",
                        *[f"{i + 1}. {r}" for i, r in enumerate(_remedies)],
                        "",
                    ]
                    (run_dir / f"concept_audit_block_{step.step_id}.md").write_text(
                        "\n".join(_ticket), encoding="utf-8"
                    )
                except Exception:  # ticket is best-effort, never fatal
                    pass
                with shared_lock:
                    findings.extend(usage_findings)
                    if checkpoint_error is not None:
                        findings.append(
                            ValidationFinding(
                                validator="resume",
                                severity="warning",
                                message=(
                                    "Could not update the blocked concept-draft "
                                    f"checkpoint for step {step.step_id}: "
                                    f"{checkpoint_error}"
                                ),
                                detail={"step_id": step.step_id},
                            )
                        )
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Concept audit blocked {step.step_id}; repair ticket written.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            blocking_usage_findings = _blocking_validator_findings(usage_findings)
            audit_log = "\n".join(
                (
                    f"{f.severity.upper()}: {f.message}"
                    + (
                        "\nDETAIL (diagnostic mirror only): "
                        + json.dumps(f.detail, ensure_ascii=False, sort_keys=True)
                        if f.detail
                        else ""
                    )
                )
                for f in blocking_usage_findings
            )
            structured_repair_ticket = typed_repair_ticket(blocking_usage_findings)
            current_concept_repair_authority = RepairPromptAuthority.create(
                typed_ticket=structured_repair_ticket,
            )
            concept_repair_authority = RepairPromptAuthority.create(
                typed_ticket=[
                    *structured_repair_ticket,
                    *_monotonic_concept_constraint_ticket(),
                ],
            )
            concept_repair_log = (
                "Static concept audit blocked this script before "
                "execution. Fix all ICU-rule violations.\n\n"
                "HUMAN-READABLE FINDINGS (diagnostic mirror only):\n" + audit_log
            )
            worker_progress.concept_repair_attempts += 1
            if not _consume_llm_repair_budget(
                "concept",
                before_code=code,
                repair_ticket=concept_repair_log,
                repair_authority=concept_repair_authority,
                current_repair_authority=current_concept_repair_authority,
                provider_category="concept_repair",
                failure_status="concept_failed",
            ):
                raise AssertionError("LLM repair budget changed without mutation")
            step_record["concept_repair_attempts"] = (
                worker_progress.concept_repair_attempts
            )
            emit_progress(
                "coder",
                f"Repairing concept-audit violation for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                repair_attempts=worker_progress.concept_repair_attempts,
            )
            _remember_concept_constraints(blocking_usage_findings)
            try:
                repaired_code = _repair_with_capsule(
                    failure_status="concept_failed",
                    context=coder_context,
                    step=step,
                    code=code,
                    run_log=concept_repair_log,
                    repair_authority=concept_repair_authority,
                    current_repair_authority=current_concept_repair_authority,
                    attempt=worker_progress.concept_repair_attempts,
                    provider_budget=provider_budget,
                    provider_category="concept_repair",
                    logical_repair_attempt_id=(step_repair_budget.llm_repair_attempts),
                )
                _sync_provider_budget()
                if (
                    quarantine_state.draft_active
                    and not _python_repair_is_materially_changed(code, repaired_code)
                ):
                    checkpoint_authority.reject_completed_repair_candidate(
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
                            "quarantined_draft_sha256": step_record.get(
                                "quarantined_draft_sha256"
                            ),
                            "repair_attempt": worker_progress.concept_repair_attempts,
                            "semantic_noop": True,
                        },
                    )
                    if not any(
                        finding.message == no_op_finding.message
                        for finding in quarantine_state.pending_errors
                    ):
                        quarantine_state.pending_errors.append(no_op_finding)
                    step_record["quarantined_repair_noop_count"] = (
                        int(step_record.get("quarantined_repair_noop_count") or 0) + 1
                    )
                    quarantine_state.repair_succeeded = False
                    step_record["quarantined_repair_succeeded"] = False
                    continue
                code = repaired_code
                worker_progress.llm_repair_used = True
                if quarantine_state.draft_active:
                    quarantine_state.draft_active = False
                    quarantine_state.repair_materially_changed = True
                    quarantine_state.pending_errors = []
                    step_record["quarantined_repair_materially_changed"] = True
            except (
                ProviderCallBudgetReceiptError,
                StepAuthorityRuntimeError,
                StepAuthorityCapsuleError,
            ):
                raise
            except BaseException as exc:
                _sync_provider_budget()
                checkpoint_error: Optional[Exception] = None
                try:
                    checkpoint = store_quarantined_concept_draft(
                        run_dir=run_dir,
                        step_id=step.step_id,
                        code=code,
                        findings=_quarantine_error_payloads(usage_findings),
                    )
                    step_record["quarantined_draft_sha256"] = checkpoint.sha256
                    step_record["quarantined_draft_relative_path"] = (
                        checkpoint.relative_path
                    )
                    step_record["quarantined_requires_repair"] = True
                except Exception as checkpoint_exc:
                    checkpoint_error = checkpoint_exc
                if not isinstance(exc, Exception):
                    raise
                fallback_code = _deterministic_fallback_code("concept_repair_failed")
                if fallback_code is not None:
                    quarantine_state.draft_active = False
                    quarantine_state.pending_errors = []
                    quarantine_state.repair_succeeded = False
                    if quarantine_state.resumed_draft_used:
                        quarantine_state.superseded_by_fallback = True
                        step_record["quarantined_repair_succeeded"] = False
                        step_record["quarantine_superseded_by_fallback"] = True
                    code = fallback_code
                    continue
                with shared_lock:
                    findings.extend(usage_findings)
                    if checkpoint_error is not None:
                        findings.append(
                            ValidationFinding(
                                validator="resume",
                                severity="warning",
                                message=(
                                    "Could not preserve the rejected concept-audit "
                                    f"draft for step {step.step_id}: {checkpoint_error}"
                                ),
                                detail={"step_id": step.step_id},
                            )
                        )
                    findings.append(
                        ValidationFinding(
                            validator="coder",
                            severity="error",
                            message=(
                                f"Coder repair failed after concept audit for "
                                f"step {step.step_id}: {exc}"
                            ),
                        )
                    )
                    step_record["status"] = "repair_failed"
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "coder",
                    f"Concept-audit repair failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

        if quarantine_state.draft_active and not quarantine_state.repair_succeeded:
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
            return step_record

        if (
            quarantine_state.repair_succeeded
            or quarantine_state.policy_superseded
            or quarantine_state.deterministic_revalidated
        ):
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
                return step_record

        worker_progress.repair_attempts = 0
        worker_progress.contract_repair_attempts = 0
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
        while True:
            code = reorder_forward_references(code)
            checkpoint_authority.ensure_candidate(
                code,
                reason="host_code_normalization_or_deterministic_mutation",
            )
            candidate_code_digest = sha256_of_bytes(code.encode("utf-8"))
            if (
                sealed_renderer_authorized_code_sha256 is not None
                and candidate_code_digest != sealed_renderer_authorized_code_sha256
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
                        "step_id": step.step_id,
                        "repair_id": sealed_renderer_state.repair_id,
                        "authorized_code_sha256": (
                            sealed_renderer_authorized_code_sha256
                        ),
                        "candidate_code_sha256": candidate_code_digest,
                    },
                )
                step_record.update(
                    {
                        "status": "execution_failed",
                        "diagnostic_only": True,
                        "sealed_renderer_terminal_reason": "code_digest_changed",
                        "llm_repair_used": False,
                        "generation_mode": "fallback",
                    }
                )
                with shared_lock:
                    findings.append(authority_finding)
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                return step_record
            final_llm_audit_due = bool(
                candidate_code_digest == deterministic_contract_approved_code_digest
                and candidate_code_digest != final_concept_gate_approved_code_digest
            )
            if (
                candidate_code_digest != concept_approved_code_digest
                or final_llm_audit_due
            ):
                # Every mutation still returns through deterministic semantic and
                # mechanical gates before execution.  The LLM concept auditor is
                # invoked only for the exact digest whose local run and early
                # deterministic contracts already passed, preventing runtime- or
                # contract-broken drafts from consuming repeated audit calls.
                usage_findings = concept_audit.findings_for_code(
                    code,
                    include_llm=final_llm_audit_due,
                )
                step_record["usage_findings"] = [
                    finding.model_dump() for finding in usage_findings
                ]
                post_mutation_errors = [
                    finding for finding in usage_findings if finding.severity == "error"
                ]
                if post_mutation_errors:
                    if (
                        final_llm_audit_due
                        and step_attempt_state.coordinates is not None
                        and step_attempt_state.current_capsule_ref is not None
                        and not any(
                            str((finding.detail or {}).get("issue_code") or "")
                            in {
                                "llm_concept_audit_provider_failure",
                                "llm_concept_audit_response_invalid",
                            }
                            for finding in usage_findings
                        )
                    ):
                        current_authority = load_verified_step_authority_capsule(
                            run_dir,
                            ref=step_attempt_state.current_capsule_ref,
                            expected_step_id=step.step_id,
                        )
                        if (
                            current_authority.capsule.concept_audit is None
                            or step_record.get("step_authority_audit_cache_miss")
                            == "audit_identity_drift"
                        ):
                            blocked_audit_key = concept_audit.tokens_by_digest.get(
                                candidate_code_digest
                            ) or canonical_sha256(
                                {
                                    "schema": (
                                        "easyicu.capsule_blocked_concept_audit/1"
                                    ),
                                    "step_id": step.step_id,
                                    "code_sha256": candidate_code_digest,
                                    "findings": [
                                        finding.model_dump(mode="json")
                                        for finding in usage_findings
                                    ],
                                }
                            )
                            blocked_ref = seal_concept_audit_capsule(
                                step_attempt_state.coordinates,
                                parent_ref=step_attempt_state.current_capsule_ref,
                                findings=usage_findings,
                                audit_key=blocked_audit_key,
                                auditor_identity_sha256=(
                                    llm_concept_auditor_identity_sha256
                                ),
                                environment_sha256=(concept_audit_environment_sha256),
                                validator_implementation_sha256=(
                                    llm_concept_auditor_implementation_sha256
                                    or canonical_sha256(
                                        "llm_concept_auditor_unavailable"
                                    )
                                ),
                            )
                            checkpoint_authority.checkpoint_capsule(
                                blocked_ref,
                                status="concept_audited_pending_review",
                            )
                    if final_llm_audit_due:
                        # These outputs came from a digest rejected by the final
                        # semantic audit.  They are never eligible for later
                        # sealing/current authority, and a repaired digest must
                        # execute afresh before it can regain contract approval.
                        deterministic_contract_approved_code_digest = None
                        _clear_output_dir(run_dir / "steps" / step.step_id / "outputs")
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
                        repair_reason_for_finding(finding)
                        for finding in post_mutation_errors
                    ]
                    if (
                        worker_progress.deterministic_concept_repairs
                        < _MAX_DETERMINISTIC_CONCEPT_REPAIRS
                    ):
                        deterministic_code, deterministic_names = (
                            _authorized_deterministic_concept_repair(
                                script_text=code,
                                error_messages=post_mutation_messages,
                                repair_reasons=post_mutation_reasons,
                                repair_findings=post_mutation_errors,
                                source=("post_mutation_deterministic_concept_repair"),
                            )
                        )
                        if deterministic_names and deterministic_code != code:
                            before_code = code
                            code = deterministic_code
                            worker_progress.deterministic_concept_repairs += 1
                            worker_progress.applied_concept_repair_names.extend(
                                deterministic_names
                            )
                            step_record["deterministic_concept_repairs"] = (
                                worker_progress.deterministic_concept_repairs
                            )
                            step_record["applied_concept_repair_names"] = list(
                                worker_progress.applied_concept_repair_names
                            )
                            for repair_name in deterministic_names:
                                _record_repair(
                                    repair_id=repair_name,
                                    step_id=step.step_id,
                                    trigger={
                                        "gate": "post_mutation_concept_audit",
                                        "audit_errors": post_mutation_messages,
                                    },
                                    transformation=(
                                        "deterministic concept repair after a "
                                        "contract/runtime mutation"
                                    ),
                                    before_code=before_code,
                                    after_code=code,
                                    selection_rule=(
                                        "applied only because a typed mechanical "
                                        "error named the anti-pattern"
                                    ),
                                )
                            _clear_output_dir(
                                run_dir / "steps" / step.step_id / "outputs"
                            )
                            continue

                    if _llm_repair_budget_available():
                        post_mutation_ticket = typed_repair_ticket(post_mutation_errors)
                        current_post_mutation_repair_authority = (
                            RepairPromptAuthority.create(
                                typed_ticket=post_mutation_ticket,
                            )
                        )
                        post_mutation_repair_authority = RepairPromptAuthority.create(
                            typed_ticket=[
                                *post_mutation_ticket,
                                *_monotonic_concept_constraint_ticket(),
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
                        worker_progress.concept_repair_attempts += 1
                        if not _consume_llm_repair_budget(
                            "post_mutation_concept",
                            before_code=code,
                            repair_ticket=post_mutation_repair_log,
                            repair_authority=post_mutation_repair_authority,
                            current_repair_authority=(
                                current_post_mutation_repair_authority
                            ),
                            provider_category="post_mutation_concept_repair",
                            failure_status="concept_failed",
                        ):
                            raise AssertionError(
                                "LLM repair budget changed without mutation"
                            )
                        step_record["concept_repair_attempts"] = (
                            worker_progress.concept_repair_attempts
                        )
                        emit_progress(
                            "coder",
                            (
                                "Repairing post-mutation concept violation for "
                                f"{step.step_id}."
                            ),
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                            repair_attempts=step_repair_budget.llm_repair_attempts,
                        )
                        _remember_concept_constraints(post_mutation_errors)
                        try:
                            code = _repair_with_capsule(
                                failure_status="concept_failed",
                                context=coder_context,
                                step=step,
                                code=code,
                                run_log=post_mutation_repair_log,
                                repair_authority=post_mutation_repair_authority,
                                current_repair_authority=(
                                    current_post_mutation_repair_authority
                                ),
                                attempt=worker_progress.concept_repair_attempts,
                                provider_budget=provider_budget,
                                provider_category="post_mutation_concept_repair",
                                logical_repair_attempt_id=(
                                    step_repair_budget.llm_repair_attempts
                                ),
                            )
                            _sync_provider_budget()
                            worker_progress.llm_repair_used = True
                            _clear_output_dir(
                                run_dir / "steps" / step.step_id / "outputs"
                            )
                            continue
                        except (
                            ProviderCallBudgetReceiptError,
                            StepAuthorityRuntimeError,
                            StepAuthorityCapsuleError,
                        ):
                            raise
                        except Exception as exc:
                            _sync_provider_budget()
                            checkpoint_error: Optional[Exception] = None
                            try:
                                checkpoint = store_quarantined_concept_draft(
                                    run_dir=run_dir,
                                    step_id=step.step_id,
                                    code=code,
                                    findings=_quarantine_error_payloads(
                                        post_mutation_errors
                                    ),
                                )
                                step_record["quarantined_draft_sha256"] = (
                                    checkpoint.sha256
                                )
                                step_record["quarantined_draft_relative_path"] = (
                                    checkpoint.relative_path
                                )
                                step_record["quarantined_requires_repair"] = True
                            except Exception as checkpoint_exc:
                                checkpoint_error = checkpoint_exc
                            fallback_code = _deterministic_fallback_code(
                                "concept_repair_failed"
                            )
                            if fallback_code is not None:
                                code = fallback_code
                                _clear_output_dir(
                                    run_dir / "steps" / step.step_id / "outputs"
                                )
                                continue
                            with shared_lock:
                                findings.extend(usage_findings)
                                if checkpoint_error is not None:
                                    findings.append(
                                        ValidationFinding(
                                            validator="resume",
                                            severity="warning",
                                            message=(
                                                "Could not preserve the rejected final "
                                                "concept-audit draft for step "
                                                f"{step.step_id}: {checkpoint_error}"
                                            ),
                                            detail={"step_id": step.step_id},
                                        )
                                    )
                                findings.append(
                                    ValidationFinding(
                                        validator="coder",
                                        severity="error",
                                        message=(
                                            "Coder repair failed after post-mutation "
                                            "concept audit for step "
                                            f"{step.step_id}: {exc}"
                                        ),
                                        detail={"step_id": step.step_id},
                                    )
                                )
                                step_record["status"] = "repair_failed"
                                _append_terminal_step_record(
                                    per_step_records, step_record
                                )
                                _flush_partial_manifest()
                            emit_progress(
                                "coder",
                                f"Concept-audit repair failed for {step.step_id}.",
                                status="error",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                            )
                            return step_record

                    if not _logical_llm_repair_budget_available():
                        step_record["step_llm_repair_budget_exhausted"] = True
                        step_record["step_llm_repair_budget"] = (
                            pipeline._max_step_llm_repair_attempts
                        )
                    checkpoint_error: Optional[Exception] = None
                    try:
                        checkpoint = store_quarantined_concept_draft(
                            run_dir=run_dir,
                            step_id=step.step_id,
                            code=code,
                            findings=_quarantine_error_payloads(post_mutation_errors),
                        )
                        step_record["quarantined_draft_sha256"] = checkpoint.sha256
                        step_record["quarantined_draft_relative_path"] = (
                            checkpoint.relative_path
                        )
                        step_record["quarantined_requires_repair"] = True
                    except Exception as checkpoint_exc:
                        checkpoint_error = checkpoint_exc
                    step_record["status"] = "blocked_by_concept_audit"
                    step_record["post_repair_concept_audit_block"] = {
                        "code_sha256": candidate_code_digest,
                        "errors": [
                            finding.model_dump(mode="json")
                            for finding in post_mutation_errors
                        ],
                    }
                    with shared_lock:
                        findings.extend(usage_findings)
                        if checkpoint_error is not None:
                            findings.append(
                                ValidationFinding(
                                    validator="resume",
                                    severity="warning",
                                    message=(
                                        "Could not preserve post-repair code rejected "
                                        f"by concept audit for step {step.step_id}: "
                                        f"{checkpoint_error}"
                                    ),
                                    detail={"step_id": step.step_id},
                                )
                            )
                        _append_terminal_step_record(per_step_records, step_record)
                        _flush_partial_manifest()
                    emit_progress(
                        "audit",
                        f"Concept audit blocked mutated code for {step.step_id}.",
                        status="error",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    return step_record
                with shared_lock:
                    findings.extend(usage_findings)
                concept_approved_code_digest = candidate_code_digest
                step_record["concept_approved_code_sha256"] = (
                    concept_approved_code_digest
                )
                step_record["deterministic_preflight_approved_code_sha256"] = (
                    concept_approved_code_digest
                )
                if final_llm_audit_due:
                    final_concept_gate_approved_code_digest = candidate_code_digest
                    step_record["final_concept_gate_approved_code_sha256"] = (
                        final_concept_gate_approved_code_digest
                    )
                    if candidate_code_digest in concept_audit.completed_digests:
                        final_audit_token = concept_audit.tokens_by_digest.get(
                            candidate_code_digest
                        )
                        if final_audit_token is not None:
                            provider_budget.complete_reserved_category(
                                "concept_audit",
                                token=final_audit_token,
                            )
                        step_record["llm_concept_audit_status"] = "completed"
                        step_record["llm_concept_approved_code_sha256"] = (
                            candidate_code_digest
                        )
                    elif not pipeline._enable_llm_concept_audit:
                        step_record["llm_concept_audit_status"] = "disabled"
                    elif (
                        worker_progress.deterministic_fallback_used
                        or worker_progress.deterministic_standard_executor_used
                    ):
                        step_record["llm_concept_audit_status"] = (
                            "skipped_trusted_deterministic_code"
                        )
                    else:
                        step_record["llm_concept_audit_status"] = (
                            "skipped_no_auditor_client"
                        )
                    if (
                        step_attempt_state.coordinates is not None
                        and step_attempt_state.current_capsule_ref is not None
                    ):
                        current_authority = load_verified_step_authority_capsule(
                            run_dir,
                            ref=step_attempt_state.current_capsule_ref,
                            expected_step_id=step.step_id,
                        )
                        if (
                            current_authority.capsule.concept_audit is None
                            or step_record.get("step_authority_audit_cache_miss")
                            == "audit_identity_drift"
                        ):
                            audit_key = concept_audit.tokens_by_digest.get(
                                candidate_code_digest
                            ) or canonical_sha256(
                                {
                                    "schema": (
                                        "easyicu.capsule_deterministic_"
                                        "concept_audit/1"
                                    ),
                                    "step_id": step.step_id,
                                    "code_sha256": candidate_code_digest,
                                    "findings": [
                                        finding.model_dump(mode="json")
                                        for finding in usage_findings
                                    ],
                                }
                            )
                            audited_ref = seal_concept_audit_capsule(
                                step_attempt_state.coordinates,
                                parent_ref=step_attempt_state.current_capsule_ref,
                                findings=usage_findings,
                                audit_key=audit_key,
                                auditor_identity_sha256=(
                                    llm_concept_auditor_identity_sha256
                                ),
                                environment_sha256=(concept_audit_environment_sha256),
                                validator_implementation_sha256=(
                                    llm_concept_auditor_implementation_sha256
                                    or canonical_sha256(
                                        "llm_concept_auditor_unavailable"
                                    )
                                ),
                            )
                            checkpoint_authority.checkpoint_capsule(
                                audited_ref,
                                status="concept_audited_pending_review",
                            )
                    # Reuse the already validated outputs.  No second execution
                    # of unchanged code is needed after the digest-bound audit.
                    break

            current_generation_mode = worker_progress.generation_mode()
            run_label = {
                "llm": "generated script",
                "resumed_code_reuse": "resumed script",
                "fallback": "fallback script",
                "deterministic_standard": "standard executor script",
            }.get(current_generation_mode, "repaired script")
            emit_progress(
                "runner",
                f"Running {run_label} for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                repair_attempts=worker_progress.repair_attempts,
            )
            execution_runner = runner
            execution_timeout_seconds = pipeline._timeout_seconds
            if worker_progress.deterministic_standard_executor_used:
                # A registered standard executes the exact typed workload the
                # planner froze. Give it the dedicated bounded timeout even
                # when the step also consumes a staged primary-cohort universe.
                execution_timeout_seconds = pipeline._standard_executor_timeout_seconds
            if step_execution_cohort_path != cohort_path or (
                worker_progress.deterministic_standard_executor_used
            ):
                execution_runner = pipeline._build_runner(
                    run_dir=run_dir,
                    cohort_path=step_execution_cohort_path,
                    target_outcome=context.target_outcome,
                    universe_path=universe_path,
                    **run_input_authority_state.runner_bindings(),
                    timeout_seconds=execution_timeout_seconds,
                )
            run_input_authority_state.require_trajectory_integrity(
                step_id=step.step_id,
            )
            runner_identity = (
                f"{type(execution_runner).__module__}."
                f"{type(execution_runner).__qualname__}"
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
                pipeline._runner_kind == "custom"
                and not (
                    isinstance(runner_authority_identity, str)
                    and re.fullmatch(r"[0-9a-f]{64}", runner_authority_identity)
                    is not None
                )
            )
            execution_context_digest = execution_context_sha256(
                code_sha256=candidate_code_digest,
                resolved_inputs_sha256=resolved_inputs_sha256,
                cohort_sha256=sha256_of_file(step_execution_cohort_path),
                universe_sha256=sha256_of_file(universe_path),
                runner_identity=runner_identity,
                timeout_seconds=execution_timeout_seconds,
                requested_network_policy=runner_network_identity,
                runtime_environment_sha256=(current_execution_runtime_sha256()),
                runner_configuration_sha256=canonical_sha256(
                    {
                        "schema": "easyicu.runner_configuration/1",
                        "runner_identity": runner_identity,
                        "configured_kind": str(pipeline._runner_kind),
                        "configured_image": str(pipeline._runner_image or ""),
                        "configured_network": str(pipeline._runner_network),
                        "effective_network": runner_network_identity,
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
                trajectory_sha256=run_input_authority_state.trajectory_sha256,
                trajectory_authority_sha256=(
                    run_input_authority_state.trajectory_authority_sha256
                ),
            )
            replay_execution = (
                step_attempt_state.selected_resume_capsule
                if (
                    custom_runner_replay_allowed
                    and not step_attempt_state.capsule_execution_replay_consumed
                    and step_attempt_state.selected_resume_capsule is not None
                    and step_attempt_state.selected_resume_capsule.capsule.execution
                    is not None
                    and step_attempt_state.selected_resume_capsule.capsule.candidate_code.sha256
                    == candidate_code_digest
                )
                else None
            )
            if not custom_runner_replay_allowed:
                step_record["step_authority_execution_cache_miss"] = (
                    "custom_runner_authority_unbound"
                )
            # DockerRunner must first prove any previous timed-out container
            # is quiescent before its bind-mounted output directory is reused;
            # it therefore owns cleanup inside ``run``. Other backends retain
            # the pipeline's established pre-execution clearing behaviour.
            step_record["execution_timeout_seconds"] = execution_timeout_seconds
            if replay_execution is not None:
                if (
                    replay_execution.capsule.execution.execution_context_sha256
                    != execution_context_digest
                ):
                    step_record["step_authority_execution_cache_miss"] = (
                        "execution_context_drift"
                    )
                    replay_execution = None
                else:
                    try:
                        run_result = materialize_sealed_run_result(
                            run_dir,
                            replay_execution,
                            expected_execution_context_sha256=(
                                execution_context_digest
                            ),
                        )
                    except StepAuthorityRuntimeError as exc:
                        replay_finding = ValidationFinding(
                            validator="step_authority_capsule",
                            severity="error",
                            message=(
                                "Checkpoint-selected execution could not be "
                                f"replayed safely for step {step.step_id}."
                            ),
                            detail={"step_id": step.step_id, "reason": str(exc)},
                        )
                        step_record["status"] = "contract_failed"
                        with shared_lock:
                            findings.append(replay_finding)
                            _append_terminal_step_record(per_step_records, step_record)
                            _flush_partial_manifest()
                        return step_record
                    step_attempt_state.capsule_execution_replay_consumed = True
                    step_record["capsule_execution_replayed"] = True
            if replay_execution is None:
                if (
                    step_attempt_state.coordinates is not None
                    and step_attempt_state.current_capsule_ref is not None
                ):
                    current_before_execution = load_verified_step_authority_capsule(
                        run_dir,
                        ref=step_attempt_state.current_capsule_ref,
                        expected_step_id=step.step_id,
                    )
                    if current_before_execution.capsule.stage not in {
                        "candidate",
                        "concept_audited",
                    }:
                        ref = seal_deterministic_candidate(
                            step_attempt_state.coordinates,
                            parent_ref=step_attempt_state.current_capsule_ref,
                            code_ref=persist_candidate_code(
                                step_attempt_state.coordinates, code
                            ),
                            reason="execution_context_changed_or_retry_requested",
                        )
                        checkpoint_authority.checkpoint_capsule(
                            ref,
                            status="candidate_checkpointed",
                        )
                run_result = step_executor.execute(
                    runner=execution_runner,
                    request=LockedStepExecutionRequest(
                        step_id=step.step_id,
                        code=code,
                        resolved_inputs_path=resolved_inputs_path,
                        output_dir=(run_dir / "steps" / step.step_id / "outputs"),
                    ),
                )

            def _seal_actual_execution_result() -> None:
                if (
                    replay_execution is not None
                    or step_attempt_state.coordinates is None
                    or step_attempt_state.current_capsule_ref is None
                ):
                    return
                executed_ref = seal_execution_capsule(
                    step_attempt_state.coordinates,
                    parent_ref=step_attempt_state.current_capsule_ref,
                    run_result=run_result,
                    execution_context_digest=execution_context_digest,
                )
                checkpoint_authority.checkpoint_capsule(
                    executed_ref,
                    status="executed_pending_review",
                )

            step_record["outputs_safe_to_collect"] = bool(
                run_result.outputs_safe_to_collect
            )
            authority_findings: List[ValidationFinding] = []
            if primary_cohort_uses_universe:
                cohort_authority_finding = _execution_input_authority_integrity_finding(
                    step_id=step.step_id,
                    universe_path=universe_path,
                    cohort_path=cohort_path,
                    expected_universe_sha256=step_record.get("execution_cohort_sha256"),
                    expected_analysis_cohort_sha256=step_record.get(
                        "authoritative_analysis_cohort_sha256"
                    ),
                )
                if cohort_authority_finding is not None:
                    authority_findings.append(cohort_authority_finding)
            trajectory_authority_finding = (
                run_input_authority_state.trajectory_integrity_finding(
                    step_id=step.step_id
                )
            )
            if trajectory_authority_finding is not None:
                authority_findings.append(trajectory_authority_finding)
            if authority_findings:
                if run_result.outputs_safe_to_collect:
                    _clear_output_dir(run_result.out_dir)
                step_record.update(
                    {
                        "status": "blocked_input_authority_mutation",
                        "input_authority_findings": [
                            item.model_dump() for item in authority_findings
                        ],
                    }
                )
                with shared_lock:
                    run_input_authority_state.mark_corrupted(step_id=step.step_id)
                    findings.extend(authority_findings)
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Rejected mutated execution authority for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record
            if not run_result.outputs_safe_to_collect:
                # The backend could not prove that a process/container with a
                # writable output mount was stopped.  Those outputs remain
                # mutable and are therefore ineligible for inspection,
                # hashing, repair, cleanup, or evidence registration. Docker
                # keeps host-owned script/log control copies, but this step is
                # still terminal until a later explicit retry resolves the
                # teardown sentinel first.
                unsafe_reason = "runner_output_teardown_unconfirmed"
                step_record.update(
                    {
                        "status": (
                            "deterministic_standard_blocked"
                            if is_trajectory_stability_standard
                            else "execution_failed"
                        ),
                        "diagnostic_only": True,
                        "runner_output_safety_reason": unsafe_reason,
                    }
                )
                if is_trajectory_stability_standard:
                    step_record["standard_executor_terminal_reason"] = (
                        "executor_runtime_failure"
                    )
                elif sealed_renderer_authorized_code_sha256 is not None:
                    step_record.update(
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
                        f"Step {step.step_id} was stopped because the execution "
                        "backend could not confirm teardown of its writable "
                        "mount; no files from that mount were inspected or "
                        "registered."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "reason": unsafe_reason,
                        "timed_out": bool(run_result.timed_out),
                        "returncode": int(run_result.returncode),
                    },
                )
                _seal_actual_execution_result()
                with shared_lock:
                    findings.append(unsafe_finding)
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "runner",
                    f"Execution mount teardown was not confirmed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record
            executed_code_digest = sha256_of_file(run_result.script_path)
            step_record["executed_code_sha256"] = executed_code_digest
            if sealed_renderer_authorized_code_sha256 is not None:
                step_record["sealed_renderer_executed_code_matches_authority"] = (
                    executed_code_digest == sealed_renderer_authorized_code_sha256
                )
            if (
                concept_approved_code_digest is None
                or executed_code_digest != concept_approved_code_digest
            ):
                integrity_finding = ValidationFinding(
                    validator="post_repair_concept_gate",
                    severity="error",
                    message=(
                        "The executed analysis script did not match the exact "
                        f"concept-approved code digest for step {step.step_id}; "
                        "outputs were rejected before evidence registration."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "concept_approved_code_sha256": concept_approved_code_digest,
                        "executed_code_sha256": executed_code_digest,
                        "script_path": str(run_result.script_path),
                    },
                )
                _clear_output_dir(run_result.out_dir)
                step_record["status"] = "blocked_script_integrity"
                step_record["script_integrity_findings"] = [
                    integrity_finding.model_dump()
                ]
                if sealed_renderer_authorized_code_sha256 is not None:
                    step_record.update(
                        {
                            "sealed_renderer_terminal_reason": (
                                "executed_code_digest_mismatch"
                            ),
                            "llm_repair_used": False,
                            "generation_mode": "fallback",
                        }
                    )
                with shared_lock:
                    findings.append(integrity_finding)
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Rejected script-integrity mismatch for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record
            _seal_actual_execution_result()
            step_record["returncode"] = run_result.returncode
            step_record["timed_out"] = run_result.timed_out
            step_record["requested_network_policy"] = (
                run_result.requested_network_policy
            )
            step_record["effective_isolation"] = run_result.effective_isolation
            step_record["isolation_degraded"] = run_result.isolation_degraded
            if run_result.isolation_degradation_reason:
                step_record["isolation_degradation_reason"] = (
                    run_result.isolation_degradation_reason
                )
            step_record["code_repair_attempts"] = worker_progress.repair_attempts

            if current_generation_mode == "llm":
                script_description = (
                    f"Generated analysis script for step {step.step_id}."
                )
            elif current_generation_mode == "resumed_code_reuse":
                script_description = (
                    f"Reused prior agent-generated analysis script for step "
                    f"{step.step_id}."
                )
            elif current_generation_mode == "fallback":
                script_description = (
                    f"Deterministic fallback analysis script for step {step.step_id}."
                )
            elif current_generation_mode == "deterministic_standard":
                script_description = (
                    "Planner-selected deterministic standard executor adapter for "
                    f"step {step.step_id}."
                )
            else:
                total_repair_attempts = (
                    worker_progress.repair_attempts
                    + worker_progress.concept_repair_attempts
                )
                script_description = (
                    f"Repaired analysis script for step {step.step_id} "
                    f"(attempt {total_repair_attempts})."
                )
            script_digest = sha256_of_file(run_result.script_path)
            script_authority = "\0".join(
                (step.step_id, script_digest, current_generation_mode)
            )
            script_evidence_id = (
                "code_analysis_"
                + hashlib.sha256(script_authority.encode("utf-8")).hexdigest()[:16]
            )
            script_record = evidence.register_file(
                kind="code",
                description=script_description,
                source_path=run_result.script_path,
                produced_by_step=step.step_id,
                inputs=resolved_input_evidence_ids or None,
                evidence_id=script_evidence_id,
                producer=(
                    "standard_executor"
                    if current_generation_mode == "deterministic_standard"
                    else "coder"
                ),
                generation_mode=current_generation_mode,
                prompt_pack_version=prompt_version,
                metadata={
                    "repair_attempts": worker_progress.repair_attempts,
                    "concept_repair_attempts": worker_progress.concept_repair_attempts,
                    "deterministic_concept_repairs": worker_progress.deterministic_concept_repairs,
                    "llm_repair_used": worker_progress.llm_repair_used,
                    "fallback_reason": step_record.get("deterministic_code_fallback"),
                    "runner_repair": worker_progress.runner_repair_name,
                    "resumed_code_evidence_id": step_record.get(
                        "resumed_code_evidence_id"
                    ),
                    "resumed_code_relative_path": step_record.get(
                        "resumed_code_relative_path"
                    ),
                    "resumed_from_generation_mode": step_record.get(
                        "resumed_from_generation_mode"
                    ),
                    "resumed_code_evidence_generation_mode": step_record.get(
                        "resumed_code_evidence_generation_mode"
                    ),
                    "resumed_quarantined_draft": quarantine_state.resumed_draft_used,
                    "quarantined_draft_sha256": step_record.get(
                        "quarantined_draft_sha256"
                    ),
                    "quarantined_repair_succeeded": quarantine_state.repair_succeeded,
                    "quarantine_policy_superseded": quarantine_state.policy_superseded,
                    "quarantine_policy_superseded_findings": step_record.get(
                        "quarantine_policy_superseded_findings"
                    ),
                    "llm_signature": llm_signature,
                },
            )
            step_record["script_evidence_id"] = script_record.evidence_id
            log_path = run_result.runner_log_path or (run_result.cwd / "run.log")
            if log_path.exists():
                evidence.register_file(
                    kind="log",
                    description=f"stdout/stderr log for step {step.step_id}.",
                    source_path=log_path,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    producer="runner",
                    generation_mode=current_generation_mode,
                    metadata={
                        "repair_attempts": worker_progress.repair_attempts,
                        "concept_repair_attempts": worker_progress.concept_repair_attempts,
                        "deterministic_concept_repairs": (
                            worker_progress.deterministic_concept_repairs
                        ),
                        "llm_repair_used": worker_progress.llm_repair_used,
                        "fallback_reason": step_record.get(
                            "deterministic_code_fallback"
                        ),
                        "runner_repair": worker_progress.runner_repair_name,
                        "resumed_from_generation_mode": step_record.get(
                            "resumed_from_generation_mode"
                        ),
                    },
                )

            if run_result.succeeded:
                # Step-summary salvage reshapes the source from which numbers are
                # registered, so each salvage is recorded in the repair ledger
                # (ENG-REPAIR1 P1.5). The salvage decision lives in
                # salvage_step_summary() so it is unit-testable end-to-end; here
                # we only record what it did.
                salvage_outcome = salvage_step_summary(run_result, step=step)
                if salvage_outcome is not None:
                    if salvage_outcome.reset_artefacts:
                        run_result.artefacts = sorted(
                            p for p in run_result.out_dir.iterdir() if p.is_file()
                        )
                    _record_repair(
                        repair_id=salvage_outcome.repair_id,
                        step_id=step.step_id,
                        trigger={
                            "source": "summary_salvage",
                            "reason": salvage_outcome.trigger_reason,
                        },
                        transformation=salvage_outcome.transformation,
                        selection_rule=salvage_outcome.selection_rule,
                    )
                if not run_result.artefacts:
                    if is_trajectory_stability_standard:
                        standard_executor_terminal_block = True
                        standard_executor_terminal_reason = "missing_executor_outputs"
                        break
                    fallback_code = _deterministic_fallback_code("no_artefacts")
                    if fallback_code is not None:
                        code = fallback_code
                        _clear_output_dir(run_result.out_dir)
                        continue
                visual_step_summary: Dict[str, Any] = {}
                visual_summary_path = run_result.out_dir / "step_summary.json"
                if visual_summary_path.exists():
                    try:
                        vloaded = json.loads(
                            visual_summary_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        vloaded = None
                    if isinstance(vloaded, dict):
                        visual_step_summary = vloaded
                    else:
                        visual_step_summary = {"raw": vloaded}
                if worker_progress.runner_repair_name and is_sealed_renderer_repair(
                    worker_progress.runner_repair_name
                ):
                    visual_step_summary = _write_host_input_binding_receipts(
                        out_dir=run_result.out_dir,
                        step_summary=visual_step_summary,
                        resolved_input_bindings=resolved_input_bindings,
                    )
                if is_trajectory_stability_standard:
                    terminal_status = (
                        str(visual_step_summary.get("status") or "").strip().lower()
                    )
                    if terminal_status != "ok":
                        standard_executor_terminal_block = True
                        standard_executor_terminal_reason = "executor_reported_" + (
                            terminal_status or "missing_status"
                        )
                        standard_executor_terminal_summary = dict(visual_step_summary)
                        break
                step_figures = [
                    art
                    for art in run_result.artefacts
                    if art.suffix.lower() in {".png", ".svg", ".tiff", ".tif"}
                ]
                visual_gate = collect_visual_gate_result(
                    enabled=pipeline._enable_visual_qa,
                    step_figures=step_figures,
                    step=step,
                    step_summary=visual_step_summary,
                )
                if visual_gate.ran:
                    visual_findings = list(visual_gate.findings)
                    step_record["visual_findings"] = [
                        f.model_dump() for f in visual_findings
                    ]
                    if visual_gate.has_errors:
                        visual_repair_decision = decide_visual_repair(
                            visual_gate,
                            sealed=sealed_renderer_authorized_code_sha256 is not None,
                            attempts_exhausted=(
                                worker_progress.visual_repair_attempts
                                >= pipeline._max_code_repair_attempts
                            ),
                            budget_available=_llm_repair_budget_available(),
                        )
                        if (
                            visual_repair_decision.action
                            is VisualRepairAction.SEALED_SUPPRESS
                        ):
                            demoted_findings = list(visual_gate.demoted_findings)
                            blocking_visual_errors = list(visual_gate.blocking_errors)
                            step_record["visual_findings"] = [
                                finding.model_dump() for finding in demoted_findings
                            ]
                            step_record["sealed_renderer_visual_repair_suppressed"] = (
                                True
                            )
                            step_record["visual_qa_demoted"] = visual_gate.was_demoted
                            with shared_lock:
                                findings.extend(demoted_findings)
                            if blocking_visual_errors:
                                step_record.update(
                                    {
                                        "status": "execution_failed",
                                        "diagnostic_only": True,
                                        "sealed_renderer_terminal_reason": (
                                            "visual_qa_failed"
                                        ),
                                        "llm_repair_used": False,
                                        "generation_mode": "fallback",
                                    }
                                )
                                with shared_lock:
                                    _append_terminal_step_record(
                                        per_step_records, step_record
                                    )
                                    _flush_partial_manifest()
                                emit_progress(
                                    "visual_qa",
                                    (
                                        "Visual QA blocked sealed renderer "
                                        f"{step.step_id}; coder repair was not "
                                        "authorized."
                                    ),
                                    status="error",
                                    run_id=run_id,
                                    step_id=step.step_id,
                                    current_step=step_current,
                                    total_steps=total_steps,
                                )
                                return step_record
                            emit_progress(
                                "visual_qa",
                                (
                                    "Cosmetic visual QA findings were retained as "
                                    "warnings for sealed renderer "
                                    f"{step.step_id}; its verified code and outputs "
                                    "were not rewritten."
                                ),
                                status="warning",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                            )
                        elif (
                            visual_repair_decision.action
                            is VisualRepairAction.EXHAUSTED
                        ):
                            fallback_code = _deterministic_fallback_code("visual_qa")
                            if fallback_code is not None:
                                code = fallback_code
                                _clear_output_dir(run_result.out_dir)
                                continue
                            demoted_findings = list(visual_gate.demoted_findings)
                            blocking_visual_errors = list(visual_gate.blocking_errors)
                            step_record["visual_findings"] = [
                                finding.model_dump() for finding in demoted_findings
                            ]
                            with shared_lock:
                                findings.extend(demoted_findings)
                            step_record["visual_qa_demoted"] = visual_gate.was_demoted
                            if blocking_visual_errors:
                                step_record["status"] = "execution_failed"
                                with shared_lock:
                                    _append_terminal_step_record(
                                        per_step_records, step_record
                                    )
                                    _flush_partial_manifest()
                                emit_progress(
                                    "visual_qa",
                                    (
                                        f"Visual QA blocked {step.step_id} after "
                                        f"{worker_progress.visual_repair_attempts} layout repair "
                                        "attempts."
                                    ),
                                    status="error",
                                    run_id=run_id,
                                    step_id=step.step_id,
                                    current_step=step_current,
                                    total_steps=total_steps,
                                )
                                return step_record
                            emit_progress(
                                "visual_qa",
                                (
                                    f"Cosmetic visual QA findings demoted to warning "
                                    f"for {step.step_id} after "
                                    f"{worker_progress.visual_repair_attempts} layout repair attempts."
                                ),
                                status="warning",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                            )
                            # Fall through to contract checks and evidence
                            # registration only when every remaining visual
                            # error was a deterministic layout/cosmetic issue.
                        else:
                            worker_progress.visual_repair_attempts += 1
                            visual_host_guidance = visual_repair_decision.host_guidance
                            current_visual_repair_authority = (
                                RepairPromptAuthority.create(
                                    typed_ticket=list(
                                        visual_repair_decision.repair_ticket
                                    ),
                                    host_guidance=visual_host_guidance,
                                )
                            )
                            visual_repair_authority = RepairPromptAuthority.create(
                                typed_ticket=[
                                    *visual_repair_decision.repair_ticket,
                                    *_monotonic_concept_constraint_ticket(),
                                ],
                                host_guidance=visual_host_guidance,
                            )
                            visual_repair_log = visual_repair_decision.repair_log
                            if not _consume_llm_repair_budget(
                                "visual",
                                before_code=code,
                                repair_ticket=visual_repair_log,
                                repair_authority=visual_repair_authority,
                                current_repair_authority=(
                                    current_visual_repair_authority
                                ),
                                provider_category="visual_repair",
                                failure_status="visual_failed",
                            ):
                                raise AssertionError(
                                    "LLM repair budget changed without mutation"
                                )
                            worker_progress.repair_attempts += 1
                            step_record["code_repair_attempts"] = (
                                worker_progress.repair_attempts
                            )
                            step_record["visual_repair_attempts"] = (
                                worker_progress.visual_repair_attempts
                            )
                            emit_progress(
                                "visual_qa",
                                f"Repairing figure layout for {step.step_id}.",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                                repair_attempts=worker_progress.repair_attempts,
                                visual_repair_attempts=worker_progress.visual_repair_attempts,
                            )
                            try:
                                code = _repair_with_capsule(
                                    failure_status="visual_failed",
                                    context=coder_context,
                                    step=step,
                                    code=code,
                                    run_log=visual_repair_log,
                                    repair_authority=visual_repair_authority,
                                    current_repair_authority=(
                                        current_visual_repair_authority
                                    ),
                                    attempt=worker_progress.visual_repair_attempts,
                                    provider_budget=provider_budget,
                                    provider_category="visual_repair",
                                    logical_repair_attempt_id=(
                                        step_repair_budget.llm_repair_attempts
                                    ),
                                )
                                _sync_provider_budget()
                                worker_progress.llm_repair_used = True
                                _clear_output_dir(run_result.out_dir)
                                continue
                            except (
                                ProviderCallBudgetReceiptError,
                                StepAuthorityRuntimeError,
                                StepAuthorityCapsuleError,
                            ):
                                raise
                            except Exception as exc:
                                _sync_provider_budget()
                                demoted_findings = list(visual_gate.demoted_findings)
                                blocking_visual_errors = list(
                                    visual_gate.blocking_errors
                                )
                                if not blocking_visual_errors:
                                    provider_finding = ValidationFinding(
                                        validator="coder",
                                        severity="warning",
                                        message=(
                                            "Cosmetic visual-layout repair was "
                                            f"unavailable for step {step.step_id}; "
                                            "the current data-valid artifacts were "
                                            f"retained: {exc}"
                                        ),
                                        detail={
                                            "step_id": step.step_id,
                                            "error_type": type(exc).__name__,
                                            "visual_repair_attempts": (
                                                worker_progress.visual_repair_attempts
                                            ),
                                        },
                                    )
                                    step_record["visual_findings"] = [
                                        finding.model_dump()
                                        for finding in demoted_findings
                                    ]
                                    step_record["visual_qa_demoted"] = True
                                    step_record["visual_repair_provider_failed"] = True
                                    with shared_lock:
                                        findings.extend(demoted_findings)
                                        findings.append(provider_finding)
                                    emit_progress(
                                        "visual_qa",
                                        (
                                            "Cosmetic visual repair unavailable; "
                                            f"retained current artifacts for {step.step_id}."
                                        ),
                                        status="warning",
                                        run_id=run_id,
                                        step_id=step.step_id,
                                        current_step=step_current,
                                        total_steps=total_steps,
                                    )
                                else:
                                    fallback_code = _deterministic_fallback_code(
                                        "visual_qa_repair_failed"
                                    )
                                    if fallback_code is not None:
                                        code = fallback_code
                                        _clear_output_dir(run_result.out_dir)
                                        continue
                                    with shared_lock:
                                        findings.extend(visual_findings)
                                        findings.append(
                                            ValidationFinding(
                                                validator="coder",
                                                severity="error",
                                                message=(
                                                    "Coder repair failed after visual QA "
                                                    f"for step {step.step_id}: {exc}"
                                                ),
                                            )
                                        )
                                        step_record["status"] = "repair_failed"
                                        _append_terminal_step_record(
                                            per_step_records, step_record
                                        )
                                        _flush_partial_manifest()
                                    emit_progress(
                                        "visual_qa",
                                        f"Visual QA repair failed for {step.step_id}.",
                                        status="error",
                                        run_id=run_id,
                                        step_id=step.step_id,
                                        current_step=step_current,
                                        total_steps=total_steps,
                                    )
                                    return step_record
                with shared_lock:
                    completed_records_snapshot = list(per_step_records)
                # Early pre-registration deterministic contract gate: the SAME
                # 14-validator sequence the final gate runs
                # (_evaluate_final_deterministic_gates), evaluated here before
                # evidence registration so contract errors enter the in-run repair
                # loop instead of becoming a terminal record. The figure-contract
                # canonicalization repair and the figure-contract / figure-source /
                # ordered-stratified validators stay below because the early gate
                # interleaves the canonicalization repair between them.
                early_contract_findings = _step_deterministic_contract_findings(
                    step=step,
                    plan=plan,
                    context=context,
                    step_summary=visual_step_summary,
                    completed_step_records=completed_records_snapshot,
                    resolved_input_bindings=resolved_input_bindings,
                    out_dir=run_result.out_dir,
                    run_dir=run_dir,
                    universe_path=universe_path,
                    cohort_path=cohort_path,
                    execution_cohort_path=step_execution_cohort_path,
                    cross_step_cohort_lock_validator=cross_step_cohort_lock_validator,
                    cross_step_registered_output_validator=(
                        cross_step_registered_output_validator
                    ),
                    cross_step_reconciliation_trace_validator=(
                        cross_step_reconciliation_trace_validator
                    ),
                    step_summary_integrity_validator=step_summary_integrity_validator,
                    step_summary_fraction_validator=step_summary_fraction_validator,
                    cross_step_source_status_validator=(
                        cross_step_source_status_validator
                    ),
                    primary_model_contract_validator=primary_model_contract_validator,
                )
                # Figure quality and source-data errors must enter the same
                # in-run repair loop as table/model contract errors. Checking
                # them only after evidence registration produces a terminal
                # contract_failed record with no opportunity to repair the
                # generated rendering script.
                for contract_path in sorted(
                    run_result.out_dir.glob("*.figure_contract.json")
                ):
                    schema_candidate = (
                        _figure_contract_source_data_canonicalization_candidate(
                            contract_path=contract_path,
                            out_dir=run_result.out_dir,
                        )
                    )
                    if schema_candidate is None:
                        continue
                    before_contract, after_contract, source_names = schema_candidate
                    repair_id = _FIGURE_CONTRACT_SOURCE_DATA_SCHEMA_REPAIR_ID
                    if not _automatic_repair_authorized(
                        repair_id,
                        step=step,
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
                    step_record.setdefault(
                        "figure_contract_schema_canonicalizations", []
                    ).append(
                        {
                            "contract": contract_path.name,
                            "source_data": list(source_names),
                            "repair_id": repair_id,
                        }
                    )
                    _record_repair(
                        repair_id=repair_id,
                        step_id=str(step.step_id),
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
                early_contract_findings += _post_canonicalization_figure_findings(
                    step=step,
                    out_dir=run_result.out_dir,
                    run_dir=run_dir,
                    step_summary=visual_step_summary,
                    completed_step_records=completed_records_snapshot,
                    resolved_input_bindings=resolved_input_bindings,
                    execution_cohort_path=step_execution_cohort_path,
                    figure_contract_validator=figure_contract_validator,
                    figure_source_validator=figure_source_validator,
                )
                unowned_sealed_markers = _unowned_sealed_authority_markers(
                    visual_step_summary,
                    authorized_code_sha256=(sealed_renderer_authorized_code_sha256),
                )
                if unowned_sealed_markers:
                    early_contract_findings.append(
                        ValidationFinding(
                            validator="sealed_renderer_authority",
                            severity="error",
                            message=(
                                "Generated code reported sealed-renderer authority "
                                "that the host did not authorize before execution."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "unowned_authority_markers": unowned_sealed_markers,
                            },
                        )
                    )
                if (
                    sealed_renderer_authorized_code_sha256 is not None
                    and visual_step_summary.get("rendering_only") is not True
                ):
                    early_contract_findings.append(
                        ValidationFinding(
                            validator="sealed_renderer_authority",
                            severity="error",
                            message=(
                                "The authorized figure adapter did not report its "
                                "required rendering-only execution scope."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "repair_id": sealed_renderer_state.repair_id,
                                "reported_rendering_only": visual_step_summary.get(
                                    "rendering_only"
                                ),
                            },
                        )
                    )
                reported_slot_bindings = visual_step_summary.get(
                    "planner_product_slot_bindings"
                )
                reported_product_slots = (
                    {
                        str(product): str(binding.get("slot") or "")
                        for product, binding in reported_slot_bindings.items()
                        if isinstance(binding, Mapping)
                    }
                    if isinstance(reported_slot_bindings, Mapping)
                    else {}
                )
                if sealed_renderer_authorized_code_sha256 is not None:
                    parent_step_id = str(step.step_id or "").removesuffix("_figure")
                    parent_out = run_dir / "steps" / parent_step_id / "outputs"
                    try:
                        read_digest_bound_artifact_snapshot(
                            parent_out=parent_out,
                            artifact_digests=sealed_renderer_state.parent_digests,
                        )
                        step_record["sealed_renderer_parent_receipt_verified"] = True
                    except ValueError:
                        step_record["sealed_renderer_parent_receipt_verified"] = False
                        early_contract_findings.append(
                            ValidationFinding(
                                validator="sealed_renderer_authority",
                                severity="error",
                                message=(
                                    "The sealed renderer's direct-parent inputs "
                                    "changed before host receipt."
                                ),
                                detail={
                                    "step_id": step.step_id,
                                    "repair_id": sealed_renderer_state.repair_id,
                                },
                            )
                        )
                if sealed_renderer_authorized_code_sha256 is not None and (
                    visual_step_summary.get("sealed_renderer_repair")
                    != sealed_renderer_state.repair_id
                    or visual_step_summary.get("sealed_renderer_implementation_sha256")
                    != sealed_renderer_state.implementation_sha256
                    or visual_step_summary.get("sealed_renderer_parent_digests")
                    != sealed_renderer_state.parent_digests
                    or reported_product_slots
                    != sealed_renderer_state.authorized_product_slots
                ):
                    early_contract_findings.append(
                        ValidationFinding(
                            validator="sealed_renderer_authority",
                            severity="error",
                            message=(
                                "The rendered summary did not preserve the exact "
                                "sealed renderer identity and implementation digest."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "expected_repair_id": sealed_renderer_state.repair_id,
                                "reported_repair_id": visual_step_summary.get(
                                    "sealed_renderer_repair"
                                ),
                                "expected_implementation_sha256": (
                                    sealed_renderer_state.implementation_sha256
                                ),
                                "reported_implementation_sha256": (
                                    visual_step_summary.get(
                                        "sealed_renderer_implementation_sha256"
                                    )
                                ),
                                "expected_parent_digests": (
                                    sealed_renderer_state.parent_digests
                                ),
                                "reported_parent_digests": visual_step_summary.get(
                                    "sealed_renderer_parent_digests"
                                ),
                                "expected_product_slots": (
                                    sealed_renderer_state.authorized_product_slots
                                ),
                                "reported_product_slots": reported_product_slots,
                            },
                        )
                    )
                # A deterministic PRIMARY runner owns its step's contract: if it
                # produced the core estimate, planner-requested extra outputs it
                # does not emit are advisory, never a reason to repair-away the
                # trustworthy estimate.
                early_contract_findings = _demote_step_contract_for_primary_runner(
                    step_record, visual_step_summary, early_contract_findings
                )
                early_contract_errors = [
                    f for f in early_contract_findings if f.severity == "error"
                ]
                if early_contract_errors:
                    locked_data_quality_issues = (
                        _locked_measurement_data_quality_issues(early_contract_errors)
                    )
                    if locked_data_quality_issues:
                        step_record.update(
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
                                    for finding in early_contract_findings
                                ],
                                "step_summary": visual_step_summary,
                                "llm_repair_used": worker_progress.llm_repair_used,
                                "generation_mode": current_generation_mode,
                                "code_repair_attempts": worker_progress.repair_attempts,
                                "contract_repair_attempts": (
                                    worker_progress.contract_repair_attempts
                                ),
                            }
                        )
                        with shared_lock:
                            findings.extend(early_contract_findings)
                            _append_terminal_step_record(per_step_records, step_record)
                            _flush_partial_manifest()
                        emit_progress(
                            "contract",
                            (
                                "Locked-cohort measurement provenance failed for "
                                f"{step.step_id}; retained diagnostics without "
                                "attempting a code repair."
                            ),
                            status="error",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        return step_record
                    if sealed_renderer_authorized_code_sha256 is not None:
                        step_record.update(
                            {
                                "status": "contract_failed",
                                "diagnostic_only": True,
                                "sealed_renderer_contract_repair_suppressed": True,
                                "sealed_renderer_terminal_reason": (
                                    "output_contract_failed"
                                ),
                                "contract_findings": [
                                    finding.model_dump()
                                    for finding in early_contract_findings
                                ],
                                "step_summary": visual_step_summary,
                                "llm_repair_used": False,
                                "generation_mode": "fallback",
                            }
                        )
                        with shared_lock:
                            findings.extend(early_contract_findings)
                            _append_terminal_step_record(per_step_records, step_record)
                            _flush_partial_manifest()
                        emit_progress(
                            "contract",
                            (
                                "Contract validation blocked sealed renderer "
                                f"{step.step_id}; its code and outputs were retained "
                                "without coder repair."
                            ),
                            status="error",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        return step_record
                    if is_trajectory_stability_standard:
                        standard_executor_terminal_block = True
                        standard_executor_terminal_reason = (
                            "executor_output_contract_failed"
                        )
                        standard_executor_terminal_summary = dict(visual_step_summary)
                        standard_executor_terminal_findings = list(
                            early_contract_findings
                        )
                        break
                    if pipeline._enable_deterministic_runner_repair:
                        before_repair_code = code
                        summary_repair = _deterministic_summary_repair(
                            code=code,
                            step_summary=visual_step_summary,
                            previous_repair=worker_progress.runner_repair_name,
                            analysis_family=local_runtime_state.analysis_family,
                        )
                        summary_repair = _authorize_automatic_repair(
                            summary_repair,
                            step=step,
                            source="deterministic_summary_repair_before_contract",
                            before_code=before_repair_code,
                        )
                    else:
                        summary_repair = None
                    if summary_repair is not None:
                        worker_progress.contract_repair_attempts += 1
                        worker_progress.repair_attempts += 1
                        worker_progress.runner_repair_name, code = summary_repair
                        step_record["runner_repair"] = (
                            worker_progress.runner_repair_name
                        )
                        step_record["code_repair_attempts"] = (
                            worker_progress.repair_attempts
                        )
                        step_record["contract_repair_attempts"] = (
                            worker_progress.contract_repair_attempts
                        )
                        _record_repair(
                            repair_id=worker_progress.runner_repair_name,
                            step_id=step.step_id,
                            trigger={
                                "source": "deterministic_summary_repair",
                                "step_summary_keys": sorted(
                                    str(key) for key in visual_step_summary.keys()
                                ),
                                "contract_findings": [
                                    f.message for f in early_contract_errors
                                ],
                            },
                            transformation=(
                                "Deterministic repair before LLM contract repair."
                            ),
                            before_code=before_repair_code,
                            after_code=code,
                        )
                        emit_progress(
                            "runner_repair",
                            (
                                f"Applied deterministic summary repair for "
                                f"{step.step_id}: {worker_progress.runner_repair_name}."
                            ),
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        _clear_output_dir(run_result.out_dir)
                        continue
                    if pipeline._enable_deterministic_runner_repair:
                        before_repair_code = code
                        contract_repair = deterministic_contract_repair(
                            code=code,
                            findings=early_contract_errors,
                            previous_repair=worker_progress.runner_repair_name,
                        )
                        contract_repair = _authorize_automatic_repair(
                            contract_repair,
                            step=step,
                            source="deterministic_contract_repair",
                            before_code=before_repair_code,
                        )
                    else:
                        contract_repair = None
                    if contract_repair is not None:
                        worker_progress.contract_repair_attempts += 1
                        worker_progress.repair_attempts += 1
                        worker_progress.runner_repair_name, code = contract_repair
                        step_record["runner_repair"] = (
                            worker_progress.runner_repair_name
                        )
                        step_record["code_repair_attempts"] = (
                            worker_progress.repair_attempts
                        )
                        step_record["contract_repair_attempts"] = (
                            worker_progress.contract_repair_attempts
                        )
                        _record_repair(
                            repair_id=worker_progress.runner_repair_name,
                            step_id=step.step_id,
                            trigger={
                                "source": "deterministic_contract_repair",
                                "contract_findings": [
                                    f.message for f in early_contract_errors
                                ],
                            },
                            transformation=(
                                "Applied a centrally authorized deterministic source "
                                "transformation for objective contract findings."
                            ),
                            before_code=before_repair_code,
                            after_code=code,
                        )
                        emit_progress(
                            "runner_repair",
                            (
                                f"Applied deterministic contract repair for "
                                f"{step.step_id}: {worker_progress.runner_repair_name}."
                            ),
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        _clear_output_dir(run_result.out_dir)
                        continue
                    if (
                        worker_progress.contract_repair_attempts
                        >= pipeline._max_code_repair_attempts
                        or not _llm_repair_budget_available()
                    ):
                        with shared_lock:
                            findings.extend(early_contract_findings)
                            step_record["status"] = "contract_failed"
                            step_record["contract_findings"] = [
                                f.model_dump() for f in early_contract_findings
                            ]
                            step_record["step_summary"] = visual_step_summary
                            _append_terminal_step_record(per_step_records, step_record)
                            _flush_partial_manifest()
                        emit_progress(
                            "contract",
                            (
                                f"Contract violation could not be repaired for "
                                f"{step.step_id}; no LLM repair budget remains."
                            ),
                            status="error",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        return step_record

                    contract_log = _contract_repair_log(early_contract_errors)
                    structured_repair_ticket = typed_repair_ticket(
                        early_contract_errors
                    )
                    current_contract_repair_authority = RepairPromptAuthority.create(
                        typed_ticket=structured_repair_ticket,
                    )
                    repair_guidance = _step_contract_repair_guidance(
                        step=step,
                        step_summary=visual_step_summary,
                        code=code,
                        input_bindings=resolved_input_bindings,
                    )
                    contract_repair_authority = RepairPromptAuthority.create(
                        typed_ticket=[
                            *structured_repair_ticket,
                            *_monotonic_concept_constraint_ticket(),
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
                            visual_step_summary,
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
                    worker_progress.contract_repair_attempts += 1
                    if not _consume_llm_repair_budget(
                        "contract",
                        before_code=code,
                        repair_ticket=contract_repair_log,
                        repair_authority=contract_repair_authority,
                        current_repair_authority=current_contract_repair_authority,
                        provider_category="contract_repair",
                        failure_status="contract_failed",
                    ):
                        raise AssertionError(
                            "LLM repair budget changed without mutation"
                        )
                    worker_progress.repair_attempts += 1
                    step_record["code_repair_attempts"] = (
                        worker_progress.repair_attempts
                    )
                    step_record["contract_repair_attempts"] = (
                        worker_progress.contract_repair_attempts
                    )
                    emit_progress(
                        "coder",
                        f"Repairing contract violation for {step.step_id}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                        repair_attempts=worker_progress.repair_attempts,
                        contract_repair_attempts=worker_progress.contract_repair_attempts,
                    )
                    try:
                        code = _repair_with_capsule(
                            failure_status="contract_failed",
                            context=coder_context,
                            step=step,
                            code=code,
                            run_log=contract_repair_log,
                            repair_authority=contract_repair_authority,
                            current_repair_authority=(
                                current_contract_repair_authority
                            ),
                            attempt=worker_progress.contract_repair_attempts,
                            provider_budget=provider_budget,
                            provider_category="contract_repair",
                            logical_repair_attempt_id=(
                                step_repair_budget.llm_repair_attempts
                            ),
                        )
                        _sync_provider_budget()
                        worker_progress.llm_repair_used = True
                        _clear_output_dir(run_result.out_dir)
                        continue
                    except (
                        ProviderCallBudgetReceiptError,
                        StepAuthorityRuntimeError,
                        StepAuthorityCapsuleError,
                    ):
                        raise
                    except Exception as exc:
                        _sync_provider_budget()
                        fallback_code = _deterministic_fallback_code(
                            "contract_repair_failed"
                        )
                        if fallback_code is not None:
                            code = fallback_code
                            _clear_output_dir(run_result.out_dir)
                            continue
                        with shared_lock:
                            findings.extend(early_contract_findings)
                            findings.append(
                                ValidationFinding(
                                    validator="coder",
                                    severity="error",
                                    message=(
                                        f"Coder repair failed after contract check "
                                        f"for step {step.step_id}: {exc}"
                                    ),
                                )
                            )
                            step_record["status"] = "repair_failed"
                            step_record["contract_findings"] = [
                                f.model_dump() for f in early_contract_findings
                            ]
                            step_record["step_summary"] = visual_step_summary
                            _append_terminal_step_record(per_step_records, step_record)
                            _flush_partial_manifest()
                        emit_progress(
                            "coder",
                            f"Contract repair failed for {step.step_id}.",
                            status="error",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        return step_record
                if (
                    pipeline._enable_deterministic_runner_repair
                    and sealed_renderer_authorized_code_sha256 is None
                ):
                    before_repair_code = code
                    summary_repair = _deterministic_summary_repair(
                        code=code,
                        step_summary=visual_step_summary,
                        previous_repair=worker_progress.runner_repair_name,
                        analysis_family=local_runtime_state.analysis_family,
                    )
                    summary_repair = _authorize_automatic_repair(
                        summary_repair,
                        step=step,
                        source="deterministic_summary_repair_after_contract",
                        before_code=before_repair_code,
                    )
                else:
                    summary_repair = None
                if summary_repair is not None:
                    worker_progress.runner_repair_name, code = summary_repair
                    step_record["runner_repair"] = worker_progress.runner_repair_name
                    _record_repair(
                        repair_id=worker_progress.runner_repair_name,
                        step_id=step.step_id,
                        trigger={
                            "source": "deterministic_summary_repair",
                            "step_summary_keys": sorted(
                                str(key) for key in visual_step_summary.keys()
                            ),
                        },
                        transformation="Deterministic repair after step_summary contract inspection.",
                        before_code=before_repair_code,
                        after_code=code,
                    )
                    emit_progress(
                        "runner_repair",
                        f"Applied deterministic summary repair for {step.step_id}: {worker_progress.runner_repair_name}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    _clear_output_dir(run_result.out_dir)
                    continue
                deterministic_contract_approved_code_digest = candidate_code_digest
                step_record["deterministic_contract_approved_code_sha256"] = (
                    deterministic_contract_approved_code_digest
                )
                # Return once through the digest gate for the single final LLM
                # concept audit.  The output directory is intentionally retained;
                # on approval it proceeds without re-executing unchanged code.
                continue

            if log_path.exists():
                run_log = log_path.read_text(encoding="utf-8", errors="replace")
            else:
                run_log = (run_result.stdout or "") + "\n" + (run_result.stderr or "")
            if is_trajectory_stability_standard:
                # A timeout can interrupt the standard executor between its
                # private streaming write and atomic rename.  That file is an
                # implementation detail, not a diagnostic product, and must
                # be gone before the generic output-directory scan below can
                # register it as evidence.
                _remove_standard_executor_pending_artifacts(run_result.out_dir)
                standard_executor_terminal_block = True
                standard_executor_terminal_reason = "executor_runtime_failure"
                break
            if sealed_renderer_authorized_code_sha256 is not None:
                runtime_finding = ValidationFinding(
                    validator="sealed_renderer_authority",
                    severity="error",
                    message=(
                        "The authorized rendering-only adapter failed at runtime; "
                        "its diagnostics were retained and no deterministic or LLM "
                        "code repair was allowed."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "repair_id": sealed_renderer_state.repair_id,
                        "returncode": run_result.returncode,
                        "timed_out": run_result.timed_out,
                    },
                )
                step_record.update(
                    {
                        "status": "execution_failed",
                        "diagnostic_only": True,
                        "sealed_renderer_runtime_repair_suppressed": True,
                        "sealed_renderer_terminal_reason": "runtime_failure",
                        "llm_repair_used": False,
                        "generation_mode": "fallback",
                    }
                )
                with shared_lock:
                    findings.append(runtime_finding)
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "runner",
                    (
                        f"Sealed renderer failed for {step.step_id}; coder repair "
                        "was not authorized."
                    ),
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record
            if pipeline._enable_deterministic_runner_repair:
                before_repair_code = code
                plugin_repair = pipeline._case_plugin_registry.repair_code(
                    context=context,
                    step=step,
                    code=code,
                    run_log=(run_log + _monotonic_concept_constraint_log()),
                )
                if (
                    plugin_repair is not None
                    and plugin_repair[0] != worker_progress.runner_repair_name
                ):
                    runner_repair = plugin_repair
                else:
                    runner_repair = _deterministic_runner_repair(
                        code=code,
                        run_log=run_log,
                        previous_repair=worker_progress.runner_repair_name,
                        analysis_family=local_runtime_state.analysis_family,
                    )
                runner_repair = _authorize_automatic_repair(
                    runner_repair,
                    step=step,
                    source=(
                        "case_plugin_repair"
                        if plugin_repair is not None and runner_repair is plugin_repair
                        else "deterministic_runner_repair"
                    ),
                    before_code=before_repair_code,
                )
            else:
                runner_repair = None
            if runner_repair is not None:
                worker_progress.runner_repair_name, code = runner_repair
                step_record["runner_repair"] = worker_progress.runner_repair_name
                _record_repair(
                    repair_id=worker_progress.runner_repair_name,
                    step_id=step.step_id,
                    trigger={
                        "source": "deterministic_runner_repair",
                        "run_log_tail": run_log[-1200:],
                    },
                    transformation="Deterministic repair after runner failure.",
                    before_code=before_repair_code,
                    after_code=code,
                )
                emit_progress(
                    "runner_repair",
                    f"Applied deterministic runner repair for {step.step_id}: {worker_progress.runner_repair_name}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                _clear_output_dir(run_result.out_dir)
                continue

            if (
                worker_progress.runtime_repair_attempts
                >= pipeline._max_code_repair_attempts
                or not _llm_repair_budget_available()
            ):
                fallback_code = _deterministic_fallback_code("execution_failure")
                if fallback_code is not None:
                    code = fallback_code
                    _clear_output_dir(run_result.out_dir)
                    continue
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="runner",
                            severity="error",
                            message=(
                                f"Step {step.step_id} "
                                f"{'timed out' if run_result.timed_out else 'failed'} "
                                f"with returncode {run_result.returncode}."
                            ),
                        )
                    )
                    step_record["status"] = "execution_failed"
                    _append_terminal_step_record(per_step_records, step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "runner",
                    f"Execution failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            runtime_repair_applied = False
            runtime_repair_fallback_applied = False
            runtime_repair_authority = RepairPromptAuthority()
            while (
                worker_progress.runtime_repair_attempts
                < pipeline._max_code_repair_attempts
                and _llm_repair_budget_available()
            ):
                worker_progress.repair_attempts += 1
                worker_progress.runtime_repair_attempts += 1
                if not _consume_llm_repair_budget(
                    "runtime",
                    before_code=code,
                    repair_ticket=run_log,
                    repair_authority=runtime_repair_authority,
                    provider_category="runtime_repair",
                    failure_status="runtime_failed",
                ):
                    raise AssertionError("LLM repair budget changed without mutation")
                step_record["code_repair_attempts"] = worker_progress.repair_attempts
                step_record["runtime_repair_attempts"] = (
                    worker_progress.runtime_repair_attempts
                )
                emit_progress(
                    "coder",
                    f"Repairing failed script for {step.step_id}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                    repair_attempts=worker_progress.repair_attempts,
                )
                try:
                    repaired_code = _repair_with_capsule(
                        failure_status="runtime_failed",
                        context=coder_context,
                        step=step,
                        code=code,
                        run_log=run_log,
                        repair_authority=runtime_repair_authority,
                        attempt=worker_progress.repair_attempts,
                        provider_budget=provider_budget,
                        provider_category="runtime_repair",
                        logical_repair_attempt_id=(
                            step_repair_budget.llm_repair_attempts
                        ),
                    )
                    _sync_provider_budget()
                    if not _python_repair_is_materially_changed(code, repaired_code):
                        checkpoint_authority.reject_completed_repair_candidate(
                            repaired_code,
                            reason="runtime_repair_semantic_noop",
                        )
                        raise RuntimeError(
                            "Runtime repair returned no material Python change."
                        )
                    code = repaired_code
                    worker_progress.llm_repair_used = True
                    runtime_repair_applied = True
                    _clear_output_dir(run_result.out_dir)
                    break
                except (
                    ProviderCallBudgetReceiptError,
                    StepAuthorityRuntimeError,
                    StepAuthorityCapsuleError,
                ):
                    raise
                except Exception as exc:
                    _sync_provider_budget()
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
                        and worker_progress.runtime_repair_attempts
                        < pipeline._max_code_repair_attempts
                        and _llm_repair_budget_available()
                        and not provider_budget.exhausted
                    )
                    if can_retry_repair:
                        emit_progress(
                            "coder",
                            (
                                f"Repair attempt did not yield usable code for "
                                f"{step.step_id} "
                                f"(attempt {worker_progress.repair_attempts}): {type(exc).__name__}; "
                                "retrying the repair without re-executing unchanged code."
                            ),
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                            repair_attempts=worker_progress.repair_attempts,
                        )
                        continue

                    # The causal failure is the unavailable repair, not a new
                    # runner failure. Preserve that reason even when the logical
                    # or provider-call budget became exhausted on this attempt.
                    fallback_code = _deterministic_fallback_code("repair_failed")
                    if fallback_code is not None:
                        code = fallback_code
                        runtime_repair_fallback_applied = True
                        _clear_output_dir(run_result.out_dir)
                        break
                    with shared_lock:
                        findings.append(
                            ValidationFinding(
                                validator="coder",
                                severity="error",
                                message=(
                                    f"Coder repair failed for step {step.step_id}: {exc}"
                                ),
                            )
                        )
                        step_record["status"] = "repair_failed"
                        _append_terminal_step_record(per_step_records, step_record)
                        _flush_partial_manifest()
                    emit_progress(
                        "coder",
                        f"Repair failed for {step.step_id}.",
                        status="error",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    return step_record

            if runtime_repair_applied or runtime_repair_fallback_applied:
                continue

        publication_step = _step_requires_publication_figure_exports(
            step
        ) and not step_record.get("deterministic_standard_analysis")
        # A deterministic data-only auxiliary produces registered tables rather
        # than an inline figure; a separate rendering step owns its export. Names
        # and narrative intent are deliberately absent from the predicate above.
        # A genuine figure method/output contract still fails closed here.
        figure_role = (
            "publication_figure"
            if publication_step
            else "analysis_figure" if _step_expects_figure(step) else None
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
            return step_record

        # Finalise every result-bearing figure before any output is copied into
        # EvidenceStore or any numeric claim is registered.  The staged repair
        # replaces the entire output directory, so running it after registration
        # would leave evidence digests and claims bound to a retired draft.
        step_summary = _load_step_summary_from_outputs(run_result.out_dir)
        if worker_progress.runner_repair_name and is_sealed_renderer_repair(
            worker_progress.runner_repair_name
        ):
            step_summary = _write_host_input_binding_receipts(
                out_dir=run_result.out_dir,
                step_summary=step_summary,
                resolved_input_bindings=resolved_input_bindings,
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
            return step_record

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
        for art in run_result.artefacts:
            if not run_result.outputs_safe_to_collect:
                # Defence in depth if a custom runner supplied an artefact
                # list despite declaring its output mount unsafe.
                continue
            # Do not rely only on deletion/enumeration timing: an isolated
            # writer interrupted during teardown could recreate its private
            # streaming file.  Internal work products are never evidence,
            # even if a runner reports one explicitly.
            if worker_progress.deterministic_standard_executor_used and (
                _is_standard_executor_internal_artifact(art)
            ):
                continue
            step_aliases = services.semantic_aliases_for(step, art)
            generation_mode = worker_progress.generation_mode()
            artifact_evidence_id = step_owned_artifact_evidence_id(
                kind=(
                    "table"
                    if art.suffix.lower() in {".csv", ".tsv", ".parquet", ".feather"}
                    else (
                        "figure"
                        if art.suffix.lower()
                        in {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}
                        else "log"
                    )
                ),
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
            elif art.suffix.lower() in {".csv", ".tsv", ".parquet", ".feather"}:
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
            elif art.suffix.lower() in {
                ".png",
                ".svg",
                ".pdf",
                ".tiff",
                ".tif",
                ".pptx",
            }:
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

        if step_summary_record_id is not None:
            step_record["step_summary_evidence_id"] = step_summary_record_id

        def _register_current_step_numeric_claims() -> None:
            """Stage numeric authority before exposing current result aliases."""

            if (
                not step_summary
                or step_summary_record_id is None
                or standard_executor_terminal_block
            ):
                return
            # Value-level provenance (A-track): every numeric leaf in the
            # step's summary is registered as a NumericClaim so the
            # manuscript binder can reverse-link numbers in prose to the
            # exact field of the exact step output that produced them.
            cap = pipeline._max_numeric_claims_per_step
            evidence.register_step_summary_numerics(
                step_id=step.step_id,
                evidence_id=step_summary_record_id,
                summary=step_summary,
                max_leaves=cap if cap > 0 else None,
            )
            # Phase-1 derived-claim hook (Commit 2). After every leaf
            # is registered, evaluate any ``derived_claims`` the coder
            # declared in step_summary. Sources must resolve to claims
            # that ALREADY exist in the registry, so this runs second.
            # Errors surface as ``derived_claim_error`` findings rather
            # than aborting — a bad formula should not kill the step.
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

        if standard_executor_terminal_block:
            terminal_summary = (
                step_summary if step_summary else standard_executor_terminal_summary
            )
            terminal_finding = ValidationFinding(
                validator="trajectory_stability_executor",
                severity="error",
                message=(
                    "The planner-specified trajectory stability computation failed "
                    "closed; its diagnostic outputs were preserved and no coder, "
                    "fallback method, seed change, or cluster-count change was used."
                ),
                detail={
                    "step_id": step.step_id,
                    "reason": standard_executor_terminal_reason,
                    "executor_errors": (
                        terminal_summary.get("errors")
                        if isinstance(terminal_summary, Mapping)
                        else None
                    ),
                },
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
                f"Trajectory stability failed closed for {step.step_id}.",
                status="error",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                reason=standard_executor_terminal_reason,
            )
            return step_record
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
            attempt_id=attempt_id,
            checkpoint_id=review_checkpoint_id,
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

        interp_generation_mode = "llm"
        final_generation_mode = str(step_record.get("generation_mode") or "")
        if final_generation_mode in {"resumed_code_reuse", "fallback"}:
            mode_label = (
                "resumed agent-generated code"
                if final_generation_mode == "resumed_code_reuse"
                else "deterministic fallback code"
            )
            interpretation = (
                f"Step `{step.step_id}` was executed from {mode_label}. "
                "Review the registered step summary and artefacts for numeric "
                "interpretation; no new LLM interpretation was requested."
            )
            interp_generation_mode = (
                "resumed_code_reuse"
                if final_generation_mode == "resumed_code_reuse"
                else "deterministic_fallback"
            )
        else:
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

    steps_to_run = (
        []
        if (
            trajectory_plan_blocked
            or typed_plan_dag_blocked
            or (
                pipeline._development_sample_size is not None
                and run_input_authority_state.development_sample is None
            )
        )
        else resume_controller.remaining_steps(
            plan=plan,
            executed_step_ids=set(preexecuted_step_ids),
        )
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

    if (
        pipeline._max_concurrent_steps <= 1
        or len(steps_to_run) <= 1
        or pipeline._enable_replanning
        or has_typed_input_dependencies
        or has_primary_cohort_universe_producer
        or requested_stop_after_step_id is not None
    ):

        def _maybe_directed_model_replan(
            *,
            failed_step: AnalysisStep,
            failed_record: Dict[str, Any],
        ) -> Optional[AnalysisPlan]:
            """Fire a forced, directive-carrying replan when a model/estimation
            step self-blocks on a task-viable cohort, else return ``None``.

            This is the active half of the self-inflicted-block fix: the
            post-hoc scorecard only *labels* the self-paralysis, whereas here we
            give the replanner a viability-conditioned override so a populated
            cohort is not silently abandoned with a non-execution stub. Bounded
            by ``_MAX_DIRECTED_MODEL_REPLANS``; conservative — silent on a hard
            crash, an unreadable cohort, or genuinely non-viable data.
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

        def _resolve_run_transition(
            step: AnalysisStep,
            record: Dict[str, Any],
            has_remaining: bool,
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
            if directed_plan is not None:
                return RunTransition.replan(
                    directed_plan,
                    rerun_current_step=True,
                )
            if (
                pipeline._enable_replanning
                and record.get("status") == "ok"
                and _successful_step_requests_replan(record)
                and has_remaining
            ):
                return RunTransition.replan(
                    _maybe_replan(
                        current_plan=plan,
                        reason=step.step_id,
                        probe_summary_payload=probe_summary,
                        completed_records=per_step_records,
                    )
                )
            return RunTransition.continue_run()

        def _apply_revised_plan(
            revised_plan: AnalysisPlan,
            executed_step_ids: set[str],
        ) -> Sequence[AnalysisStep]:
            nonlocal plan, total_steps
            plan = revised_plan
            step_order.clear()
            step_order.update({s.step_id: i for i, s in enumerate(plan.steps)})
            remaining = resume_controller.remaining_steps(
                plan=plan,
                executed_step_ids=executed_step_ids,
            )
            total_steps = len(plan.steps)
            return remaining

        # ``steps_to_run`` already carries the fail-closed plan-preflight
        # decision above. Recomputing from the full plan here would revive
        # every step after a typed-DAG/trajectory contract ERROR and spend
        # Coder calls on a plan the host has declared non-executable.
        run_coordinator.run_sequential(
            state=RunExecutionState(
                remaining_steps=list(steps_to_run),
                executed_step_ids=set(preexecuted_step_ids),
            ),
            execute_step=_execute_one_step,
            resolve_transition=_resolve_run_transition,
            apply_revised_plan=_apply_revised_plan,
        )
    else:

        def _record_parallel_worker_error(exc: BaseException) -> None:
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="step_executor",
                        severity="error",
                        message=f"Worker raised an unhandled exception: {exc!r}",
                    )
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
            probe_summary=probe_summary,
            runtime_state=runtime_state,
            flush_partial_manifest=_flush_partial_manifest,
        )

    if (
        not trajectory_plan_blocked
        and not typed_plan_dag_blocked
        and trajectory_plan_contract_applies(plan=plan, context=context)
    ):
        run_level_trajectory_findings = trajectory_bundle_findings(
            context=context,
            plan=plan,
            per_step_records=per_step_records,
            evidence=evidence,
            run_dir=run_dir,
            cohort_path=cohort_path,
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

    try:
        robustness_specs = robustness_specs_for_execution(run_dir=run_dir, plan=plan)
        if robustness_specs and not list(getattr(plan, "robustness_specs", []) or []):
            findings.append(
                ValidationFinding(
                    validator="robustness_panel",
                    severity="warning",
                    message=(
                        "Recovered robustness_specs from the plan-time lock because "
                        "the active replanned AnalysisPlan no longer carried them."
                    ),
                )
            )
        adapter_rows, adapter_warnings = fit_robustness_rows_from_records(
            specs=robustness_specs,
            per_step_records=per_step_records,
            primary_cohort=getattr(plan, "cohort", None),
            cohort_path=cohort_path,
            context=context,
            run_dir=run_dir,
            allow_implicit_cohort_refit=False,
        )
        for warning in adapter_warnings:
            findings.append(
                ValidationFinding(
                    validator="robustness_estimator",
                    severity="warning",
                    message=warning,
                )
            )
        robustness_panel = build_robustness_panel_from_records(
            specs=robustness_specs,
            per_step_records=per_step_records,
            adapter_rows=adapter_rows,
        )
        write_robustness_panel(
            run_dir=run_dir,
            panel=robustness_panel,
            evidence=evidence,
            prompt_pack_version=prompt_version,
        )
        _flush_partial_manifest(
            {
                "robustness_panel_path": "robustness_panel.json",
                "robustness_n_variants": robustness_panel.n_variants,
                "robustness_range_low": robustness_panel.range_low,
                "robustness_range_high": robustness_panel.range_high,
            }
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="robustness_panel",
                severity="warning",
                message=f"Robustness panel artifact could not be built: {exc}",
            )
        )

    if pipeline._enable_visual_qa:
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
            client = pipeline._vlm_client or role_resolver("analyzer")
            if client is not None:
                vlm_adapter = VLMVisualQAAdapter(client)
        final_visual_findings = VisualQAAuditor(vlm_adapter=vlm_adapter).audit(
            figure_paths=fig_paths
        )
        demoted_final_findings, _ = _demote_cosmetic_visual_findings(
            final_visual_findings
        )
        findings += demoted_final_findings

    try:
        article_contract_status = summarize_article_contract_coverage(
            context=context,
            plan=plan,
            evidence_records=evidence.records(),
            per_step_records=per_step_records,
            run_dir=run_dir,
        )
        article_contract_path = run_dir / "article_contract_audit.json"
        article_contract_path.write_text(
            json.dumps(
                article_contract_audit_payload(article_contract_status),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        if evidence.get("article_contract_audit") is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Run-level article analysis contract audit: compares "
                    "registered artifacts against required article display roles."
                ),
                source_path=article_contract_path,
                evidence_id="article_contract_audit",
                producer="article_contract",
                generation_mode="system",
            )
        findings.extend(
            validate_run_against_article_contract(
                context=context,
                plan=plan,
                evidence_records=evidence.records(),
                per_step_records=per_step_records,
                run_dir=run_dir,
            )
        )
        _flush_partial_manifest(
            {"article_contract_audit": str(article_contract_path.relative_to(run_dir))}
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="article_analysis_contract",
                severity="warning",
                message=(
                    "Run-level article analysis contract audit failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        )

    plan_result.plan = plan
    plan_result.plan_path = plan_path
    return _ExecutePhaseResult(
        plan=plan,
        per_step_records=per_step_records,
        probe_summary=probe_summary,
        runtime_state=runtime_state,
        flush_partial_manifest=_flush_partial_manifest,
    )
