"""End-to-end orchestration of the research-agent loop.

A single class, :class:`ResearchAgentPipeline`, ties together the
context builder, agents, runner, validators and evidence store. The
public ``run()`` method is the only thing most users need.

Loop shape::

    research_question + cohort
        ↓
    build_research_context        → context.json   (registered evidence)
        ↓
    PlannerAgent.run              → plan.json      (registered evidence)
        ↓
    for each AnalysisStep:
        CoderAgent.run            → analysis.py    (registered evidence)
        ConceptUsageAuditor.audit → findings       (errors block step)
        CodeRunner.run            → outputs/, run.log
    StatisticalValidator.audit → findings
    register every artefact   → evidence ids
    AnalyzerAgent.run         → interpretation paragraph
        ↓
    PublicationFigureSkill.run → claim-first figure contract + exports
        ↓
    WriterAgent.run               → manuscript_scaffold.md
    EvidenceStore.bind_manuscript → manuscript_scaffold_bound.md
        ↓
    write manifest.json + results_report.md
"""

from __future__ import annotations

import ast
import asyncio
import csv
import io
import json
import functools
import logging
import math
import os
from copy import deepcopy
import re
import shutil
import textwrap
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

from .agents.core import (
    AnalyzerAgent,
    infer_analysis_type,
    ClinicalSemanticsAgent,
    CoderAgent,
    CriticAgent,
    DataExtractionAgent,
    ManuscriptAgent,
    PlannerAgent,
    PlannerArticleContractError,
    ReplannerAgent,
    RuntimeSupervisor,
    StatisticalAnalysisAgent,
    VisualizationAgent,
)
from .agents.progressive_planner import (
    ProgressivePlannerAgent,
    progressive_cohort_concept_ids,
)
from .architecture import architecture_profile_markdown, default_architecture_profile
from .authority.parent_artifact import (
    _resolve_upstream_manifest_analysis_request,
    _resolve_upstream_manifest_step,
    _verified_direct_parent_artifact_digests,
    _verified_direct_parent_table_names,
)
from .authority.figure_renderer import (
    _ordered_distribution_availability_parent_digest_seal,
    _sealed_renderer_figure_step_matches_parent,
)
from .providers.cost import CostMeter, metered_role_resolver
from .providers.hard_stop import HardStopClient
from .figures.distribution_availability import (
    _distribution_availability_figure_step_matches_parent,
)
from .figures.missingness_publication import (
    render_missingness_publication_bundle_from_prior_outputs as _render_missingness_publication_bundle_from_prior_outputs,
)
from .figures.sealed_registry import sealed_renderer_adapter
from .replication.envelope import (
    ENVELOPE_SCHEMA_VERSION,
    ReproCallRecord,
    ReproEnvelope,
    envelope_role_resolver,
)
from .methods.multiple_testing import build_multiple_testing_report
from .review.causal_audit import run_causal_audit
from .reporting.reporting_checklist import (
    build_strobe_checklist,
    build_tripod_ai_checklist,
    choose_checklist,
)
from .reporting.reviewer import run_reviewer_round
from .authority.provenance import (
    ProvenanceBundle,
    SourceFileRecord,
    build_provenance_bundle,
    hash_sources,
)
from .methods.sensitivity import (
    EValueResult,
    NegativeControlResult,
    compute_e_value,
    run_negative_control_check,
)
from .replication.paper import (
    build_paper_replication_spec,
    build_paper_result_ledger,
    canonical_outcome_name,
    compare_paper_to_easyicu,
    load_paper_source,
    parse_paper_profile,
    postprocess_paper_replication,
    render_deviation_report,
    render_replication_report,
    render_showcase_manuscript,
    write_claim_csv,
    write_fail_closed_paper_package,
)
from .replication.notebook import (
    NotebookStep,
    build_notebook,
    build_requirements_lockfile,
    write_notebook,
)
from .discovery.hypothesis_generator import generate_hypotheses
from .reporting.pdf_render import render_pdf_for_run
from .research_context.builder import (
    build_naive_research_context,
    build_research_context,
    build_retrieved_research_context,
)
from .research_context.typed import parse_research_context_json
from .gates.preplan import preplan_data_failure_reason, preplan_data_findings
from .authority.context_numeric_claims import register_context_numeric_claims
from .authority.declared_levels import bind_step_declared_levels
from .authority.plan_input_closure import close_measurement_companion_inputs
from .authority.table_one_binding import (
    bind_table_one_execution_spec,
    restore_table_one_private_checkpoint,
    write_table_one_private_checkpoint,
)
from .authority.resume_plan import (
    load_compatible_resume_plan as _load_compatible_resume_plan,
)
from .authority import (
    pipeline_cache as _pipeline_cache,
    plan_lifecycle as _plan_lifecycle,
)
from .planning.analysis_blueprint import (
    build_analysis_blueprint,
    render_analysis_blueprint_for_prompt,
    validate_plan_against_analysis_blueprint,
)
from .planning.adjustment_authority import (
    validate_plan_against_adjustment_authority,
)
from .planning.cohort_contract import (
    cohort_concept_id_scope,
    cohort_definition_has_explicit_selection,
)
from .planning.dependence_authority import (
    DependenceAuthorityError,
    bind_context_dependence_authority,
)
from .reporting.article_contract import (
    build_article_analysis_contract,
    validate_plan_against_article_contract,
)
from .planning.figure_strategy import build_article_figure_strategy
from .planning.final_article_design import (
    materialize_final_article_design_authority,
)
from .planning.progressive_artifacts import persist_progressive_planning_authority
from .orchestration import progressive_planning as _progressive_planning
from .planning.scientific_review import (
    render_plan_scientific_guardrails,
)
from .orchestration.config import (
    PipelineConfig,
    assert_step_provider_budget_funds_its_repairs,
)
from .orchestration.instance_lifecycle import (
    PipelineInstanceLifecycleLease,
    pipeline_instance_lifecycle as _pipeline_instance_lifecycle,
)
from .orchestration.human_review_checkpoint import (
    HumanReviewCheckpoint,
    HumanReviewCheckpointError,
    checkpoint_path as human_review_checkpoint_path,
    load_checkpoint as load_human_review_checkpoint,
    write_checkpoint as write_human_review_checkpoint,
)
from .orchestration.human_review_restore import (
    bind_checkpoint_decision_payloads,
    commit_human_review_decision,
    fail_human_review_checkpoint,
    mark_human_review_execution_phase,
    mark_human_review_execution_started,
    persist_human_review_records,
    prepare_human_review_decision,
    restore_durable_human_review_pause,
)
from .orchestration.services import PipelineServices
from .orchestration.progress import (
    ResumableProgressChannel,
    planner_retry_progress_callback,
)
from .orchestration.scientific_runtime import ScientificRuntimeAuthorities
from .orchestration.workflow import PipelineRunOutcome, PlannerDesignCanaryComplete
from .resources.capability_runtime import CapabilityWorkflowRuntime
from .contracts.runtime import (
    RunResult,
    _ExecutePhaseResult,
    _PlanPhaseResult,
    _WritePhaseResult,
)
from .execution.host_services import (
    ExecutePhaseServices,
    PublicationFigureAuthorityServices,
)
from .execution.cohort_routing import PreselectionUniverseOwnerCapability
from .execution.output_files import _clear_output_dir, _has_figure_exports
from .concept_dict_audit import (
    assert_dict_matches as assert_concept_dict_matches,
    verify_recorded_dict_match,
    write_concept_dict_fingerprint,
)
from .cohort.schema import (
    CohortAuthorityError,
    ensure_cohort_definition,
    materialize_locked_analysis_cohort,
    write_locked_cohort_definition,
)
from .intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    MaterializedMetadataError,
    implementation_bundle_sha256 as materialized_implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
    materialized_provenance_path,
    stage_materialized_cohort_authority,
)
from .intake.materialized_trajectory import (
    MaterializedTrajectoryAuthorityRef,
    MaterializedTrajectoryError,
    StagedTrajectoryBinding,
    VerifiedLegacyTrajectoryCapsuleReceipt,
    VerifiedMaterializedTrajectoryAuthority,
    load_verified_materialized_trajectory_authority,
    long_trajectory_is_bound,
    stage_legacy_trajectory_exact,
    stage_materialized_trajectory_authority,
)
from .robustness.panel import (
    ensure_robustness_specs,
    load_locked_robustness_specs,
    robustness_specs_sha,
    write_locked_robustness_specs,
)
from .trajectory.plan_contract import (
    augment_trajectory_plan_products,
    non_trajectory_clustering_stability_guide,
    trajectory_plan_dag_findings,
    trajectory_planner_contract_guide,
    trajectory_step_roles,
)
from .reporting.readiness import (
    execution_gate_status,
    render_report,
    write_readiness_artifacts,
)

# Back-compat aliases. Tests (and any downstream code) that imported the
# leading-underscore names from this module before the readiness/report
# helpers were moved to ``reporting.readiness`` keep working unchanged.
from .reporting.readiness import (
    _publication_figure_bundle_ready,  # noqa: F401
    _compute_readiness_gates,  # noqa: F401
    _count_missing_evidence_markers,  # noqa: F401
    _extract_claim_ledger_rows,  # noqa: F401
    _render_author_review_note,  # noqa: F401
)

_execution_gate_status = execution_gate_status  # noqa: F841 (legacy alias)
_write_readiness_artifacts = write_readiness_artifacts  # noqa: F841 (legacy alias)
_render_report = render_report  # noqa: F841 (legacy alias)
from .audits.manuscript_claims import audit_manuscript_numeric_claims  # noqa: E402
from .audits.manuscript_claims import (  # noqa: E402,F401
    _first_summary_scalar,
    _extract_metric_claims,
    _extract_percent_claims_near,
)

_audit_manuscript_numeric_claims = audit_manuscript_numeric_claims  # noqa: F841 (legacy alias)

from .authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
    _coerce_enforcement_mode,
    sha256_of_file,
)
from .learning.experience import (
    ExperienceBank,
    ExperienceBankCorruptError,
    ExperienceRecord,
    mine_experience_from_run,
)
from .reporting.manuscript_post import (
    bind_numeric_values,
    _demote_unresolved_evidence_placeholders,
    _first_resolvable_name,
    _remove_tbd_sentences,
    _repair_common_writer_citation_omissions,
    _repair_common_writer_placeholders,
)
from .repairs.summary import (
    _extract_last_json_object,
    _salvage_minimal_contract_step_summary,
    _salvage_named_json_step_summary,
    _salvage_stdout_json_step_summary,
)
from .repairs.source import (
    _KEYERROR_NOT_IN_INDEX_RE,
    _NAME_ERROR_HELPER_RE,
    _deterministic_runner_repair,
    _deterministic_summary_repair,
    _extract_missing_index_columns,
    _patch_json_dump_numpy_key_sanitizer,
    _patch_primary_predictor_into_design_matrix,
    _strip_columns_from_list_literals,
)
from .scalar_utils import (
    _coerce_scalar,
    _expected_numeric_annotations_for_step,
    _first_numeric_effect_from_text,
    _first_numeric_scalar_with_key_fragment,
    _first_present_scalar,
    _flatten_scalar_dict,
)
from .replication.report import (
    _build_replication_notes,
    _extract_cross_database_run_summary,
    _literature_provenance_note,
    _render_cross_database_comparison_markdown,
    _render_cross_database_summary_markdown,
    _render_cross_database_validation_report,
)
from .robustness.primary_effect import (
    _extract_primary_effect_row,
    _infer_primary_predictor_from_run_dir,
    _primary_effect_candidate_score,
)
from .reporting.writer_evidence import (
    _preferred_writer_evidence_names,
    _render_writer_evidence_digest,
    _render_writer_evidence_digest_v2,
    _resolve_writer_aux_path,
    _summarise_primary_association_table,
    _summarise_table_one_rows,
)
from .plan_utils import (
    _augment_report_typed_product_inputs,
    _cap_plan_preserving_figure_steps,
    _clustering_contract_applies,
    _cohort_definition_contract_findings,
    endpoint_contract_findings,
    _cohort_definition_is_empty,
    _ensure_publication_figure_step_in_plan,
    _effect_figure_semantics_supported_by_inputs,
    _effect_figure_semantics_supported_by_model_roster,
    _enforce_advanced_plan_contract,
    _plan_expects_analysis_cohort,
    _infer_primary_predictor_from_context,
    _migrate_render_step_contract,
    _parent_step_id_for_figure_step,
    _prediction_contract_applies,
    _predictor_tokens,
    _preserve_figure_steps_after_replan,
    _research_question_implies_figure,
    _render_only_figure_step_intent,
    _split_table_and_figure_outputs_in_plan,
    _step_contract_findings,
    _step_contract_repair_guidance,
    _step_expects_figure,
    _step_produces_figure,
    effect_output_authorized,
)
from .planning import figure_plan_shaping as _figure_plan
from .planning.final_plan_shape import validate_final_plan_shape
from .orchestration.experiment_spec import ExperimentSpec, dump_experiment_spec
from .orchestration import scientific_plan_review_gate as _scientific_plan_gate
from .figures.skill import PublicationFigureSkill
from .figures.prior_output_support import (
    figure_parent_candidate_step_dirs as _figure_parent_candidate_step_dirs,
    publication_label as _publication_label,
    short_figure_label as _short_figure_label,
)
from .reporting.bibtex import render_bibtex
from .reporting.latex import scaffold_to_latex
from .literature import (
    HypothesisBlueprintAgent,
    LiteratureAgent,
    LiteratureBundle,
    manuscript_citable_keys,
    manuscript_citable_records,
    render_hypothesis_blueprint_for_prompt,
)
from .planning.preplan_literature import prepare_preplan_literature
from .planning import literature_design_authority as _literature_design
from .planning.preplan_know_how import (
    PlannerKnowHowBinding,
    prepare_preplan_know_how,
)
from .providers.llm import (
    LLMRouter,
    llm_is_mockish,
    llm_supports_vision,
    resolve_role_client,
)
from .providers.mocks import MockLLMClient
from .providers.prompt_budget import budgeted_role_client
from .providers.protocol import LLMClient, LLMMessage
from .learning.memory import RunMemory
from .learning.runtime import ReviewedMemoryRuntime
from .learning.store import FileSystemMemoryStore
from .providers.prompts import PROMPT_PACK_VERSION, prompt_pack_files
from .execution.runner import (
    HOST_OWNED_RUNNER_ENV_KEYS,
    CodeRunner,
    DockerRunner,
    reject_reserved_runner_env,
    select_safe_runner_kind,
)
from .execution.method_capabilities import (
    runtime_capability_job_scope,
    set_runtime_capability_snapshot_provider,
)
from .concept_availability import normalize_database_name
from .schema import (
    AgentRuntimeState,
    AnalysisManifest,
    AnalysisPlan,
    AnalysisStep,
    CritiqueReport,
    EvidenceRecord,
    EvidenceRef,
    EndpointSpec,
    ManuscriptDraftPacket,
    PaperProfile,
    PaperReplicationSpec,
    PaperResultLedger,
    PipelineResult,
    ResearchContext,
    ReplicationDeviationReport,
    TimeWindow,
    ValidationFinding,
    VariableRole,
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
    PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
    PlannedModelRequirement,
)
from .planning.study_design import (
    build_study_design_brief,
    validate_plan_against_study_design_brief,
)
from .providers.efficiency_budget import wrap_planner_efficiency_budget
from .skills import ClinicalSkill, get_skill, list_skills
from .authority.runtime_artifacts import (
    AuditLogger,
    build_execution_replay,
    build_workflow_graph,
    current_successful_step_records,
    current_step_records,
    load_run_artifact_authority,
    render_workflow_graph_mermaid,
    verified_run_evidence_path,
    write_json_artifact,
)
from .authority.run_input import (
    RUN_INPUT_CAPSULE_EVIDENCE_ID,
    RUN_INPUT_CAPSULE_FILENAME,
    RunInputIdentityError,
    build_environment_identity,
    build_scientific_identity,
    load_verified_run_input_capsule,
    prepare_existing_resume_input,
    seal_run_input_capsule,
    verify_legacy_trajectory_capsule_receipt,
)
from .authority.plan_review import PlanReviewAuthority, ReviewExecutionAuthority
from .canonical_json import canonical_sha256
from .authority.run_lock import (
    acquire_run_execution_lock,
    current_locked_run_id,
    exclusive_run_execution,
)
from .authority.run_heartbeat import (
    bind_active_run_heartbeat,
    run_heartbeat_scope,
)
from .audits.validators import (
    ClinicalConstraintValidator,
    ConceptUsageAuditor,
    LLMConceptAuditor,
    PublicationClaimAuditor,
    ReplicationDesignAuditor,
    ReplicationResultComparator,
    StatisticalGuard,
    StatisticalValidator,
    dedupe_findings,
)
from .gates.figure_egress import FigureEgressPolicy
from .gates.visual_qa import VLMVisualQAAdapter, VisualQAAuditor
from .orchestration.finalize import (
    _concept_dictionary_manifest_fields,  # noqa: F401
    _render_cost_summary,  # noqa: F401
)

# Compatibility seam for callers and tests that patch the historical pipeline
# symbol while the implementation remains owned by figure_plan_shaping.
_ensure_audit_panel_step_in_plan = _figure_plan.ensure_data_quality_figure_step


def _one_capability_job(method: Callable[..., Any]) -> Callable[..., Any]:
    """Bind a public entry point to one runtime-capability publication scope.

    Applied at the entry point rather than around each
    ``set_runtime_capability_snapshot_provider`` call because publication has
    to outlive the runner constructor that performs it — the coder prompt is
    rendered later. The boundary that matters is the job: whatever a job
    publishes is visible for the whole job and gone once the job returns,
    including when it returns a ``HumanReviewPending`` or raises.

    A decorator keeps this a three-line change to a ~900-line method; the
    alternative would be re-indenting the whole body into a ``with`` block.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with runtime_capability_job_scope():
            return method(self, *args, **kwargs)

    return wrapper


def _defer_typed_plan_dag_findings_until_probe(
    candidate_findings: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Mark pre-probe typed-DAG errors as pending, never as current errors.

    Plan shaping happens before the probe-aware replanner gets its one focused
    chance to repair missing or ambiguous typed edges. Retaining those initial
    errors after a successful replan makes a repaired plan look blocked. Only
    ``plan_typed_dag`` errors are deferred; cap failures and every unrelated
    planner error keep their original severity. The execute phase recomputes
    the final typed DAG and emits current errors before any Coder call.
    """

    deferred: List[ValidationFinding] = []
    for finding in candidate_findings:
        if finding.validator != "plan_typed_dag" or finding.severity != "error":
            deferred.append(finding)
            continue
        deferred.append(
            finding.model_copy(
                update={
                    "validator": "plan_contract_pending",
                    "severity": "warning",
                    "detail": {
                        **dict(finding.detail or {}),
                        "pending_probe_replan": True,
                        "original_validator": "plan_typed_dag",
                    },
                }
            )
        )
    return deferred


from .orchestration.resume_plan_migration import (  # noqa: F401 — owner module
    LegacyResumePlanMigrationError,
    _is_closed_adjusted_association_step,
    _legacy_resume_model_roster_targets,
    _LegacyModelRosterPacket,
    _LegacyModelRosterStepPacket,
    _migrate_legacy_resume_figure_render_edges,
    _migrate_legacy_resume_model_requirements,
    _migrate_resume_scientific_runtime_binding,
    _migrate_resume_trajectory_products,
    _next_analysis_plan_revision,
    _normalise_plan_contract_token,
    _parse_legacy_model_roster_packet,
    _project_legacy_model_roster_packet,
    _restore_resume_plan_robustness_lock,
    _resume_completed_records_for_plan_migration,
)


def _load_resume_state(run_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        loaded = load_run_artifact_authority(run_dir)
    except Exception as exc:
        raise ValueError(
            f"Cannot resume from corrupt checkpoint authority: {run_dir}"
        ) from exc
    if loaded is not None:
        return dict(loaded)
    # ``None`` is the explicit legacy signal: no checkpoint in this run has a
    # per-step ledger. Preserve read-only legacy adoption without weakening the
    # modern monotonic selector above.
    legacy_candidates = [
        path
        for path in (run_dir / "manifest_partial.json", run_dir / "manifest.json")
        if path.is_file()
    ]
    if not legacy_candidates:
        return None
    checkpoint = max(legacy_candidates, key=lambda path: path.stat().st_mtime_ns)
    try:
        legacy = json.loads(checkpoint.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Cannot resume from corrupt checkpoint: {checkpoint}"
        ) from exc
    if not isinstance(legacy, dict):
        raise ValueError(f"Cannot resume from non-object checkpoint: {checkpoint}")
    return legacy


@dataclass(frozen=True)
class _PlanGenerationResult:
    """Immutable handoff between plan generation and host validation."""

    plan: AnalysisPlan
    reused_prior_plan: bool
    reused_plan_path: Optional[Path]
    migrated_plan_path: Optional[Path]
    plan_generation_mode: str
    used_mock_llm: bool
    planner_prompt_metrics: Optional[Dict[str, Any]]
    proposed_plan: AnalysisPlan


def _resume_compatible_plan(
    *,
    run_dir: Path,
    resume_state: Any,
    context: Any,
    agent_context: Any,
    evidence: Any,
    know_how_binding: Any,
    enable_know_how: Any,
    findings: Any,
) -> Tuple[Any, Optional[Path], Optional[AnalysisPlan], bool, str]:
    # Resume: reuse the locked plan from the prior run instead of re-planning.
    # (see _generate_or_resume_plan for the full rationale)
    reused_prior_plan = False
    reused_plan_path: Optional[Path] = None
    proposed_plan: Optional[AnalysisPlan] = None
    if resume_state is not None:
        plan, _prior_plan_path = _load_compatible_resume_plan(
            run_dir=run_dir,
            resume_state=resume_state,
            context=context,
            evidence=evidence,
            prompt_pack_version=PROMPT_PACK_VERSION,
        )
        if plan is not None and plan.steps:
            proposed_plan = plan.model_copy(deep=True)
            restore_table_one_private_checkpoint(
                run_dir=run_dir,
                plan=plan,
                context=agent_context,
            )
            # The resumed plan is the one on disk, so it still carries the
            # host's opaque placeholders; a resume that skipped this would
            # execute a different declaration than the first attempt did.
            for resumed_step in plan.steps:
                bind_step_declared_levels(resumed_step, agent_context)
            know_how_binding.verify_resume(
                plan.know_how_decisions,
                enabled=enable_know_how,
            )
            reused_prior_plan = True
            reused_plan_path = _prior_plan_path
            plan_generation_mode = "resumed"
            findings.append(
                ValidationFinding(
                    validator="planner",
                    severity="warning",
                    message=(
                        "Resuming prior run: reused the latest compatible "
                        "saved analysis plan (skipped re-planning) so "
                        "completed step_ids stay aligned and execution "
                        "continues from the failed step."
                    ),
                    detail={
                        "generation_mode": "resumed",
                        "n_steps": len(plan.steps),
                        "plan_path": (
                            str(_prior_plan_path.relative_to(run_dir))
                            if _prior_plan_path is not None
                            and _prior_plan_path.is_relative_to(run_dir)
                            else str(_prior_plan_path)
                        ),
                    },
                )
            )
        else:
            raise LegacyResumePlanMigrationError(
                "resume checkpoint has no digest-verified analysis plan "
                "evidence compatible with every completed step"
            )
    return (
        plan,
        reused_plan_path,
        proposed_plan,
        reused_prior_plan,
        plan_generation_mode,
    )


def _apply_resume_plan_migrations(
    *,
    plan: AnalysisPlan,
    agent_context: Any,
    run_dir: Path,
    resume_state: Optional[Dict[str, Any]],
    resume_from_step_id: Optional[str],
    role_resolver: Callable[[str], Any],
    evidence: Any,
    prompt_version: str,
    llm_signature: str,
    max_prompt_tokens: Optional[int],
    submission_profile_name: Optional[str],
    plan_generation_mode: str,
    migrated_plan_path: Optional[Path],
    findings: List[ValidationFinding],
    scientific_runtime_authorities: Any,
) -> Tuple[AnalysisPlan, Optional[Path], str]:
    """Apply every resume-time migration a reused plan needs, in order.

    A resumed plan may predate the typed model roster, the immutable
    robustness lock, the canonical trajectory replay-product schema, or exact
    typed parent edges on framework-split rendering steps. Each migration
    below rewrites the plan, records its own revision path, and appends the
    finding that makes the rewrite auditable; the last one to fire owns
    ``migrated_plan_path``.

    The migration implementations live in
    ``orchestration/resume_plan_migration.py``. This coordinator stays in the
    host module so the resume tests keep their module-global monkeypatch seams
    on ``easyicu.research_agent.pipeline``.
    """

    from .orchestration.profiles import is_paper_facing_profile
    from .orchestration.step_selector import resolve_resume_from_step_selector

    resume_from_step_id = resolve_resume_from_step_selector(
        plan,
        resume_from_step_id,
    )

    plan, migrated_plan_path, migrated_step_ids = (
        _migrate_legacy_resume_model_requirements(
            plan=plan,
            context=agent_context,
            run_dir=run_dir,
            resume_state=resume_state,
            resume_from_step_id=resume_from_step_id,
            role_resolver=role_resolver,
            evidence=evidence,
            prompt_version=prompt_version,
            llm_signature=llm_signature,
            max_prompt_tokens=max_prompt_tokens,
            allow_scientific_migration=not is_paper_facing_profile(
                submission_profile_name
            ),
        )
    )
    if migrated_plan_path is not None:
        plan_generation_mode = "resume_with_scientific_migration"
        findings.append(
            ValidationFinding(
                validator="planner_schema_migration",
                severity="error",
                message=(
                    "A new Planner decision migrated legacy remaining "
                    "adjusted-association step(s) to a typed model roster; "
                    "the revised plan requires fresh human approval."
                ),
                detail={
                    "reason": "resume_scientific_migration_requires_review",
                    "human_review_required": True,
                    "approval_allowed": True,
                    "generation_mode": "resume_with_scientific_migration",
                    "target_step_ids": list(migrated_step_ids),
                    "plan_path": str(migrated_plan_path.relative_to(run_dir)),
                },
            )
        )
    plan, lock_restore_path = _restore_resume_plan_robustness_lock(
        plan=plan,
        run_dir=run_dir,
        evidence=evidence,
        prompt_version=prompt_version,
        llm_signature=llm_signature,
    )
    if lock_restore_path is not None:
        migrated_plan_path = lock_restore_path
        if plan_generation_mode != "resume_with_scientific_migration":
            plan_generation_mode = "resume_with_authority_restore"
        findings.append(
            ValidationFinding(
                validator="robustness_spec_lock",
                severity="warning",
                message=(
                    "Resume restored the verified plan-time robustness "
                    "specifications after an older replan drifted from "
                    "the immutable lock."
                ),
                detail={
                    "plan_path": str(lock_restore_path.relative_to(run_dir)),
                    "lock_path": "robustness_specs_locked.json",
                },
            )
        )
    plan, trajectory_migration_path, trajectory_migration_findings = (
        _migrate_resume_trajectory_products(
            plan=plan,
            context=agent_context,
            run_dir=run_dir,
            evidence=evidence,
            prompt_version=prompt_version,
            llm_signature=llm_signature,
        )
    )
    findings.extend(trajectory_migration_findings)
    if trajectory_migration_path is not None:
        migrated_plan_path = trajectory_migration_path
        if plan_generation_mode == "resumed":
            plan_generation_mode = "resume_with_schema_migration"
        findings.append(
            ValidationFinding(
                validator="plan_contract",
                severity="warning",
                message=(
                    "Resume migrated an older trajectory plan to the "
                    "canonical replay-product schema without changing "
                    "scientific ownership or step identities."
                ),
                detail={
                    "kind": "resume_trajectory_schema_migration",
                    "plan_path": str(trajectory_migration_path.relative_to(run_dir)),
                },
            )
        )
    (
        plan,
        figure_edge_migration_path,
        figure_edge_step_ids,
    ) = _migrate_legacy_resume_figure_render_edges(
        plan=plan,
        run_dir=run_dir,
        resume_state=resume_state,
        resume_from_step_id=resume_from_step_id,
        evidence=evidence,
        prompt_version=prompt_version,
        llm_signature=llm_signature,
    )
    if figure_edge_migration_path is not None:
        migrated_plan_path = figure_edge_migration_path
        if plan_generation_mode == "resumed":
            plan_generation_mode = "resume_with_schema_migration"
        findings.append(
            ValidationFinding(
                validator="planner_schema_migration",
                severity="warning",
                message=(
                    "Resume restored exact typed parent edges on "
                    "legacy framework-split rendering steps."
                ),
                detail={
                    "kind": "legacy_figure_render_edge",
                    "target_step_ids": list(figure_edge_step_ids),
                    "plan_path": str(figure_edge_migration_path.relative_to(run_dir)),
                },
            )
        )
    (
        plan,
        scientific_runtime_migration_path,
        scientific_runtime_step_ids,
        scientific_runtime_findings,
    ) = _migrate_resume_scientific_runtime_binding(
        plan=plan,
        resume_state=resume_state,
        resume_from_step_id=resume_from_step_id,
        scientific_runtime_authorities=scientific_runtime_authorities,
        run_dir=run_dir,
        evidence=evidence,
        prompt_version=prompt_version,
        llm_signature=llm_signature,
    )
    findings.extend(scientific_runtime_findings)
    if scientific_runtime_migration_path is not None:
        migrated_plan_path = scientific_runtime_migration_path
        if plan_generation_mode == "resumed":
            plan_generation_mode = "resume_with_authority_restore"
        findings.append(
            ValidationFinding(
                validator="scientific_runtime_plan_compiler",
                severity="warning",
                message=(
                    "Resume recompiled the signed scientific runtime binding "
                    "only on steps selected for replay."
                ),
                detail={
                    "reason_code": "resume_scientific_runtime_binding_restored",
                    "target_step_ids": list(scientific_runtime_step_ids),
                    "plan_path": str(
                        scientific_runtime_migration_path.relative_to(run_dir)
                    ),
                },
            )
        )
    return plan, migrated_plan_path, plan_generation_mode


def _run_preplan_literature_and_hypothesis(
    self: "ResearchAgentPipeline",
    *,
    skill_obj: Any,
    emit_progress: Any,
    run_id: str,
    agent_context: Any,
    run_dir: Path,
    evidence: Any,
    findings: Any,
    context: Any,
    context_path: Path,
    llm: Any,
    resume_state: Any,
) -> Tuple[Optional[Any], Any, list[str], list[str], Any]:
    _scientific_plan_gate.require_strict_planner_route(
        self._config.require_literature_design_authority, skill_obj
    )
    allowed_literature_citation_keys: list[str] = []
    direct_comparator_literature_keys: list[str] = []
    preplan_literature: Optional[LiteratureBundle] = None
    abort_context = _scientific_plan_gate.PreplanAbortContext(
        run_id,
        run_dir,
        context,
        context_path,
        agent_context,
        evidence,
        findings,
        llm,
        resume_state,
    )

    if self._enable_literature and skill_obj is None:
        try:
            emit_progress(
                "hypothesis",
                "Building pre-plan literature and hypothesis blueprint.",
                run_id=run_id,
            )
            preplan_literature = prepare_preplan_literature(
                context=agent_context,
                run_dir=run_dir,
                evidence=evidence,
                enable_pubmed=self._enable_pubmed,
                pubmed_email=self._pubmed_email,
                pubmed_api_key=self._pubmed_api_key,
                enable_tavily=self._enable_tavily,
                tavily_api_key=self._tavily_api_key,
                tavily_retmax=self._tavily_retmax,
                tavily_include_domains=self._tavily_include_domains,
                bound_seed=self._bound_preplan_literature,
                reuse_bound_seed_exact=(
                    self._development_resume_reuse_bound_literature
                ),
            )
            if self._config.require_literature_design_authority:
                _literature_design.validate_preplan_literature_design_authority(
                    preplan_literature
                )
            allowed_literature_citation_keys = list(
                manuscript_citable_keys(preplan_literature)
            )
            direct_comparator_literature_keys = [
                decision.citation_key
                for decision in preplan_literature.screening_decisions
                if decision.disposition == "include"
                and decision.evidence_role == "direct_comparator"
            ]
            # O17 — Front-door hypothesis generation. Opt-in; writes
            # ``hypothesis_candidates.json`` + ``.md`` so the paper
            # Methods section can quote "Out of N candidates we
            # preregistered Q" rather than "we picked Q".
            if self._enable_hypothesis_generator:
                try:
                    hg_result = generate_hypotheses(
                        context=agent_context,
                        citations=list(manuscript_citable_records(preplan_literature)),
                        top_k=self._hypothesis_generator_top_k,
                    )
                    hg_json = run_dir / "hypothesis_candidates.json"
                    hg_md = run_dir / "hypothesis_candidates.md"
                    hg_json.write_text(
                        json.dumps(hg_result.to_json(), indent=2, default=str),
                        encoding="utf-8",
                    )
                    hg_md.write_text(hg_result.to_markdown(), encoding="utf-8")
                    if evidence.get("hypothesis_candidates") is None:
                        evidence.register_file(
                            kind="log",
                            description=(
                                "Ranked front-door hypothesis candidates "
                                "(predictor × outcome) with coverage, "
                                "literature-saturation and ICU-gate scores (O17)."
                            ),
                            source_path=hg_json,
                            evidence_id="hypothesis_candidates",
                            producer="hypothesis_generator",
                            generation_mode="deterministic_skill",
                        )
                    if evidence.get("hypothesis_candidates_summary") is None:
                        evidence.register_file(
                            kind="log",
                            description=(
                                "Human-readable hypothesis-candidate table (O17)."
                            ),
                            source_path=hg_md,
                            evidence_id="hypothesis_candidates_summary",
                            producer="hypothesis_generator",
                            generation_mode="deterministic_skill",
                        )
                    findings.append(
                        ValidationFinding(
                            validator="hypothesis_generator",
                            severity="info",
                            message=(
                                f"Ranked {len(hg_result.candidates)} candidate "
                                f"hypotheses; top={hg_result.candidates[0].question if hg_result.candidates else 'n/a'}"
                            ),
                            evidence_ids=["hypothesis_candidates"],
                        )
                    )
                except Exception as exc:
                    findings.append(
                        ValidationFinding(
                            validator="hypothesis_generator",
                            severity="warning",
                            message=(
                                f"Hypothesis generator failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
                    )
            blueprint = HypothesisBlueprintAgent().run(
                context=agent_context,
                literature=preplan_literature,
            )
            blueprint_path = run_dir / "hypothesis_blueprint.json"
            blueprint_path.write_text(
                blueprint.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if evidence.get("hypothesis_blueprint") is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        "Literature-aware hypothesis, feasibility critique, "
                        "and recommended plan skeleton fed to the planner."
                    ),
                    source_path=blueprint_path,
                    evidence_id="hypothesis_blueprint",
                    producer="hypothesis_blueprint",
                    generation_mode="deterministic_skill",
                )
            if blueprint.feasibility_status == "blocked":
                findings.append(
                    ValidationFinding(
                        validator="hypothesis_blueprint",
                        severity="error",
                        message=(
                            "Hypothesis blueprint marked this request as blocked; "
                            "pipeline stopped before planning executable analysis steps."
                        ),
                        evidence_ids=["hypothesis_blueprint"],
                    )
                )
                emit_progress(
                    "hypothesis",
                    "Hypothesis blueprint is blocked; aborting before planner execution.",
                    status="error",
                    run_id=run_id,
                )
                return (
                    abort_context.finish(self, reason="hypothesis_blueprint_blocked"),
                    agent_context,
                    allowed_literature_citation_keys,
                    direct_comparator_literature_keys,
                    preplan_literature,
                )
            note = render_hypothesis_blueprint_for_prompt(
                blueprint,
                literature=preplan_literature,
            )
            agent_notes = (
                f"{agent_context.notes}\n\n{note}" if agent_context.notes else note
            )
            agent_context = agent_context.model_copy(update={"notes": agent_notes})
        except _literature_design.LiteratureDesignAuthorityError as exc:
            _scientific_plan_gate.record_literature_authority_abort(
                findings, emit_progress, run_id, exc
            )
            return (
                abort_context.finish(self, reason=exc.reason_code),
                agent_context,
                allowed_literature_citation_keys,
                direct_comparator_literature_keys,
                preplan_literature,
            )
        except Exception as exc:
            _scientific_plan_gate.fail_if_strict_prompt_compilation_failed(
                self._config.require_literature_design_authority, exc
            )
            findings.append(
                ValidationFinding(
                    validator="hypothesis_blueprint",
                    severity="warning",
                    message=(
                        "Hypothesis blueprint failed; planner will use "
                        f"context only: {exc}"
                    ),
                )
            )
    return (
        None,
        agent_context,
        allowed_literature_citation_keys,
        direct_comparator_literature_keys,
        preplan_literature,
    )


class ResearchAgentPipeline:
    """One-shot orchestration. Construct, call :meth:`run`, read the result."""

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        services: Optional[PipelineServices] = None,
    ) -> "ResearchAgentPipeline":
        """Construct from the canonical configuration and service objects."""
        return cls(config=config, services=services)

    def __init__(
        self,
        *,
        config: Optional[PipelineConfig] = None,
        services: Optional[PipelineServices] = None,
        **legacy_options: Any,
    ) -> None:
        """Construct a pipeline from declarative settings and live services.

        The flat keyword form is retained as an EasyICU 1.x compatibility
        adapter. New code should pass config=PipelineConfig(...) and, when
        needed, services=PipelineServices(...).
        """
        if config is not None and legacy_options:
            names = ", ".join(sorted(legacy_options))
            raise TypeError(
                "config= is the complete declarative source; do not combine it "
                f"with legacy option(s): {names}"
            )
        if config is None:
            warnings.warn(
                "Flat ResearchAgentPipeline keyword construction is deprecated; "
                "pass config=PipelineConfig(...) and services=PipelineServices(...).",
                DeprecationWarning,
                stacklevel=2,
            )
            services, config_options = PipelineServices.split_legacy_kwargs(
                legacy_options,
                services=services,
            )
            config = PipelineConfig.from_kwargs(**config_options)
        else:
            services = services or PipelineServices()

        self._config = config
        self._services = services
        # Lazy import to avoid pulling the plugin registry module if no
        # plugins are configured (the default).
        from .fallback import CasePluginRegistry as _CasePluginRegistry

        self._case_plugin_registry = (
            services.case_plugin_registry or _CasePluginRegistry()
        )
        self.workdir = Path(config.workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._llm = services.llm
        self._provider_hard_stop = services.provider_hard_stop
        hard_stop_config = {
            "max_provider_attempts_per_run": config.max_provider_attempts_per_run,
            "max_provider_attempts_per_batch": config.max_provider_attempts_per_batch,
            "max_total_tokens_per_run": config.max_total_tokens_per_run,
            "max_total_tokens_per_batch": config.max_total_tokens_per_batch,
            "max_estimated_cost_usd_per_batch": (
                config.max_estimated_cost_usd_per_batch
            ),
            "max_wall_clock_seconds_per_task": (config.max_wall_clock_seconds_per_task),
            "input_cost_usd_per_million_tokens": (
                config.provider_input_cost_usd_per_million_tokens
            ),
            "output_cost_usd_per_million_tokens": (
                config.provider_output_cost_usd_per_million_tokens
            ),
        }
        hard_stop_configured = any(
            value is not None for value in hard_stop_config.values()
        )
        if hard_stop_configured != (self._provider_hard_stop is not None):
            raise ValueError(
                "Provider hard-stop limits and live enforcement service must "
                "be supplied together"
            )
        if self._provider_hard_stop is not None:
            from .authority.provider_hard_stop import ProviderHardStopLimits

            expected_hard_stop = ProviderHardStopLimits(
                **hard_stop_config  # type: ignore[arg-type]
            )
            if self._provider_hard_stop.ledger.limits != expected_hard_stop:
                raise ValueError(
                    "Provider hard-stop service limits differ from PipelineConfig"
                )
        self._timeout_seconds = config.timeout_seconds
        self._standard_executor_timeout_seconds = (
            config.standard_executor_timeout_seconds
        )
        self._python_executable = config.python_executable
        self._enable_literature = config.enable_literature
        self._enable_visual_qa = config.enable_visual_qa
        self._enable_publication_figure_skill = bool(
            config.enable_publication_figure_skill
        )
        self._enable_nature_writing_skill = bool(config.enable_nature_writing_skill)
        from .user_extensions import compile_user_extension_activation

        self._user_extension_activation = compile_user_extension_activation(
            config.extension_activation
        )
        self._vlm_client = services.vlm_client
        if (
            self._provider_hard_stop is not None
            and self._vlm_client is not None
            and not isinstance(self._vlm_client, HardStopClient)
        ):
            # An explicitly injected VLM bypasses the normal analyzer role
            # resolver. Keep its image and text transports under the same
            # task/batch stop-loss as every other Provider-backed role.
            self._vlm_client = HardStopClient(
                self._vlm_client,
                role="visual_qa",
                task=self._provider_hard_stop,
            )
        self._visual_qa_adapter = services.visual_qa_adapter
        # Deny-by-default: a rendered figure is not covered by the text
        # outbound projection, so uploading its bytes to an external VLM is a
        # separate decision from authorizing the provider.
        self._allow_external_figure_upload = bool(config.allow_external_figure_upload)
        self._llm_concept_auditor_client = services.llm_concept_auditor_client
        if (
            self._provider_hard_stop is not None
            and self._llm_concept_auditor_client is not None
            and not isinstance(self._llm_concept_auditor_client, HardStopClient)
        ):
            # An explicitly injected concept auditor bypasses the normal role
            # resolver in execution/phase.py. Wrap it here so a paid audit
            # cannot escape the same durable run/batch stop-loss.
            self._llm_concept_auditor_client = HardStopClient(
                self._llm_concept_auditor_client,
                role="concept_auditor",
                task=self._provider_hard_stop,
            )
        if config.enable_vlm_visual_qa is None:
            self._enable_vlm_visual_qa = bool(
                services.visual_qa_adapter is not None
                or llm_supports_vision(self._vlm_client)
                or llm_supports_vision(services.llm)
            )
        else:
            self._enable_vlm_visual_qa = bool(config.enable_vlm_visual_qa)
        if config.enable_llm_concept_audit is None:
            concept_client = services.llm_concept_auditor_client or services.llm
            self._enable_llm_concept_audit = bool(
                concept_client is not None and not llm_is_mockish(concept_client)
            )
        else:
            self._enable_llm_concept_audit = bool(config.enable_llm_concept_audit)
        self._enable_memory = config.enable_memory
        self._enable_latex = config.enable_latex
        self._latex_venue_template = config.latex_venue_template or "article"
        self._latex_draft_watermark = bool(config.latex_draft_watermark)
        lang = (config.manuscript_language or "en").lower()
        self._manuscript_language = (
            "zh" if lang.startswith(("zh", "cn", "chinese")) else "en"
        )
        # Stored as the canonical Enum so downstream code can compare
        # against EvidenceEnforcementMode.STRICT without string casing.
        self._evidence_enforcement_mode = _coerce_enforcement_mode(
            config.evidence_enforcement_mode
        )
        # T1.4 — when set, the pipeline strips the ICU rules out of the
        # context that drives planning, coding and validation. This is the
        # historical *untyped* naive ablation: a generic data agent sees only
        # column names + dtypes + ANY-aggregation. Typed export authority is
        # deliberately rejected at run entry because exposing its V2 physical
        # facts would contaminate that ablation, while sealing it as V1 would
        # discard the authority contract.
        self._disable_icu_context = bool(config.disable_icu_context)
        self._context_top_k = (
            int(config.context_top_k) if config.context_top_k else None
        )
        self._max_code_repair_attempts = max(0, int(config.max_code_repair_attempts))
        self._max_step_llm_repair_attempts = max(
            0, int(config.max_step_llm_repair_attempts)
        )
        self._max_step_provider_calls = max(0, int(config.max_step_provider_calls))
        self._max_prompt_tokens_per_call = max(
            1, int(config.max_prompt_tokens_per_call)
        )
        # Re-check after resolving the optional service-dependent concept-audit
        # flag. PipelineConfig cannot know whether an injected client makes an
        # ``enable_llm_concept_audit=None`` decision true, so only this layer
        # can count the reserved audit call exactly.
        assert_step_provider_budget_funds_its_repairs(
            max_step_provider_calls=self._max_step_provider_calls,
            max_code_repair_attempts=self._max_code_repair_attempts,
            max_step_llm_repair_attempts=self._max_step_llm_repair_attempts,
            llm_concept_audit_enabled=bool(self._enable_llm_concept_audit),
            allow_underfunded=bool(config.allow_underfunded_step_provider_calls),
        )
        self._enable_deterministic_code_fallback = bool(
            config.enable_deterministic_code_fallback
        )
        self._enable_deterministic_planner_fallback = bool(
            config.enable_deterministic_planner_fallback
        )
        self._planner_strategy = config.planner_strategy
        self._enable_deterministic_runner_repair = bool(
            config.enable_deterministic_runner_repair
        )
        # T2.2 — opt-in PubMed live search. Off by default so CI and
        # the offline demo stay deterministic; the LiteratureAgent
        # handles network failure gracefully (empty list → curated
        # registry only).
        self._enable_pubmed = bool(config.enable_pubmed)
        self._pubmed_email = config.pubmed_email
        self._pubmed_api_key = config.pubmed_api_key
        self._bound_preplan_literature = (
            LiteratureBundle.model_validate(config.bound_preplan_literature)
            if config.bound_preplan_literature is not None
            else None
        )
        self._development_resume_reuse_bound_literature = bool(
            config.development_progressive_resume_reuse_bound_literature
        )
        self._bound_plan_revision_contract = str(
            config.bound_plan_revision_contract or ""
        ).strip()
        # O5 — opt-in Tavily web search for preprints/guidelines/trial
        # registries that PubMed may not index. Off by default so CI
        # and offline demos remain deterministic.
        self._enable_tavily = bool(config.enable_tavily)
        self._tavily_api_key = config.tavily_api_key
        self._tavily_retmax = int(config.tavily_retmax)
        self._tavily_include_domains = list(config.tavily_include_domains or [])
        self._tavily_exclude_domains = list(config.tavily_exclude_domains or [])
        # T3.5 — cohort cache. When enabled, identical re-runs (same
        # cohort hash + same skill/question/llm signature) short-circuit
        # to the prior run_dir's PipelineResult instead of repeating
        # the entire pipeline. Off by default so production users opt
        # in deliberately and tests that *want* full pipeline
        # execution every time keep their existing semantics.
        self._enable_cache = bool(config.enable_cache)
        self._cache_dir = (
            Path(config.cache_dir).resolve()
            if config.cache_dir is not None
            else self.workdir / ".cache"
        )
        self._cache = _pipeline_cache.PipelineCache(self._cache_dir)
        # T3.2 — opt-in LLM cost tracking. When enabled, every per-role
        # client is wrapped in a ``MeteredClient`` that records token
        # counts (prompt / completion) and an estimated USD cost into a
        # ``CostMeter``. The records are persisted to
        # ``manifest.cost_records`` plus ``cost_summary.md`` and
        # ``cost_records.json`` artefacts. Off by default so the
        # default pipeline behaviour stays bit-identical.
        self._enable_cost_tracking = bool(config.enable_cost_tracking)
        self._cost_price_table = config.cost_price_table
        # O20 — Reproducibility envelope. Records prompt/response
        # sha256, requested seed, temperature, provider/model, and a
        # PHI-safe environment snapshot for every LLM call the pipeline
        # makes. Off by default so the base pipeline behaviour stays
        # bit-identical; turn on when a paper reviewer asks for an
        # auditable replay bundle.
        self._enable_reproducibility_envelope = bool(
            config.enable_reproducibility_envelope
        )
        self._llm_seed = int(config.llm_seed) if config.llm_seed is not None else None
        # The seed reaches a provider request through exactly one path: the
        # envelope's recording client, which forwards it when the inner client
        # accepts it. With the envelope off there is no such path, yet the
        # execution identity stamps ``llm_seed`` unconditionally -- so a run
        # was frozen claiming a seed no request ever carried. Submission
        # profiles turn the envelope on, so the paper path was honest and only
        # development runs recorded the false claim; that is the harder case to
        # notice, not the safer one. Refuse instead of stamping a promise the
        # transport cannot keep.
        if self._llm_seed is not None and not self._enable_reproducibility_envelope:
            raise ValueError(
                "llm_seed is set but enable_reproducibility_envelope is False: "
                "the seed would be recorded in the execution identity without "
                "being sent to any provider request. Enable the envelope, or "
                "leave llm_seed unset."
            )
        self._envelope_include_previews = bool(config.envelope_include_previews)
        self._submission_profile_name = config.submission_profile_name
        self._submission_profile_version = config.submission_profile_version
        self._submission_profile_ref = (
            f"{config.submission_profile_name}/{config.submission_profile_version}"
        )
        self._submission_profile_locked_at = config.submission_profile_locked_at
        self._expected_concept_dict_sha = config.expected_concept_dict_sha
        self._expected_sofa2_dict_sha = config.expected_sofa2_dict_sha
        # O22 — family-aware multiple-testing correction. Defaults to ON
        # because reviewers of a research-agent paper will always ask
        # for it; the correction is cheap to compute and does not
        # rewrite existing artefacts.
        self._enable_multiple_testing_correction = bool(
            config.enable_multiple_testing_correction
        )
        self._multiple_testing_alpha = float(config.multiple_testing_alpha)
        # O18 — Causal audit. Deterministic: labels every primary
        # effect as associational vs causal, scans bound manuscript
        # for causal language over associational effects, emits
        # warnings/errors. Default ON because the cost of a paper
        # that silently sells an OR as causal is high.
        self._enable_causal_audit = bool(config.enable_causal_audit)
        # O16 — Reporting-guideline checklist. STROBE always; TRIPOD+AI
        # when the analysis looks like a prediction model. Default ON
        # because ICU journals routinely require the checklist.
        self._enable_reporting_checklist = bool(config.enable_reporting_checklist)
        self._reporting_checklist_names = (
            tuple(config.reporting_checklist_names)
            if config.reporting_checklist_names
            else None
        )
        # Authoritative benchmark task kind (e.g. "subphenotype_clustering"),
        # used to decide kind-specific reporting-checklist applicability rather
        # than relying on fragile manuscript wording. Optional outside the bench.
        self._benchmark_task_kind = config.task_kind
        self._required_primary_cohort_selection_mode = (
            config.required_primary_cohort_selection_mode
        )
        self._scientific_runtime_authorities = ScientificRuntimeAuthorities.load(
            trajectory=config.trajectory_scientific_runtime_authority,
            current_case=config.current_case_scientific_runtime_authority,
        )
        self._scientific_runtime_projection_sha256 = (
            config.scientific_runtime_projection_sha256
        )
        # O15 — Three-role reviewer round (statistician / clinician /
        # methodologist) driven off already-computed evidence and
        # findings. Deterministic; no extra LLM calls. Default ON.
        self._enable_reviewer_round = bool(config.enable_reviewer_round)
        # O24 — Fairness / subgroup analysis for the primary effect.
        # Deterministic (pure numpy); runs after E-values.
        self._enable_fairness_subgroups = bool(config.enable_fairness_subgroups)
        # O17 — Front-door hypothesis generator. Runs early (plan phase)
        # to emit a ranked candidate list; does not change the
        # downstream plan unless the user reassigns ``question``.
        self._enable_hypothesis_generator = bool(config.enable_hypothesis_generator)
        self._hypothesis_generator_top_k = int(config.hypothesis_generator_top_k)
        # PDF rendering (TexLive optional). Off by default because not
        # every CI environment has a LaTeX install; turn on when the
        # user actually wants ``manuscript_scaffold.pdf`` next to the
        # ``.tex`` and ``.bib``.
        self._enable_pdf_render = bool(config.enable_pdf_render)
        # T3.3 — concurrent step execution. The canonical AnalysisPlan
        # has independent steps (each step reads the cohort and writes
        # to its own out_dir), so a small thread pool can shrink the
        # critical-path latency by ~k× for k workers. Default 1 keeps
        # bit-identical sequential behaviour for users who don't opt
        # in. EvidenceStore guards its mutators with an RLock so a
        # higher value is safe.
        self._max_concurrent_steps = max(1, int(config.max_concurrent_steps))
        self._development_sample_size = (
            int(config.development_sample_size)
            if config.development_sample_size is not None
            else None
        )
        if (
            self._development_sample_size is not None
            and self._development_sample_size <= 0
        ):
            raise ValueError("development_sample_size must be positive")
        self._development_sample_seed = int(config.development_sample_seed)
        self._development_diagnostic = bool(config.development_diagnostic)
        from .orchestration.profiles import is_paper_facing_profile

        if (
            self._development_sample_size is not None
            and config.submission_profile_name is not None
        ):
            raise ValueError(
                "development cohort sampling is non-paper authority and cannot "
                "be combined with a submission profile"
            )
        if self._development_diagnostic and is_paper_facing_profile(
            config.submission_profile_name
        ):
            raise ValueError(
                "development diagnostics are non-paper authority and cannot "
                "be combined with a paper-facing submission profile"
            )

        if config.allow_underfunded_step_provider_calls and is_paper_facing_profile(
            config.submission_profile_name
        ):
            # Declaring the shortfall makes it visible; it does not make it
            # paper authority. A submission run whose steps could exhaust their
            # provider budget mid-repair reports an accounting failure in the
            # shape of a scientific one. Keyed off paper-facing rather than
            # "a profile was supplied", so `*_dev` profiles can still exercise
            # exhaustion.
            raise ValueError(
                "an under-funded provider budget is non-paper authority and "
                "cannot be combined with a paper-facing submission profile"
            )
        self._enable_probe_step = bool(config.enable_probe_step)
        self._enable_replanning = bool(config.enable_replanning)
        # 0 / None means "no cap". Anything positive enforces the cap in
        # the replanner overflow guard (see execution/phase.py).
        self._max_total_steps = (
            int(config.max_total_steps)
            if config.max_total_steps and config.max_total_steps > 0
            else 0
        )
        # Replan convergence guards (see pipeline_config / execution.phase).
        # 0 disables the guard.
        self._max_consecutive_noop_replans = (
            int(config.max_consecutive_noop_replans)
            if config.max_consecutive_noop_replans
            and config.max_consecutive_noop_replans > 0
            else 0
        )
        self._max_replans = (
            int(config.max_replans)
            if config.max_replans and config.max_replans > 0
            else 0
        )
        # Stabilization / primary-only iterations tighten the replan budget so a
        # non-converging run fails closed fast (~3 revisions) instead of burning
        # the full-run budget of 6. A caller that already set a smaller positive
        # cap keeps it; a disabled cap (0) is re-armed to 3 under stabilization.
        self._stabilization_mode = bool(config.stabilization_mode)
        if self._stabilization_mode:
            self._max_replans = min(self._max_replans, 3) if self._max_replans else 3
        self._max_numeric_claims_per_step = (
            int(config.max_numeric_claims_per_step)
            if config.max_numeric_claims_per_step
            and config.max_numeric_claims_per_step > 0
            else 0
        )
        # Phase-1 writer-digest widening (default off). When True, the
        # writer's evidence digest is augmented with a "secondary
        # numbers" block enumerating NumericClaim entries outside the
        # ``WRITER_DIGEST_PREFERRED_KEYS`` primary subset. The binder
        # already accepts these; the flag controls only what the
        # writer SEES. See reporting.writer_evidence._render_writer_evidence_digest_v2.
        self._writer_digest_widened = bool(config.writer_digest_widened)
        self._writer_digest_secondary_cap_per_step = max(
            0, int(config.writer_digest_secondary_cap_per_step)
        )
        # Legacy cross-run stores are write-only compatibility sources.
        # Their free-text records are never injected into Planner context.
        # Run-derived records are additionally mirrored into the v2
        # permissioned quarantine store during finalization.
        self._enable_experience_bank = bool(config.enable_experience_bank)
        self._experience_bank_path: Optional[Path] = (
            Path(config.experience_bank_path) if config.experience_bank_path else None
        )
        self._experience_bank_top_k = max(0, int(config.experience_bank_top_k))
        self._experience_bank_min_similarity = float(
            config.experience_bank_min_similarity
        )
        self._enable_know_how = bool(config.enable_know_how)
        self._allow_curated_mvp_know_how = bool(
            config.allow_curated_mvp_know_how
            # Preserve the archived 20260721 profile's historical behaviour
            # without changing its serialized replay contract.
            or config.submission_profile_name == "npj_dm_know_how_dev"
        )
        self._enable_coder_resources = bool(config.enable_coder_resources)
        self._enable_reviewed_memory = bool(config.enable_reviewed_memory)
        self._reviewed_memory_namespaces = tuple(config.reviewed_memory_namespaces)
        self._capability_runtime = CapabilityWorkflowRuntime.create(
            enabled=config.enable_capability_workflow,
            profile_name=config.submission_profile_name,
            profile_version=config.submission_profile_version,
            expected_image_digest=config.expected_runner_image_digest,
            request=config.capability_request,
            approval=config.capability_approval,
            activation=config.capability_activation,
        )
        # Operator-supplied control plane for the human-review pause. The
        # optional ``reviewer_identity_resolver`` supplies authenticated
        # identity. Left ``None`` a run cannot answer a pause, so a plan that
        # raises one fails closed rather than continuing unattended.
        self._human_review_gate = services.human_review_gate
        # Set by ``run()`` when the workflow pauses; consumed by
        # ``resume_human_review()``. Holds the state machine because its phase
        # invokers close over this run's evidence store and services.
        self._pending_human_review: Optional[Dict[str, Any]] = None
        # Instance-local owner for mutable run/runtime/review state.  Unlike
        # the run-id file lock, this lease spans a human-review pause and sees
        # fresh run ids, so one object cannot be driven by two calls at once.
        self._instance_lifecycle_lease = PipelineInstanceLifecycleLease()
        self._know_how_paths = tuple(Path(path) for path in config.know_how_paths)
        self._know_how_top_k = int(config.know_how_top_k)
        self._know_how_min_score = float(config.know_how_min_score)
        if not 0 <= self._know_how_top_k <= 5:
            raise ValueError("know_how_top_k must be between 0 and 5")
        if not 0.0 <= self._know_how_min_score <= 1.0:
            raise ValueError("know_how_min_score must be between 0 and 1")
        from .orchestration.profiles import require_profile_know_how_setting

        require_profile_know_how_setting(
            name=config.submission_profile_name,
            version=config.submission_profile_version,
            enabled=self._enable_know_how,
        )
        from .orchestration.profiles import require_profile_curated_know_how_setting

        require_profile_curated_know_how_setting(
            name=config.submission_profile_name,
            version=config.submission_profile_version,
            enabled=bool(config.allow_curated_mvp_know_how),
        )
        from .orchestration.profiles import require_profile_coder_resource_setting

        require_profile_coder_resource_setting(
            name=config.submission_profile_name,
            version=config.submission_profile_version,
            enabled=self._enable_coder_resources,
        )
        from .orchestration.profiles import require_profile_reviewed_memory_setting

        require_profile_reviewed_memory_setting(
            name=config.submission_profile_name,
            version=config.submission_profile_version,
            enabled=self._enable_reviewed_memory,
            namespaces=self._reviewed_memory_namespaces,
        )
        # T3.1 — runner backend selection. ``auto`` prefers a probed Docker
        # image and uses macOS sandbox-exec only when Docker is unavailable;
        # ``docker`` explicitly selects :class:`DockerRunner`
        # which mounts the cohort read-only inside a container with
        # ``--network none`` by default. Users with their own sandbox
        # (e.g. OpenHands) can pass an arbitrary ``runner_factory``
        # that accepts ``workdir=, cohort_parquet=, timeout_seconds=``
        # and returns a runner with a ``run(step_id, code)`` method.
        kind = (config.runner_kind or "auto").lower()
        if services.runner_factory is not None:
            self._runner_kind = "custom"
        elif kind in {"auto", "default"}:
            self._runner_kind = "auto"
        elif kind in {"subprocess", "host"}:
            self._runner_kind = "subprocess"
        elif kind in {"docker", "container", "openhands"}:
            self._runner_kind = "docker"
        else:
            raise ValueError(
                f"Unknown runner_kind {config.runner_kind!r}; "
                "expected 'auto', 'subprocess', 'docker', or pass a runner_factory."
            )
        self._runner_image = config.runner_image
        self._runner_network = config.runner_network
        self._expected_runner_image_digest = config.expected_runner_image_digest
        self._host_runner_authorized = bool(config.host_runner_authorized)
        self._runner_factory = services.runner_factory
        self._runner_kwargs = dict(config.runner_kwargs or {})
        self._validated_runtime_capabilities: Optional[Tuple[str, ...]] = None
        self._validated_runtime_bundle: Optional[Dict[str, object]] = None
        self._memory = RunMemory(self.workdir) if config.enable_memory else None
        self._permissioned_memory_store = (
            FileSystemMemoryStore(self.workdir / ".memory_v2")
            if config.enable_memory
            or config.enable_experience_bank
            or config.enable_reviewed_memory
            else None
        )
        self._reviewed_memory_runtime = ReviewedMemoryRuntime(
            enabled=self._enable_reviewed_memory,
            store=self._permissioned_memory_store,
            allowed_namespaces=self._reviewed_memory_namespaces,
        )

    def _figure_egress_policy(
        self,
        *,
        evidence: Optional[Any] = None,
        run_dir: Optional[Path] = None,
        active_step_evidence_ids: Optional[frozenset] = None,
    ) -> FigureEgressPolicy:
        """Authority for putting rendered figure bytes on the network."""

        return FigureEgressPolicy(
            allow_external_upload=self._allow_external_figure_upload,
            evidence=evidence,
            run_dir=run_dir,
            active_step_evidence_ids=active_step_evidence_ids,
        )

    def _build_runner(
        self,
        *,
        run_dir: Path,
        cohort_path: Path,
        target_outcome: Optional[str] = None,
        universe_path: Optional[Path] = None,
        preselection_universe_capability: Optional[
            PreselectionUniverseOwnerCapability
        ] = None,
        universe_is_typed: bool = False,
        universe_authority_ref: Optional[MaterializedCohortAuthorityRef] = None,
        trajectory_path: Optional[Path] = None,
        trajectory_authority_ref: Optional[MaterializedTrajectoryAuthorityRef] = None,
        trajectory_legacy_capsule_receipt: Optional[
            VerifiedLegacyTrajectoryCapsuleReceipt
        ] = None,
        timeout_seconds: Optional[float] = None,
        cap_provider_timeout: bool = True,
    ):
        """Return the configured runner backend for a single ``run()``.

        Kept as a method (not a closure) so subclasses or tests can
        stub it cleanly. Returns any object that exposes
        ``run(step_id=..., code=...) -> RunResult``.

        ``cohort_path`` is the canonical analysis cohort the steps read as
        ``COHORT_PARQUET``. ``universe_path`` remains host authority metadata;
        it is exposed as ``EASYICU_UNIVERSE_PARQUET`` only when the caller has
        supplied this exact typed robustness/cohort-construction capability.
        """
        # Runner capability discovery is scoped to the backend selected for
        # this build.  Clear a Docker snapshot left in the current ContextVar
        # before a custom factory (which need not instantiate CodeRunner) can
        # inherit it.  DockerRunner installs its own provider below.
        set_runtime_capability_snapshot_provider(None)
        runner_kwargs = dict(self._runner_kwargs)
        extra_env = dict(runner_kwargs.pop("extra_env", {}) or {})
        trajectory_aliases = (
            "TRAJECTORY_PARQUET",
            "EASYICU_TRAJECTORY_PARQUET",
            "COHORT_TRAJECTORY_PARQUET",
        )
        pipeline_owned_env = {
            *HOST_OWNED_RUNNER_ENV_KEYS,
            "OUTCOME_COL",
            "EASYICU_UNIVERSE_PARQUET",
            *trajectory_aliases,
        }
        reject_reserved_runner_env(
            extra_env,
            reserved=tuple(pipeline_owned_env),
            owner="ResearchAgentPipeline",
        )
        if target_outcome:
            extra_env["OUTCOME_COL"] = target_outcome
        if preselection_universe_capability is not None and not isinstance(
            preselection_universe_capability,
            PreselectionUniverseOwnerCapability,
        ):
            raise TypeError(
                "preselection_universe_capability must be a "
                "PreselectionUniverseOwnerCapability"
            )
        if preselection_universe_capability is not None and universe_path is None:
            raise ValueError(
                "pre-selection universe authorization requires universe_path"
            )
        if preselection_universe_capability is not None:
            extra_env["EASYICU_UNIVERSE_PARQUET"] = str(universe_path)
        if trajectory_path is not None:
            candidate = Path(trajectory_path).expanduser()
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate.suffix.lower() not in {".parquet", ".pq"}
            ):
                raise MaterializedTrajectoryError(
                    "staged trajectory must be a regular parquet file"
                )
            selected_trajectory_path = candidate.resolve(strict=True)
            if universe_is_typed:
                if universe_authority_ref is None:
                    raise MaterializedTrajectoryError(
                        "typed cohort trajectory lost its cohort authority"
                    )
                if trajectory_authority_ref is not None:
                    verified_trajectory = (
                        load_verified_materialized_trajectory_authority(
                            selected_trajectory_path,
                            expected_authority=trajectory_authority_ref,
                            expected_universe_authority=universe_authority_ref,
                        )
                    )
                    if verified_trajectory is None:  # pragma: no cover - exact ref
                        raise MaterializedTrajectoryError(
                            "typed trajectory authority is missing"
                        )
                elif trajectory_legacy_capsule_receipt is not None:
                    verify_legacy_trajectory_capsule_receipt(
                        run_dir=run_dir,
                        trajectory_path=selected_trajectory_path,
                        receipt=trajectory_legacy_capsule_receipt,
                        expected_universe_authority=universe_authority_ref,
                    )
                else:
                    raise MaterializedTrajectoryError(
                        "typed cohort trajectory requires an exact sealed authority"
                    )
            elif (
                trajectory_authority_ref is not None
                or trajectory_legacy_capsule_receipt is not None
            ):
                raise MaterializedTrajectoryError(
                    "typed or legacy capsule trajectory authority requires a typed universe"
                )
            for traj_alias in trajectory_aliases:
                extra_env[traj_alias] = str(selected_trajectory_path)
        elif (
            trajectory_authority_ref is not None
            or trajectory_legacy_capsule_receipt is not None
        ):
            raise MaterializedTrajectoryError(
                "trajectory authority cannot be supplied without a staged trajectory"
            )
        effective_timeout_seconds = (
            self._timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        )
        if self._provider_hard_stop is not None and cap_provider_timeout:
            effective_timeout_seconds = self._provider_hard_stop.cap_timeout(
                effective_timeout_seconds
            )
        if self._runner_factory is not None:
            # A user-supplied factory (OpenHands, firecracker, ...) also needs
            # the run's outcome column so deterministic repairs resolve it from
            # OUTCOME_COL rather than guessing a column name.
            runner = self._runner_factory(
                workdir=run_dir,
                cohort_parquet=cohort_path,
                timeout_seconds=effective_timeout_seconds,
                extra_env=extra_env,
                **runner_kwargs,
            )
        else:
            runner_kind = self._runner_kind
            if runner_kind == "auto":
                runner_kind = select_safe_runner_kind(
                    image=self._runner_image,
                    docker_executable=runner_kwargs.get("docker_executable"),
                )
            if runner_kind == "docker":
                runner = DockerRunner(
                    workdir=run_dir,
                    cohort_parquet=cohort_path,
                    timeout_seconds=effective_timeout_seconds,
                    image=self._runner_image,
                    network=self._runner_network,
                    extra_env=extra_env,
                    submission_profile_name=self._submission_profile_name,
                    **runner_kwargs,
                )
            else:
                runner = CodeRunner(
                    workdir=run_dir,
                    cohort_parquet=cohort_path,
                    timeout_seconds=effective_timeout_seconds,
                    python_executable=self._python_executable,
                    extra_env=extra_env,
                    **runner_kwargs,
                )
        # ``_build_runner`` intentionally clears any previous ContextVar at
        # entry.  After the pre-plan probe, every later runner rebuild (for
        # example after cohort materialization) must restore that exact
        # verified snapshot before the Coder prompt is rendered.  Otherwise a
        # host CodeRunner would silently fall back to probing the Codex process
        # rather than the configured execution interpreter.
        if self._validated_runtime_bundle is not None:
            adopt_runtime = getattr(runner, "adopt_validated_runtime_bundle", None)
            if not callable(adopt_runtime):
                raise RuntimeError(
                    "Execution runner changed after runtime preflight and cannot "
                    "adopt the verified environment receipt."
                )
            adopt_runtime(self._validated_runtime_bundle)
        if self._validated_runtime_capabilities is not None:
            frozen_snapshot = self._validated_runtime_capabilities
            set_runtime_capability_snapshot_provider(lambda: frozen_snapshot)
        return runner

    def _preflight_execution_runtime(
        self,
        *,
        run_dir: Path,
        cohort_path: Path,
        target_outcome: Optional[str],
        cap_provider_timeout: bool = True,
    ) -> Tuple[str, ...]:
        """Validate the real execution environment before Planner spend.

        A stale Docker tag or incomplete custom backend used to be discovered
        only after planning. Built-in and custom runners now expose one small
        capability contract so that failure happens before the first planning
        provider call and the Coder sees the exact sandbox allow-list.
        """

        runner = self._build_runner(
            run_dir=run_dir,
            cohort_path=cohort_path,
            target_outcome=target_outcome,
            universe_path=cohort_path,
            cap_provider_timeout=cap_provider_timeout,
        )
        validate = getattr(runner, "validate_runtime_capabilities", None)
        if not callable(validate):
            raise RuntimeError(
                "Custom research-agent runners must implement "
                "validate_runtime_capabilities() and return the import names "
                "available in their immutable execution environment."
            )
        raw_snapshot = validate()
        if isinstance(raw_snapshot, (str, bytes)):
            raise RuntimeError(
                "Runner validate_runtime_capabilities() must return a collection "
                "of import names, not a string."
            )
        snapshot = tuple(
            sorted({str(name).strip() for name in raw_snapshot if str(name).strip()})
        )
        if not snapshot:
            raise RuntimeError(
                "Runner capability validation returned an empty package snapshot."
            )
        frozen_snapshot = tuple(snapshot)
        self._validated_runtime_capabilities = frozen_snapshot
        export_runtime = getattr(runner, "export_validated_runtime_bundle", None)
        self._validated_runtime_bundle = (
            dict(export_runtime()) if callable(export_runtime) else None
        )
        set_runtime_capability_snapshot_provider(lambda: frozen_snapshot)
        return frozen_snapshot

    def _clear_validated_runtime(self) -> None:
        """Remove capabilities published by an earlier run on this instance."""

        self._validated_runtime_capabilities = ()
        self._validated_runtime_bundle = None

    def _generate_or_resume_plan(
        self,
        *,
        agent_context: Any,
        allowed_literature_citation_keys: Sequence[str],
        cohort_path: Path,
        context: Any,
        direct_comparator_literature_keys: Sequence[str],
        emit_progress: Callable[..., None],
        evidence: Any,
        findings: List[ValidationFinding],
        know_how_binding: PlannerKnowHowBinding,
        llm_signature: str,
        planning_contract_context: str,
        preplan_literature: Optional[LiteratureBundle],
        planner_prompt_metrics: Optional[Dict[str, Any]],
        prompt_version: str,
        resume_from_step_id: Optional[str],
        resume_state: Optional[Dict[str, Any]],
        role_resolver: Callable[[str], Any],
        run_dir: Path,
        run_id: str,
        skill_obj: Optional[ClinicalSkill],
        used_mock_llm: bool,
    ) -> _PlanGenerationResult | _progressive_planning.ProgressiveDesignCanaryDraft:
        """Resume or generate one plan without shaping or persisting it."""
        # Resume: reuse the locked plan from the prior run instead of
        # re-planning. A non-deterministic planner would otherwise emit a
        # *different* plan on resume, whose step_ids no longer match the
        # completed-step skip set — so the "resume" would silently re-run the
        # whole analysis under new names. Reusing the saved plan keeps the
        # already-completed step_ids aligned and continues from the failed step.
        reused_prior_plan = False
        reused_plan_path: Optional[Path] = None
        migrated_plan_path: Optional[Path] = None
        proposed_plan: Optional[AnalysisPlan] = None
        development_authority_plan = None
        development_locked_plan_loaded = False
        if resume_state is not None:
            (
                plan,
                reused_plan_path,
                proposed_plan,
                reused_prior_plan,
                plan_generation_mode,
            ) = _resume_compatible_plan(
                run_dir=run_dir,
                resume_state=resume_state,
                context=context,
                agent_context=agent_context,
                evidence=evidence,
                know_how_binding=know_how_binding,
                enable_know_how=self._enable_know_how,
                findings=findings,
            )
        elif self._config.development_locked_analysis_plan_path is not None:
            locked_plan_path = Path(
                self._config.development_locked_analysis_plan_path
            ).expanduser()
            expected_digest = str(
                self._config.development_locked_analysis_plan_sha256 or ""
            )
            if not locked_plan_path.is_file():
                raise ValueError(
                    "development locked analysis plan is not a regular file: "
                    f"{locked_plan_path}"
                )
            observed_digest = sha256_of_file(locked_plan_path)
            if observed_digest != expected_digest:
                raise ValueError(
                    "development locked analysis plan SHA-256 mismatch: "
                    f"expected={expected_digest} observed={observed_digest}"
                )
            try:
                plan = AnalysisPlan.model_validate_json(
                    locked_plan_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise ValueError("development locked analysis plan is invalid") from exc
            if plan.research_question != agent_context.research_question:
                raise ValueError(
                    "development locked analysis plan research question mismatch"
                )
            primary_steps = [
                step for step in plan.steps if step.planned_analysis_role == "primary"
            ]
            if len(primary_steps) == 1 and primary_steps[0].scientific_capability:
                from .planning.analysis_types import (
                    get_analysis_type,
                    optional_analysis_type_for_capability,
                )
                from .planning.capability_registry import get_capability_by_id

                current_type = get_analysis_type(plan.analysis_type)
                declared_capability = get_capability_by_id(
                    primary_steps[0].scientific_capability
                )
                expected_capability = get_capability_by_id(current_type.capability_id)
                rebound_type = None
                if (
                    declared_capability is not None
                    and expected_capability is not None
                    and declared_capability.family == expected_capability.family
                    and declared_capability.capability_id
                    != expected_capability.capability_id
                ):
                    rebound_type = optional_analysis_type_for_capability(
                        declared_capability.capability_id
                    )
                if rebound_type is not None:
                    findings.append(
                        ValidationFinding(
                            validator="scientific_capability",
                            severity="warning",
                            message=(
                                "Rebound a locked development plan's analysis "
                                "subtype to the capability declared by its sole "
                                "primary owner; no estimator or result changed."
                            ),
                            detail={
                                "reason_code": (
                                    "development_locked_plan_primary_capability_rebound"
                                ),
                                "source_analysis_type": plan.analysis_type,
                                "rebound_analysis_type": rebound_type.key,
                                "primary_step_id": primary_steps[0].step_id,
                                "primary_capability_id": (
                                    declared_capability.capability_id
                                ),
                            },
                        )
                    )
                    plan = plan.model_copy(update={"analysis_type": rebound_type.key})
            plan_generation_mode = "development_locked_analysis_plan"
            development_locked_plan_loaded = True
            findings.append(
                ValidationFinding(
                    validator="planner",
                    severity="warning",
                    message=(
                        "Used an exact-digest locked development AnalysisPlan "
                        "without another Planner call; the current host still "
                        "validates and shapes every execution contract."
                    ),
                    detail={
                        "reason_code": "development_locked_analysis_plan_loaded",
                        "analysis_only": True,
                        "source_plan_sha256": observed_digest,
                    },
                )
            )
        elif self._development_diagnostic:
            development_authority_plan = (
                self._scientific_runtime_authorities.development_execution_only_plan(
                    research_question=agent_context.research_question,
                )
            )

        if reused_prior_plan:
            plan, migrated_plan_path, plan_generation_mode = (
                _apply_resume_plan_migrations(
                    plan=plan,
                    agent_context=agent_context,
                    run_dir=run_dir,
                    resume_state=resume_state,
                    resume_from_step_id=resume_from_step_id,
                    role_resolver=role_resolver,
                    evidence=evidence,
                    prompt_version=prompt_version,
                    llm_signature=llm_signature,
                    max_prompt_tokens=self._max_prompt_tokens_per_call,
                    submission_profile_name=self._submission_profile_name,
                    plan_generation_mode=plan_generation_mode,
                    migrated_plan_path=migrated_plan_path,
                    findings=findings,
                    scientific_runtime_authorities=(
                        self._scientific_runtime_authorities
                    ),
                )
            )
        elif development_locked_plan_loaded:
            pass
        elif development_authority_plan is not None:
            plan, development_authority_finding = development_authority_plan
            findings.append(development_authority_finding)
            plan_generation_mode = "development_execution_only_runtime_authority"
        elif skill_obj is not None:
            plan_generation_mode = "deterministic_skill"
            issues = skill_obj.validate_against(pd.read_parquet(cohort_path))
            for msg in issues:
                findings.append(
                    ValidationFinding(
                        validator="clinical_skill",
                        severity="warning",
                        message=msg,
                    )
                )
            plan = skill_obj.plan(context)
        else:
            progressive = self._planner_strategy == "progressive_v2"
            plan_generation_mode = "llm_progressive_v2" if progressive else "llm"
            planner_class = ProgressivePlannerAgent if progressive else PlannerAgent
            planner = planner_class(
                wrap_planner_efficiency_budget(
                    budgeted_role_client(
                        role_resolver,
                        "planner",
                        "planner_plan_generation",
                        limit_tokens=self._max_prompt_tokens_per_call,
                    ),
                    max_calls=(
                        self._config.development_planner_efficiency_max_calls
                        if progressive
                        else None
                    ),
                    max_reported_tokens=(
                        self._config.development_planner_efficiency_max_reported_tokens
                    ),
                    max_wall_seconds=(
                        self._config.development_planner_efficiency_max_wall_seconds
                    ),
                )
            )

            try:
                planner_run_kwargs = dict(
                    know_how_binding.planner_kwargs,
                    allowed_literature_citation_keys=allowed_literature_citation_keys,
                    direct_comparator_literature_keys=(
                        direct_comparator_literature_keys
                    ),
                    enforce_article_contract=True,
                    article_contract_context=context,
                    planning_contract_context=planning_contract_context,
                    progress_callback=planner_retry_progress_callback(
                        emit_progress, run_id=run_id
                    ),
                )
                if progressive:
                    progressive_result = _progressive_planning.run_pipeline_progressive_planner(
                        planner=planner,
                        context=agent_context,
                        run_dir=run_dir,
                        evidence=evidence,
                        prompt_pack_version=prompt_version,
                        resume_checkpoint_path=self._config.development_progressive_resume_checkpoint_path,
                        resume_checkpoint_sha256=self._config.development_progressive_resume_checkpoint_sha256,
                        stop_after_outline=self._config.development_stop_after_planner_outline,
                        cohort_path=cohort_path,
                        llm_signature=llm_signature,
                        planner_kwargs=planner_run_kwargs,
                        preplan_literature=preplan_literature,
                        required_primary_cohort_selection_mode=self._required_primary_cohort_selection_mode,
                        know_how_binding=know_how_binding,
                        planning_contract_context=planning_contract_context,
                        finding_sink=findings.append,
                    )
                    if isinstance(progressive_result, _progressive_planning.ProgressiveDesignCanaryDraft):
                        return progressive_result
                    plan = progressive_result.plan
                    plan_generation_mode = progressive_result.generation_mode
                    planner_prompt_metrics = dict(progressive_result.prompt_metrics)
                else:
                    plan = planner.run(agent_context, **planner_run_kwargs)
                    planner_prompt_metrics = know_how_binding.prompt_metrics(
                        planner,
                        agent_context,
                        planning_contract_context=planning_contract_context,
                    )
            except PlannerArticleContractError:
                raise
            except Exception as exc:
                if not self._enable_deterministic_planner_fallback:
                    raise
                findings.append(
                    ValidationFinding(
                        validator="planner",
                        severity="warning",
                        message=(
                            "Planner agent failed; using deterministic fallback plan: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        detail={"generation_mode": "fallback"},
                    )
                )
                plan = PlannerAgent(MockLLMClient(context=agent_context)).run(
                    agent_context,
                    **know_how_binding.planner_kwargs,
                    allowed_literature_citation_keys=(allowed_literature_citation_keys),
                    direct_comparator_literature_keys=(
                        direct_comparator_literature_keys
                    ),
                )
                used_mock_llm = True
                plan_generation_mode = "fallback"
            dropped_plan_keys = getattr(planner, "last_dropped_plan_keys", None) or {}
            dropped_keys = list(dropped_plan_keys.get("top_level", [])) + list(
                dropped_plan_keys.get("steps", [])
            )
            if dropped_keys:
                findings.append(
                    ValidationFinding(
                        validator="planner_schema",
                        severity="warning",
                        message=(
                            "Planner returned unsupported plan fields that were dropped "
                            "before schema validation."
                        ),
                        detail={"dropped_keys": dropped_keys},
                    )
                )
            plan_normalizations = list(dropped_plan_keys.get("normalizations", []))
            if plan_normalizations:
                findings.append(
                    ValidationFinding(
                        validator="planner_schema",
                        severity="warning",
                        message=(
                            "Planner mixed result generation with typed panel "
                            "rendering; the host compiled each exact panel contract "
                            "into a rendering-only child step."
                        ),
                        detail={
                            "reason_code": "planner_mixed_figure_panels_compiled",
                            "normalizations": plan_normalizations,
                        },
                    )
                )
            # A hosted model occasionally emits structurally-broken plan JSON
            # (e.g. a stray time-window at the top level and no usable steps
            # array) that normalises to 0 steps. An empty plan must never run:
            # retry the real planner once, then fall back to the deterministic
            # plan so the pipeline always executes a real analysis.
            if not plan.steps and self._enable_deterministic_planner_fallback:
                retry_plan = None
                try:
                    retry_plan = planner.run(
                        agent_context,
                        **know_how_binding.planner_kwargs,
                        allowed_literature_citation_keys=allowed_literature_citation_keys,
                        direct_comparator_literature_keys=(
                            direct_comparator_literature_keys
                        ),
                        enforce_article_contract=True,
                        article_contract_context=context,
                        planning_contract_context=planning_contract_context,
                    )
                except Exception:
                    retry_plan = None
                if retry_plan is not None and retry_plan.steps:
                    plan = retry_plan
                    findings.append(
                        ValidationFinding(
                            validator="planner",
                            severity="warning",
                            message="Planner returned an empty plan; recovered on retry.",
                            detail={"generation_mode": "retry"},
                        )
                    )
                else:
                    findings.append(
                        ValidationFinding(
                            validator="planner",
                            severity="warning",
                            message=(
                                "Planner returned an empty plan (0 steps after schema "
                                "validation) twice; using deterministic fallback plan."
                            ),
                            detail={"generation_mode": "fallback"},
                        )
                    )
                    plan = PlannerAgent(MockLLMClient(context=agent_context)).run(
                        agent_context,
                        **know_how_binding.planner_kwargs,
                        allowed_literature_citation_keys=(
                            allowed_literature_citation_keys
                        ),
                        direct_comparator_literature_keys=(
                            direct_comparator_literature_keys
                        ),
                    )
                    used_mock_llm = True
                    plan_generation_mode = "fallback"
            # Structured-纳排 retry: if the plan implies an analysis cohort (a
            # cohort / eligibility / attrition step) but left plan.cohort
            # unstructured, the 纳排 is unenforceable free text. Give the planner
            # one focused retry; adopt it only if it actually structures the
            # cohort, so a good plan is never discarded when the retry doesn't.
            if (
                not used_mock_llm
                and _plan_expects_analysis_cohort(plan)
                and _cohort_definition_is_empty(plan)
            ):
                cohort_retry = None
                try:
                    cohort_retry = planner.run(
                        agent_context,
                        **know_how_binding.planner_kwargs,
                        allowed_literature_citation_keys=allowed_literature_citation_keys,
                        direct_comparator_literature_keys=(
                            direct_comparator_literature_keys
                        ),
                        enforce_article_contract=True,
                        article_contract_context=context,
                        planning_contract_context=planning_contract_context,
                    )
                except Exception:
                    cohort_retry = None
                if (
                    cohort_retry is not None
                    and cohort_retry.steps
                    and not _cohort_definition_is_empty(cohort_retry)
                ):
                    plan = cohort_retry
                    findings.append(
                        ValidationFinding(
                            validator="cohort_contract",
                            severity="warning",
                            message=(
                                "Planner initially left the analysis cohort "
                                "unstructured; recovered structured inclusion/"
                                "exclusion on retry."
                            ),
                            detail={"generation_mode": "cohort_retry"},
                        )
                    )
        if proposed_plan is None:
            proposed_plan = plan.model_copy(deep=True)
        return _PlanGenerationResult(
            plan=plan,
            reused_prior_plan=reused_prior_plan,
            reused_plan_path=reused_plan_path,
            migrated_plan_path=migrated_plan_path,
            plan_generation_mode=plan_generation_mode,
            used_mock_llm=used_mock_llm,
            planner_prompt_metrics=planner_prompt_metrics,
            proposed_plan=proposed_plan,
        )

    def _validate_and_persist_plan(
        self,
        *,
        generation: _PlanGenerationResult | _progressive_planning.ProgressiveDesignCanaryDraft,
        agent_context: Any,
        allowed_literature_citation_keys: Sequence[str],
        analysis_blueprint: Any,
        article_contract: Any,
        article_figure_strategy: Any,
        cohort_path: Path,
        context: Any,
        context_path: Path,
        cost_meter: Optional[CostMeter],
        direct_comparator_literature_keys: Sequence[str],
        emit_progress: Callable[..., None],
        evidence: Any,
        findings: List[ValidationFinding],
        know_how_binding: PlannerKnowHowBinding,
        llm_signature: str,
        long_trajectory_bound: bool,
        preplan_literature: Optional[LiteratureBundle],
        prompt_files: Sequence[Path],
        prompt_version: str,
        repro_envelope: Optional[ReproEnvelope],
        resume_state: Optional[Dict[str, Any]],
        role_resolver: Callable[[str], Any],
        run_dir: Path,
        run_id: str,
        skill_obj: Optional[ClinicalSkill],
        study_design_brief: Any,
    ) -> _PlanPhaseResult | PlannerDesignCanaryComplete:
        """Shape, validate, bind, and persist the generated analysis plan."""
        if isinstance(generation, _progressive_planning.ProgressiveDesignCanaryDraft):
            return _progressive_planning.finalize_progressive_design_canary(
                generation, run_id, run_dir, evidence, cost_meter,
                self._provider_hard_stop, prompt_version, emit_progress,
            )
        plan = generation.plan
        reused_prior_plan = generation.reused_prior_plan
        reused_plan_path = generation.reused_plan_path
        migrated_plan_path = generation.migrated_plan_path
        plan_generation_mode = generation.plan_generation_mode
        used_mock_llm = generation.used_mock_llm
        planner_prompt_metrics = generation.planner_prompt_metrics
        host_normalization_input = plan.model_copy(deep=True)
        # Skip the plan-shaping transforms when resuming: the saved plan is
        # already in its final, transformed form, and re-running split/cap/
        # ensure_* could rename or reorder step_ids and break the resume skip
        # set. A freshly generated plan still gets the full treatment.
        if not reused_prior_plan:
            plan, plan_contract_findings = _enforce_advanced_plan_contract(
                plan=plan,
                context=context,
                long_trajectory_bound=long_trajectory_bound,
            )
            findings.extend(plan_contract_findings)
            plan, split_findings = _split_table_and_figure_outputs_in_plan(plan=plan)
            findings.extend(split_findings)
            plan = _figure_plan.apply_required_plan_obligations(plan, context, findings)
            plan, report_input_findings = _augment_report_typed_product_inputs(
                plan=plan
            )
            findings.extend(report_input_findings)
            # Bind before deterministic figure selection; the universal gate below rechecks every source.
            plan = bind_context_dependence_authority(plan=plan, context=agent_context)
            # Force a declared figure step whenever the publication-figure skill
            # will produce one regardless of the plan: the scorer reads
            # analysis_plan.json, and a question-only heuristic misses tasks
            # that never say "figure" yet still require one. Likewise
            # ensure a declared audit/robustness panel, since that evidence is
            # produced (locked robustness specs, data-quality summaries) but the
            # plan often never presents it.
            plan, result_renderer_findings = (
                _figure_plan.select_deterministic_result_renderers(plan=plan)
            )
            findings.extend(result_renderer_findings)
            plan, figure_guard_findings = _ensure_publication_figure_step_in_plan(
                plan=plan,
                context=context,
                force=self._enable_publication_figure_skill,
            )
            findings.extend(figure_guard_findings)
            plan, cohort_figure_findings = (
                _figure_plan.ensure_cohort_accounting_figure_step(
                    plan=plan,
                )
            )
            findings.extend(cohort_figure_findings)
            plan, audit_panel_findings = _figure_plan.ensure_data_quality_figure_step(
                plan=plan,
                context=context,
            )
            findings.extend(audit_panel_findings)
            plan, empty_figure_findings = (
                _figure_plan.close_empty_deterministic_figure_contracts(plan=plan)
            )
            findings.extend(empty_figure_findings)
            plan = _figure_plan.apply_deterministic_figure_panels(plan, findings)
            # Measurement provenance companions are public Coder inputs. Close
            # them before lifecycle sealing and human review so Execute cannot
            # change the exact Plan payload that the decision approved.
            plan, companion_input_findings = close_measurement_companion_inputs(
                plan=plan,
                context=context,
            )
            findings.extend(companion_input_findings)
            cap = self._max_total_steps
            plan, cap_findings = _cap_plan_preserving_figure_steps(plan=plan, cap=cap)
            findings.extend(_defer_typed_plan_dag_findings_until_probe(cap_findings))
            plan, trajectory_product_findings = augment_trajectory_plan_products(
                plan=plan,
                context=context,
            )
            findings.extend(trajectory_product_findings)
            # The probe-aware replanner receives these structural issues before
            # execution. Keep the initial snapshot advisory so a successfully
            # repaired plan is not blocked by its superseded pre-probe shape.
            findings.extend(
                finding.model_copy(
                    update={
                        "validator": "plan_contract_pending",
                        "severity": "warning",
                        "detail": {
                            **dict(finding.detail or {}),
                            "pending_probe_replan": True,
                        },
                    }
                )
                for finding in trajectory_plan_dag_findings(
                    plan=plan,
                    context=context,
                    long_trajectory_bound=long_trajectory_bound,
                )
            )
            with cohort_concept_id_scope(
                progressive_cohort_concept_ids(
                    agent_context,
                    tuple(variable.name for variable in agent_context.variables),
                )
            ):
                plan = ensure_cohort_definition(plan)
            plan = ensure_robustness_specs(plan)
            # Final gate: if the plan implies a cohort but still has no
            # structured inclusion/exclusion (the retry above didn't recover
            # it), record a loud, auditable contract error instead of silently
            # running the analysis on the full universe.
            findings.extend(_cohort_definition_contract_findings(plan))
        # One boundary for every plan source: LLM, deterministic skill and
        # digest-verified resume. Planner parsing also binds early for prompt
        # diagnostics, but execution authority cannot depend on which producer
        # happened to create the plan.
        bound_dependence_plan = bind_context_dependence_authority(
            plan=plan,
            context=agent_context,
        )
        if reused_prior_plan and bound_dependence_plan != plan:
            raise DependenceAuthorityError(
                "digest-verified resume plan lacks the current repeated-unit "
                "authority projection; start a fresh plan rather than changing "
                "a previously executed plan in memory"
            )
        plan = bound_dependence_plan
        if not reused_prior_plan:
            # The signed runtime owner is the final public-input compiler. Run
            # it after generic input closure so measurement companions or other
            # host shaping cannot drift its exact executor contract.
            plan, scientific_runtime_compile_findings = (
                self._scientific_runtime_authorities.bind_plan(plan)
            )
            findings.extend(scientific_runtime_compile_findings)
        # The endpoint half of the same declaration, checked for every plan
        # rather than only inside the cohort branch above: a family can require
        # a typed endpoint whether or not it also defines an analysis cohort.
        findings.extend(endpoint_contract_findings(plan, context=context))
        # Planner output was validated before shaping, but the host has since
        # split mixed outputs, closed typed dependencies, and applied the step
        # cap. Never offer a human an approval request for a transform-corrupted
        # plan (for example a visualization step whose sole figure declaration
        # was moved away during de-duplication).
        validate_final_plan_shape(plan)
        if study_design_brief is not None:
            if (
                plan.analysis_type
                and article_contract is not None
                and plan.analysis_type != article_contract.source_analysis_type
            ):
                # The pre-plan contract is a prompt profile.  Once the Planner
                # has selected a valid analysis type, seal a separate final
                # contract instead of letting provisional inference retain
                # scientific headline authority.
                final_design = materialize_final_article_design_authority(
                    context=context,
                    analysis_type=plan.analysis_type,
                    run_dir=run_dir,
                    evidence=evidence,
                )
                study_design_brief = final_design.brief
                article_contract = final_design.contract
                article_figure_strategy = final_design.figure_strategy
                analysis_blueprint = final_design.blueprint
            if article_figure_strategy is not None:
                plan = _figure_plan.apply_article_figure_strategy_placements(
                    plan=plan,
                    strategy=article_figure_strategy,
                )
            findings.extend(
                validate_plan_against_study_design_brief(
                    plan=plan,
                    brief=study_design_brief,
                )
            )
        if article_contract is not None:
            findings.extend(
                validate_plan_against_article_contract(
                    plan=plan,
                    contract=article_contract,
                )
            )
        if analysis_blueprint is not None:
            findings.extend(
                validate_plan_against_analysis_blueprint(
                    plan=plan,
                    blueprint=analysis_blueprint,
                )
            )
        if self._required_primary_cohort_selection_mode is not None:
            observed_mode = str(getattr(plan.cohort, "selection_mode", "") or "")
            if observed_mode != self._required_primary_cohort_selection_mode:
                raise CohortAuthorityError(
                    "Planner primary cohort selection mode does not match the "
                    "caller-bound contract: expected "
                    f"{self._required_primary_cohort_selection_mode!r}, observed "
                    f"{observed_mode!r}"
                )
            if not cohort_definition_has_explicit_selection(plan.cohort):
                raise CohortAuthorityError(
                    "Planner primary cohort selection is not explicit"
                )
        validate_plan_against_adjustment_authority(plan=plan, context=agent_context)
        self._scientific_runtime_authorities.validate_plan(plan)
        plan_path = (
            migrated_plan_path or reused_plan_path or (run_dir / "analysis_plan.json")
        )
        for planned_step in plan.steps:
            bind_table_one_execution_spec(planned_step, agent_context)
            bind_step_declared_levels(planned_step, agent_context)
        cohort_concept_ids = progressive_cohort_concept_ids(
            agent_context,
            tuple(variable.name for variable in agent_context.variables),
        )
        normalized_plan = _plan_lifecycle.build_normalized_plan_lineage(
            proposed_plan=generation.proposed_plan,
            proposed_source=plan_generation_mode,
            pre_normalization_plan=host_normalization_input,
            normalized_plan=plan,
            resume_scientific_semantics_changed=plan_generation_mode
            in {
                "resume_with_scientific_migration",
                "resume_with_authority_restore",
            },
            host_scientific_semantics_changed=not reused_prior_plan,
            cohort_concept_ids=cohort_concept_ids,
        )
        cohort_concept_ids = normalized_plan.proposed.cohort_concept_ids
        write_table_one_private_checkpoint(run_dir=run_dir, plan=plan)
        if not reused_prior_plan:
            plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        if evidence.get("analysis_plan") is None:
            evidence.register_file(
                kind="log",
                description=(
                    f"Analysis plan from ClinicalSkill '{skill_obj.key}'."
                    if skill_obj
                    else "Analysis plan emitted by PlannerAgent."
                ),
                source_path=plan_path,
                evidence_id="analysis_plan",
                producer="planner" if skill_obj is None else "clinical_skill",
                generation_mode=plan_generation_mode,
                prompt_pack_version=prompt_version,
                metadata={
                    "llm_signature": llm_signature,
                    "used_mock_llm": used_mock_llm,
                },
            )
        _plan_lifecycle.persist_normalized_plan(
            run_dir=run_dir, evidence=evidence, normalized=normalized_plan
        )
        know_how_binding.persist_prompt_metrics(
            planner_prompt_metrics,
            run_dir=run_dir,
            evidence=evidence,
        )
        if plan_generation_mode in {
            "llm_progressive_v2",
            "llm_progressive_v2_dev_resume",
        }:
            lifecycle_evidence_id = _plan_lifecycle.plan_lifecycle_evidence_id(
                plan.revision
            )
            persist_progressive_planning_authority(
                run_dir=run_dir,
                evidence=evidence,
                proposed_plan_sha256=normalized_plan.proposed.plan_sha256,
                normalized_plan_sha256=normalized_plan.plan_sha256,
                normalized_plan_authority_sha256=normalized_plan.authority_sha256,
                normalized_plan_evidence_id=lifecycle_evidence_id,
                normalized_plan_filename=f"{lifecycle_evidence_id}.json",
                prompt_pack_version=prompt_version,
            )
        if self._config.require_human_plan_review:
            if self._config.require_literature_design_authority:
                _scientific_plan_gate.append_literature_design_authority_finding(
                    findings, plan, preplan_literature
                )
            review_gate = _scientific_plan_gate.prepare_scientific_plan_review_gate(
                context=context,
                plan=plan,
                literature=preplan_literature,
                figure_strategy=article_figure_strategy,
                run_dir=run_dir,
                evidence=evidence,
                require_reportable_capability=(
                    self._config.require_reportable_scientific_capability
                ),
                reuse_existing_review=reused_prior_plan,
            )
            findings.append(review_gate.finding)
        write_locked_cohort_definition(
            run_dir=run_dir,
            plan=plan,
            evidence=evidence,
            prompt_pack_version=prompt_version,
            llm_signature=llm_signature,
            cohort_concept_ids=cohort_concept_ids,
        )
        write_locked_robustness_specs(
            run_dir=run_dir,
            plan=plan,
            evidence=evidence,
            prompt_pack_version=prompt_version,
            llm_signature=llm_signature,
        )
        # Materialize the locked cohort definition into the canonical analysis
        # cohort so the declared inclusion/exclusion is enforced once, on the
        # data every downstream step reads — instead of relying on each
        # LLM-generated step to re-apply 纳排 (which run10/run11 showed it does
        # not, so the primary model ran on the full universe). The full universe
        # is exposed only to typed robustness/cohort-construction steps.
        if not reused_prior_plan:
            analysis_cohort = materialize_locked_analysis_cohort(
                run_dir=run_dir,
                plan=plan,
                universe_path=cohort_path,
                context=context,
                cohort_concept_ids=cohort_concept_ids,
            )
            if analysis_cohort["status"] == "applied":
                findings.append(
                    ValidationFinding(
                        validator="cohort_materializer",
                        severity="info",
                        message=(
                            "Applied the locked cohort definition: analysis cohort "
                            f"n={analysis_cohort['n_cohort']} of universe "
                            f"n={analysis_cohort['n_universe']}. Downstream steps read "
                            "the filtered cohort (COHORT_PARQUET); the full universe "
                            "is available only to explicitly authorized typed steps."
                        ),
                        detail={
                            "n_universe": analysis_cohort["n_universe"],
                            "n_analysis_cohort": analysis_cohort["n_cohort"],
                            "cohort_definition_sha256": analysis_cohort[
                                "cohort_definition_sha256"
                            ],
                            "materialized_cohort_authority_ref": analysis_cohort[
                                "authority_ref"
                            ],
                        },
                    )
                )
            elif analysis_cohort["status"] == "error":
                # The comment above this block records why the materializer
                # exists: run10/run11 left 纳排 to each generated step, and the
                # primary model silently ran on the full universe. Falling back
                # to "downstream steps must apply it themselves" on failure
                # reinstates exactly that path, on a run that has already
                # *locked* a cohort definition. A locked cohort that cannot be
                # applied is a stop, not a warning.
                findings.append(
                    ValidationFinding(
                        validator="cohort_materializer",
                        severity="error",
                        message=(
                            "Could not apply the locked cohort definition to the "
                            "universe. The run is stopped rather than analysing "
                            "the unfiltered universe under a cohort the plan "
                            f"declared. Reason: {analysis_cohort['error']}"
                        ),
                        detail={
                            "reason": "locked_cohort_not_materialized",
                            "materializer_error": str(analysis_cohort["error"]),
                            "cohort_definition_sha256": analysis_cohort.get(
                                "cohort_definition_sha256"
                            ),
                        },
                    )
                )
                raise CohortAuthorityError(
                    "the locked cohort definition could not be materialised "
                    f"({analysis_cohort['error']}); refusing to continue on the "
                    "unfiltered universe"
                )
        emit_progress(
            "plan",
            f"Analysis plan ready with {len(plan.steps)} step(s).",
            run_id=run_id,
            total_steps=len(plan.steps),
        )

        started_at = datetime.now(timezone.utc)
        if resume_state and resume_state.get("started_at"):
            try:
                started_at = datetime.fromisoformat(resume_state["started_at"])
            except Exception:
                pass

        return _PlanPhaseResult(
            context=context,
            agent_context=agent_context,
            context_path=context_path,
            evidence=evidence,
            findings=findings,
            plan=plan,
            plan_path=plan_path,
            llm_signature=llm_signature,
            used_mock_llm=used_mock_llm,
            prompt_version=prompt_version,
            prompt_files=prompt_files,
            role_resolver=role_resolver,
            cost_meter=cost_meter,
            repro_envelope=repro_envelope,
            started_at=started_at,
            resume_state=resume_state,
            allowed_literature_citation_keys=tuple(allowed_literature_citation_keys),
            direct_comparator_literature_keys=tuple(direct_comparator_literature_keys),
            preplan_literature=preplan_literature,
            cohort_concept_ids=cohort_concept_ids,
        )

    def _run_plan_phase(
        self,
        *,
        question: str,
        cohort_path: Path,
        cohort_name: str,
        database: str,
        target_outcome: Optional[str],
        endpoint: Optional[EndpointSpec],
        primary_exposure: Optional[str],
        cross_database_validation: Optional[Sequence[str]],
        inclusion_criteria: Optional[Sequence[str]],
        exclusion_criteria: Optional[Sequence[str]],
        id_columns: Optional[Sequence[str]],
        time_columns: Optional[Sequence[str]],
        outcome_columns: Optional[Sequence[str]],
        time_windows: Optional[Sequence[TimeWindow]],
        concept_descriptions: Optional[Dict[str, str]],
        user_preferences: Optional[Dict[str, Any]],
        notes: Optional[str],
        skill_obj: Optional[ClinicalSkill],
        llm: LLMClient,
        run_dir: Path,
        run_id: str,
        run_language: str,
        experiment_spec_path: Optional[Path],
        resume_state: Optional[Dict[str, Any]],
        resume_context_evidence_path: Optional[Path],
        trajectory_binding: Optional[StagedTrajectoryBinding],
        run_scientific_identity: Dict[str, Any],
        run_environment_identity: Dict[str, Any],
        resume_from_step_id: Optional[str],
        emit_progress: Callable[..., None],
    ) -> _PlanPhaseResult:
        """Build context, attach memory, and emit an execution plan."""
        # The Planner is refused a trajectory design unless the host can see a
        # trajectory, and ResearchContext only ever shows the wide fixed-window
        # representation.  Answering here, from the same predicate the execution
        # phase uses, is what lets the trajectory contract be raised while the
        # Planner can still act on it -- H3 previously met that contract only
        # after its last revision, so it never got to satisfy it.
        long_trajectory_bound = long_trajectory_is_bound(trajectory_binding)
        context_path = run_dir / "research_context.json"
        if resume_context_evidence_path is not None:
            # Resume context authority is the digest-verified evidence copy,
            # never a newly built context from the incoming call. Scientific
            # identity was compared before this phase and before any run-dir
            # write. Restore a stale mutable working copy only from that sealed
            # authority; do not reserialize it (timestamps/provenance paths are
            # part of the original evidence bytes).
            context = parse_research_context_json(
                resume_context_evidence_path.read_text(encoding="utf-8")
            )
            if not context_path.is_file() or sha256_of_file(
                context_path
            ) != sha256_of_file(resume_context_evidence_path):
                shutil.copy2(resume_context_evidence_path, context_path)
        else:
            builder = (
                build_naive_research_context
                if self._disable_icu_context
                else build_research_context
            )
            context_kwargs = dict(
                research_question=question,
                cohort=cohort_path,
                cohort_name=cohort_name,
                database=database,
                target_outcome=target_outcome,
                endpoint=endpoint,
                primary_exposure=primary_exposure,
                cross_database_validation=cross_database_validation,
                inclusion_criteria=inclusion_criteria,
                exclusion_criteria=exclusion_criteria,
                id_columns=id_columns,
                time_columns=time_columns,
                outcome_columns=outcome_columns,
                concept_descriptions=concept_descriptions,
                time_windows=time_windows,
                user_preferences=user_preferences,
                notes=notes,
            )
            if builder is build_research_context:
                context_kwargs["trajectory_binding"] = trajectory_binding
            context = builder(**context_kwargs)
            context_path.write_text(
                context.model_dump_json(indent=2),
                encoding="utf-8",
            )
        emit_progress(
            "context",
            "Research context built.",
            run_id=run_id,
            n_stays=context.cohort.n_stays,
            n_variables=len(context.variables),
        )

        evidence = EvidenceStore(
            root=run_dir,
            enforcement_mode=self._evidence_enforcement_mode,
        )
        if evidence.get("research_context") is None:
            evidence.register_file(
                kind="log",
                description="ResearchContext (frozen at run time).",
                source_path=context_path,
                evidence_id="research_context",
                aliases=["RUN_CONTEXT", "run_context"],
                producer="pipeline",
                generation_mode="system",
            )
        register_context_numeric_claims(evidence, context=context)
        capability_finding = self._capability_runtime.prepare(
            run_dir=run_dir,
            evidence=evidence,
            runtime_import_names=self._validated_runtime_capabilities or (),
            runtime_bundle=self._validated_runtime_bundle,
            is_resume=resume_state is not None,
        )
        self._approved_capability_resources = (
            self._capability_runtime.approved_resources
        )
        concept_fingerprint_path = run_dir / "concept_dict_fingerprint.json"
        concept_fingerprint = write_concept_dict_fingerprint(concept_fingerprint_path)
        assert_concept_dict_matches(
            concept_fingerprint,
            expected_concept_dict_sha=self._expected_concept_dict_sha,
            expected_sofa2_dict_sha=self._expected_sofa2_dict_sha,
            mode="strict",
        )
        if evidence.get("concept_dict_fingerprint") is None:
            evidence.register_file(
                kind="log",
                description="SHA-256 fingerprint for EasyICU concept dictionaries.",
                source_path=concept_fingerprint_path,
                evidence_id="concept_dict_fingerprint",
                producer="pipeline",
                generation_mode="system",
            )
        if (
            experiment_spec_path is not None
            and experiment_spec_path.exists()
            and evidence.get("experiment_spec") is None
        ):
            evidence.register_file(
                kind="log",
                description="Config-driven YAML/JSON experiment specification for this run.",
                source_path=experiment_spec_path,
                evidence_id="experiment_spec",
                producer="pipeline",
                generation_mode="system",
            )
        if evidence.get(RUN_INPUT_CAPSULE_EVIDENCE_ID) is None:
            seal_run_input_capsule(
                run_dir=run_dir,
                evidence=evidence,
                scientific_identity=run_scientific_identity,
                initial_environment=run_environment_identity,
                context_path=context_path,
                cohort_path=cohort_path,
                experiment_spec_path=experiment_spec_path,
            )
        architecture_profile = default_architecture_profile()
        architecture_json = run_dir / "architecture_profile.json"
        architecture_md = run_dir / "architecture_profile.md"
        write_json_artifact(architecture_json, architecture_profile)
        architecture_md.write_text(
            architecture_profile_markdown(architecture_profile),
            encoding="utf-8",
        )
        if evidence.get("architecture_profile") is None:
            evidence.register_file(
                kind="log",
                description="Declared four-layer architecture profile for this runtime.",
                source_path=architecture_json,
                evidence_id="architecture_profile",
                producer="pipeline",
                generation_mode="system",
            )
        if evidence.get("architecture_profile_markdown") is None:
            evidence.register_file(
                kind="log",
                description="Human-readable architecture profile for this runtime.",
                source_path=architecture_md,
                evidence_id="architecture_profile_markdown",
                producer="pipeline",
                generation_mode="system",
            )

        findings = (
            [capability_finding]
            if capability_finding is not None
            else preplan_data_findings(context=context, cohort_path=cohort_path)
        )
        if any(f.severity == "error" for f in findings):
            capability_blocked = capability_finding is not None
            emit_progress(
                "capability" if capability_blocked else "audit",
                (
                    "Capability review is required before provider execution."
                    if capability_blocked
                    else "Pre-plan data gate failed; aborting before provider execution."
                ),
                status="error",
                run_id=run_id,
            )
            aborted = self._finalise_aborted(
                run_id=run_id,
                run_dir=run_dir,
                context=context,
                context_path=context_path,
                evidence=evidence,
                findings=findings,
                reason=(
                    "capability_review_required"
                    if capability_blocked
                    else preplan_data_failure_reason(findings)
                ),
            )
            return _PlanPhaseResult(
                context=context,
                agent_context=context,
                context_path=context_path,
                evidence=evidence,
                findings=findings,
                plan=AnalysisPlan(
                    research_question=context.research_question,
                    steps=[],
                ),
                plan_path=run_dir / "analysis_plan.json",
                llm_signature=self._llm_signature(llm),
                used_mock_llm=any(True for _ in self._iter_mock_clients(llm)),
                prompt_version=PROMPT_PACK_VERSION,
                prompt_files=prompt_pack_files(),
                role_resolver=lambda _role: resolve_role_client(llm, _role),
                cost_meter=None,
                repro_envelope=None,
                started_at=datetime.now(timezone.utc),
                resume_state=resume_state,
                aborted_result=aborted,
            )
        emit_progress(
            "audit",
            "Initial cohort audit passed.",
            run_id=run_id,
            findings=len(findings),
        )

        agent_context = context
        if self._context_top_k and skill_obj is None:
            agent_context = build_retrieved_research_context(
                agent_context,
                query=question,
                top_k=self._context_top_k,
            )
            agent_context_path = run_dir / "research_context_agent_prompt.json"
            agent_context_path.write_text(
                agent_context.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if evidence.get("research_context_agent_prompt") is None:
                evidence.register_file(
                    kind="log",
                    description="Prompt-sized ResearchContext after concept retrieval (O6).",
                    source_path=agent_context_path,
                    evidence_id="research_context_agent_prompt",
                    producer="pipeline",
                    generation_mode="system",
                )

        (
            preplan_terminal_result,
            agent_context,
            allowed_literature_citation_keys,
            direct_comparator_literature_keys,
            preplan_literature,
        ) = _run_preplan_literature_and_hypothesis(
            self,
            skill_obj=skill_obj,
            emit_progress=emit_progress,
            run_id=run_id,
            agent_context=agent_context,
            run_dir=run_dir,
            evidence=evidence,
            findings=findings,
            context=context,
            context_path=context_path,
            llm=llm,
            resume_state=resume_state,
        )
        if preplan_terminal_result is not None:
            return preplan_terminal_result

        planner_prompt_metrics: Optional[Dict[str, Any]] = None
        know_how_binding = PlannerKnowHowBinding()
        if self._enable_know_how and skill_obj is None:
            prepared_know_how = prepare_preplan_know_how(
                context=agent_context,
                run_dir=run_dir,
                evidence=evidence,
                database=database,
                paths=self._know_how_paths,
                top_k=self._know_how_top_k,
                min_score=self._know_how_min_score,
                allow_curated_mvp=self._allow_curated_mvp_know_how,
            )
            know_how_binding = PlannerKnowHowBinding.from_prepared(prepared_know_how)

        study_design_brief = None
        article_contract = None
        article_figure_strategy = None
        analysis_blueprint = None
        planning_contract_context = ""
        try:
            # Contract scope comes only from the frozen user/data context.
            # Prompt-enrichment notes include generic examples such as
            # "external validation" and "transportability"; feeding those
            # generated notes back into adaptive trigger inference would
            # silently widen a single-database study.
            study_design_brief = build_study_design_brief(context)
            study_design_path = run_dir / "study_design_brief.json"
            study_design_path.write_text(
                study_design_brief.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if evidence.get("study_design_brief") is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        "Pre-plan study-design brief: analysis family, expected "
                        "methods, main-text displays, supplementary displays, "
                        "sensitivity requirements, and covariate strategy."
                    ),
                    source_path=study_design_path,
                    evidence_id="study_design_brief",
                    producer="study_design_scout",
                    generation_mode="deterministic_skill",
                )
            article_contract = build_article_analysis_contract(
                context,
                brief=study_design_brief,
            )
            article_contract_path = run_dir / "article_analysis_contract.json"
            article_contract_path.write_text(
                article_contract.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if evidence.get("article_analysis_contract") is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        "Pre-plan article analysis contract: required "
                        "article-level roles and display modules derived from "
                        "the study-design brief."
                    ),
                    source_path=article_contract_path,
                    evidence_id="article_analysis_contract",
                    producer="study_design_scout",
                    generation_mode="deterministic_skill",
                )
            article_figure_strategy = build_article_figure_strategy(context)
            figure_strategy_path = run_dir / "article_figure_strategy.json"
            figure_strategy_path.write_text(
                article_figure_strategy.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if evidence.get("article_figure_strategy") is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        "Pre-plan article figure strategy: required visual "
                        "roles, chart families, hero-role expectation, and "
                        "figure anti-patterns derived from the study-design family."
                    ),
                    source_path=figure_strategy_path,
                    evidence_id="article_figure_strategy",
                    producer="study_design_scout",
                    generation_mode="deterministic_skill",
                )
            analysis_blueprint = build_analysis_blueprint(
                context,
                brief=study_design_brief,
                contract=article_contract,
                figure_strategy=article_figure_strategy,
            )
            analysis_blueprint_path = run_dir / "analysis_blueprint.json"
            analysis_blueprint_path.write_text(
                analysis_blueprint.model_dump_json(indent=2),
                encoding="utf-8",
            )
            if evidence.get("analysis_blueprint") is None:
                evidence.register_file(
                    kind="log",
                    description=(
                        "Pre-plan analysis blueprint: question family, prior-art "
                        "design brief, article roles, figure strategy, and "
                        "validation gates used to shape planner output."
                    ),
                    source_path=analysis_blueprint_path,
                    evidence_id="analysis_blueprint",
                    producer="study_design_scout",
                    generation_mode="deterministic_skill",
                )
            design_note = render_analysis_blueprint_for_prompt(analysis_blueprint)
            planning_contract_context = design_note
            agent_notes = (
                f"{agent_context.notes}\n\n{design_note}"
                if agent_context.notes
                else design_note
            )
            agent_context = agent_context.model_copy(update={"notes": agent_notes})
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="study_design_brief",
                    severity="warning",
                    message=(
                        "Study-design brief failed; planner will use context only: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
        scientific_plan_guardrails = render_plan_scientific_guardrails(agent_context)
        planning_contract_context = "\n\n".join(
            value
            for value in (
                planning_contract_context,
                scientific_plan_guardrails,
                self._bound_plan_revision_contract,
            )
            if value
        )
        if self._required_primary_cohort_selection_mode is not None:
            population_contract = (
                "CALLER-BOUND PRIMARY COHORT MODE: set "
                "AnalysisPlan.cohort.selection_mode exactly to "
                f"{self._required_primary_cohort_selection_mode!r}. "
            )
            if self._required_primary_cohort_selection_mode == "all_input_rows":
                population_contract += (
                    "Keep cohort.inclusion and cohort.exclusion empty; do not "
                    "invent a completeness, anchor, or proxy eligibility filter."
                )
            else:
                population_contract += (
                    "Declare at least one typed inclusion/exclusion predicate."
                )
            planning_contract_context = "\n\n".join(
                value
                for value in (planning_contract_context, population_contract)
                if value
            )

        # The Planner prompt renders this guide itself, but from the context
        # alone -- which carries only the wide representation. A run whose
        # trajectory is bound as the long typed input therefore saw nothing,
        # and was then refused by a gate that DOES know about that tier. Only
        # the pipeline holds the flag, so it supplies the guide the prompt
        # could not build, and only when the prompt's own attempt came back
        # empty, so a wide-column run is never told twice.
        analysis_type_key = infer_analysis_type(agent_context).key
        trajectory_planning_guides = []
        if long_trajectory_bound and not trajectory_planner_contract_guide(
            context=agent_context,
            analysis_type=analysis_type_key,
        ):
            trajectory_planning_guides.append(
                trajectory_planner_contract_guide(
                    context=agent_context,
                    analysis_type=analysis_type_key,
                    long_trajectory_bound=True,
                )
            )
        # The converse case: a group-discovery study with no trajectory in
        # either representation is still asked for a stability audit, and the
        # only typed stability field it can see belongs to the trajectory
        # calculator. Declaring that field is what refused m3's whole plan.
        trajectory_planning_guides.append(
            non_trajectory_clustering_stability_guide(
                context=agent_context,
                analysis_type=analysis_type_key,
                long_trajectory_bound=long_trajectory_bound,
            )
        )
        planning_contract_context = "\n\n".join(
            value
            for value in (planning_contract_context, *trajectory_planning_guides)
            if value
        )

        for client in self._iter_mock_clients(llm):
            client.context = agent_context
        llm_signature = self._llm_signature(llm)
        used_mock_llm = any(True for _ in self._iter_mock_clients(llm))
        prompt_version = PROMPT_PACK_VERSION
        prompt_files = prompt_pack_files()

        cost_meter: Optional[CostMeter] = None
        repro_envelope: Optional[ReproEnvelope] = None
        if self._enable_reproducibility_envelope:
            repro_envelope = ReproEnvelope(
                run_id=run_id,
                seed=self._llm_seed,
                include_previews=self._envelope_include_previews,
            )
        if repro_envelope is not None:
            base_role_resolver = envelope_role_resolver(
                llm,
                repro_envelope,
                seed=self._llm_seed,
            )
        else:

            def base_role_resolver(role: str):
                return resolve_role_client(llm, role)

        if self._provider_hard_stop is not None:

            def stopped_role_resolver(role: str):
                base = base_role_resolver(role)
                if base is None or isinstance(base, HardStopClient):
                    return base
                return HardStopClient(
                    base,
                    role=role,
                    task=self._provider_hard_stop,
                )

        else:
            stopped_role_resolver = base_role_resolver

        if self._enable_cost_tracking:
            cost_meter = (
                CostMeter(
                    price_table=(
                        dict(self._cost_price_table) if self._cost_price_table else None
                    ),
                    runtime_dir=run_dir / ".runtime",
                )
                if self._cost_price_table is not None
                else CostMeter(runtime_dir=run_dir / ".runtime")
            )

            # Order: envelope -> hard stop -> meter. The hard-stop wrapper
            # reserves every raw transport retry before delivery; the meter
            # receives usage from that same call for the normal run manifest.
            class _RoleResolverShim:
                name = "role_resolver_shim"

                def __init__(self, resolver):
                    self._resolver = resolver

                def for_role(self, role: str):
                    return self._resolver(role)

                def complete(self, *args, **kwargs):  # pragma: no cover
                    raise RuntimeError(
                        "RoleResolverShim is a dispatcher; call for_role() first."
                    )

            role_resolver = metered_role_resolver(
                _RoleResolverShim(stopped_role_resolver),
                cost_meter,
            )
        else:
            role_resolver = stopped_role_resolver
        generation = self._generate_or_resume_plan(
            agent_context=agent_context,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            cohort_path=cohort_path,
            context=context,
            direct_comparator_literature_keys=direct_comparator_literature_keys,
            emit_progress=emit_progress,
            evidence=evidence,
            findings=findings,
            know_how_binding=know_how_binding,
            llm_signature=llm_signature,
            planning_contract_context=planning_contract_context,
            preplan_literature=preplan_literature,
            planner_prompt_metrics=planner_prompt_metrics,
            prompt_version=prompt_version,
            resume_from_step_id=resume_from_step_id,
            resume_state=resume_state,
            role_resolver=role_resolver,
            run_dir=run_dir,
            run_id=run_id,
            skill_obj=skill_obj,
            used_mock_llm=used_mock_llm,
        )
        return self._validate_and_persist_plan(
            generation=generation,
            agent_context=agent_context,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            analysis_blueprint=analysis_blueprint,
            article_contract=article_contract,
            article_figure_strategy=article_figure_strategy,
            cohort_path=cohort_path,
            context=context,
            context_path=context_path,
            cost_meter=cost_meter,
            direct_comparator_literature_keys=direct_comparator_literature_keys,
            emit_progress=emit_progress,
            evidence=evidence,
            findings=findings,
            know_how_binding=know_how_binding,
            llm_signature=llm_signature,
            long_trajectory_bound=long_trajectory_bound,
            preplan_literature=preplan_literature,
            prompt_files=prompt_files,
            prompt_version=prompt_version,
            repro_envelope=repro_envelope,
            resume_state=resume_state,
            role_resolver=role_resolver,
            run_dir=run_dir,
            run_id=run_id,
            skill_obj=skill_obj,
            study_design_brief=study_design_brief,
        )

    def _execute_phase_services(self) -> ExecutePhaseServices:
        """Build a fresh dependency snapshot for one execute-phase call."""

        return ExecutePhaseServices(
            build_probe_summary=_build_probe_summary,
            deterministic_figure_family_supported_for_upstream=(
                deterministic_figure_family_supported_for_upstream
            ),
            promote_prior_publication_bundle=_promote_prior_publication_bundle,
            promote_sibling_figure_exports=_promote_sibling_figure_exports,
            render_publication_bundle_from_prior_outputs_for_step=(
                _render_publication_bundle_from_prior_outputs_for_step
            ),
            semantic_aliases_for=_semantic_aliases_for,
            publication_figure_authority=PublicationFigureAuthorityServices(
                distribution_availability_step_matches_parent=(
                    _distribution_availability_figure_step_matches_parent
                ),
                sealed_renderer_step_matches_parent=(
                    _sealed_renderer_figure_step_matches_parent
                ),
                sealed_renderer_parent_digest_seal=(
                    _sealed_renderer_parent_digest_seal
                ),
                deterministic_repair_id_for_upstream=(
                    deterministic_figure_repair_id_for_upstream
                ),
            ),
        )

    def _run_execute_phase(
        self,
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
    ) -> "_ExecutePhaseResult":
        """Delegate to :mod:`execution.phase`.

        The execute loop body is in :mod:`execution.phase` so this
        file does not have to host both the orchestration shell and the
        execute-phase guts. Late-imported to keep ``import pipeline``
        free of a cycle.
        """
        from .execution.phase import run_execute_phase

        return run_execute_phase(
            self,
            plan_result=plan_result,
            cohort_path=cohort_path,
            trajectory_binding=trajectory_binding,
            run_dir=run_dir,
            run_id=run_id,
            skill_obj=skill_obj,
            notes=notes,
            emit_progress=emit_progress,
            resume_from_step_id=resume_from_step_id,
            stop_after_step_id=stop_after_step_id,
        )

    def _run_write_phase(
        self,
        *,
        plan_result: _PlanPhaseResult,
        execute_result: _ExecutePhaseResult,
        run_dir: Path,
        run_id: str,
        stop_after_analysis: bool,
        manuscript_title: Optional[str],
        manuscript_authors: Optional[Sequence[str]],
        run_language: str,
        emit_progress: Callable[..., None],
        force_writer_probe: bool = False,
    ) -> _WritePhaseResult:
        """Delegate to :mod:`reporting.write_phase`."""
        from .reporting.write_phase import run_write_phase

        return run_write_phase(
            self,
            plan_result=plan_result,
            execute_result=execute_result,
            run_dir=run_dir,
            run_id=run_id,
            stop_after_analysis=stop_after_analysis,
            manuscript_title=manuscript_title,
            manuscript_authors=manuscript_authors,
            run_language=run_language,
            emit_progress=emit_progress,
            force_writer_probe=force_writer_probe,
        )

    def _finalise_success(
        self,
        *,
        plan_result: _PlanPhaseResult,
        execute_result: _ExecutePhaseResult,
        write_result: _WritePhaseResult,
        run_id: str,
        run_dir: Path,
        cohort_path: Path,
        notes: Optional[str],
        database: str,
        target_outcome: Optional[str],
        stop_after_analysis: bool,
        cache_key: Optional[str],
        scientific_identity: Mapping[str, Any],
        experiment_spec_path: Optional[Path],
        audit_logger: Optional[AuditLogger],
        emit_progress: Callable[..., None],
    ) -> PipelineResult:
        """Delegate to :mod:`orchestration.finalize`."""
        from .orchestration.finalize import finalise_success

        return finalise_success(
            self,
            plan_result=plan_result,
            execute_result=execute_result,
            write_result=write_result,
            run_id=run_id,
            run_dir=run_dir,
            cohort_path=cohort_path,
            notes=notes,
            database=database,
            target_outcome=target_outcome,
            stop_after_analysis=stop_after_analysis,
            cache_key=cache_key,
            scientific_identity=scientific_identity,
            experiment_spec_path=experiment_spec_path,
            audit_logger=audit_logger,
            emit_progress=emit_progress,
        )

    # ------------------------------------------------------------------
    # Cross-run experience bank (Phase-1, Commit 3 — opt-in)
    # ------------------------------------------------------------------

    def _experience_bank(self) -> Optional[ExperienceBank]:
        """Lazy-construct the ExperienceBank from the configured path.

        Returns ``None`` when the feature is disabled or no path was
        configured. Callers must treat the bank as optional.
        """
        if not self._enable_experience_bank or self._experience_bank_path is None:
            return None
        return ExperienceBank(path=self._experience_bank_path)

    def retrieve_experience_hints(
        self,
        *,
        research_question: str,
        database: Optional[str] = None,
    ) -> List[Tuple[ExperienceRecord, float]]:
        """Top-k experience records most lexically similar to the question.

        Returns an empty list when the bank is disabled, missing, or
        contains no records above the configured similarity floor.
        Surface the returned ``summary`` lines verbatim in the planner
        prompt — the bank guarantees they are self-contained
        sentences. Callers that need the raw records (e.g. for
        debugging) can read ``ExperienceBank(path).records()``
        directly.
        """
        bank = self._experience_bank()
        if bank is None:
            return []
        try:
            return bank.retrieve(
                research_question=research_question,
                database=database,
                top_k=self._experience_bank_top_k,
                min_similarity=self._experience_bank_min_similarity,
            )
        except ExperienceBankCorruptError:
            # Plan with no bank rather than with an unprovable subset of one.
            # Logged loudly: a corrupt shared bank is an operator problem that
            # will otherwise recur on every run that reads it.
            logger.error(
                "experience bank %s is corrupt; planning without it",
                self._experience_bank_path,
            )
            return []

    def reflect_and_persist_experience(
        self,
        *,
        run_dir: Path,
        context: ResearchContext,
        database: str,
        cohort_name: str,
    ) -> List[ExperienceRecord]:
        """Mine legacy records and mirror them into permissioned quarantine.

        Reads ``run_status.json`` from ``run_dir`` for the gates +
        findings + superseded-error partition. Returns the records
        that were registered (after dedup against the existing bank); finalization
        mirrors these records into ``run_lessons/quarantine`` and never injects
        them into Planner context. Returns an empty list when the feature is
        disabled or the run dir
        does not yet have a run_status.

        Idempotent on ``(kind, summary)``: re-running over the same
        run_dir does not duplicate records, only refreshes
        ``produced_at`` / ``producer_run_id``.
        """
        bank = self._experience_bank()
        if bank is None:
            return []
        run_status_path = Path(run_dir) / "run_status.json"
        if not run_status_path.exists():
            return []
        try:
            run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "experience-bank: run_status.json at %s could not be read: %s",
                run_status_path,
                exc,
            )
            return []
        gates = run_status.get("gates") or run_status.get("readiness_gates") or {}
        findings = run_status.get("findings") or []
        superseded_errors = (
            gates.get("superseded_errors") or run_status.get("superseded_errors") or []
        )
        plan_step_ids = []
        for step in run_status.get("plan_steps") or []:
            sid = step.get("step_id") if isinstance(step, dict) else None
            if sid:
                plan_step_ids.append(str(sid))
        if not plan_step_ids:
            # Fall back to per_step_records if plan_steps not surfaced.
            for record in run_status.get("per_step_records") or []:
                sid = record.get("step_id") if isinstance(record, dict) else None
                if sid:
                    plan_step_ids.append(str(sid))
        records = mine_experience_from_run(
            research_question=context.research_question,
            database=database,
            cohort_name=cohort_name,
            gates=gates,
            findings=findings,
            superseded_errors=superseded_errors,
            plan_step_ids=plan_step_ids,
            producer_run_id=Path(run_dir).name,
        )
        bank.extend(records)
        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _relative_run_path(run_dir: Path, value: Optional[Path]) -> Optional[str]:
        if value is None:
            return None
        resolved = Path(value).resolve()
        try:
            return resolved.relative_to(run_dir.resolve()).as_posix()
        except ValueError as exc:
            raise HumanReviewCheckpointError(
                "human-review handoff path escapes its run directory"
            ) from exc

    def _persist_review_checkpoint(
        self,
        *,
        plan_result: _PlanPhaseResult,
        requests: Sequence[Any],
        run_id: str,
        run_dir: Path,
        cohort_path: Path,
        trajectory_binding: Optional[StagedTrajectoryBinding],
        runtime_capabilities: Sequence[str],
        runtime_bundle: Optional[Mapping[str, Any]],
        run_environment_identity: Mapping[str, Any],
        notes: Optional[str],
        database: str,
        target_outcome: Optional[str],
        stop_after_step_id: Optional[str],
        stop_after_analysis: bool,
        manuscript_title: Optional[str],
        manuscript_authors: Optional[Sequence[str]],
        run_language: str,
        force_writer_probe: bool,
        scientific_identity: Mapping[str, Any],
        experiment_spec_path: Optional[Path],
        cache_key: Optional[str],
        skill_obj: Optional[ClinicalSkill],
    ) -> Optional[HumanReviewCheckpoint]:
        """Persist a fresh Plan→Review handoff without serializing live objects.

        A run already resumed from a prior execution checkpoint may contain
        legacy scientific migrations.  That is deliberately left on the
        existing same-process path: making it durable would require a second
        authority contract for migration history, which this narrow P0 does
        not invent.
        """

        if plan_result.resume_state is not None:
            return None
        capsule = plan_result.evidence.get(RUN_INPUT_CAPSULE_EVIDENCE_ID)
        if capsule is None:
            raise HumanReviewCheckpointError(
                "cannot persist human review without a run-input capsule"
            )
        execution_authorities: list[ReviewExecutionAuthority] = []
        for request in requests:
            payload = request.payload if hasattr(request, "payload") else {}
            raw_authority = (
                payload.get("plan_review_authority")
                if isinstance(payload, Mapping)
                else None
            )
            if not isinstance(raw_authority, Mapping):
                raise HumanReviewCheckpointError(
                    "human-review request lacks plan-review authority"
                )
            authority = PlanReviewAuthority.model_validate(raw_authority)
            if authority.execution is None:
                raise HumanReviewCheckpointError(
                    "human-review request lacks execution authority"
                )
            execution_authorities.append(authority.execution)
        if not execution_authorities or any(
            item != execution_authorities[0] for item in execution_authorities[1:]
        ):
            raise HumanReviewCheckpointError(
                "human-review requests do not share one execution authority"
            )
        execution_authority = execution_authorities[0]
        if execution_authority.run_input_capsule_sha256 != str(capsule.sha256):
            raise HumanReviewCheckpointError(
                "review execution authority does not bind the run-input capsule"
            )
        trajectory_payload: Optional[dict[str, Any]] = None
        if trajectory_binding is not None:
            trajectory_payload = {
                "path": self._relative_run_path(run_dir, trajectory_binding.path),
                "sha256": trajectory_binding.sha256,
                "size": trajectory_binding.size,
                "authority_ref": (
                    trajectory_binding.authority_ref.to_dict()
                    if trajectory_binding.authority_ref is not None
                    else None
                ),
                "legacy_capsule_receipt": (
                    {
                        "capsule_sha256": (
                            trajectory_binding.legacy_capsule_receipt.capsule_sha256
                        ),
                        "trajectory_relative_path": (
                            trajectory_binding.legacy_capsule_receipt.trajectory_relative_path
                        ),
                        "trajectory_sha256": (
                            trajectory_binding.legacy_capsule_receipt.trajectory_sha256
                        ),
                        "trajectory_size": (
                            trajectory_binding.legacy_capsule_receipt.trajectory_size
                        ),
                        "universe_authority_sha256": (
                            trajectory_binding.legacy_capsule_receipt.universe_authority_sha256
                        ),
                        "schema_version": (
                            trajectory_binding.legacy_capsule_receipt.schema_version
                        ),
                    }
                    if trajectory_binding.legacy_capsule_receipt is not None
                    else None
                ),
            }
        repro_payload = None
        if plan_result.repro_envelope is not None:
            repro_payload = {
                "seed": plan_result.repro_envelope.seed,
                "preview_max_chars": plan_result.repro_envelope.preview_max_chars,
                "include_previews": plan_result.repro_envelope.include_previews,
                "env_snapshot": dict(plan_result.repro_envelope.env_snapshot),
                "calls": [item.to_json() for item in plan_result.repro_envelope.calls],
            }
        plan_handoff = {
            "context": plan_result.context.model_dump(mode="json"),
            "agent_context": plan_result.agent_context.model_dump(mode="json"),
            "context_path": self._relative_run_path(run_dir, plan_result.context_path),
            "findings": [item.model_dump(mode="json") for item in plan_result.findings],
            "plan": plan_result.plan.model_dump(mode="json"),
            "plan_path": self._relative_run_path(run_dir, plan_result.plan_path),
            "llm_signature": plan_result.llm_signature,
            "used_mock_llm": bool(plan_result.used_mock_llm),
            "prompt_version": plan_result.prompt_version,
            "prompt_files": dict(plan_result.prompt_files),
            "started_at": plan_result.started_at.isoformat(),
            "allowed_literature_citation_keys": list(
                plan_result.allowed_literature_citation_keys
            ),
            "direct_comparator_literature_keys": list(
                plan_result.direct_comparator_literature_keys
            ),
            "preplan_literature": (
                plan_result.preplan_literature.model_dump(mode="json")
                if plan_result.preplan_literature is not None
                else None
            ),
            "cohort_concept_ids": list(plan_result.cohort_concept_ids),
            "repro_envelope": repro_payload,
        }
        execution_coordinates = {
            "cohort_path": self._relative_run_path(run_dir, cohort_path),
            "trajectory_binding": trajectory_payload,
            "notes": notes,
            "database": database,
            "target_outcome": target_outcome,
            "stop_after_step_id": stop_after_step_id,
            "stop_after_analysis": bool(stop_after_analysis),
            "manuscript_title": manuscript_title,
            "manuscript_authors": list(manuscript_authors or ()),
            "run_language": run_language,
            "force_writer_probe": bool(force_writer_probe),
            "scientific_identity": dict(scientific_identity),
            "experiment_spec_path": self._relative_run_path(
                run_dir, experiment_spec_path
            ),
            "cache_key": cache_key,
            "skill_key": skill_obj.key if skill_obj is not None else None,
        }
        checkpoint = HumanReviewCheckpoint.create(
            run_id=run_id,
            pipeline_config_sha256=self._config.canonical_digest(),
            environment_identity=run_environment_identity,
            llm_signature_sha256=canonical_sha256(plan_result.llm_signature),
            run_input_capsule_sha256=str(capsule.sha256),
            capability_activation_sha256=(
                execution_authority.capability_activation_sha256
            ),
            runtime_capabilities=runtime_capabilities,
            runtime_bundle=runtime_bundle,
            requests=tuple(requests),
            plan_handoff=plan_handoff,
            execution_coordinates=execution_coordinates,
        )
        write_human_review_checkpoint(human_review_checkpoint_path(run_dir), checkpoint)
        return checkpoint

    def _restore_role_handoff(
        self,
        *,
        run_id: str,
        run_dir: Path,
        repro_payload: Optional[Mapping[str, Any]],
    ) -> tuple[Callable[[str], Any], Optional[CostMeter], Optional[ReproEnvelope]]:
        llm = self._llm
        if llm is None:
            raise HumanReviewCheckpointError(
                "durable human-review resume requires the configured provider"
            )
        repro_envelope: Optional[ReproEnvelope] = None
        if self._enable_reproducibility_envelope:
            if not isinstance(repro_payload, Mapping):
                raise HumanReviewCheckpointError(
                    "reproducibility envelope is absent from the plan handoff"
                )
            try:
                calls = [
                    ReproCallRecord(**dict(item))
                    for item in list(repro_payload.get("calls") or ())
                ]
                repro_envelope = ReproEnvelope(
                    run_id=run_id,
                    seed=repro_payload.get("seed"),
                    preview_max_chars=int(
                        repro_payload.get("preview_max_chars") or 280
                    ),
                    include_previews=bool(repro_payload.get("include_previews")),
                    calls=calls,
                    env_snapshot=dict(repro_payload.get("env_snapshot") or {}),
                )
            except Exception as exc:
                raise HumanReviewCheckpointError(
                    "reproducibility envelope cannot be rehydrated"
                ) from exc
        base_role_resolver = (
            envelope_role_resolver(llm, repro_envelope, seed=self._llm_seed)
            if repro_envelope is not None
            else lambda role: resolve_role_client(llm, role)
        )
        if self._provider_hard_stop is not None:

            def stopped_role_resolver(role: str):
                base = base_role_resolver(role)
                if base is None or isinstance(base, HardStopClient):
                    return base
                return HardStopClient(base, role=role, task=self._provider_hard_stop)

        else:
            stopped_role_resolver = base_role_resolver
        cost_meter = None
        if self._enable_cost_tracking:
            cost_meter = (
                CostMeter(
                    price_table=dict(self._cost_price_table),
                    runtime_dir=run_dir / ".runtime",
                )
                if self._cost_price_table is not None
                else CostMeter(runtime_dir=run_dir / ".runtime")
            )

            class _RoleResolverShim:
                name = "restored_role_resolver_shim"

                def for_role(self, role: str):
                    return stopped_role_resolver(role)

                def complete(self, *_args: Any, **_kwargs: Any):
                    raise RuntimeError("call for_role() before provider use")

            role_resolver = metered_role_resolver(_RoleResolverShim(), cost_meter)
        else:
            role_resolver = stopped_role_resolver
        return role_resolver, cost_meter, repro_envelope

    @staticmethod
    def _checkpoint_run_path(
        *,
        run_dir: Path,
        relative_path: Any,
        required: bool = True,
    ) -> Optional[Path]:
        """Resolve one checkpoint path without permitting path substitution."""

        if relative_path is None:
            if required:
                raise HumanReviewCheckpointError(
                    "durable human-review checkpoint is missing a required path"
                )
            return None
        text = str(relative_path).strip()
        candidate = (run_dir / text).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise HumanReviewCheckpointError(
                "durable human-review checkpoint path escapes its run directory"
            ) from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise HumanReviewCheckpointError(
                "durable human-review checkpoint references a missing artifact"
            )
        return candidate

    def _record_human_review_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        run_id: str,
        run_dir: Path,
        evidence: EvidenceStore,
    ) -> None:
        """Delegate decision persistence to the checkpoint owner."""
        persist_human_review_records(
            records,
            run_id=run_id,
            run_dir=run_dir,
            evidence=evidence,
            submission_profile_name=self._submission_profile_name,
        )

    @_pipeline_instance_lifecycle("run")
    @exclusive_run_execution
    @_one_capability_job
    def run(
        self,
        *,
        question: Optional[str] = None,
        cohort: Union[str, Path, pd.DataFrame],
        cohort_authority_path: Optional[Union[str, Path]] = None,
        cohort_authority_ref: Optional[
            Union[MaterializedCohortAuthorityRef, Mapping[str, object]]
        ] = None,
        trajectory_path: Optional[Union[str, Path]] = None,
        trajectory_authority_path: Optional[Union[str, Path]] = None,
        trajectory_authority_ref: Optional[
            Union[MaterializedTrajectoryAuthorityRef, Mapping[str, object]]
        ] = None,
        cohort_name: str = "cohort",
        database: str = "miiv",
        target_outcome: Optional[str] = None,
        endpoint: Optional[EndpointSpec] = None,
        primary_exposure: Optional[str] = None,
        cross_database_validation: Optional[Sequence[str]] = None,
        inclusion_criteria: Optional[Sequence[str]] = None,
        exclusion_criteria: Optional[Sequence[str]] = None,
        id_columns: Optional[Sequence[str]] = None,
        time_columns: Optional[Sequence[str]] = None,
        outcome_columns: Optional[Sequence[str]] = None,
        time_windows: Optional[Sequence[TimeWindow]] = None,
        concept_descriptions: Optional[Dict[str, str]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        skill: Optional[Union[str, ClinicalSkill]] = None,
        manuscript_title: Optional[str] = None,
        manuscript_authors: Optional[Sequence[str]] = None,
        manuscript_language: Optional[str] = None,
        resume_run_id: Optional[str] = None,
        resume_from_step_id: Optional[str] = None,
        stop_after_step_id: Optional[str] = None,
        stop_after_analysis: bool = False,
        experiment_spec: Optional[Union[ExperimentSpec, Dict[str, Any]]] = None,
        source_files: Optional[Sequence[Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        force_writer_probe: bool = False,
    ) -> PipelineRunOutcome:
        """Run the explicit Plan → Review → Execute → Write workflow.

        Returns a :class:`PipelineResult` for a run that finished, or a
        :class:`HumanReviewPending` for a run that stopped at the human-review
        gate. The pause is not an error and carries no result: nothing
        downstream of it executed. Answer it with :meth:`resume_human_review`.
        """
        # A requested execution checkpoint is an analysis-only boundary.  The
        # execute coordinator already stops after the selected step; carry the
        # same boundary into Write/Finalise so a checkpoint replay cannot spend
        # Writer calls or overwrite a prior manuscript after execution stops.
        stop_after_analysis = bool(stop_after_analysis or stop_after_step_id)
        skill_obj: Optional[ClinicalSkill] = None
        if skill is not None:
            skill_obj = get_skill(skill) if isinstance(skill, str) else skill
            if question is None:
                question = skill_obj.question_for(database=database)
            if target_outcome is None:
                target_outcome = skill_obj.target_outcome
            if not time_windows:
                time_windows = skill_obj.time_windows or None
        if question is None:
            raise ValueError(
                "`question` is required (or pass `skill=...` to derive one)."
            )
        if self._llm is None:
            raise ValueError(
                "ResearchAgentPipeline.run() now requires an explicit `llm=` "
                "client. Pass MockLLMClient() only for tests or deterministic "
                "demo runs; the pipeline no longer falls back to mock silently."
            )
        if resume_run_id and self._capability_runtime.activation is not None:
            raise ValueError(
                "Approved capability activation requires a new run; resume is forbidden"
            )
        # One instance holds exactly one pause. Starting a second run here used
        # to overwrite ``_pending_human_review`` (on pause) or clear it (on
        # completion), silently destroying a paused run that an operator was
        # still deciding on: its live plan handoff cannot be rebuilt, so the
        # Planner work was simply gone and ``resume_human_review`` reported no
        # pending review at all. Refuse instead of discarding it. Each run also
        # gets a fresh run id, so the run-level file lock cannot catch this.
        blocking_pause = self._pending_human_review
        if blocking_pause:
            blocked_by = blocking_pause["pending"]
            raise RuntimeError(
                f"run {blocked_by.run_id!r} on this pipeline instance is paused "
                "for human review and would be discarded by starting another "
                "run. A pause holds a live plan handoff that cannot be "
                "reconstructed, so it must be answered with "
                "resume_human_review() (or abandoned by using a separate "
                "ResearchAgentPipeline instance for the new run) before this "
                "instance can start again. Pending review ids: "
                + ", ".join(blocked_by.review_ids)
            )
        verified_source_authority = None
        authority_declared = (
            cohort_authority_path is not None or cohort_authority_ref is not None
        )
        if authority_declared and (
            cohort_authority_path is None or cohort_authority_ref is None
        ):
            raise MaterializedMetadataError(
                "cohort authority path and reference must be declared together"
            )
        expected_cohort_authority: Optional[MaterializedCohortAuthorityRef] = None
        if authority_declared:
            if not isinstance(cohort, (str, Path)):
                raise MaterializedMetadataError(
                    "materialized cohort authority requires a parquet path input"
                )
            source_cohort_path = Path(cohort).expanduser()
            if source_cohort_path.suffix.lower() not in {".parquet", ".pq"}:
                raise MaterializedMetadataError(
                    "materialized cohort authority requires a parquet path input"
                )
            expected_cohort_authority = (
                cohort_authority_ref
                if isinstance(cohort_authority_ref, MaterializedCohortAuthorityRef)
                else MaterializedCohortAuthorityRef.from_dict(cohort_authority_ref)
            )
            declared_authority_path = Path(cohort_authority_path).expanduser()
            expected_authority_path = (
                source_cohort_path.parent / expected_cohort_authority.file
            )
            if (
                declared_authority_path.is_symlink()
                or declared_authority_path.resolve()
                != expected_authority_path.resolve()
            ):
                raise MaterializedMetadataError(
                    "cohort authority path does not match the declared reference"
                )
            verified_source_authority = load_verified_materialized_cohort_authority(
                source_cohort_path,
                expected_authority=expected_cohort_authority,
            )
            if verified_source_authority is None:  # pragma: no cover - exact ref
                raise MaterializedMetadataError(
                    "declared typed cohort lost its selected authority"
                )
        elif isinstance(cohort, (str, Path)):
            source_cohort_path = Path(cohort).expanduser()
            if source_cohort_path.suffix.lower() in {".parquet", ".pq"}:
                # Classify and bind typed inputs before scientific identity,
                # cache lookup, resume comparison, or any run-directory write.
                # This keeps the direct Python API safe without requiring every
                # caller to manually rediscover the sibling selector.
                verified_source_authority = load_verified_materialized_cohort_authority(
                    source_cohort_path
                )
                if verified_source_authority is not None:
                    expected_cohort_authority = verified_source_authority.reference
        if verified_source_authority is not None and self._disable_icu_context:
            raise MaterializedMetadataError(
                "Typed materialized cohorts require ICU-aware ResearchContext v2; "
                "disable_icu_context=True is reserved for untyped historical "
                "ablation inputs. Use ICU-aware context/--arms aware, or supply "
                "an untyped DataFrame or legacy parquet for the naive arm."
            )
        if verified_source_authority is not None:
            normalized_database = normalize_database_name(database)
            if verified_source_authority.sidecar.source_database != normalized_database:
                raise MaterializedMetadataError(
                    "declared database does not match typed cohort authority"
                )
            from easyicu.config import load_src_cfg

            expected_prefixes = tuple(
                str(value).strip().lower()
                for value in load_src_cfg(normalized_database).class_prefix
                if str(value).strip()
            )
            if (
                verified_source_authority.sidecar.source_database_class_prefixes
                != expected_prefixes
            ):
                raise MaterializedMetadataError(
                    "typed cohort source class policy does not match host registry"
                )
            # Typed authority owns the actual source database. Preserve the
            # public API's accepted aliases (for example ``mimiciv``) while
            # passing one canonical coordinate into scientific identity,
            # ResearchContext v2, cache, and resume authority.
            database = normalized_database
        verified_source_trajectory: Optional[
            VerifiedMaterializedTrajectoryAuthority
        ] = None
        expected_trajectory_authority: Optional[MaterializedTrajectoryAuthorityRef] = (
            None
        )
        selected_trajectory_path: Optional[Path] = None
        trajectory_authority_declared = (
            trajectory_authority_path is not None
            or trajectory_authority_ref is not None
        )
        if trajectory_authority_declared and (
            trajectory_authority_path is None
            or trajectory_authority_ref is None
            or trajectory_path is None
        ):
            raise MaterializedTrajectoryError(
                "trajectory path, authority path, and authority reference must be "
                "declared together"
            )
        raw_trajectory_path: Optional[Path] = None
        if trajectory_path is not None:
            raw_trajectory_path = Path(trajectory_path).expanduser()
        elif isinstance(cohort, (str, Path)):
            raw_cohort_path = Path(cohort).expanduser()
            candidate = raw_cohort_path.with_name(
                f"{raw_cohort_path.stem}_trajectory.parquet"
            )
            if candidate.exists():
                raw_trajectory_path = candidate
        if raw_trajectory_path is not None:
            if (
                raw_trajectory_path.is_symlink()
                or not raw_trajectory_path.is_file()
                or raw_trajectory_path.suffix.lower() not in {".parquet", ".pq"}
            ):
                raise MaterializedTrajectoryError(
                    "trajectory input must be a regular parquet file"
                )
            selected_trajectory_path = raw_trajectory_path.resolve(strict=True)
        if trajectory_authority_declared:
            if verified_source_authority is None:
                raise MaterializedTrajectoryError(
                    "typed trajectory authority requires a typed cohort authority"
                )
            expected_trajectory_authority = (
                trajectory_authority_ref
                if isinstance(
                    trajectory_authority_ref, MaterializedTrajectoryAuthorityRef
                )
                else MaterializedTrajectoryAuthorityRef.from_dict(
                    trajectory_authority_ref
                )
            )
            assert selected_trajectory_path is not None
            declared_trajectory_authority_path = Path(
                trajectory_authority_path
            ).expanduser()
            expected_trajectory_authority_path = (
                selected_trajectory_path.parent / expected_trajectory_authority.file
            )
            if (
                declared_trajectory_authority_path.is_symlink()
                or declared_trajectory_authority_path.resolve()
                != expected_trajectory_authority_path.resolve()
            ):
                raise MaterializedTrajectoryError(
                    "trajectory authority path does not match the declared reference"
                )
        if (
            selected_trajectory_path is not None
            and verified_source_authority is not None
        ):
            verified_source_trajectory = (
                load_verified_materialized_trajectory_authority(
                    selected_trajectory_path,
                    expected_authority=expected_trajectory_authority,
                    expected_universe_authority=verified_source_authority.reference,
                )
            )
            if verified_source_trajectory is None:
                raise MaterializedTrajectoryError(
                    "typed cohort trajectory requires a sealed trajectory authority"
                )
            expected_trajectory_authority = verified_source_trajectory.reference
        elif expected_trajectory_authority is not None:
            raise MaterializedTrajectoryError(
                "typed trajectory authority requires a typed cohort authority"
            )
        run_language = self._normalise_manuscript_language(
            manuscript_language or self._manuscript_language
        )
        progress_channel = ResumableProgressChannel(progress_callback)
        _emit_progress = progress_channel.emit

        _emit_progress("run", "Starting research-agent run.")

        spec_obj: Optional[ExperimentSpec] = None
        if experiment_spec is not None:
            spec_obj = (
                experiment_spec
                if isinstance(experiment_spec, ExperimentSpec)
                else ExperimentSpec.model_validate(experiment_spec)
            )
        llm = self._llm
        if llm is None:
            raise RuntimeError("LLM client is unexpectedly missing after validation.")
        run_scientific_identity = build_scientific_identity(
            cohort=cohort,
            question=question,
            cohort_name=cohort_name,
            database=database,
            target_outcome=target_outcome,
            endpoint=endpoint,
            primary_exposure=primary_exposure,
            cross_database_validation=cross_database_validation,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            id_columns=id_columns,
            time_columns=time_columns,
            outcome_columns=outcome_columns,
            time_windows=time_windows,
            concept_descriptions=concept_descriptions,
            user_preferences=user_preferences,
            notes=notes,
            skill_key=(skill_obj.key if skill_obj is not None else None),
            experiment_spec=spec_obj,
            source_files=source_files,
            disable_icu_context=self._disable_icu_context,
            development_sampling=(
                {
                    "schema": "easyicu.development_execution_sample/1",
                    "paper_authority": False,
                    "stage": "after_locked_cohort_materialization_and_qc",
                    "algorithm": "sha256_identity_rank_v1",
                    "target_rows": self._development_sample_size,
                    "seed": self._development_sample_seed,
                }
                if self._development_sample_size is not None
                else None
            ),
            materialized_cohort_authority_ref=(
                expected_cohort_authority.to_dict()
                if expected_cohort_authority is not None
                else None
            ),
            trajectory_path=(
                selected_trajectory_path
                if trajectory_path is not None
                or expected_trajectory_authority is not None
                else None
            ),
            materialized_trajectory_authority_ref=(
                expected_trajectory_authority.to_dict()
                if expected_trajectory_authority is not None
                else None
            ),
            capability_workflow=self._capability_runtime.scientific_coordinate(),
        )
        run_environment_identity = build_environment_identity(
            llm_signature=self._llm_signature(llm)
        )

        resume_state: Optional[Dict[str, Any]] = None
        resume_context_evidence_path: Optional[Path] = None
        resume_input_verified = False
        resume_trajectory_binding: Optional[StagedTrajectoryBinding] = None
        experiment_spec_path: Optional[Path] = None
        run_id = current_locked_run_id()
        if resume_run_id:
            run_dir = self.workdir / run_id
            if run_dir.exists():
                resume_state = _load_resume_state(run_dir)
                if resume_state is None and any(run_dir.iterdir()):
                    raise RunInputIdentityError(
                        "Cannot resume safely: the requested run directory is "
                        "non-empty but has no readable checkpoint."
                    )
                if resume_state is not None:
                    if expected_cohort_authority is not None:
                        staged_authority = load_verified_materialized_cohort_authority(
                            run_dir / "cohort.parquet"
                        )
                        if (
                            staged_authority is None
                            or staged_authority.authority.parent_authority_sha256
                            != expected_cohort_authority.sha256
                        ):
                            raise MaterializedMetadataError(
                                "resume cohort no longer descends from the declared "
                                "source authority"
                            )
                        source_authority = verified_source_authority
                        if (
                            source_authority is None
                        ):  # pragma: no cover - declared above
                            raise MaterializedMetadataError(
                                "declared source authority was not verified"
                            )
                        staged_binding = staged_authority.sidecar.files[0]
                        source_binding = source_authority.sidecar.files[0]
                        if (
                            staged_authority.authority.cohort_sha256
                            != source_authority.authority.cohort_sha256
                            or staged_authority.authority.cohort_size
                            != source_authority.authority.cohort_size
                            or staged_authority.authority.cohort_rows
                            != source_authority.authority.cohort_rows
                            or staged_authority.authority.cohort_columns
                            != source_authority.authority.cohort_columns
                            or staged_authority.authority.cohort_schema_sha256
                            != source_authority.authority.cohort_schema_sha256
                            or staged_authority.authority.row_identity_sha256
                            != source_authority.authority.row_identity_sha256
                            or staged_binding.identity_column
                            != source_binding.identity_column
                            or staged_binding.time_coordinates
                            != source_binding.time_coordinates
                            or staged_binding.columns != source_binding.columns
                        ):
                            raise MaterializedMetadataError(
                                "resume cohort is not an exact typed copy of the "
                                "declared source authority"
                            )
                    # Dictionary identity is part of the interrupted run's
                    # provenance authority.  Check the checkpoint-selected
                    # manifest before prepare_existing_resume_input writes a
                    # resume receipt or the plan phase refreshes the mutable
                    # root fingerprint file.  Historical manifests that
                    # predate dictionary fingerprints remain compatible.
                    verify_recorded_dict_match(resume_state, mode="strict")
                    prepared_resume = prepare_existing_resume_input(
                        run_dir=run_dir,
                        resume_state=resume_state,
                        scientific_identity=run_scientific_identity,
                        current_environment=run_environment_identity,
                        cohort=cohort,
                        question=question,
                        resume_from_step_id=resume_from_step_id,
                        enforcement_mode=self._evidence_enforcement_mode,
                        load_compatible_plan=_load_compatible_resume_plan,
                    )
                    resume_state = prepared_resume.resume_state
                    resume_input_verified = prepared_resume.input_verified
                    resume_context_evidence_path = prepared_resume.context_evidence_path
                    resume_trajectory_binding = prepared_resume.trajectory_binding
                    experiment_spec_path = prepared_resume.experiment_spec_path
                    if prepared_resume.cohort_path is not None:
                        cohort_path = prepared_resume.cohort_path
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = self.workdir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
        bind_active_run_heartbeat(
            run_dir,
            task_timeout_seconds=self._heartbeat_wall_clock_remaining(),
        )

        if not resume_input_verified and spec_obj is not None:
            experiment_spec_path = dump_experiment_spec(
                spec_obj,
                run_dir / "experiment_spec.yaml",
            )

        if not resume_input_verified:
            cohort_path = self._materialise_cohort(
                cohort,
                run_dir,
                expected_source_authority=expected_cohort_authority,
            )
            _emit_progress(
                "cohort",
                "Cohort materialised to parquet.",
                run_id=run_id,
                path=str(cohort_path),
            )
            staged_trajectory_binding: Optional[StagedTrajectoryBinding] = None
            if selected_trajectory_path is not None:
                trajectory_identity = run_scientific_identity.get("trajectory")
                if not isinstance(trajectory_identity, Mapping):
                    cohort_identity = run_scientific_identity.get("cohort")
                    trajectory_identity = (
                        cohort_identity.get("trajectory")
                        if isinstance(cohort_identity, Mapping)
                        else None
                    )
                if not isinstance(trajectory_identity, Mapping):
                    raise MaterializedTrajectoryError(
                        "trajectory scientific identity is missing"
                    )
                staged_trajectory_binding = self._materialise_trajectory(
                    source_path=selected_trajectory_path,
                    target_path=run_dir / "cohort_trajectory.parquet",
                    source_cohort_path=(
                        Path(cohort).expanduser().resolve()
                        if verified_source_trajectory is not None
                        and isinstance(cohort, (str, Path))
                        else None
                    ),
                    target_cohort_path=cohort_path,
                    source_authority=verified_source_trajectory,
                    expected_sha256=str(trajectory_identity.get("sha256") or ""),
                    expected_size=int(trajectory_identity.get("size_bytes") or -1),
                )
                _emit_progress(
                    "cohort",
                    "Trajectory staged with exact input authority.",
                    run_id=run_id,
                    path=str(staged_trajectory_binding.path),
                )
        else:
            staged_trajectory_binding = resume_trajectory_binding
            _emit_progress(
                "cohort",
                "Verified immutable staged cohort for resume.",
                run_id=run_id,
                path=str(cohort_path),
            )
        audit_logger = AuditLogger(run_dir / "audit_log.jsonl")
        progress_channel.bind_audit_logger(audit_logger)

        cache_key: Optional[str] = None
        if self._enable_cache and not resume_run_id:
            cache_key = self._cache.compute_key(
                cohort_path=cohort_path,
                question=question,
                target_outcome=target_outcome,
                skill_key=(skill_obj.key if skill_obj is not None else None),
                database=database,
                llm=self._llm,
                stop_after_analysis=stop_after_analysis,
                manuscript_language=run_language,
                flags=self._cache_flag_payload(),
                science_inputs=run_scientific_identity,
            )
            cached = self._cache.lookup(
                cache_key,
                scientific_identity=run_scientific_identity,
            )
            if cached is not None:
                shutil.rmtree(run_dir, ignore_errors=True)
                _emit_progress(
                    "cache",
                    f"Reused cached run {cached.run_id}.",
                    status="complete",
                    run_id=cached.run_id,
                )
                return cached

        if self._config.planner_only:
            self._clear_validated_runtime()
            runtime_capabilities = ()
            _emit_progress(
                "runtime",
                "Execution runtime preflight skipped for planner-only authority.",
                run_id=run_id,
                method_capabilities=[],
            )
        else:
            runtime_capabilities = self._preflight_execution_runtime(
                run_dir=run_dir, cohort_path=cohort_path, target_outcome=target_outcome
            )
            _emit_progress(
                "runtime",
                "Execution runtime validated before planning.",
                run_id=run_id,
                method_capabilities=list(runtime_capabilities),
            )

        def _plan_invoker():
            return _pipeline_run___plan_invoker(
                _emit_progress=_emit_progress,
                cohort_name=cohort_name,
                cohort_path=cohort_path,
                concept_descriptions=concept_descriptions,
                cross_database_validation=cross_database_validation,
                database=database,
                endpoint=endpoint,
                exclusion_criteria=exclusion_criteria,
                experiment_spec_path=experiment_spec_path,
                id_columns=id_columns,
                inclusion_criteria=inclusion_criteria,
                llm=llm,
                notes=notes,
                outcome_columns=outcome_columns,
                primary_exposure=primary_exposure,
                question=question,
                resume_context_evidence_path=resume_context_evidence_path,
                resume_from_step_id=resume_from_step_id,
                resume_state=resume_state,
                run_dir=run_dir,
                run_environment_identity=run_environment_identity,
                run_id=run_id,
                run_language=run_language,
                run_scientific_identity=run_scientific_identity,
                self=self,
                skill_obj=skill_obj,
                staged_trajectory_binding=staged_trajectory_binding,
                target_outcome=target_outcome,
                time_columns=time_columns,
                time_windows=time_windows,
                user_preferences=user_preferences,
            )

        from .orchestration.workflow import orchestration_runtime_receipt

        orchestration_receipt = orchestration_runtime_receipt()
        orchestration_receipt_path = run_dir / "orchestration_runtime.json"
        orchestration_receipt_path.write_text(
            orchestration_receipt.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

        def _provenance_hook(plan_result):
            return _pipeline_run___provenance_hook(
                plan_result,
                cohort_path=cohort_path,
                orchestration_receipt_path=orchestration_receipt_path,
                run_dir=run_dir,
                self=self,
                source_files=source_files,
            )

        def _execute_invoker(plan_result):
            return _pipeline_run___execute_invoker(
                plan_result,
                _emit_progress=_emit_progress,
                cohort_path=cohort_path,
                notes=notes,
                resume_from_step_id=resume_from_step_id,
                run_dir=run_dir,
                run_id=run_id,
                self=self,
                skill_obj=skill_obj,
                staged_trajectory_binding=staged_trajectory_binding,
                stop_after_step_id=stop_after_step_id,
            )

        def _write_invoker(plan_result, execute_result):
            return _pipeline_run___write_invoker(
                plan_result,
                execute_result,
                _emit_progress=_emit_progress,
                force_writer_probe=force_writer_probe,
                manuscript_authors=manuscript_authors,
                manuscript_title=manuscript_title,
                run_dir=run_dir,
                run_id=run_id,
                run_language=run_language,
                self=self,
                stop_after_analysis=stop_after_analysis,
            )

        def _finalise_invoker(plan_result, execute_result, write_result):
            return _pipeline_run___finalise_invoker(
                plan_result,
                execute_result,
                write_result,
                _emit_progress=_emit_progress,
                audit_logger=audit_logger,
                cache_key=cache_key,
                cohort_path=cohort_path,
                database=database,
                experiment_spec_path=experiment_spec_path,
                notes=notes,
                run_dir=run_dir,
                run_id=run_id,
                run_scientific_identity=run_scientific_identity,
                self=self,
                stop_after_analysis=stop_after_analysis,
                target_outcome=target_outcome,
            )

        # The recorder needs this run's own EvidenceStore, which lives on the
        # plan result rather than in ``run()``'s locals. Keep the live handoff
        # for the supported same-process resume; the defensive reopen also
        # makes direct recorder diagnostics fail closed instead of indexing an
        # empty list.
        reviewed_plan: List[Any] = []

        def _review_evidence_store():
            return _pipeline_run___review_evidence_store(
                reviewed_plan=reviewed_plan, run_dir=run_dir, self=self
            )

        def _human_review_invoker(plan_result):
            return _pipeline_run___human_review_invoker(
                plan_result, reviewed_plan=reviewed_plan, self=self
            )

        def _human_review_recorder(records):
            return _pipeline_run___human_review_recorder(
                records,
                _review_evidence_store=_review_evidence_store,
                run_dir=run_dir,
                run_id=run_id,
                self=self,
            )

        gate = self._human_review_gate
        from .orchestration.workflow import WorkflowPaused, build_pipeline_workflow

        checkpoint_commit: Dict[str, Any] = {
            "path": None,
            "decision_sha256": None,
            "decision_payloads": None,
        }

        def _prepare_human_review_execution(
            decision_records: Sequence[Mapping[str, Any]],
        ):
            return _pipeline_run___prepare_human_review_execution(
                decision_records, checkpoint_commit=checkpoint_commit
            )

        def _commit_human_review_execution(
            decision_records: Sequence[Mapping[str, Any]],
        ):
            return _pipeline_run___commit_human_review_execution(
                decision_records,
                checkpoint_commit=checkpoint_commit,
                reviewed_plan=reviewed_plan,
                run_dir=run_dir,
            )

        def _commit_human_review_execution_start():
            return _pipeline_run___commit_human_review_execution_start(
                checkpoint_commit=checkpoint_commit
            )

        def _commit_human_review_write_start():
            return _pipeline_run___commit_human_review_write_start(
                checkpoint_commit=checkpoint_commit
            )

        def _commit_human_review_finalize_start():
            return _pipeline_run___commit_human_review_finalize_start(
                checkpoint_commit=checkpoint_commit
            )

        workflow = build_pipeline_workflow(
            plan_invoker=_plan_invoker,
            execute_invoker=_execute_invoker,
            write_invoker=_write_invoker,
            finalise_invoker=_finalise_invoker,
            provenance_hook=_provenance_hook,
            human_review_invoker=_human_review_invoker,
            human_review_recorder=_human_review_recorder,
            human_review_decision_prepare=_prepare_human_review_execution,
            human_review_execution_commit=_commit_human_review_execution,
            human_review_execution_start=_commit_human_review_execution_start,
            human_review_write_start=_commit_human_review_write_start,
            human_review_finalize_start=_commit_human_review_finalize_start,
            reviewer_identity_resolver=(
                getattr(gate, "reviewer_identity_resolver", None)
                if gate is not None
                else None
            ),
        )
        outcome = workflow.start()
        durable_checkpoint = None
        if isinstance(outcome, WorkflowPaused):
            if not reviewed_plan:
                raise HumanReviewCheckpointError(
                    "workflow paused without a typed plan handoff"
                )
            durable_checkpoint = self._persist_review_checkpoint(
                plan_result=reviewed_plan[-1],
                requests=outcome.requests,
                run_id=run_id,
                run_dir=run_dir,
                cohort_path=cohort_path,
                trajectory_binding=staged_trajectory_binding,
                runtime_capabilities=runtime_capabilities,
                runtime_bundle=self._validated_runtime_bundle,
                run_environment_identity=run_environment_identity,
                notes=notes,
                database=database,
                target_outcome=target_outcome,
                stop_after_step_id=stop_after_step_id,
                stop_after_analysis=stop_after_analysis,
                manuscript_title=manuscript_title,
                manuscript_authors=manuscript_authors,
                run_language=run_language,
                force_writer_probe=force_writer_probe,
                scientific_identity=run_scientific_identity,
                experiment_spec_path=experiment_spec_path,
                cache_key=cache_key,
                skill_obj=skill_obj,
            )
            if durable_checkpoint is not None:
                checkpoint_commit["path"] = str(human_review_checkpoint_path(run_dir))
                if self._provider_hard_stop is not None:
                    self._provider_hard_stop.pause()
        return self._pipeline_result_or_pending(
            outcome,
            workflow=workflow,
            run_id=run_id,
            run_dir=run_dir,
            progress_channel=progress_channel,
            durable_checkpoint=durable_checkpoint,
            checkpoint_commit=checkpoint_commit,
        )

    def _pipeline_result_or_pending(
        self,
        outcome: Any,
        *,
        workflow: Any,
        run_id: str,
        run_dir: Path,
        progress_channel: Optional[ResumableProgressChannel] = None,
        durable_checkpoint: Optional[HumanReviewCheckpoint] = None,
        checkpoint_commit: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Return the run's result, or the typed pause that replaced it.

        A run that stopped for human review has no ``PipelineResult`` because
        nothing downstream of the pause executed.
        """

        from .orchestration.workflow import (
            HumanReviewPending,
            WorkflowCompleted,
            WorkflowPaused,
        )

        if isinstance(outcome, WorkflowCompleted):
            self._pending_human_review = None
            if checkpoint_commit and checkpoint_commit.get("path"):
                path = Path(str(checkpoint_commit["path"]))
                checkpoint = load_human_review_checkpoint(path, require_pending=False)
                if checkpoint.state in {
                    "executing",
                    "write_in_progress",
                    "finalize_in_progress",
                    "consumed",
                }:
                    write_human_review_checkpoint(
                        path, checkpoint.transitioned("completed")
                    )
            return outcome.final_result

        if not isinstance(outcome, WorkflowPaused):
            raise RuntimeError(
                "the pipeline workflow returned neither a completed result nor "
                f"a human-review pause (outcome={type(outcome).__name__})"
            )
        pending = HumanReviewPending(
            run_id=run_id,
            thread_id=run_id,
            run_dir=str(run_dir),
            requests=outcome.requests,
            resume_scope=(
                "durable_checkpoint"
                if durable_checkpoint is not None
                else "same_process"
            ),
            resume_pid=(None if durable_checkpoint is not None else os.getpid()),
        )
        # Held so ``resume_human_review`` can drive the same state machine: its
        # invokers close over this run's evidence store, run dir and services.
        self._pending_human_review = {
            "workflow": workflow,
            "pending": pending,
            # Captured here, not read off the instance at resume time. A second
            # run on the same pipeline overwrites
            # ``_validated_runtime_capabilities`` during its own preflight, so
            # a review approved against image A could otherwise be resumed
            # under image B's allow-list — the environment the reviewer signed
            # off is not the one that would finish the analysis. A tuple of
            # import names, copied, not a provider callable.
            # ``getattr`` rather than attribute access: this method is reached
            # by test doubles that never ran ``__init__``, and a pause must not
            # start failing because the capability fields are absent.
            "runtime_capabilities": tuple(
                getattr(self, "_validated_runtime_capabilities", None) or ()
            ),
            "runtime_bundle": deepcopy(
                getattr(self, "_validated_runtime_bundle", None)
            ),
            # Mutable indirection owned by this paused run.  Resume may replace
            # only the transport callback; the workflow, plan and scientific
            # authority remain the exact objects reviewed by the operator.
            "progress_sink": progress_channel,
            "checkpoint_commit": checkpoint_commit,
        }
        return pending

    @property
    def has_resumable_human_review(self) -> bool:
        """Whether the live workflow still owns an answerable review pause."""

        state = self._pending_human_review
        if not isinstance(state, Mapping):
            return False
        workflow = state.get("workflow")
        pending = state.get("pending")
        return bool(
            getattr(workflow, "state", None) == "paused"
            and getattr(pending, "resumable_here", False)
        )

    @_pipeline_instance_lifecycle("resume")
    @_one_capability_job
    def resume_human_review(
        self,
        decisions: Sequence[Union[Any, Mapping[str, Any]]],
        *,
        run_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Any:
        """Answer the review that paused :meth:`run` and finish the run.

        ``decisions`` must contain exactly one entry per paused request, each
        carrying the request's own ``authority_sha256`` — the workflow rejects
        a decision that does not bind the request it claims to answer, so an
        approval cannot be replayed against a different pause.

        A fresh Plan-phase pause is also persisted as a digest-bound checkpoint.
        When ``run_id`` names that checkpoint, a new pipeline instance may
        reconstruct the reviewed handoff without invoking Planner again.
        """

        from .orchestration.workflow import (
            HUMAN_REVIEW_RESUME_SCOPE,
            HumanReviewRejected,
        )
        from .authority.provider_hard_stop import ProviderHardStopExceeded

        all_rejected = bool(decisions) and all(
            str(
                item.get("decision")
                if isinstance(item, Mapping)
                else getattr(item, "decision", "")
            )
            == "rejected"
            for item in decisions
        )
        if self._config.planner_only and not all_rejected:
            raise RuntimeError(
                "this pipeline authority is planner-only and cannot resume into Execute"
            )

        pending_state = self._pending_human_review
        if not pending_state and run_id is not None:
            pending_state = restore_durable_human_review_pause(
                self,
                run_id=str(run_id),
                progress_callback=progress_callback,
                plan_result_factory=_PlanPhaseResult,
                load_resume_state=_load_resume_state,
                rejection_only=all_rejected,
            )
        if not pending_state:
            raise RuntimeError(
                "no human review is pending on this pipeline instance. "
                "resume_human_review() requires either the live pause returned "
                "by run() or its exact durable checkpoint coordinate "
                f"(legacy scope {HUMAN_REVIEW_RESUME_SCOPE!r})."
            )
        pending = pending_state["pending"]
        if not pending.resumable_here:
            raise RuntimeError(
                f"human review for run {pending.run_id!r} was paused in process "
                f"{pending.resume_pid} and cannot be answered from process "
                f"{os.getpid()}"
            )
        if run_id is not None and str(run_id) != pending.run_id:
            raise RuntimeError(
                f"pending human review belongs to run {pending.run_id!r}, "
                f"not {str(run_id)!r}"
            )
        # The pause ended the run's capability scope, so the provider the
        # runner published is gone. Republish the snapshot captured *at the
        # pause* rather than whatever the instance holds now: a second run on
        # this pipeline overwrites the instance field during its own preflight,
        # and resuming under another image's allow-list would finish the
        # analysis in an environment the reviewer never saw.
        resumed_snapshot = tuple(pending_state.get("runtime_capabilities") or ())
        if resumed_snapshot:
            set_runtime_capability_snapshot_provider(lambda: resumed_snapshot)

        payload = [
            item if isinstance(item, Mapping) else item.model_dump(mode="json")
            for item in decisions
        ]
        checkpoint_commit = pending_state.get("checkpoint_commit")
        if isinstance(checkpoint_commit, dict) and checkpoint_commit.get("path"):
            bind_checkpoint_decision_payloads(
                checkpoint_commit, requests=pending.requests, payload=payload
            )
        # Same writer lease ``run`` holds, bound to the paused run's own id
        # rather than a fresh one: resuming writes into that run's directory
        # and evidence store, so it must not proceed while another call is
        # writing there. ``run`` returns when it pauses, releasing its lease,
        # which is exactly why resume has to take one of its own.
        workflow = pending_state["workflow"]
        progress_channel = pending_state.get("progress_sink")
        if progress_callback is not None:
            if isinstance(progress_channel, ResumableProgressChannel):
                progress_channel.replace_callback(progress_callback)
            elif isinstance(progress_channel, dict):
                # Same-process pauses created before a hot reload retain the
                # historical mutable callback sink.
                progress_channel["callback"] = progress_callback
        provider_resumed = False
        review_checkpoint_at: Optional[str] = None
        try:
            with acquire_run_execution_lock(
                workdir=Path(self.workdir), run_id=pending.run_id
            ):
                if self._provider_hard_stop is not None and not all_rejected:
                    # A process may die after reopening the Provider clock but
                    # before the decision commit changes a still-pending review
                    # checkpoint. Converge that exact window back to the durable
                    # pause before reopening it; paused tasks make this a no-op.
                    if isinstance(checkpoint_commit, dict):
                        checkpoint_path = checkpoint_commit.get("path")
                        reconcile_pause = getattr(
                            self._provider_hard_stop,
                            "reconcile_review_pause",
                            None,
                        )
                        if checkpoint_path and callable(reconcile_pause):
                            checkpoint = load_human_review_checkpoint(
                                Path(str(checkpoint_path)), require_pending=False
                            )
                            if checkpoint.state == "pending":
                                review_checkpoint_at = checkpoint.created_at
                                reconcile_pause(paused_at=review_checkpoint_at)
                    self._provider_hard_stop.resume()
                    provider_resumed = True
                with run_heartbeat_scope(run_id=pending.run_id):
                    bind_active_run_heartbeat(
                        Path(pending.run_dir),
                        task_timeout_seconds=(
                            None
                            if all_rejected
                            else self._heartbeat_wall_clock_remaining()
                        ),
                    )
                    outcome = workflow.resume(payload)
                return self._pipeline_result_or_pending(
                    outcome,
                    workflow=workflow,
                    run_id=pending.run_id,
                    run_dir=Path(pending.run_dir),
                    progress_channel=(
                        progress_channel
                        if isinstance(progress_channel, ResumableProgressChannel)
                        else None
                    ),
                    checkpoint_commit=(
                        checkpoint_commit
                        if isinstance(checkpoint_commit, dict)
                        else None
                    ),
                )
        except HumanReviewRejected:
            # The workflow has recorded the rejection and discarded its live
            # handoff. Do not keep presenting the public pipeline pause as
            # answerable or allow a later approval attempt against it.
            if isinstance(checkpoint_commit, dict):
                fail_human_review_checkpoint(checkpoint_commit)
            self._pending_human_review = None
            raise
        except Exception as exc:
            # Validation failures leave the workflow paused so the caller can
            # correct and resubmit the exact decision set. Once execution,
            # writing or finalisation terminalises the workflow, however, the
            # live handoff is no longer resumable and must not be retained.
            if isinstance(exc, ProviderHardStopExceeded) or getattr(
                workflow, "state", None
            ) in {"failed", "rejected", "completed"}:
                if isinstance(checkpoint_commit, dict):
                    fail_human_review_checkpoint(checkpoint_commit)
                self._pending_human_review = None
            elif provider_resumed and self._provider_hard_stop is not None:
                reconcile_pause = getattr(
                    self._provider_hard_stop,
                    "reconcile_review_pause",
                    None,
                )
                if review_checkpoint_at and callable(reconcile_pause):
                    reconcile_pause(paused_at=review_checkpoint_at)
                else:
                    self._provider_hard_stop.pause()
            raise

    def run_from_spec(
        self,
        spec: Union[ExperimentSpec, Dict[str, Any]],
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PipelineRunOutcome:
        """Run the pipeline from a typed YAML/JSON experiment specification.

        Same two outcomes as :meth:`run`, which this delegates to.
        """
        spec_obj = (
            spec
            if isinstance(spec, ExperimentSpec)
            else ExperimentSpec.model_validate(spec)
        )
        kwargs = spec_obj.run_kwargs()
        kwargs["experiment_spec"] = spec_obj
        kwargs["progress_callback"] = progress_callback
        return self.run(**kwargs)

    async def run_async(self, **kwargs: Any) -> PipelineRunOutcome:
        """Async wrapper for UI/API runtimes that need non-blocking orchestration."""
        return await asyncio.to_thread(self.run, **kwargs)

    def run_with_graph(self, **kwargs: Any) -> PipelineRunOutcome:
        """Deprecated alias retained for EasyICU 1.x callers."""
        warnings.warn(
            "run_with_graph() is deprecated; run() uses the sole explicit "
            "EasyICU workflow.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run(**kwargs)

    def replicate(
        self,
        *,
        cohorts: Dict[str, Union[str, Path, pd.DataFrame]],
        question: Optional[str] = None,
        target_outcome: Optional[str] = None,
        cohort_name_prefix: str = "cohort",
        skill: Optional[Union[str, ClinicalSkill]] = None,
        stop_after_analysis: bool = True,
        inclusion_criteria: Optional[Sequence[str]] = None,
        exclusion_criteria: Optional[Sequence[str]] = None,
        id_columns: Optional[Sequence[str]] = None,
        time_columns: Optional[Sequence[str]] = None,
        outcome_columns: Optional[Sequence[str]] = None,
        time_windows: Optional[Sequence[TimeWindow]] = None,
        concept_descriptions: Optional[Dict[str, str]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        manuscript_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the same plan/question across multiple cohorts and compare effects.

        This is the pipeline-native cross-database replication path requested
        by the methods critique: instead of pushing replication into a separate
        CLI layer, the orchestrator can now launch one analysis run per cohort
        and emit a harmonised effect-comparison table.
        """
        if not cohorts:
            raise ValueError(
                "`cohorts` must contain at least one database -> cohort entry."
            )

        run_results: Dict[str, PipelineResult] = {}
        comparison_rows: List[Dict[str, Any]] = []
        databases = list(cohorts.keys())
        for database, cohort in cohorts.items():
            result = self.run(
                question=question,
                cohort=cohort,
                cohort_name=f"{database}_{cohort_name_prefix}",
                database=database,
                target_outcome=target_outcome,
                cross_database_validation=[db for db in databases if db != database],
                inclusion_criteria=inclusion_criteria,
                exclusion_criteria=exclusion_criteria,
                id_columns=id_columns,
                time_columns=time_columns,
                outcome_columns=outcome_columns,
                time_windows=time_windows,
                concept_descriptions=concept_descriptions,
                user_preferences=user_preferences,
                notes=notes,
                skill=skill,
                manuscript_language=manuscript_language,
                stop_after_analysis=stop_after_analysis,
            )
            run_results[database] = result
            comparison_rows.append(
                _extract_primary_effect_row(database=database, result=result)
            )

        replication_id = "replication_" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S"
        )
        replication_dir = self.workdir / replication_id
        replication_dir.mkdir(parents=True, exist_ok=True)

        comparison_df = pd.DataFrame(comparison_rows)
        csv_path = replication_dir / "cross_database_effect_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        md_path = replication_dir / "cross_database_effect_comparison.md"
        md_path.write_text(
            _render_cross_database_comparison_markdown(comparison_rows),
            encoding="utf-8",
        )
        run_summaries = [
            _extract_cross_database_run_summary(database=db, result=res)
            for db, res in run_results.items()
        ]
        summary_df = pd.DataFrame(run_summaries)
        summary_csv_path = replication_dir / "cross_database_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        summary_md_path = replication_dir / "cross_database_summary.md"
        summary_md_path.write_text(
            _render_cross_database_summary_markdown(run_summaries),
            encoding="utf-8",
        )
        validation_report_path = replication_dir / "cross_database_validation_report.md"
        validation_report_path.write_text(
            _render_cross_database_validation_report(
                question=question,
                target_outcome=target_outcome,
                rows=comparison_rows,
                run_summaries=run_summaries,
            ),
            encoding="utf-8",
        )
        summary_path = replication_dir / "cross_database_runs.json"
        summary_path.write_text(
            json.dumps(
                {
                    "question": question,
                    "target_outcome": target_outcome,
                    "stop_after_analysis": stop_after_analysis,
                    "comparison_csv": str(csv_path),
                    "summary_csv": str(summary_csv_path),
                    "validation_report": str(validation_report_path),
                    "runs": {
                        db: {
                            "run_id": res.run_id,
                            "manifest_path": res.manifest_path,
                            "report_path": res.report_path,
                            "manuscript_path": res.manuscript_path,
                        }
                        for db, res in run_results.items()
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "replication_id": replication_id,
            "replication_dir": str(replication_dir),
            "comparison_csv": str(csv_path),
            "comparison_md": str(md_path),
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
            "validation_report": str(validation_report_path),
            "summary_json": str(summary_path),
            "runs": run_results,
        }

    async def replicate_async(self, **kwargs: Any) -> Dict[str, Any]:
        """Async wrapper for cross-database replication."""
        return await asyncio.to_thread(self.replicate, **kwargs)

    def reproduce_paper(
        self,
        *,
        paper: Union[str, Path],
        cohort: Union[str, Path, pd.DataFrame],
        database: str,
        mode: str = "replication",
        cohort_name: str = "paper_replication_cohort",
        manuscript_language: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        """Run the paper-reproduction dual-track workflow."""
        paper_profile = parse_paper_profile(paper)
        replication_spec, deviation_report = build_paper_replication_spec(paper_profile)
        if not deviation_report.supported:
            return self._write_fail_closed_paper_package(
                paper=paper,
                cohort=cohort,
                database=database,
                cohort_name=cohort_name,
                paper_profile=paper_profile,
                replication_spec=replication_spec,
                deviation_report=deviation_report,
            )

        target_outcome = (
            replication_spec.mapped_concepts.get("target_outcome")
            or replication_spec.mapped_concepts.get("outcome")
            or canonical_outcome_name(paper_profile.target_outcome)
        )
        question = paper_profile.research_question or (
            f"Replicate the published ICU study '{paper_profile.paper_title or paper_profile.paper_source}' "
            "using EasyICU with design-and-conclusion alignment."
        )
        notes = _build_replication_notes(
            paper_profile=paper_profile,
            replication_spec=replication_spec,
            mode=mode,
        )
        source_files: List[Any] = []
        raw_paper = str(paper)
        if "\n" not in raw_paper and len(raw_paper) <= 240:
            paper_candidate = Path(raw_paper).expanduser()
            try:
                if paper_candidate.exists():
                    source_files.append(paper_candidate)
            except OSError:
                pass
        result = self.run(
            question=question,
            cohort=cohort,
            cohort_name=cohort_name,
            database=database,
            target_outcome=target_outcome,
            notes=notes,
            manuscript_language=manuscript_language,
            progress_callback=progress_callback,
            source_files=source_files or None,
            user_preferences={
                "inferred_analysis_family": paper_profile.paper_type,
                "timing_and_design": "paper_replication",
                "must_have_outputs": ", ".join(replication_spec.required_outputs),
                "covariates": paper_profile.covariates,
                "extra_notes": (
                    "Replication mode: use EasyICU actual results only; if the original paper "
                    "is referenced, frame it as 'original paper reported ...'."
                ),
            },
            stop_after_analysis=(mode == "replication"),
        )
        return self._postprocess_paper_replication(
            result=result,
            paper_profile=paper_profile,
            replication_spec=replication_spec,
            deviation_report=deviation_report,
            mode=mode,
        )

    def compare_with_paper(
        self,
        *,
        paper: Union[str, Path],
        result: Optional[PipelineResult] = None,
        run_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Return comparison rows between a parsed paper and one EasyICU run."""
        if result is None and run_dir is None:
            raise ValueError("Provide either `result` or `run_dir`.")
        paper_profile = parse_paper_profile(paper)
        actual_run_dir = Path(
            result.workdir if result is not None else run_dir
        ).resolve()
        manifest = json.loads(
            (actual_run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        context_payload: Optional[Dict[str, Any]] = None
        context_rel = manifest.get("context_path")
        if context_rel:
            context_path = actual_run_dir / str(context_rel)
            if context_path.exists():
                context_payload = json.loads(context_path.read_text(encoding="utf-8"))
        ledger = build_paper_result_ledger(
            paper_profile=paper_profile,
            manifest=manifest,
            context_payload=context_payload,
        )
        rows = compare_paper_to_easyicu(
            paper_profile=paper_profile,
            ledger=ledger,
        )
        return {"paper_profile": paper_profile, "ledger": ledger, "rows": rows}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_manuscript_language(language: str) -> str:
        lang = (language or "en").lower()
        return "zh" if lang.startswith(("zh", "cn", "chinese")) else "en"

    def _materialise_cohort(
        self,
        cohort: Union[str, Path, pd.DataFrame],
        run_dir: Path,
        *,
        expected_source_authority: Optional[MaterializedCohortAuthorityRef] = None,
    ) -> Path:
        if isinstance(cohort, (str, Path)):
            src = Path(cohort).resolve()
            if not src.exists():
                raise FileNotFoundError(f"Cohort path not found: {src}")
            target = run_dir / "cohort.parquet"
            if src.suffix.lower() in {".parquet", ".pq"}:
                typed_source = load_verified_materialized_cohort_authority(
                    src,
                    expected_authority=expected_source_authority,
                )
                staged = stage_materialized_cohort_authority(
                    src,
                    target,
                    producer_implementation_sha256=(
                        materialized_implementation_bundle_sha256(
                            (
                                Path(__file__),
                                Path(__file__).resolve().parent
                                / "intake"
                                / "materialized_metadata.py",
                            )
                        )
                    ),
                    expected_source_authority=expected_source_authority,
                )
                if staged is None and src.resolve() != target.resolve():
                    # Legacy materializations bind provenance to the exact
                    # Parquet bytes. Re-serializing an otherwise identical
                    # frame changes that file authority and needlessly loads
                    # the full cohort into memory, so stage it byte-for-byte.
                    shutil.copy2(src, target)
                    source_provenance = materialized_provenance_path(src)
                    if source_provenance.is_file():
                        shutil.copy2(
                            source_provenance,
                            materialized_provenance_path(target),
                        )
            elif src.suffix.lower() in {".csv", ".tsv"}:
                if expected_source_authority is not None:
                    raise MaterializedMetadataError(
                        "materialized cohort authority cannot bind CSV/TSV input"
                    )
                sep = "\t" if src.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(src, sep=sep)
                df.to_parquet(target, index=False)
            else:
                raise ValueError(
                    f"Unsupported cohort file extension: {src.suffix}. "
                    "Use .parquet, .csv or pass a DataFrame."
                )
            return target
        if isinstance(cohort, pd.DataFrame):
            if expected_source_authority is not None:
                raise MaterializedMetadataError(
                    "materialized cohort authority cannot bind DataFrame input"
                )
            target = run_dir / "cohort.parquet"
            cohort.to_parquet(target, index=False)
            return target
        raise TypeError("cohort must be a path or a pandas DataFrame")

    def _materialise_trajectory(
        self,
        *,
        source_path: Path,
        target_path: Path,
        source_cohort_path: Optional[Path],
        target_cohort_path: Path,
        source_authority: Optional[VerifiedMaterializedTrajectoryAuthority],
        expected_sha256: str,
        expected_size: int,
    ) -> StagedTrajectoryBinding:
        """Stage a trajectory exactly; typed inputs also rebind cohort lineage."""

        if source_authority is None:
            staged_path = stage_legacy_trajectory_exact(
                source_path,
                target_path,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
            return StagedTrajectoryBinding(
                path=staged_path,
                sha256=expected_sha256,
                size=expected_size,
            )
        if source_cohort_path is None:
            raise MaterializedTrajectoryError(
                "typed trajectory requires its source cohort path"
            )
        target_cohort = load_verified_materialized_cohort_authority(target_cohort_path)
        if target_cohort is None:
            raise MaterializedTrajectoryError(
                "typed trajectory target cohort lost its authority"
            )
        staged = stage_materialized_trajectory_authority(
            source_path,
            target_path,
            source_universe_path=source_cohort_path,
            target_universe_path=target_cohort_path,
            expected_source_authority=source_authority.reference,
            expected_target_universe_authority=target_cohort.reference,
            producer_implementation_sha256=(
                materialized_implementation_bundle_sha256(
                    (
                        Path(__file__),
                        Path(__file__).resolve().parent
                        / "intake"
                        / "materialized_trajectory.py",
                    )
                )
            ),
        )
        return StagedTrajectoryBinding(
            path=target_path,
            sha256=staged.authority.trajectory_sha256,
            size=staged.authority.trajectory_size,
            authority_ref=staged.reference,
        )

    def _postprocess_paper_replication(
        self,
        *,
        result: PipelineResult,
        paper_profile: PaperProfile,
        replication_spec: PaperReplicationSpec,
        deviation_report: ReplicationDeviationReport,
        mode: str,
    ) -> PipelineResult:
        """Thin delegate; real logic lives in :func:`replication.paper.postprocess_paper_replication`."""
        return postprocess_paper_replication(
            result=result,
            paper_profile=paper_profile,
            replication_spec=replication_spec,
            deviation_report=deviation_report,
            mode=mode,
        )

    def _write_fail_closed_paper_package(
        self,
        *,
        paper: Union[str, Path],
        cohort: Union[str, Path, pd.DataFrame],
        database: str,
        cohort_name: str,
        paper_profile: PaperProfile,
        replication_spec: PaperReplicationSpec,
        deviation_report: ReplicationDeviationReport,
    ) -> PipelineResult:
        """Thin delegate; real logic lives in
        :func:`replication.paper.write_fail_closed_paper_package`."""
        return write_fail_closed_paper_package(
            workdir=self.workdir,
            llm=self._llm,
            materialise_cohort=self._materialise_cohort,
            paper=paper,
            cohort=cohort,
            database=database,
            cohort_name=cohort_name,
            paper_profile=paper_profile,
            replication_spec=replication_spec,
            deviation_report=deviation_report,
        )

    # ------------------------------------------------------------------
    # T3.5 — cohort cache (delegates; real logic in authority/pipeline_cache.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _llm_signature(llm: Any) -> str:
        return _pipeline_cache.llm_signature(llm)

    @staticmethod
    def _iter_mock_clients(llm: Any):
        yield from _pipeline_cache.iter_mock_clients(llm)

    def _heartbeat_wall_clock_remaining(self) -> Optional[float]:
        """Return the live task window rather than restarting it on resume."""

        if self._provider_hard_stop is not None:
            return self._provider_hard_stop.assert_active()
        return self._config.max_wall_clock_seconds_per_task

    def _cache_flag_payload(self) -> Dict[str, Any]:
        """Return the bag of pipeline-level flags that participate in
        the cache key. Kept here (and not in :mod:`pipeline_cache`)
        because the canonical list of flags is owned by the pipeline.
        """
        return {
            "disable_icu_context": bool(self._disable_icu_context),
            "enable_pubmed": bool(self._enable_pubmed),
            "enable_tavily": bool(self._enable_tavily),
            "context_top_k": self._context_top_k,
            "enable_llm_concept_audit": bool(self._enable_llm_concept_audit),
            "enable_probe_step": bool(self._enable_probe_step),
            "enable_replanning": bool(self._enable_replanning),
            "max_code_repair_attempts": self._max_code_repair_attempts,
            "max_step_llm_repair_attempts": self._max_step_llm_repair_attempts,
            "max_step_provider_calls": self._max_step_provider_calls,
            "enable_deterministic_code_fallback": bool(
                self._enable_deterministic_code_fallback
            ),
            "enable_deterministic_planner_fallback": bool(
                self._enable_deterministic_planner_fallback
            ),
            "planner_strategy": self._planner_strategy,
            "development_progressive_resume_checkpoint_sha256": (
                self._config.development_progressive_resume_checkpoint_sha256
            ),
            "development_locked_analysis_plan_sha256": (
                self._config.development_locked_analysis_plan_sha256
            ),
            "enable_deterministic_runner_repair": bool(
                self._enable_deterministic_runner_repair
            ),
            "enable_latex": bool(self._enable_latex),
            "enable_pdf_render": bool(self._enable_pdf_render),
            "latex_venue_template": self._latex_venue_template,
            "latex_draft_watermark": bool(self._latex_draft_watermark),
        }

    def _finalise_aborted(
        self,
        *,
        run_id: str,
        run_dir: Path,
        context: ResearchContext,
        context_path: Path,
        evidence: EvidenceStore,
        findings: List[ValidationFinding],
        reason: str,
    ) -> PipelineResult:
        """Delegate to :mod:`orchestration.finalize`."""
        from .orchestration.finalize import finalise_aborted

        return finalise_aborted(
            self,
            run_id=run_id,
            run_dir=run_dir,
            context=context,
            context_path=context_path,
            evidence=evidence,
            findings=findings,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from .reporting.publication_bundles import (  # noqa: F401 — owner module
    _AMBIGUOUS_FIGURE_DATA_FAMILY,
    _BINARY_GROUP_EXCLUDED_TOKENS,
    _INCOMPATIBLE_FIGURE_DATA_FAMILY,
    _RESULT_TABLE_COLS,
    _TABLE_ONE_ROWTYPE_COLS,
    _TABLE_ONE_VALUE_TOKENS,
    _UPSTREAM_FAMILY_TO_RENDERER_KEY,
    _UPSTREAM_FIGURE_DATA_FAMILY_TO_RENDERER_KEY,
    _UPSTREAM_METHOD_TO_RENDERER_KEY,
    _as_percent,
    _association_descriptive_context,
    _binary_group_column,
    _binary_group_label,
    _build_probe_summary,
    _context_axis_label,
    _event_count_column,
    _explicit_false_figure_value,
    _find_column,
    _is_risk_difference_row,
    _iter_prior_output_tables,
    _label_column,
    _promote_prior_publication_bundle,
    _promote_sibling_figure_exports,
    _publication_bundle_has_any_role,
    _publication_bundle_has_resolvable_sources,
    _publication_contract_file_references,
    _render_absolute_risk_publication_bundle_from_prior_outputs,
    _render_cohort_flow_publication_bundle_from_prior_outputs,
    _render_cohort_overlap_publication_bundle_from_prior_outputs,
    _render_descriptive_publication_bundle_from_prior_outputs,
    _render_phenotype_publication_bundle_from_prior_outputs,
    _render_prediction_publication_bundle_from_prior_outputs,
    _renderer_for_upstream_figure_data_family,
    _resolve_upstream_analysis_family,
    _sensitivity_plot_label,
    _truthy_figure_value,
    deterministic_figure_family_supported,
)


def _renderer_for_upstream_family(family: Optional[str]):
    """Map a parent ``analysis_family`` to its deterministic figure renderer."""

    key = _UPSTREAM_FAMILY_TO_RENDERER_KEY.get(str(family or "").strip().lower())
    if key is None:
        return None
    if key == "survival":
        from .figures.survival import (
            render_survival_bundle_from_prior_outputs as _render_survival_bundle,
        )

        return _render_survival_bundle
    return {
        "association": _render_association_publication_bundle_from_prior_outputs,
        "prediction": _render_prediction_publication_bundle_from_prior_outputs,
        "sensitivity": _render_sensitivity_publication_bundle_from_prior_outputs,
        "cohort": _render_cohort_overlap_publication_bundle_from_prior_outputs,
        "missingness": _render_missingness_publication_bundle_from_prior_outputs,
        "absolute_risk": _render_absolute_risk_publication_bundle_from_prior_outputs,
        "phenotype": _render_phenotype_publication_bundle_from_prior_outputs,
        "descriptive": _render_descriptive_publication_bundle_from_prior_outputs,
    }.get(key)


def _render_publication_bundle_from_prior_outputs_for_step(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    step_text: str = "",
    preverified_parent_digests: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Route by the direct parent's artifact, exact method, or family contract."""

    del step_text

    # An explicit artifact family has first priority, followed by the exact
    # parent-method compatibility adapter.  A supporting QC figure can
    # contain a generic token such as ``quality`` in its stochastic step id; the
    # token router would otherwise steal it for the missingness renderer even
    # though the parent recorded a more precise controlled method.
    _upstream_artifact_family = _resolve_upstream_figure_data_family(
        run_dir, current_step_id
    )
    _upstream_artifact_renderer = _renderer_for_upstream_figure_data_family(
        _upstream_artifact_family
    )
    _upstream_method_renderer = _renderer_for_upstream_method(
        _resolve_upstream_analysis_method(run_dir, current_step_id)
    )

    # Parent-family renderer (used as the else branch AND as a fallback for the
    # strict phenotype/descriptive renderers when their guard returns None, so a
    # mis-routed step still reaches the correct renderer rather than the coder).
    _upstream_family = _resolve_upstream_analysis_family(run_dir, current_step_id)
    _upstream_renderer = _renderer_for_upstream_family(_upstream_family)
    _upstream_fallback = (_upstream_renderer,) if _upstream_renderer is not None else ()
    # Cohort sensitivity, overlap, and attrition/flow are sibling renderings of
    # one closed cohort-definition family.  A direct-parent family declaration
    # should outrank stochastic step text, but a sensitivity/overlap renderer
    # that rejects an attrition-shaped parent must still hand off to the flow
    # renderer in the same family.  Keep this as an exact-family fallback; do
    # not reintroduce token routing for unrelated methods.
    if _upstream_family == "cohort_definition_sensitivity":
        _upstream_fallback = (
            _render_sensitivity_publication_bundle_from_prior_outputs,
            _render_cohort_overlap_publication_bundle_from_prior_outputs,
            _render_cohort_flow_publication_bundle_from_prior_outputs,
        )
    elif _upstream_family == "cohort_definition":
        # The automatic path admits this family only when the current direct
        # parent has digest-bound cohort_flow.csv + attrition.csv.  Render that
        # exact closed product; never probe an overlap renderer first and let a
        # schema coincidence choose a different scientific display.
        _upstream_fallback = (
            _render_cohort_flow_publication_bundle_from_prior_outputs,
        )

    if _upstream_artifact_family is not None:
        if _upstream_artifact_renderer is None:
            return None
        renderers = (_upstream_artifact_renderer,)
    elif _upstream_method_renderer is not None:
        renderers = (_upstream_method_renderer,)
    elif _upstream_fallback:
        renderers = _upstream_fallback
    else:
        return None

    # A split association-figure step should preserve a source-backed figure
    # that the direct analysis parent already produced.  Re-rendering from the
    # first OR-like CSV can discard panel semantics and flatten primary,
    # secondary, sensitivity, and adjustment rows into one forest plot.  Role
    # filtering plus the direct-parent rule in the promoter keeps this scoped;
    # when no eligible bundle exists, the deterministic renderer below remains
    # the fallback.
    if _render_association_publication_bundle_from_prior_outputs in renderers:
        promoted = _promote_prior_publication_bundle(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            required_roles=("primary_estimand",),
            require_declared_sources=True,
        )
        if promoted is not None:
            return promoted

    for renderer in renderers:
        renderer_kwargs: Dict[str, Any] = {}
        if preverified_parent_digests is not None:
            from .figures.distribution_availability import (
                render_distribution_availability_bundle_from_prior_outputs,
            )

            if renderer is render_distribution_availability_bundle_from_prior_outputs:
                renderer_kwargs["preverified_parent_digests"] = dict(
                    preverified_parent_digests
                )
        repair_id = renderer(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            **renderer_kwargs,
        )
        if repair_id is not None:
            return repair_id
    return None


def _resolve_upstream_analysis_method(
    run_dir: Path, current_step_id: str
) -> Optional[str]:
    """Return the controlled ``method`` recorded by a figure step's parent."""

    parent = str(current_step_id or "").removesuffix("_figure")
    if not parent or parent == str(current_step_id):
        return None
    summ = Path(run_dir) / "steps" / parent / "outputs" / "step_summary.json"
    try:
        method = json.loads(summ.read_text("utf-8")).get("method")
    except Exception:
        method = None
    if method:
        return str(method).strip().lower()

    # Free-model summaries need not repeat planning metadata.  The partial
    # manifest retains the exact structured AnalysisStep that produced the
    # parent, so use that method as a closed fallback instead of inferring a
    # renderer from the stochastic step id or intent prose.
    request_step = _resolve_upstream_manifest_step(run_dir, current_step_id)
    method = request_step.get("method") if request_step else None
    if method:
        return str(method).strip().lower()
    return None


def _planned_primary_association_contract(
    run_dir: Path,
    figure_step_id: str,
    summary: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Resolve one Planner-required primary model to its validated contract."""

    request_step = _resolve_upstream_manifest_step(run_dir, figure_step_id)
    if not isinstance(request_step, Mapping):
        return None
    requirements = request_step.get("model_requirements")
    primary_requirements = [
        item
        for item in requirements or []
        if isinstance(item, Mapping)
        and str(item.get("analysis_role") or "").strip().lower() == "primary"
        and item.get("required_for_step_success") is not False
    ]
    if len(primary_requirements) != 1:
        return None
    requirement_id = str(primary_requirements[0].get("requirement_id") or "")
    contracts = summary.get("model_contracts")
    matching_contracts = [
        item
        for item in contracts or []
        if isinstance(item, Mapping)
        and str(item.get("requirement_id") or "") == requirement_id
        and str(item.get("analysis_role") or "").strip().lower() == "primary"
        and str(item.get("fit_status") or "").strip().lower() == "fitted"
    ]
    if len(matching_contracts) != 1:
        return None
    contract = dict(matching_contracts[0])
    if not str(contract.get("model_id") or "").strip():
        return None
    if not str(contract.get("exposure_source") or "").strip():
        return None
    return contract


def _absolute_risk_parent_digest_seal(
    run_dir: Path,
    figure_step_id: str,
) -> Optional[dict[str, str]]:
    """Validate and seal the absolute-risk renderer's exact parent product."""

    digests = _verified_direct_parent_artifact_digests(run_dir, figure_step_id)
    required_names = {
        "step_summary.json",
        "outcome_incidence.csv",
        "exposure_prevalence.csv",
    }
    if not digests or not required_names <= set(digests):
        return None
    request_step = _resolve_upstream_manifest_step(run_dir, figure_step_id)
    if not isinstance(request_step, Mapping):
        return None
    from .contracts.declared_product import (
        read_digest_bound_artifact_snapshot,
        typed_product,
    )
    from .figures.absolute_risk import (
        CONTROLLED_METHOD,
        prepare_absolute_risk_inputs,
    )

    if str(request_step.get("method") or "").strip().lower() != CONTROLLED_METHOD:
        return None
    declared_tables = {
        parsed
        for raw in (request_step.get("expected_outputs") or [])
        if (parsed := typed_product(raw)) is not None and parsed[0] == "table"
    }
    if declared_tables != {
        ("table", "outcome_incidence"),
        ("table", "exposure_prevalence"),
    }:
        return None
    sealed = {name: digests[name] for name in sorted(required_names)}
    parent_step_id = str(figure_step_id or "").removesuffix("_figure")
    parent_out = Path(run_dir) / "steps" / parent_step_id / "outputs"
    try:
        context_payload = json.loads(
            (Path(run_dir) / "research_context.json").read_text(encoding="utf-8")
        )
        expected_primary_exposure = str(
            context_payload.get("primary_exposure") or ""
        ).strip()
        expected_target_outcome = str(
            context_payload.get("target_outcome") or ""
        ).strip()
        if not expected_primary_exposure or not expected_target_outcome:
            return None
        snapshot = read_digest_bound_artifact_snapshot(
            parent_out=parent_out,
            artifact_digests=sealed,
        )
        summary = json.loads(snapshot["step_summary.json"].decode("utf-8"))
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if (
        not isinstance(summary, Mapping)
        or prepare_absolute_risk_inputs(
            summary,
            snapshot["outcome_incidence.csv"],
            snapshot["exposure_prevalence.csv"],
            expected_primary_exposure=expected_primary_exposure,
            expected_target_outcome=expected_target_outcome,
        )
        is None
    ):
        return None
    return sealed


def _sealed_renderer_parent_digest_seal(
    run_dir: Path,
    figure_step_id: str,
    repair_id: str,
) -> Optional[dict[str, str]]:
    """Return the exact evidence digests one sealed renderer may consume."""

    if adapter := sealed_renderer_adapter(repair_id):
        return adapter.seal(run_dir, figure_step_id)
    if repair_id == "absolute_risk_incidence_prevalence_publication_bundle_v1":
        return _absolute_risk_parent_digest_seal(run_dir, figure_step_id)
    digests = _verified_direct_parent_artifact_digests(run_dir, figure_step_id)
    if not digests or "step_summary.json" not in digests:
        return None
    csv_names = {name for name in digests if Path(name).suffix.lower() == ".csv"}
    if repair_id == "ordered_category_distribution_publication_bundle_v1":
        required_names = {"step_summary.json", *csv_names}
        if not csv_names:
            return None
    elif repair_id == "cohort_flow_publication_bundle_from_parent_outputs_v1":
        required_names = {
            "step_summary.json",
            "cohort_flow.csv",
            "attrition.csv",
        }
    elif repair_id == "sensitivity_publication_bundle_from_locked_summary_v1":
        required_names = {"step_summary.json", "robustness_summary.csv"}
    elif repair_id == ("association_publication_bundle_from_planned_model_contract_v1"):
        required_names = {
            "step_summary.json",
            "adjusted_association_estimates.csv",
        }
    else:
        return None
    if not required_names <= set(digests):
        return None
    sealed = {name: digests[name] for name in sorted(required_names)}
    if repair_id != "association_publication_bundle_from_planned_model_contract_v1":
        return sealed

    # Association rendering is automatic only after the Planner and the
    # validated parent summary identify one exact primary model contract.  The
    # renderer then selects by model_id + exposure_source; it never searches
    # table prose, variable-name fragments, or benchmark vocabulary.
    parent_step_id = str(figure_step_id or "").removesuffix("_figure")
    parent_out = Path(run_dir) / "steps" / parent_step_id / "outputs"
    try:
        from .contracts.declared_product import read_digest_bound_artifact_snapshot

        snapshot = read_digest_bound_artifact_snapshot(
            parent_out=parent_out,
            artifact_digests=sealed,
        )
        summary = json.loads(snapshot["step_summary.json"].decode("utf-8"))
        estimates = pd.read_csv(
            io.BytesIO(snapshot["adjusted_association_estimates.csv"])
        )
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(summary, Mapping):
        return None
    contract = _planned_primary_association_contract(
        run_dir,
        figure_step_id,
        summary,
    )
    if contract is None:
        return None
    model_id = str(contract.get("model_id") or "").strip()
    exposure = str(contract.get("exposure_source") or "").strip()
    required_columns = {"model_id", "term_role", "source_variable"}
    if not model_id or not exposure or not required_columns <= set(estimates.columns):
        return None
    selected = estimates.loc[
        estimates["model_id"].astype(str).eq(model_id)
        & estimates["term_role"].astype(str).str.lower().eq("exposure")
        & estimates["source_variable"].astype(str).eq(exposure)
    ]
    if selected.empty:
        return None
    return sealed


def _resolve_upstream_figure_data_family(
    run_dir: Path, current_step_id: str
) -> Optional[str]:
    """Return one unambiguous figure-data family declared by the parent."""

    parent = str(current_step_id or "").removesuffix("_figure")
    if not parent or parent == str(current_step_id):
        return None
    summ = Path(run_dir) / "steps" / parent / "outputs" / "step_summary.json"
    try:
        summary = json.loads(summ.read_text("utf-8"))
    except Exception:
        return None
    if not isinstance(summary, dict):
        return None
    families = {str(summary.get("figure_data_family") or "").strip().lower()}
    contracts = summary.get("figure_data_contracts")
    if isinstance(contracts, list):
        families.update(
            str(item.get("family") or "").strip().lower()
            for item in contracts
            if isinstance(item, dict)
        )
    families.discard("")
    if not families:
        return None
    if len(families) > 1:
        return _AMBIGUOUS_FIGURE_DATA_FAMILY
    family = next(iter(families))
    if family == "ordered_category_distribution":
        request_step = _resolve_upstream_manifest_step(run_dir, current_step_id)
        declared_outputs = (
            request_step.get("expected_outputs") if request_step else None
        )
        if isinstance(declared_outputs, list) and declared_outputs:
            names = {
                re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    str(item).split(":", 1)[-1].strip().lower(),
                ).strip("_")
                for item in declared_outputs
            }
            has_distribution = any(
                name.endswith("_distribution")
                or name in {"category_distribution", "level_distribution"}
                for name in names
            )
            has_result_product = any(
                token in name
                for name in names
                for token in (
                    "association",
                    "effect",
                    "estimate",
                    "outcome",
                    "trend",
                )
            )
            if has_result_product or not has_distribution:
                return _INCOMPATIBLE_FIGURE_DATA_FAMILY
    return family


def deterministic_figure_repair_id_for_upstream(
    run_dir: Path, step_id: str
) -> Optional[str]:
    """Return one Planner-authorized, evidence-bound renderer repair id.

    Sealed preflight may replace the coder, so coder-written summary fields are
    never routing authority.  The host-recorded parent planning request selects
    one exact standard method family; registered table names and the closed
    renderer schema then provide the structural evidence.  A summary method, if
    present, is only an equality check against the Planner method.
    """

    verified_tables = _verified_direct_parent_table_names(run_dir, step_id)
    if not verified_tables:
        return None
    request = _resolve_upstream_manifest_analysis_request(run_dir, step_id)
    request_step = request.get("step") if isinstance(request, Mapping) else None
    if not isinstance(request_step, Mapping):
        return None
    from .contracts.declared_product import typed_product

    planner_table_tokens = {
        tuple(part for part in parsed[1].split("_") if part)
        for raw in (request_step.get("expected_outputs") or [])
        if (parsed := typed_product(raw)) is not None
        and parsed[0] in {"table", "artifact", "dataset"}
    }

    def _declares(role_suffixes: Sequence[Sequence[str]]) -> bool:
        return any(
            len(tokens) >= len(suffix) and tokens[-len(suffix) :] == tuple(suffix)
            for tokens in planner_table_tokens
            for suffix in role_suffixes
        )

    planner_method = str(request_step.get("method") or "").strip().lower()
    if not planner_method:
        return None
    reported_method = _resolve_upstream_analysis_method(run_dir, step_id)
    if reported_method and reported_method != planner_method:
        return None
    from .repair_registry import sealed_renderer_metadata

    candidates = [
        metadata
        for metadata in sealed_renderer_metadata()
        if planner_method in metadata.planner_methods
        and all(
            _declares(role_alternatives)
            for role_alternatives in metadata.planner_parent_output_role_groups
        )
    ]
    if not candidates:
        structural_v2 = (
            "ordered_category_distribution_availability_publication_bundle_v2"
        )
        if (
            _ordered_distribution_availability_parent_digest_seal(run_dir, step_id)
            is not None
        ):
            return structural_v2
    if len(candidates) != 1:
        return None
    repair_id = candidates[0].repair_id
    if adapter := sealed_renderer_adapter(repair_id):
        return repair_id if adapter.seal(run_dir, step_id) is not None else None
    if repair_id == "absolute_risk_incidence_prevalence_publication_bundle_v1":
        required_tables = {"outcome_incidence.csv", "exposure_prevalence.csv"}
        return (
            repair_id
            if required_tables <= verified_tables
            and _absolute_risk_parent_digest_seal(run_dir, step_id) is not None
            else None
        )
    if repair_id == ("association_publication_bundle_from_planned_model_contract_v1"):
        return (
            repair_id
            if "adjusted_association_estimates.csv" in verified_tables
            and _sealed_renderer_parent_digest_seal(
                run_dir,
                step_id,
                repair_id,
            )
            is not None
            else None
        )
    if repair_id == "sensitivity_publication_bundle_from_locked_summary_v1":
        return repair_id if "robustness_summary.csv" in verified_tables else None
    if repair_id == "cohort_flow_publication_bundle_from_parent_outputs_v1":
        return (
            repair_id
            if {"cohort_flow.csv", "attrition.csv"} <= verified_tables
            else None
        )
    if repair_id == "ordered_category_distribution_publication_bundle_v1":
        return repair_id
    return None


def _render_association_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
    authorized_repair_id: Optional[str] = None,
) -> Optional[str]:
    """Deterministically build a multi-panel figure from a prior association step.

    Mirror of the prediction repair for adjusted-association analyses. Small
    models sometimes write a coefficient table (``odds_ratio`` + ``or_ci_low`` /
    ``or_ci_high`` columns) in the regression step but fail the follow-up
    figure-only step (e.g. hard-coding a wrong results filename). Rather than
    accepting a one-panel placeholder, render a source-data-backed association
    figure with uncertainty context from the registered parent table.
    """
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    # Resolve the odds-ratio + CI columns by common name variants, so the
    # rescue works whether the parent step wrote ``or_ci_low/or_ci_high`` (our
    # deterministic fallback) or ``ci_lower/ci_upper`` etc. (free-model code).
    # Without this the rescue silently skips a perfectly good coefficient table
    # and the figure-only step fails the whole run.
    _OR_ALIASES = ("odds_ratio", "oddsratio", "adjusted_or", "aor", "or")
    _CI_LOW_ALIASES = (
        "or_ci_low",
        "or_ci_lower",
        "ci_lower",
        "ci_low",
        "or_lower",
        "conf_low",
        "ci95_low",
        "ci_low_95",
        "lower",
    )
    _CI_HIGH_ALIASES = (
        "or_ci_high",
        "or_ci_upper",
        "ci_upper",
        "ci_high",
        "or_upper",
        "conf_high",
        "ci95_high",
        "ci_high_95",
        "upper",
    )

    def _resolve_or_ci_columns(frame: pd.DataFrame):
        lower_to_orig = {str(c).lower(): c for c in frame.columns}
        or_c = next((lower_to_orig[a] for a in _OR_ALIASES if a in lower_to_orig), None)
        if or_c is None and any(
            key in lower_to_orig
            for key in ("estimate", "point_estimate", "effect_estimate")
        ):
            scale_col = next(
                (
                    lower_to_orig[a]
                    for a in ("effect_scale", "scale", "measure")
                    if a in lower_to_orig
                ),
                None,
            )
            if scale_col is not None:
                scale_text = (
                    frame[scale_col]
                    .astype(str)
                    .str.lower()
                    .str.replace(r"[_-]+", " ", regex=True)
                )
                if scale_text.str.contains(
                    r"\b(?:odds ratio|or)\b",
                    regex=True,
                    na=False,
                ).any():
                    estimate_key = next(
                        (
                            key
                            for key in (
                                "estimate",
                                "point_estimate",
                                "effect_estimate",
                            )
                            if key in lower_to_orig
                        ),
                        None,
                    )
                    if estimate_key is not None:
                        or_c = lower_to_orig[estimate_key]
        lo_c = next(
            (lower_to_orig[a] for a in _CI_LOW_ALIASES if a in lower_to_orig), None
        )
        hi_c = next(
            (lower_to_orig[a] for a in _CI_HIGH_ALIASES if a in lower_to_orig), None
        )
        if or_c and lo_c and hi_c:
            return or_c, lo_c, hi_c
        return None

    sealed_repair_id = "association_publication_bundle_from_planned_model_contract_v1"
    parent: Optional[tuple[Path, pd.DataFrame, tuple[str, str, str]]] = None
    if preverified_parent_artifacts is not None:
        if authorized_repair_id != sealed_repair_id:
            return None
        try:
            candidate_frame = pd.read_csv(
                io.BytesIO(
                    preverified_parent_artifacts["adjusted_association_estimates.csv"]
                )
            )
        except (KeyError, OSError, ValueError):
            return None
        resolved = _resolve_or_ci_columns(candidate_frame)
        if resolved is None:
            return None
        parent_step_id = str(current_step_id or "").removesuffix("_figure")
        table_path = (
            Path(run_dir)
            / "steps"
            / parent_step_id
            / "outputs"
            / "adjusted_association_estimates.csv"
        )
        parent = (table_path, candidate_frame, resolved)
    else:
        candidate_step_dirs, _direct_parent_only = _figure_parent_candidate_step_dirs(
            steps_dir=steps_dir, current_step_id=current_step_id
        )
        for step_dir in candidate_step_dirs:
            outputs_dir = step_dir / "outputs"
            if not outputs_dir.exists():
                continue
            candidates: List[
                tuple[tuple[int, int], Path, pd.DataFrame, tuple[str, str, str]]
            ] = []
            for csv_path in sorted(outputs_dir.glob("*.csv")):
                try:
                    candidate_frame = pd.read_csv(csv_path)
                except Exception:
                    continue
                resolved = _resolve_or_ci_columns(candidate_frame)
                if resolved is None:
                    continue
                columns = {str(column).lower() for column in candidate_frame.columns}
                structured_coefficients = {
                    "model_id",
                    "term",
                    "term_role",
                    "source_variable",
                }.issubset(columns)
                score = (
                    int(structured_coefficients),
                    int(
                        structured_coefficients
                        and "coefficient" in csv_path.stem.lower()
                    ),
                )
                candidates.append((score, csv_path, candidate_frame, resolved))
            if candidates:
                _, csv_path, candidate_frame, resolved = max(
                    candidates,
                    key=lambda item: item[0],
                )
                parent = (csv_path, candidate_frame, resolved)
                break
    if parent is None:
        return None

    table_path, frame, (or_col, lo_col, hi_col) = parent
    lower_to_orig = {str(c).lower(): c for c in frame.columns}

    parent_summary: Dict[str, Any] = {}
    summary_path = table_path.parent / "step_summary.json"
    if preverified_parent_artifacts is not None:
        try:
            loaded = json.loads(
                preverified_parent_artifacts["step_summary.json"].decode("utf-8")
            )
            if isinstance(loaded, dict):
                parent_summary = loaded
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
            return None
    elif summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                parent_summary = loaded
        except Exception:
            parent_summary = {}
    primary_model_id = str(parent_summary.get("primary_model_id") or "").strip()
    model_contracts = parent_summary.get("model_contracts") or []
    if not isinstance(model_contracts, list):
        model_contracts = []
    primary_contract: Optional[Mapping[str, Any]] = None
    if authorized_repair_id == sealed_repair_id:
        primary_contract = _planned_primary_association_contract(
            run_dir,
            current_step_id,
            parent_summary,
        )
        if primary_contract is None:
            return None
        primary_model_id = str(primary_contract.get("model_id") or "").strip()
    else:
        primary_contract = next(
            (
                contract
                for contract in model_contracts
                if isinstance(contract, dict)
                and primary_model_id
                and str(contract.get("model_id") or "") == primary_model_id
            ),
            None,
        )
        if primary_contract is None:
            primary_contract = next(
                (
                    contract
                    for contract in model_contracts
                    if isinstance(contract, dict)
                    and str(contract.get("analysis_role") or "").lower() == "primary"
                    and str(contract.get("exposure_role") or "primary").lower()
                    == "primary"
                ),
                None,
            )
    if not primary_model_id and isinstance(primary_contract, dict):
        primary_model_id = str(primary_contract.get("model_id") or "").strip()
    primary_exposure = (
        str(primary_contract.get("exposure_source") or "").strip()
        if isinstance(primary_contract, dict)
        else ""
    )
    matching_model_ids = (
        {primary_model_id}
        if authorized_repair_id == sealed_repair_id
        else {
            str(contract.get("model_id") or "").strip()
            for contract in model_contracts
            if isinstance(contract, dict)
            and primary_exposure
            and str(contract.get("exposure_source") or "").strip() == primary_exposure
            and str(contract.get("exposure_role") or "primary").lower() == "primary"
            and str(contract.get("analysis_role") or "").lower()
            in {"primary", "sensitivity"}
        }
    )
    matching_model_ids.discard("")

    plot_df = frame.copy()
    term_role_col = lower_to_orig.get("term_role")
    if term_role_col is not None:
        exposure_rows = plot_df[term_role_col].astype(str).str.lower().eq("exposure")
        if exposure_rows.any():
            plot_df = plot_df.loc[exposure_rows].copy()
    model_id_col = lower_to_orig.get("model_id")
    if model_id_col is not None:
        if matching_model_ids:
            selected_models = plot_df[model_id_col].astype(str).isin(matching_model_ids)
            if selected_models.any():
                plot_df = plot_df.loc[selected_models].copy()
        elif primary_model_id:
            selected_primary = plot_df[model_id_col].astype(str).eq(primary_model_id)
            if selected_primary.any():
                plot_df = plot_df.loc[selected_primary].copy()
    source_variable_col = lower_to_orig.get("source_variable")
    if source_variable_col is not None and primary_exposure:
        selected_exposure = (
            plot_df[source_variable_col].astype(str).eq(primary_exposure)
        )
        if selected_exposure.any():
            plot_df = plot_df.loc[selected_exposure].copy()

    def _n_distinct(col: str) -> int:
        try:
            return int(plot_df[col].astype(str).nunique(dropna=True))
        except Exception:
            return 0

    # Pick the column that LABELS / keys each forest row. Prefer a known
    # variable/exposure-descriptor column, but only if it actually VARIES across
    # rows: an association table for a single graded exposure keeps the exposure
    # name constant (e.g. exposure_variable='sofa2_liver_cat' on every row) and
    # distinguishes rows by an ordinal level/band column. Keying on the constant
    # column collapses every forest row to one label and drops the per-row trace
    # key. Skip constant candidates and fall back to
    # the first varying column rather than blindly to columns[0].
    _LABEL_CANDIDATES = (
        "term",
        "variable",
        "exposure",
        "predictor",
        "feature",
        "covariate",
        "exposure_variable",
        "level",
        "band",
        "category",
        "stage",
        "group",
        "bin",
        "quantile",
        "tertile",
        "quartile",
        "decile",
    )
    _present = [
        str(lower_to_orig[key]) for key in _LABEL_CANDIDATES if key in lower_to_orig
    ]
    var_col = (
        # a named candidate that VARIES across rows (avoids the collapse) ...
        next((c for c in _present if _n_distinct(c) > 1), None)
        # ... else the first named candidate (single-row / genuinely all-constant
        # forests have no collapse risk; keep the original semantic label) ...
        or next(iter(_present), None)
        # ... else the first varying column, else the first column.
        or next((str(c) for c in plot_df.columns if _n_distinct(str(c)) > 1), None)
        or str(plot_df.columns[0])
    )
    # Drop the intercept term; it is not an interpretable effect estimate.
    intercept_col = lower_to_orig.get("term", var_col)
    plot_df = plot_df[
        ~plot_df[intercept_col].astype(str).str.lower().isin({"const", "intercept"})
    ]
    for _c in (or_col, lo_col, hi_col):
        plot_df = plot_df.assign(**{_c: pd.to_numeric(plot_df[_c], errors="coerce")})
    plot_df = plot_df.dropna(subset=[or_col, lo_col, hi_col])
    if plot_df.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from easyicu.research_agent.figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = plot_df[var_col].astype(str).tolist()
    full_display_labels = [_publication_label(label) for label in labels]
    analysis_set_col = lower_to_orig.get("analysis_set")
    if len(plot_df) > 1 and model_id_col is not None:
        qualified_labels: List[str] = []
        for row_idx, base_label in enumerate(full_display_labels):
            row = plot_df.iloc[row_idx]
            qualifier = (
                row.get(analysis_set_col)
                if analysis_set_col is not None
                else row.get(model_id_col)
            )
            qualified_labels.append(f"{base_label} ({_publication_label(qualifier)})")
        full_display_labels = qualified_labels
    display_labels = [
        _short_figure_label(label.replace("Maximum ", "Max "), limit=32)
        for label in full_display_labels
    ]
    or_vals = plot_df[or_col].astype(float).to_numpy()
    lo = plot_df[lo_col].astype(float).to_numpy()
    hi = plot_df[hi_col].astype(float).to_numpy()
    ci_width = hi - lo
    y = list(range(len(labels)))
    source_row_indices = plot_df.index.to_list()
    source_data = plot_df.copy().reset_index(drop=True)
    source_data = source_data.assign(
        source_row_index=source_row_indices,
        display_label=full_display_labels,
        plot_label=display_labels,
        point_estimate=or_vals,
        odds_ratio=or_vals,
        ci_low=lo,
        ci_high=hi,
        ci_width=ci_width,
        source_table=table_path.name,
    )
    source_data.to_csv(out_dir / "publication_figure_source_data.csv", index=False)
    descriptive_context = (
        {
            "plot_rows": [],
            "source_files": [],
            "has_prevalence": False,
            "has_outcome_risk": False,
            "title": "",
            "claim": "",
        }
        if preverified_parent_artifacts is not None
        else _association_descriptive_context(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            primary_exposure=primary_exposure or None,
        )
    )
    descriptive_rows = list(descriptive_context.get("plot_rows") or [])
    association_panel_title = (
        "Primary adjusted association" if len(labels) <= 3 else "Adjusted association"
    )
    association_panel_claim = (
        "The primary adjusted odds ratio and 95% CI are read from the parent association table."
        if len(labels) <= 3
        else (
            "Per-covariate adjusted odds ratios and 95% CIs are read "
            "from the parent association table."
        )
    )
    association_chart_type = "dot_interval" if len(labels) <= 3 else "forest"

    palette = apply_publication_style()
    if descriptive_rows:
        fig_height_mm = max(82, 18 * len(labels) + 22, 16 * len(descriptive_rows) + 28)
        fig = plt.figure(
            figsize=(183 / 25.4, fig_height_mm / 25.4),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.02, 1.28],
            left=0.18,
            right=0.98,
            top=0.90,
            bottom=0.18,
            wspace=0.82,
        )
        ax_context = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[0, 1])

        context_df = pd.DataFrame(descriptive_rows)
        context_labels = []
        for _, row in context_df.iterrows():
            metric = str(row.get("plot_metric") or "").strip()
            group = str(row.get("plot_group_label") or "").strip()
            context_labels.append(_context_axis_label(metric, group))
        context_x = pd.to_numeric(
            context_df["plot_estimate_pct"], errors="coerce"
        ).to_numpy()
        context_lo = (
            pd.to_numeric(
                context_df.get("plot_ci_low_pct", context_df["plot_estimate_pct"]),
                errors="coerce",
            )
            .fillna(pd.Series(context_x))
            .to_numpy()
        )
        context_hi = (
            pd.to_numeric(
                context_df.get("plot_ci_high_pct", context_df["plot_estimate_pct"]),
                errors="coerce",
            )
            .fillna(pd.Series(context_x))
            .to_numpy()
        )
        y_context = list(range(len(context_labels)))
        ax_context.errorbar(
            context_x,
            y_context,
            xerr=[
                [
                    max(0.0, center - lower)
                    for center, lower in zip(context_x, context_lo)
                ],
                [
                    max(0.0, upper - center)
                    for center, upper in zip(context_x, context_hi)
                ],
            ],
            fmt="o",
            color=palette.get("teal", "#42949E"),
            ecolor=palette.get("teal", "#42949E"),
            elinewidth=1.0,
            capsize=2.3,
            markersize=4.0,
        )
        max_context = max(
            [float(x) for x in context_hi if math.isfinite(float(x))] or [1.0]
        )
        ax_context.set_xlim(0, max(5.0, max_context + 8.0, max_context * 1.35))
        ax_context.set_yticks(y_context)
        ax_context.set_yticklabels(context_labels, fontsize=6.8)
        ax_context.set_xlabel("Percent (95% CI)")
        ax_context.set_title(str(descriptive_context["title"]), loc="left", pad=4)
        ax_context.set_ylim(max(len(context_labels) + 1.8, 4.2), -0.5)
        ax_context.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
            alpha=0.8,
        )
        for row_idx, row in context_df.iterrows():
            event_n = pd.to_numeric(
                pd.Series([row.get("plot_event_n")]), errors="coerce"
            ).iloc[0]
            denom = pd.to_numeric(
                pd.Series([row.get("plot_denominator")]), errors="coerce"
            ).iloc[0]
            if pd.notna(event_n) and pd.notna(denom):
                label = f"{float(context_x[row_idx]):.1f}% ({int(event_n):,}/{int(denom):,})"
            else:
                label = f"{float(context_x[row_idx]):.1f}%"
            ax_context.text(
                max(float(context_hi[row_idx]), float(context_x[row_idx])) + 0.6,
                row_idx,
                label,
                va="center",
                fontsize=6.3,
                color=palette.get("baseline", "#272727"),
            )
        add_panel_label(ax_context, "A", x=0.0, y=1.08)
    else:
        fig = plt.figure(
            figsize=(183 / 25.4, max(72, 18 * len(labels) + 18) / 25.4),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.45, 0.85],
            left=0.22,
            right=0.98,
            top=0.90,
            bottom=0.18,
            wspace=0.42,
        )
        ax = fig.add_subplot(grid[0, 0])
        ax_width = fig.add_subplot(grid[0, 1], sharey=ax)
    ax.errorbar(
        or_vals,
        y,
        xerr=[
            [max(0.0, center - lower) for center, lower in zip(or_vals, lo)],
            [max(0.0, upper - center) for center, upper in zip(or_vals, hi)],
        ],
        fmt="o",
        color=palette.get("blue", "#0F4D92"),
        ecolor=palette.get("blue", "#0F4D92"),
        elinewidth=1.0,
        capsize=2.3,
        markersize=4.0,
    )
    ax.axvline(
        1.0,
        color=palette.get("neutral", "#8F8F8F"),
        linewidth=0.9,
        linestyle="--",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels, fontsize=6.6)
    ax.set_xlabel("Adjusted odds ratio (95% CI)")
    ax.set_title(association_panel_title, loc="left", pad=4)
    if len(labels) <= 3:
        max_hi = max(float(value) for value in hi if math.isfinite(float(value)))
        ax.set_xlim(
            left=max(0.01, min(float(value) for value in lo) * 0.96),
            right=max_hi * 1.28,
        )
        for row_idx, (center, lower, upper) in enumerate(zip(or_vals, lo, hi)):
            ax.text(
                float(upper) * 1.025,
                row_idx,
                f"OR {float(center):.2f} ({float(lower):.2f}-{float(upper):.2f})",
                va="center",
                fontsize=6.2,
                color=palette.get("baseline", "#272727"),
            )
    ax.invert_yaxis()
    ax.grid(
        axis="x",
        color=palette.get("neutral_light", "#D8D8D8"),
        linewidth=0.55,
        alpha=0.8,
    )
    add_panel_label(ax, "B" if descriptive_rows else "A", x=0.0, y=1.08)

    if not descriptive_rows:
        ax_width.barh(
            y,
            ci_width,
            color=palette.get("orange", "#E69F00"),
            height=0.5,
        )
        ax_width.set_xlabel("95% CI width")
        ax_width.set_title("Estimate precision", loc="left", pad=4)
        ax_width.tick_params(axis="y", labelleft=False)
        ax_width.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
            alpha=0.8,
        )
        add_panel_label(ax_width, "B", x=0.0, y=1.08)

    source_data_files = [
        "publication_figure_source_data.csv",
        *[str(item) for item in descriptive_context.get("source_files", [])],
    ]
    if descriptive_rows:
        panels = [
            {
                "panel_id": "A",
                "title": str(descriptive_context["title"]),
                "role": "descriptive_result",
                "chart_type": "dot_interval_absolute_risk",
                "claim": str(descriptive_context["claim"]),
                "evidence_ids": [
                    item for item in descriptive_context.get("source_files", [])
                ],
            },
            {
                "panel_id": "B",
                "title": association_panel_title,
                "role": "primary_estimand",
                "chart_type": association_chart_type,
                "claim": association_panel_claim,
                "evidence_ids": ["publication_figure_source_data.csv"],
                "metadata": {"planner_product_slots": ["primary_estimand"]},
            },
        ]
        core_claim = (
            "The figure pairs reader-facing prevalence or absolute-risk context "
            "with the adjusted association estimate and uncertainty."
        )
    else:
        panels = [
            {
                "panel_id": "A",
                "title": association_panel_title,
                "role": "primary_estimand",
                "chart_type": association_chart_type,
                "claim": association_panel_claim,
                "evidence_ids": ["publication_figure_source_data.csv"],
                "metadata": {"planner_product_slots": ["primary_estimand"]},
            },
            {
                "panel_id": "B",
                "title": "Interval-width audit",
                "role": "robustness",
                "chart_type": "bar",
                "claim": (
                    "The width of each 95% CI is shown to expose estimate "
                    "precision rather than hiding uncertainty in the forest plot."
                ),
                "evidence_ids": ["publication_figure_source_data.csv"],
                "metadata": {"planner_product_slots": ["precision_audit"]},
            },
        ]
        core_claim = (
            "Adjusted associations and their uncertainty are summarised from "
            "the registered association coefficient table."
        )
    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=core_claim,
        panels=panels,
        source_data=source_data_files,
        statistics_note=(
            "Generated deterministically from registered parent-step tables; "
            "the association panel uses the coefficient table and any context "
            "panel uses prevalence or outcome-risk source tables when present."
        ),
    )
    outputs = save_publication_figure(
        fig, out_dir / "publication_figure", contract=contract, dpi=300
    )
    plt.close(fig)

    existing_summary: Dict[str, Any] = {}
    step_summary_path = out_dir / "step_summary.json"
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_summary = loaded
        except Exception:
            existing_summary = {}
    observed_repair_id = (
        sealed_repair_id
        if authorized_repair_id == sealed_repair_id
        else (
            "association_publication_bundle_from_parent_outputs_v3"
            if descriptive_rows
            else "association_publication_bundle_from_parent_outputs_v2"
        )
    )
    existing_summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_association_publication_figure_repair",
            "rendering_only": True,
            "deterministic_publication_figure_rescue": observed_repair_id,
            "source_step_id": current_step_id.removesuffix("_figure"),
            "figure_contract": "publication_figure.figure_contract.json",
        }
    )
    existing_summary.setdefault("publication_figure_repair", {})
    existing_summary["publication_figure_repair"].update(
        {
            "mode": "association_forest_from_parent_outputs",
            "source_association_table": str(table_path),
            "source_data": "publication_figure_source_data.csv",
            "descriptive_source_data": descriptive_context.get("source_files", []),
            "primary_model_id": primary_model_id or None,
            "primary_exposure_source": primary_exposure or None,
            "selected_model_ids": sorted(matching_model_ids),
            "n_association_rows": int(len(plot_df)),
        }
    )
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    existing_summary["figure_files"] = figure_files
    if figure_files:
        existing_summary["figure_path"] = figure_files[0]
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return observed_repair_id


def _render_sensitivity_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
    authorized_repair_id: Optional[str] = None,
) -> Optional[str]:
    """Deterministically rebuild a sensitivity figure from parent outputs."""

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    parent_step_id = current_step_id.removesuffix("_figure")
    candidate_sources: List[Tuple[Path, Union[Path, bytes]]] = []
    direct_outputs = steps_dir / parent_step_id / "outputs"
    if parent_step_id and parent_step_id != current_step_id and direct_outputs.exists():
        # A split rendering step must not silently borrow a similarly-shaped
        # table from an unrelated sensitivity step.  Search the direct parent
        # only when it exists.  Older plans whose rendering-step name does not
        # share the exact parent stem retain the conservative fallback below.
        if preverified_parent_artifacts is None:
            direct_candidates = [
                (path, path) for path in sorted(direct_outputs.glob("*.csv"))
            ]
        elif authorized_repair_id == (
            "sensitivity_publication_bundle_from_locked_summary_v1"
        ):
            payload = preverified_parent_artifacts.get("robustness_summary.csv")
            if (
                payload is None
                or "step_summary.json" not in preverified_parent_artifacts
            ):
                return None
            direct_candidates = [(direct_outputs / "robustness_summary.csv", payload)]
        else:
            direct_candidates = [
                (direct_outputs / name, payload)
                for name, payload in sorted(preverified_parent_artifacts.items())
                if Path(name).name == name and Path(name).suffix.lower() == ".csv"
            ]
        declared_names: set[str] = set()
        try:
            summary_payload = (
                preverified_parent_artifacts.get("step_summary.json")
                if preverified_parent_artifacts is not None
                else None
            )
            direct_summary = json.loads(
                summary_payload.decode("utf-8")
                if summary_payload is not None
                else (direct_outputs / "step_summary.json").read_text(encoding="utf-8")
            )
        except Exception:
            direct_summary = {}
        if isinstance(direct_summary, dict):
            for mapping_key in ("output_files", "aliases"):
                mapping = direct_summary.get(mapping_key)
                if isinstance(mapping, dict):
                    declared_items = mapping.items()
                elif isinstance(mapping, list):
                    declared_items = ((str(value), value) for value in mapping)
                else:
                    continue
                for alias, value in declared_items:
                    if not any(
                        token in str(alias).lower()
                        for token in ("robustness", "sensitivity")
                    ):
                        continue
                    if isinstance(value, str) and value.lower().endswith(".csv"):
                        declared_names.add(Path(value).name)
        candidate_sources.extend(
            sorted(
                direct_candidates,
                key=lambda item: (
                    item[0].name not in declared_names,
                    item[0].name,
                ),
            )
        )
    elif preverified_parent_artifacts is None:
        for step_dir in sorted(steps_dir.iterdir()):
            if not step_dir.is_dir() or step_dir.name == current_step_id:
                continue
            if "sensitivity" not in step_dir.name.lower():
                continue
            outputs_dir = step_dir / "outputs"
            if outputs_dir.exists():
                candidate_sources.extend(
                    (path, path) for path in sorted(outputs_dir.glob("*.csv"))
                )

    parent: Optional[tuple[Path, pd.DataFrame]] = None
    for csv_path, source in candidate_sources:
        try:
            # Preserve round-trippable confidence-limit values; the default
            # fast parser can change the final binary digit and create a false
            # source-trace mismatch on an otherwise identical table.
            frame = pd.read_csv(
                io.BytesIO(source) if isinstance(source, bytes) else source,
                float_precision="round_trip",
            )
        except Exception:
            continue
        required = {"spec_id", "effect_scale", "point_estimate", "ci_low", "ci_high"}
        if required <= set(frame.columns):
            parent = (csv_path, frame)
            break
    if parent is None:
        return None

    table_path, frame = parent
    source_step_id = table_path.parents[1].name
    source_data = frame.copy()
    source_data["source_table"] = table_path.name
    source_data["source_step_id"] = source_step_id
    for col in (
        "point_estimate",
        "ci_low",
        "ci_high",
        "modeled_analytic_n",
        "event_n",
        "membership_n",
    ):
        if col in source_data.columns:
            source_data[col] = pd.to_numeric(source_data[col], errors="coerce")
    if "modeled_analytic_n" not in source_data.columns:
        for count_alias in ("analysis_n", "n"):
            if count_alias in source_data.columns:
                source_data["modeled_analytic_n"] = pd.to_numeric(
                    source_data[count_alias], errors="coerce"
                )
                break
    for count_col in ("modeled_analytic_n", "event_n", "membership_n"):
        if count_col not in source_data.columns:
            continue
        numeric = pd.to_numeric(source_data[count_col], errors="coerce")
        finite = numeric.dropna()
        if finite.empty or ((finite % 1) == 0).all():
            source_data[count_col] = numeric.astype("Int64")
    if "display_label" not in source_data.columns:
        source_data["display_label"] = source_data["spec_id"].map(_publication_label)
    if "axis" not in source_data.columns:
        source_data["axis"] = "sensitivity"
    if "converged" not in source_data.columns:
        source_data["converged"] = source_data["point_estimate"].notna()
    source_data["axis_label"] = source_data["axis"].map(_publication_label)
    source_data["plot_label"] = [
        _sensitivity_plot_label(row) for row in source_data.to_dict(orient="records")
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    estimated_mask = (
        source_data[["point_estimate", "ci_low", "ci_high"]].notna().all(axis=1)
    )
    if "converged" in source_data.columns:
        estimated_mask &= source_data["converged"].map(_truthy_figure_value)
    if "reportable" in source_data.columns:
        estimated_mask &= source_data["reportable"].map(_truthy_figure_value)
    if "independent_variant" in source_data.columns:
        independent = source_data["independent_variant"]
        estimated_mask &= ~independent.map(_explicit_false_figure_value)
    plot_df = source_data.loc[estimated_mask].copy()
    if plot_df.empty:
        return None
    ratio_df = plot_df[
        plot_df["effect_scale"].astype(str).str.upper().isin({"OR", "RR", "HR"})
    ].copy()
    additive_scales = {
        "RD",
        "RISK_DIFFERENCE",
        "MEAN_DIFFERENCE",
        "MEDIAN_DIFFERENCE",
        "CONDITIONAL_MEAN_DIFFERENCE",
        "CONDITIONAL_MEDIAN_DIFFERENCE",
    }
    rd_df = plot_df[
        plot_df["effect_scale"].astype(str).str.upper().isin(additive_scales)
    ].copy()
    plotted_indexes = ratio_df.index.union(rd_df.index)
    figure_source_data = source_data.loc[plotted_indexes].copy()
    estimability_source_data = source_data.drop(index=plotted_indexes).copy()
    estimability_source_data = estimability_source_data.drop(
        columns=[
            "modeled_analytic_n",
            "model_contract_n",
            "event_n",
            "model_id",
            "source_model_id",
            "exposure_source",
            "exposure_expression",
            "exposure_role",
            "analysis_role",
            "analysis_set",
            "baseline_missing_policy",
            "fit_status",
            "fit_method",
            "replay_mode",
            "coefficient_source_table",
            "coefficient_term",
            "model_contract_source",
            "source_script_sha256",
        ],
        errors="ignore",
    )
    figure_source_data.to_csv(
        out_dir / "sensitivity_forest_source_data.csv",
        index=False,
    )
    estimability_source_filename: Optional[str] = None
    if not estimability_source_data.empty:
        estimability_source_filename = "sensitivity_estimability_source_data.csv"
        estimability_source_data.to_csv(
            out_dir / estimability_source_filename,
            index=False,
        )
    if not rd_df.empty:
        rd_df["plot_label"] = [
            _sensitivity_plot_label(row) for row in rd_df.to_dict(orient="records")
        ]
    n_df = figure_source_data.copy()
    if "modeled_analytic_n" in n_df.columns:
        n_df["modeled_analytic_n"] = pd.to_numeric(
            n_df["modeled_analytic_n"],
            errors="coerce",
        )
    else:
        n_df["modeled_analytic_n"] = pd.NA
    n_df = n_df[n_df["modeled_analytic_n"].fillna(0).gt(0)].copy()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from easyicu.research_agent.figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    if ratio_df.empty and rd_df.empty:
        return None
    n_plot = n_df.dropna(subset=["modeled_analytic_n"]).copy()
    max_rows = max(len(ratio_df), len(rd_df), len(n_plot), 1)
    figure_height_mm = float(max(88, min(145, 24 + 15 * max_rows)))
    fig = plt.figure(
        figsize=(183 / 25.4, figure_height_mm / 25.4),
        constrained_layout=False,
    )
    ax_ratio = None
    ax_rd = None
    ax_n = None
    if not ratio_df.empty and not rd_df.empty:
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.42, 0.92],
            height_ratios=[1.0, 0.82],
            left=0.28,
            right=0.98,
            top=0.92,
            bottom=0.17,
            wspace=0.78,
            hspace=0.62,
        )
        ax_ratio = fig.add_subplot(grid[:, 0])
        ax_rd = fig.add_subplot(grid[0, 1])
        if not n_plot.empty:
            ax_n = fig.add_subplot(grid[1, 1])
    else:
        has_denominator = not n_plot.empty
        grid = fig.add_gridspec(
            1,
            2 if has_denominator else 1,
            width_ratios=[1.42, 0.92] if has_denominator else [1.0],
            left=0.25 if has_denominator else 0.30,
            right=0.98,
            top=0.90,
            bottom=0.20,
            wspace=0.82,
        )
        effect_axis = fig.add_subplot(grid[0, 0])
        if not ratio_df.empty:
            ax_ratio = effect_axis
        else:
            ax_rd = effect_axis
        if has_denominator:
            ax_n = fig.add_subplot(grid[0, 1])

    def _plot_interval_panel(
        ax,
        data: pd.DataFrame,
        *,
        title: str,
        xlabel: str,
        null_value: float,
        color: str,
    ) -> None:
        data = data.reset_index(drop=True)
        y = list(range(len(data)))
        center = data["point_estimate"].astype(float).to_numpy()
        lo = data["ci_low"].astype(float).to_numpy()
        hi = data["ci_high"].astype(float).to_numpy()
        labels = [
            _short_figure_label(label)
            for label in data["plot_label"].fillna(data["display_label"]).astype(str)
        ]
        ax.errorbar(
            center,
            y,
            xerr=[
                [max(0.0, c - lower) for c, lower in zip(center, lo)],
                [max(0.0, h - c) for c, h in zip(center, hi)],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            markersize=3.9,
        )
        ax.axvline(
            null_value,
            color=palette.get("neutral", "#8F8F8F"),
            linestyle="--",
            linewidth=0.8,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", pad=4)
        ax.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
            alpha=0.8,
        )

    contract_panels: List[Dict[str, Any]] = []
    next_panel_ord = ord("A")

    def _next_panel_id() -> str:
        nonlocal next_panel_ord
        panel_id = chr(next_panel_ord)
        next_panel_ord += 1
        return panel_id

    source_evidence = ["sensitivity_forest_source_data.csv"]
    all_source_evidence = list(source_evidence)
    if estimability_source_filename:
        all_source_evidence.append(estimability_source_filename)
    if ax_ratio is not None:
        ratio_scales = sorted(
            set(ratio_df["effect_scale"].dropna().astype(str).str.upper())
        )
        ratio_xlabel = {
            ("OR",): "Adjusted odds ratio (95% CI)",
            ("RR",): "Adjusted risk ratio (95% CI)",
            ("HR",): "Hazard ratio (95% CI)",
        }.get(tuple(ratio_scales), "Ratio estimate (95% CI)")
        panel_id = _next_panel_id()
        _plot_interval_panel(
            ax_ratio,
            ratio_df,
            title="Ratio-scale sensitivity",
            xlabel=ratio_xlabel,
            null_value=1.0,
            color=palette.get("blue", "#0F4D92"),
        )
        add_panel_label(ax_ratio, panel_id, x=-0.24)
        contract_panels.append(
            {
                "panel_id": panel_id,
                "title": "Ratio-scale sensitivity",
                "role": "robustness",
                "claim": (
                    "Converged, independently estimable ratio-scale sensitivity "
                    "estimates are read from the registered parent table."
                ),
                "evidence_ids": source_evidence,
            }
        )

    if ax_rd is not None:
        additive_values = sorted(
            set(rd_df["effect_scale"].dropna().astype(str).str.upper())
        )
        if set(additive_values) <= {"RD", "RISK_DIFFERENCE"}:
            additive_title = "Risk-difference sensitivity"
            additive_xlabel = "Risk difference (95% CI)"
        elif additive_values and all("MEDIAN" in value for value in additive_values):
            additive_title = "Median-difference sensitivity"
            additive_xlabel = "Adjusted median difference (95% CI)"
        elif additive_values and all("MEAN" in value for value in additive_values):
            additive_title = "Mean-difference sensitivity"
            additive_xlabel = "Adjusted mean difference (95% CI)"
        else:
            additive_title = "Additive-scale sensitivity"
            additive_xlabel = "Adjusted difference (95% CI)"
        panel_id = _next_panel_id()
        _plot_interval_panel(
            ax_rd,
            rd_df,
            title=additive_title,
            xlabel=additive_xlabel,
            null_value=0.0,
            color=palette.get("green", "#008B5E"),
        )
        add_panel_label(ax_rd, panel_id, x=-0.24, y=1.06, fontsize=10.0)
        contract_panels.append(
            {
                "panel_id": panel_id,
                "title": additive_title,
                "role": "robustness",
                "claim": (
                    "Converged, reportable additive-scale sensitivity estimates "
                    "are shown on their declared scale."
                ),
                "evidence_ids": source_evidence,
            }
        )

    non_independent_count = 0
    if "independent_variant" in estimability_source_data.columns:
        non_independent_count = int(
            estimability_source_data["independent_variant"]
            .map(_explicit_false_figure_value)
            .sum()
        )
    if ax_n is not None:
        n_plot = n_plot.reset_index(drop=True)
        y_n = list(range(len(n_plot)))
        colors = [
            (
                palette.get("blue", "#0F4D92")
                if _truthy_figure_value(value)
                else palette.get("neutral_light", "#D8D8D8")
            )
            for value in n_plot["converged"].fillna(False)
        ]
        ax_n.barh(
            y_n,
            n_plot["modeled_analytic_n"].astype(float),
            color=colors,
            height=0.56,
        )
        ax_n.set_yticks(y_n)
        ax_n.set_yticklabels(
            [
                _short_figure_label(label, limit=26)
                for label in n_plot["plot_label"]
                .fillna(n_plot["display_label"])
                .astype(str)
            ]
        )
        ax_n.invert_yaxis()
        if non_independent_count:
            # Reserve an in-axis status row below the final denominator bar;
            # placing the note at a negative axes fraction pushes SVG text
            # outside the canvas even when the raster preview looks acceptable.
            ax_n.set_ylim(len(n_plot) + 0.75, -0.6)
        event_values = (
            pd.to_numeric(n_plot["event_n"], errors="coerce")
            if "event_n" in n_plot.columns
            else pd.Series([pd.NA] * len(n_plot))
        )
        max_n = float(n_plot["modeled_analytic_n"].max())
        if event_values.notna().any():
            for row_index, (analytic_n, event_n) in enumerate(
                zip(n_plot["modeled_analytic_n"], event_values)
            ):
                if pd.isna(event_n):
                    continue
                ax_n.text(
                    float(analytic_n) + max_n * 0.015,
                    row_index,
                    f"{int(event_n):,} events",
                    va="center",
                    fontsize=6.0,
                    color=palette.get("baseline", "#272727"),
                )
            ax_n.set_xlim(0, max_n * 1.29)
            ax_n.set_xlabel("Analytic sample size")
        else:
            ax_n.set_xlabel("Analytic sample size")
        ax_n.set_title("Model denominator audit", loc="left", pad=4)
        ax_n.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
            alpha=0.8,
        )
        if non_independent_count:
            ax_n.text(
                0.0,
                len(n_plot) + 0.25,
                f"Non-independent outcome variants: {non_independent_count}",
                ha="left",
                va="center",
                fontsize=6.0,
                color=palette.get("neutral", "#8F8F8F"),
            )
        panel_id = _next_panel_id()
        add_panel_label(ax_n, panel_id, x=-0.24, y=1.06, fontsize=10.0)
        contract_panels.append(
            {
                "panel_id": panel_id,
                "title": "Model denominator audit",
                "role": "audit",
                "claim": (
                    "Positive analytic sample sizes and available event counts "
                    "are shown for fitted sensitivity models; non-independent "
                    "variants are reported separately rather than encoded as N=0."
                ),
                "evidence_ids": all_source_evidence,
            }
        )

    for panel in contract_panels:
        panel_role = str(panel.get("role") or "")
        if panel_role == "robustness":
            panel["metadata"] = {"planner_product_slots": ["robustness_plot"]}
        elif panel_role == "audit":
            panel["metadata"] = {
                "planner_product_slots": ["robustness_denominator_audit"]
            }

    contract = make_figure_contract(
        figure_id="sensitivity_forest",
        core_claim=(
            "Pre-specified sensitivity estimates are rendered from the "
            "registered sensitivity-comparison table with effect-scale and "
            "denominator context."
        ),
        panels=contract_panels,
        height_mm=figure_height_mm,
        source_data=all_source_evidence,
        statistics_note=(
            "Generated deterministically from the registered parent-step "
            "sensitivity-comparison table after the rendering step lacked a "
            "canonical figure contract."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "sensitivity_forest",
        contract=contract,
        dpi=300,
    )
    plt.close(fig)

    step_summary_path = out_dir / "step_summary.json"
    existing_summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_summary = loaded
        except Exception:
            existing_summary = {}
    existing_summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_sensitivity_publication_figure_repair",
            "rendering_only": True,
            "source_step_id": source_step_id,
            "source_sensitivity_table": str(table_path),
            "source_data_csv": str(out_dir / "sensitivity_forest_source_data.csv"),
            "source_data_files": all_source_evidence,
            "n_rows_plotted": int(len(figure_source_data)),
            "n_denominator_rows": int(len(n_plot)),
            "n_non_independent_variants": non_independent_count,
            "source_model_ids": sorted(
                set(
                    figure_source_data.get("model_id", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                )
            ),
            "effect_scales_plotted": sorted(
                set(figure_source_data["effect_scale"].dropna().astype(str))
            ),
            "figure_files": [
                path.name for key, path in outputs.items() if key != "contract"
            ],
            "figure_path": "sensitivity_forest.png",
            "figure_contract": "sensitivity_forest.figure_contract.json",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if table_path.name == "robustness_summary.csv" and (
        authorized_repair_id == "sensitivity_publication_bundle_from_locked_summary_v1"
        or _resolve_upstream_analysis_method(run_dir, current_step_id)
        == "cohort_definition_sensitivity"
    ):
        return "sensitivity_publication_bundle_from_locked_summary_v1"
    return "sensitivity_publication_bundle_from_parent_outputs_v2"


def deterministic_figure_family_supported_for_upstream(
    run_dir: Path, step_id: str
) -> bool:
    """Compatibility boolean for the typed automatic-renderer gate."""

    return deterministic_figure_repair_id_for_upstream(run_dir, step_id) is not None


def _renderer_for_upstream_method(method: Optional[str]):
    """Map an exact controlled parent method to a deterministic renderer."""

    key = _UPSTREAM_METHOD_TO_RENDERER_KEY.get(str(method or "").strip().lower())
    if key == "ordered_distribution":
        from .figures.ordered_distribution import (
            render_ordered_distribution_bundle_from_prior_outputs,
        )

        return render_ordered_distribution_bundle_from_prior_outputs
    if key == "distribution_availability":
        from .figures.distribution_availability import (
            render_distribution_availability_bundle_from_prior_outputs,
        )

        return render_distribution_availability_bundle_from_prior_outputs
    return {
        "sensitivity": _render_sensitivity_publication_bundle_from_prior_outputs,
        "missingness": _render_missingness_publication_bundle_from_prior_outputs,
    }.get(key)


def _render_authorized_sealed_publication_bundle(
    *,
    repair_id: str,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    parent_artifact_digests: Mapping[str, str],
) -> Optional[str]:
    """Render one host-selected closed adapter from one immutable byte snapshot.

    This dispatcher never chooses a scientific method or probes alternative
    renderers.  The host has already authorized the exact repair ID from the
    direct parent's registered method/family and evidence digests.
    """

    from .contracts.declared_product import read_digest_bound_artifact_snapshot

    parent_step_id = str(current_step_id or "").removesuffix("_figure")
    if not parent_step_id or parent_step_id == str(current_step_id or ""):
        return None
    parent_out = Path(run_dir) / "steps" / parent_step_id / "outputs"
    try:
        snapshot = read_digest_bound_artifact_snapshot(
            parent_out=parent_out,
            artifact_digests=parent_artifact_digests,
        )
    except ValueError:
        return None
    if "step_summary.json" not in snapshot:
        return None

    adapter = sealed_renderer_adapter(repair_id)
    if adapter:
        observed = adapter.render(run_dir, current_step_id, out_dir, snapshot)
    elif repair_id == "absolute_risk_incidence_prevalence_publication_bundle_v1":
        from .figures.absolute_risk import (
            render_absolute_risk_bundle_from_prior_outputs,
        )

        observed = render_absolute_risk_bundle_from_prior_outputs(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            preverified_parent_artifacts=snapshot,
        )
    elif repair_id == "ordered_category_distribution_publication_bundle_v1":
        from .figures.ordered_distribution import (
            render_ordered_distribution_bundle_from_prior_outputs,
        )

        observed = render_ordered_distribution_bundle_from_prior_outputs(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            preverified_parent_artifacts=snapshot,
        )
    elif repair_id == "cohort_flow_publication_bundle_from_parent_outputs_v1":
        observed = _render_cohort_flow_publication_bundle_from_prior_outputs(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            preverified_parent_artifacts=snapshot,
        )
    elif repair_id == "sensitivity_publication_bundle_from_locked_summary_v1":
        observed = _render_sensitivity_publication_bundle_from_prior_outputs(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            preverified_parent_artifacts=snapshot,
            authorized_repair_id=repair_id,
        )
    elif repair_id == ("association_publication_bundle_from_planned_model_contract_v1"):
        observed = _render_association_publication_bundle_from_prior_outputs(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            preverified_parent_artifacts=snapshot,
            authorized_repair_id=repair_id,
        )
    else:
        return None
    return observed if observed == repair_id else None


# ---------------------------------------------------------------------------
# Semantic-alias map
# ---------------------------------------------------------------------------
#
# This is the bridge between the writer agent's stable placeholder
# vocabulary (``{evidence:table_one}``, ``{evidence:outcome_rate}``)
# and the per-run, hash-suffixed evidence_ids the EvidenceStore
# actually emits. The map below is intentionally small and
# inspectable; new writer placeholders should be added here rather
# than hidden inside ad-hoc string surgery elsewhere in the pipeline.

# A mapping of (step_id_substring, artefact_basename) -> tuple of aliases.
# step_id_substring is matched as ``in step.step_id`` so that step ids
# the planner generates with arbitrary ordering ("01_table_one",
# "02_outcome_incidence", "04_primary_association")
# all resolve correctly.
_SEMANTIC_ALIAS_MAP: Dict[tuple, tuple] = {
    # Cohort/table-one summaries often carry the same mortality
    # incidence statistic that manuscript scaffolds cite as outcome_rate.
    ("cohort_summary", "step_summary.json"): (
        "cohort_summary",
        "outcome_rate",
        "mortality_rate",
        "outcome_incidence",
    ),
    ("table_one", "step_summary.json"): (
        "table_one",
        "cohort_summary",
        "mortality_rate",
    ),
    # Outcome incidence summary should also answer to "outcome_rate".
    ("outcome_incidence", "step_summary.json"): ("outcome_rate", "outcome_incidence"),
    ("stratified_incidence", "step_summary.json"): (
        "stratified_mortality",
        "outcome_rate",
        "mortality_rate",
        "primary_association",
    ),
    ("stratified_mortality", "step_summary.json"): (
        "stratified_mortality",
        "outcome_rate",
        "mortality_rate",
        "primary_association",
    ),
    ("correlation_analysis", "step_summary.json"): (
        "primary_association",
        "correlation_summary",
        "spearman_correlation",
        "model_performance",
    ),
    ("visualization", "step_summary.json"): (
        "correlation_summary",
        "spearman_correlation",
    ),
    ("missing_data_audit", "step_summary.json"): (
        "missingness",
        "outcome_rate",
        "mortality_rate",
    ),
    # The primary-association step_summary doubles as the OR / model
    # statistic the writer points at.
    ("primary_association", "step_summary.json"): ("primary_association",),
    ("association_model", "step_summary.json"): (
        "primary_association",
        "association_model",
        "primary_association_table",
    ),
    ("sensitivity_complete_case", "step_summary.json"): (
        "primary_association",
        "complete_case_model",
        "kdigo_sensitivity",
    ),
    ("sensitivity_reduced_vars", "step_summary.json"): (
        "primary_association",
        "association_model",
        "reduced_model",
        "kdigo_sensitivity",
    ),
    ("sensitivity_reduced", "step_summary.json"): (
        "primary_association",
        "association_model",
        "reduced_model",
        "kdigo_sensitivity",
    ),
    ("model_fitting_complete_case", "step_summary.json"): (
        "primary_association",
        "complete_case_model",
    ),
    ("model_fitting_missing_indicator", "step_summary.json"): (
        "primary_association",
        "missing_indicator_model",
    ),
    ("model_fitting_reduced_variables", "step_summary.json"): (
        "primary_association",
        "reduced_variable_model",
    ),
    ("complete_case_robustness", "step_summary.json"): (
        "primary_association",
        "outcome_rate",
        "mortality_rate",
        "robustness_summary",
    ),
    ("composite", "step_summary.json"): ("primary_association",),
    ("mortality_association", "step_summary.json"): (
        "primary_association",
        "outcome_rate",
        "mortality_rate",
    ),
    # Generic table outputs from any step.
    ("", "table_one.csv"): ("table_one",),
    ("", "missingness.csv"): ("missingness",),
    ("", "sofa2_stratum_balance.csv"): ("primary_association_table",),
    ("", "stratified_mortality_incidence.csv"): (
        "stratified_mortality_incidence",
        "stratified_mortality",
        "outcome_rate",
        "mortality_rate",
    ),
    ("", "primary_association.csv"): ("primary_association_table",),
    ("", "correlation_matrix.csv"): (
        "correlation_matrix",
        "correlation_summary",
        "primary_association_table",
    ),
    ("", "cluster_characteristics.csv"): (
        "cluster_characteristics",
        "cluster_summary",
        "table_one",
    ),
    ("", "cluster_mortality.csv"): ("cluster_mortality", "outcome_rate"),
    ("", "clustering_algorithm_details.json"): ("clustering_algorithm_details",),
    ("", "clustering_methodology.json"): ("clustering_methodology",),
    # Figures.
    ("", "mortality_by_sofa2_stratum.png"): (
        "mortality_by_sofa2_stratum",
        "figure_mortality_by_sofa2_stratum",
    ),
    ("", "missingness_heatmap.png"): ("missingness_heatmap",),
    ("", "primary_association_curve.png"): ("primary_association_figure",),
    ("", "sofa2_correlation_heatmap.png"): (
        "sofa2_correlation_heatmap",
        "correlation_heatmap",
        "correlation_figure",
    ),
    ("", "correlation_heatmap.png"): (
        "correlation_heatmap",
        "sofa2_correlation_heatmap",
        "correlation_figure",
    ),
    ("", "clustering_visualization.png"): (
        "clustering_visualization",
        "clustering_figure",
    ),
    ("", "survival_curves.png"): (
        "survival_curves",
        "survival_figure",
        "kaplan_meier",
    ),
    ("", "time_varying_discrimination.png"): (
        "time_varying_discrimination",
        "dynamic_prediction_figure",
    ),
    ("", "covariate_balance.png"): (
        "covariate_balance",
        "love_plot",
        "causal_balance_figure",
    ),
    ("", "subgroup_forest.png"): (
        "subgroup_forest",
        "treatment_response_figure",
    ),
    ("", "external_validation.png"): (
        "external_validation",
        "validation_figure",
    ),
}


def _semantic_aliases_for(step: AnalysisStep, artefact: Path) -> List[str]:
    """Return the semantic aliases for an artefact registered under a step.

    First-write-wins on the EvidenceStore side: if multiple steps emit
    a ``table_one.csv``, only the first registration claims the
    ``table_one`` alias. That matches the convention that the
    Methods/Results scaffold cites Table 1 once.
    """
    out: List[str] = []
    if artefact.name == "step_summary.json":
        step_id = step.step_id or ""
        if step_id:
            out.append(step_id)
            stripped = re.sub(r"^\d+[_-]+", "", step_id)
            if stripped and stripped != step_id:
                out.append(stripped)
        expected = " ".join(str(item).lower() for item in (step.expected_outputs or []))
        intent = (step.intent or "").lower()
        if _prediction_contract_applies(step):
            out.extend(
                [
                    "model_performance",
                    "prediction_performance",
                    "baseline_prevalence",
                ]
            )
        if _clustering_contract_applies(
            method=str(step.method or ""),
            step_id=step_id,
            intent=intent,
            expected_outputs=step.expected_outputs or [],
        ):
            out.append("clustering_performance")
            if not (artefact.parent / "cluster_characteristics.csv").exists():
                out.extend(["cluster_summary", "cluster_characteristics", "table_one"])
            if not (artefact.parent / "cluster_mortality.csv").exists():
                out.append("cluster_mortality")
            if not any(
                (artefact.parent / name).exists()
                for name in (
                    "clustering_methodology.json",
                    "clustering_algorithm_details.json",
                )
            ):
                out.append("clustering_methodology")
        if (
            "robustness" in expected
            or "robustness" in intent
            or "complete_case_robustness" in step_id.lower()
        ):
            out.extend(["outcome_rate", "mortality_rate", "robustness_summary"])
            if _step_summary_has_primary_effect(artefact):
                out.append("primary_association")
        if ("table_one" in step_id.lower() or "table:table_one" in expected) and not (
            artefact.parent / "table_one.csv"
        ).exists():
            out.append("table_one")
        if "report" in step_id.lower() and _step_summary_has_any_key(
            artefact,
            ("mortality_rate", "outcome_rate", "death_rate", "icu_mortality_rate"),
        ):
            out.extend(
                [
                    "outcome_rate",
                    "mortality_rate",
                    "cohort_summary",
                ]
            )
    for (step_substr, basename), aliases in _SEMANTIC_ALIAS_MAP.items():
        if basename != artefact.name:
            continue
        if step_substr and step_substr not in (step.step_id or "").lower():
            continue
        out.extend(aliases)
    if (
        artefact.name == "clustering_algorithm_details.json"
        and not (artefact.parent / "clustering_methodology.json").exists()
    ):
        out.append("clustering_methodology")
    return out


def _step_summary_has_primary_effect(path: Path) -> bool:
    """Return True when an existing step_summary has a finite primary effect.

    Robustness/missingness steps frequently mention "robustness" in their
    intent but are not themselves primary association artefacts. The
    ``primary_association`` alias should therefore be attached only when the
    registered summary actually contains an effect estimate. For synthetic
    tests that pass a non-existent path, preserve the historical aliasing
    behaviour by returning True.
    """

    if path.name != "step_summary.json" or not path.exists():
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False

    def _finite(value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    direct_keys = {
        "primary_or",
        "odds_ratio",
        "adjusted_or",
        "estimate",
        "primary_association_estimate",
        "primary_point_estimate",
    }
    for key in direct_keys:
        value = payload.get(key)
        if _finite(value):
            return True
        if isinstance(value, dict) and any(_finite(v) for v in value.values()):
            return True
    for value in payload.values():
        if isinstance(value, dict):
            association = value.get("primary_association")
            if isinstance(association, dict) and any(
                _finite(association.get(k))
                for k in ("odds_ratio", "or", "estimate", "adjusted_or")
            ):
                return True
    return False


def _step_summary_has_any_key(path: Path, keys: Sequence[str]) -> bool:
    """Return True when an existing step_summary contains any requested key."""

    if path.name != "step_summary.json" or not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in keys)


__all__ = ["ResearchAgentPipeline"]


def _pipeline_run___plan_invoker(
    *,
    _emit_progress: Any,
    cohort_name: Any,
    cohort_path: Any,
    concept_descriptions: Any,
    cross_database_validation: Any,
    database: Any,
    endpoint: Any,
    exclusion_criteria: Any,
    experiment_spec_path: Any,
    id_columns: Any,
    inclusion_criteria: Any,
    llm: Any,
    notes: Any,
    outcome_columns: Any,
    primary_exposure: Any,
    question: Any,
    resume_context_evidence_path: Any,
    resume_from_step_id: Any,
    resume_state: Any,
    run_dir: Any,
    run_environment_identity: Any,
    run_id: Any,
    run_language: Any,
    run_scientific_identity: Any,
    self: Any,
    skill_obj: Any,
    staged_trajectory_binding: Any,
    target_outcome: Any,
    time_columns: Any,
    time_windows: Any,
    user_preferences: Any,
):
    return self._run_plan_phase(
        question=question,
        cohort_path=cohort_path,
        cohort_name=cohort_name,
        database=database,
        target_outcome=target_outcome,
        endpoint=endpoint,
        primary_exposure=primary_exposure,
        cross_database_validation=cross_database_validation,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        id_columns=id_columns,
        time_columns=time_columns,
        outcome_columns=outcome_columns,
        time_windows=time_windows,
        concept_descriptions=concept_descriptions,
        user_preferences=user_preferences,
        notes=notes,
        skill_obj=skill_obj,
        llm=llm,
        run_dir=run_dir,
        run_id=run_id,
        run_language=run_language,
        experiment_spec_path=experiment_spec_path,
        resume_state=resume_state,
        resume_context_evidence_path=resume_context_evidence_path,
        trajectory_binding=staged_trajectory_binding,
        run_scientific_identity=run_scientific_identity,
        run_environment_identity=run_environment_identity,
        resume_from_step_id=resume_from_step_id,
        emit_progress=_emit_progress,
    )


def _pipeline_run___provenance_hook(
    plan_result,
    *,
    cohort_path: Any,
    orchestration_receipt_path: Any,
    run_dir: Any,
    self: Any,
    source_files: Any,
):
    try:
        provenance = build_provenance_bundle(
            cohort_path=cohort_path,
            source_files=source_files,
        )
        provenance_path = run_dir / "provenance_sources.json"
        provenance.to_disk(provenance_path)
        if plan_result.evidence.get("provenance_sources") is None:
            plan_result.evidence.register_file(
                kind="log",
                description=(
                    "Raw EHR and cohort provenance hashes (O27): "
                    "sha256 + size + mtime for every source file "
                    "and the materialised cohort parquet."
                ),
                source_path=provenance_path,
                evidence_id="provenance_sources",
                producer="pipeline",
                generation_mode="system",
            )
    except Exception as exc:
        # A paper-facing profile claims the full raw EHR → cohort →
        # analysis → manuscript chain. If the first link cannot be
        # hashed, that claim is unsupported, so the run stops instead
        # of finishing with a warning nobody reads. Development runs
        # keep the warning: they make no provenance claim.
        from .orchestration.profiles import is_paper_facing_profile

        paper_facing = is_paper_facing_profile(self._submission_profile_name)
        plan_result.findings.append(
            ValidationFinding(
                validator="provenance",
                severity="error" if paper_facing else "warning",
                message=(
                    f"Failed to compute raw-EHR provenance bundle: "
                    f"{type(exc).__name__}: {exc}"
                ),
                detail={
                    "reason": "raw_ehr_provenance_unavailable",
                    "submission_profile_name": self._submission_profile_name,
                },
            )
        )
        if paper_facing:
            raise RuntimeError(
                "raw-EHR provenance could not be computed under submission "
                f"profile {self._submission_profile_name!r}; the "
                "manuscript's provenance chain would be incomplete"
            ) from exc
    if plan_result.evidence.get("orchestration_runtime") is None:
        plan_result.evidence.register_file(
            kind="log",
            description=(
                "Non-scientific phase-dispatch runtime identity; EasyICU "
                "receipts/capsules/checkpoints remain authoritative."
            ),
            source_path=orchestration_receipt_path,
            evidence_id="orchestration_runtime",
            producer="pipeline",
            generation_mode="system",
        )


def _pipeline_run___execute_invoker(
    plan_result,
    *,
    _emit_progress: Any,
    cohort_path: Any,
    notes: Any,
    resume_from_step_id: Any,
    run_dir: Any,
    run_id: Any,
    self: Any,
    skill_obj: Any,
    staged_trajectory_binding: Any,
    stop_after_step_id: Any,
):
    return self._run_execute_phase(
        plan_result=plan_result,
        cohort_path=cohort_path,
        trajectory_binding=staged_trajectory_binding,
        run_dir=run_dir,
        run_id=run_id,
        skill_obj=skill_obj,
        notes=notes,
        emit_progress=_emit_progress,
        resume_from_step_id=resume_from_step_id,
        stop_after_step_id=stop_after_step_id,
    )


def _pipeline_run___write_invoker(
    plan_result,
    execute_result,
    *,
    _emit_progress: Any,
    force_writer_probe: Any,
    manuscript_authors: Any,
    manuscript_title: Any,
    run_dir: Any,
    run_id: Any,
    run_language: Any,
    self: Any,
    stop_after_analysis: Any,
):
    try:
        return self._run_write_phase(
            plan_result=plan_result,
            execute_result=execute_result,
            run_dir=run_dir,
            run_id=run_id,
            stop_after_analysis=stop_after_analysis,
            manuscript_title=manuscript_title,
            manuscript_authors=manuscript_authors,
            run_language=run_language,
            emit_progress=_emit_progress,
            force_writer_probe=force_writer_probe,
        )
    except EvidenceEnforcementError as exc:
        validator = (
            "manuscript_numeric_auditor"
            if "untraced" in getattr(exc, "detail", {})
            else "evidence_bound_writer"
        )
        plan_result.findings.append(
            ValidationFinding(
                validator=validator,
                severity="error",
                message=(
                    f"STRICT evidence enforcement blocked manuscript generation: {exc}"
                ),
                detail=getattr(exc, "detail", {}) or None,
            )
        )
        bound_path = run_dir / "manuscript_scaffold_bound.md"
        bound_path.write_text(
            "# Manuscript scaffold not generated\n\n"
            "STRICT evidence enforcement failed before final binding.\n\n"
            f"Error: {exc}\n",
            encoding="utf-8",
        )
        _emit_progress(
            "writer",
            "STRICT evidence enforcement blocked manuscript generation.",
            status="error",
            run_id=run_id,
        )
        return _WritePhaseResult(literature=None, bound_path=bound_path)


def _pipeline_run___finalise_invoker(
    plan_result,
    execute_result,
    write_result,
    *,
    _emit_progress: Any,
    audit_logger: Any,
    cache_key: Any,
    cohort_path: Any,
    database: Any,
    experiment_spec_path: Any,
    notes: Any,
    run_dir: Any,
    run_id: Any,
    run_scientific_identity: Any,
    self: Any,
    stop_after_analysis: Any,
    target_outcome: Any,
):
    return self._finalise_success(
        plan_result=plan_result,
        execute_result=execute_result,
        write_result=write_result,
        run_id=run_id,
        run_dir=run_dir,
        cohort_path=cohort_path,
        notes=notes,
        database=database,
        target_outcome=target_outcome,
        stop_after_analysis=stop_after_analysis,
        cache_key=cache_key,
        scientific_identity=run_scientific_identity,
        experiment_spec_path=experiment_spec_path,
        audit_logger=audit_logger,
        emit_progress=_emit_progress,
    )


def _pipeline_run___review_evidence_store(
    *, reviewed_plan: Any, run_dir: Any, self: Any
):
    if reviewed_plan:
        return reviewed_plan[-1].evidence
    return EvidenceStore(run_dir, enforcement_mode=self._evidence_enforcement_mode)


def _pipeline_run___human_review_invoker(plan_result, *, reviewed_plan: Any, self: Any):
    from .orchestration.workflow import human_review_requests_for_plan

    reviewed_plan.append(plan_result)
    plan_evidence = getattr(plan_result, "evidence", None)
    capsule_record = (
        plan_evidence.get(RUN_INPUT_CAPSULE_EVIDENCE_ID)
        if plan_evidence is not None
        else None
    )
    if capsule_record is None:
        requests_without_execution = human_review_requests_for_plan(
            findings=plan_result.findings,
            plan=plan_result.plan,
            evidence=plan_evidence,
            require_plan_review=self._config.require_human_plan_review,
        )
        if not requests_without_execution:
            return requests_without_execution
        raise RuntimeError(
            "human review cannot bind execution identity because the "
            "run input capsule is absent"
        )
    submission_profile_ref = (
        self._submission_profile_ref
        if self._submission_profile_name and self._submission_profile_version
        else None
    )
    execution_authority = ReviewExecutionAuthority(
        pipeline_config_sha256=self._config.canonical_digest(),
        submission_profile_ref=submission_profile_ref,
        capability_activation_sha256=canonical_sha256(
            self._capability_runtime.activation.model_dump(mode="json")
            if self._capability_runtime.activation is not None
            else None
        ),
        run_input_capsule_sha256=str(capsule_record.sha256),
    )
    requests = human_review_requests_for_plan(
        findings=plan_result.findings,
        plan=plan_result.plan,
        evidence=plan_evidence,
        execution_authority=execution_authority,
        require_plan_review=self._config.require_human_plan_review,
    )
    return requests


def _pipeline_run___human_review_recorder(
    records, *, _review_evidence_store: Any, run_dir: Any, run_id: Any, self: Any
):
    self._record_human_review_records(
        records,
        run_id=run_id,
        run_dir=run_dir,
        evidence=_review_evidence_store(),
    )


def _pipeline_run___prepare_human_review_execution(
    decision_records: Sequence[Mapping[str, Any]], *, checkpoint_commit: Any
) -> None:
    path = checkpoint_commit.get("path")
    if path is None:
        return
    prepare_human_review_decision(
        checkpoint_file=Path(path),
        decision_payloads=checkpoint_commit.get("decision_payloads") or (),
        decision_records=decision_records,
        decision_sha256=str(checkpoint_commit.get("decision_sha256") or ""),
    )


def _pipeline_run___commit_human_review_execution(
    decision_records: Sequence[Mapping[str, Any]],
    *,
    checkpoint_commit: Any,
    reviewed_plan: Any,
    run_dir: Any,
) -> None:
    path = checkpoint_commit.get("path")
    if path is None:
        return
    if not reviewed_plan:
        raise HumanReviewCheckpointError(
            "approved execution has no restored typed plan handoff"
        )
    commit_human_review_decision(
        checkpoint_file=Path(path),
        run_dir=run_dir,
        evidence=reviewed_plan[-1].evidence,
        plan_revision=reviewed_plan[-1].plan.revision,
        decision_payloads=checkpoint_commit.get("decision_payloads") or (),
        decision_records=decision_records,
        decision_sha256=str(checkpoint_commit.get("decision_sha256") or ""),
    )


def _pipeline_run___commit_human_review_execution_start(
    *, checkpoint_commit: Any
) -> None:
    path = checkpoint_commit.get("path")
    if path is None:
        return
    mark_human_review_execution_started(Path(path))


def _pipeline_run___commit_human_review_write_start(*, checkpoint_commit: Any) -> None:
    path = checkpoint_commit.get("path")
    if path is not None:
        mark_human_review_execution_phase(Path(path), "write_in_progress")


def _pipeline_run___commit_human_review_finalize_start(
    *, checkpoint_commit: Any
) -> None:
    path = checkpoint_commit.get("path")
    if path is not None:
        mark_human_review_execution_phase(Path(path), "finalize_in_progress")
