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
import logging
import math
import re
import shutil
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

from .agents.core import (
    AnalyzerAgent,
    ClinicalSemanticsAgent,
    CoderAgent,
    CriticAgent,
    DataExtractionAgent,
    ManuscriptAgent,
    PlannerAgent,
    ReplannerAgent,
    RuntimeSupervisor,
    StatisticalAnalysisAgent,
    VisualizationAgent,
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
from .figures.distribution_availability import (
    _distribution_availability_parent_digest_seal,
    _distribution_availability_figure_step_matches_parent,
)
from .figures.continuous_measurement_audit import (
    _continuous_measurement_audit_parent_digest_seal,
)
from .replication.envelope import (
    ENVELOPE_SCHEMA_VERSION,
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
from .authority.plan_scope import (
    _serializable_plan_scientific_scope_signature,
    completed_step_record_matches_plan,
    verified_plan_evidence_rank,
)
from .authority import pipeline_cache as _pipeline_cache
from .planning.analysis_blueprint import (
    build_analysis_blueprint,
    render_analysis_blueprint_for_prompt,
    validate_plan_against_analysis_blueprint,
)
from .reporting.article_contract import (
    build_article_analysis_contract,
    validate_plan_against_article_contract,
)
from .planning.figure_strategy import build_article_figure_strategy
from .orchestration.config import PipelineConfig
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
from .execution.output_files import _clear_output_dir, _has_figure_exports
from .concept_dict_audit import (
    assert_dict_matches as assert_concept_dict_matches,
    verify_recorded_dict_match,
    write_concept_dict_fingerprint,
)
from .cohort.schema import (
    COHORT_LOCK_FILENAME,
    _load_locked_cohort_definition,
    ensure_cohort_definition,
    materialize_locked_analysis_cohort,
    write_locked_cohort_definition,
)
from .planning.cohort_contract import cohort_definition_sha
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
    trajectory_plan_dag_findings,
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

_audit_manuscript_numeric_claims = (
    audit_manuscript_numeric_claims  # noqa: F841 (legacy alias)
)

from .authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
    _coerce_enforcement_mode,
    sha256_of_file,
)
from .authority.evidence_snapshot import load_current_evidence_snapshot
from .learning.experience import (
    ExperienceBank,
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
    _cohort_definition_is_empty,
    _ensure_audit_panel_step_in_plan,
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
from .orchestration.experiment_spec import ExperimentSpec, dump_experiment_spec
from .figures.skill import PublicationFigureSkill
from .reporting.bibtex import render_bibtex
from .reporting.latex import scaffold_to_latex
from .literature import (
    HypothesisBlueprintAgent,
    LiteratureAgent,
    LiteratureBundle,
    render_hypothesis_blueprint_for_prompt,
)
from .planning.preplan_literature import prepare_preplan_literature
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
from .providers.protocol import LLMClient, LLMMessage
from .learning.memory import RunMemory
from .providers.prompts import PROMPT_PACK_VERSION, prompt_pack_files
from .execution.runner import (
    HOST_OWNED_RUNNER_ENV_KEYS,
    CodeRunner,
    DockerRunner,
    reject_reserved_runner_env,
    select_safe_runner_kind,
)
from .execution.method_capabilities import set_runtime_capability_snapshot_provider
from .concept_availability import normalize_database_name
from .schema import (
    AgentRuntimeState,
    AnalysisManifest,
    AnalysisPlan,
    AnalysisStep,
    CritiqueReport,
    EvidenceRecord,
    EvidenceRef,
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
    prepare_existing_resume_input,
    seal_run_input_capsule,
    verify_legacy_trajectory_capsule_receipt,
)
from .authority.run_lock import current_locked_run_id, exclusive_run_execution
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
from .gates.visual_qa import VLMVisualQAAdapter, VisualQAAuditor


from .orchestration.finalize import (
    _concept_dictionary_manifest_fields,  # noqa: F401
    _render_cost_summary,  # noqa: F401
)


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


def _resume_plan_candidate_paths(
    *,
    run_dir: Path,
    resume_state: Optional[Dict[str, Any]],
) -> List[Path]:
    """Return digest-verified immutable plan evidence, newest first.

    Live ``analysis_plan*.json`` files are mutable runtime conveniences and can
    be re-serialized under a newer schema during resume. They are never plan
    authority. The evidence copies retain the planner/replanner bytes and are
    usable only after path containment and SHA-256 verification.
    """

    del resume_state  # Evidence authority supersedes a mutable manifest path.
    records = list(load_current_evidence_snapshot(run_dir).records)

    ranked: List[tuple[int, int, Path]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        revision = verified_plan_evidence_rank(record)
        if revision is None:
            continue
        verified_path = verified_run_evidence_path(run_dir, record)
        if verified_path is not None:
            ranked.append((revision, index, verified_path))

    candidates = [path for _revision, _index, path in sorted(ranked, reverse=True)]

    unique: List[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def _load_compatible_resume_plan(
    *,
    run_dir: Path,
    resume_state: Optional[Dict[str, Any]],
) -> tuple[Optional[AnalysisPlan], Optional[Path]]:
    """Load the newest saved plan compatible with completed resume steps."""
    locked_cohort_sha256: Optional[str] = None
    if (run_dir / COHORT_LOCK_FILENAME).exists():
        locked_cohort_sha256 = cohort_definition_sha(
            _load_locked_cohort_definition(run_dir)
        )
    completed_records = [
        record
        for record in current_successful_step_records(
            (resume_state or {}).get("per_step_records") or []
        )
        if record.get("step_id") and record.get("step_id") != "00_probe"
    ]
    completed_step_ids = {str(record.get("step_id")) for record in completed_records}
    candidates = _resume_plan_candidate_paths(
        run_dir=run_dir,
        resume_state=resume_state,
    )
    for candidate in candidates:
        try:
            plan = AnalysisPlan.model_validate(
                json.loads(candidate.read_text(encoding="utf-8"))
            )
        except Exception:
            continue
        if locked_cohort_sha256 is not None and (
            plan.cohort is None
            or cohort_definition_sha(plan.cohort) != locked_cohort_sha256
        ):
            # Plan revisions are allowed to change unfinished steps, never the
            # already sealed cohort authority.  Skip an incomplete/drifted
            # revision and try the next digest-verified ancestor.
            continue
        step_by_id = {step.step_id: step for step in plan.steps}
        if not plan.steps or not completed_step_ids <= set(step_by_id):
            continue
        compatible = True
        expected_plan_scope = _serializable_plan_scientific_scope_signature(plan)
        for record in completed_records:
            step_id = str(record.get("step_id") or "")
            if not completed_step_record_matches_plan(
                record,
                step=step_by_id[step_id],
                expected_plan_scope=expected_plan_scope,
                plan_candidate_count=len(candidates),
                completed_records=completed_records,
            ):
                compatible = False
                break
        if compatible:
            return plan, candidate
    return None, None


class LegacyResumePlanMigrationError(RuntimeError):
    """A legacy resume plan could not be migrated without scientific drift."""


def _normalise_plan_contract_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _is_closed_adjusted_association_step(step: AnalysisStep) -> bool:
    """Match the typed roster's exact method-and-product contract only."""

    method_head = str(step.method or "").lower().split(" with ", 1)[0]
    if (
        _normalise_plan_contract_token(method_head)
        != PLANNED_MODEL_REQUIREMENTS_STEP_METHOD
    ):
        return False
    products = set()
    for output in step.expected_outputs or []:
        kind, separator, name = str(output or "").partition(":")
        if separator:
            products.add(
                (
                    _normalise_plan_contract_token(kind),
                    _normalise_plan_contract_token(name),
                )
            )
    return (
        PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
        PLANNED_MODEL_REQUIREMENTS_OUTPUT,
    ) in products


def _resume_completed_records_for_plan_migration(
    *,
    plan: AnalysisPlan,
    resume_state: Optional[Dict[str, Any]],
    resume_from_step_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Return current successful records that remain completed after a cut."""

    current_records = [
        dict(record)
        for record in current_step_records(
            [
                record
                for record in ((resume_state or {}).get("per_step_records") or [])
                if isinstance(record, dict) and record.get("step_id")
            ]
        )
    ]
    cut_step_id = str(resume_from_step_id or "").strip()
    if not cut_step_id:
        return [record for record in current_records if record.get("status") == "ok"]

    step_order = {step.step_id: index for index, step in enumerate(plan.steps)}
    if cut_step_id == "00_probe":
        cut_index = -1
    elif cut_step_id in step_order:
        cut_index = step_order[cut_step_id]
    else:
        raise LegacyResumePlanMigrationError(
            f"resume_from_step_id={cut_step_id!r} is not in the active analysis plan"
        )

    completed: List[Dict[str, Any]] = []
    for record in current_records:
        if record.get("status") != "ok":
            continue
        step_id = str(record.get("step_id") or "")
        record_index = -1 if step_id == "00_probe" else step_order.get(step_id)
        if record_index is not None and record_index < cut_index:
            completed.append(record)
    return completed


def _legacy_resume_model_roster_targets(
    *,
    plan: AnalysisPlan,
    completed_step_ids: set[str],
) -> tuple[str, ...]:
    """Select only remaining, exact closed-contract steps with an empty roster."""

    return tuple(
        step.step_id
        for step in plan.steps
        if step.step_id not in completed_step_ids
        and _is_closed_adjusted_association_step(step)
        and not step.model_requirements
    )


class _LegacyModelRosterStepPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    model_requirements: List[PlannedModelRequirement] = Field(min_length=1)


class _LegacyModelRosterPacket(BaseModel):
    """Planner-owned roster patch with no surface for broader plan edits."""

    model_config = ConfigDict(extra="forbid")

    steps: List[_LegacyModelRosterStepPacket] = Field(min_length=1)


def _parse_legacy_model_roster_packet(
    raw: str,
    *,
    target_step_ids: tuple[str, ...],
) -> _LegacyModelRosterPacket:
    packet = _LegacyModelRosterPacket.model_validate(json.loads(raw.strip()))
    returned_step_ids = [step.step_id for step in packet.steps]
    if returned_step_ids != list(target_step_ids):
        raise ValueError(
            "roster packet steps must exactly match the ordered target ids: "
            f"expected={list(target_step_ids)!r}, returned={returned_step_ids!r}"
        )
    for step in packet.steps:
        requirement_ids = [
            requirement.requirement_id for requirement in step.model_requirements
        ]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError(
                f"duplicate requirement_id in roster packet for {step.step_id!r}"
            )
        primary_count = sum(
            requirement.analysis_role == "primary"
            for requirement in step.model_requirements
        )
        if primary_count != 1:
            raise ValueError(
                "each target step roster must contain exactly one "
                "analysis_role='primary'; the Planner chooses which requirement "
                f"is primary (step={step.step_id!r}, returned={primary_count})"
            )
    return packet


def _project_legacy_model_roster_packet(
    *,
    plan: AnalysisPlan,
    packet: _LegacyModelRosterPacket,
) -> AnalysisPlan:
    """Project only validated roster values onto an otherwise frozen plan."""

    rosters = {
        step.step_id: [
            requirement.model_dump(mode="json")
            for requirement in step.model_requirements
        ]
        for step in packet.steps
    }
    payload = plan.model_dump(mode="json")
    for step_payload in payload["steps"]:
        step_id = str(step_payload.get("step_id") or "")
        if step_id in rosters:
            step_payload["model_requirements"] = rosters[step_id]
    return AnalysisPlan.model_validate(payload)


def _next_analysis_plan_revision(
    *,
    run_dir: Path,
    plan: AnalysisPlan,
    evidence: EvidenceStore,
) -> int:
    revision = int(plan.revision) + 1
    for path in run_dir.glob("analysis_plan_revision_*.json"):
        match = re.fullmatch(r"analysis_plan_revision_(\d+)\.json", path.name)
        if match:
            revision = max(revision, int(match.group(1)) + 1)
    while evidence.get(f"analysis_plan_revision_{revision}") is not None:
        revision += 1
    return revision


def _restore_resume_plan_robustness_lock(
    *,
    plan: AnalysisPlan,
    run_dir: Path,
    evidence: EvidenceStore,
    prompt_version: str,
    llm_signature: str,
) -> tuple[AnalysisPlan, Optional[Path]]:
    """Project the verified plan-time robustness lock onto a resume plan.

    A probe-time replanner from older runs could reword or drop robustness
    specifications after the immutable lock was written. Execution correctly
    rejects that drift, but resume must load a plan that agrees with the lock.
    The lock remains the authority: this migration writes a new immutable plan
    revision and never rewrites the locked specifications.
    """

    lock_path = Path(run_dir) / "robustness_specs_locked.json"
    if not lock_path.is_file():
        return plan, None
    locked_specs = load_locked_robustness_specs(run_dir)
    active_specs = list(plan.robustness_specs or [])
    if robustness_specs_sha(active_specs) == robustness_specs_sha(locked_specs):
        return plan, None

    revision = _next_analysis_plan_revision(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
    )
    restored = plan.model_copy(
        update={
            "robustness_specs": list(locked_specs),
            "revision": revision,
        }
    )
    revision_path = run_dir / f"analysis_plan_revision_{revision}.json"
    if revision_path.exists():
        raise LegacyResumePlanMigrationError(
            f"refusing to overwrite existing plan revision {revision_path.name}"
        )
    revision_path.write_text(restored.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Resume migration restoring the immutable plan-time robustness "
            "specification lock."
        ),
        source_path=revision_path,
        evidence_id=f"analysis_plan_revision_{revision}",
        producer="planner",
        generation_mode="system",
        prompt_pack_version=prompt_version,
        metadata={
            "reason": "restore_locked_robustness_specs",
            "llm_signature": llm_signature,
        },
    )
    return restored, revision_path


def _migrate_legacy_resume_figure_render_edges(
    *,
    plan: AnalysisPlan,
    run_dir: Path,
    resume_state: Optional[Dict[str, Any]],
    resume_from_step_id: Optional[str],
    evidence: EvidenceStore,
    prompt_version: str,
    llm_signature: str,
) -> tuple[AnalysisPlan, Optional[Path], tuple[str, ...]]:
    """Restore exact typed parent edges on legacy system-split figure steps.

    Older framework-generated render children copied the parent's raw inputs and
    scientific method. Current source-data authority requires a host-resolved
    typed edge. This migration is intentionally narrower than ordinary plan
    shaping: it recognizes only the full legacy splitter fingerprint or an
    already-visualization child with one globally unique exact typed table role.
    Raw artifacts, sibling tables, datasets, and models remain excluded so
    rendering cannot reopen the scientific analysis. Any ambiguity remains
    fail-closed for the Planner.
    """

    from .contracts.declared_product import (
        effect_bearing_product,
        typed_product,
    )

    completed_records = _resume_completed_records_for_plan_migration(
        plan=plan,
        resume_state=resume_state,
        resume_from_step_id=resume_from_step_id,
    )
    completed_step_ids = {
        str(record.get("step_id") or "") for record in completed_records
    }
    if len(plan.steps) < 2:
        return plan, None, ()

    step_order = {str(step.step_id): index for index, step in enumerate(plan.steps)}
    cut_step_id = str(resume_from_step_id or "").strip()
    cut_index: Optional[int] = None
    if cut_step_id:
        if cut_step_id == "00_probe":
            cut_index = 0
        elif cut_step_id in step_order:
            cut_index = step_order[cut_step_id]
        else:
            raise LegacyResumePlanMigrationError(
                f"resume_from_step_id={cut_step_id!r} is not in the active analysis plan"
            )

    producer_ids: Dict[tuple[str, str], set[str]] = {}
    producer_tokens: Dict[tuple[str, str], List[tuple[str, str]]] = {}
    for producer_step in plan.steps:
        for output in producer_step.expected_outputs or []:
            parsed = typed_product(output)
            if parsed is not None and parsed[0] in {"statistic", "table"}:
                producer_ids.setdefault(parsed, set()).add(str(producer_step.step_id))
                producer_tokens.setdefault(parsed, []).append(
                    (str(producer_step.step_id), str(output))
                )

    def _exact_role_dependencies(
        figure_outputs: Sequence[str],
        *,
        required_producer_id: Optional[str] = None,
        required_producer_step: Optional[AnalysisStep] = None,
    ) -> tuple[List[str], set[str]]:
        dependencies: List[str] = []
        dependency_producers: set[str] = set()
        for figure_output in figure_outputs:
            parsed_figure = typed_product(figure_output)
            if parsed_figure is None or parsed_figure[0] != "figure":
                return [], set()
            candidates = [
                candidate
                for kind in ("table", "statistic")
                for candidate in producer_tokens.get((kind, parsed_figure[1]), [])
            ]
            if required_producer_id is not None:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[0] == required_producer_id
                ]
            if not candidates and effect_bearing_product(figure_output):
                semantic_candidates: List[tuple[str, str]] = []
                for identity, raw_candidates in producer_tokens.items():
                    if required_producer_id is not None:
                        raw_candidates = [
                            candidate
                            for candidate in raw_candidates
                            if candidate[0] == required_producer_id
                        ]
                    if not raw_candidates:
                        continue
                    supported = _effect_figure_semantics_supported_by_inputs(
                        figure_outputs=[figure_output],
                        effect_input_products={identity},
                    ) or (
                        required_producer_step is not None
                        and _effect_figure_semantics_supported_by_model_roster(
                            step=required_producer_step,
                            figure_outputs=[figure_output],
                            effect_input_products={identity},
                        )
                    )
                    if supported:
                        semantic_candidates.extend(raw_candidates)
                candidates = semantic_candidates
            if len(candidates) != 1:
                return [], set()
            producer_id, source_token = candidates[0]
            dependencies.append(source_token)
            dependency_producers.add(producer_id)
        return list(dict.fromkeys(dependencies)), dependency_producers

    revised_steps = list(plan.steps)
    migrated_step_ids: List[str] = []

    for index in range(1, len(plan.steps)):
        parent = plan.steps[index - 1]
        child = plan.steps[index]
        parent_id = str(parent.step_id)
        child_id = str(child.step_id)
        child_is_in_resume_window = cut_index is None or index >= cut_index
        parent_is_available_or_scheduled = parent_id in completed_step_ids or (
            cut_index is not None and index - 1 >= cut_index
        )
        if (
            not child_is_in_resume_window
            or not parent_is_available_or_scheduled
            or child_id in completed_step_ids
            or child_id != f"{parent_id}_figure"
            or str(child.method) != str(parent.method)
            or list(child.inputs or []) != list(parent.inputs or [])
            or list(child.icu_rule_refs or [])
            != [*list(parent.icu_rule_refs or []), "visualization_rule"]
            or child.model_requirements
            or child.trajectory_stability_spec is not None
        ):
            continue

        parent_outputs = list(parent.expected_outputs or [])
        if any(
            (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
            for raw in parent_outputs
        ):
            continue

        figure_outputs: List[str] = []
        child_contract_valid = True
        for raw in child.expected_outputs or []:
            parsed = typed_product(raw)
            if parsed is None:
                child_contract_valid = False
                break
            if parsed[0] == "figure":
                figure_outputs.append(str(raw))
            elif parsed[0] != "log":
                child_contract_valid = False
                break
        figure_identities = [typed_product(raw) for raw in figure_outputs]
        if (
            not child_contract_valid
            or not figure_outputs
            or len(figure_identities) != len(set(figure_identities))
            or child.intent
            != _render_only_figure_step_intent(
                source_step_id=parent_id,
                figure_outputs=figure_outputs,
            )
        ):
            continue

        source_tokens, dependency_producers = _exact_role_dependencies(
            figure_outputs,
            required_producer_id=parent_id,
            required_producer_step=parent,
        )
        source_identities = {
            parsed
            for raw in source_tokens
            if (parsed := typed_product(raw)) is not None
        }
        source_names = [identity[1] for identity in source_identities]
        if (
            not source_tokens
            or dependency_producers != {parent_id}
            or len(source_identities) != len(source_tokens)
            or len(source_names) != len(set(source_names))
            or any(
                producer_ids.get(identity) != {parent_id}
                for identity in source_identities
            )
        ):
            continue

        if any(effect_bearing_product(raw) for raw in figure_outputs) and (
            not effect_output_authorized(parent)
            or not (
                _effect_figure_semantics_supported_by_inputs(
                    figure_outputs=figure_outputs,
                    effect_input_products=source_identities,
                )
                or _effect_figure_semantics_supported_by_model_roster(
                    step=parent,
                    figure_outputs=figure_outputs,
                    effect_input_products=source_identities,
                )
            )
        ):
            continue

        revised_steps[index] = _migrate_render_step_contract(
            child, source_tokens, method="visualization"
        )
        migrated_step_ids.append(child_id)

    # A later framework splitter already emitted a visualization child, but an
    # older plan may still bind it to a sibling table even though the declared
    # figure has one globally unique exact typed table role elsewhere. Preserve
    # step ids/order and replace only that closed edge; never infer multi-table
    # dependencies or grant every table owned by the producer.
    for index, original_child in enumerate(plan.steps):
        child = revised_steps[index]
        child_id = str(child.step_id)
        if (
            child_id in completed_step_ids
            or (cut_index is not None and index < cut_index)
            or not child_id.endswith("_figure")
            or _normalise_plan_contract_token(str(child.method or "")).split(
                "_with_", 1
            )[0]
            != "visualization"
            or not child.inputs
            or any(
                (parsed := typed_product(raw)) is None
                or parsed[0] not in {"statistic", "table"}
                for raw in child.inputs
            )
        ):
            continue
        figure_outputs = [
            str(raw)
            for raw in child.expected_outputs or []
            if (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
        ]
        parsed_child_outputs = [
            typed_product(raw) for raw in child.expected_outputs or []
        ]
        if (
            not figure_outputs
            or any(parsed is None for parsed in parsed_child_outputs)
            or any(
                parsed[0] not in {"figure", "log"}
                for parsed in parsed_child_outputs
                if parsed is not None
            )
        ):
            continue
        source_tokens, dependency_producers = _exact_role_dependencies(figure_outputs)
        if len(dependency_producers) != 1 or not source_tokens:
            continue
        source_step_id = next(iter(dependency_producers))
        source_index = step_order.get(source_step_id)
        if (
            source_index is None
            or source_index >= index
            or (
                source_step_id not in completed_step_ids
                and not (cut_index is not None and source_index >= cut_index)
            )
        ):
            continue
        intended = _render_only_figure_step_intent(
            source_step_id=source_step_id,
            figure_outputs=figure_outputs,
        )
        migrated_child = _migrate_render_step_contract(
            child, source_tokens, intent=intended
        )
        if child == migrated_child:
            continue
        revised_steps[index] = migrated_child
        if child_id not in migrated_step_ids:
            migrated_step_ids.append(child_id)

    if not migrated_step_ids:
        return plan, None, ()

    revision = _next_analysis_plan_revision(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
    )
    migrated = plan.model_copy(
        update={
            "steps": revised_steps,
            "revision": revision,
        }
    )
    revision_path = run_dir / f"analysis_plan_revision_{revision}.json"
    if revision_path.exists():
        raise LegacyResumePlanMigrationError(
            f"refusing to overwrite existing plan revision {revision_path.name}"
        )
    revision_path.write_text(migrated.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Resume migration restoring exact typed parent edges on legacy "
            "framework-split rendering steps."
        ),
        source_path=revision_path,
        evidence_id=f"analysis_plan_revision_{revision}",
        producer="planner",
        generation_mode="system",
        prompt_pack_version=prompt_version,
        metadata={
            "reason": "resume_legacy_figure_render_edges",
            "target_step_ids": migrated_step_ids,
            "llm_signature": llm_signature,
        },
    )
    return migrated, revision_path, tuple(migrated_step_ids)


def _migrate_resume_trajectory_products(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    run_dir: Path,
    evidence: EvidenceStore,
    prompt_version: str,
    llm_signature: str,
) -> tuple[AnalysisPlan, Optional[Path], List[ValidationFinding]]:
    """Apply schema-only trajectory products to a reused legacy plan.

    Resume normally skips plan-shaping transforms to preserve step identities.
    Canonical trajectory products are the safe exception: augmentation changes
    neither step ids/order nor any scientific method, input, horizon, threshold,
    or cluster choice. Older checkpoints may predate the role recognizer, so
    treating their saved plan as already normalized silently removes the replay
    contracts from resumed execution.
    """

    augmented, augmentation_findings = augment_trajectory_plan_products(
        plan=plan,
        context=context,
    )
    if augmented == plan:
        return plan, None, augmentation_findings

    revision = _next_analysis_plan_revision(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
    )
    augmented = augmented.model_copy(update={"revision": revision})
    revision_path = run_dir / f"analysis_plan_revision_{revision}.json"
    if revision_path.exists():
        raise LegacyResumePlanMigrationError(
            f"refusing to overwrite existing plan revision {revision_path.name}"
        )
    revision_path.write_text(augmented.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Resume migration adding canonical trajectory replay products to "
            "the existing agent-owned role DAG."
        ),
        source_path=revision_path,
        evidence_id=f"analysis_plan_revision_{revision}",
        producer="planner",
        generation_mode="system",
        prompt_pack_version=prompt_version,
        metadata={
            "reason": "resume_trajectory_schema_products",
            "llm_signature": llm_signature,
        },
    )
    return augmented, revision_path, augmentation_findings


def _migrate_legacy_resume_model_requirements(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    run_dir: Path,
    resume_state: Optional[Dict[str, Any]],
    resume_from_step_id: Optional[str],
    role_resolver: Callable[[str], Any],
    evidence: EvidenceStore,
    prompt_version: str,
    llm_signature: str,
) -> tuple[AnalysisPlan, Optional[Path], tuple[str, ...]]:
    """Ask the planner LLM to migrate an old empty typed-model roster.

    The framework identifies only the schema surface that requires migration.
    It never derives an outcome, exposure, analysis role, analysis set, or model
    family from prose. Those scientific commitments must come back in a small,
    strictly typed PlannerAgent packet. The framework projects only that roster
    onto the frozen plan before a revision is written or registered.
    """

    completed_records = _resume_completed_records_for_plan_migration(
        plan=plan,
        resume_state=resume_state,
        resume_from_step_id=resume_from_step_id,
    )
    completed_step_ids = {str(record.get("step_id")) for record in completed_records}
    target_step_ids = _legacy_resume_model_roster_targets(
        plan=plan,
        completed_step_ids=completed_step_ids,
    )
    if not target_step_ids:
        return plan, None, ()

    target_steps = [
        {
            "step_id": step.step_id,
            "intent": step.intent,
            "method": step.method,
            "inputs": list(step.inputs),
            "expected_outputs": list(step.expected_outputs),
            "icu_rule_refs": list(step.icu_rule_refs),
        }
        for step in plan.steps
        if step.step_id in set(target_step_ids)
    ]
    required_fields = [
        "requirement_id",
        "outcome",
        "outcome_type",
        "method_family",
        "exposure_source",
        "analysis_role",
        "analysis_set",
        "required_for_step_success",
    ]
    format_reminder = (
        'Return exactly {"steps": [{"step_id": <target id>, '
        '"model_requirements": [<one or more complete requirement objects>]}]}. '
        f"Every requirement object must contain all fields {required_fields!r}. "
        "Allowed outcome_type: binary, continuous. Allowed analysis_role: "
        "primary, secondary, sensitivity. Allowed analysis_set: source_aware, "
        "complete_case. Primary/secondary requirements must set "
        "required_for_step_success=true; only sensitivity may be false. Each "
        "target step must contain exactly one analysis_role=primary; the Planner "
        "chooses which requirement is primary and labels all others secondary "
        "or sensitivity."
    )
    plan_payload = plan.model_dump(mode="json")
    plan_level_commitments = {
        key: plan_payload.get(key)
        for key in (
            "research_question",
            "analysis_type",
            "rationale",
            "cohort",
            "robustness_specs",
        )
    }
    messages = [
        LLMMessage(
            role="system",
            content=(
                "You are the PlannerAgent's typed legacy-plan migration worker. "
                "Choose the scientific model roster; the framework will only "
                "validate and project it. Return JSON only. Never rewrite the "
                "AnalysisPlan and never invent a default model when the supplied "
                "plan and ResearchContext do not justify one."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                "Populate model_requirements for every target step. Each roster "
                "entry is a complete PlannedModelRequirement. Choose outcome, "
                "exposure_source, method_family, role, and analysis set from the "
                "unchanged scientific commitments below. Emit a separate "
                "requirement for every adjusted outcome/model pre-specified in "
                "the target step intent; ResearchContext.target_outcome is not "
                "an exhaustive roster and must not cause an intent-committed "
                "secondary outcome/model to be omitted. The Planner decides the "
                "roster count and contents; the framework does not infer either "
                "from prose. For each target step, choose exactly one roster "
                "entry as analysis_role=primary and label the other entries "
                "secondary or sensitivity. Binary method_family "
                f"must be one of {sorted(ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES)!r}; "
                "continuous method_family must be one of "
                f"{sorted(ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES)!r}. "
                "Do not return plan fields, prose, or requirements for any other "
                "step.\n\n"
                f"REQUIRED JSON SHAPE:\n{format_reminder}\n\n"
                "TARGET STEPS (verbatim from the saved planner plan):\n"
                f"{json.dumps(target_steps, indent=2, ensure_ascii=False)}\n\n"
                "READ-ONLY PLAN-LEVEL COMMITMENTS (context only; do not return "
                "these fields):\n"
                f"{json.dumps(plan_level_commitments, indent=2, ensure_ascii=False)}\n\n"
                "RESEARCH CONTEXT:\n"
                f"{context.model_dump_json(indent=2)}"
            ),
        ),
    ]
    try:
        from .providers.structured_retry import call_llm_with_structured_retry

        packet = call_llm_with_structured_retry(
            role_resolver("planner"),
            messages,
            parser=lambda raw: _parse_legacy_model_roster_packet(
                raw,
                target_step_ids=target_step_ids,
            ),
            role="legacy_model_roster_migration",
            max_retries=2,
            max_tokens=4096,
            temperature=0.1,
            format_reminder=format_reminder,
        )
    except Exception as exc:
        raise LegacyResumePlanMigrationError(
            "planner LLM failed while migrating legacy model_requirements; "
            "resume stopped without a default model"
        ) from exc
    revised = _project_legacy_model_roster_packet(
        plan=plan,
        packet=packet,
    )
    revision = _next_analysis_plan_revision(
        run_dir=run_dir,
        plan=plan,
        evidence=evidence,
    )
    revised = revised.model_copy(update={"revision": revision})
    revision_path = run_dir / f"analysis_plan_revision_{revision}.json"
    if revision_path.exists():
        raise LegacyResumePlanMigrationError(
            f"refusing to overwrite existing plan revision {revision_path.name}"
        )
    revision_path.write_text(revised.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description=(
            "Planner-owned legacy resume migration for typed model requirements."
        ),
        source_path=revision_path,
        evidence_id=f"analysis_plan_revision_{revision}",
        producer="planner",
        generation_mode="llm",
        prompt_pack_version=prompt_version,
        metadata={
            "reason": "legacy_missing_model_requirements",
            "target_step_ids": list(target_step_ids),
            "llm_signature": llm_signature,
        },
    )
    return revised, revision_path, target_step_ids


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


class ResearchAgentPipeline:
    """One-shot orchestration. Construct, call :meth:`run`, read the result."""

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "ResearchAgentPipeline":
        """Construct a pipeline from a :class:`PipelineConfig` object.

        The legacy keyword-argument form of ``__init__`` continues to
        work; this classmethod is the recommended new-code entry point
        because the config object is typed, copyable
        (``config.with_overrides(...)``) and serialisable
        (``config.as_kwargs()``).
        """
        return cls(**config.as_kwargs())

    def __init__(
        self,
        *,
        workdir: Union[str, Path],
        llm: Optional[LLMClient] = None,
        timeout_seconds: float = 300.0,
        standard_executor_timeout_seconds: float = 3_600.0,
        python_executable: Optional[str] = None,
        enable_literature: bool = True,
        enable_visual_qa: bool = True,
        enable_publication_figure_skill: bool = True,
        enable_vlm_visual_qa: Optional[bool] = None,
        vlm_client: Optional[LLMClient] = None,
        visual_qa_adapter: Optional[VLMVisualQAAdapter] = None,
        enable_llm_concept_audit: Optional[bool] = None,
        llm_concept_auditor_client: Optional[LLMClient] = None,
        enable_memory: bool = True,
        enable_latex: bool = True,
        latex_venue_template: str = "article",
        manuscript_language: str = "en",
        evidence_enforcement_mode: str = "soft",
        disable_icu_context: bool = False,
        context_top_k: Optional[int] = None,
        max_code_repair_attempts: int = 3,
        max_step_llm_repair_attempts: int = 2,
        max_step_provider_calls: int = 7,
        enable_deterministic_code_fallback: bool = False,
        enable_deterministic_planner_fallback: bool = False,
        enable_deterministic_runner_repair: bool = True,
        enable_pubmed: bool = False,
        pubmed_email: Optional[str] = None,
        pubmed_api_key: Optional[str] = None,
        enable_tavily: bool = False,
        tavily_api_key: Optional[str] = None,
        tavily_retmax: int = 5,
        tavily_include_domains: Optional[Sequence[str]] = None,
        tavily_exclude_domains: Optional[Sequence[str]] = None,
        enable_cache: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_cost_tracking: bool = False,
        cost_price_table: Optional[Dict[str, Any]] = None,
        enable_reproducibility_envelope: bool = False,
        llm_seed: Optional[int] = None,
        envelope_include_previews: bool = False,
        submission_profile_name: Optional[str] = None,
        submission_profile_version: Optional[str] = None,
        submission_profile_locked_at: Optional[str] = None,
        expected_concept_dict_sha: Optional[str] = None,
        expected_sofa2_dict_sha: Optional[str] = None,
        enable_multiple_testing_correction: bool = True,
        multiple_testing_alpha: float = 0.05,
        enable_causal_audit: bool = True,
        enable_reporting_checklist: bool = True,
        reporting_checklist_names: Optional[Sequence[str]] = None,
        task_kind: Optional[str] = None,
        enable_reviewer_round: bool = True,
        enable_fairness_subgroups: bool = True,
        enable_hypothesis_generator: bool = False,
        hypothesis_generator_top_k: int = 5,
        enable_pdf_render: bool = False,
        max_concurrent_steps: int = 1,
        development_sample_size: Optional[int] = None,
        development_sample_seed: int = 20260719,
        enable_probe_step: bool = True,
        enable_replanning: bool = True,
        max_total_steps: int = 12,
        max_consecutive_noop_replans: int = 2,
        max_replans: int = 6,
        stabilization_mode: bool = False,
        max_numeric_claims_per_step: int = 100,
        writer_digest_widened: bool = False,
        writer_digest_secondary_cap_per_step: int = 20,
        enable_experience_bank: bool = False,
        experience_bank_path: Optional[Union[str, Path]] = None,
        experience_bank_top_k: int = 5,
        experience_bank_min_similarity: float = 0.2,
        enable_know_how: bool = False,
        know_how_paths: Sequence[Union[str, Path]] = (),
        know_how_top_k: int = 3,
        know_how_min_score: float = 0.15,
        runner_kind: str = "auto",
        runner_image: Optional[str] = None,
        runner_network: str = "none",
        runner_factory: Optional[Callable[..., Any]] = None,
        runner_kwargs: Optional[Dict[str, Any]] = None,
        case_plugin_registry: Optional[Any] = None,
    ) -> None:
        # Snapshot the construction kwargs into a typed config object for
        # introspection / replay / serialisation. The body below still reads
        # from local variables directly (so the legacy ``__init__(**kwargs)``
        # signature stays unchanged); ``self._config`` is purely a view onto
        # what we were given.
        self._config: PipelineConfig = PipelineConfig.from_kwargs(
            **{k: v for k, v in locals().items() if k != "self"}
        )
        # Lazy import to avoid pulling the plugin registry module if no
        # plugins are configured (the default).
        from .fallback import CasePluginRegistry as _CasePluginRegistry

        self._case_plugin_registry = case_plugin_registry or _CasePluginRegistry()
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._llm = llm
        self._timeout_seconds = timeout_seconds
        self._standard_executor_timeout_seconds = standard_executor_timeout_seconds
        self._python_executable = python_executable
        self._enable_literature = enable_literature
        self._enable_visual_qa = enable_visual_qa
        self._enable_publication_figure_skill = bool(enable_publication_figure_skill)
        self._vlm_client = vlm_client
        self._visual_qa_adapter = visual_qa_adapter
        self._llm_concept_auditor_client = llm_concept_auditor_client
        if enable_vlm_visual_qa is None:
            self._enable_vlm_visual_qa = bool(
                visual_qa_adapter is not None
                or llm_supports_vision(vlm_client)
                or llm_supports_vision(llm)
            )
        else:
            self._enable_vlm_visual_qa = bool(enable_vlm_visual_qa)
        if enable_llm_concept_audit is None:
            concept_client = llm_concept_auditor_client or llm
            self._enable_llm_concept_audit = bool(
                concept_client is not None and not llm_is_mockish(concept_client)
            )
        else:
            self._enable_llm_concept_audit = bool(enable_llm_concept_audit)
        self._enable_memory = enable_memory
        self._enable_latex = enable_latex
        self._latex_venue_template = latex_venue_template or "article"
        lang = (manuscript_language or "en").lower()
        self._manuscript_language = (
            "zh" if lang.startswith(("zh", "cn", "chinese")) else "en"
        )
        # Stored as the canonical Enum so downstream code can compare
        # against EvidenceEnforcementMode.STRICT without string casing.
        self._evidence_enforcement_mode = _coerce_enforcement_mode(
            evidence_enforcement_mode
        )
        # T1.4 — when set, the pipeline strips the ICU rules out of the
        # context that drives planning, coding and validation. This is the
        # historical *untyped* naive ablation: a generic data agent sees only
        # column names + dtypes + ANY-aggregation. Typed export authority is
        # deliberately rejected at run entry because exposing its V2 physical
        # facts would contaminate that ablation, while sealing it as V1 would
        # discard the authority contract.
        self._disable_icu_context = bool(disable_icu_context)
        self._context_top_k = int(context_top_k) if context_top_k else None
        self._max_code_repair_attempts = max(0, int(max_code_repair_attempts))
        self._max_step_llm_repair_attempts = max(0, int(max_step_llm_repair_attempts))
        self._max_step_provider_calls = max(0, int(max_step_provider_calls))
        self._enable_deterministic_code_fallback = bool(
            enable_deterministic_code_fallback
        )
        self._enable_deterministic_planner_fallback = bool(
            enable_deterministic_planner_fallback
        )
        self._enable_deterministic_runner_repair = bool(
            enable_deterministic_runner_repair
        )
        # T2.2 — opt-in PubMed live search. Off by default so CI and
        # the offline demo stay deterministic; the LiteratureAgent
        # handles network failure gracefully (empty list → curated
        # registry only).
        self._enable_pubmed = bool(enable_pubmed)
        self._pubmed_email = pubmed_email
        self._pubmed_api_key = pubmed_api_key
        # O5 — opt-in Tavily web search for preprints/guidelines/trial
        # registries that PubMed may not index. Off by default so CI
        # and offline demos remain deterministic.
        self._enable_tavily = bool(enable_tavily)
        self._tavily_api_key = tavily_api_key
        self._tavily_retmax = int(tavily_retmax)
        self._tavily_include_domains = list(tavily_include_domains or [])
        self._tavily_exclude_domains = list(tavily_exclude_domains or [])
        # T3.5 — cohort cache. When enabled, identical re-runs (same
        # cohort hash + same skill/question/llm signature) short-circuit
        # to the prior run_dir's PipelineResult instead of repeating
        # the entire pipeline. Off by default so production users opt
        # in deliberately and tests that *want* full pipeline
        # execution every time keep their existing semantics.
        self._enable_cache = bool(enable_cache)
        self._cache_dir = (
            Path(cache_dir).resolve()
            if cache_dir is not None
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
        self._enable_cost_tracking = bool(enable_cost_tracking)
        self._cost_price_table = cost_price_table
        # O20 — Reproducibility envelope. Records prompt/response
        # sha256, requested seed, temperature, provider/model, and a
        # PHI-safe environment snapshot for every LLM call the pipeline
        # makes. Off by default so the base pipeline behaviour stays
        # bit-identical; turn on when a paper reviewer asks for an
        # auditable replay bundle.
        self._enable_reproducibility_envelope = bool(enable_reproducibility_envelope)
        self._llm_seed = int(llm_seed) if llm_seed is not None else None
        self._envelope_include_previews = bool(envelope_include_previews)
        self._submission_profile_name = submission_profile_name
        self._submission_profile_version = submission_profile_version
        self._submission_profile_locked_at = submission_profile_locked_at
        self._expected_concept_dict_sha = expected_concept_dict_sha
        self._expected_sofa2_dict_sha = expected_sofa2_dict_sha
        # O22 — family-aware multiple-testing correction. Defaults to ON
        # because reviewers of a research-agent paper will always ask
        # for it; the correction is cheap to compute and does not
        # rewrite existing artefacts.
        self._enable_multiple_testing_correction = bool(
            enable_multiple_testing_correction
        )
        self._multiple_testing_alpha = float(multiple_testing_alpha)
        # O18 — Causal audit. Deterministic: labels every primary
        # effect as associational vs causal, scans bound manuscript
        # for causal language over associational effects, emits
        # warnings/errors. Default ON because the cost of a paper
        # that silently sells an OR as causal is high.
        self._enable_causal_audit = bool(enable_causal_audit)
        # O16 — Reporting-guideline checklist. STROBE always; TRIPOD+AI
        # when the analysis looks like a prediction model. Default ON
        # because ICU journals routinely require the checklist.
        self._enable_reporting_checklist = bool(enable_reporting_checklist)
        self._reporting_checklist_names = (
            tuple(reporting_checklist_names) if reporting_checklist_names else None
        )
        # Authoritative benchmark task kind (e.g. "subphenotype_clustering"),
        # used to decide kind-specific reporting-checklist applicability rather
        # than relying on fragile manuscript wording. Optional outside the bench.
        self._benchmark_task_kind = task_kind
        # O15 — Three-role reviewer round (statistician / clinician /
        # methodologist) driven off already-computed evidence and
        # findings. Deterministic; no extra LLM calls. Default ON.
        self._enable_reviewer_round = bool(enable_reviewer_round)
        # O24 — Fairness / subgroup analysis for the primary effect.
        # Deterministic (pure numpy); runs after E-values.
        self._enable_fairness_subgroups = bool(enable_fairness_subgroups)
        # O17 — Front-door hypothesis generator. Runs early (plan phase)
        # to emit a ranked candidate list; does not change the
        # downstream plan unless the user reassigns ``question``.
        self._enable_hypothesis_generator = bool(enable_hypothesis_generator)
        self._hypothesis_generator_top_k = int(hypothesis_generator_top_k)
        # PDF rendering (TexLive optional). Off by default because not
        # every CI environment has a LaTeX install; turn on when the
        # user actually wants ``manuscript_scaffold.pdf`` next to the
        # ``.tex`` and ``.bib``.
        self._enable_pdf_render = bool(enable_pdf_render)
        # T3.3 — concurrent step execution. The canonical AnalysisPlan
        # has independent steps (each step reads the cohort and writes
        # to its own out_dir), so a small thread pool can shrink the
        # critical-path latency by ~k× for k workers. Default 1 keeps
        # bit-identical sequential behaviour for users who don't opt
        # in. EvidenceStore guards its mutators with an RLock so a
        # higher value is safe.
        self._max_concurrent_steps = max(1, int(max_concurrent_steps))
        self._development_sample_size = (
            int(development_sample_size)
            if development_sample_size is not None
            else None
        )
        if (
            self._development_sample_size is not None
            and self._development_sample_size <= 0
        ):
            raise ValueError("development_sample_size must be positive")
        self._development_sample_seed = int(development_sample_seed)
        if (
            self._development_sample_size is not None
            and submission_profile_name is not None
        ):
            raise ValueError(
                "development cohort sampling is non-paper authority and cannot "
                "be combined with a submission profile"
            )
        self._enable_probe_step = bool(enable_probe_step)
        self._enable_replanning = bool(enable_replanning)
        # 0 / None means "no cap". Anything positive enforces the cap in
        # the replanner overflow guard (see execution/phase.py).
        self._max_total_steps = (
            int(max_total_steps) if max_total_steps and max_total_steps > 0 else 0
        )
        # Replan convergence guards (see pipeline_config / execution.phase).
        # 0 disables the guard.
        self._max_consecutive_noop_replans = (
            int(max_consecutive_noop_replans)
            if max_consecutive_noop_replans and max_consecutive_noop_replans > 0
            else 0
        )
        self._max_replans = int(max_replans) if max_replans and max_replans > 0 else 0
        # Stabilization / primary-only iterations tighten the replan budget so a
        # non-converging run fails closed fast (~3 revisions) instead of burning
        # the full-run budget of 6. A caller that already set a smaller positive
        # cap keeps it; a disabled cap (0) is re-armed to 3 under stabilization.
        self._stabilization_mode = bool(stabilization_mode)
        if self._stabilization_mode:
            self._max_replans = min(self._max_replans, 3) if self._max_replans else 3
        self._max_numeric_claims_per_step = (
            int(max_numeric_claims_per_step)
            if max_numeric_claims_per_step and max_numeric_claims_per_step > 0
            else 0
        )
        # Phase-1 writer-digest widening (default off). When True, the
        # writer's evidence digest is augmented with a "secondary
        # numbers" block enumerating NumericClaim entries outside the
        # ``WRITER_DIGEST_PREFERRED_KEYS`` primary subset. The binder
        # already accepts these; the flag controls only what the
        # writer SEES. See reporting.writer_evidence._render_writer_evidence_digest_v2.
        self._writer_digest_widened = bool(writer_digest_widened)
        self._writer_digest_secondary_cap_per_step = max(
            0, int(writer_digest_secondary_cap_per_step)
        )
        # Phase-1 cross-run experience bank (default off). When True,
        # the planner sees retrieved hints in its context block and
        # the post-run reflector mines + writes back records via the
        # deterministic helper in experience.py. The bank is read in
        # ``_retrieve_experience_hints`` and written in
        # ``_reflect_and_persist_experience``.
        self._enable_experience_bank = bool(enable_experience_bank)
        self._experience_bank_path: Optional[Path] = (
            Path(experience_bank_path) if experience_bank_path else None
        )
        self._experience_bank_top_k = max(0, int(experience_bank_top_k))
        self._experience_bank_min_similarity = float(experience_bank_min_similarity)
        self._enable_know_how = bool(enable_know_how)
        self._know_how_paths = tuple(Path(path) for path in know_how_paths)
        self._know_how_top_k = int(know_how_top_k)
        self._know_how_min_score = float(know_how_min_score)
        if not 0 <= self._know_how_top_k <= 5:
            raise ValueError("know_how_top_k must be between 0 and 5")
        if not 0.0 <= self._know_how_min_score <= 1.0:
            raise ValueError("know_how_min_score must be between 0 and 1")
        from .orchestration.profiles import require_profile_know_how_setting

        require_profile_know_how_setting(
            name=submission_profile_name,
            version=submission_profile_version,
            enabled=self._enable_know_how,
        )
        # T3.1 — runner backend selection. ``auto`` prefers a probed Docker
        # image and uses macOS sandbox-exec only when Docker is unavailable;
        # ``docker`` explicitly selects :class:`DockerRunner`
        # which mounts the cohort read-only inside a container with
        # ``--network none`` by default. Users with their own sandbox
        # (e.g. OpenHands) can pass an arbitrary ``runner_factory``
        # that accepts ``workdir=, cohort_parquet=, timeout_seconds=``
        # and returns a runner with a ``run(step_id, code)`` method.
        kind = (runner_kind or "auto").lower()
        if runner_factory is not None:
            self._runner_kind = "custom"
        elif kind in {"auto", "default"}:
            self._runner_kind = "auto"
        elif kind in {"subprocess", "host"}:
            self._runner_kind = "subprocess"
        elif kind in {"docker", "container", "openhands"}:
            self._runner_kind = "docker"
        else:
            raise ValueError(
                f"Unknown runner_kind {runner_kind!r}; "
                "expected 'auto', 'subprocess', 'docker', or pass a runner_factory."
            )
        self._runner_image = runner_image
        self._runner_network = runner_network
        self._runner_factory = runner_factory
        self._runner_kwargs = dict(runner_kwargs or {})
        self._validated_runtime_capabilities: Optional[Tuple[str, ...]] = None
        self._validated_runtime_bundle: Optional[Dict[str, object]] = None
        self._memory = RunMemory(self.workdir) if enable_memory else None

    def _build_runner(
        self,
        *,
        run_dir: Path,
        cohort_path: Path,
        target_outcome: Optional[str] = None,
        universe_path: Optional[Path] = None,
        universe_is_typed: bool = False,
        universe_authority_ref: Optional[MaterializedCohortAuthorityRef] = None,
        trajectory_path: Optional[Path] = None,
        trajectory_authority_ref: Optional[MaterializedTrajectoryAuthorityRef] = None,
        trajectory_legacy_capsule_receipt: Optional[
            VerifiedLegacyTrajectoryCapsuleReceipt
        ] = None,
        timeout_seconds: Optional[float] = None,
    ):
        """Return the configured runner backend for a single ``run()``.

        Kept as a method (not a closure) so subclasses or tests can
        stub it cleanly. Returns any object that exposes
        ``run(step_id=..., code=...) -> RunResult``.

        ``cohort_path`` is the canonical analysis cohort the steps read as
        ``COHORT_PARQUET``. ``universe_path`` (when given) is exposed as
        ``EASYICU_UNIVERSE_PARQUET`` so explicit robustness steps can reach the
        pre-纳排 universe without re-running extraction.
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
        if universe_path is not None:
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

    def _run_plan_phase(
        self,
        *,
        question: str,
        cohort_path: Path,
        cohort_name: str,
        database: str,
        target_outcome: Optional[str],
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

        findings = preplan_data_findings(context=context, cohort_path=cohort_path)
        if any(f.severity == "error" for f in findings):
            emit_progress(
                "audit",
                "Pre-plan data gate failed; aborting before provider execution.",
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
                reason=preplan_data_failure_reason(findings),
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

        memory_digest_text: Optional[str] = None
        if self._memory is not None:
            digest = self._memory.digest_for_prompt(
                research_question=question,
                database=database,
                target_outcome=target_outcome,
            )
            memory_digest_text = digest
            if skill_obj is None:
                try:
                    meta_digest = self._memory.meta_planner_digest(
                        skill_keys=[s.key for s in list_skills()],
                        research_question=question,
                        database=database,
                        target_outcome=target_outcome,
                    )
                    memory_digest_text = digest + "\n\n" + meta_digest
                except Exception:
                    pass
            digest_path = run_dir / "memory_digest.md"
            digest_path.write_text(memory_digest_text, encoding="utf-8")
            if evidence.get("memory_digest") is None:
                evidence.register_file(
                    kind="log",
                    description="Cross-run memory digest fed to the planner.",
                    source_path=digest_path,
                    evidence_id="memory_digest",
                    producer="memory",
                    generation_mode="system",
                )

        experience_digest_text: Optional[str] = None
        experience_hits = self.retrieve_experience_hints(
            research_question=question,
            database=database,
        )
        if experience_hits:
            bank = self._experience_bank()
            bank_record_count = (
                len(bank.records()) if bank is not None else len(experience_hits)
            )
            retired_count = max(0, bank_record_count - len(experience_hits))
            lines = [
                "# Experience Hints",
                "",
                "Only deterministic, audit-safe experience buckets are shown: "
                "concept_usage_hint and failure_counter_example. These are hints, "
                "not ICU rules; cohort definitions and clinical rules still come "
                "from the concept dictionary and validators.",
                "",
                f"Selected {len(experience_hits)} card(s); withheld/retired "
                f"{retired_count} card(s) below the retrieval threshold or top-k.",
            ]
            for idx, (record, score) in enumerate(experience_hits, start=1):
                lines.extend(
                    [
                        "",
                        f"## Card {idx}: {record.kind}",
                        f"- score: {score:.3f}",
                        f"- database: {record.database}",
                        f"- cohort: {record.cohort_name}",
                        f"- produced_by: {record.producer_run_id or 'unknown'}",
                        f"- reason: lexical overlap with the current question"
                        + (
                            " plus same-database boost"
                            if database and record.database == database
                            else ""
                        ),
                        f"- summary: {record.summary}",
                    ]
                )
            experience_digest_text = "\n".join(lines) + "\n"
            experience_path = run_dir / "experience_hints.md"
            experience_path.write_text(experience_digest_text, encoding="utf-8")
            if evidence.get("experience_hints") is None:
                evidence.register_file(
                    kind="log",
                    description="Audit log of cross-run experience cards fed to the planner.",
                    source_path=experience_path,
                    evidence_id="experience_hints",
                    producer="experience_bank",
                    generation_mode="system",
                )

        agent_context = context
        planner_notes: List[str] = []
        if memory_digest_text:
            planner_notes.append("RunMemory digest for planner:\n" + memory_digest_text)
        if experience_digest_text:
            planner_notes.append(
                "Experience hints for planner:\n" + experience_digest_text
            )
        if planner_notes:
            note = "\n\n".join(planner_notes)
            agent_notes = f"{context.notes}\n\n{note}" if context.notes else note
            agent_context = agent_context.model_copy(update={"notes": agent_notes})
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
                )
                # O17 — Front-door hypothesis generation. Opt-in; writes
                # ``hypothesis_candidates.json`` + ``.md`` so the paper
                # Methods section can quote "Out of N candidates we
                # preregistered Q" rather than "we picked Q".
                if self._enable_hypothesis_generator:
                    try:
                        hg_result = generate_hypotheses(
                            context=agent_context,
                            citations=list(preplan_literature.citations),
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
                    aborted = self._finalise_aborted(
                        run_id=run_id,
                        run_dir=run_dir,
                        context=context,
                        context_path=context_path,
                        evidence=evidence,
                        findings=findings,
                        reason="hypothesis_blueprint_blocked",
                    )
                    return _PlanPhaseResult(
                        context=context,
                        agent_context=agent_context,
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
                note = render_hypothesis_blueprint_for_prompt(
                    blueprint,
                    literature=preplan_literature,
                )
                agent_notes = (
                    f"{agent_context.notes}\n\n{note}" if agent_context.notes else note
                )
                agent_context = agent_context.model_copy(update={"notes": agent_notes})
            except Exception as exc:
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
            )
            know_how_binding = PlannerKnowHowBinding.from_prepared(prepared_know_how)

        study_design_brief = None
        article_contract = None
        article_figure_strategy = None
        analysis_blueprint = None
        try:
            study_design_brief = build_study_design_brief(agent_context)
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
                agent_context,
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
            article_figure_strategy = build_article_figure_strategy(agent_context)
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
                agent_context,
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
        if self._enable_cost_tracking:
            cost_meter = (
                CostMeter(
                    price_table=(
                        dict(self._cost_price_table) if self._cost_price_table else None
                    ),
                )
                if self._cost_price_table is not None
                else CostMeter()
            )
            # Order: envelope wraps the innermost client so prompt /
            # response hashes are computed on the exact strings the
            # agent sent / received; the metered layer then observes
            # ``last_usage`` the envelope passes through.
            if repro_envelope is not None:
                env_resolver = envelope_role_resolver(
                    llm,
                    repro_envelope,
                    seed=self._llm_seed,
                )

                class _EnvelopeShim:
                    """Bridges a ``role -> client`` resolver back into an LLMClient-like object.

                    ``metered_role_resolver`` expects an object it can
                    call ``resolve_role_client`` on. The shim exposes
                    ``for_role`` so the downstream resolver takes our
                    envelope-wrapped client as the inner client.
                    """

                    name = "envelope_shim"

                    def __init__(self, resolver):
                        self._resolver = resolver

                    def for_role(self, role: str):
                        return self._resolver(role)

                    # A no-op ``complete`` so static checks don't trip;
                    # the real call always goes through ``for_role``.
                    def complete(self, *args, **kwargs):  # pragma: no cover
                        raise RuntimeError(
                            "EnvelopeShim is a role dispatcher; call for_role() first."
                        )

                role_resolver = metered_role_resolver(
                    _EnvelopeShim(env_resolver),
                    cost_meter,
                )
            else:
                role_resolver = metered_role_resolver(llm, cost_meter)
        elif repro_envelope is not None:
            role_resolver = envelope_role_resolver(
                llm,
                repro_envelope,
                seed=self._llm_seed,
            )
        else:

            def role_resolver(role: str):
                return resolve_role_client(llm, role)

        # Resume: reuse the locked plan from the prior run instead of
        # re-planning. A non-deterministic planner would otherwise emit a
        # *different* plan on resume, whose step_ids no longer match the
        # completed-step skip set — so the "resume" would silently re-run the
        # whole analysis under new names. Reusing the saved plan keeps the
        # already-completed step_ids aligned and continues from the failed step.
        reused_prior_plan = False
        reused_plan_path: Optional[Path] = None
        migrated_plan_path: Optional[Path] = None
        if resume_state is not None:
            plan, _prior_plan_path = _load_compatible_resume_plan(
                run_dir=run_dir,
                resume_state=resume_state,
            )
            if plan is not None and plan.steps:
                know_how_binding.verify_resume(
                    plan.know_how_decisions,
                    enabled=self._enable_know_how,
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

        if reused_prior_plan:
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
                )
            )
            if migrated_plan_path is not None:
                plan_generation_mode = "resumed_planner_migration"
                findings.append(
                    ValidationFinding(
                        validator="planner_schema_migration",
                        severity="warning",
                        message=(
                            "Migrated legacy remaining adjusted-association "
                            "step(s) to the planner-owned typed model roster."
                        ),
                        detail={
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
                plan_generation_mode = "resumed_planner_migration"
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
                plan_generation_mode = "resumed_planner_migration"
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
                            "plan_path": str(
                                trajectory_migration_path.relative_to(run_dir)
                            ),
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
                plan_generation_mode = "resumed_planner_migration"
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
                            "plan_path": str(
                                figure_edge_migration_path.relative_to(run_dir)
                            ),
                        },
                    )
                )
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
            plan_generation_mode = "llm"
            planner = PlannerAgent(role_resolver("planner"))
            try:
                plan = planner.run(
                    agent_context,
                    **know_how_binding.planner_kwargs,
                )
                planner_prompt_metrics = know_how_binding.prompt_metrics(
                    planner, agent_context
                )
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
        # Skip the plan-shaping transforms when resuming: the saved plan is
        # already in its final, transformed form, and re-running split/cap/
        # ensure_* could rename or reorder step_ids and break the resume skip
        # set. A freshly generated plan still gets the full treatment.
        if not reused_prior_plan:
            plan, plan_contract_findings = _enforce_advanced_plan_contract(
                plan=plan,
                context=context,
            )
            findings.extend(plan_contract_findings)
            plan, split_findings = _split_table_and_figure_outputs_in_plan(plan=plan)
            findings.extend(split_findings)
            plan, report_input_findings = _augment_report_typed_product_inputs(
                plan=plan
            )
            findings.extend(report_input_findings)
            # Force a declared figure step whenever the publication-figure skill
            # will produce one regardless of the plan: the scorer reads
            # analysis_plan.json, and a question-only heuristic misses tasks
            # that never say "figure" yet still require one. Likewise
            # ensure a declared audit/robustness panel, since that evidence is
            # produced (locked robustness specs, data-quality summaries) but the
            # plan often never presents it.
            plan, figure_guard_findings = _ensure_publication_figure_step_in_plan(
                plan=plan,
                context=context,
                force=self._enable_publication_figure_skill,
            )
            findings.extend(figure_guard_findings)
            plan, audit_panel_findings = _ensure_audit_panel_step_in_plan(
                plan=plan,
                context=context,
            )
            findings.extend(audit_panel_findings)

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
                )
            )
            plan = ensure_cohort_definition(plan)
            plan = ensure_robustness_specs(plan)
            # Final gate: if the plan implies a cohort but still has no
            # structured inclusion/exclusion (the retry above didn't recover
            # it), record a loud, auditable contract error instead of silently
            # running the analysis on the full universe.
            findings.extend(_cohort_definition_contract_findings(plan))
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
                final_brief = build_study_design_brief(
                    agent_context,
                    analysis_type=plan.analysis_type,
                )
                final_contract = build_article_analysis_contract(
                    agent_context,
                    brief=final_brief,
                    analysis_type=plan.analysis_type,
                )
                final_strategy = build_article_figure_strategy(
                    agent_context,
                    analysis_family=final_brief.analysis_family,
                )
                final_blueprint = build_analysis_blueprint(
                    agent_context,
                    brief=final_brief,
                    contract=final_contract,
                    figure_strategy=final_strategy,
                )
                final_payloads = (
                    (
                        "study_design_brief_final",
                        "study_design_brief.final.json",
                        final_brief,
                    ),
                    (
                        "article_analysis_contract_final",
                        "article_analysis_contract.final.json",
                        final_contract,
                    ),
                    (
                        "article_figure_strategy_final",
                        "article_figure_strategy.final.json",
                        final_strategy,
                    ),
                    (
                        "analysis_blueprint_final",
                        "analysis_blueprint.final.json",
                        final_blueprint,
                    ),
                )
                for evidence_id, filename, payload in final_payloads:
                    final_path = run_dir / filename
                    final_path.write_text(
                        payload.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    if evidence.get(evidence_id) is None:
                        evidence.register_file(
                            kind="log",
                            description=(
                                "Planner-final article design authority bound to "
                                f"analysis_type={plan.analysis_type}."
                            ),
                            source_path=final_path,
                            evidence_id=evidence_id,
                            producer="planner_contract_finalizer",
                            generation_mode="deterministic_skill",
                        )
                study_design_brief = final_brief
                article_contract = final_contract
                article_figure_strategy = final_strategy
                analysis_blueprint = final_blueprint
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
        plan_path = (
            migrated_plan_path or reused_plan_path or (run_dir / "analysis_plan.json")
        )
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
        know_how_binding.persist_prompt_metrics(
            planner_prompt_metrics,
            run_dir=run_dir,
            evidence=evidence,
        )
        write_locked_cohort_definition(
            run_dir=run_dir,
            plan=plan,
            evidence=evidence,
            prompt_pack_version=prompt_version,
            llm_signature=llm_signature,
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
        # stays reachable via EASYICU_UNIVERSE_PARQUET for robustness steps.
        if not reused_prior_plan:
            analysis_cohort = materialize_locked_analysis_cohort(
                run_dir=run_dir,
                plan=plan,
                universe_path=cohort_path,
                context=context,
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
                            "stays available as EASYICU_UNIVERSE_PARQUET."
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
                findings.append(
                    ValidationFinding(
                        validator="cohort_materializer",
                        severity="warning",
                        message=(
                            "Could not auto-apply the locked cohort definition to the "
                            "universe; downstream steps read the unfiltered universe "
                            "and must apply inclusion/exclusion themselves. Reason: "
                            f"{analysis_cohort['error']}"
                        ),
                    )
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
        return bank.retrieve(
            research_question=research_question,
            database=database,
            top_k=self._experience_bank_top_k,
            min_similarity=self._experience_bank_min_similarity,
        )

    def reflect_and_persist_experience(
        self,
        *,
        run_dir: Path,
        context: ResearchContext,
        database: str,
        cohort_name: str,
    ) -> List[ExperienceRecord]:
        """Mine experience records from a completed run and write them back.

        Reads ``run_status.json`` from ``run_dir`` for the gates +
        findings + superseded-error partition. Returns the records
        that were registered (after dedup against the existing bank);
        an empty list when the feature is disabled or the run dir
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

    @exclusive_run_execution
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
        _use_graph: bool = False,
    ) -> PipelineResult:
        """Run the explicit Plan → Execute → Write phases for one cohort.

        Pass ``_use_graph=True`` (or call :meth:`run_with_graph`) to
        dispatch the phases through the opt-in langgraph wrapper in
        :mod:`easyicu.research_agent.graph`. Behaviour is identical
        either way; the graph path is a PoC for future branching /
        checkpointing work.
        """
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
        audit_logger: Optional[AuditLogger] = None

        def _emit_progress(stage: str, message: str, **extra: Any) -> None:
            if audit_logger is not None:
                try:
                    audit_logger.emit(
                        phase=stage,
                        event=message,
                        status=str(extra.get("status", "running")),
                        step_id=(
                            str(extra.get("step_id")) if extra.get("step_id") else None
                        ),
                        detail={
                            k: v
                            for k, v in extra.items()
                            if k not in {"status", "step_id"}
                        },
                    )
                except Exception:
                    pass
            if progress_callback is None:
                return
            payload = {
                "stage": stage,
                "message": message,
                "status": extra.pop("status", "running"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            payload.update(extra)
            try:
                progress_callback(payload)
            except Exception:
                pass

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

        runtime_capabilities = self._preflight_execution_runtime(
            run_dir=run_dir,
            cohort_path=cohort_path,
            target_outcome=target_outcome,
        )
        _emit_progress(
            "runtime",
            "Execution runtime validated before planning.",
            run_id=run_id,
            method_capabilities=list(runtime_capabilities),
        )

        def _plan_invoker():
            return self._run_plan_phase(
                question=question,
                cohort_path=cohort_path,
                cohort_name=cohort_name,
                database=database,
                target_outcome=target_outcome,
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

        def _provenance_hook(plan_result):
            # O27 — Raw EHR provenance. Hash the cohort parquet and any
            # user-supplied source files and register as evidence so the
            # manuscript's provenance chain goes: raw EHR -> cohort ->
            # analysis artefacts -> manuscript. Only runs after the plan
            # phase succeeded.
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
                plan_result.findings.append(
                    ValidationFinding(
                        validator="provenance",
                        severity="warning",
                        message=(
                            f"Failed to compute raw-EHR provenance bundle: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                )

        def _execute_invoker(plan_result):
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

        def _write_invoker(plan_result, execute_result):
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
                            "STRICT evidence enforcement blocked manuscript "
                            f"generation: {exc}"
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

        def _finalise_invoker(plan_result, execute_result, write_result):
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

        if _use_graph:
            from .graph import build_pipeline_graph

            graph = build_pipeline_graph(
                plan_invoker=_plan_invoker,
                execute_invoker=_execute_invoker,
                write_invoker=_write_invoker,
                finalise_invoker=_finalise_invoker,
                provenance_hook=_provenance_hook,
            )
            final_state = graph.invoke({})
            return final_state["final_result"]

        plan_result = _plan_invoker()
        if plan_result.aborted_result is not None:
            return plan_result.aborted_result
        _provenance_hook(plan_result)
        execute_result = _execute_invoker(plan_result)
        write_result = _write_invoker(plan_result, execute_result)
        return _finalise_invoker(plan_result, execute_result, write_result)

    def run_from_spec(
        self,
        spec: Union[ExperimentSpec, Dict[str, Any]],
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        """Run the pipeline from a typed YAML/JSON experiment specification."""
        spec_obj = (
            spec
            if isinstance(spec, ExperimentSpec)
            else ExperimentSpec.model_validate(spec)
        )
        kwargs = spec_obj.run_kwargs()
        kwargs["experiment_spec"] = spec_obj
        kwargs["progress_callback"] = progress_callback
        return self.run(**kwargs)

    async def run_async(self, **kwargs: Any) -> PipelineResult:
        """Async wrapper for UI/API runtimes that need non-blocking orchestration."""
        return await asyncio.to_thread(self.run, **kwargs)

    def run_with_graph(self, **kwargs: Any) -> PipelineResult:
        """Opt-in PoC: dispatch the phases through a langgraph StateGraph.

        Requires the ``agentic`` extra (``pip install easyicu[agentic]``).
        Behaviour is identical to :meth:`run`; see
        :mod:`easyicu.research_agent.graph` for the wrapper design.
        """
        kwargs.pop("_use_graph", None)
        return self.run(_use_graph=True, **kwargs)

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
            "enable_deterministic_runner_repair": bool(
                self._enable_deterministic_runner_repair
            ),
            "latex_venue_template": self._latex_venue_template,
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


def _build_probe_summary(
    *,
    context: ResearchContext,
    cohort_path: Path,
    out_dir: Path,
) -> tuple[Dict[str, Any], List[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cohort_path)
    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_outcome": context.target_outcome,
        "top_missing_columns": [],
        "score_completeness": [],
    }
    missing_rows = []
    for col in df.columns:
        frac = float(df[col].isna().mean()) if len(df) else 0.0
        missing_rows.append(
            {
                "variable": col,
                "fraction_missing": frac,
                "n_missing": int(df[col].isna().sum()),
                "n_unique_non_missing": int(df[col].dropna().nunique()),
            }
        )
    missing_df = pd.DataFrame(missing_rows).sort_values(
        ["fraction_missing", "variable"], ascending=[False, True]
    )
    summary["top_missing_columns"] = missing_df.head(10).to_dict(orient="records")
    files: List[Path] = []
    missing_path = out_dir / "probe_variable_profile.csv"
    missing_df.to_csv(missing_path, index=False)
    files.append(missing_path)

    from easyicu.io.data_quality import composite_score_completeness

    for variable in context.variables:
        if variable.role not in {
            VariableRole.ORDINAL_SCORE,
            VariableRole.COMPOSITE_SCORE,
        }:
            continue
        if variable.name not in df.columns:
            continue
        observed = df[variable.name].dropna()
        if observed.empty:
            continue
        stats: Dict[str, Any] = {
            "variable": variable.name,
            "min": float(observed.min()),
            "max": float(observed.max()),
            "n_zero": (
                int((observed == 0).sum())
                if pd.api.types.is_numeric_dtype(observed)
                else None
            ),
        }
        n_components_col = f"{variable.name}_n_components"
        if n_components_col in df.columns:
            stats["completeness"] = composite_score_completeness(
                df,
                variable.name,
                n_components_col=n_components_col,
            )
            summary["score_completeness"].append(stats)
    summary_path = out_dir / "probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    files.append(summary_path)
    return summary, files


def _promote_sibling_figure_exports(*, out_dir: Path) -> Optional[str]:
    """Promote figure files written beside ``outputs/`` into ``outputs/``.

    Some generated scripts treat ``STEP_OUT_DIR`` as a filename stem and
    write ``outputs.svg`` / ``outputs.png`` beside the canonical
    ``outputs/`` directory. The execution contract only registers files inside
    ``outputs/``, so normalize that common layout before declaring the
    publication figure missing.
    """
    parent = out_dir.parent
    source_stem = out_dir.name
    figure_suffixes = (".pdf", ".png", ".svg", ".tiff", ".tif", ".pptx")
    figure_sources = [
        parent / f"{source_stem}{suffix}"
        for suffix in figure_suffixes
        if (parent / f"{source_stem}{suffix}").is_file()
    ]
    if not figure_sources:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    target_stem = "publication_figure"
    exported_figure_files: List[str] = []
    for source in figure_sources:
        target = out_dir / f"{target_stem}{source.suffix.lower()}"
        shutil.copy2(source, target)
        exported_figure_files.append(target.name)

    contract_source = parent / f"{source_stem}.figure_contract.json"
    if contract_source.is_file():
        shutil.copy2(contract_source, out_dir / f"{target_stem}.figure_contract.json")

    step_summary_path = out_dir / "step_summary.json"
    summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary = loaded
        except Exception:
            summary = {}
    summary.setdefault("publication_figure_rescue", {})
    summary["publication_figure_rescue"].update(
        {
            "mode": "sibling_outputs_stem",
            "source_stem": source_stem,
            "source_dir": str(parent),
        }
    )
    summary["figure_files"] = sorted(exported_figure_files)
    if exported_figure_files:
        summary["figure_path"] = sorted(exported_figure_files)[0]
    step_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "sibling_figure_exports_promote_v1"


def _promote_prior_publication_bundle(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    required_roles: Optional[Sequence[str]] = None,
    require_declared_sources: bool = False,
) -> Optional[str]:
    """Promote the strongest earlier figure bundle into a publication step."""
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    figure_suffixes = {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}
    contract_suffix = ".figure_contract.json"
    best: Optional[tuple[tuple[int, int, int], str, Dict[str, Path]]] = None
    role_filter = {
        str(role).strip().lower()
        for role in (required_roles or [])
        if str(role).strip()
    }

    # A split ``<parent>_figure`` step may only promote exports produced by its
    # direct parent.  If that parent has no figure bundle, copying an unrelated
    # earlier figure is scientifically worse than failing closed (for example,
    # a cohort-flow figure must never satisfy an absolute-risk figure step).
    # Generic terminal publication steps that do not have a sibling parent keep
    # the historical run-wide strongest-bundle behaviour.
    parent_step_id = str(current_step_id or "").removesuffix("_figure")
    direct_parent = steps_dir / parent_step_id
    if parent_step_id != str(current_step_id or "") and direct_parent.is_dir():
        candidate_step_dirs = [direct_parent]
    else:
        candidate_step_dirs = sorted(steps_dir.iterdir())

    for step_dir in candidate_step_dirs:
        if not step_dir.is_dir() or step_dir.name == current_step_id:
            continue
        outputs_dir = step_dir / "outputs"
        if not outputs_dir.exists():
            continue
        bundles: Dict[str, Dict[str, Path]] = {}
        for path in outputs_dir.iterdir():
            if not path.is_file():
                continue
            if path.name.endswith(contract_suffix):
                stem = path.name[: -len(contract_suffix)]
                bundles.setdefault(stem, {})["contract"] = path
                continue
            if path.suffix.lower() in figure_suffixes:
                bundles.setdefault(path.stem, {})[path.suffix.lower()] = path
        for stem, files in bundles.items():
            figure_count = sum(1 for key in files if key.startswith("."))
            if figure_count == 0:
                continue
            if role_filter and not _publication_bundle_has_any_role(files, role_filter):
                continue
            if (
                require_declared_sources
                and not _publication_bundle_has_resolvable_sources(files)
            ):
                continue
            score = (
                1 if "publication_figure" in stem else 0,
                1 if "primary_association" in stem else 0,
                figure_count,
            )
            if best is None or score > best[0]:
                best = (score, stem, files)

    if best is None:
        return None

    _, source_stem, files = best
    target_stem = "publication_figure"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, source in files.items():
        if key == "contract":
            target = out_dir / f"{target_stem}.figure_contract.json"
        else:
            target = out_dir / f"{target_stem}{key}"
        shutil.copy2(source, target)

    # A figure contract is not self-contained when its source-data and panel
    # evidence files remain behind in the analysis step.  Promotion previously
    # copied only the rendered exports + JSON contract, leaving a formally
    # untraceable bundle in the split figure step.  Copy every file-like local
    # reference while preserving safe relative names; logical evidence IDs
    # without a file suffix are intentionally left alone.
    copied_trace_files: List[str] = []
    contract_path = files.get("contract")
    if contract_path is not None and contract_path.is_file():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception:
            contract = {}

        artifact_refs = _publication_contract_file_references(contract)

        source_outputs = contract_path.parent.resolve()
        for ref in dict.fromkeys(artifact_refs):
            relative_ref = Path(ref)
            if relative_ref.is_absolute() or ".." in relative_ref.parts:
                relative_ref = Path(relative_ref.name)
            source = (source_outputs / relative_ref).resolve()
            if not source.is_relative_to(source_outputs) or not source.is_file():
                continue
            target = out_dir / relative_ref
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_trace_files.append(str(relative_ref))

    step_summary_path = out_dir / "step_summary.json"
    summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            summary = json.loads(step_summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    summary.setdefault("publication_figure_rescue", {})
    source_outputs_dir = files[next(iter(files))].parent
    source_step_id = source_outputs_dir.parent.name
    summary["publication_figure_rescue"].update(
        {
            "mode": "promotion",
            "source_step_stem": source_stem,
            "source_outputs_dir": str(source_outputs_dir),
            "copied_trace_files": sorted(copied_trace_files),
        }
    )
    exported_figure_files = [
        str((out_dir / f"{target_stem}{key}").name)
        for key in sorted(files)
        if key != "contract"
    ]
    summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_publication_bundle_promotion",
            "rendering_only": True,
            "source_step_id": source_step_id,
            "source_data_files": sorted(copied_trace_files),
        }
    )
    summary["figure_files"] = exported_figure_files
    if exported_figure_files:
        summary["figure_path"] = exported_figure_files[0]
    step_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "publication_bundle_promote_v1"


def _publication_contract_file_references(contract: Any) -> List[str]:
    """Return local file-like source/evidence references from a contract."""

    artifact_refs: List[str] = []

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            token = value.strip()
            if token and Path(token).suffix:
                artifact_refs.append(token)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _collect(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                _collect(item)

    if isinstance(contract, dict):
        _collect(contract.get("source_data"))
        panels = contract.get("panels") or []
        if isinstance(panels, list):
            for panel in panels:
                if isinstance(panel, dict):
                    _collect(panel.get("evidence_ids"))
    return list(dict.fromkeys(artifact_refs))


def _publication_bundle_has_resolvable_sources(files: Mapping[str, Path]) -> bool:
    """Require every declared file reference to exist beside the parent bundle."""

    contract_path = files.get("contract")
    if contract_path is None or not contract_path.is_file():
        return False
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    refs = _publication_contract_file_references(contract)
    if not refs:
        return False
    source_outputs = contract_path.parent.resolve()
    for ref in refs:
        relative_ref = Path(ref)
        if relative_ref.is_absolute() or ".." in relative_ref.parts:
            relative_ref = Path(relative_ref.name)
        source = (source_outputs / relative_ref).resolve()
        if not source.is_relative_to(source_outputs) or not source.is_file():
            return False
    return True


def _publication_bundle_has_any_role(
    files: Mapping[str, Path], required_roles: set[str]
) -> bool:
    contract_path = files.get("contract")
    if contract_path is None or not contract_path.exists():
        return False
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(contract, dict):
        return False
    roles: set[str] = set()
    panels = contract.get("panels") or []
    if isinstance(panels, list):
        for panel in panels:
            if not isinstance(panel, dict):
                continue
            role = str(panel.get("role") or "").strip().lower()
            if role:
                roles.add(role)
    top_role = str(contract.get("role") or "").strip().lower()
    if top_role:
        roles.add(top_role)
    return bool(roles & required_roles)


def _figure_parent_candidate_step_dirs(
    *, steps_dir: Path, current_step_id: str
) -> tuple[List[Path], bool]:
    """Return direct parent only for split figures, else legacy prior steps.

    A ``<analysis>_figure`` child is an ownership edge: it may render only the
    standardized products emitted by ``<analysis>``.  Searching an older step
    for a same-shaped CSV can silently switch estimand or cohort while remaining
    formally source-backed.  Legacy terminal overview figures without an
    existing direct parent retain run-wide rescue behavior.
    """

    parent_step_id = str(current_step_id or "").removesuffix("_figure")
    direct_parent = steps_dir / parent_step_id
    is_split = parent_step_id != str(current_step_id or "")
    if is_split and direct_parent.is_dir():
        return [direct_parent], True
    return (
        [
            step_dir
            for step_dir in sorted(steps_dir.iterdir())
            if step_dir.is_dir() and step_dir.name != current_step_id
        ],
        False,
    )


def _render_prediction_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically build a validation figure from prior prediction outputs.

    Some small models successfully write ``model_performance.csv`` and
    ``step_summary.json`` in the parent model-training step but fail to
    render the follow-up figure step. When that happens, we can still
    construct a publication-style validation bundle from the structured
    parent artefacts instead of failing the entire run.
    """
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    best_parent: Optional[tuple[Path, Path, Dict[str, Any]]] = None
    candidate_step_dirs, _direct_parent_only = _figure_parent_candidate_step_dirs(
        steps_dir=steps_dir, current_step_id=current_step_id
    )
    for step_dir in candidate_step_dirs:
        outputs_dir = step_dir / "outputs"
        perf_path = outputs_dir / "model_performance.csv"
        summary_path = outputs_dir / "step_summary.json"
        if not perf_path.exists() or not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        best_parent = (perf_path, summary_path, summary)
        break
    if best_parent is None:
        return None

    perf_path, summary_path, summary = best_parent
    try:
        frame = pd.read_csv(perf_path)
    except Exception:
        return None
    metric_cols = [col for col in ("auroc", "brier_score") if col in frame.columns]
    calib_cols = [
        col
        for col in ("calibration_slope", "calibration_intercept")
        if col in frame.columns
    ]
    if not metric_cols and not calib_cols:
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
    fig, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 82 / 25.4), constrained_layout=True
    )
    apply_publication_style(fig)
    if not isinstance(axes, (list, tuple)):
        axes = axes.ravel()
    folds = frame.get("fold")
    if folds is None:
        folds = pd.Series([f"Fold {idx + 1}" for idx in range(len(frame))])
    folds = folds.astype(str)

    ax1, ax2 = axes[0], axes[1]
    if "auroc" in frame.columns:
        ax1.plot(
            folds,
            frame["auroc"].astype(float),
            marker="o",
            linewidth=1.4,
            label="AUROC",
        )
    if "brier_score" in frame.columns:
        ax1.plot(
            folds,
            frame["brier_score"].astype(float),
            marker="s",
            linewidth=1.2,
            label="Brier",
        )
    ax1.set_title("Cross-validation discrimination", loc="left", pad=4)
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Metric value")
    ax1.tick_params(axis="x", rotation=35)
    ax1.legend(frameon=False, fontsize=7)
    add_panel_label(ax1, "A", x=-0.1)

    if "calibration_slope" in frame.columns:
        ax2.plot(
            folds,
            frame["calibration_slope"].astype(float),
            marker="o",
            linewidth=1.4,
            label="Slope",
        )
        ax2.axhline(1.0, linestyle="--", linewidth=0.8, color="#8F8F8F")
    if "calibration_intercept" in frame.columns:
        ax2.plot(
            folds,
            frame["calibration_intercept"].astype(float),
            marker="s",
            linewidth=1.2,
            label="Intercept",
        )
        ax2.axhline(0.0, linestyle=":", linewidth=0.8, color="#B64342")
    ax2.set_title("Cross-validation calibration", loc="left", pad=4)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Calibration statistic")
    ax2.tick_params(axis="x", rotation=35)
    if calib_cols:
        ax2.legend(frameon=False, fontsize=7)
    add_panel_label(ax2, "B", x=-0.1)

    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=(
            "Prediction-model validation metrics are summarised from the "
            "registered cross-validation performance table."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Discrimination",
                "role": "validation",
                "claim": "Fold-level AUROC and Brier score are derived from the model-performance table.",
                "evidence_ids": ["model_performance", "01_model_training"],
            },
            {
                "panel_id": "B",
                "title": "Calibration",
                "role": "validation",
                "claim": "Fold-level calibration slope and intercept are derived from the registered step summary and performance table.",
                "evidence_ids": ["model_performance", "01_model_training"],
            },
        ],
        source_data=["model_performance", "01_model_training"],
        statistics_note=(
            "Deterministic rescue figure generated from parent-step outputs "
            "when the figure-only child step did not emit exports."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "publication_figure",
        contract=contract,
        dpi=300,
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
    existing_summary.setdefault("publication_figure_rescue", {})
    existing_summary["publication_figure_rescue"].update(
        {
            "mode": "prediction_validation_from_parent_outputs",
            "source_model_performance": str(perf_path),
            "source_step_summary": str(summary_path),
        }
    )
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    existing_summary["figure_files"] = figure_files
    if figure_files:
        existing_summary["figure_path"] = figure_files[0]
    existing_summary.setdefault("cv_auroc_mean", summary.get("statistic:auroc"))
    existing_summary.setdefault("cv_brier_mean", summary.get("statistic:brier_score"))
    existing_summary.setdefault(
        "calibration_slope", summary.get("statistic:calibration_slope")
    )
    existing_summary.setdefault(
        "calibration_intercept", summary.get("statistic:calibration_intercept")
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "prediction_publication_bundle_from_parent_outputs_v1"


def _render_cohort_overlap_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically build a cohort-definition overlap figure.

    Cohort eligibility and overlap steps do not emit OR/CI tables, so they
    should never fall through to the generic association forest rescue. This
    renderer consumes the immediate parent step's attrition and overlap tables
    and writes traceable source-data copies keyed by cohort-definition ids.
    """

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    parent_step_id = current_step_id.removesuffix("_figure")
    parent_outputs = steps_dir / parent_step_id / "outputs"
    attrition_path = parent_outputs / "alternative_cohort_attrition.csv"
    overlap_path = parent_outputs / "cohort_overlap_matrix.csv"
    audit_path = parent_outputs / "cohort_definition_empirical_equivalence_audit.csv"
    if not attrition_path.exists() or not overlap_path.exists():
        return None

    try:
        attrition = pd.read_csv(attrition_path)
        overlap = pd.read_csv(overlap_path)
    except Exception:
        return None

    attrition_required = {
        "definition_id",
        "definition_label",
        "definition_type",
        "n_included",
        "included_pct_of_rows",
        "overlap_with_primary_pct_of_primary",
        "moved_in_vs_primary_n",
        "moved_out_vs_primary_n",
    }
    overlap_required = {"definition_a", "definition_b", "jaccard"}
    if not attrition_required <= set(attrition.columns):
        return None
    if not overlap_required <= set(overlap.columns):
        return None
    if attrition.empty or overlap.empty:
        return None

    source_attrition = attrition.copy()
    for col in (
        "n_included",
        "n_excluded",
        "included_pct_of_rows",
        "overlap_with_primary_n",
        "overlap_with_primary_pct_of_primary",
        "overlap_with_primary_pct_of_definition",
        "moved_in_vs_primary_n",
        "moved_out_vs_primary_n",
    ):
        if col in source_attrition.columns:
            source_attrition[col] = pd.to_numeric(
                source_attrition[col], errors="coerce"
            )

    def _cohort_definition_display_label(row: Mapping[str, Any]) -> str:
        definition_id = str(row.get("definition_id") or "").strip()
        known = {
            "primary_adult_los1_all_vitals_sepsis3_derivable": "Primary",
            "alt_adult_no_los_all_vitals_sepsis3_derivable": "No LOS threshold",
            "alt_adult_los1_three_of_four_vitals_sepsis3_derivable": ">=3 of 4 vitals",
            "alt_adult_los1_no_temp_requirement_sepsis3_derivable": "No temperature",
            "alt_adult_los2_all_vitals_sepsis3_derivable": "LOS >=2 d",
            "primary_adult_los1_all_vitals_sep3_measured": "Primary",
            "alt_adult_no_los_all_vitals_sep3_measured": "No LOS threshold",
            "alt_adult_los1_three_of_four_vitals_sep3_measured": ">=3 of 4 vitals",
            "alt_adult_los1_no_temp_requirement_sep3_measured": "No temperature",
            "alt_adult_los2_all_vitals_sep3_measured": "LOS >=2 d",
        }
        if definition_id in known:
            return known[definition_id]
        label = str(row.get("definition_label") or definition_id or "").strip()
        return label or "Definition"

    source_attrition["display_label"] = [
        _cohort_definition_display_label(row)
        for row in source_attrition.to_dict(orient="records")
    ]

    source_overlap = overlap.copy()
    for col in (
        "n_a",
        "n_b",
        "intersection_n",
        "union_n",
        "jaccard",
        "a_in_b_pct",
        "b_in_a_pct",
    ):
        if col in source_overlap.columns:
            source_overlap[col] = pd.to_numeric(source_overlap[col], errors="coerce")

    out_dir.mkdir(parents=True, exist_ok=True)
    source_attrition_path = out_dir / "publication_figure_definition_source_data.csv"
    source_overlap_path = out_dir / "publication_figure_overlap_source_data.csv"
    source_attrition.to_csv(source_attrition_path, index=False)
    source_overlap.to_csv(source_overlap_path, index=False)

    plot_df = source_attrition.reset_index(drop=True).copy()
    labels = plot_df["display_label"].astype(str).tolist()
    y = list(range(len(plot_df)))

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
    fig = plt.figure(figsize=(183 / 25.4, 132 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.05, 1.05],
        height_ratios=[1.0, 0.88],
        left=0.18,
        right=0.98,
        top=0.92,
        bottom=0.16,
        wspace=0.45,
        hspace=0.58,
    )
    ax_n = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[1, 0])
    ax_heat = fig.add_subplot(grid[:, 1])

    colors = [
        (
            palette.get("blue", "#0F4D92")
            if str(row.get("definition_type", "")).lower() == "primary"
            else palette.get("teal", "#42949E")
        )
        for row in plot_df.to_dict(orient="records")
    ]
    ax_n.barh(y, plot_df["n_included"].astype(float), color=colors, height=0.58)
    ax_n.set_yticks(y)
    ax_n.set_yticklabels(labels)
    ax_n.invert_yaxis()
    ax_n.set_xlabel("Included ICU stays")
    ax_n.set_title("Eligibility definitions", loc="left", pad=4)
    ax_n.grid(axis="x", color=palette.get("neutral_light", "#D8D8D8"), linewidth=0.55)
    add_panel_label(ax_n, "A", x=-0.20)

    moved_in = plot_df["moved_in_vs_primary_n"].astype(float)
    moved_out = -plot_df["moved_out_vs_primary_n"].astype(float)
    ax_delta.barh(
        y,
        moved_in,
        color=palette.get("green", "#008B5E"),
        height=0.36,
        label="Added vs primary",
    )
    ax_delta.barh(
        y,
        moved_out,
        color=palette.get("orange", "#E28E2C"),
        height=0.36,
        label="Removed vs primary",
    )
    ax_delta.axvline(0, color=palette.get("neutral", "#8F8F8F"), linewidth=0.8)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels(labels)
    ax_delta.invert_yaxis()
    ax_delta.set_xlabel("ICU-stay count change")
    ax_delta.set_title("Movement relative to primary", loc="left", pad=4)
    ax_delta.grid(
        axis="x", color=palette.get("neutral_light", "#D8D8D8"), linewidth=0.55
    )
    ax_delta.legend(
        frameon=False,
        fontsize=6.2,
        loc="lower center",
        bbox_to_anchor=(0.54, -0.36),
        ncol=2,
    )
    add_panel_label(ax_delta, "B", x=-0.20)

    definition_order = plot_df["definition_id"].astype(str).tolist()
    label_map = dict(zip(definition_order, labels))
    heat = (
        source_overlap.pivot_table(
            index="definition_a",
            columns="definition_b",
            values="jaccard",
            aggfunc="first",
        )
        .reindex(index=definition_order, columns=definition_order)
        .astype(float)
    )
    image = ax_heat.imshow(
        heat.to_numpy() * 100.0,
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="auto",
    )
    ax_heat.set_xticks(range(len(definition_order)))
    ax_heat.set_xticklabels(
        [
            _short_figure_label(label_map.get(item, item), limit=18)
            for item in definition_order
        ],
        rotation=45,
        ha="right",
    )
    ax_heat.set_yticks(range(len(definition_order)))
    ax_heat.set_yticklabels(
        [
            _short_figure_label(label_map.get(item, item), limit=18)
            for item in definition_order
        ]
    )
    ax_heat.set_title("Pairwise cohort overlap", loc="left", pad=4)
    for row_idx in range(len(definition_order)):
        for col_idx in range(len(definition_order)):
            value = heat.iat[row_idx, col_idx]
            if pd.isna(value):
                continue
            ax_heat.text(
                col_idx,
                row_idx,
                f"{value * 100:.0f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="#1F1F1F" if value < 0.72 else "white",
            )
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.set_label("Jaccard overlap (%)")
    add_panel_label(ax_heat, "C", x=-0.12)

    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=(
            "Alternative eligibility definitions change the cohort denominator "
            "and overlap structure, which must be visible before interpreting "
            "model sensitivity."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Eligibility denominators",
                "role": "overview",
                "claim": "Included ICU-stay counts are read from the parent attrition table.",
                "evidence_ids": ["alternative_cohort_attrition"],
            },
            {
                "panel_id": "B",
                "title": "Movement relative to primary",
                "role": "audit",
                "claim": "Each alternative definition's added and removed stays are explicit.",
                "evidence_ids": ["alternative_cohort_attrition"],
            },
            {
                "panel_id": "C",
                "title": "Pairwise overlap",
                "role": "robustness",
                "claim": "Jaccard overlap is computed from the parent overlap matrix.",
                "evidence_ids": ["cohort_overlap_matrix"],
            },
        ],
        source_data=[
            "alternative_cohort_attrition",
            "cohort_overlap_matrix",
            "publication_figure_definition_source_data.csv",
            "publication_figure_overlap_source_data.csv",
        ],
        statistics_note=(
            "Generated deterministically from the parent cohort-definition "
            "attrition and overlap tables; no values are inferred from the image."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "publication_figure",
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
            "method": "deterministic_cohort_overlap_publication_figure_repair",
            "rendering_only": True,
            "source_step_id": parent_step_id,
            "source_attrition_table": str(attrition_path),
            "source_overlap_table": str(overlap_path),
            "source_equivalence_audit": (
                str(audit_path) if audit_path.exists() else None
            ),
            "source_data_files": [
                source_attrition_path.name,
                source_overlap_path.name,
            ],
            "n_definitions": int(len(plot_df)),
            "figure_files": [
                path.name for key, path in outputs.items() if key != "contract"
            ],
            "figure_path": "publication_figure.png",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "cohort_overlap_publication_bundle_from_parent_outputs_v1"


def _render_cohort_flow_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
) -> Optional[str]:
    """Deterministically render a simple sequential cohort-flow contract.

    This is deliberately narrower than the cohort-definition overlap renderer:
    it accepts only a parent ``cohort_flow.csv`` plus ``attrition.csv`` with
    explicit stages, denominators, removals, and exclusion categories.  The
    overlap renderer remains the first choice for multi-definition analyses.
    """

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    parent_step_id = current_step_id.removesuffix("_figure")
    parent_outputs = steps_dir / parent_step_id / "outputs"
    flow_path = parent_outputs / "cohort_flow.csv"
    attrition_path = parent_outputs / "attrition.csv"
    flow_payload = (
        preverified_parent_artifacts.get("cohort_flow.csv")
        if preverified_parent_artifacts is not None
        else None
    )
    attrition_payload = (
        preverified_parent_artifacts.get("attrition.csv")
        if preverified_parent_artifacts is not None
        else None
    )
    if preverified_parent_artifacts is None and (
        not flow_path.exists() or not attrition_path.exists()
    ):
        return None
    if preverified_parent_artifacts is not None and (
        flow_payload is None
        or attrition_payload is None
        or "step_summary.json" not in preverified_parent_artifacts
    ):
        return None

    try:
        flow = pd.read_csv(
            io.BytesIO(flow_payload) if flow_payload is not None else flow_path
        )
        attrition = pd.read_csv(
            io.BytesIO(attrition_payload)
            if attrition_payload is not None
            else attrition_path
        )
    except Exception:
        return None

    flow_required = {
        "stage",
        "n",
        "percent_of_universe",
        "n_removed_from_prior_stage",
        "criterion",
    }
    attrition_required = {
        "attrition_category",
        "n",
        "percent_of_universe",
        "status",
        "reason",
        "partition_role",
    }
    if not flow_required <= set(flow.columns):
        return None
    if not attrition_required <= set(attrition.columns):
        return None
    if flow.empty or attrition.empty:
        return None

    flow_plot = flow.copy()
    attrition_plot = attrition.copy()
    for frame, numeric_columns in (
        (
            flow_plot,
            ("n", "percent_of_universe", "n_removed_from_prior_stage"),
        ),
        (attrition_plot, ("n", "percent_of_universe")),
    ):
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any() or (~frame[column].map(math.isfinite)).any():
                return None
            if (frame[column] < 0).any():
                return None

    if flow_plot["stage"].fillna("").astype(str).str.strip().eq("").any():
        return None
    if (
        attrition_plot["attrition_category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    source_flow = flow.copy()
    source_flow["source_table"] = flow_path.name
    source_attrition = attrition.copy()
    source_attrition["source_table"] = attrition_path.name
    source_flow_path = out_dir / "publication_figure_source_data.csv"
    source_attrition_path = out_dir / "publication_figure_attrition_source_data.csv"
    source_flow.to_csv(source_flow_path, index=False)
    source_attrition.to_csv(source_attrition_path, index=False)

    excluded = attrition_plot[
        attrition_plot["status"].fillna("").astype(str).str.lower().eq("excluded")
    ].copy()

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
    height_mm = max(112.0, min(170.0, 70.0 + 13.0 * len(flow_plot)))
    fig, (ax_flow, ax_attrition) = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, height_mm / 25.4),
        gridspec_kw={"width_ratios": [1.12, 0.88]},
        constrained_layout=True,
    )

    n_stages = len(flow_plot)
    if n_stages == 1:
        y_positions = [0.50]
    else:
        y_positions = [
            0.90 - index * 0.78 / (n_stages - 1) for index in range(n_stages)
        ]
    box_height = min(0.13, 0.52 / max(n_stages, 1))
    flow_rows = flow_plot.reset_index(drop=True).to_dict(orient="records")
    for index, (row, y_pos) in enumerate(zip(flow_rows, y_positions)):
        stage_label = str(row["stage"]).replace("_", " ").strip().title()
        stage_label = _short_figure_label(stage_label, limit=34)
        count = int(round(float(row["n"])))
        percent = float(row["percent_of_universe"])
        facecolor = (
            palette.get("blue", "#0F4D92")
            if index in (0, n_stages - 1)
            else palette.get("teal", "#42949E")
        )
        ax_flow.text(
            0.46,
            y_pos,
            f"{stage_label}\n{count:,} ({percent:.1f}% of universe)",
            ha="center",
            va="center",
            fontsize=7.0,
            color="white",
            transform=ax_flow.transAxes,
            bbox={
                "boxstyle": "round,pad=0.42",
                "facecolor": facecolor,
                "edgecolor": "none",
            },
        )
        if index + 1 >= n_stages:
            continue
        next_y = y_positions[index + 1]
        ax_flow.annotate(
            "",
            xy=(0.46, next_y + box_height / 2),
            xytext=(0.46, y_pos - box_height / 2),
            xycoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": palette.get("neutral", "#8F8F8F"),
                "linewidth": 0.9,
            },
        )
        removed = int(round(float(flow_rows[index + 1]["n_removed_from_prior_stage"])))
        ax_flow.text(
            0.62,
            (y_pos + next_y) / 2,
            f"Removed: {removed:,}",
            ha="left",
            va="center",
            fontsize=6.3,
            color=palette.get("neutral_dark", "#4A4A4A"),
            transform=ax_flow.transAxes,
        )
    ax_flow.set_title("Registered eligibility sequence", loc="left", pad=4)
    ax_flow.set_axis_off()
    add_panel_label(ax_flow, "A", x=-0.04)

    if excluded.empty:
        ax_attrition.text(
            0.5,
            0.5,
            "No excluded categories were registered",
            ha="center",
            va="center",
            transform=ax_attrition.transAxes,
            fontsize=7.0,
        )
        ax_attrition.set_xticks([])
        ax_attrition.set_yticks([])
    else:
        excluded = excluded.reset_index(drop=True)
        y = list(range(len(excluded)))
        values = excluded["n"].astype(float)
        labels = [
            _short_figure_label(
                str(value).replace("_", " ").strip().title(),
                limit=28,
            )
            for value in excluded["attrition_category"]
        ]
        ax_attrition.barh(
            y,
            values,
            color=palette.get("orange", "#E28E2C"),
            height=0.58,
        )
        ax_attrition.set_yticks(y)
        ax_attrition.set_yticklabels(labels)
        ax_attrition.invert_yaxis()
        max_count = float(values.max())
        ax_attrition.set_xlim(0, max(1.0, max_count * 1.28))
        for index, row in excluded.iterrows():
            count = int(round(float(row["n"])))
            percent = float(row["percent_of_universe"])
            x_pos = max(float(row["n"]), max(max_count, 1.0) * 0.015)
            ax_attrition.text(
                x_pos,
                index,
                f" {count:,} ({percent:.1f}%)",
                ha="left",
                va="center",
                fontsize=6.2,
            )
        ax_attrition.set_xlabel("Excluded records")
        ax_attrition.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
        )
    ax_attrition.set_title("Recorded attrition", loc="left", pad=4)
    add_panel_label(ax_attrition, "B", x=-0.16)

    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=(
            "The registered eligibility sequence defines the analysis cohort "
            "and explicitly accounts for exclusions from the supplied study universe."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Eligibility sequence",
                "role": "overview",
                "claim": (
                    "Stage-specific denominators and removals are read from the "
                    "registered cohort-flow table."
                ),
                "evidence_ids": ["cohort_flow"],
                "metadata": {"planner_product_slots": ["cohort_flow"]},
            },
            {
                "panel_id": "B",
                "title": "Attrition accounting",
                "role": "audit",
                "claim": (
                    "Explicit exclusion categories and percentages are read from "
                    "the registered attrition table."
                ),
                "evidence_ids": ["attrition"],
                "metadata": {"planner_product_slots": ["attrition_audit"]},
            },
        ],
        height_mm=height_mm,
        source_data=[
            "cohort_flow",
            "attrition",
            source_flow_path.name,
            source_attrition_path.name,
        ],
        statistics_note=(
            "Counts, percentages, and stage removals are rendered directly from "
            "the parent cohort-flow and attrition tables; no values are inferred "
            "from the image."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "publication_figure",
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
    if (
        existing_summary.get("deterministic_publication_figure_rescue")
        == "no_parent_outputs"
    ):
        existing_summary.pop("warning", None)
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    existing_summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_cohort_flow_publication_figure_repair",
            "rendering_only": True,
            "deterministic_publication_figure_rescue": (
                "cohort_flow_publication_bundle_from_parent_outputs_v1"
            ),
            "source_step_id": parent_step_id,
            "source_cohort_flow_table": str(flow_path),
            "source_attrition_table": str(attrition_path),
            "source_data_files": [
                source_flow_path.name,
                source_attrition_path.name,
            ],
            "n_flow_stages": int(len(flow_plot)),
            "n_exclusion_categories": int(len(excluded)),
            "figure_files": figure_files,
            "figure_path": "publication_figure.png",
            "figure_contract": "publication_figure.figure_contract.json",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "cohort_flow_publication_bundle_from_parent_outputs_v1"


def _render_missingness_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically rebuild a missingness/measurement audit figure."""

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None
    parent_step_id = current_step_id.removesuffix("_figure")
    candidate_paths: List[Path] = []
    candidate_step_dirs, direct_parent_only = _figure_parent_candidate_step_dirs(
        steps_dir=steps_dir, current_step_id=current_step_id
    )
    for step_dir in candidate_step_dirs:
        text = step_dir.name.lower()
        if not direct_parent_only and not any(
            token in text for token in ("missing", "measurement", "quality")
        ):
            continue
        outputs_dir = step_dir / "outputs"
        if outputs_dir.exists():
            candidate_paths.extend(sorted(outputs_dir.glob("*.csv")))

    def _first_col(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
        for name in names:
            if name in frame.columns:
                return name
        return None

    parent: Optional[tuple[Path, pd.DataFrame]] = None
    parent_score = -1
    for csv_path in candidate_paths:
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        has_label = _first_col(
            frame,
            ("variable", "exposure_or_variable", "concept", "label", "value_col"),
        )
        has_total = _first_col(
            frame,
            ("total_n", "n_total", "denominator", "denominator_n", "n"),
        )
        has_missing = _first_col(
            frame,
            ("missing_n", "n_missing", "value_missing_n", "raw_missing_n"),
        )
        has_unavailable = _first_col(
            frame,
            ("analysis_unavailable_n", "unavailable_n", "invalid_or_missing_n"),
        )
        has_measured = _first_col(
            frame, ("measured_n", "measured_one_n", "n_nonmissing")
        )
        has_pct = _first_col(
            frame,
            (
                "missing_pct",
                "value_missing_pct",
                "raw_missing_pct",
                "analysis_unavailable_pct",
                "measured_pct",
                "measured_one_pct",
                "percentage",
            ),
        )
        if not (
            has_label
            and (
                has_total
                and (has_missing or has_unavailable or has_measured)
                or has_pct
            )
        ):
            continue
        name = csv_path.name.lower()
        score = 0
        if "missingness" in name:
            score += 100
        if "measurement" in name:
            score += 100
        if has_missing:
            score += 20
        if has_unavailable:
            score += 20
        if "metric" in frame.columns and any(
            column in frame.columns for column in ("table_section", "section")
        ):
            score += 10
        if score > parent_score:
            parent = (csv_path, frame)
            parent_score = score
    if parent is None:
        return None

    table_path, frame = parent
    label_col = _first_col(
        frame,
        ("variable", "exposure_or_variable", "concept", "value_col", "label"),
    )
    display_col = _first_col(
        frame,
        (
            "display_label",
            "label",
            "concept",
            "variable",
            "exposure_or_variable",
            "value_col",
        ),
    )
    section_col = _first_col(frame, ("table_section", "section"))
    metric_col = _first_col(frame, ("metric",))
    cohort_col = _first_col(frame, ("cohort", "scope"))
    category_col = _first_col(frame, ("category", "status_category"))
    total_col = _first_col(
        frame,
        ("total_n", "n_total", "denominator", "denominator_n", "n"),
    )
    missing_n_col = _first_col(
        frame,
        ("missing_n", "n_missing", "value_missing_n", "raw_missing_n"),
    )
    unavailable_n_col = _first_col(
        frame,
        ("analysis_unavailable_n", "unavailable_n", "invalid_or_missing_n"),
    )
    measured_n_col = _first_col(frame, ("measured_n", "measured_one_n", "n_nonmissing"))
    missing_pct_col = _first_col(
        frame,
        ("missing_pct", "value_missing_pct", "raw_missing_pct"),
    )
    unavailable_pct_col = _first_col(
        frame,
        ("analysis_unavailable_pct", "unavailable_pct", "invalid_or_missing_pct"),
    )
    measured_pct_col = _first_col(frame, ("measured_pct", "measured_one_pct"))
    if label_col is None:
        return None

    rich_process_table = section_col is not None and metric_col is not None
    source_all = frame.copy()
    if rich_process_table:
        source_all["source_row_index"] = range(len(source_all))
    source = source_all.copy()
    source_row_filter = "all_compatible_rows"
    if rich_process_table:
        section = source[section_col].astype(str).str.lower()
        metric = source[metric_col].astype(str).str.lower()
        raw_rows = section.eq("column_missingness") & metric.eq("raw_missing")
        unavailable_rows = section.eq("column_missingness") & metric.str.contains(
            "analysis_unavailable",
            regex=False,
        )
        if raw_rows.any():
            source = source.loc[raw_rows].copy()
            source_row_filter = "column_missingness:raw_missing"
        elif unavailable_rows.any():
            source = source.loc[unavailable_rows].copy()
            source_row_filter = "column_missingness:analysis_unavailable"
        if missing_n_col is None and "n" in source.columns:
            missing_n_col = "n"
        if missing_pct_col is None and "percentage" in source.columns:
            missing_pct_col = "percentage"
    total = (
        pd.to_numeric(source[total_col], errors="coerce")
        if total_col is not None
        else pd.Series(pd.NA, index=source.index, dtype="Float64")
    )
    missing_n = (
        pd.to_numeric(source[missing_n_col], errors="coerce")
        if missing_n_col is not None
        else pd.Series(pd.NA, index=source.index, dtype="Float64")
    )
    measured_n = (
        pd.to_numeric(source[measured_n_col], errors="coerce")
        if measured_n_col is not None
        else pd.Series(pd.NA, index=source.index, dtype="Float64")
    )
    unavailable_n = (
        pd.to_numeric(source[unavailable_n_col], errors="coerce")
        if unavailable_n_col is not None
        else pd.Series(pd.NA, index=source.index, dtype="Float64")
    )
    if measured_n_col is None and total_col is not None:
        if unavailable_n_col is not None:
            measured_n = total - unavailable_n
        elif missing_n_col is not None:
            measured_n = total - missing_n
    present_but_measured_zero_col = _first_col(
        source,
        ("value_present_but_measured_zero_n", "value_present_but_n_zero_n"),
    )
    if (
        present_but_measured_zero_col is not None
        and total_col is not None
        and missing_n_col is not None
    ):
        present_but_unflagged = pd.to_numeric(
            source[present_but_measured_zero_col],
            errors="coerce",
        ).fillna(0)
        use_value_availability = present_but_unflagged > 0
        measured_n = measured_n.mask(use_value_availability, total - missing_n)
    missing_pct = (
        100.0 * missing_n / total
        if total_col is not None and missing_n_col is not None
        else (
            pd.to_numeric(source[missing_pct_col], errors="coerce")
            if missing_pct_col is not None
            else pd.Series(pd.NA, index=source.index, dtype="Float64")
        )
    )
    unavailable_pct = (
        100.0 * unavailable_n / total
        if total_col is not None and unavailable_n_col is not None
        else (
            pd.to_numeric(source[unavailable_pct_col], errors="coerce")
            if unavailable_pct_col is not None
            else pd.Series(pd.NA, index=source.index, dtype="Float64")
        )
    )
    measured_pct = (
        100.0 * measured_n / total
        if total_col is not None and measured_n.notna().any()
        else (
            pd.to_numeric(source[measured_pct_col], errors="coerce")
            if measured_pct_col is not None
            else 100.0 - missing_pct
        )
    )
    labels = source[label_col].astype(str)
    display_labels = (
        source[display_col].astype(str)
        if display_col is not None
        else labels.map(_publication_label)
    )
    indicator_semantics_col = _first_col(source, ("indicator_semantics",))
    event_status_mask = pd.Series(False, index=source.index)
    if indicator_semantics_col is not None:
        event_status_mask = (
            source[indicator_semantics_col].astype(str).eq("binary_event_presence")
        )
        display_labels = display_labels.mask(
            event_status_mask,
            display_labels.astype(str) + " — analytic event status",
        )
    label_output_col = "variable_name" if rich_process_table else "variable"
    source_data_payload: Dict[str, Any] = {
        label_output_col: labels,
        "display_label": display_labels,
        "missing_pct": missing_pct.astype(float),
        "missing_n": missing_n.astype(float),
        "n_nonmissing": measured_n.astype(float),
        "total_n": total.astype(float),
        "measured_pct": measured_pct.astype(float),
        "measured_n": measured_n.astype(float),
        "source_table": table_path.name,
        "source_transform": "missingness_measurement_summary_v1",
        "source_row_filter": source_row_filter,
    }
    if rich_process_table:
        source_data_payload["source_row_index"] = source["source_row_index"].astype(int)
    else:
        if "concept" in source.columns:
            source_data_payload["concept"] = source["concept"].astype(str)
        else:
            source_data_payload["concept"] = labels
        if "label" in source.columns:
            source_data_payload["label"] = source["label"].astype(str)
    if missing_n_col is not None:
        source_data_payload["value_missing_n"] = pd.to_numeric(
            source[missing_n_col],
            errors="coerce",
        )
        if rich_process_table:
            source_data_payload[missing_n_col] = pd.to_numeric(
                source[missing_n_col],
                errors="coerce",
            )
    if missing_pct_col is not None:
        source_data_payload["value_missing_pct"] = pd.to_numeric(
            source[missing_pct_col],
            errors="coerce",
        )
        if rich_process_table:
            source_data_payload[missing_pct_col] = pd.to_numeric(
                source[missing_pct_col],
                errors="coerce",
            )
    if unavailable_n_col is not None:
        source_data_payload["analysis_unavailable_n"] = unavailable_n
    if unavailable_pct_col is not None or unavailable_n_col is not None:
        source_data_payload["analysis_unavailable_pct"] = unavailable_pct
    if total_col is not None:
        source_data_payload["n_total"] = pd.to_numeric(
            source[total_col], errors="coerce"
        )
        if rich_process_table:
            source_data_payload[total_col] = pd.to_numeric(
                source[total_col],
                errors="coerce",
            )
    if cohort_col is not None:
        cohort_output_col = "cohort_name" if rich_process_table else "cohort"
        source_data_payload[cohort_output_col] = source[cohort_col].astype(str)
    if "measured_one_n" in source.columns:
        source_data_payload["measured_one_n"] = pd.to_numeric(
            source["measured_one_n"],
            errors="coerce",
        )
    if "measured_one_pct" in source.columns:
        source_data_payload["measured_one_pct"] = pd.to_numeric(
            source["measured_one_pct"],
            errors="coerce",
        )
    if indicator_semantics_col is not None:
        source_data_payload["indicator_semantics"] = source[
            indicator_semantics_col
        ].astype(str)
    if "raw_indicator_one_n" in source.columns:
        source_data_payload["raw_indicator_one_n"] = pd.to_numeric(
            source["raw_indicator_one_n"],
            errors="coerce",
        )
    if "event_count_column" in source.columns:
        source_data_payload["event_count_column"] = (
            source["event_count_column"].fillna("").astype(str)
        )
    source_data = pd.DataFrame(source_data_payload).dropna(
        subset=["missing_pct", "measured_pct"],
        how="all",
    )
    if source_data.empty:
        return None
    if rich_process_table:
        variable_order = (
            source_data.groupby(label_output_col, sort=False)["missing_pct"]
            .max()
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        source_data = source_data[
            source_data[label_output_col].isin(variable_order)
        ].copy()
        source_data[label_output_col] = pd.Categorical(
            source_data[label_output_col],
            categories=list(variable_order),
            ordered=True,
        )
        source_data = source_data.sort_values(
            [label_output_col, "cohort_name"],
        )
        source_data[label_output_col] = source_data[label_output_col].astype(str)
    else:
        source_data = source_data.sort_values("missing_pct", ascending=False).head(12)

    has_event_status_rows = bool(
        "indicator_semantics" in source_data.columns
        and source_data["indicator_semantics"]
        .astype(str)
        .eq("binary_event_presence")
        .any()
    )
    availability_title = (
        "Analytic availability" if has_event_status_rows else "Measurement availability"
    )
    availability_axis_label = (
        "Value / event status available (%)"
        if has_event_status_rows
        else "Available observations (%)"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    availability_source_path = out_dir / "missingness_measurement_panel_source_data.csv"
    source_data.to_csv(availability_source_path, index=False)

    status_source_data = pd.DataFrame()
    status_source_path = out_dir / "missingness_status_matrix_source_data.csv"
    if (
        rich_process_table
        and label_col is not None
        and category_col is not None
        and cohort_col is not None
        and total_col is not None
        and {"n", "percentage"}.issubset(source_all.columns)
    ):
        status_mask = source_all[section_col].astype(str).str.lower().eq(
            "source_status"
        ) & source_all[metric_col].astype(str).str.lower().isin(
            ("mutually_exclusive_source_status", "source_status")
        )
        status_rows = source_all.loc[status_mask].copy()
        if not status_rows.empty:
            status_source_data = pd.DataFrame(
                {
                    "variable_name": status_rows[label_col].astype(str),
                    "display_label": status_rows[label_col]
                    .astype(str)
                    .map(_publication_label),
                    "cohort_name": status_rows[cohort_col].astype(str),
                    "status_category": status_rows[category_col].astype(str),
                    "n": pd.to_numeric(status_rows["n"], errors="coerce"),
                    "denominator": pd.to_numeric(
                        status_rows[total_col], errors="coerce"
                    ),
                    "percentage": pd.to_numeric(
                        status_rows["percentage"], errors="coerce"
                    ),
                    "source_row_index": status_rows["source_row_index"].astype(int),
                    "source_table": table_path.name,
                    "source_transform": "source_status_matrix_v1",
                }
            ).dropna(subset=["percentage"])
            if not status_source_data.empty:
                status_source_data.to_csv(status_source_path, index=False)

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
    rich_matrix_rendered = not status_source_data.empty
    if rich_matrix_rendered:
        fig = plt.figure(
            figsize=(183 / 25.4, 126 / 25.4),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.18, 0.82],
            left=0.16,
            right=0.98,
            top=0.88,
            bottom=0.25,
            wspace=0.62,
        )
        ax_missing = fig.add_subplot(grid[0, 0])
        ax_measured = fig.add_subplot(grid[0, 1])

        status_plot = status_source_data.copy()
        multi_cohort = status_plot["cohort_name"].nunique() > 1
        status_plot["row_label"] = status_plot["display_label"].astype(str)
        if multi_cohort:
            status_plot["row_label"] = (
                status_plot["row_label"]
                + "\n"
                + status_plot["cohort_name"].map(_publication_label)
            )
        status_rows = list(dict.fromkeys(status_plot["row_label"].tolist()))
        status_columns = list(
            dict.fromkeys(status_plot["status_category"].astype(str).tolist())
        )
        status_matrix = status_plot.pivot_table(
            index="row_label",
            columns="status_category",
            values="percentage",
            aggfunc="first",
        ).reindex(index=status_rows, columns=status_columns)

        def _status_display(value: str) -> str:
            text = value.lower().replace("_", " ")
            if "valid observed" in text:
                return "Observed"
            if "no recorded source" in text:
                return "No source"
            if "summary missing" in text:
                return "Source present;\nsummary missing"
            if "contradictory" in text or "invalid" in text:
                return "Contradictory /\ninvalid"
            return _short_figure_label(value, limit=24)

        status_values = status_matrix.to_numpy(dtype=float)
        ax_missing.imshow(
            status_values,
            aspect="auto",
            vmin=0,
            vmax=100,
            cmap="Blues",
        )
        ax_missing.set_xticks(range(len(status_columns)))
        ax_missing.set_xticklabels(
            [_status_display(value) for value in status_columns],
            rotation=28,
            ha="right",
        )
        ax_missing.set_yticks(range(len(status_rows)))
        ax_missing.set_yticklabels(status_rows)
        for row_idx in range(len(status_rows)):
            for col_idx in range(len(status_columns)):
                value = status_values[row_idx, col_idx]
                if pd.isna(value):
                    continue
                ax_missing.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value >= 55 else "black",
                )
        ax_missing.set_xlabel("Source-status share of cohort (%)")
        ax_missing.set_title("Measurement-source status", loc="left", pad=4)
        add_panel_label(ax_missing, "A", x=0.0, y=1.08)

        availability_plot = source_data.copy()
        variable_rows = list(
            dict.fromkeys(availability_plot["variable_name"].astype(str).tolist())
        )
        cohort_columns = list(
            dict.fromkeys(availability_plot["cohort_name"].astype(str).tolist())
        )
        availability_matrix = availability_plot.pivot_table(
            index="variable_name",
            columns="cohort_name",
            values="measured_pct",
            aggfunc="first",
        ).reindex(index=variable_rows, columns=cohort_columns)
        availability_values = availability_matrix.to_numpy(dtype=float)

        def _measurement_display_map(values: Sequence[str]) -> Dict[str, str]:
            raw_values = list(dict.fromkeys(str(value) for value in values))
            display = {value: _publication_label(value) for value in raw_values}
            counts: Dict[str, int] = {}
            for label in display.values():
                counts[label] = counts.get(label, 0) + 1
            suffix_labels = {
                "_first": "First value",
                "_max": "Maximum",
                "_min": "Minimum",
                "_mean": "Mean",
                "_n": "Observation count",
                "_measured": "Measured flag",
            }
            always_expand = ("_first", "_n", "_measured")
            for value in raw_values:
                lower_value = value.lower()
                if counts.get(display[value], 0) <= 1 and not lower_value.endswith(
                    always_expand
                ):
                    continue
                for suffix, suffix_label in suffix_labels.items():
                    if lower_value.endswith(suffix):
                        base = value[: -len(suffix)]
                        display[value] = f"{_publication_label(base)} — {suffix_label}"
                        break
                else:
                    display[value] = value.replace("_", " ").title()
            return display

        variable_display = _measurement_display_map(variable_rows)
        ax_measured.imshow(
            availability_values,
            aspect="auto",
            vmin=0,
            vmax=100,
            cmap="Blues",
        )
        ax_measured.set_xticks(range(len(cohort_columns)))
        ax_measured.set_xticklabels(
            [_publication_label(value) for value in cohort_columns],
            rotation=28,
            ha="right",
        )
        ax_measured.set_yticks(range(len(variable_rows)))
        ax_measured.set_yticklabels(
            [
                _short_figure_label(variable_display[value], limit=32).replace(
                    " — ", "\n"
                )
                for value in variable_rows
            ]
        )
        for row_idx in range(len(variable_rows)):
            for col_idx in range(len(cohort_columns)):
                value = availability_values[row_idx, col_idx]
                if pd.isna(value):
                    continue
                ax_measured.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value >= 55 else "black",
                )
        ax_measured.set_xlabel(availability_axis_label)
        ax_measured.set_title(availability_title, loc="left", pad=4)
        add_panel_label(ax_measured, "B", x=0.0, y=1.08)
    else:
        plot_df = source_data.reset_index(drop=True)
        y = list(range(len(plot_df)))
        labels = [
            _short_figure_label(label, limit=30)
            for label in plot_df["display_label"].astype(str)
        ]
        fig = plt.figure(
            figsize=(183 / 25.4, 104 / 25.4),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.05, 0.95],
            left=0.28,
            right=0.98,
            top=0.88,
            bottom=0.16,
            wspace=0.50,
        )
        ax_missing = fig.add_subplot(grid[0, 0])
        ax_measured = fig.add_subplot(grid[0, 1], sharey=ax_missing)

        missing = pd.to_numeric(plot_df["missing_pct"], errors="coerce").fillna(0)
        measured = pd.to_numeric(plot_df["measured_pct"], errors="coerce").fillna(0)
        ax_missing.barh(
            y,
            missing.clip(0, 100),
            color=palette.get("red", "#B2182B"),
            height=0.56,
        )
        ax_missing.axvline(
            20,
            color=palette.get("neutral", "#8F8F8F"),
            linestyle="--",
            linewidth=0.8,
        )
        ax_missing.set_yticks(y)
        ax_missing.set_yticklabels(labels)
        ax_missing.invert_yaxis()
        ax_missing.set_xlabel("Missing values (%)")
        ax_missing.set_title("Value missingness", loc="left", pad=4)
        ax_missing.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
        )
        add_panel_label(ax_missing, "A", x=0.0, y=1.08)

        ax_measured.barh(
            y,
            measured.clip(0, 100),
            color=palette.get("blue", "#0F4D92"),
            height=0.56,
        )
        ax_measured.set_xlim(0, 100)
        ax_measured.set_xlabel(availability_axis_label)
        ax_measured.set_title(availability_title, loc="left", pad=4)
        ax_measured.tick_params(axis="y", labelleft=False)
        ax_measured.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
        )
        add_panel_label(ax_measured, "B", x=0.0, y=1.08)

    parent_evidence_id = table_path.stem
    availability_source_id = availability_source_path.stem
    status_source_id = status_source_path.stem
    panel_a_title = (
        "Measurement-source status" if rich_matrix_rendered else "Value missingness"
    )
    panel_a_claim = (
        "Mutually exclusive source-status percentages are shown for each audited "
        "measurement summary and cohort."
        if rich_matrix_rendered
        else "Missing percentages are recomputed from missing counts and denominators "
        "in the parent audit table."
    )
    panel_a_evidence = [
        parent_evidence_id,
        status_source_id if rich_matrix_rendered else availability_source_id,
    ]
    # Figure contracts name concrete local CSV files; upstream evidence ids stay
    # in the panel bindings above. A stem-only list leaves the strict source-data
    # validator with no verifiable file and must not pass.
    contract_source_data = [availability_source_path.name]
    if rich_matrix_rendered:
        contract_source_data.append(status_source_path.name)

    contract = make_figure_contract(
        figure_id="missingness_measurement_panel",
        core_claim=(
            "First-24h variable availability is shown directly from the "
            "registered missingness and measurement audit table."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": panel_a_title,
                "role": "data_quality",
                "chart_type": (
                    "missingness_matrix" if rich_matrix_rendered else "missingness_bar"
                ),
                "claim": panel_a_claim,
                "evidence_ids": panel_a_evidence,
            },
            {
                "panel_id": "B",
                "title": availability_title,
                "role": "data_quality",
                "chart_type": "availability_panel",
                "claim": (
                    "Value availability percentages are recomputed from counts and "
                    "denominators in the parent audit table. Registered binary-event "
                    "rows report analytic status availability under the locked "
                    "absence-as-negative convention, not literal measurement capture."
                    if has_event_status_rows
                    else "Measured or available percentages are recomputed from "
                    "measurement counts and denominators in the parent audit table."
                ),
                "evidence_ids": [parent_evidence_id, availability_source_id],
            },
        ],
        source_data=contract_source_data,
        statistics_note=(
            "Generated deterministically from the registered parent-step "
            "missingness/measurement audit; percentages are count-derived."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "missingness_measurement_panel",
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
            "method": "deterministic_missingness_publication_figure_repair",
            "rendering_only": True,
            "source_step_id": parent_step_id,
            "source_missingness_table": str(table_path),
            "source_row_filter": source_row_filter,
            "source_data_csv": str(availability_source_path),
            "source_data_files": [
                availability_source_path.name,
                *([status_source_path.name] if rich_matrix_rendered else []),
            ],
            "n_variables_plotted": int(source_data[label_output_col].nunique()),
            "n_availability_rows": int(len(source_data)),
            "n_source_status_rows": int(len(status_source_data)),
            "n_binary_event_status_rows": int(
                source_data.get("indicator_semantics", pd.Series(dtype=str))
                .astype(str)
                .eq("binary_event_presence")
                .sum()
            ),
            "rich_missingness_matrix_rendered": rich_matrix_rendered,
            "figure_files": [
                path.name for key, path in outputs.items() if key != "contract"
            ],
            "figure_path": "missingness_measurement_panel.png",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "missingness_publication_bundle_from_parent_outputs_v1"


def _render_phenotype_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically rebuild a phenotyping figure from agent-produced,
    source-backed standardized clustering products.

    Reads ``outcome_by_cluster.csv`` (the descriptive outcome contrast, which
    carries ``ci_low``/``ci_high`` -- validator-checked value columns) as the
    primary traceable source, falling back to ``cluster_sizes.csv``. Renders a
    two-panel figure (cluster sizes + descriptive outcome-by-cluster with CIs) and
    emits a validator-conformant ``*_source_data.csv`` traced positionally via
    ``source_row_index`` (like the prediction/association renderers). Returns
    ``None`` when no cluster table with >= 2 clusters is found, so a run that never
    produced a partition falls through cleanly rather than emitting an empty
    figure. Descriptive by construction -- no OR/HR is drawn or claimed.
    """

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None
    parent_step_id = current_step_id.removesuffix("_figure")
    candidate_paths: List[Path] = []
    candidate_step_dirs, direct_parent_only = _figure_parent_candidate_step_dirs(
        steps_dir=steps_dir, current_step_id=current_step_id
    )
    for step_dir in candidate_step_dirs:
        text = step_dir.name.lower()
        if not direct_parent_only and not any(
            token in text
            for token in ("cluster", "phenotype", "subphenotype", "trajectory")
        ):
            continue
        outputs_dir = step_dir / "outputs"
        if outputs_dir.exists():
            candidate_paths.extend(sorted(outputs_dir.glob("*.csv")))

    def _first_col(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
        for name in names:
            if name in frame.columns:
                return name
        return None

    # Prefer the outcome-by-cluster table (traceable ci_low/ci_high); else sizes.
    outcome: Optional[tuple[Path, pd.DataFrame]] = None
    sizes: Optional[tuple[Path, pd.DataFrame]] = None
    for csv_path in candidate_paths:
        name = csv_path.name.lower()
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        cluster_col = _first_col(frame, ("cluster", "cluster_id", "phenotype", "label"))
        if cluster_col is None:
            continue
        if outcome is None and "outcome_by_cluster" in name:
            outcome = (csv_path, frame)
        if sizes is None and (
            "cluster_sizes" in name or "size" in frame.columns or "n" in frame.columns
        ):
            sizes = (csv_path, frame)
    primary = outcome or sizes
    if primary is None:
        return None
    table_path, frame = primary
    cluster_col = _first_col(frame, ("cluster", "cluster_id", "phenotype", "label"))
    if cluster_col is None or frame[cluster_col].nunique() < 2:
        return None

    # Positional trace: keep the ORIGINAL row order so source_row_index maps 1:1
    # into the upstream table the validator re-reads.
    plot_df = frame.reset_index(drop=True)
    plot_df.insert(0, "source_row_index", plot_df.index.astype(int))
    n_col = _first_col(plot_df, ("n", "n_stays", "cluster_size", "size", "count"))
    rate_col = _first_col(
        plot_df, ("mortality_rate", "outcome_rate", "event_rate", "rate")
    )
    ci_low_col = _first_col(plot_df, ("ci_low", "ci_lower", "lower"))
    ci_high_col = _first_col(plot_df, ("ci_high", "ci_upper", "upper"))

    clusters = plot_df[cluster_col].astype(str)
    source_payload: Dict[str, Any] = {
        "source_row_index": plot_df["source_row_index"].astype(int),
        # The cluster label is carried under a NON-key column name so the validator
        # traces POSITIONALLY (source_row_index) and value-checks ci_low/ci_high,
        # rather than joining on a key column that is either unshared or non-unique.
        "cluster_label": clusters,
        "source_table": table_path.name,
        "source_transform": "phenotype_cluster_outcome_summary_v1",
    }
    if n_col is not None:
        source_payload["n"] = pd.to_numeric(plot_df[n_col], errors="coerce")
    if rate_col is not None:
        source_payload["mortality_rate"] = pd.to_numeric(
            plot_df[rate_col], errors="coerce"
        )
    if ci_low_col is not None:
        source_payload["ci_low"] = pd.to_numeric(plot_df[ci_low_col], errors="coerce")
    if ci_high_col is not None:
        source_payload["ci_high"] = pd.to_numeric(plot_df[ci_high_col], errors="coerce")
    source_data = pd.DataFrame(source_payload)
    if source_data.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(out_dir / "phenotype_cluster_panel_source_data.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from easyicu.research_agent.figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    labels = [f"C{c}" for c in clusters.tolist()]
    x = list(range(len(labels)))
    has_outcome = rate_col is not None
    ncols = 2 if (n_col is not None and has_outcome) else 1
    fig = plt.figure(figsize=(183 / 25.4, 92 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1, ncols, left=0.12, right=0.97, top=0.88, bottom=0.18, wspace=0.42
    )
    col = 0
    if n_col is not None:
        ax_size = fig.add_subplot(grid[0, col])
        sizes_v = pd.to_numeric(plot_df[n_col], errors="coerce").fillna(0).to_numpy()
        ax_size.bar(x, sizes_v, color=palette.get("blue", "#0F4D92"), width=0.62)
        ax_size.set_xticks(x, labels, fontsize=6.2)
        ax_size.set_ylabel("Cluster size (n)")
        ax_size.set_title("Cluster sizes", loc="left", pad=4)
        add_panel_label(ax_size, "A", x=-0.12)
        col += 1
    if has_outcome:
        ax_out = fig.add_subplot(grid[0, col])
        rate = pd.to_numeric(plot_df[rate_col], errors="coerce").fillna(0).to_numpy()
        # Scale a 0-1 proportion to percent for display; leave an already-percent
        # column untouched.
        scale = 100.0 if float(np.nanmax(rate)) <= 1.0 else 1.0
        rate_pct = rate * scale
        yerr = None
        if ci_low_col is not None and ci_high_col is not None:
            lo = (
                pd.to_numeric(plot_df[ci_low_col], errors="coerce").fillna(0).to_numpy()
                * scale
            )
            hi = (
                pd.to_numeric(plot_df[ci_high_col], errors="coerce")
                .fillna(0)
                .to_numpy()
                * scale
            )
            yerr = np.vstack(
                [np.clip(rate_pct - lo, 0, None), np.clip(hi - rate_pct, 0, None)]
            )
        ax_out.bar(
            x,
            rate_pct,
            yerr=yerr,
            color=palette.get("red", "#B2182B"),
            width=0.62,
            capsize=3,
        )
        ax_out.set_xticks(x, labels, fontsize=6.2)
        ax_out.set_ylabel("Outcome rate (%)")
        ax_out.set_title("Outcome by cluster (descriptive)", loc="left", pad=4)
        add_panel_label(ax_out, "B" if n_col is not None else "A", x=-0.12)

    contract = make_figure_contract(
        figure_id="phenotype_cluster_panel",
        core_claim=(
            "Discovered phenotype clusters are shown by size and a DESCRIPTIVE "
            "outcome-by-cluster comparison, rendered from the agent-produced, "
            "source-backed clustering products (no causal claim)."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Cluster sizes",
                "role": "phenotype_structure",
                "claim": "Cluster sizes come directly from the declared parent table.",
                "evidence_ids": ["phenotype_cluster_panel_source_data.csv"],
            },
            {
                "panel_id": "B",
                "title": "Outcome by cluster",
                "role": "phenotype_outcome",
                "claim": (
                    "Outcome rates and confidence intervals are copied from the "
                    "agent's descriptive outcome-by-cluster product; comparison is "
                    "descriptive, explicitly not causal."
                ),
                "evidence_ids": ["phenotype_cluster_panel_source_data.csv"],
            },
        ],
        source_data=["phenotype_cluster_panel_source_data.csv"],
        statistics_note=(
            "Rendered deterministically from agent-produced standardized "
            "clustering products; outcome-by-cluster is descriptive (no adjusted "
            "effect)."
        ),
    )
    outputs = save_publication_figure(
        fig, out_dir / "phenotype_cluster_panel", contract=contract, dpi=300
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
            "method": "deterministic_phenotype_publication_figure_repair",
            "rendering_only": True,
            "source_step_id": parent_step_id,
            "source_cluster_table": str(table_path),
            "source_data_csv": str(out_dir / "phenotype_cluster_panel_source_data.csv"),
            "n_clusters_plotted": int(frame[cluster_col].nunique()),
            "figure_files": [
                path.name for key, path in outputs.items() if key != "contract"
            ],
            "figure_path": "phenotype_cluster_panel.png",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "phenotype_publication_bundle_from_parent_outputs_v1"


# Descriptive "table one" / baseline-characteristics signature. Historically the
# deterministic figure path deliberately EXCLUDED descriptive figures so an empty
# renderer would not claim an association forest. This renderer reverses that only
# for a table that is UNAMBIGUOUSLY a table-one summary (variable + row-type +
# per-group median/percentage cells) and returns None otherwise, so a result table
# still falls through to its own renderer / the coder.
_TABLE_ONE_ROWTYPE_COLS = ("row_type", "summary_type", "variable_class")
_TABLE_ONE_VALUE_TOKENS = ("median", "mean", "percentage", "count", "q25", "q75")
_RESULT_TABLE_COLS = (
    "odds_ratio",
    "hazard_ratio",
    "risk_ratio",
    "estimate",
    "point_estimate",
    "coef",
    "auroc",
)


def _render_descriptive_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically rebuild a descriptive baseline / table-one figure.

    Fires ONLY for a genuine table-one summary (a ``variable`` key column, a
    row-type column, and per-group median/percentage cells) and returns ``None``
    for anything else -- an association/result table (has an odds_ratio/estimate
    column) is left to its own renderer, and a run without a descriptive table
    falls through cleanly. Renders continuous (median) and categorical (percent)
    baseline summaries and emits a validator-conformant ``*_source_data.csv``
    traced positionally via ``source_row_index`` into the parent table.
    """

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None
    parent_step_id = current_step_id.removesuffix("_figure")
    candidate_paths: List[Path] = []
    candidate_step_dirs, direct_parent_only = _figure_parent_candidate_step_dirs(
        steps_dir=steps_dir, current_step_id=current_step_id
    )
    for step_dir in candidate_step_dirs:
        text = step_dir.name.lower()
        if not direct_parent_only and not any(
            token in text
            for token in (
                "baseline",
                "table_one",
                "table1",
                "descriptive",
                "characteristic",
            )
        ):
            continue
        outputs_dir = step_dir / "outputs"
        if outputs_dir.exists():
            candidate_paths.extend(sorted(outputs_dir.glob("*.csv")))

    def _first_col(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
        for name in names:
            if name in frame.columns:
                return name
        return None

    def _is_table_one(frame: pd.DataFrame) -> bool:
        cols = {str(c).lower() for c in frame.columns}
        if "variable" not in cols:
            return False
        # A result/effect table is NOT a table one.
        if any(c in cols for c in _RESULT_TABLE_COLS):
            return False
        if not any(c in cols for c in _TABLE_ONE_ROWTYPE_COLS):
            return False
        return any(any(tok in c for c in cols) for tok in _TABLE_ONE_VALUE_TOKENS)

    parent: Optional[tuple[Path, pd.DataFrame]] = None
    for csv_path in candidate_paths:
        name = csv_path.name.lower()
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        if not _is_table_one(frame):
            continue
        parent = (csv_path, frame)
        if "table_one" in name or "baseline" in name or "characteristic" in name:
            break
    if parent is None:
        return None

    table_path, frame = parent
    frame = frame.reset_index(drop=True)
    row_type_col = _first_col(frame, _TABLE_ONE_ROWTYPE_COLS)
    label_col = _first_col(frame, ("label", "variable"))
    category_col = _first_col(frame, ("category",))
    median_col = _first_col(frame, ("overall_median", "median"))
    pct_col = _first_col(frame, ("overall_percentage", "percentage"))
    if label_col is None or (median_col is None and pct_col is None):
        return None

    # Positional trace: keep original row order; source_row_index maps 1:1 into the
    # upstream table the validator re-reads.
    rows: List[Dict[str, Any]] = []
    for idx, row in frame.iterrows():
        rtype = str(row.get(row_type_col, "") if row_type_col else "").lower()
        label = str(row.get(label_col, "")).strip()
        cat = str(row.get(category_col, "")).strip() if category_col else ""
        median_v = (
            pd.to_numeric(pd.Series([row.get(median_col)]), errors="coerce").iloc[0]
            if median_col
            else float("nan")
        )
        pct_v = (
            pd.to_numeric(pd.Series([row.get(pct_col)]), errors="coerce").iloc[0]
            if pct_col
            else float("nan")
        )
        is_cont = ("continuous" in rtype) or (pd.notna(median_v) and pd.isna(pct_v))
        display = label if not cat or cat.lower() == "nan" else f"{label} ({cat})"
        rows.append(
            {
                # ``variable``/``category`` are _KEY_COLUMNS and non-unique in a
                # table one (e.g. sex -> Female + Male); carrying them under NON-key
                # names forces the validator to trace POSITIONALLY via
                # source_row_index rather than an ambiguous named-key join that
                # false-flags disagreement.
                "source_row_index": int(idx),
                "variable_name": str(row.get("variable", label)),
                "row_category": cat,
                "display_label": display,
                "is_continuous": bool(is_cont),
                "overall_median": (
                    float(median_v) if pd.notna(median_v) else float("nan")
                ),
                "overall_percentage": float(pct_v) if pd.notna(pct_v) else float("nan"),
                "source_table": table_path.name,
                "source_transform": "table_one_baseline_summary_v1",
            }
        )
    source_data = pd.DataFrame(rows)
    cont_rows = source_data[
        source_data["is_continuous"] & source_data["overall_median"].notna()
    ].head(12)
    cat_rows = source_data[
        (~source_data["is_continuous"]) & source_data["overall_percentage"].notna()
    ].head(12)
    if cont_rows.empty and cat_rows.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    keep = pd.concat([cont_rows, cat_rows]).sort_values("source_row_index")
    keep.to_csv(out_dir / "baseline_table_one_source_data.csv", index=False)

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
    ncols = int(not cont_rows.empty) + int(not cat_rows.empty)
    fig = plt.figure(figsize=(183 / 25.4, 104 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1, max(1, ncols), left=0.30, right=0.97, top=0.88, bottom=0.14, wspace=0.55
    )
    col = 0
    if not cont_rows.empty:
        ax = fig.add_subplot(grid[0, col])
        y = list(range(len(cont_rows)))
        ax.barh(
            y,
            cont_rows["overall_median"].to_numpy(),
            color=palette.get("blue", "#0F4D92"),
            height=0.6,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_short_figure_label(v, limit=28) for v in cont_rows["display_label"]]
        )
        ax.invert_yaxis()
        ax.set_xlabel("Median (overall)")
        ax.set_title("Continuous characteristics", loc="left", pad=4)
        add_panel_label(ax, "A", x=0.0, y=1.06)
        col += 1
    if not cat_rows.empty:
        ax = fig.add_subplot(grid[0, col])
        y = list(range(len(cat_rows)))
        ax.barh(
            y,
            cat_rows["overall_percentage"].clip(0, 100).to_numpy(),
            color=palette.get("green", "#2E7D32"),
            height=0.6,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_short_figure_label(v, limit=28) for v in cat_rows["display_label"]]
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage (overall)")
        ax.set_title("Categorical characteristics", loc="left", pad=4)
        add_panel_label(ax, "B" if not cont_rows.empty else "A", x=0.0, y=1.06)

    contract = make_figure_contract(
        figure_id="baseline_table_one",
        core_claim=(
            "Baseline cohort characteristics are shown directly from the "
            "registered descriptive table-one summary."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": "Baseline characteristics",
                "role": "descriptive",
                "claim": (
                    "Median and percentage summaries are copied from the parent "
                    "table-one; no effect is estimated."
                ),
                "evidence_ids": ["table_one"],
            }
        ],
        source_data=["table_one"],
        statistics_note=(
            "Generated deterministically from the registered parent-step "
            "table-one; descriptive only (no adjusted effect)."
        ),
    )
    outputs = save_publication_figure(
        fig, out_dir / "baseline_table_one", contract=contract, dpi=300
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
            "method": "deterministic_descriptive_publication_figure_repair",
            "rendering_only": True,
            "source_step_id": parent_step_id,
            "source_table_one": str(table_path),
            "source_data_csv": str(out_dir / "baseline_table_one_source_data.csv"),
            "n_rows_plotted": int(len(keep)),
            "figure_files": [
                path.name for key, path in outputs.items() if key != "contract"
            ],
            "figure_path": "baseline_table_one.png",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "descriptive_publication_bundle_from_parent_outputs_v1"


def _iter_prior_output_tables(
    *,
    run_dir: Path,
    current_step_id: str,
) -> Sequence[Tuple[Path, pd.DataFrame]]:
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return []
    tables: List[Tuple[Path, pd.DataFrame]] = []
    for step_dir in sorted(steps_dir.iterdir()):
        if not step_dir.is_dir() or step_dir.name == current_step_id:
            continue
        outputs_dir = step_dir / "outputs"
        if not outputs_dir.exists():
            continue
        for csv_path in sorted(outputs_dir.glob("*.csv")):
            try:
                tables.append((csv_path, pd.read_csv(csv_path)))
            except Exception:
                continue
    return tables


def _find_column(
    frame: pd.DataFrame,
    *,
    exact: Sequence[str] = (),
    suffixes: Sequence[str] = (),
    contains: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Optional[str]:
    excluded = {item.lower() for item in exclude}
    lower_to_orig = {str(c).lower(): c for c in frame.columns}
    for candidate in exact:
        key = candidate.lower()
        if key in lower_to_orig and key not in excluded:
            return str(lower_to_orig[key])
    for column in frame.columns:
        key = str(column).lower()
        if key in excluded:
            continue
        if suffixes and any(key.endswith(suffix.lower()) for suffix in suffixes):
            return str(column)
        if contains and any(token.lower() in key for token in contains):
            return str(column)
    return None


def _as_percent(row: pd.Series, column: Optional[str]) -> Optional[float]:
    if not column:
        return None
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def _event_count_column(
    frame: pd.DataFrame, denominator_col: Optional[str]
) -> Optional[str]:
    excluded = {
        "n",
        "n_total",
        "total_n",
        "denominator",
        "n_denominator",
        str(denominator_col or "").lower(),
    }
    column = _find_column(
        frame,
        exact=("event_n", "events", "outcome_n", "n_positive"),
        suffixes=("_event_n", "_events", "_n"),
        exclude=tuple(excluded),
    )
    return column


def _label_column(frame: pd.DataFrame) -> Optional[str]:
    return _find_column(
        frame,
        exact=(
            "label",
            "group_label",
            "exposure_label",
            "stratum_label",
            "category_label",
        ),
        suffixes=("_label",),
    )


_BINARY_GROUP_EXCLUDED_TOKENS = (
    "n",
    "count",
    "event",
    "events",
    "death",
    "mort",
    "risk",
    "rate",
    "prevalence",
    "incidence",
    "ci",
    "lower",
    "upper",
    "pct",
    "percent",
    "source",
    "row",
)


def _binary_group_column(frame: pd.DataFrame) -> Optional[str]:
    binary_tokens = {"0", "1", "0.0", "1.0", "false", "true", "no", "yes"}
    for column in frame.columns:
        key = str(column).lower()
        if any(token in key for token in _BINARY_GROUP_EXCLUDED_TOKENS):
            continue
        values = [
            str(value).strip().lower()
            for value in frame[column].dropna().tolist()
            if str(value).strip()
        ]
        if not values:
            continue
        binary_values = [value for value in values if value in binary_tokens]
        allowed_extra_values = [
            value
            for value in values
            if value not in binary_tokens and "risk_difference" not in value
        ]
        if len(set(binary_values)) >= 2 and not allowed_extra_values:
            return str(column)
    return None


def _binary_group_label(column: str, value: Any) -> str:
    normalized = str(value).strip().lower()
    base = _publication_label(column)
    if normalized in {"1", "1.0", "true", "yes"}:
        return f"{base} positive"
    if normalized in {"0", "0.0", "false", "no"}:
        return f"{base} negative"
    return _publication_label(value)


def _is_risk_difference_row(row: pd.Series, *values: Any) -> bool:
    haystack = " ".join([str(value or "") for value in values])
    haystack = (
        f"{haystack} {' '.join(str(value or '') for value in row.to_dict().values())}"
    )
    return (
        "risk_difference" in haystack.lower() or "risk difference" in haystack.lower()
    )


def _context_axis_label(metric: Any, group: Any) -> str:
    metric_text = str(metric or "").strip()
    group_text = str(group or "").strip()
    if metric_text.lower() == "exposure prevalence":
        suffix = " prevalence"
        if group_text.lower().endswith(suffix) and len(group_text) > len(suffix):
            return (
                f"{_short_figure_label(group_text[: -len(suffix)].strip(), limit=24)}\n"
                "prevalence"
            )
        return _short_figure_label(group_text or metric_text, limit=28)
    if metric_text and group_text and metric_text.lower() not in group_text.lower():
        return (
            f"{_short_figure_label(group_text, limit=24)}\n"
            f"{_short_figure_label(metric_text, limit=24)}"
        )
    return _short_figure_label(group_text or metric_text or "Context", limit=28)


def _association_descriptive_context(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    primary_exposure: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect source-backed prevalence or absolute-risk rows for association figures.

    The helper is deliberately keyed to generic column semantics
    (prevalence/risk/event-rate), not to a benchmark variable name.
    """

    plot_rows: List[Dict[str, Any]] = []
    source_files: List[str] = []
    has_prevalence = False
    has_outcome_risk = False

    def _canonical_exposure_token(value: Any) -> str:
        token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
        suffixes = (
            "_log1p_active",
            "_log1p",
            "_active",
            "_maximum",
            "_minimum",
            "_median",
            "_mean",
            "_first",
            "_last",
            "_value",
            "_max",
            "_min",
        )
        changed = True
        while changed and token:
            changed = False
            for suffix in suffixes:
                if token.endswith(suffix) and len(token) > len(suffix):
                    token = token[: -len(suffix)]
                    changed = True
                    break
        return token

    primary_token = _canonical_exposure_token(primary_exposure)

    for table_path, frame in _iter_prior_output_tables(
        run_dir=run_dir,
        current_step_id=current_step_id,
    ):
        if frame.empty:
            continue
        frame = frame.copy()
        if primary_token:
            exposure_col = _find_column(
                frame,
                exact=(
                    "exposure",
                    "exposure_source",
                    "source_variable",
                    "concept",
                    "variable",
                ),
            )
            if exposure_col:
                matches = (
                    frame[exposure_col].map(_canonical_exposure_token).eq(primary_token)
                )
                if matches.any():
                    frame = frame.loc[matches].copy()
                elif frame[exposure_col].nunique(dropna=True) > 1:
                    # This is explicitly a multi-exposure context table but it
                    # contains no row for the locked primary exposure.
                    continue
        prevalence_pct_col = _find_column(
            frame,
            exact=("prevalence_pct", "incidence_pct"),
        )
        prevalence_prop_col = _find_column(frame, exact=("prevalence", "incidence"))
        if not has_prevalence and (prevalence_pct_col or prevalence_prop_col):
            denominator_col = _find_column(
                frame,
                exact=("n_denominator", "denominator", "n_total", "total_n", "n"),
            )
            event_col = _find_column(frame, exact=("n_positive", "event_n", "events"))
            label_col = _find_column(
                frame,
                exact=("label", "exposure", "variable", "concept"),
            )
            source_rows: List[Dict[str, Any]] = []
            for idx, row in frame.iterrows():
                estimate = _as_percent(row, prevalence_pct_col or prevalence_prop_col)
                if estimate is None:
                    continue
                base_label = row.get(label_col) if label_col else "Exposure"
                display_label = f"{_publication_label(base_label)} prevalence"
                record = row.to_dict()
                record.update(
                    {
                        "plot_metric": "Exposure prevalence",
                        "plot_group_label": display_label,
                        "plot_estimate_pct": estimate,
                        "plot_ci_low_pct": _as_percent(
                            row,
                            _find_column(frame, exact=("ci_low_pct", "lower_pct"))
                            or _find_column(frame, exact=("ci_low", "lower")),
                        ),
                        "plot_ci_high_pct": _as_percent(
                            row,
                            _find_column(frame, exact=("ci_high_pct", "upper_pct"))
                            or _find_column(frame, exact=("ci_high", "upper")),
                        ),
                        "plot_denominator": (
                            row.get(denominator_col) if denominator_col else None
                        ),
                        "plot_event_n": row.get(event_col) if event_col else None,
                        "source_table": table_path.name,
                        "source_row_index": int(idx),
                    }
                )
                source_rows.append(record)
                plot_rows.append(record)
            if source_rows:
                source_path = out_dir / "publication_figure_prevalence_source_data.csv"
                pd.DataFrame(source_rows).to_csv(source_path, index=False)
                source_files.append(source_path.name)
                has_prevalence = True

        risk_pct_col = _find_column(
            frame,
            exact=("outcome_risk_pct", "risk_pct", "event_rate_pct"),
            suffixes=("_risk_pct", "_rate_pct"),
            exclude=("prevalence_pct", "incidence_pct"),
        )
        risk_prop_col = _find_column(
            frame,
            exact=("outcome_risk", "risk", "event_rate"),
            suffixes=("_risk", "_rate"),
            exclude=("prevalence", "incidence"),
        )
        if not has_outcome_risk and (risk_pct_col or risk_prop_col):
            denominator_col = _find_column(
                frame,
                exact=("n", "n_total", "total_n", "denominator", "n_denominator"),
            )
            event_col = _event_count_column(frame, denominator_col)
            label_col = _label_column(frame) or _find_column(
                frame,
                exact=("group", "category", "stratum", "exposure"),
            )
            binary_group_col = None if label_col else _binary_group_column(frame)
            metric_source = str(risk_pct_col or risk_prop_col or "outcome risk")
            metric_label = _publication_label(
                metric_source.replace("_pct", "").replace("_risk", " risk")
            )
            source_rows = []
            for idx, row in frame.iterrows():
                estimate = _as_percent(row, risk_pct_col or risk_prop_col)
                if estimate is None:
                    continue
                if _is_risk_difference_row(
                    row,
                    row.get(label_col) if label_col else None,
                    row.get(binary_group_col) if binary_group_col else None,
                ):
                    continue
                if label_col:
                    group_label = row.get(label_col)
                elif binary_group_col:
                    group_label = _binary_group_label(
                        binary_group_col,
                        row.get(binary_group_col),
                    )
                else:
                    group_label = f"Group {idx + 1}"
                record = row.to_dict()
                record.update(
                    {
                        "plot_metric": metric_label,
                        "plot_group_label": _publication_label(group_label),
                        "plot_estimate_pct": estimate,
                        "plot_ci_low_pct": _as_percent(
                            row,
                            _find_column(frame, exact=("ci_low_pct", "lower_pct"))
                            or _find_column(
                                frame,
                                exact=("ci_low", "lower"),
                                suffixes=(
                                    "_ci_low_pct",
                                    "_ci_low",
                                    "_lower_pct",
                                    "_lower",
                                ),
                            ),
                        ),
                        "plot_ci_high_pct": _as_percent(
                            row,
                            _find_column(frame, exact=("ci_high_pct", "upper_pct"))
                            or _find_column(
                                frame,
                                exact=("ci_high", "upper"),
                                suffixes=(
                                    "_ci_high_pct",
                                    "_ci_high",
                                    "_upper_pct",
                                    "_upper",
                                ),
                            ),
                        ),
                        "plot_denominator": (
                            row.get(denominator_col) if denominator_col else None
                        ),
                        "plot_event_n": row.get(event_col) if event_col else None,
                        "source_table": table_path.name,
                        "source_row_index": int(idx),
                    }
                )
                source_rows.append(record)
                plot_rows.append(record)
            if source_rows:
                source_path = (
                    out_dir / "publication_figure_absolute_risk_source_data.csv"
                )
                pd.DataFrame(source_rows).to_csv(source_path, index=False)
                source_files.append(source_path.name)
                has_outcome_risk = True

        if has_prevalence and has_outcome_risk:
            break

    if has_prevalence and has_outcome_risk:
        title = "Prevalence and absolute outcome risk"
        claim = "Exposure prevalence and absolute outcome risk are shown before adjusted relative estimates."
    elif has_prevalence:
        title = "Exposure prevalence"
        claim = "Exposure prevalence is shown before adjusted relative estimates."
    elif has_outcome_risk:
        title = "Absolute outcome risk"
        claim = "Absolute outcome risk is shown before adjusted relative estimates."
    else:
        title = ""
        claim = ""
    return {
        "plot_rows": plot_rows,
        "source_files": source_files,
        "has_prevalence": has_prevalence,
        "has_outcome_risk": has_outcome_risk,
        "title": title,
        "claim": claim,
    }


def _render_absolute_risk_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Render measurement availability and unadjusted outcome risk.

    The renderer accepts only the direct parent's tidy absolute-risk contract
    (``exposure``, ``group_type``, ``estimate_type``).  It never re-reads the
    cohort and every plotted row carries a positional trace back to the parent
    CSV.  This is intentionally separate from the association renderer: an
    absolute-risk context step has no adjusted estimand to invent or borrow.
    """

    parent_step_id = str(current_step_id or "").removesuffix("_figure")
    if not parent_step_id or parent_step_id == str(current_step_id or ""):
        return None
    parent_outputs = Path(run_dir) / "steps" / parent_step_id / "outputs"
    if not parent_outputs.exists():
        return None

    candidates = [parent_outputs / "exposure_outcome_summary.csv"]
    candidates.extend(
        path for path in sorted(parent_outputs.glob("*.csv")) if path not in candidates
    )
    table_path: Optional[Path] = None
    frame: Optional[pd.DataFrame] = None
    required = {"exposure", "group_type", "estimate_type"}
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            loaded = pd.read_csv(candidate).reset_index(drop=True)
        except Exception:
            continue
        if loaded.empty or not required.issubset(set(loaded.columns)):
            continue
        if not any(
            column in loaded.columns
            for column in ("outcome_risk_pct", "outcome_risk", "estimate")
        ):
            continue
        table_path, frame = candidate, loaded
        break
    if table_path is None or frame is None:
        return None

    estimate_type = frame["estimate_type"].astype(str).str.lower()
    group_type = frame["group_type"].astype(str).str.lower()
    group_value = (
        frame.get("group_value", pd.Series("", index=frame.index, dtype="object"))
        .astype(str)
        .str.lower()
    )

    availability_mask = (
        estimate_type.eq("prevalence")
        & group_type.eq("source_state")
        & group_value.eq("observed")
    )
    availability = frame.loc[availability_mask].copy()

    risk_mask = estimate_type.eq("outcome_risk")
    level_risk = frame.loc[risk_mask & group_type.eq("exposure_level")].copy()
    if len(level_risk) >= 2:
        counts = level_risk["exposure"].astype(str).value_counts()
        primary_exposure = str(counts.index[0])
        risk = level_risk[
            level_risk["exposure"].astype(str).eq(primary_exposure)
        ].copy()
        no_source = frame.loc[
            risk_mask
            & group_type.eq("source_state")
            & group_value.eq("no_source")
            & frame["exposure"].astype(str).eq(primary_exposure)
        ].copy()
        risk = pd.concat([risk, no_source], axis=0)
    else:
        risk = frame.loc[risk_mask & group_type.eq("source_state")].copy()

    distribution = frame.loc[
        estimate_type.eq("continuous_distribution")
        & frame.get("median", pd.Series(float("nan"), index=frame.index)).notna()
    ].copy()
    if availability.empty or risk.empty:
        return None

    def traced(rows: pd.DataFrame, transform: str) -> pd.DataFrame:
        traced_rows = rows.copy()
        traced_rows.insert(0, "source_row_index", traced_rows.index.astype(int))
        # The parent table repeats exposure/label across prevalence and risk
        # rows.  Those names are generic validator key columns, so preserving
        # them would trigger an ambiguous many-to-many join.  Rename only the
        # display identifiers and let ``source_row_index`` provide the exact
        # positional trace; numeric source columns remain byte-for-byte intact.
        traced_rows = traced_rows.rename(
            columns={
                column: f"source_{column}"
                for column in (
                    "label",
                    "variable",
                    "term",
                    "exposure",
                    "contrast",
                    "stage",
                    "level",
                    "band",
                    "category",
                )
                if column in traced_rows.columns
            }
        )
        traced_rows["source_table"] = table_path.name
        traced_rows["source_transform"] = transform
        return traced_rows.reset_index(drop=True)

    availability_source = traced(availability, "observed_source_prevalence_rows_v1")
    risk_source = traced(risk, "absolute_outcome_risk_rows_v1")
    distribution_source = traced(distribution, "continuous_distribution_rows_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    availability_name = "absolute_risk_availability_source_data.csv"
    risk_name = "absolute_risk_outcome_source_data.csv"
    availability_source.to_csv(out_dir / availability_name, index=False)
    risk_source.to_csv(out_dir / risk_name, index=False)
    source_files = [availability_name, risk_name]
    distribution_name = "absolute_risk_distribution_source_data.csv"
    if not distribution_source.empty:
        distribution_source.to_csv(out_dir / distribution_name, index=False)
        source_files.append(distribution_name)

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
    has_distribution = not distribution.empty
    fig = plt.figure(
        figsize=(183 / 25.4, (112 if has_distribution else 88) / 25.4),
        constrained_layout=False,
    )
    if has_distribution:
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[0.82, 1.35],
            height_ratios=[1.0, 0.72],
            left=0.16,
            right=0.98,
            top=0.84,
            bottom=0.14,
            wspace=0.78,
            hspace=0.70,
        )
        ax_availability = fig.add_subplot(grid[0, 0])
        ax_distribution = fig.add_subplot(grid[1, 0])
        ax_risk = fig.add_subplot(grid[:, 1])
    else:
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[0.88, 1.32],
            left=0.16,
            right=0.98,
            top=0.84,
            bottom=0.18,
            wspace=0.78,
        )
        ax_availability = fig.add_subplot(grid[0, 0])
        ax_distribution = None
        ax_risk = fig.add_subplot(grid[0, 1])

    availability_x = [
        _as_percent(
            row, "prevalence_pct" if "prevalence_pct" in frame.columns else "prevalence"
        )
        for _, row in availability.iterrows()
    ]
    availability_lo = [_as_percent(row, "ci_low") for _, row in availability.iterrows()]
    availability_hi = [
        _as_percent(row, "ci_high") for _, row in availability.iterrows()
    ]
    availability_labels = [
        _short_figure_label(_publication_label(value), limit=25)
        for value in availability["exposure"].astype(str)
    ]
    y_availability = list(range(len(availability)))
    ax_availability.errorbar(
        availability_x,
        y_availability,
        xerr=[
            [
                max(0.0, x - (lo if lo is not None else x))
                for x, lo in zip(availability_x, availability_lo)
            ],
            [
                max(0.0, (hi if hi is not None else x) - x)
                for x, hi in zip(availability_x, availability_hi)
            ],
        ],
        fmt="o",
        color=palette.get("teal", "#42949E"),
        ecolor=palette.get("teal", "#42949E"),
        elinewidth=1.0,
        capsize=2.3,
        markersize=4.2,
    )
    ax_availability.set_yticks(y_availability)
    ax_availability.set_yticklabels(availability_labels, fontsize=6.7)
    ax_availability.set_xlim(0, 100)
    ax_availability.set_xlabel("Observed stays, % (95% CI)")
    ax_availability.set_title("Measurement availability", loc="left", pad=4)
    ax_availability.invert_yaxis()
    ax_availability.grid(axis="x", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    add_panel_label(ax_availability, "A", x=0.0, y=1.08)

    risk_x = [
        _as_percent(
            row,
            (
                "outcome_risk_pct"
                if "outcome_risk_pct" in frame.columns
                else "outcome_risk"
            ),
        )
        for _, row in risk.iterrows()
    ]
    risk_lo = [_as_percent(row, "ci_low") for _, row in risk.iterrows()]
    risk_hi = [_as_percent(row, "ci_high") for _, row in risk.iterrows()]
    risk_labels = []
    for _, row in risk.iterrows():
        exposure_label = _publication_label(row.get("exposure"))
        if str(row.get("group_type") or "").lower() == "exposure_level":
            label = f"{exposure_label} = {row.get('group_value')}"
        elif str(row.get("group_value") or "").lower() == "no_source":
            label = "No recorded source"
        else:
            label = _publication_label(row.get("label") or row.get("group_value"))
        risk_labels.append(_short_figure_label(label, limit=30))
    y_risk = list(range(len(risk)))
    ax_risk.errorbar(
        risk_x,
        y_risk,
        xerr=[
            [
                max(0.0, x - (lo if lo is not None else x))
                for x, lo in zip(risk_x, risk_lo)
            ],
            [
                max(0.0, (hi if hi is not None else x) - x)
                for x, hi in zip(risk_x, risk_hi)
            ],
        ],
        fmt="o",
        color=palette.get("blue", "#0F4D92"),
        ecolor=palette.get("blue", "#0F4D92"),
        elinewidth=1.0,
        capsize=2.3,
        markersize=4.2,
    )
    max_risk = max([value for value in risk_hi if value is not None] or risk_x)
    ax_risk.set_xlim(0, max(10.0, max_risk * 1.28))
    ax_risk.set_yticks(y_risk)
    ax_risk.set_yticklabels(risk_labels, fontsize=6.8)
    ax_risk.set_xlabel("Outcome risk, % (95% CI)")
    ax_risk.set_title("Absolute outcome risk", loc="left", pad=4)
    ax_risk.invert_yaxis()
    ax_risk.grid(axis="x", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    for row_index, (value, upper) in enumerate(zip(risk_x, risk_hi)):
        ax_risk.text(
            max(value, upper if upper is not None else value) + 0.45,
            row_index,
            f"{value:.1f}%",
            va="center",
            fontsize=6.2,
            color=palette.get("baseline", "#272727"),
        )
    add_panel_label(ax_risk, "B", x=0.0, y=1.06)

    panels: List[Dict[str, Any]] = [
        {
            "panel_id": "A",
            "title": "Measurement availability",
            "role": "descriptive_result",
            "chart_type": "dot_interval_prevalence",
            "claim": "Source-consistent observed prevalence is shown for each requested exposure.",
            "evidence_ids": [availability_name],
        },
        {
            "panel_id": "B",
            "title": "Absolute outcome risk",
            "role": "descriptive_result",
            "chart_type": "dot_interval_absolute_risk",
            "claim": "Unadjusted outcome risks and Wilson 95% confidence intervals are shown for prespecified exposure levels, retaining the no-source group.",
            "evidence_ids": [risk_name],
        },
    ]
    if ax_distribution is not None:
        medians = pd.to_numeric(distribution["median"], errors="coerce").to_numpy()
        q25 = pd.to_numeric(distribution["q25"], errors="coerce").to_numpy()
        q75 = pd.to_numeric(distribution["q75"], errors="coerce").to_numpy()
        y_distribution = list(range(len(distribution)))
        ax_distribution.errorbar(
            medians,
            y_distribution,
            xerr=[medians - q25, q75 - medians],
            fmt="o",
            color=palette.get("orange", "#E69F00"),
            ecolor=palette.get("orange", "#E69F00"),
            elinewidth=1.0,
            capsize=2.3,
            markersize=4.2,
        )
        ax_distribution.set_yticks(y_distribution)
        ax_distribution.set_yticklabels(
            [
                _short_figure_label(_publication_label(value), limit=25)
                for value in distribution["exposure"].astype(str)
            ],
            fontsize=6.7,
        )
        ax_distribution.set_xlabel("Median (IQR)")
        ax_distribution.set_title("Observed distribution", loc="left", pad=4)
        ax_distribution.invert_yaxis()
        ax_distribution.grid(axis="x", color="#D8D8D8", linewidth=0.55, alpha=0.8)
        add_panel_label(ax_distribution, "C", x=0.0, y=1.10)
        panels.append(
            {
                "panel_id": "C",
                "title": "Observed distribution",
                "role": "descriptive_result",
                "chart_type": "median_iqr",
                "claim": "Continuous observed exposures are summarised by median and interquartile range without post-hoc binning.",
                "evidence_ids": [distribution_name],
            }
        )

    fig.suptitle(
        "Measurement availability and unadjusted outcome context",
        x=0.16,
        ha="left",
        y=0.98,
        fontsize=9.2,
        fontweight="bold",
    )
    contract = make_figure_contract(
        figure_id="publication_figure",
        core_claim=(
            "The figure shows measurement availability, absolute outcome risk, "
            "and continuous-distribution context before adjusted modelling."
        ),
        panels=panels,
        source_data=source_files,
        statistics_note=(
            "Rendered deterministically from the direct parent's tidy summary. "
            "Risk and prevalence intervals are Wilson 95% confidence intervals; "
            "continuous values are median (IQR)."
        ),
    )
    outputs = save_publication_figure(
        fig, out_dir / "publication_figure", contract=contract, dpi=300
    )
    plt.close(fig)

    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    summary = {
        "step_id": current_step_id,
        "method": "deterministic_absolute_risk_publication_figure",
        "rendering_only": True,
        "source_step_id": parent_step_id,
        "source_table": str(table_path),
        "source_data_files": source_files,
        "n_availability_rows": int(len(availability)),
        "n_risk_rows": int(len(risk)),
        "n_distribution_rows": int(len(distribution)),
        "figure_files": figure_files,
        "figure_path": "publication_figure.png",
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "absolute_risk_publication_bundle_from_parent_outputs_v1"


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


# A ``*_figure`` step renders its parent analysis step's outputs. The parent's
# deterministic runner records a CONTROLLED ``analysis_family`` in its
# step_summary.json (e.g. the ordinal dose-response runner writes
# ``analysis_family='association'``). Routing the figure by that PROVEN family is
# the structural signal to use when the (stochastically named) figure step id
# carries no family token — e.g. ``05_primary_stage_outcome_analysis_figure`` has
# no association token, yet its parent produced a canonical association forest.
# This is a small CLOSED enum (the families the runners emit) mapped to renderers,
# NOT a fragile substring match against a free-form step id / intent prose. Only
# RESULT families are mapped; ``descriptive``/baseline figures are intentionally
# absent so they keep their existing (LLM-coder) path rather than being claimed by
# a renderer that would emit an empty figure.
_UPSTREAM_FAMILY_TO_RENDERER_KEY: dict[str, str] = {
    "association": "association",
    "dose_response": "association",
    "prediction": "prediction",
    "prediction_model": "prediction",
    "survival": "survival",
    "survival_analysis": "survival",
    "cohort_definition": "cohort",
    "cohort_definition_sensitivity": "sensitivity",
    "sensitivity_analysis": "sensitivity",
    "missingness": "missingness",
    "measurement": "missingness",
    "data_quality": "missingness",
    "absolute_risk_context": "absolute_risk",
    "phenotyping": "phenotype",
    "clustering": "phenotype",
    "descriptive": "descriptive",
    "table_one": "descriptive",
    "baseline": "descriptive",
}


# Some supporting/QC analyses intentionally retain a broad analysis family
# (for example, ``association_study``) while recording a narrower controlled
# method.  Exact-method routing lets those steps use a schema-matched renderer
# without teaching the global router any clinical variable, benchmark item, or
# figure-specific name.  Keep this map closed and exact: free-text method-token
# matching would recreate the same accidental routing problem as step-id prose.
_UPSTREAM_METHOD_TO_RENDERER_KEY: dict[str, str] = {
    "ordinal_exposure_derivation_and_quality_control": "ordered_distribution",
    "exposure_distribution_and_missingness_audit": "distribution_availability",
    "cohort_definition_sensitivity": "sensitivity",
    "missingness": "missingness",
    "missingness_audit": "missingness",
    "missingness_measurement_audit": "missingness",
}


# A step-level artifact contract is more precise than either the whole-step
# analysis family or the planner's free-form step id.  New producers should
# declare this closed enum; exact-method routing below remains a compatibility
# adapter for completed runs that predate the field.
_UPSTREAM_FIGURE_DATA_FAMILY_TO_RENDERER_KEY: dict[str, str] = {
    "ordered_category_distribution": "ordered_distribution",
}
_AMBIGUOUS_FIGURE_DATA_FAMILY = "__ambiguous_figure_data_family__"
_INCOMPATIBLE_FIGURE_DATA_FAMILY = "__incompatible_figure_data_family__"


def _resolve_upstream_analysis_family(
    run_dir: Path, current_step_id: str
) -> Optional[str]:
    """Return the ``analysis_family`` recorded by a figure step's parent step."""

    parent = str(current_step_id or "").removesuffix("_figure")
    if not parent or parent == str(current_step_id):
        return None
    summ = Path(run_dir) / "steps" / parent / "outputs" / "step_summary.json"
    try:
        fam = json.loads(summ.read_text("utf-8")).get("analysis_family")
    except Exception:
        return None
    return str(fam).strip().lower() if fam else None


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


def _renderer_for_upstream_figure_data_family(family: Optional[str]):
    """Map an explicit step-level figure-data contract to its renderer."""

    key = _UPSTREAM_FIGURE_DATA_FAMILY_TO_RENDERER_KEY.get(
        str(family or "").strip().lower()
    )
    if key == "ordered_distribution":
        from .figures.ordered_distribution import (
            render_ordered_distribution_bundle_from_prior_outputs,
        )

        return render_ordered_distribution_bundle_from_prior_outputs
    return None


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


def _sealed_renderer_parent_digest_seal(
    run_dir: Path,
    figure_step_id: str,
    repair_id: str,
) -> Optional[dict[str, str]]:
    """Return the exact evidence digests one sealed renderer may consume."""

    distribution_id = (
        "distribution_availability_publication_bundle_from_parent_outputs_v1"
    )
    if repair_id == distribution_id:
        return _distribution_availability_parent_digest_seal(run_dir, figure_step_id)
    if repair_id == "continuous_measurement_audit_publication_bundle_v1":
        return _continuous_measurement_audit_parent_digest_seal(run_dir, figure_step_id)
    if repair_id == "absolute_risk_incidence_prevalence_publication_bundle_v1":
        return _absolute_risk_parent_digest_seal(run_dir, figure_step_id)
    if repair_id == (
        "ordered_category_distribution_availability_publication_bundle_v2"
    ):
        return _ordered_distribution_availability_parent_digest_seal(
            run_dir, figure_step_id
        )
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

    if repair_id == "absolute_risk_incidence_prevalence_publication_bundle_v1":
        from .figures.absolute_risk import (
            render_absolute_risk_bundle_from_prior_outputs,
        )

        observed = render_absolute_risk_bundle_from_prior_outputs(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            preverified_parent_artifacts=snapshot,
        )
    elif repair_id == (
        "distribution_availability_publication_bundle_from_parent_outputs_v1"
    ):
        from .figures.distribution_availability import (
            render_distribution_availability_bundle_from_prior_outputs,
        )

        observed = render_distribution_availability_bundle_from_prior_outputs(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            preverified_parent_artifacts=snapshot,
        )
    elif repair_id == "continuous_measurement_audit_publication_bundle_v1":
        from .figures.continuous_measurement_audit import (
            render_continuous_measurement_audit_bundle,
        )

        observed = render_continuous_measurement_audit_bundle(
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
    elif repair_id == (
        "ordered_category_distribution_availability_publication_bundle_v2"
    ):
        from .figures.ordered_distribution import (
            render_ordered_distribution_bundle_from_prior_outputs,
        )

        observed = render_ordered_distribution_bundle_from_prior_outputs(
            run_dir=run_dir,
            current_step_id=current_step_id,
            out_dir=out_dir,
            preverified_parent_artifacts=snapshot,
            authorized_repair_id=repair_id,
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
    if repair_id == (
        "distribution_availability_publication_bundle_from_parent_outputs_v1"
    ):
        return (
            repair_id
            if _distribution_availability_parent_digest_seal(run_dir, step_id)
            is not None
            else None
        )
    if repair_id == "continuous_measurement_audit_publication_bundle_v1":
        return (
            repair_id
            if _continuous_measurement_audit_parent_digest_seal(run_dir, step_id)
            is not None
            else None
        )
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


def deterministic_figure_family_supported_for_upstream(
    run_dir: Path, step_id: str
) -> bool:
    """Compatibility boolean for the typed automatic-renderer gate."""

    return deterministic_figure_repair_id_for_upstream(run_dir, step_id) is not None


def deterministic_figure_family_supported(step_id: str) -> bool:
    """Deprecated name-only compatibility probe; names never establish ownership."""

    del step_id
    return False


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


def _publication_label(value: Any) -> str:
    token = str(value or "").strip()
    mapping = {
        "sepsis3": "Sepsis-3",
        "sep3_sofa2_max": "Sepsis-3",
        "age": "Age",
        "age_filled": "Age",
        "age_per_10y": "Age, per 10 years",
        "sex_m": "Male sex",
        "sex_male": "Male sex",
        "male": "Male sex",
        "hr_first": "Heart rate",
        "hr_first_filled": "Heart rate",
        "hr_max_per_10bpm": "Maximum heart rate, per 10 bpm",
        "map_first": "Mean arterial pressure",
        "map_first_filled": "Mean arterial pressure",
        "map_min": "Minimum mean arterial pressure",
        "resp_max_per_5": "Maximum respiratory rate, per 5/min",
        "temp_max_c": "Maximum temperature, per 1 deg C",
        "lactate": "Lactate",
        "lact": "Lactate",
        "lact_max_mmol_l": "Maximum lactate, per 1 mmol/L",
        "lact_measured": "Lactate measured",
        "bun_max_per_10": "Maximum BUN, per 10 units",
        "wbc_max_per_10": "Maximum WBC, per 10 units",
        "sofa2": "SOFA-2",
        "death": "In-hospital mortality",
        "alt_adult_no_los_all_vitals_sepsis3_derivable": "No LOS threshold",
        "alt_adult_los1_three_of_four_vitals_sepsis3_derivable": ">=3 of 4 vitals",
        "alt_adult_los1_no_temp_requirement_sepsis3_derivable": "No temperature",
        "alt_adult_los2_all_vitals_sepsis3_derivable": "ICU LOS >=2 d",
    }
    lower = token.lower()
    if lower in mapping:
        return mapping[lower]
    cleaned = lower
    for suffix in ("_filled", "_first", "_measured"):
        cleaned = cleaned.removesuffix(suffix)
    return cleaned.replace("_", " ").strip().title() or token


def _short_figure_label(value: Any, *, limit: int = 38) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "..."


def _truthy_figure_value(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes"}


def _explicit_false_figure_value(value: Any) -> bool:
    if value is False:
        return True
    if value is True or value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"false", "0", "no"}


def _sensitivity_plot_label(row: Mapping[str, Any]) -> str:
    spec_id = str(row.get("spec_id") or "").strip().lower()
    if spec_id.endswith("_crude_rd"):
        spec_id = spec_id.removesuffix("_crude_rd")
    mapping = {
        "full_export_step03_scope": "Full export",
        "primary_los_ge_1d": "Primary cohort",
        "primary": "Primary cohort",
        "cohort_no_los_restriction": "No ICU LOS restriction",
        "cohort_los_ge_2d": "ICU LOS >=2 d",
        "cohort_core_physiology_present": "Core physiology present",
        "primary_adult_los1_all_vitals_sepsis3_derivable": "Primary cohort",
        "alt_adult_no_los_all_vitals_sepsis3_derivable": "No LOS threshold",
        "alt_adult_los1_three_of_four_vitals_sepsis3_derivable": ">=3 of 4 vitals",
        "alt_adult_los1_no_temp_requirement_sepsis3_derivable": "No temperature",
        "alt_adult_los2_all_vitals_sepsis3_derivable": "ICU LOS >=2 d",
        "primary_lactate_complete_case": "Lactate obs.",
        "primary_without_lactate_adjustment": "No lactate adj.",
        "missing_raw_complete_case": "Complete-case",
        "missing_drop_lactate": "Drop lactate",
        "effect_robust_poisson_rr": "Risk ratio",
        "effect_marginal_standardized_rd": "Risk difference",
    }
    if spec_id in mapping:
        return mapping[spec_id]
    los_match = re.fullmatch(r"alt_cohort_los_ge_(\d+(?:p\d+)?)([hd])", spec_id)
    if los_match:
        value = los_match.group(1).replace("p", ".")
        unit = "h" if los_match.group(2) == "h" else "d"
        return f"ICU LOS >= {value} {unit}"
    if "complete_case" in spec_id:
        return "Complete-case"
    if "source_aware" in spec_id:
        return "Source-aware"
    label = str(row.get("display_label") or row.get("label") or spec_id).strip()
    return _short_figure_label(label.replace("LOS ≥", "LOS >="), limit=30)


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
    ("", "clustering_algorithm_details.json"): (
        "clustering_algorithm_details",
        "clustering_methodology",
    ),
    ("", "clustering_methodology.json"): (
        "clustering_methodology",
        "cluster_summary",
    ),
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
            out.extend(
                [
                    "cluster_summary",
                    "cluster_characteristics",
                    "cluster_mortality",
                    "clustering_performance",
                    "clustering_methodology",
                    "table_one",
                ]
            )
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
