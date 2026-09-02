"""Typed project-level research workflow projection for Pi Copilot.

This module owns only orchestration state.  Idea Mining, extraction, Research
Agent execution, evidence gates, and manuscript artefacts keep their existing
owners; the Copilot shell receives an immutable, path-free projection of those
owners' receipts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from easyicu.webserver import agent_pipeline_runs, agent_runs, jobs, sources
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.study_scientific_configuration import (
    ScientificConfiguration,
    SetupFacts,
)

from . import cohort_eligibility, plan_decisions, plan_review_progress
from .contracts import (
    EXECUTION_RETRY_REPLAYABLE_GATE_REASONS,
    PLAN_RESUME_OFFER_GATE_REASONS,
    plan_approval_allowed,
)
from .plan_projection import project_plan_conversation_preview
from .projections import (
    StudySetupReceipt,
    project_job,
    project_run_outcome,
    project_study_setup_receipt,
)
from .run_authority import (
    list_bound_run_history,
    research_pipeline_project_root,
    workflow_authoritative_run,
)

WorkflowStatus = Literal[
    "blocked",
    "ready",
    "running",
    "complete",
    "optional",
    "review_required",
]


class ResearchWorkflowStage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Literal[
        "question",
        "idea",
        "setup",
        "extraction",
        "plan",
        "analysis",
        "interpretation",
        "manuscript",
    ]
    label: str
    status: WorkflowStatus
    owner: str
    reason_code: str


class ResearchWorkflowSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.pi-research-workflow/1"] = (
        "easyicu.pi-research-workflow/1"
    )
    current_stage: str
    next_action_code: str
    missing_setup_fields: List[str] = Field(default_factory=list, max_length=16)
    planning_prerequisites_missing: List[str] = Field(
        default_factory=list, max_length=8
    )
    stages: List[ResearchWorkflowStage] = Field(min_length=8, max_length=8)
    completed_required_stages: int = Field(ge=0, le=7)
    required_stage_count: Literal[7] = 7
    scientific_authority: Literal["EasyICU"] = "EasyICU"
    pi_role: Literal["conversation_and_orchestration"] = (
        "conversation_and_orchestration"
    )
    study_setup_receipt: StudySetupReceipt
    plan_review_summary: Optional[Mapping[str, Any]] = None
    plan_conversation_preview: Optional[Mapping[str, Any]] = None
    plan_execution_ready: bool = False
    analysis_validation_retry_available: bool = False


class ProjectWorkflowProjection(BaseModel):
    """One adapter-neutral projection compiled from raw owner receipts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow: ResearchWorkflowSnapshot
    active_job: Mapping[str, Any]
    latest_run: Mapping[str, Any]


def active_export_matches_study(
    study: Optional[Mapping[str, Any]], active_export_path: Any
) -> bool:
    """Return whether the source owner's active export belongs to this study."""

    study_row: Mapping[str, Any] = study or {}
    source = study_row.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    expected = str(source.get("path") or "").strip()
    active = str(active_export_path or "").strip()
    return bool(expected and active and expected == active)


def registered_export_matches_study(
    study: Optional[Mapping[str, Any]], registry: Optional[Mapping[str, Any]]
) -> bool:
    """Return whether the project's exact bound export is still registered."""

    study_row: Mapping[str, Any] = study or {}
    confirmations = study_row.get("confirmations")
    confirmations = confirmations if isinstance(confirmations, Mapping) else {}
    if confirmations.get("extraction_completed") is not True:
        return False
    source = study_row.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    expected = str(source.get("path") or "").strip()
    if not expected:
        return False
    registry_row: Mapping[str, Any] = registry or {}
    return any(
        isinstance(row, Mapping)
        and bool(row.get("ok"))
        and str(row.get("path") or "").strip() == expected
        for row in (registry_row.get("sources") or [])
    )


# These are Planner proposal fields, not pre-plan setup questions.  The
# researcher reviews the complete candidate plan; they should not have to
# invent an endpoint contract or a sensitivity implementation before seeing
# that plan.  Full execution still requires the reviewed proposal to be
# promoted into typed StudyContext authority.
_PLANNER_PROPOSAL_FINDING_CODES = frozenset(
    {
        "OUTCOME_DEFINITION_UNRESOLVED",
        "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED",
    }
)


def _identified_data_source(study: Mapping[str, Any]) -> bool:
    """Whether the study is bound to a data source EasyICU has identified."""

    source = study.get("data_source")
    if not isinstance(source, Mapping):
        return False
    return bool(str(source.get("database") or "").strip())


def build_research_workflow_snapshot(
    *,
    study: Optional[Mapping[str, Any]],
    active_export_present: bool,
    active_job: Optional[Mapping[str, Any]],
    latest_run: Optional[Mapping[str, Any]],
    plan_review_authority: Optional[Mapping[str, Any]] = None,
    continuing_review_choices: bool = False,
) -> ResearchWorkflowSnapshot:
    """Compile owner receipts into one deterministic Copilot workflow state."""

    study_row: Mapping[str, Any] = study or {}
    job_row: Mapping[str, Any] = active_job or {}
    run_row: Mapping[str, Any] = latest_run or {}
    question_ready = bool(str(study_row.get("question") or "").strip())
    idea_handoff = study_row.get("idea_handoff")
    idea_handoff = idea_handoff if isinstance(idea_handoff, Mapping) else {}
    idea_accepted = bool(
        idea_handoff.get("status") == "accepted"
        and str(idea_handoff.get("run_id") or "").strip()
        and str(idea_handoff.get("idea_id") or "").strip()
        and len(str(idea_handoff.get("canonical_handoff_sha256") or "")) == 64
    )
    idea_recommendation = str(idea_handoff.get("go_no_go") or "").strip()
    idea_blocks_execution = bool(idea_accepted and idea_recommendation != "recommend")
    assessment = ScientificConfiguration.inspect(study_row).assess_setup(
        SetupFacts(
            active_export_present=bool(active_export_present),
            eligibility_stated=cohort_eligibility.eligibility_stated(study_row),
            dependence_finding=study_context_owner.analysis_dependence_finding(
                dict(study_row)
            ),
            window_finding=study_context_owner.materialization_window_finding(
                dict(study_row)
            ),
        )
    )
    missing = list(assessment.missing_fields)
    planning_prerequisites_missing = list(assessment.planning_prerequisites_missing)
    setup_ready = not missing
    job_kind = str(job_row.get("kind") or "")
    job_status = str(job_row.get("status") or "")
    extraction_running = job_kind == "extract" and job_status == "running"
    analysis_running = job_kind == "agent-run" and job_status == "running"
    artifact_names = {
        str(item) for item in (run_row.get("artifact_names") or []) if item
    }
    run_type = str(run_row.get("run_type") or "")
    run_engine = str(run_row.get("engine") or "")
    has_plan = "agent_plan.json" in artifact_names
    has_evidence = "evidence_ledger.json" in artifact_names
    has_outputs = bool(
        artifact_names
        & {
            "result_tables.json",
            "table1_summary.json",
            "missingness_audit.json",
            "roc_curve.json",
            "calibration_curve.json",
            "figure_gallery.json",
        }
    )
    manuscript_artifact_present = "manuscript_draft.json" in artifact_names
    full_run = run_type == "full"
    pipeline_run = run_engine == "easyicu.research_agent.pipeline"
    pipeline_receipt = "source_run_manifest.json" in artifact_names
    gate_status = str(run_row.get("gate_status") or "")
    run_blocked = gate_status == "blocked"
    raw_gate_checks = run_row.get("gate_checks")
    gate_checks = dict(raw_gate_checks) if isinstance(raw_gate_checks, Mapping) else {}
    # The projection always writes a bounded manuscript_draft.json, including
    # a diagnostic explanation when Writer fails closed.  Only the Research
    # Agent's manuscript_ready gate proves that the file contains a real,
    # evidence-bound draft suitable for human review.
    has_manuscript = bool(
        manuscript_artifact_present and gate_checks.get("manuscript_ready") is True
    )
    executed_analysis_validated = bool(
        gate_checks.get("execution_complete") is True
        and gate_checks.get("analysis_validated") is True
    )
    analysis_outputs_available = bool(
        full_run
        and pipeline_run
        and pipeline_receipt
        and has_plan
        and has_evidence
        and has_outputs
        and gate_checks.get("execution_complete") is True
        and (
            gate_checks.get("analysis_validated") is True
            or (
                gate_checks.get("evidence_complete") is True
                and gate_checks.get("numeric_verified") is True
            )
        )
    )
    preflight_complete = bool(
        run_type == "preflight"
        and has_evidence
        and gate_status == "analysis_only"
        and str(run_row.get("readiness_status") or "") == "awaiting_human_signoff"
    )
    # A completed local preflight is an owner-issued receipt that the exact
    # bound registered export was resolved and reviewed.  Treat it as stronger
    # evidence than a legacy StudyContext extraction flag so prepared-export
    # reuse cannot fall backward to extraction after preflight succeeds.
    prepared_export_receipted = bool(active_export_present or preflight_complete)
    # Planning starts from the user's question plus a data source EasyICU has
    # identified -- a prepared export or a bound, identified local database.
    # The Planner proposes unresolved analysis choices, including population
    # eligibility, in its reviewable plan; Copilot must not create a shadow
    # plan or turn those proposals into a pre-plan questionnaire.
    planning_data_ready = bool(
        prepared_export_receipted or _identified_data_source(study_row)
    )
    eligibility_confirmation_required = bool(
        (
            "cohort_eligibility" in planning_prerequisites_missing
            # Initial planning may propose eligibility. Once a plan exists, a
            # StudyContext change invalidates that receipt and must be
            # reconfirmed before regenerating or executing the old plan.
            or (has_plan and "cohort_eligibility" in missing)
        )
        and question_ready
        and planning_data_ready
    )
    plan_generation_ready = bool(
        question_ready
        and planning_data_ready
        and not planning_prerequisites_missing
        and not has_plan
    )
    review_authority = (
        plan_review_authority if isinstance(plan_review_authority, Mapping) else {}
    )
    authority_requests = review_authority.get("requests")
    authority_reason_codes = {
        str(item.get("reason_code") or "").strip()
        for item in (authority_requests if isinstance(authority_requests, list) else [])
        if isinstance(item, Mapping) and str(item.get("reason_code") or "").strip()
    }
    pending_review_reason_codes = authority_reason_codes or {
        str(item).strip()
        for item in (run_row.get("pending_review_reason_codes") or [])
        if str(item).strip()
    }
    plan_review_codes = {
        "operator_plan_approval_required",
        "plan_scientific_changes_required",
        "scientific_plan_review_policy_stale",
    }
    active_plan_review_codes = sorted(pending_review_reason_codes & plan_review_codes)
    plan_review_declared = bool(
        has_plan
        and str(run_row.get("run_status") or "") == "human_review_pending"
        and active_plan_review_codes
    )
    raw_scientific_review = review_authority.get("scientific_plan_review")
    raw_scientific_review = (
        raw_scientific_review if isinstance(raw_scientific_review, Mapping) else {}
    )
    dimension_scores = raw_scientific_review.get("dimension_scores")
    dimension_scores = dimension_scores if isinstance(dimension_scores, Mapping) else {}
    review_findings = raw_scientific_review.get("findings")
    review_findings = review_findings if isinstance(review_findings, list) else []
    raw_facts = raw_scientific_review.get("facts")
    raw_facts = raw_facts if isinstance(raw_facts, Mapping) else {}
    raw_remediation_buckets = raw_facts.get("remediation_buckets")
    raw_remediation_buckets = (
        raw_remediation_buckets if isinstance(raw_remediation_buckets, Mapping) else {}
    )

    def projected_remediation_codes(route: str) -> List[str]:
        values = raw_remediation_buckets.get(route)
        rows = [
            str(code)[:120]
            for code in (values if isinstance(values, list) else [])[:40]
            if str(code).strip()
        ]
        if route == "study_authority_change":
            return [
                code for code in rows if code not in _PLANNER_PROPOSAL_FINDING_CODES
            ]
        if route == "agent_plan_revision":
            proposal_codes = [
                str(item.get("code") or "")[:120]
                for item in review_findings[:40]
                if isinstance(item, Mapping)
                and str(item.get("code") or "") in _PLANNER_PROPOSAL_FINDING_CODES
            ]
            return list(dict.fromkeys([*rows, *proposal_codes]))[:40]
        return rows

    def projected_authorization_question(item: Mapping[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "code": str(item.get("code") or "")[:120],
            "question": str(item.get("authorization_question") or "")[:1_200],
        }
        evidence = str(item.get("message") or "").strip()
        if evidence:
            row["evidence"] = evidence[:1_600]
        evidence_refs = item.get("evidence_refs")
        if isinstance(evidence_refs, list):
            refs = [
                str(value)[:240] for value in evidence_refs[:12] if str(value).strip()
            ]
            if refs:
                row["evidence_refs"] = refs
        remediation = str(item.get("remediation") or "").strip()
        if remediation:
            row["remediation"] = remediation[:1_600]
        return row

    plan_review_summary = (
        {
            "status": str(raw_scientific_review.get("status") or "")[:40],
            "score": raw_scientific_review.get("score"),
            "top_journal_candidate": bool(
                raw_scientific_review.get("top_journal_candidate")
            ),
            "review_scope": str(
                raw_scientific_review.get("review_scope") or "pre_execution_plan"
            )[:80],
            "rendered_outputs_assessed": bool(
                raw_scientific_review.get("rendered_outputs_assessed")
            ),
            "dimension_scores": {
                str(key)[:80]: int(value)
                for key, value in list(dimension_scores.items())[:12]
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
            "finding_codes": [
                str(item.get("code") or "")[:120]
                for item in review_findings[:40]
                if isinstance(item, Mapping) and str(item.get("code") or "").strip()
            ],
            "authorization_questions": [
                projected_authorization_question(item)
                for item in review_findings[:40]
                if isinstance(item, Mapping)
                and bool(item.get("requires_user_authorization"))
                and str(item.get("code") or "") not in _PLANNER_PROPOSAL_FINDING_CODES
                and str(item.get("authorization_question") or "").strip()
            ],
            "remediation_buckets": {
                route: projected_remediation_codes(route)
                for route in (
                    "agent_plan_revision",
                    "study_authority_change",
                    "external_evidence",
                    "independent_review",
                )
            },
        }
        if raw_scientific_review
        else None
    )
    current_scientific_digest = study_context_owner.scientific_configuration_sha256(
        study_row
    )
    planned_scientific_digest = str(
        review_authority.get("scientific_configuration_sha256")
        or run_row.get("scientific_configuration_sha256")
        or ""
    ).strip()
    review_authority_available = bool(
        review_authority
        and str(review_authority.get("run_id") or "").strip()
        == str(run_row.get("run_id") or "").strip()
        and bool(review_authority.get("resumable_here"))
    )
    plan_configuration_matches = bool(
        len(planned_scientific_digest) == 64
        and planned_scientific_digest == current_scientific_digest
    )
    plan_review_pending = bool(
        plan_review_declared
        and review_authority_available
        and plan_configuration_matches
    )
    # Sibling choices may continue across host-receipted edits, but the
    # superseded candidate still cannot be approved or executed.
    choices_pending = bool(
        continuing_review_choices
        and plan_review_declared
        and "plan_scientific_changes_required" in active_plan_review_codes
        and "scientific_plan_review_policy_stale" not in active_plan_review_codes
        and not analysis_running
    )
    plan_execution_ready = bool(
        plan_review_pending
        and not choices_pending
        and plan_approval_allowed(review_authority)
        and str(review_authority.get("budget_mode") or "full_reviewed")
        != "planner_canary"
    )
    live_plan_reason = (
        "scientific_plan_review_policy_stale"
        if "scientific_plan_review_policy_stale" in active_plan_review_codes
        else "plan_scientific_changes_required"
        if "plan_scientific_changes_required" in active_plan_review_codes
        else "operator_plan_approval_required"
        if plan_execution_ready
        else "plan_execution_upgrade_required"
    )
    plan_review_reason_code = (
        "plan_scientific_changes_required"
        if choices_pending
        else live_plan_reason
        if plan_review_pending
        else "plan_configuration_superseded"
        if plan_review_declared
        and planned_scientific_digest
        and not plan_configuration_matches
        else "plan_review_not_resumable"
        if plan_review_declared
        else ""
    )
    # A live, digest-matching review is an approval gate. A stale or
    # non-resumable plan remains historical evidence, but the next governed
    # action is a fresh planning run rather than approval or in-place editing.
    plan_attention_required = bool(plan_review_pending or choices_pending)
    plan_regeneration_required = bool(
        plan_review_declared and not plan_attention_required and not analysis_running
    )
    analysis_complete = bool(
        full_run
        and pipeline_run
        and pipeline_receipt
        and has_plan
        and has_evidence
        and has_outputs
        and (gate_status == "analysis_only" or executed_analysis_validated)
    )
    # A completed, validated full run is a stronger downstream receipt than
    # legacy blank setup slots. The approved Agent plan owns the exact study
    # design used for execution, so an older StudyContext must not pull the
    # visible workflow backward from result interpretation to setup.
    setup_receipted = bool(
        setup_ready or analysis_complete or analysis_outputs_available
    )
    extraction_receipted = bool(
        prepared_export_receipted or analysis_complete or analysis_outputs_available
    )
    pipeline_attempt_blocked = bool(
        full_run
        and pipeline_run
        and pipeline_receipt
        and run_blocked
        and not analysis_complete
        and not analysis_outputs_available
    )
    # A terminal fail-closed run is historical evidence, not a completed
    # analysis stage.  The ordinary Copilot journey must return to a fresh Plan
    # confirmation instead of advancing to interpretation (where the user can
    # only be told that the old run failed).  The failed run stays immutable;
    # a newly authorized provider turn receives a new run id and Plan review.
    failed_pipeline_regeneration_required = bool(
        pipeline_attempt_blocked and not analysis_running and not plan_review_declared
    )
    failed_execution_retry_available = bool(
        failed_pipeline_regeneration_required
        and has_plan
        and has_evidence
        and str(run_row.get("gate_reason") or "")
        in EXECUTION_RETRY_REPLAYABLE_GATE_REASONS
        and len(planned_scientific_digest) == 64
        and planned_scientific_digest == current_scientific_digest
    )
    analysis_validation_retry_available = bool(
        analysis_outputs_available
        and run_blocked
        and str(run_row.get("gate_reason") or "")
        in EXECUTION_RETRY_REPLAYABLE_GATE_REASONS
        and len(planned_scientific_digest) == 64
        and planned_scientific_digest == current_scientific_digest
    )
    # Offering "resume this plan" is narrower than seeding a fresh planning run
    # from a preserved prefix (PLANNER_CHECKPOINT_GATE_REASONS): only a
    # budget-exhausted plan is itself still intact.
    planner_checkpoint_resume_available = bool(
        failed_pipeline_regeneration_required
        and str(run_row.get("gate_reason") or "") in PLAN_RESUME_OFFER_GATE_REASONS
        and bool(run_row.get("development_planner_checkpoint_available"))
        and len(planned_scientific_digest) == 64
        and planned_scientific_digest == current_scientific_digest
    )
    plan_regeneration_required = bool(
        plan_regeneration_required or failed_pipeline_regeneration_required
    )
    plan_regeneration_reason_code = (
        "planner_checkpoint_resume_available"
        if planner_checkpoint_resume_available
        else "failed_pipeline_execution_retry_available"
        if failed_execution_retry_available
        else "failed_pipeline_requires_fresh_plan"
        if failed_pipeline_regeneration_required
        else plan_review_reason_code
    )
    legacy_full_scaffold = bool(full_run and not pipeline_run and has_plan)

    stages = [
        ResearchWorkflowStage(
            id="idea",
            label="Idea mining",
            status=(
                "review_required"
                if idea_blocks_execution
                else "complete"
                if idea_accepted
                else "optional"
                if question_ready
                else "ready"
            ),
            owner="easyicu.webserver.ideas.mining",
            reason_code=(
                "idea_feasibility_refresh_required"
                if idea_blocks_execution
                else "idea_handoff_accepted"
                if idea_accepted
                else "idea_mining_available"
            ),
        ),
        ResearchWorkflowStage(
            id="question",
            label="Scientific question",
            status="complete" if question_ready else "ready",
            owner="easyicu.webserver.study_contexts",
            reason_code=("question_bound" if question_ready else "question_required"),
        ),
        ResearchWorkflowStage(
            id="setup",
            label="Study setup",
            status="complete"
            if setup_receipted
            else ("ready" if question_ready else "blocked"),
            owner="easyicu.webserver.study_contexts",
            reason_code=(
                "approved_plan_setup_receipt"
                if (analysis_complete or analysis_outputs_available) and not setup_ready
                else "study_setup_complete"
                if setup_ready
                else "cohort_eligibility_confirmation_required"
                if eligibility_confirmation_required
                else "study_setup_incomplete"
            ),
        ),
        ResearchWorkflowStage(
            id="plan",
            label="Analysis plan",
            status=(
                "blocked"
                if idea_blocks_execution
                else "review_required"
                if plan_attention_required
                else "running"
                if analysis_running
                else "ready"
                if plan_regeneration_required
                else "complete"
                if has_plan
                else "ready"
                if plan_generation_ready
                else "blocked"
            ),
            owner="easyicu.research_agent.planning",
            reason_code=(
                "idea_feasibility_refresh_required"
                if idea_blocks_execution
                else plan_review_reason_code
                if plan_attention_required
                else "analysis_running"
                if analysis_running
                else plan_regeneration_reason_code
                if plan_regeneration_required
                else "agent_plan_ready"
                if has_plan
                else "provider_ready_to_generate_plan"
                if plan_generation_ready
                else "cohort_eligibility_confirmation_required"
                if eligibility_confirmation_required
                else "active_export_or_setup_required"
            ),
        ),
        ResearchWorkflowStage(
            id="extraction",
            label="Feature extraction",
            status=(
                "complete"
                if extraction_receipted
                else "running"
                if extraction_running
                else "ready"
                if setup_ready
                else "blocked"
            ),
            owner="easyicu.webserver.routes.jobs",
            reason_code=(
                "approved_analysis_input_receipt"
                if (analysis_complete or analysis_outputs_available)
                and not prepared_export_receipted
                else "active_export_ready"
                if extraction_receipted
                else "extraction_running"
                if extraction_running
                else "extraction_ready"
                if setup_ready
                else "study_setup_incomplete"
            ),
        ),
        ResearchWorkflowStage(
            id="analysis",
            label="Analysis and validation",
            status=(
                "blocked"
                if idea_blocks_execution
                else "blocked"
                if plan_attention_required or plan_regeneration_required
                else "running"
                if analysis_running
                else "complete"
                if analysis_complete
                else "review_required"
                if analysis_outputs_available
                else "review_required"
                if pipeline_attempt_blocked
                else "ready"
                if prepared_export_receipted and setup_ready
                else "blocked"
            ),
            owner="easyicu.research_agent.pipeline",
            reason_code=(
                "idea_feasibility_refresh_required"
                if idea_blocks_execution
                else (
                    plan_review_reason_code
                    if plan_attention_required
                    else plan_regeneration_reason_code
                )
                if plan_attention_required or plan_regeneration_required
                else "analysis_running"
                if analysis_running
                else "validated_analysis_ready"
                if analysis_complete
                else "analysis_outputs_require_validation"
                if analysis_outputs_available
                else "analysis_gate_blocked"
                if pipeline_attempt_blocked
                else "research_pipeline_required"
                if legacy_full_scaffold
                else "analysis_ready"
                if prepared_export_receipted and setup_ready
                else "active_export_or_setup_required"
            ),
        ),
        ResearchWorkflowStage(
            id="interpretation",
            label="Result interpretation",
            status=(
                "review_required"
                if analysis_complete or analysis_outputs_available
                else "blocked"
            ),
            owner="easyicu.research_agent.reporting",
            reason_code=(
                "evidence_bound_interpretation_ready"
                if analysis_complete or analysis_outputs_available
                else "validated_analysis_required"
            ),
        ),
        ResearchWorkflowStage(
            id="manuscript",
            label="Manuscript",
            status=(
                "review_required" if analysis_complete and has_manuscript else "blocked"
            ),
            owner="easyicu.research_agent.reporting",
            reason_code=(
                "manuscript_draft_ready_for_review"
                if analysis_complete and has_manuscript
                else "full_agent_manuscript_required"
            ),
        ),
    ]

    required = [row for row in stages if row.id != "idea"]
    # ``review_required`` is an outstanding human action, never a completed
    # stage.  Counting it as done made analysis-only runs appear as 7/7 even
    # though their interpretation and manuscript were still awaiting review.
    completed = sum(1 for row in required if row.status == "complete")
    if (
        eligibility_confirmation_required
        and not plan_attention_required
        and not analysis_complete
        and not analysis_outputs_available
    ):
        # A StudyContext change invalidates its cohort receipt by design.  The
        # new population must be confirmed before a stale-plan regeneration
        # action can be offered; otherwise the visible button only submits a
        # run that the launch owner must reject. A live, digest-matching
        # candidate is different: show its plan and evidence first and resolve
        # its choices in that review, without granting execution authority.
        next_stage = next(row for row in required if row.id == "setup")
    elif (
        plan_attention_required
        or plan_regeneration_required
        or (plan_generation_ready and not idea_blocks_execution)
    ):
        next_stage = next(row for row in required if row.id == "plan")
    elif not question_ready and not idea_accepted:
        # A blank project starts in divergent Idea Mining. The researcher may
        # still state a complete scientific question directly, but the progress
        # rail must not imply that question construction precedes exploration.
        next_stage = next(row for row in stages if row.id == "idea")
    else:
        next_stage = next(
            (row for row in required if row.status != "complete"),
            stages[-1],
        )
    next_action = next_stage.reason_code
    if all(row.status == "complete" for row in required):
        next_action = "human_review_and_reporting"
    return ResearchWorkflowSnapshot(
        current_stage=next_stage.id,
        next_action_code=next_action,
        planning_prerequisites_missing=planning_prerequisites_missing,
        missing_setup_fields=missing,
        stages=stages,
        completed_required_stages=completed,
        study_setup_receipt=project_study_setup_receipt(study_row),
        plan_review_summary=plan_review_summary,
        plan_execution_ready=plan_execution_ready,
        analysis_validation_retry_available=(analysis_validation_retry_available),
    )


def _enrich_plan_review(
    snapshot: ResearchWorkflowSnapshot,
    *,
    study: Mapping[str, Any],
    review: Mapping[str, Any],
) -> ResearchWorkflowSnapshot:
    """Attach conversational Plan review fields after workflow compilation."""

    payloads = review.get("artifact_payloads")
    payloads = payloads if isinstance(payloads, Mapping) else {}
    agent_plan = payloads.get("agent_plan.json")
    review_summary = snapshot.plan_review_summary
    if (
        isinstance(review_summary, Mapping)
        and isinstance(agent_plan, Mapping)
        # Once StudyContext changes, the immutable review is historical
        # evidence.  Its old authorization questions must not replace the
        # host-owned ``plan_configuration_superseded`` action or ask the user
        # to answer a decision that no longer exists in current authority.
        and snapshot.next_action_code == "plan_scientific_changes_required"
    ):
        questions = plan_decisions.pending_authorization_questions(
            study,
            review_summary.get("authorization_questions"),
        )
        enriched_questions = []
        for question in questions:
            item = dict(question) if isinstance(question, Mapping) else {}
            decision_context = plan_decisions.plan_decision_context(
                agent_plan, str(item.get("code") or "")
            )
            if decision_context:
                item["decision_context"] = decision_context
            if item.get("code") == "ADJUSTMENT_SET_NOT_USER_CONFIRMED":
                item["proposed_covariates"] = plan_decisions.proposed_adjustment_set(
                    agent_plan
                )
            enriched_questions.append(item)
        snapshot = snapshot.model_copy(
            update={
                **(
                    {"next_action_code": "plan_scientific_changes_required"}
                    if enriched_questions
                    else {}
                ),
                "plan_review_summary": {
                    **dict(review_summary),
                    "authorization_questions": enriched_questions,
                },
            }
        )
    plan_preview = project_plan_conversation_preview(agent_plan)
    if plan_preview:
        snapshot = snapshot.model_copy(
            update={"plan_conversation_preview": plan_preview}
        )
    return snapshot


def build_project_workflow_projection(
    *,
    study_context_id: Optional[str],
    study_override: Optional[Mapping[str, Any]] = None,
) -> ProjectWorkflowProjection:
    """Collect raw receipts once, compile once, and project only at the end."""

    clean_study_id = str(study_context_id or "").strip()
    if study_override is not None:
        study: Mapping[str, Any] = dict(study_override)
    elif clean_study_id:
        study = study_context_owner.get_context(clean_study_id) or {}
    else:
        study = study_context_owner.get_active_context() or {}
    if not clean_study_id:
        clean_study_id = str(study.get("id") or "").strip()

    registry = sources.load_registry()
    active_job: Optional[Mapping[str, Any]] = None
    active_job_id = str(study.get("active_job_id") or "").strip()
    if active_job_id:
        job = jobs.MANAGER.get(active_job_id)
        active_job = job.snapshot() if job else None
    rows = list_bound_run_history(
        study_context_id=clean_study_id or None,
        project_root=research_pipeline_project_root(clean_study_id or None),
        limit=10,
    )
    latest_run = workflow_authoritative_run(rows)
    plan_review_authority = (
        agent_pipeline_runs.pending_review(str(latest_run.get("run_id") or ""))
        if latest_run
        else None
    )

    review = (
        agent_runs.read_run_review(str(latest_run.get("project_dir") or ""))
        if latest_run else {}
    )

    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=registered_export_matches_study(study, registry),
        active_job=active_job,
        latest_run=latest_run,
        plan_review_authority=plan_review_authority,
        continuing_review_choices=plan_review_progress.has_pending_choices(
            study, latest_run or {}, review,
        ),
    )
    latest_run_outcome: Mapping[str, Any] = {"present": False}
    if latest_run:
        latest_run_outcome = project_run_outcome(review)
        snapshot = _enrich_plan_review(snapshot, study=study, review=review)

    return ProjectWorkflowProjection(
        workflow=snapshot,
        active_job=project_job(active_job),
        latest_run=latest_run_outcome,
    )


__all__ = [
    "ProjectWorkflowProjection",
    "ResearchWorkflowSnapshot",
    "ResearchWorkflowStage",
    "StudySetupReceipt",
    "WorkflowStatus",
    "active_export_matches_study",
    "registered_export_matches_study",
    "build_research_workflow_snapshot",
    "build_project_workflow_projection",
]
