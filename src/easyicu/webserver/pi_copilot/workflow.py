"""Typed project-level research workflow projection for Pi Copilot.

This module owns only orchestration state.  Idea Mining, extraction, Research
Agent execution, evidence gates, and manuscript artefacts keep their existing
owners; the Copilot shell receives an immutable, path-free projection of those
owners' receipts.
"""

from __future__ import annotations

from typing import Any, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

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
    stages: List[ResearchWorkflowStage] = Field(min_length=8, max_length=8)
    completed_required_stages: int = Field(ge=0, le=7)
    required_stage_count: Literal[7] = 7
    scientific_authority: Literal["EasyICU"] = "EasyICU"
    pi_role: Literal["conversation_and_orchestration"] = (
        "conversation_and_orchestration"
    )


def _has_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


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


def _setup_missing(
    study: Mapping[str, Any], *, active_export_present: bool
) -> List[str]:
    missing: List[str] = []
    checks = (
        ("question", bool(str(study.get("question") or "").strip())),
        (
            "data_source",
            active_export_present or _has_mapping(study.get("data_source")),
        ),
        ("cohort", _has_mapping(study.get("cohort"))),
        ("modules", bool(study.get("modules"))),
        ("outcome", bool(str(study.get("outcome") or "").strip())),
        ("time_window", _has_mapping(study.get("time_window"))),
        (
            "export_format",
            bool(str(study.get("export_format") or "").strip()),
        ),
        (
            "analysis_goal",
            bool(str(study.get("analysis_goal") or "").strip()),
        ),
    )
    for name, present in checks:
        if not present:
            missing.append(name)
    return missing


def build_research_workflow_snapshot(
    *,
    study: Optional[Mapping[str, Any]],
    active_export_present: bool,
    active_job: Optional[Mapping[str, Any]],
    latest_run: Optional[Mapping[str, Any]],
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
    idea_blocks_execution = bool(
        idea_accepted and idea_recommendation != "recommend"
    )
    missing = _setup_missing(
        study_row,
        active_export_present=bool(active_export_present),
    )
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
    has_manuscript = "manuscript_draft.json" in artifact_names
    full_run = run_type == "full"
    pipeline_run = run_engine == "easyicu.research_agent.pipeline"
    pipeline_receipt = "source_run_manifest.json" in artifact_names
    gate_status = str(run_row.get("gate_status") or "")
    run_blocked = gate_status == "blocked"
    pending_review_reason_codes = {
        str(item).strip()
        for item in (run_row.get("pending_review_reason_codes") or [])
        if str(item).strip()
    }
    plan_review_pending = bool(
        has_plan
        and str(run_row.get("run_status") or "") == "human_review_pending"
        and "operator_plan_approval_required" in pending_review_reason_codes
    )
    analysis_complete = bool(
        full_run
        and pipeline_run
        and pipeline_receipt
        and has_plan
        and has_evidence
        and has_outputs
        and gate_status == "analysis_only"
    )
    pipeline_attempt_blocked = bool(
        full_run and pipeline_run and pipeline_receipt and run_blocked
    )
    legacy_full_scaffold = bool(full_run and not pipeline_run and has_plan)

    stages = [
        ResearchWorkflowStage(
            id="question",
            label="Scientific question",
            status="complete" if question_ready else "ready",
            owner="easyicu.webserver.study_contexts",
            reason_code=("question_bound" if question_ready else "question_required"),
        ),
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
                else "blocked"
            ),
            owner="easyicu.webserver.ideas.mining",
            reason_code=(
                "idea_feasibility_refresh_required"
                if idea_blocks_execution
                else "idea_handoff_accepted"
                if idea_accepted
                else "idea_mining_available"
                if question_ready
                else "question_required"
            ),
        ),
        ResearchWorkflowStage(
            id="setup",
            label="Study setup",
            status="complete"
            if setup_ready
            else ("ready" if question_ready else "blocked"),
            owner="easyicu.webserver.study_contexts",
            reason_code=(
                "study_setup_complete" if setup_ready else "study_setup_incomplete"
            ),
        ),
        ResearchWorkflowStage(
            id="extraction",
            label="Feature extraction",
            status=(
                "complete"
                if active_export_present
                else "running"
                if extraction_running
                else "ready"
                if setup_ready
                else "blocked"
            ),
            owner="easyicu.webserver.routes.jobs",
            reason_code=(
                "active_export_ready"
                if active_export_present
                else "extraction_running"
                if extraction_running
                else "extraction_ready"
                if setup_ready
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
                if plan_review_pending
                else "complete"
                if has_plan
                else "running"
                if analysis_running
                else "ready"
                if active_export_present and setup_ready
                else "blocked"
            ),
            owner="easyicu.research_agent.planning",
            reason_code=(
                "idea_feasibility_refresh_required"
                if idea_blocks_execution
                else "operator_plan_approval_required"
                if plan_review_pending
                else "agent_plan_ready"
                if has_plan
                else "analysis_running"
                if analysis_running
                else "plan_ready"
                if active_export_present and setup_ready
                else "active_export_or_setup_required"
            ),
        ),
        ResearchWorkflowStage(
            id="analysis",
            label="Analysis and validation",
            status=(
                "blocked"
                if idea_blocks_execution
                else "blocked"
                if plan_review_pending
                else "complete"
                if analysis_complete and not run_blocked
                else "review_required"
                if pipeline_attempt_blocked
                else "running"
                if analysis_running
                else "ready"
                if active_export_present and setup_ready
                else "blocked"
            ),
            owner="easyicu.research_agent.pipeline",
            reason_code=(
                "idea_feasibility_refresh_required"
                if idea_blocks_execution
                else "operator_plan_approval_required"
                if plan_review_pending
                else "validated_analysis_ready"
                if analysis_complete and not run_blocked
                else "analysis_gate_blocked"
                if pipeline_attempt_blocked
                else "research_pipeline_required"
                if legacy_full_scaffold
                else "analysis_running"
                if analysis_running
                else "analysis_ready"
                if active_export_present and setup_ready
                else "active_export_or_setup_required"
            ),
        ),
        ResearchWorkflowStage(
            id="interpretation",
            label="Result interpretation",
            status=("review_required" if analysis_complete else "blocked"),
            owner="easyicu.research_agent.reporting",
            reason_code=(
                "evidence_bound_interpretation_ready"
                if analysis_complete
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
    completed = sum(
        1 for row in required if row.status in {"complete", "review_required"}
    )
    if plan_review_pending:
        completed -= 1
        next_stage = next(row for row in required if row.id == "plan")
    else:
        next_stage = next(
            (
                row
                for row in required
                if row.status not in {"complete", "review_required"}
            ),
            stages[-1],
        )
    next_action = next_stage.reason_code
    if all(row.status in {"complete", "review_required"} for row in required):
        next_action = "human_review_and_reporting"
    return ResearchWorkflowSnapshot(
        current_stage=next_stage.id,
        next_action_code=next_action,
        missing_setup_fields=missing,
        stages=stages,
        completed_required_stages=completed,
    )


__all__ = [
    "ResearchWorkflowSnapshot",
    "ResearchWorkflowStage",
    "WorkflowStatus",
    "active_export_matches_study",
    "build_research_workflow_snapshot",
]
