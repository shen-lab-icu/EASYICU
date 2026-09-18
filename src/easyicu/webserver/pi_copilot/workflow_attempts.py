"""Read-only attempt facts, distinct from the plan that retains authority."""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict

from .contracts import PLAN_RESUME_OFFER_GATE_REASONS
from .run_authority import workflow_authoritative_run


class PreservedPlanFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    candidate_run_id: str
    reason: Literal["provider_unavailable", "planner_budget_exhausted", "preparation_failed"]
    checkpoint_resume_available: bool


def preserved_plan_failure(
    *,
    latest_attempt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    study_id: str,
    scientific_configuration_sha256: str,
) -> PreservedPlanFailure | None:
    """Keep a newer failed preparation visible without transferring authority.

    A checkpoint flag offers a route only. Launch still validates the owned
    path, input authority, configuration and complete checkpoint hash chain.
    """

    run_id = latest_attempt.get("run_id")
    candidate_id = candidate.get("run_id")
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}", value) is None
        for value in (run_id, candidate_id)
    ):
        return None
    if (
        run_id == candidate_id
        or latest_attempt.get("engine") != "easyicu.research_agent.pipeline"
        or latest_attempt.get("study_id") != study_id
        or candidate.get("study_id") != study_id
        or latest_attempt.get("scientific_configuration_sha256")
        != scientific_configuration_sha256
        or workflow_authoritative_run([latest_attempt, candidate]) is not candidate
    ):
        return None
    reason_code = str(latest_attempt.get("gate_reason") or "")
    reason = {
        "research_pipeline_planner_provider_unavailable": "provider_unavailable",
        "research_pipeline_planner_efficiency_budget_exhausted": "planner_budget_exhausted",
    }.get(reason_code, "preparation_failed")
    return PreservedPlanFailure(
        run_id=run_id,
        candidate_run_id=candidate_id,
        reason=reason,
        checkpoint_resume_available=(
            reason_code in PLAN_RESUME_OFFER_GATE_REASONS
            and latest_attempt.get("development_planner_checkpoint_available") is True
        ),
    )


def research_job_has_execution_progress(job: Mapping[str, Any]) -> bool:
    """Job kind alone does not prove that a research plan reached execution.

    Read host lifecycle stage codes, never their human/model-facing labels.
    Subsequent audit events cannot move an executing job back into planning.
    This projection does not grant execution or count an analysis as complete.
    """

    events = job.get("events")
    if not isinstance(events, list):
        return False
    return any(
        isinstance(event, Mapping)
        and event.get("type") == "progress"
        and isinstance(event.get("step"), str)
        and event.get("step") in {"step", "coder", "runner", "runner_repair", "critic", "visual_qa"}
        and isinstance(event.get("status", "running"), str)
        and event.get("status", "running") in {"running", "complete"}
        for event in events
    )


def research_job_has_report_repair_progress(job: Mapping[str, Any]) -> bool:
    """A report-only lifecycle event is neither planning nor new execution."""

    events = job.get("events")
    if not isinstance(events, list):
        return False
    return any(
        isinstance(event, Mapping)
        and event.get("type") == "progress"
        and event.get("step") == "report_repair"
        and isinstance(event.get("status", "running"), str)
        and event.get("status", "running") in {"running", "complete"}
        for event in events
    )
