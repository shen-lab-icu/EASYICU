"""Synthetic pending plan-review records shared by webserver tests."""

from __future__ import annotations

from pathlib import Path

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewPending,
    HumanReviewRequest,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def pending_plan_review(run_dir: Path) -> HumanReviewPending:
    plan = AnalysisPlan(
        research_question="Is the grouped value associated with death?",
        steps=[
            AnalysisStep(
                step_id="primary",
                intent="Compare the groups",
                method="descriptive",
                inputs=[],
                expected_outputs=["table:groups"],
            )
        ],
    )
    authority = PlanReviewAuthority.create(plan=plan)
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="a" * 64,
        payload={
            "reason": "operator_plan_approval_required",
            "plan_review_authority": authority.model_dump(mode="json"),
        },
    )
    return HumanReviewPending(
        run_id="run-groupings",
        thread_id="thread-groupings",
        run_dir=str(run_dir),
        requests=(request,),
    )
