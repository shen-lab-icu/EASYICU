"""Fail-closed execution-graph gate for proposed plan revisions."""

from __future__ import annotations

from ..contracts.declared_product import primary_analysis_cohort_plan_findings
from ..contracts.runtime import ValidationFinding
from ..plan_utils import _typed_plan_dag_findings
from ..schema import AnalysisPlan, ResearchContext
from ..trajectory.plan_contract import trajectory_plan_dag_findings


def replan_candidate_contract_findings(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> list[ValidationFinding]:
    """Return execution-blocking graph defects in one proposed revision."""

    return [
        *_typed_plan_dag_findings(plan),
        *primary_analysis_cohort_plan_findings(plan=plan),
        *trajectory_plan_dag_findings(plan=plan, context=context),
    ]


def replan_candidate_rejection_finding(
    *,
    contract_errors: list[ValidationFinding],
    trigger: str,
    candidate_revision: int,
) -> ValidationFinding:
    """Summarize a rejected invalid candidate without activating its errors."""

    return ValidationFinding(
        validator="replanner",
        severity="warning",
        message=(
            "Rejected a proposed plan revision because its typed execution "
            "graph was invalid; retained the current plan."
        ),
        detail={
            "reason": "replan_candidate_execution_graph_invalid",
            "trigger": trigger,
            "candidate_revision": candidate_revision,
            "contract_findings": [
                {
                    "validator": finding.validator,
                    "message": finding.message,
                    "detail": finding.detail,
                }
                for finding in contract_errors
            ],
        },
    )


__all__ = [
    "replan_candidate_contract_findings",
    "replan_candidate_rejection_finding",
]
