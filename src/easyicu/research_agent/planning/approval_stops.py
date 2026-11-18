"""The reasons a reviewed plan cannot be approved, gathered from their owners.

A plan stops for approval when its population cannot be applied as stated
(``population_compile.POPULATION_APPROVAL_STOPS``) or when the research
question asks for an analysis the plan does not answer
(``question_requirements.QUESTION_REQUIREMENT_STOP_CODES``).  Every reader
of a plan under review -- the conversation's workflow, the run projection, the
agent route -- spreads this one tuple, so a stop an owner adds reaches all of
them.  The order is the order a reader names them in: the population first.

The plan phase judges these stops on the compiled plan, and the pipeline then
shapes it, so where review requests are derived (``orchestration.workflow``)
each owner judges its stops again on the plan the requests offer
(:func:`judged_on_plan_under_review`).  The question requirements owner does;
the population owner's stops are still judged on the compiled plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..contracts.claim_ceiling import first_result
from ..contracts.product_files import (
    DeliveryRequest,
    HOST_DELIVERY_CLARIFICATION,
    HOST_DELIVERY_CLARIFICATION_MESSAGE,
    HOST_DELIVERY_KINDS,
    RUN_DELIVERABLES,
    delivery_requests,
)
from ..schema import AnalysisPlan, ValidationFinding
from .population_compile import POPULATION_APPROVAL_STOPS
from .question_requirements import (
    QUESTION_REQUIREMENT_STOP_CODES,
    question_requirements_on_plan_under_review,
)

FIRST_RESULT_DELIVERY_STOP = "first_result_delivery_requires_article"
DELIVERY_APPROVAL_STOPS = (HOST_DELIVERY_CLARIFICATION, FIRST_RESULT_DELIVERY_STOP)

PLAN_APPROVAL_STOPS: tuple[str, ...] = (
    *DELIVERY_APPROVAL_STOPS,
    *POPULATION_APPROVAL_STOPS.values(),
    "population_criterion_decision_unconfirmed",
    *QUESTION_REQUIREMENT_STOP_CODES,
)


def first_result_delivery_stop(plan: AnalysisPlan, question: str | None = None) -> ValidationFinding | None:
    """Keep the desktop analysis profile sealed when the question needs the Writer."""
    return delivery_approval_stop(plan, question) if first_result(plan) else None


def delivery_approval_stop(plan: AnalysisPlan, question: str | None = None) -> ValidationFinding | None:
    """Resolve a delivery decision before offering the plan for approval."""
    requests = delivery_requests(question or plan.research_question)
    ambiguous = [kind for kind in RUN_DELIVERABLES if requests.get(kind) is DeliveryRequest.AMBIGUOUS]
    if ambiguous:
        return ValidationFinding(
            validator="question_requirements",
            severity="error",
            message=HOST_DELIVERY_CLARIFICATION_MESSAGE,
            detail={
                "reason": HOST_DELIVERY_CLARIFICATION, "deliverable_kinds": ambiguous,
                "human_review_required": True, "approval_allowed": False,
            },
        )
    if not first_result(plan):
        return None
    existing = {step.host_delivery_kind for step in plan.steps}
    kinds = [
        kind for kind in HOST_DELIVERY_KINDS
        if kind in existing or requests.get(kind) is DeliveryRequest.REQUESTED
    ]
    if not kinds:
        return None
    return ValidationFinding(
        validator="question_requirements",
        severity="error",
        message=(
            "Analysis-only first results cannot produce the requested article deliverables. "
            "Revise the request in the conversation, or start a separate article run."
        ),
        detail={
            "reason": FIRST_RESULT_DELIVERY_STOP, "deliverable_kinds": kinds,
            "human_review_required": True, "approval_allowed": False,
        },
    )


def judged_on_plan_under_review(
    findings: Iterable[ValidationFinding],
    *,
    plan: AnalysisPlan,
    run_dir: Path,
    question: str | None = None,
) -> list[ValidationFinding]:
    """The plan phase's findings, with each owner's stops judged on the plan offered for review."""

    return question_requirements_on_plan_under_review(
        findings, plan=plan, run_dir=run_dir, question=question,
    )


__all__ = ["PLAN_APPROVAL_STOPS", "judged_on_plan_under_review"]
