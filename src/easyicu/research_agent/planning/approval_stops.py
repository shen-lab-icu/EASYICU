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
    FIRST_RESULT_DELIVERY_STOP,
    HOST_DELIVERY_CLARIFICATION,
    HOST_DELIVERY_CLARIFICATION_MESSAGE,
    HOST_DELIVERY_KINDS,
)
from ..schema import AnalysisPlan, ValidationFinding
from .population_compile import POPULATION_APPROVAL_STOPS
from .question_requirements import (
    QUESTION_REQUIREMENT_APPROVAL_STOPS,
    QUESTION_REQUIREMENT_STOP_CODES,
    question_requirements_on_plan_under_review,
)

DELIVERY_APPROVAL_STOPS = (HOST_DELIVERY_CLARIFICATION, FIRST_RESULT_DELIVERY_STOP)

PLAN_APPROVAL_STOPS: tuple[str, ...] = (
    *DELIVERY_APPROVAL_STOPS,
    *POPULATION_APPROVAL_STOPS.values(),
    "population_criterion_decision_unconfirmed",
    *(code for code in QUESTION_REQUIREMENT_STOP_CODES if code not in DELIVERY_APPROVAL_STOPS),
)


def first_result_delivery_stop(plan: AnalysisPlan) -> ValidationFinding | None:
    """Keep the desktop analysis profile sealed when the question needs the Writer."""
    return delivery_approval_stop(plan) if first_result(plan) else None


def delivery_approval_stop(plan: AnalysisPlan, *, article_allowed: bool = True) -> ValidationFinding | None:
    """Resolve a delivery decision before offering the plan for approval."""
    selection = plan.deliverables
    existing = {step.host_delivery_kind for step in plan.steps}
    requested = selection.requested if selection is not None else tuple(
        kind for kind in HOST_DELIVERY_KINDS if kind in existing
    )
    if selection is not None and not selection.confirmed:
        kinds = selection.check
        reason = HOST_DELIVERY_CLARIFICATION
        message = HOST_DELIVERY_CLARIFICATION_MESSAGE
    elif not article_allowed or first_result(plan):
        kinds = requested
        reason = FIRST_RESULT_DELIVERY_STOP
        message = (
            "Analysis-only first results cannot produce the requested article deliverables. "
            "Revise the request in the conversation, or start a separate article run."
        )
    else:
        kinds = tuple(kind for kind in requested if kind not in existing)
        reason = QUESTION_REQUIREMENT_APPROVAL_STOPS["not_covered"]
        message = "The plan lacks a typed owner for at least one selected deliverable."
    if not kinds:
        return None
    return ValidationFinding(
        validator="question_requirements", severity="error", message=message,
        detail={
            "reason": reason, "deliverable_kinds": list(kinds),
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
