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

from ..schema import AnalysisPlan, ValidationFinding
from .population_compile import POPULATION_APPROVAL_STOPS
from .question_requirements import (
    QUESTION_REQUIREMENT_STOP_CODES,
    question_requirements_on_plan_under_review,
)

PLAN_APPROVAL_STOPS: tuple[str, ...] = (
    *POPULATION_APPROVAL_STOPS.values(),
    *QUESTION_REQUIREMENT_STOP_CODES,
)


def judged_on_plan_under_review(
    findings: Iterable[ValidationFinding],
    *,
    plan: AnalysisPlan,
    run_dir: Path,
) -> list[ValidationFinding]:
    """The plan phase's findings, with each owner's stops judged on the plan offered for review."""

    return question_requirements_on_plan_under_review(
        findings, plan=plan, run_dir=run_dir
    )


__all__ = ["PLAN_APPROVAL_STOPS", "judged_on_plan_under_review"]
