"""Whether a Planner's capability gap is true of the study (host check).

A Planner that cannot express a design element the question needs declares a
capability gap instead of drafting steps that quietly answer another question
(``progressive_contract.ProgressiveCapabilityGap``).  A gap stops planning, so
the host first checks the claim against the study's typed context: a claim the
context contradicts goes back to the Planner as a repair.  Three requirements
are checkable here.  The host has no typed evidence for or against an
unsupported estimand or design element, so those stop as unverifiable, and the
run record says so.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from ..authority.declared_levels import closed_planning_levels_for
from ..schema import ResearchContext
from ..trajectory.plan_contract import trajectory_context_is_bound
from .family_spec.request import sealed_trajectory_suite_coordinates
from .progressive_contract import ProgressiveCapabilityGap

CapabilityGapVerification = Literal["verified", "unverified", "unverifiable"]


class CapabilityGapCheck(NamedTuple):
    verification: CapabilityGapVerification
    #: What the context shows, for the Planner's repair or the run record.
    fact: str


def threshold_grouping_expressible(context: ResearchContext, concept: str) -> bool:
    """Whether a typed exposure grouping could group ``concept`` by thresholds.

    The named hook for ExposureGroupSpec.  Until that owner lands nothing can,
    so a thresholds claim stands on the variable's closed levels alone; then
    the claim holds only when the spec cannot group this concept either.
    """

    del context, concept
    return False


def check_capability_gap(
    gap: ProgressiveCapabilityGap,
    *,
    context: ResearchContext,
    planning_contract_context: str = "",
) -> CapabilityGapCheck:
    """Check one declared gap against the study's typed context."""

    if gap.requirement == "levels_from_thresholds_unavailable":
        variables = {variable.name: variable for variable in context.variables}
        concept = str(gap.concept or "")
        if concept not in variables:
            return CapabilityGapCheck(
                "unverified", f"{concept!r} is not a variable of this study"
            )
        levels = closed_planning_levels_for(name=concept, variables=variables)
        if len(levels) >= 2:
            return CapabilityGapCheck(
                "unverified",
                f"{concept!r} already has {len(levels)} closed levels to compare",
            )
        if threshold_grouping_expressible(context, concept):
            return CapabilityGapCheck(
                "unverified",
                f"{concept!r} can be grouped by thresholds with a typed exposure grouping",
            )
        return CapabilityGapCheck(
            "verified",
            f"{concept!r} has no closed levels, and nothing groups it by thresholds",
        )
    if gap.requirement == "multiple_sources_required":
        sources = {
            str(value).strip()
            for value in (context.cohort.database, *context.cross_database_validation)
            if str(value or "").strip()
        }
        if len(sources) >= 2:
            return CapabilityGapCheck(
                "unverified", f"this study uses {len(sources)} sources"
            )
        return CapabilityGapCheck("verified", "this study uses one source")
    if gap.requirement == "longitudinal_representation_unavailable":
        if (
            trajectory_context_is_bound(context)
            or sealed_trajectory_suite_coordinates(planning_contract_context)
            is not None
        ):
            return CapabilityGapCheck(
                "unverified", "this study has a longitudinal representation"
            )
        return CapabilityGapCheck(
            "verified", "this study has no longitudinal representation"
        )
    return CapabilityGapCheck(
        "unverifiable", f"the host has no typed evidence about {gap.requirement}"
    )


__all__ = [
    "CapabilityGapCheck",
    "CapabilityGapVerification",
    "check_capability_gap",
    "threshold_grouping_expressible",
]
