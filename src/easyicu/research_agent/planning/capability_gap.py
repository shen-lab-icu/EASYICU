"""Whether a Planner's capability gap is true of the study (host check).

A Planner that cannot express a design element the question needs declares a
capability gap instead of drafting steps that quietly answer another question
(``progressive_contract.ProgressiveCapabilityGap``).  A gap stops planning, so
the host first checks the claim against the study's typed context: a claim the
context contradicts goes back to the Planner as a repair. Concrete requirements
are checked here. The host has no typed evidence for or against an
unsupported estimand or unnamed design element. An unnamed design gap gets
one confirmation against typed host facts before it stops as unverifiable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, NamedTuple, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..authority.declared_levels import closed_planning_levels_for
from ..schema import ResearchContext
from ..trajectory.plan_contract import trajectory_context_is_bound
from .family_spec.request import sealed_trajectory_suite_coordinates
from .progressive_contract import (
    CapabilityGapRequirement, ProgressiveCapabilityGap, ProgressivePlanCompileError,
    ProgressivePlanOutline,
)

CapabilityGapVerification = Literal["verified", "unverified", "unverifiable"]


class CapabilityGapCheck(NamedTuple):
    verification: CapabilityGapVerification
    #: What the context shows, for the Planner's repair or the run record.
    fact: str


class CapabilityGapDiagnostic(BaseModel):
    """Host-owned stopped-outline receipt, for private planning artifacts only.

    This is not part of the Planner's output schema or the browser projection.
    Rationale is validated planning text, as in an outline checkpoint, never
    an exception message or a raw provider response.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement: CapabilityGapRequirement
    concept: str | None = Field(default=None, min_length=1, max_length=128)
    verification: CapabilityGapVerification
    fact: str = Field(min_length=1, max_length=2_000)
    rationale: str = Field(min_length=8, max_length=1_200)


@dataclass
class CapabilityGapConfirmation:
    """One confirmation, charged to the existing outline retry owner."""

    progress_callback: Callable[[Any], None] | None = None
    requested: bool = False
    retry_available: bool = False

    def on_progress(self, event: Any) -> None:
        # This also accounts for transport resends without creating a second
        # budget. A gap on the final attempt must remain a definite stop.
        self.retry_available = event.attempt < event.total_attempts
        if self.progress_callback is not None:
            self.progress_callback(event)

    def check(self, outline: ProgressivePlanOutline, check: CapabilityGapCheck) -> None:
        if (
            check.verification == "unverifiable"
            and outline.capability_gap == "design_element_unsupported"
            and outline.capability_gap_concept is None
            and not self.requested
            and self.retry_available
        ):
            self.requested = True
            raise ProgressivePlanCompileError(
                "progressive_capability_gap_confirmation_required",
                "Confirm the unnamed design-element gap against the host facts below. "
                "Availability alone does not prove the whole design is expressible. "
                "If the needed element is one of these, draft the steps for the original "
                "question; otherwise keep the gap and name in rationale what the question "
                "asks that none of the offered families and typed elements expresses. "
                "Do not drop a requirement or change the question to fit a family.",
                path="capability_gap", cause_code=outline.capability_gap,
            )


def capability_gap_stop(
    outline: ProgressivePlanOutline, check: CapabilityGapCheck,
) -> ProgressivePlanCompileError:
    """Bind the typed stop to its private, validated planning receipt."""

    gap = outline.declared_capability_gap()
    if gap is None:
        raise ValueError("a capability-gap stop requires a declared gap")
    diagnostic = CapabilityGapDiagnostic(
        requirement=gap.requirement, concept=gap.concept,
        verification=check.verification, fact=check.fact, rationale=outline.rationale,
    )
    stop = ProgressivePlanCompileError(
        "progressive_capability_gap",
        "the question needs something no offered family can express "
        f"({gap.requirement}, {check.verification}): {outline.rationale}",
        path="capability_gap", cause_code=gap.requirement,
        concepts=(gap.concept,) if gap.concept else (),
        element=gap.requirement,
        question_required=True,
        findings=[diagnostic.model_dump(mode="json", exclude={"rationale"})],
    )
    stop.easyicu_planning_capability_gap = diagnostic
    return stop


def capability_gap_confirmation_facts(
    context: ResearchContext, *, offered_families: Sequence[str],
) -> str:
    """Project existing context owners, without data rows or guessed semantics.

    Keep the outer materialization windows even when a column has no named
    analysis_window. They bound its observations without declaring time zero.
    This goes in the retry reminder, outside the validator-message clipping.
    """

    from .adjustment_authority import STATED_VARIABLE_FIELDS, context_stated_variables
    from .analysis_types import infer_analysis_type_with_basis

    def column(name: str | None) -> dict:
        variable = context.variable(name) if name else None
        return {
            "column": name,
            "metadata": variable.model_dump(mode="json", include={
                "role", "dtype", "source_concept", "derived_from_concepts",
                "analysis_window", "analysis_window_role", "aggregation_default",
                "allowed_aggregations", "unit_normalization", "temporal_resolution",
            }) if variable else None,
        }

    family, basis = infer_analysis_type_with_basis(context)
    facts = {
        "family": family.key, "family_basis": basis,
        "offered_families": list(offered_families),
        "exposure": column(context.primary_exposure),
        "outcome": column(context.target_outcome),
        "endpoint": context.endpoint.model_dump(mode="json") if context.endpoint else None,
        "time_windows": [window.model_dump(mode="json", exclude={"rationale"}) for window in context.time_windows],
        "temporal_constraints": [item.model_dump(mode="json", exclude={"raw_text", "executable_repr"})
                                 for item in context.temporal_constraints],
        **{field: [item.model_dump(mode="json") for item in context_stated_variables(context, field)]
           for field in STATED_VARIABLE_FIELDS},
    }
    return "Host facts for capability-gap confirmation:\n" + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))


def threshold_grouping_expressible(context: ResearchContext, concept: str) -> bool:
    """Whether a typed exposure grouping could group ``concept`` by thresholds.

    The named hook for ExposureGroupSpec.  Until that owner lands nothing can,
    so a thresholds claim stands on the variable's closed levels alone; then
    the claim holds only when the spec cannot group this concept either.
    """

    del context, concept
    return False


#: A stated reading no source concept was matched to proves no gap: the host
#: could not name the variable, which is not the source lacking it.
_UNRESOLVED_FACT = (
    "no source concept was matched to the stated words {words!r}, which does not show "
    "the source lacks the variable; use the roster column that is this variable, or "
    "plan without it, and scientific review shows the unresolved requirement"
)


def check_capability_gap(
    gap: ProgressiveCapabilityGap,
    *,
    context: ResearchContext,
    planning_contract_context: str = "",
) -> CapabilityGapCheck:
    """Check one declared gap against the study's typed context."""

    if gap.requirement == "design_element_unsupported" and gap.concept is not None:
        # Source, timing and family owners can prove a named requirement gap.
        # A variable can have both roles; a verified gap in either still stops.
        from .adjustment_authority import (
            context_stated_adjustments, context_stated_variables, stated_adjustment_after_time_zero,
            stated_adjustment_columns,
        )
        from .analysis_types import infer_analysis_type
        from .requirement_coverage import family_describes_by_group, subgroup_actions_for_family

        adjustment = next((item for item in context_stated_adjustments(context)
                           if (item.concept or item.variable) == gap.concept), None)
        if adjustment is not None:
            if adjustment.source_status == "unavailable":
                return CapabilityGapCheck("verified", f"the bound source lacks {gap.concept!r}")
            if stated_adjustment_after_time_zero(context, adjustment):
                return CapabilityGapCheck("verified", f"{gap.concept!r} is measured after typed time zero")
        subgroup = next((item for item in context_stated_variables(context, "stated_subgroups")
                         if (item.concept or item.variable) == gap.concept), None)
        if subgroup is not None:
            if subgroup.source_status == "unavailable":
                return CapabilityGapCheck("verified", f"the bound source lacks subgroup {gap.concept!r}")
            family = infer_analysis_type(context).key
            if not subgroup_actions_for_family(family) and not family_describes_by_group(family):
                return CapabilityGapCheck("verified", f"{family} has no subgroup owner for {gap.concept!r}")
            if subgroup.source_status == "unresolved":
                return CapabilityGapCheck("unverified", _UNRESOLVED_FACT.format(words=subgroup.words))
            if subgroup.source_status != "available" or subgroup.concept is None:
                return CapabilityGapCheck("unverifiable", f"source availability for subgroup {gap.concept!r} needs resolution")
            return CapabilityGapCheck("unverified", f"{family} offers a subgroup owner and the source supplies {gap.concept!r}")

        if adjustment is None:
            return CapabilityGapCheck("unverified", f"{gap.concept!r} is not a stated adjustment or subgroup")
        if adjustment.source_status == "available" or stated_adjustment_columns(context, adjustment):
            return CapabilityGapCheck(
                "unverified", f"the study offers {gap.concept!r} by time zero"
            )
        if adjustment.source_status == "unresolved":
            return CapabilityGapCheck("unverified", _UNRESOLVED_FACT.format(words=adjustment.words))
        return CapabilityGapCheck("unverifiable", f"source availability for {gap.concept!r} needs resolution")

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
    "CapabilityGapConfirmation",
    "CapabilityGapDiagnostic",
    "CapabilityGapVerification",
    "capability_gap_confirmation_facts",
    "capability_gap_stop",
    "check_capability_gap",
    "threshold_grouping_expressible",
]
