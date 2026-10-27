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
import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, NamedTuple, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..authority.declared_levels import closed_planning_levels_for
from ..schema import ConceptDescriptor, ResearchContext
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
    if check.verification == "unverified":
        # A saved outline bypasses response parsing. It must still never turn
        # a contradicted claim into a terminal capability-gap receipt.
        raise ProgressivePlanCompileError(
            "progressive_capability_gap_claim_unverified",
            f"the declared capability gap does not hold: {check.fact}; "
            "draft the steps the offered families can express, or declare a gap the study shows",
            path="capability_gap", cause_code=gap.requirement,
        )
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
        "fixed_window_association": fixed_window_association_facts(context),
        "stay_level_representation": stay_level_representation_facts(context),
        "longitudinal_representation": longitudinal_representation_facts(context),
        **{field: [item.model_dump(mode="json") for item in context_stated_variables(context, field)]
           for field in STATED_VARIABLE_FIELDS},
    }
    return "Host facts for capability-gap confirmation:\n" + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))


def longitudinal_representation_facts(context: ResearchContext) -> dict[str, Any] | None:
    """The per-timepoint representation a trajectory-class question is planned on.

    A study planned on a registered source's metadata holds no row yet: its
    run materializes each stay's timed values of the trajectory design's
    coordinates as a long panel over the design's own window, which the outer
    stay-summary window does not bound (``acquisition``, the web trajectory
    projection).  So the planning catalog lacking a time column is not the
    source lacking repeated measurements.  ``None`` unless the question asks
    for trajectories, the family router reads it as discovering their classes,
    the study is planned on metadata, the window the question states is one
    the signed fixed-window owner counts, and the roster holds a time-varying
    variable that owner can count eligible windows on
    (``contracts.trajectory_design``).  Which variables the question asks
    about stays the Planner's reading; the host names the roster's candidates
    and the exposure its reader bound.
    """

    from ..concept_availability import variable_source_unavailability
    from ..contracts.concept_values import names_without_values
    from ..contracts.trajectory_design import (
        TRAJECTORY_PRIMARY_ACTION,
        executable_trajectory_coordinates,
        time_varying_role,
        trajectory_window_design,
    )
    from ..research_context.stay_events import stay_outcome_columns
    from ..research_context.temporal_semantics import trajectory_window_statements
    from .analysis_types import infer_analysis_type, longitudinal_trajectory_requested
    from .exposure_group_compile import planned_on_metadata

    if (
        not longitudinal_trajectory_requested(context)
        or not planned_on_metadata(context)
        or infer_analysis_type(context).key != "trajectory_clustering"
    ):
        return None
    window = trajectory_window_design(
        trajectory_window_statements(str(context.research_question or ""))
    )
    if window.get("executable") is not True:
        return None
    excluded = {*context.cohort.id_columns, *stay_outcome_columns(context)}
    unavailable = names_without_values(context)
    candidates = [
        variable.name
        for variable in context.variables
        if variable.name not in excluded
        and time_varying_role(variable.role)
        and not variable_source_unavailability(variable, context.cohort.database)
        and not {variable.name, variable.source_concept, *variable.derived_from_concepts} & unavailable
        and executable_trajectory_coordinates([variable.name])
    ]
    if not candidates:
        return None
    named = [name for name in (context.primary_exposure,) if name in candidates]
    span = (
        f"{window['window_start_hours']}-{window['window_end_hours']} h after ICU admission "
        f"in {window['grid_width_hours']}-hour windows"
    )
    window_clause = (
        f"over {span}, the window the question states"
        if window["source"] == "question"
        else f"over its default {span}; the question states no trajectory window"
    )
    return {
        "analysis_family": "trajectory_clustering",
        "requires_longitudinal_representation": True,
        "representation": "per_timepoint_panel_materialized_by_the_run",
        "time_varying_variables": candidates,
        "question_reading_variables": named,
        "trajectory_window": window,
        "planning_rule": (
            "This study is planned on the source's metadata; its run materializes each "
            "stay's timed values of the trajectory design's coordinates as a "
            "per-timepoint panel over the design's window, which the outer stay-summary "
            "window does not bound. Time-varying variables the signed fixed-window owner "
            f"can model: {', '.join(candidates)}"
            + (f" (the question's reading names {', '.join(named)})" if named else "")
            + ". Plan trajectory_clustering with one primary "
            f"{TRAJECTORY_PRIMARY_ACTION} step whose inputs are the time-varying "
            "variables the question asks about, and no outcome or one-value-per-stay "
            "variable; the host compiles them into the signed fixed-window trajectory "
            f"design {window_clause}. An analysis the question asks of the frozen "
            "classes that no offered step expresses (for example assigning a stay to a "
            "class from features available earlier) goes in cannot_prove and the "
            "rationale; the plan on the signed suite states it as a question "
            "requirement for review."
        ),
    }


def stay_level_representation_facts(context: ResearchContext) -> dict[str, Any] | None:
    """Prove the question's required values fit one record per ICU stay.

    Only bound question variables and required adjustment/subgroup receipts
    participate; an optional roster column does not add a requirement. Read
    source prohibitions, materialization receipts and stay-value kinds through
    their owners. Neither a time window in prose nor a summary-like suffix on
    prepared data establishes a summary. This is representation evidence, not
    approval of time alignment, prediction eligibility or patient dependence.
    """

    from ..concept_availability import variable_source_unavailability
    from ..contracts.concept_values import names_without_values
    from .adjustment_authority import (
        AdjustmentSetAuthority, STATED_VARIABLE_FIELDS, context_stated_variables,
        stated_adjustment_columns,
    )
    from .analysis_types import infer_analysis_type, longitudinal_trajectory_requested

    if longitudinal_trajectory_requested(context) or len(context.cohort.id_columns) != 1:
        return None
    required = {name for name in (context.primary_exposure, context.target_outcome) if name}
    if not required:
        return None
    authority = AdjustmentSetAuthority.from_context(context)
    if authority.selection == "exact":
        required.update(authority.operational_covariates)
    for field in STATED_VARIABLE_FIELDS:
        for item in context_stated_variables(context, field):
            columns = stated_adjustment_columns(context, item)
            if not columns and item.concept:
                # Representation sufficiency does not impose model-term
                # eligibility: a grouping may itself be a stay's outcome.
                columns = tuple(variable.name for variable in context.variables
                                if item.concept in {variable.name, variable.source_concept})
            if item.source_status != "available" or not columns:
                return None
            required.update(columns)
    unavailable = names_without_values(context)
    stay_values: list[str] = []
    summaries: list[dict[str, Any]] = []
    for name in sorted(required):
        variable = context.variable(name)
        if (
            variable is None
            or variable_source_unavailability(variable, context.cohort.database)
            or {variable.name, variable.source_concept, *variable.derived_from_concepts} & unavailable
        ):
            return None
        representation = _stay_variable_representation(context, variable)
        if representation is None:
            return None
        if representation:
            summaries.append(representation)
        else:
            stay_values.append(name)
    family = infer_analysis_type(context).key
    summary_labels = [
        f"{item['column']} ({item['summary']} over {item['window_hours'][0]:g}-{item['window_hours'][1]:g} h "
        f"from {item['anchor']})"
        for item in summaries
    ]
    return {
        "analysis_family": family,
        "stay_level_variables": stay_values, "window_summary_variables": summaries,
        "requires_longitudinal_representation": False,
        "planning_rule": (
            f"{family} can use stay-level variables {stay_values} and fixed-window summaries {summary_labels} "
            "without repeated measurements. Preserve the question's outcome, grouping and predictors. "
            "Check time alignment, eligibility and patient dependence separately; "
            "their unresolved status does not establish a longitudinal representation requirement."
        ),
    }


def _stay_variable_representation(context: ResearchContext, variable: ConceptDescriptor) -> dict[str, Any] | None:
    """A supplied summary receipt, an empty stay-value receipt, or no proof."""

    from ..cohort.materializer import EVENT_SUMMARIES, summary_column_name
    from ..research_context.materialization_window import column_materialized_window
    from ..research_context.stay_events import column_kind, stay_outcome_columns
    from .exposure_group_compile import grouping_sources, planned_on_metadata

    concept = variable.source_concept or variable.name
    summary = next((
        item for item in sorted(EVENT_SUMMARIES)
        if variable.unit_normalization == f"window_numeric_{item}"
        or (planned_on_metadata(context) and variable.name == summary_column_name(concept, item))
    ), None)
    if summary is not None:
        window = column_materialized_window(context, variable.name)
        if (
            window is None or window.anchor is None
            or window.start_hours is None or window.end_hours is None
            or not all(math.isfinite(hour) for hour in (window.start_hours, window.end_hours))
            or not window.start_hours < window.end_hours
        ):
            return None
        if variable.unit_normalization != f"window_numeric_{summary}" and not any(
            source.concept == concept and summary in source.summaries
            and source.window == (window.start_hours, window.end_hours)
            for source in grouping_sources(context)
        ):
            return None
        return {"column": variable.name, "summary": summary, "anchor": window.anchor,
                "window_hours": [window.start_hours, window.end_hours]}
    if variable.unit_normalization == "stay_level_unique_value" or column_kind(
        variable, column=variable.name, concept=concept, outcomes=stay_outcome_columns(context),
    ) != "window_summary":
        return {}
    return None


def fixed_window_association_facts(context: ResearchContext) -> dict[str, Any] | None:
    """A window summary can propose a landmark without a trajectory.

    This proves representation sufficiency, not a runnable risk set. Read the
    materialization/grouping owners, including their metadata-only projection;
    a suggestive column suffix or a time window in prose proves nothing. Keep
    death timing and patient grouping separate from measurement representation.
    Neither a missing death time nor an unbound patient identifier establishes
    that the question requires repeated measurements.
    """

    from ..cohort.materializer import summary_column_name
    from ..concept_availability import variable_source_unavailability
    from ..contracts.concept_values import names_without_values
    from ..contracts.patient_grouping_need import PATIENT_GROUPING_STATUSES, PATIENT_GROUPING_STATUS_KEY
    from ..research_context.event_time_semantics import recorded_event_time_semantics
    from ..research_context.materialization_window import column_materialized_window
    from .analysis_types import infer_analysis_type, longitudinal_trajectory_requested
    from .exposure_group_compile import grouping_sources, planned_on_metadata

    if longitudinal_trajectory_requested(context) or infer_analysis_type(context).key != "association_study":
        return None
    exposure = context.variable(context.primary_exposure) if context.primary_exposure else None
    outcome = context.variable(context.target_outcome) if context.target_outcome else None
    if exposure is None or outcome is None or len(context.cohort.id_columns) != 1:
        return None
    if any(variable_source_unavailability(item, context.cohort.database) for item in (exposure, outcome)):
        return None
    unavailable = names_without_values(context)
    if any({item.name, item.source_concept, *item.derived_from_concepts} & unavailable for item in (exposure, outcome)):
        return None
    variables = {item.name: item for item in context.variables}
    levels = closed_planning_levels_for(name=outcome.name, variables=variables)
    endpoint = context.endpoint
    if len(levels) != 2 and not (endpoint and endpoint.name == outcome.name and endpoint.kind == "binary"):
        return None
    window = column_materialized_window(context, exposure.name)
    if (
        window is None or window.anchor != "icu_admission"
        or window.start_hours is None or window.end_hours is None
        or not all(math.isfinite(hour) for hour in (window.start_hours, window.end_hours))
        or not 0 <= window.start_hours < window.end_hours
    ):
        return None
    concept = exposure.source_concept or exposure.name
    summary = next((
        summary for source in grouping_sources(context)
        if source.concept == concept and source.window == (window.start_hours, window.end_hours)
        for summary in source.summaries
        if exposure.unit_normalization == f"window_numeric_{summary}"
        or (planned_on_metadata(context) and exposure.name == summary_column_name(concept, summary))
    ), None)
    if summary is None:
        return None
    semantics = recorded_event_time_semantics(context) or {}
    grouping = (context.cohort.provenance or {}).get(PATIENT_GROUPING_STATUS_KEY)
    return {
        "analysis_family": "association_study", "exposure": exposure.name, "summary": summary,
        "window_hours": [window.start_hours, window.end_hours],
        "proposed_time_zero_hours": window.end_hours,
        "requires_longitudinal_representation": False,
        "death_time_semantics": semantics.get("death_time"),
        "patient_grouping_status": grouping if grouping in PATIENT_GROUPING_STATUSES else None,
        "planning_rule": (
            f"association_study can propose a fixed-landmark association for {exposure.name}, "
            f"a {summary} summary over {window.start_hours:g}-{window.end_hours:g} h after ICU admission; "
            "it does not require repeated measurements. "
            f"Explicitly propose time zero at {window.end_hours:g} h, with deaths before the landmark excluded "
            "and rows still under observation. Preserve the outcome, subgroup and adjustment requirements. "
            "Verify landmark eligibility separately before execution; do not invent death times or "
            "equate ICU length of stay with survival. If eligibility cannot be established, name that "
            "design_element_unsupported gap. Never silently substitute an admission-time estimand."
        ),
    }


def threshold_grouping_expressible(context: ResearchContext, concept: str) -> bool:
    """Whether a typed exposure grouping could group ``concept`` by thresholds.

    Read the grouping compiler's source menu, including units and physical
    windows. Availability of a numeric dtype alone proves no grouping source.
    """

    from .exposure_group_compile import grouping_sources

    return any(source.concept == concept for source in grouping_sources(context))


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
        from .analysis_types import longitudinal_trajectory_requested

        if (
            trajectory_context_is_bound(context)
            or sealed_trajectory_suite_coordinates(planning_contract_context)
            is not None
        ):
            return CapabilityGapCheck(
                "unverified", "this study has a longitudinal representation"
            )
        if longitudinal_trajectory_requested(context):
            planned = longitudinal_representation_facts(context)
            if planned is not None:
                return CapabilityGapCheck("unverified", planned["planning_rule"])
            return CapabilityGapCheck(
                "verified", "the question requests within-stay trajectories or repeated measurements, "
                "and this study has no longitudinal representation",
            )
        scalar = stay_level_representation_facts(context)
        if scalar is not None:
            fixed_window = fixed_window_association_facts(context)
            fact = scalar["planning_rule"]
            if fixed_window is not None:
                fact += " " + fixed_window["planning_rule"]
            return CapabilityGapCheck("unverified", fact)
        return CapabilityGapCheck(
            "unverifiable", "this study has no longitudinal representation, but the host has no evidence "
            "that the question requires within-stay trajectories or repeated measurements",
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
    "fixed_window_association_facts",
    "longitudinal_representation_facts",
    "stay_level_representation_facts",
    "threshold_grouping_expressible",
]
