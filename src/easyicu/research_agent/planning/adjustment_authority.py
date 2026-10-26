"""Origin-aware adjustment sets and host-verified question variables.

Owner
-----
The typed :class:`~easyicu.research_agent.schema.UserPreferences` contract owns
whether an adjustment set is still open to Planner selection or was fixed by
the user-facing study configuration.  This module compiles that small public
contract and validates Planner output. When adjustment is unstated, it binds
the study family's versioned minimal set before either Planner strategy runs.
This metadata-only rule never inspects rows or searches arbitrary covariates.

The host's structured question reading is a separate required subset in
``data_constraints.stated_adjustments``. The shared variable receipt also
carries ``stated_subgroups``; its coverage belongs to question_requirements.
Neither role closes the Planner's remaining choices or claims baseline timing.
Requirement coverage checks the primary model and capability-gap validation
checks any declared inability.

The validator is intentionally called both while structured Planner retry is
available and again at the execution boundary.  A stored/resumed plan must not
be able to bypass the same scientific authority that constrained a fresh plan.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ..authority.declared_levels import observed_levels_for
from ..concept_availability import concept_records_one_value_per_stay, variable_source_unavailability
from ..contracts.exposure_group_rules import EXPOSURE_GROUP_TRANSFORMS, read_grouping_rules
from ..research_context.materialization_window import bound_feature_window_end_hours, context_column_windows
from ..research_context.stay_events import (
    column_kind,
    stay_outcome_columns,
    whole_stay_event_columns,
)
from ..research_context.typed import declared_domain_for_variable


# A repeated concept is materialized into explicit value summaries before the
# Planner sees it.  Those suffixes change representation, not the user-owned
# covariate identity.  Measurement/count/time companions are deliberately not
# included: ``charlson_n`` or ``charlson_measured`` is not the Charlson score.
_VALUE_AGGREGATION_SUFFIXES = ("_first", "_mean", "_max", "_min")
_MODEL_TERM_INELIGIBLE_ROLES = frozenset(
    {"id", "time", "index", "meta", "outcome"}
)
_MODEL_TERM_DYNAMIC_ROLES = frozenset(
    {"vital", "lab", "intervention", "ordinal_score", "composite_score"}
)
#: The roles a descriptive baseline or a fitted feature (a prediction's
#: predictor, a phenotype's input) may hold when its window is placed: the
#: model-term roles and ``other``.
WINDOW_DESCRIPTION_ROLES = frozenset({*_MODEL_TERM_DYNAMIC_ROLES, "other"})
HostTemporalRole = Literal["baseline_static", "at_or_before_time_zero"]


StatedVariableField = Literal["stated_adjustments", "stated_subgroups"]
STATED_VARIABLE_FIELDS = ("stated_adjustments", "stated_subgroups")


class StatedVariable(BaseModel):
    """A model-read variable and quote whose identity the host checked.

    The containing field states its role. Availability never proves baseline
    timing; unresolved means no source concept was matched to the words.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    variable: str = Field(min_length=1, max_length=160)
    words: str = Field(min_length=1, max_length=240)
    concept: str | None = Field(default=None, min_length=1, max_length=128)
    source_status: Literal["available", "unavailable", "unbound", "unresolved"]


def normalize_stated_variables(value: Any) -> list[dict[str, Any]]:
    """Validate receipts without inferring a scientific role from keywords."""

    items = TypeAdapter(list[StatedVariable]).validate_python([] if value is None else value)
    if len(items) > 16 or len({(item.variable, item.words) for item in items}) != len(items):
        raise ValueError("stated variables must be distinct and contain at most 16 entries")
    return [item.model_dump(mode="json") for item in items]


def context_stated_variables(context: Any, field: StatedVariableField) -> tuple[StatedVariable, ...]:
    """Read a required subset from the existing data-constraints channel."""

    raw = getattr(getattr(context, "user_preferences", None), "data_constraints", None)
    try:
        constraints = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return ()  # Legacy free-form constraints carry no structured requirements.
    if not isinstance(constraints, Mapping):
        return ()
    return tuple(StatedVariable.model_validate(item) for item in
                 normalize_stated_variables(constraints.get(field)))


# Saved adjustments retain their record shape, digest and public reader names.
StatedAdjustment = StatedVariable
normalize_stated_adjustments = normalize_stated_variables


def context_stated_adjustments(context: Any) -> tuple[StatedAdjustment, ...]:
    return context_stated_variables(context, "stated_adjustments")


def stated_adjustment_columns(context: Any, adjustment: StatedAdjustment) -> tuple[str, ...]:
    """Resolve a stay value, or retain every repeated-measurement alternative.

    Older preparation repeated native stay values on a time grid and emitted
    several summaries. The dictionary's one-value declaration makes the
    recorded first value a representation of that same value, not a choice of
    clinical aggregation. Require matching source/window receipts; a suffix
    alone grants no such equivalence. Timing remains the window owner's job.
    """

    if adjustment.concept is None:
        return ()
    candidates = [
        variable for variable in context.variables
        if _matches_declared_covariate(adjustment.concept, variable.name)
        and str(getattr(variable.role, "value", variable.role)) not in _MODEL_TERM_INELIGIBLE_ROLES
    ]
    if concept_records_one_value_per_stay(adjustment.concept):
        for variable in candidates:
            if (variable.name == adjustment.concept
                    and variable.source_concept in {None, adjustment.concept}):
                return (variable.name,)
        for variable in candidates:
            if (variable.name == f"{adjustment.concept}_first"
                    and variable.unit_normalization == "window_numeric_first"
                    and variable.analysis_window
                    and all(item.source_concept == adjustment.concept
                            and item.analysis_window == variable.analysis_window for item in candidates)):
                return (variable.name,)
    return tuple(variable.name for variable in candidates)


def stated_adjustment_after_time_zero(context: Any, adjustment: StatedAdjustment) -> bool:
    """Only an owner-declared window after a typed landmark proves this gap."""

    landmark = primary_landmark_hours(context)
    columns = stated_adjustment_columns(context, adjustment)
    if landmark is None or not columns:
        return False
    variables = {item.name: item for item in context.variables}
    ends = [
        None if owner_declared_baseline_static(variables[name]) else
        _analysis_window_end_hours(variables[name].analysis_window)
        for name in columns
    ]
    return all(end is not None and end > landmark for end in ends)

# ``first_24h``, ``0-24h``, ``0_24h``, ``24h``: the trailing hour bound of a
# named window is the only fact this owner reads from a window label.
_WINDOW_END_HOURS = re.compile(r"(?:^|[^0-9])(\d+(?:\.\d+)?)\s*h(?:ours?)?\s*$", re.IGNORECASE)
#: The words a free-text window label may carry besides its hours: the ICU-
#: admission anchor and window words.  Any other word names another anchor
#: ("sepsis_onset_0_24h", "hospital admission 0-24 hours"), which the trailing
#: hour bound cannot place on the ICU-admission axis.
_ICU_AXIS_LABEL_WORDS = frozenset({
    "icu", "admission", "admit", "admitted", "first", "from", "after", "since",
    "within", "window", "the", "of", "h", "hr", "hrs", "hour", "hours",
})
#: The typed label form, ``<anchor>[start,end]h``; the research-context
#: builder writes ``icu_admission[start,end]h``, and concept metadata may name
#: another anchor (``event_onset[0,72]h``).
_WINDOW_INTERVAL_HOURS = re.compile(
    r"(?:^|[^a-z0-9_])(?P<anchor>[a-z][a-z0-9_]*)\[\s*-?\d+(?:\.\d+)?\s*,\s*"
    r"(?P<end>-?\d+(?:\.\d+)?)\s*\]\s*h(?:ours?)?\s*$",
    re.IGNORECASE,
)


def primary_landmark_hours(context: Any) -> Optional[float]:
    """Return the typed landmark (hours after ICU admission) or ``None``.

    The landmark is a study-context fact: an explicit ``landmark_hours``
    preference or the prespecified landmark timing sensitivity.  Prose is not
    authority, so no text is parsed here.
    """

    preferences = getattr(context, "user_preferences", None)
    if preferences is None:
        return None
    declared = getattr(preferences, "landmark_hours", None)
    if declared is not None:
        return float(declared)
    for spec in getattr(preferences, "sensitivity_specs", ()) or ():
        if (
            str(getattr(spec, "axis", "") or "") == "timing"
            and str(getattr(spec, "strategy", "") or "") == "landmark"
            and getattr(spec, "landmark_hours", None) is not None
        ):
            return float(spec.landmark_hours)
    return None


def host_outer_feature_window_end_hours(context: Any) -> Optional[float]:
    """Return the end (hours after ICU admission) of the feature window that binds the run.

    The cohort carries no measurement outside it.  The materialization owner
    answers: the host's materialization record, else the windows the run's
    caller declared.  A window the context builder synthesized -- its default
    roster, or one inferred from the question's wording -- binds nothing, so
    the answer is then ``None``.
    """

    return bound_feature_window_end_hours(context)


def _analysis_window_end_hours(label: Any) -> Optional[float]:
    """The window's end in hours after ICU admission, or None if it names none.

    A window anchored elsewhere cannot be placed on the ICU-admission axis
    without the anchor's time, so it proves no timing here: a typed interval
    (``event_onset[0,72]h``) or free text naming another anchor
    (``sepsis_onset_0_24h``).  Neither does a label this owner cannot read.
    """

    text = str(label or "").strip()
    if not text:
        return None
    interval = _WINDOW_INTERVAL_HOURS.search(text)
    if interval:
        if interval.group("anchor").lower() != "icu_admission":
            return None
        return float(interval.group("end"))
    match = _WINDOW_END_HOURS.search(text)
    if match is None:
        return None
    if any(word not in _ICU_AXIS_LABEL_WORDS for word in re.findall(r"[a-z]+", text.lower())):
        return None
    return float(match.group(1))


def owner_declared_baseline_static(variable: Any) -> bool:
    """Whether the concept owner declares this variable a baseline attribute.

    The concept dictionary's ``demographic`` role is an owner declaration that
    the value describes the admission itself (age, sex, admission type), not a
    measurement taken during the stay.  It is the same authority that lets the
    host prove a baseline covariate's timing without a rationale, so it is
    shared rather than re-derived from a column name or a physical window.
    """

    role = str(getattr(getattr(variable, "role", None), "value", None) or getattr(variable, "role", "") or "")
    return role == "demographic"


def baseline_static_grouping(variable: Any, context: Any) -> bool:
    """Whether a study's grouping column holds only baseline attributes.

    The host forms a grouping column (an exposure-group transform) from the
    concepts it names.  One with no window of its own, formed only from
    columns the owner declares baseline attributes (age groups), holds
    nothing observed after time zero.  A source the context does not carry
    proves nothing.
    """

    if str(getattr(variable, "unit_normalization", "") or "") not in EXPOSURE_GROUP_TRANSFORMS:
        return False
    if str(getattr(variable, "analysis_window", "") or "").strip():
        return False
    sources = [str(name) for name in getattr(variable, "derived_from_concepts", None) or ()]
    return bool(sources) and all(
        owner_declared_baseline_static(context.variable(name)) for name in sources
    )


def exposure_adjustment_columns(context: Any) -> frozenset[str]:
    """The exposure and the input columns of its declared grouping are not adjustments."""
    exposure = str(context.primary_exposure or "")
    variable = context.variable(exposure)
    columns = {exposure}
    if variable is not None and variable.unit_normalization in EXPOSURE_GROUP_TRANSFORMS:
        concepts = set(variable.derived_from_concepts)
        columns.update(v.name for v in context.variables if v.name in concepts or v.source_concept in concepts)
        for group in (context.cohort.provenance or {}).get("exposure_groups", ()):
            if group.get("variable") == exposure and group.get("parameters"):
                columns.update(read_grouping_rules(group["parameters"]).read_columns)
    return frozenset(columns)


def host_window_bound_roles(
    context: Any,
    *,
    reference_hours: Optional[float],
    dynamic_roles: frozenset[str] = _MODEL_TERM_DYNAMIC_ROLES,
    outer_window_fallback_roles: frozenset[str] = _MODEL_TERM_DYNAMIC_ROLES,
) -> dict[str, HostTemporalRole]:
    """Project which variables the host can prove observed by ``reference_hours``.

    An owner-declared baseline demographic is static. A window-derived variable
    with one of ``dynamic_roles`` is available at or before the reference time
    only when its materialization window ends at or before it: its own
    ``analysis_window`` label, or -- only when it declares none, and only for
    the clinical roles listed in ``outer_window_fallback_roles`` -- the outer
    host-bound feature window.  Only a column that summarizes observations
    over a window inherits that window (``stay_events.column_kind``, the
    reading the materialization owner gives it): a stay-level value, an
    outcome, an event's time or an event status recorded over the whole stay
    does not, whatever its role.
    A native score with a source-specific definition window is known only
    once that window ends, including legacy descriptors with role ``other``.
    Every other variable is absent from the mapping. With no reference time
    only the demographics are provable.
    """

    outer_end = host_outer_feature_window_end_hours(context)
    windows = context_column_windows(context)
    outcomes = stay_outcome_columns(context)
    whole_stay = whole_stay_event_columns(context)
    roles: dict[str, HostTemporalRole] = {}
    for variable in getattr(context, "variables", ()) or ():
        name = str(getattr(variable, "name", "") or "").strip()
        role = str(getattr(variable.role, "value", variable.role) or "")
        if not name:
            continue
        if owner_declared_baseline_static(variable):
            roles[name] = "baseline_static"
            continue
        native = windows.get(name)
        if native is not None and native.source == "concept":
            if reference_hours is not None and native.end_hours <= reference_hours:
                roles[name] = "at_or_before_time_zero"
            continue
        if role not in dynamic_roles or reference_hours is None:
            continue
        label = str(getattr(variable, "analysis_window", None) or "").strip()
        window_end = _analysis_window_end_hours(label)
        # Only a variable without its own window inherits the outer one. A
        # declared window this owner cannot place before the reference time
        # proves nothing; the outer window is not a stand-in for it.
        if (
            not label
            and role in outer_window_fallback_roles
            and name not in whole_stay
            and column_kind(
                variable,
                column=name,
                concept=str(getattr(variable, "source_concept", "") or "").strip() or name,
                outcomes=outcomes,
            )
            == "window_summary"
        ):
            window_end = outer_end
        if window_end is not None and window_end <= reference_hours:
            roles[name] = "at_or_before_time_zero"
    return roles


def host_proven_temporal_roles(
    context: Any, *, reference_hours: Optional[float] = None
) -> dict[str, HostTemporalRole]:
    """Project the covariate timing the host can prove without a rationale.

    An owner-declared baseline demographic is static.  A window-derived clinical
    measurement or score is available at or before time zero only when its
    materialization window (its own ``analysis_window`` label, else the outer
    host-bound feature window) ends at or before the typed landmark.  Every
    other variable is absent from the mapping: a generated clinical rationale
    never replaces this timing authority, so such a variable can enter an
    adjustment set only through an exact user-reviewed roster.

    ``reference_hours`` is a host-typed time zero that is not yet a study
    preference: the landmark of a design the host itself proposes for review.
    Without it the study's declared landmark is used.
    """

    if reference_hours is None:
        reference_hours = primary_landmark_hours(context)
    return host_window_bound_roles(context, reference_hours=reference_hours)


def _matches_declared_covariate(declared: str, observed: str) -> bool:
    if observed == declared:
        return True
    return any(
        observed == f"{declared}{suffix}"
        for suffix in _VALUE_AGGREGATION_SUFFIXES
    )


def concept_relatives(context: Any) -> Callable[[str], frozenset[str]]:
    """The run's columns the concept dictionary relates to a column.

    A column's identity is its name, its source concept and the concepts it
    derives from; two columns are related when their identities meet.
    """

    identities = {
        str(variable.name): frozenset(
            str(value)
            for value in (
                variable.name,
                getattr(variable, "source_concept", None),
                *(getattr(variable, "derived_from_concepts", None) or ()),
            )
            if str(value or "").strip()
        )
        for variable in context.variables
    }

    def relatives(column: str) -> frozenset[str]:
        own = identities.get(column, frozenset({column}))
        return frozenset(
            {column, *(name for name, ids in identities.items() if ids & own)}
        )

    return relatives


class AdjustmentAuthorityError(ValueError):
    """A plan changed a user-locked or host-bound adjustment set."""

    code = "adjustment_set_authority_mismatch"


FAMILY_MINIMAL_ADJUSTMENT_RULE = "family_minimal_adjustment/1"
FAMILY_MINIMAL_ADJUSTMENTS = {
    "association_study": ("age", "sex"),
    "ordinal_dose_response": ("age", "sex"),
}

ADJUSTED_ASSOCIATION_SUBGROUP_GUIDANCE = (
    "For adjusted_association, model_terms encode the exposure and adjustment "
    "covariates only; they do not express stratified/subgroup analysis. Keep a "
    "stated modifier or stratifier outside the adjustment set unless it is "
    "explicitly in the bound roster. Plan its analysis in a separate "
    "custom_analysis step with scientific_action_id=association.effect_modification, "
    "an analysis role such as secondary, and the stratifier in raw_inputs, "
    "together with the exposure, outcome and bound covariates. Preserve that "
    "subgroup step when repairing model_terms; if it is absent, the outline "
    "needs a subgroup owner. Table 1 grouping or a covariate term does not "
    "satisfy the subgroup requirement."
)


def stated_subgroup_columns(context: Any) -> frozenset[str]:
    """Resolve stated modifiers to analysis columns through concept lineage."""
    relatives = concept_relatives(context)
    return frozenset(
        name
        for item in context_stated_variables(context, "stated_subgroups")
        if item.concept
        for name in relatives(item.concept)
    )


def _family_default_adjustment(context: Any) -> "AdjustmentSetAuthority | None":
    """Bind the same minimal roster for family templates and general planning.

    Use the existing study-family router and source metadata. An incomplete
    context cannot establish a default; other families retain their owners.
    A modifier's main effect and interaction remain its model owner's job.
    """
    from .analysis_types import infer_analysis_type

    if (not getattr(context, "primary_exposure", None) or not getattr(context, "target_outcome", None)
            or context.variable(context.primary_exposure) is None
            or context.variable(context.target_outcome) is None):
        return None
    family = infer_analysis_type(context).key
    if family not in FAMILY_MINIMAL_ADJUSTMENTS:
        return None
    relatives = concept_relatives(context)
    excluded = set(exposure_adjustment_columns(context))
    for name in (context.primary_exposure, context.target_outcome, *context.cohort.outcome_columns):
        excluded.update(relatives(name))
    for item in context_stated_variables(context, "stated_subgroups"):
        if item.concept:
            excluded.update(relatives(item.concept))
    selected = []
    for concept in FAMILY_MINIMAL_ADJUSTMENTS[family]:
        candidates = [v.name for v in context.variables
                      if v.name not in excluded and owner_declared_baseline_static(v)
                      and not v.derived_from_concepts
                      and (v.source_concept == concept or (v.name == concept and v.source_concept is None))
                      and not variable_source_unavailability(v, str(context.cohort.database or ""))]
        if concept in candidates:
            selected.append(concept)
        elif len(candidates) == 1:
            selected.append(candidates[0])
    reason = (
        f"{FAMILY_MINIMAL_ADJUSTMENT_RULE}; family={family}: prespecified age, sex "
        "in that order when uniquely available as baseline demographics, excluding "
        "exposure sources, outcomes and modifiers. No adjustment was stated. "
        "This minimal association model does not eliminate residual confounding."
    )
    return AdjustmentSetAuthority(
        selection="exact", covariates=tuple(selected), authority="agent_plan",
        rationales=tuple((name, reason) for name in selected),
        temporal_roles=tuple((name, "baseline_static") for name in selected),
        operationalizations=tuple((name, name) for name in selected),
        rule=FAMILY_MINIMAL_ADJUSTMENT_RULE,
    )


@dataclass(frozen=True)
class AdjustmentSetAuthority:
    """Immutable projection of the exact roster and its declared author."""

    selection: Literal["planner_selectable", "exact"]
    covariates: tuple[str, ...]
    authority: Optional[Literal["user", "agent_plan"]] = None
    rationales: tuple[tuple[str, str], ...] = ()
    temporal_roles: tuple[tuple[str, str], ...] = ()
    operationalizations: tuple[tuple[str, str], ...] = ()
    rule: Optional[str] = None  # Host-only provenance; not a Planner output field.

    @classmethod
    def from_context(cls, context: Any) -> "AdjustmentSetAuthority":
        preferences = getattr(context, "user_preferences", None)
        selection = str(
            getattr(preferences, "covariate_selection", "planner_selectable")
            or "planner_selectable"
        ).strip()
        covariates = tuple(
            str(value or "").strip()
            for value in (getattr(preferences, "covariates", ()) or ())
            if str(value or "").strip()
        )
        if selection not in {"planner_selectable", "exact"}:
            raise AdjustmentAuthorityError(
                "adjustment_set_authority_invalid: covariate_selection must be "
                "'planner_selectable' or 'exact'"
            )
        rationales = getattr(preferences, "covariate_rationales", {}) or {}
        temporal_roles = getattr(preferences, "covariate_temporal_roles", {}) or {}
        operationalizations = (
            getattr(preferences, "covariate_operationalizations", {}) or {}
        )
        authority = getattr(preferences, "covariate_authority", None)
        if selection != "exact" and not context_stated_adjustments(context):
            default = _family_default_adjustment(context)
            if default is not None:
                return default
        return cls(
            selection=selection,
            covariates=covariates,
            authority=authority,
            rationales=tuple((name, str(rationales[name])) for name in covariates if name in rationales),
            temporal_roles=tuple(
                (name, str(temporal_roles[name]))
                for name in covariates
                if name in temporal_roles
            ),
            operationalizations=tuple(
                (name, str(operationalizations[name]))
                for name in covariates
                if name in operationalizations
            ),
        )

    @property
    def operational_covariates(self) -> tuple[str, ...]:
        mapping = dict(self.operationalizations)
        return tuple(mapping.get(name, name) for name in self.covariates)

    @property
    def operational_rationales(self) -> dict[str, str]:
        mapping = dict(self.operationalizations)
        return {mapping.get(name, name): value for name, value in self.rationales}

    @property
    def operational_temporal_roles(self) -> dict[str, str]:
        mapping = dict(self.operationalizations)
        return {mapping.get(name, name): value for name, value in self.temporal_roles}

    def prompt_projection(self) -> dict[str, Any]:
        """Preserve authorship wherever a bound adjustment set is described.

        Exactness is a constraint on the roster, not evidence that the user
        authored it or that a plan-review action has occurred.
        """

        if self.selection != "exact":
            boundary = "Planner-selectable; available variables are not user-specified covariates."
        elif self.rule:
            boundary = f"Host family default {self.rule}, not user-specified; preserve the bound roster."
        elif self.authority == "agent_plan":
            boundary = "Agent-selected, not user-specified; describe this as the Agent's proposed adjustment set."
        elif self.authority == "user":
            boundary = "User-authored adjustment set; preserve the exact roster."
        else:
            boundary = "Authorship is unrecorded; do not infer user or Agent authorship."
        return {
            "selection": self.selection,
            "authority": self.authority or "unrecorded",
            "scientific_covariates": list(self.covariates),
            "operational_covariates": list(self.operational_covariates),
            "operationalizations": dict(self.operationalizations),
            "authorship_boundary": boundary + " This binding is not a plan-approval receipt.",
            "covariate_roster_rule": (
                "Include every bound covariate in order: model covariates must equal "
                "operational_covariates, including [] for no adjustment. The host "
                "will not add, drop or reorder a Planner term."
                if self.selection == "exact" else
                "Choose justified covariates from the eligible variables."
            ),
            "adjusted_association_subgroups": ADJUSTED_ASSOCIATION_SUBGROUP_GUIDANCE,
            **({"selection_rule": self.rule} if self.rule else {}),
        }

    def plan_rationale(self, rationale: str) -> str:
        """Persist the host decision even for an empty default or an LLM omission."""
        if not self.rule:
            return rationale
        return (
            f"{rationale} Adjustment policy {self.rule}: no adjustment was stated; "
            "bind age then sex when uniquely available as baseline demographics, "
            "excluding exposure sources, outcomes and modifiers. The bound roster is "
            f"{list(self.operational_covariates)!r}. Stated/exact study rosters take priority, "
            "including explicit no adjustment. Additions require a recorded rationale "
            "and a reviewed study revision. This minimal set does not eliminate residual confounding."
        )

    def validate_plan(self, plan: Any) -> None:
        """Require every declared fitted model to honor an exact roster.

        ``covariates=[]`` is a meaningful scientific decision: an unadjusted
        model.  It is not equivalent to an omitted roster, and available age or
        sex columns do not authorize the Planner to add them.
        """

        if self.selection != "exact":
            return

        mismatches: list[str] = []
        operationalizations = dict(self.operationalizations)
        for step in getattr(plan, "steps", ()) or ():
            step_id = str(getattr(step, "step_id", "") or "<unnamed>")
            for requirement in getattr(step, "model_requirements", ()) or ():
                declared = getattr(requirement, "covariates", None)
                observed: Optional[tuple[str, ...]] = (
                    None
                    if declared is None
                    else tuple(str(value or "").strip() for value in declared)
                )
                roster_matches = (
                    observed is not None
                    and len(observed) == len(self.covariates)
                    and all(
                        (
                            actual == operationalizations[expected]
                            if expected in operationalizations
                            else _matches_declared_covariate(expected, actual)
                        )
                        for expected, actual in zip(self.covariates, observed)
                    )
                )
                if not roster_matches:
                    mismatches.append(
                        f"{step_id}/{getattr(requirement, 'requirement_id', '<unnamed>')}: "
                        f"declared={list(observed) if observed is not None else None!r}"
                    )

        if not mismatches:
            return
        binding = f"host rule {self.rule}" if self.rule else "user_preferences.covariate_selection='exact'"
        raise AdjustmentAuthorityError(
            f"{AdjustmentAuthorityError.code}: {binding} binds "
            f"every planned model to covariates={list(self.covariates)!r}; "
            "the Planner may not add, remove, infer, or reorder covariates. "
            "Mismatches: "
            + "; ".join(mismatches[:6])
        )


def adjusted_model_term_planning_authority(
    context: Any,
    variable_names: Sequence[str],
    *,
    reference_hours: Optional[float] = None,
) -> dict[str, Any]:
    """Project model-term eligibility for a bounded Planner prompt.

    This metadata-only projection cannot select an adjustment set or infer
    clinical timing. It exposes which catalog variables the current typed
    authority permits as model terms and the encodings supported by their
    sealed domains. ``reference_hours`` is passed to
    :func:`host_proven_temporal_roles`.
    """

    adjustment = AdjustmentSetAuthority.from_context(context)
    temporal_roles = adjustment.operational_temporal_roles
    authorized_time_zero = frozenset(
        name
        for name in adjustment.operational_covariates
        if adjustment.selection == "exact"
        and temporal_roles.get(name) in {"baseline_static", "at_or_before_time_zero"}
    )
    primary_exposure = str(context.primary_exposure or "").strip()
    exposure_columns = exposure_adjustment_columns(context)
    outcome = str(context.target_outcome or "").strip()
    variables = {item.name: item for item in context.variables}
    host_timed: Mapping[str, HostTemporalRole] = host_proven_temporal_roles(
        context, reference_hours=reference_hours
    )
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for name in dict.fromkeys(str(value or "").strip() for value in variable_names):
        if not name or name in {primary_exposure, outcome}:
            continue
        if name in exposure_columns:
            excluded.append({"name": name, "reason": "exposure_group_source"})
            continue
        variable = variables.get(name)
        if variable is None:
            continue
        role = str(getattr(variable.role, "value", variable.role) or "")
        if role in _MODEL_TERM_INELIGIBLE_ROLES:
            excluded.append({"name": name, "reason": f"semantic_role:{role}"})
            continue
        if adjustment.selection != "exact" and name not in host_timed:
            excluded.append(
                {"name": name, "reason": "planner_baseline_authority_missing"}
            )
            continue
        if (
            role in _MODEL_TERM_DYNAMIC_ROLES
            and name not in authorized_time_zero
            and host_timed.get(name) != "at_or_before_time_zero"
        ):
            excluded.append({"name": name, "reason": "time_zero_authority_missing"})
            continue
        if adjustment.selection == "exact" and name not in adjustment.operational_covariates:
            excluded.append({"name": name, "reason": "outside_bound_adjustment_roster"})
            continue
        observed = observed_levels_for(name=name, variables=variables)
        declared, declared_basis = declared_domain_for_variable(variable)
        closed_domain = observed or list(declared or ())
        eligible.append(
            {
                "name": name,
                "semantic_role": role,
                "host_temporal_role": host_timed.get(name),
                "allowed_codings": [
                    "binary" if len(closed_domain) == 2 else "categorical"
                ]
                if closed_domain
                else ["continuous"],
                "closed_domain_size": len(closed_domain) or None,
                "domain_authority": "observed" if observed else declared_basis,
            }
        )
    return {
        "selection": adjustment.selection,
        "primary_exposure": primary_exposure,
        "outcome": outcome,
        "eligible_covariates": eligible,
        "excluded_covariates": excluded,
    }


def validate_plan_against_adjustment_authority(*, plan: Any, context: Any) -> None:
    """Public fail-closed boundary used by planning and execution."""

    AdjustmentSetAuthority.from_context(context).validate_plan(plan)


__all__ = [
    "StatedAdjustment",
    "normalize_stated_adjustments",
    "AdjustmentAuthorityError",
    "AdjustmentSetAuthority",
    "HostTemporalRole",
    "adjusted_model_term_planning_authority",
    "host_outer_feature_window_end_hours",
    "host_proven_temporal_roles",
    "WINDOW_DESCRIPTION_ROLES",
    "host_window_bound_roles",
    "primary_landmark_hours",
    "validate_plan_against_adjustment_authority",
]
