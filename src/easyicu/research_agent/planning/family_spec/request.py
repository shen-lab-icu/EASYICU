"""Project the sealed StudyContext into one ``FamilySpecRequest``.

Everything here is a projection of typed facts already owned elsewhere: the
adjustment authority (roster and host-proven timing), the ordinal multi-outcome
contract, the prespecified sensitivity specs, the dependence authority, the
closed variable domains, and the sealed literature roster.  Nothing is inferred
from prose except the research question text that is copied verbatim.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from ...authority.declared_levels import closed_planning_levels_for
from ...concept_availability import variable_source_unavailability
from ...contracts.model_terms import level_spelling
from ...schema import ResearchContext
from ..accepted_analysis_inputs import analysis_input_value_columns
from ..adjustment_authority import (
    AdjustmentSetAuthority,
    adjusted_model_term_planning_authority,
    host_outer_feature_window_end_hours,
    host_window_bound_roles,
    primary_landmark_hours,
)
from ..dependence_authority import (
    context_dependence_authority,
    descriptive_counts_only_required,
)
from ..ordinal_multi_outcome import resolve_ordinal_multi_outcome_contract
from .contract import (
    DESCRIPTIVE_FAMILY_ID,
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    LANDMARK_CATEGORICAL_FAMILY_ID,
    LANDMARK_SPLINE_FAMILY_ID,
    LANDMARK_SURVIVAL_FAMILY_ID,
    PHENOTYPING_FAMILY_ID,
    PREDICTION_FAMILY_ID,
    MAX_FIT_FEATURES,
    SOURCE_FEASIBILITY_FAMILY_ID,
    AcceptedFeatureGroup,
    AdjustmentCandidate,
    ExposureKind,
    FamilySpecError,
    FamilySpecRequest,
    SealedFeasibilityCoordinates,
    SealedSuiteCoordinates,
    SealedTrajectoryCoordinates,
    SensitivityAxisBinding,
)

#: Marker lines the sealed authorities print before their JSON coordinates
#: (``LandmarkSurvivalRuntimeAuthority.planning_contract_context`` and
#: ``TrajectoryScientificRuntimeAuthority.planning_contract_context``).
SEALED_SURVIVAL_SUITE_MARKER = "CALLER-BOUND LANDMARK SURVIVAL SUITE:"
SEALED_TRAJECTORY_SUITE_MARKER = "CALLER-BOUND FIXED-WINDOW TRAJECTORY SUITE:"
SEALED_FEASIBILITY_MARKER = "CALLER-BOUND SOURCE FEASIBILITY DECISION:"


def _sealed_disclosure_payload(planning_contract_context: str, marker: str) -> dict | None:
    """Return the JSON object following ``marker`` in the disclosure text, or None.

    The pipeline hands the planner each authority's own disclosure text; this
    reads only the object after the marker line.  Any other prose, a missing
    object, or a malformed object yields ``None``: the family then falls back
    to Progressive v2 instead of guessing a suite.
    """

    text = str(planning_contract_context or "")
    start = text.find(marker)
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    end = None
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end is None:
        return None
    try:
        payload = json.loads(text[brace : end + 1])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def sealed_trajectory_suite_coordinates(
    planning_contract_context: str,
) -> SealedTrajectoryCoordinates | None:
    """Parse the sealed fixed-window trajectory disclosure, or ``None`` when absent."""

    payload = _sealed_disclosure_payload(
        planning_contract_context, SEALED_TRAJECTORY_SUITE_MARKER
    )
    if payload is None or "sealed_representation_owner" not in payload:
        return None
    try:
        window = payload.get("window_hours") or []
        return SealedTrajectoryCoordinates(
            representation_owner=str(payload.get("sealed_representation_owner") or ""),
            candidate_owner=str(payload.get("sealed_candidate_owner") or ""),
            coordinate_concepts=[str(value) for value in payload.get("coordinate_concepts") or []],
            descriptive_only_concepts=[
                str(value) for value in payload.get("descriptive_only_concepts") or []
            ],
            window_hours=(int(window[0]), int(window[1])),
            grid_width_hours=int(payload.get("grid_width_hours") or 0),
            candidate_cluster_counts=[
                int(value) for value in payload.get("candidate_cluster_counts") or []
            ],
            representation_outputs=[
                str(value) for value in payload.get("representation_outputs") or []
            ],
        )
    except (TypeError, ValueError, IndexError):
        return None


def sealed_feasibility_coordinates(
    planning_contract_context: str,
) -> SealedFeasibilityCoordinates | None:
    """Parse the sealed source-feasibility disclosure, or ``None`` when absent."""

    payload = _sealed_disclosure_payload(planning_contract_context, SEALED_FEASIBILITY_MARKER)
    if payload is None or "sealed_owner" not in payload:
        return None
    try:
        window = payload.get("audited_window_hours") or []
        return SealedFeasibilityCoordinates(
            sealed_owner=str(payload.get("sealed_owner") or ""),
            plan_intent=str(payload.get("plan_intent") or ""),
            plan_outputs=[str(value) for value in payload.get("plan_outputs") or []],
            source=str(payload.get("source") or ""),
            audited_window_hours=(int(window[0]), int(window[1])),
            decision=str(payload.get("decision") or ""),
            reason_code=str(payload.get("reason_code") or ""),
            forbidden_plan_tokens=[
                str(value) for value in payload.get("forbidden_plan_tokens") or []
            ],
        )
    except (TypeError, ValueError, IndexError):
        return None


def sealed_survival_suite_coordinates(
    planning_contract_context: str,
) -> SealedSuiteCoordinates | None:
    """Parse the sealed survival suite disclosure, or ``None`` when absent."""

    payload = _sealed_disclosure_payload(
        planning_contract_context, SEALED_SURVIVAL_SUITE_MARKER
    )
    if payload is None or "sealed_primary_owner" not in payload:
        return None
    try:
        return SealedSuiteCoordinates(
            primary_owner=str(payload.get("sealed_primary_owner") or ""),
            exposure_status_column=str(payload.get("exposure_status_column") or ""),
            exposure_onset_column=str(payload.get("exposure_onset_column") or ""),
            event_column=str(payload.get("event_column") or ""),
            followup_time_column=str(payload.get("followup_time_column") or ""),
            landmark_hours=float(payload.get("landmark_hours") or 0.0),
            endpoint_horizon_days=float(payload.get("endpoint_horizon_days") or 0.0),
            adjustment_columns=[str(value) for value in payload.get("adjustment_columns") or []],
            plan_outputs=[str(value) for value in payload.get("plan_outputs") or []],
        )
    except (TypeError, ValueError):
        return None

_CONTINUOUS_EXPOSURE_ROLES = frozenset({"lab", "vital", "composite_score", "other"})
_NUMERIC_DTYPES = ("float", "int", "double", "decimal")
#: Roles a descriptive baseline table may describe when the host can prove the
#: measurement sits inside the sealed observation window. ``other`` covers
#: window-aggregated indices (a comorbidity score's first value, for example)
#: that are legitimate baseline descriptors without being adjustment terms.
_BASELINE_DESCRIPTION_ROLES = frozenset(
    {"vital", "lab", "intervention", "ordinal_score", "composite_score", "other"}
)



def _landmark_timing_spec(context: ResearchContext) -> Any | None:
    preferences = context.user_preferences
    if preferences is None:
        return None
    for spec in preferences.sensitivity_specs:
        if (
            str(spec.axis or "") == "timing"
            and str(spec.strategy or "") == "landmark"
            and spec.event_time_variable
            and spec.observation_duration_variable
        ):
            return spec
    return None


def _typed_cohort_constraints(context: ResearchContext) -> Mapping[str, Any]:
    preferences = context.user_preferences
    raw = getattr(preferences, "data_constraints", None)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    cohort = payload.get("cohort")
    return cohort if isinstance(cohort, Mapping) else {}


def _typed_cohort_fields(
    context: ResearchContext,
    required_primary_cohort_selection_mode: str | None,
    *,
    require_typed_bound: bool,
) -> dict[str, Any]:
    """The primary cohort's typed row bounds and the selection mode they imply.

    Age bounds and a minimum ICU stay are the only typed predicates a family
    template applies; prose criteria are not authority.  A minimum stay is
    checked against ``los_icu``, the stay-level duration the predicate reads,
    so a roster without it fails here instead of losing the criterion.
    """

    cohort = _typed_cohort_constraints(context)
    age_min = _optional_number(cohort.get("age_min"))
    age_max = _optional_number(cohort.get("age_max"))
    minimum = _optional_number(cohort.get("min_icu_los_hours"))
    minimum_icu_hours = minimum if minimum is not None and minimum > 0 else None
    typed = any(value is not None for value in (age_min, age_max, minimum_icu_hours))
    if required_primary_cohort_selection_mode in {"all_input_rows", "predicate_filtered"}:
        selection_mode = required_primary_cohort_selection_mode
    else:
        selection_mode = "predicate_filtered" if typed else "all_input_rows"
    if require_typed_bound and selection_mode == "predicate_filtered" and not typed:
        raise FamilySpecError(
            "family_spec_cohort_predicate_unavailable",
            "a predicate-filtered primary cohort needs a typed age bound or minimum ICU "
            "stay in data_constraints.cohort; prose criteria are not authority",
            path="cohort",
        )
    if minimum_icu_hours is not None and not any(
        variable.name == "los_icu" for variable in context.variables
    ):
        raise FamilySpecError(
            "family_spec_cohort_predicate_unavailable",
            "a minimum ICU stay needs the stay's ICU length of stay (los_icu) in the "
            "sealed roster",
            path="cohort",
        )
    return {
        "cohort_selection_mode": selection_mode,
        "age_min": age_min,
        "age_max": age_max,
        "minimum_icu_hours": minimum_icu_hours,
    }


def _optional_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _levels(context: ResearchContext, name: str) -> list[str]:
    variables = {item.name: item for item in context.variables}
    return [
        level_spelling(value)
        for value in closed_planning_levels_for(name=name, variables=variables)
    ]


_BINARY_OUTCOME_DOMAINS = (["0", "1"], ["false", "true"])


def _binary_outcome_levels(context: ResearchContext, outcome: str) -> list[str] | None:
    """Return the outcome's closed binary spellings (event spelling last) or ``None``."""

    levels = _levels(context, outcome)
    return levels if levels in _BINARY_OUTCOME_DOMAINS else None


def _structurally_available_roster(
    context: ResearchContext, roster: Sequence[str]
) -> list[str]:
    """Drop optional columns whose source the outline authority would refuse.

    Design coordinates (exposure, outcome, timing) are never filtered here: if
    one of them is structurally unavailable the outline must fail closed with
    that reason. Optional audit and candidate columns, however, are simply not
    offered, because offering an unavailable column only guarantees the later
    refusal.
    """

    variables = {item.name: item for item in context.variables}
    database = str(context.cohort.database or "")
    kept: list[str] = []
    for name in roster:
        variable = variables.get(name)
        if variable is not None and variable_source_unavailability(variable, database):
            continue
        kept.append(name)
    return kept


def _exposure_kind(context: ResearchContext, exposure: str) -> ExposureKind | None:
    """Classify the exposure from its closed domain and typed descriptor only."""

    variable = context.variable(exposure)
    if variable is None:
        return None
    if len(_levels(context, exposure)) >= 2:
        return "categorical"
    role = str(getattr(variable.role, "value", variable.role) or "")
    dtype = str(variable.dtype or "").lower()
    if (
        role in _CONTINUOUS_EXPOSURE_ROLES
        and not variable.is_ordinal
        and any(token in dtype for token in _NUMERIC_DTYPES)
    ):
        return "continuous"
    return None


def exposure_companion_columns(context: ResearchContext, exposure: str) -> list[str]:
    """Columns derived from the exposure's own source concept (window audit context).

    A continuous exposure such as a peak lab value travels with its companion
    aggregates and provenance flags (minimum, mean, count, measured flag, first
    and last observation time). They describe the measurement process, never
    a second exposure or a covariate, so the template routes them to the
    measurement audit and keeps them out of the candidate roster.
    """

    variable = context.variable(exposure)
    concept = str(getattr(variable, "source_concept", "") or "") if variable is not None else ""
    if not concept:
        return []
    return [
        item.name
        for item in context.variables
        if item.name != exposure and str(getattr(item, "source_concept", "") or "") == concept
    ]


def family_template_id_for_context(
    context: ResearchContext,
    *,
    analysis_types: Sequence[str],
    planning_contract_context: str = "",
) -> str | None:
    """Return the family template that fits the sealed context, or ``None``.

    Both landmark association families need a 0/1 endpoint, a typed landmark,
    a prespecified landmark timing spec naming the event-time and
    observation-duration columns, and one identity column. The categorical
    family additionally needs a closed exposure with at least two levels; the
    spline family a numeric, non-ordinal exposure. Anything less is not a
    template gap to guess around; the caller falls back to Progressive v2.
    """

    headline = str(analysis_types[0]) if analysis_types else ""
    exposure = str(context.primary_exposure or "").strip()
    outcome = str(context.target_outcome or "").strip()
    if (
        headline == "causal_inference"
        and sealed_feasibility_coordinates(planning_contract_context) is not None
        and len(context.cohort.id_columns) == 1
    ):
        # The reviewed protocol found the contrast non-identifiable: the only
        # current-run result is the sealed fail-closed decision, which needs
        # no outcome, exposure or adjustment coordinate from the context.
        return SOURCE_FEASIBILITY_FAMILY_ID
    if not outcome or context.variable(outcome) is None:
        return None
    if _binary_outcome_levels(context, outcome) is None:
        return None
    if len(context.cohort.id_columns) != 1:
        return None
    if (
        headline == "trajectory_clustering"
        and sealed_trajectory_suite_coordinates(planning_contract_context) is not None
    ):
        # Longitudinal trajectory clustering is templated only when the host
        # has sealed the fixed-window trajectory suite for this run; the
        # sealed authority, not the context's exposure, is the design proof.
        return FIXED_WINDOW_TRAJECTORY_FAMILY_ID
    if headline == "survival":
        # Survival is templated only when the host has already sealed the
        # landmark survival suite for this run: the Planner names that owner
        # and labels its columns; it never composes a Cox contract itself.
        sealed = sealed_survival_suite_coordinates(planning_contract_context)
        if (
            sealed is None
            or sealed.exposure_status_column != exposure
            or sealed.event_column != outcome
            or any(context.variable(name) is None for name in sealed.source_columns)
        ):
            return None
        return LANDMARK_SURVIVAL_FAMILY_ID
    if headline == "prediction_model":
        # A static binary prediction model has predictors and an outcome but
        # no primary exposure; longitudinal designs are not templated.
        return (
            PREDICTION_FAMILY_ID
            if getattr(context, "fixed_window_trajectory", None) is None
            else None
        )
    if not exposure or context.variable(exposure) is None:
        return None
    kind = _exposure_kind(context, exposure)
    if kind is None:
        return None
    if headline == "association_study":
        if primary_landmark_hours(context) is None or _landmark_timing_spec(context) is None:
            return None
        return LANDMARK_CATEGORICAL_FAMILY_ID if kind == "categorical" else LANDMARK_SPLINE_FAMILY_ID
    if headline == "descriptive_epidemiology" and kind == "categorical":
        return DESCRIPTIVE_FAMILY_ID
    if (
        headline == "trajectory_clustering"
        and kind == "categorical"
        and getattr(context, "fixed_window_trajectory", None) is None
    ):
        # Cross-sectional phenotyping: one feature vector per row inside the
        # sealed observation window. Longitudinal trajectory clustering carries
        # fixed-window trajectory metadata and is not templated.
        return PHENOTYPING_FAMILY_ID
    return None


def _candidates(
    context: ResearchContext,
    *,
    variable_roster: Sequence[str],
    adjustment: AdjustmentSetAuthority,
    design_columns: frozenset[str] = frozenset(),
) -> list[AdjustmentCandidate]:
    """Project the candidate roster from the host model-term authority.

    ``design_columns`` are coordinates the family already uses for another
    purpose (alternate exposure definitions, landmark timing, the first-stay
    flag, secondary outcomes); they are never offered as covariates.
    """

    authority = adjusted_model_term_planning_authority(
        context,
        [name for name in variable_roster if name not in design_columns],
    )
    candidates: list[AdjustmentCandidate] = []
    if adjustment.selection == "exact":
        roles = dict(adjustment.operational_temporal_roles)
        eligible = {row["name"]: row for row in authority["eligible_covariates"]}
        for name in adjustment.operational_covariates:
            row = eligible.get(name)
            codings = list(row["allowed_codings"]) if row else ["continuous"]
            candidates.append(
                AdjustmentCandidate(
                    name=name,
                    semantic_role=str(row["semantic_role"]) if row else "other",
                    host_temporal_role=(
                        roles.get(name)
                        if roles.get(name) in {"baseline_static", "at_or_before_time_zero"}
                        else None
                    ),
                    allowed_codings=codings,
                    closed_domain_size=row.get("closed_domain_size") if row else None,
                    selectable=True,
                    boundary="exact user-reviewed roster; timing and rationale are user authority",
                )
            )
        return candidates
    for row in authority["eligible_covariates"]:
        temporal = row.get("host_temporal_role")
        candidates.append(
            AdjustmentCandidate(
                name=str(row["name"]),
                semantic_role=str(row["semantic_role"]),
                host_temporal_role=temporal,
                allowed_codings=list(row["allowed_codings"]),
                closed_domain_size=row.get("closed_domain_size"),
                selectable=True,
                boundary=(
                    "owner-declared baseline demographic"
                    if temporal == "baseline_static"
                    else "window-derived measurement bound at or before the landmark"
                ),
            )
        )
    variables = {item.name: item for item in context.variables}
    for row in authority["excluded_covariates"]:
        name = str(row["name"])
        variable = variables.get(name)
        if variable is None:
            continue
        candidates.append(
            AdjustmentCandidate(
                name=name,
                semantic_role=str(getattr(variable.role, "value", variable.role) or ""),
                host_temporal_role=None,
                allowed_codings=["continuous"],
                closed_domain_size=None,
                selectable=False,
                boundary=(
                    "pre-time-zero availability is not provable by the host; "
                    f"reason={row['reason']}; enters a model only through an exact "
                    "user-reviewed roster"
                ),
            )
        )
    return candidates


def _baseline_candidates(
    context: ResearchContext,
    *,
    variable_roster: Sequence[str],
    design_columns: frozenset[str],
) -> list[AdjustmentCandidate]:
    """Offer baseline-description variables the host can place in the window.

    A descriptive baseline table has no adjustment role, but "baseline" is still
    a timing claim: only owner-declared demographics and variables whose
    materialization window closes inside the sealed observation window are
    offered. Everything else stays out of Table 1 and in the measurement audit.
    """

    reference = host_outer_feature_window_end_hours(context)
    roles = host_window_bound_roles(
        context, reference_hours=reference, dynamic_roles=_BASELINE_DESCRIPTION_ROLES
    )
    variables = {item.name: item for item in context.variables}
    candidates: list[AdjustmentCandidate] = []
    for name in variable_roster:
        variable = variables.get(name)
        if variable is None or name in design_columns:
            continue
        role = str(getattr(variable.role, "value", variable.role) or "")
        if role in {"id", "time", "meta", "index", "outcome"}:
            continue
        temporal = roles.get(name)
        levels = _levels(context, name)
        if temporal is None:
            candidates.append(
                AdjustmentCandidate(
                    name=name,
                    semantic_role=role,
                    host_temporal_role=None,
                    allowed_codings=["continuous"],
                    closed_domain_size=None,
                    selectable=False,
                    boundary=(
                        "measurement timing inside the sealed observation window is not "
                        "provable by the host; described in the measurement audit only"
                    ),
                )
            )
            continue
        closed = len(levels) if len(levels) >= 2 else None
        candidates.append(
            AdjustmentCandidate(
                name=name,
                semantic_role=role,
                host_temporal_role=temporal,
                allowed_codings=(
                    ["binary"] if closed == 2 else ["categorical"] if closed else ["continuous"]
                ),
                closed_domain_size=closed,
                selectable=True,
                boundary=(
                    "owner-declared baseline demographic"
                    if temporal == "baseline_static"
                    else "window-bound measurement closed inside the sealed observation window"
                ),
            )
        )
    return candidates


def build_family_spec_request(
    context: ResearchContext,
    *,
    analysis_types: Sequence[str],
    variable_roster: Sequence[str],
    allowed_literature_citation_keys: Sequence[str],
    direct_comparator_literature_keys: Sequence[str] = (),
    comparison_literature_keys: Sequence[str] = (),
    comparator_titles: Mapping[str, str] | None = None,
    required_primary_cohort_selection_mode: str | None = None,
    planning_contract_context: str = "",
) -> FamilySpecRequest:
    """Seal the host authority for one family attempt, before any Planner call."""

    request = _family_spec_request(
        context,
        analysis_types=analysis_types,
        variable_roster=variable_roster,
        allowed_literature_citation_keys=allowed_literature_citation_keys,
        direct_comparator_literature_keys=direct_comparator_literature_keys,
        comparison_literature_keys=comparison_literature_keys,
        comparator_titles=comparator_titles,
        required_primary_cohort_selection_mode=required_primary_cohort_selection_mode,
        planning_contract_context=planning_contract_context,
    )
    request = _bind_accepted_feature_groups(context, request)
    _refuse_eligibility_after_time_zero(request)
    return request


def _bind_accepted_feature_groups(
    context: ResearchContext, request: FamilySpecRequest
) -> FamilySpecRequest:
    """Keep the reviewed candidate's primary inputs as the fit's features.

    After data preparation each accepted input is a family of value columns.
    The Planner chooses which of them represents the input, not whether the
    input stays; an accepted input with no selectable column, or more accepted
    inputs than the design holds, is a host contradiction refused before any
    Provider call.
    """

    if request.family_id not in {PHENOTYPING_FAMILY_ID, PREDICTION_FAMILY_ID}:
        return request
    accepted = analysis_input_value_columns(context)
    if not accepted:
        return request
    selectable = {item.name for item in request.feature_candidates if item.selectable}
    groups = [
        AcceptedFeatureGroup(
            concept=concept, columns=[name for name in columns if name in selectable]
        )
        for concept, columns in accepted.items()
        if any(name in selectable for name in columns)
    ]
    unavailable = [
        concept
        for concept, columns in accepted.items()
        if not any(name in selectable for name in columns)
    ]
    if unavailable:
        raise FamilySpecError(
            "family_spec_accepted_input_not_selectable",
            "accepted primary inputs have no selectable fit feature in the prepared data: "
            + ", ".join(unavailable),
            path="feature_candidates",
        )
    if len(groups) > MAX_FIT_FEATURES:
        raise FamilySpecError(
            "family_spec_accepted_inputs_exceed_design",
            f"{len(groups)} accepted primary inputs exceed the {MAX_FIT_FEATURES} fit "
            "features a design names",
            path="feature_candidates",
        )
    return FamilySpecRequest.model_validate(
        {
            **request.model_dump(mode="json"),
            "accepted_feature_groups": [group.model_dump(mode="json") for group in groups],
        }
    )


def _refuse_eligibility_after_time_zero(request: FamilySpecRequest) -> None:
    """A typed minimum ICU stay must be decided by the plan's time zero.

    A stay reaches the minimum exactly when it is still in the ICU at that
    hour.  A minimum beyond time zero would select on survival after it, so
    the host refuses instead of fitting a plan on that population.
    """

    minimum = request.minimum_icu_hours
    time_zero = request.cohort_time_zero_hours
    if minimum is not None and time_zero is not None and minimum > time_zero:
        raise FamilySpecError(
            "family_spec_cohort_eligibility_after_time_zero",
            f"a minimum ICU stay of {minimum:g} h ends after the plan's time zero at "
            f"{time_zero:g} h after ICU admission; eligibility would depend on survival "
            "after time zero",
            path="cohort",
        )


def _family_spec_request(
    context: ResearchContext,
    *,
    analysis_types: Sequence[str],
    variable_roster: Sequence[str],
    allowed_literature_citation_keys: Sequence[str],
    direct_comparator_literature_keys: Sequence[str],
    comparison_literature_keys: Sequence[str],
    comparator_titles: Mapping[str, str] | None,
    required_primary_cohort_selection_mode: str | None,
    planning_contract_context: str,
) -> FamilySpecRequest:
    family_id = family_template_id_for_context(
        context,
        analysis_types=analysis_types,
        planning_contract_context=planning_contract_context,
    )
    if family_id is None:
        raise FamilySpecError(
            "family_spec_template_unavailable",
            "the sealed context does not fit a host-templated family",
            path="family_id",
        )
    if family_id == SOURCE_FEASIBILITY_FAMILY_ID:
        sealed_feasibility = sealed_feasibility_coordinates(planning_contract_context)
        assert sealed_feasibility is not None
        return _build_feasibility_request(
            context,
            sealed=sealed_feasibility,
            variable_roster=variable_roster,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            direct_comparator_literature_keys=direct_comparator_literature_keys,
            comparison_literature_keys=comparison_literature_keys,
            comparator_titles=comparator_titles,
            required_primary_cohort_selection_mode=required_primary_cohort_selection_mode,
        )
    if family_id == FIXED_WINDOW_TRAJECTORY_FAMILY_ID:
        sealed_trajectory = sealed_trajectory_suite_coordinates(planning_contract_context)
        assert sealed_trajectory is not None
        return _build_trajectory_request(
            context,
            sealed=sealed_trajectory,
            variable_roster=variable_roster,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            direct_comparator_literature_keys=direct_comparator_literature_keys,
            comparison_literature_keys=comparison_literature_keys,
            comparator_titles=comparator_titles,
            required_primary_cohort_selection_mode=required_primary_cohort_selection_mode,
        )
    if family_id == LANDMARK_SURVIVAL_FAMILY_ID:
        sealed = sealed_survival_suite_coordinates(planning_contract_context)
        assert sealed is not None
        return _build_survival_request(
            context,
            sealed=sealed,
            variable_roster=variable_roster,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            direct_comparator_literature_keys=direct_comparator_literature_keys,
            comparison_literature_keys=comparison_literature_keys,
            comparator_titles=comparator_titles,
            required_primary_cohort_selection_mode=required_primary_cohort_selection_mode,
        )
    if family_id == PREDICTION_FAMILY_ID:
        return _build_prediction_request(
            context,
            variable_roster=variable_roster,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            direct_comparator_literature_keys=direct_comparator_literature_keys,
            comparison_literature_keys=comparison_literature_keys,
            comparator_titles=comparator_titles,
            required_primary_cohort_selection_mode=required_primary_cohort_selection_mode,
        )
    if family_id == PHENOTYPING_FAMILY_ID:
        return _build_phenotyping_request(
            context,
            variable_roster=variable_roster,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            direct_comparator_literature_keys=direct_comparator_literature_keys,
            comparison_literature_keys=comparison_literature_keys,
            comparator_titles=comparator_titles,
            required_primary_cohort_selection_mode=required_primary_cohort_selection_mode,
        )
    if family_id == DESCRIPTIVE_FAMILY_ID:
        return _build_descriptive_request(
            context,
            variable_roster=variable_roster,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            direct_comparator_literature_keys=direct_comparator_literature_keys,
            comparison_literature_keys=comparison_literature_keys,
            comparator_titles=comparator_titles,
            required_primary_cohort_selection_mode=required_primary_cohort_selection_mode,
        )
    roster = list(dict.fromkeys(str(value).strip() for value in variable_roster if str(value).strip()))
    variables = {item.name: item for item in context.variables}
    exposure = str(context.primary_exposure or "").strip()
    outcome = str(context.target_outcome or "").strip()
    optional_roster = set(_structurally_available_roster(context, roster))
    exposure_kind: ExposureKind = (
        "categorical" if family_id == LANDMARK_CATEGORICAL_FAMILY_ID else "continuous"
    )
    exposure_levels = _levels(context, exposure) if exposure_kind == "categorical" else []
    companions = (
        [name for name in exposure_companion_columns(context, exposure) if name in roster]
        if exposure_kind == "continuous"
        else []
    )
    outcome_levels = _levels(context, outcome)
    timing = _landmark_timing_spec(context)
    assert timing is not None  # guaranteed by family_template_id_for_context
    landmark = primary_landmark_hours(context)
    assert landmark is not None
    preferences = context.user_preferences
    specs = list(preferences.sensitivity_specs) if preferences is not None else []
    adjustment = AdjustmentSetAuthority.from_context(context)
    for column in (timing.event_time_variable, timing.observation_duration_variable):
        if column not in variables:
            raise FamilySpecError(
                "family_spec_landmark_column_unavailable",
                f"landmark timing column {column!r} is absent from the sealed context",
                path="landmark_spec",
            )
    alternates = [
        SensitivityAxisBinding(
            spec_id=spec.spec_id,
            axis=str(spec.axis),
            strategy=str(spec.strategy),
            execution_variables=list(spec.execution_variables),
        )
        for spec in specs
        if exposure_kind == "categorical"
        and str(spec.axis) == "exposure_definition"
        and str(spec.strategy) == "alternate_exposure"
        and len(spec.execution_variables) == 1
        and spec.execution_variables[0] in variables
    ]
    first_stay = next(
        (
            SensitivityAxisBinding(
                spec_id=spec.spec_id,
                axis=str(spec.axis),
                strategy=str(spec.strategy),
                execution_variables=list(spec.execution_variables),
            )
            for spec in specs
            if str(spec.axis) == "repeated_stays"
            and str(spec.strategy) == "first_stay"
            and len(spec.execution_variables) == 1
            and spec.execution_variables[0] in variables
        ),
        None,
    )
    complete_case_spec_id = next(
        (
            spec.spec_id
            for spec in specs
            if str(spec.axis) == "missing_data" and str(spec.strategy) == "complete_case"
        ),
        None,
    )
    operationalizations = dict(adjustment.operationalizations)
    functional_form_spec_ids = {
        operationalizations.get(spec.execution_variables[0], spec.execution_variables[0]): spec.spec_id
        for spec in specs
        if str(spec.axis) == "functional_form"
        and str(spec.strategy) == "restricted_cubic_spline"
        and len(spec.execution_variables) == 1
        # The exposure's own spline is the signed primary estimator, not a
        # covariate functional-form check.
        and operationalizations.get(spec.execution_variables[0], spec.execution_variables[0]) != exposure
    }
    ordinal = resolve_ordinal_multi_outcome_contract(context)
    secondary = ordinal.continuous_outcome if ordinal is not None else None
    dependence = context_dependence_authority(context)
    used = {
        exposure,
        outcome,
        timing.event_time_variable,
        timing.observation_duration_variable,
        *(item.execution_variables[0] for item in alternates),
        *(first_stay.execution_variables if first_stay else ()),
        *([secondary] if secondary else []),
        *adjustment.operational_covariates,
        *context.cohort.outcome_columns,
        *companions,
    }
    candidates = _candidates(
        context,
        variable_roster=[name for name in roster if name in optional_roster],
        adjustment=adjustment,
        design_columns=frozenset(used) - set(adjustment.operational_covariates),
    )
    selectable_names = {item.name for item in candidates if item.selectable}
    # Unassigned ``other``-role columns are measurement context for the audit
    # step.  A column the host cannot time stays audit context even though it
    # is also listed as an unselectable candidate: the audit describes it, and
    # only an exact user roster can model it.
    measurement_audit_columns = [
        name
        for name in roster
        if name not in used
        and name not in selectable_names
        and name in variables
        and name in optional_roster
        and str(getattr(variables[name].role, "value", variables[name].role)) == "other"
    ]
    # Every variable the template can place in the selected design's
    # ``required_variables`` needs a Planner-authored reader label; the
    # identity column is rendered as a coordinate, not a label.
    required_label_keys = [
        name
        for name in dict.fromkeys(
            [
                exposure,
                outcome,
                timing.event_time_variable,
                timing.observation_duration_variable,
                *([secondary] if secondary else []),
                *(item.execution_variables[0] for item in alternates),
                *(first_stay.execution_variables if first_stay else ()),
                *measurement_audit_columns,
                *(item.name for item in candidates if item.selectable),
            ]
        )
        if name in variables and name != context.cohort.id_columns[0]
    ]
    cohort_fields = _typed_cohort_fields(
        context, required_primary_cohort_selection_mode, require_typed_bound=True
    )
    reference_index = 0
    contrast_index = len(exposure_levels) - 1 if exposure_levels else 0
    return FamilySpecRequest(
        family_id=family_id,
        research_question=str(context.research_question or "").strip() or "(no question text)",
        cohort_name=str(context.cohort.cohort_name),
        **cohort_fields,
        identity_column=context.cohort.id_columns[0],
        cluster_unit="patient" if dependence is not None else None,
        primary_exposure=exposure,
        exposure_kind=exposure_kind,
        exposure_levels=exposure_levels,
        reference_level_index=reference_index,
        primary_contrast_level_index=contrast_index,
        exposure_is_ordered=bool(
            exposure_kind == "categorical"
            and (
                variables[exposure].is_ordinal
                or str(getattr(variables[exposure].role, "value", variables[exposure].role)) == "ordinal_score"
            )
        ),
        exposure_companion_columns=companions,
        outcome=outcome,
        outcome_levels=outcome_levels,
        event_level_index=len(outcome_levels) - 1,
        landmark_hours=float(landmark),
        landmark_spec_id=timing.spec_id,
        event_time_column=timing.event_time_variable,
        observation_duration_column=timing.observation_duration_variable,
        observation_duration_unit=str(timing.observation_duration_unit or "hours"),
        secondary_continuous_outcome=secondary,
        adjustment_selection=adjustment.selection,
        exact_roster=list(adjustment.operational_covariates) if adjustment.selection == "exact" else [],
        exact_rationales=dict(adjustment.operational_rationales) if adjustment.selection == "exact" else {},
        exact_temporal_roles=dict(adjustment.operational_temporal_roles) if adjustment.selection == "exact" else {},
        adjustment_candidates=candidates,
        alternate_exposures=alternates,
        first_stay=first_stay,
        complete_case_spec_id=complete_case_spec_id,
        functional_form_spec_ids=functional_form_spec_ids,
        measurement_audit_columns=measurement_audit_columns,
        required_reader_label_keys=required_label_keys,
        allowed_literature_citation_keys=list(dict.fromkeys(allowed_literature_citation_keys)),
        direct_comparator_literature_keys=list(dict.fromkeys(direct_comparator_literature_keys)),
        comparison_literature_keys=list(dict.fromkeys(comparison_literature_keys)),
        comparator_titles={
            str(key): " ".join(str(value or "").split())
            for key, value in (comparator_titles or {}).items()
            if str(value or "").strip()
        },
        variable_roster=roster,
    )


def _binary_level_label_keys(context: ResearchContext, exposure: str) -> list[str]:
    """Level label keys the foundation validator demands for a binary primary exposure."""

    variable = context.variable(exposure)
    domain = getattr(variable, "observed_domain", None) if variable is not None else None
    if not isinstance(domain, Mapping) or not domain.get("is_binary"):
        return []
    levels = _levels(context, exposure)
    if sorted(levels) != ["0", "1"]:
        return []
    return [f"{exposure}=0", f"{exposure}=1"]


def _build_descriptive_request(
    context: ResearchContext,
    *,
    variable_roster: Sequence[str],
    allowed_literature_citation_keys: Sequence[str],
    direct_comparator_literature_keys: Sequence[str],
    comparison_literature_keys: Sequence[str],
    comparator_titles: Mapping[str, str] | None,
    required_primary_cohort_selection_mode: str | None,
) -> FamilySpecRequest:
    """Seal the host authority for one descriptive exposure–outcome attempt."""

    roster = list(dict.fromkeys(str(value).strip() for value in variable_roster if str(value).strip()))
    variables = {item.name: item for item in context.variables}
    exposure = str(context.primary_exposure or "").strip()
    outcome = str(context.target_outcome or "").strip()
    exposure_levels = _levels(context, exposure)
    outcome_levels = _levels(context, outcome)
    companions = [name for name in exposure_companion_columns(context, exposure) if name in roster]
    dependence = context_dependence_authority(context)
    design_columns = frozenset(
        {exposure, outcome, context.cohort.id_columns[0], *companions, *context.cohort.outcome_columns}
    )
    optional_roster = _structurally_available_roster(context, roster)
    candidates = _baseline_candidates(
        context, variable_roster=optional_roster, design_columns=design_columns
    )
    selectable_names = {item.name for item in candidates if item.selectable}
    measurement_audit_columns = [
        name
        for name in optional_roster
        if name not in design_columns
        and name not in selectable_names
        and name in variables
        and str(getattr(variables[name].role, "value", variables[name].role)) == "other"
    ]
    level_label_keys = _binary_level_label_keys(context, exposure)
    required_label_keys = [
        name
        for name in dict.fromkeys([exposure, outcome, *(item.name for item in candidates if item.selectable)])
        if name in variables and name != context.cohort.id_columns[0]
    ]
    cohort_fields = _typed_cohort_fields(
        context, required_primary_cohort_selection_mode, require_typed_bound=True
    )
    return FamilySpecRequest(
        family_id=DESCRIPTIVE_FAMILY_ID,
        analysis_type="descriptive_epidemiology",
        research_question=str(context.research_question or "").strip() or "(no question text)",
        cohort_name=str(context.cohort.cohort_name),
        **cohort_fields,
        identity_column=context.cohort.id_columns[0],
        cluster_unit="patient" if dependence is not None else None,
        primary_exposure=exposure,
        exposure_kind="categorical",
        exposure_levels=exposure_levels,
        reference_level_index=0,
        primary_contrast_level_index=len(exposure_levels) - 1,
        exposure_is_ordered=bool(
            variables[exposure].is_ordinal
            or str(getattr(variables[exposure].role, "value", variables[exposure].role)) == "ordinal_score"
        ),
        exposure_companion_columns=[],
        outcome=outcome,
        outcome_levels=outcome_levels,
        event_level_index=len(outcome_levels) - 1,
        observation_window_hours=host_outer_feature_window_end_hours(context),
        level_label_keys=level_label_keys,
        counts_only=descriptive_counts_only_required(
            context, analysis_type="descriptive_epidemiology"
        ),
        adjustment_selection="planner_selectable",
        adjustment_candidates=candidates,
        measurement_audit_columns=measurement_audit_columns,
        required_reader_label_keys=required_label_keys,
        allowed_literature_citation_keys=list(dict.fromkeys(allowed_literature_citation_keys)),
        direct_comparator_literature_keys=list(dict.fromkeys(direct_comparator_literature_keys)),
        comparison_literature_keys=list(dict.fromkeys(comparison_literature_keys)),
        comparator_titles={
            str(key): " ".join(str(value or "").split())
            for key, value in (comparator_titles or {}).items()
            if str(value or "").strip()
        },
        variable_roster=roster,
    )


def _build_feasibility_request(
    context: ResearchContext,
    *,
    sealed: SealedFeasibilityCoordinates,
    variable_roster: Sequence[str],
    allowed_literature_citation_keys: Sequence[str],
    direct_comparator_literature_keys: Sequence[str],
    comparison_literature_keys: Sequence[str],
    comparator_titles: Mapping[str, str] | None,
    required_primary_cohort_selection_mode: str | None,
) -> FamilySpecRequest:
    """Seal the host authority for one fail-closed source-feasibility attempt.

    The Planner supplies comparator applications only; there is no exposure,
    outcome, roster or label to decide because the reviewed protocol found
    the requested contrast non-identifiable from the current source.
    """

    roster = list(dict.fromkeys(str(value).strip() for value in variable_roster if str(value).strip()))
    identity = context.cohort.id_columns[0]
    variables = {item.name for item in context.variables}
    # The audited source columns the cohort-accounting step reads and the
    # reviewer sees in the design: the declared exposure/outcome first, then
    # the remaining roster, never the identity column.
    audited_columns = [
        name
        for name in dict.fromkeys(
            [
                str(context.primary_exposure or "").strip(),
                str(context.target_outcome or "").strip(),
                *roster,
            ]
        )
        if name and name != identity and name in variables
    ][:3]
    dependence = context_dependence_authority(context)
    cohort_fields = _typed_cohort_fields(
        context, required_primary_cohort_selection_mode, require_typed_bound=False
    )
    return FamilySpecRequest(
        family_id=SOURCE_FEASIBILITY_FAMILY_ID,
        analysis_type="causal_inference",
        research_question=str(context.research_question or "").strip() or "(no question text)",
        cohort_name=str(context.cohort.cohort_name),
        **cohort_fields,
        identity_column=context.cohort.id_columns[0],
        cluster_unit="patient" if dependence is not None else None,
        primary_exposure="",
        exposure_kind="none",
        exposure_levels=[],
        reference_level_index=0,
        primary_contrast_level_index=0,
        exposure_is_ordered=False,
        outcome="",
        outcome_levels=[],
        event_level_index=0,
        sealed_feasibility=sealed,
        adjustment_selection="planner_selectable",
        adjustment_candidates=[],
        measurement_audit_columns=audited_columns,
        required_reader_label_keys=audited_columns,
        allowed_literature_citation_keys=list(dict.fromkeys(allowed_literature_citation_keys)),
        direct_comparator_literature_keys=list(dict.fromkeys(direct_comparator_literature_keys)),
        comparison_literature_keys=list(dict.fromkeys(comparison_literature_keys)),
        comparator_titles={
            str(key): " ".join(str(value or "").split())
            for key, value in (comparator_titles or {}).items()
            if str(value or "").strip()
        },
        variable_roster=list(dict.fromkeys([identity, *audited_columns, *roster])),
    )


def _build_trajectory_request(
    context: ResearchContext,
    *,
    sealed: SealedTrajectoryCoordinates,
    variable_roster: Sequence[str],
    allowed_literature_citation_keys: Sequence[str],
    direct_comparator_literature_keys: Sequence[str],
    comparison_literature_keys: Sequence[str],
    comparator_titles: Mapping[str, str] | None,
    required_primary_cohort_selection_mode: str | None,
) -> FamilySpecRequest:
    """Seal the host authority for one fixed-window trajectory suite attempt.

    The Planner labels the sealed coordinate concepts and the outcome and
    writes comparator applications; it selects no feature, model, cluster
    count or threshold.
    """

    roster = list(dict.fromkeys(str(value).strip() for value in variable_roster if str(value).strip()))
    variables = {item.name: item for item in context.variables}
    outcome = str(context.target_outcome or "").strip()
    outcome_levels = _levels(context, outcome)
    dependence = context_dependence_authority(context)
    optional_roster = _structurally_available_roster(context, roster)
    design_columns = frozenset(
        {outcome, context.cohort.id_columns[0], *context.cohort.outcome_columns}
    )
    measurement_audit_columns = [
        name
        for name in optional_roster
        if name not in design_columns
        and name in variables
        and (
            name in sealed.coordinate_concepts
            or name in sealed.descriptive_only_concepts
        )
    ]
    required_label_keys = [
        name
        for name in dict.fromkeys([outcome, *sealed.coordinate_concepts, *sealed.descriptive_only_concepts])
        if name in variables and name != context.cohort.id_columns[0]
    ]
    cohort_fields = _typed_cohort_fields(
        context, required_primary_cohort_selection_mode, require_typed_bound=False
    )
    return FamilySpecRequest(
        family_id=FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
        analysis_type="trajectory_clustering",
        research_question=str(context.research_question or "").strip() or "(no question text)",
        cohort_name=str(context.cohort.cohort_name),
        **cohort_fields,
        identity_column=context.cohort.id_columns[0],
        cluster_unit="patient" if dependence is not None else None,
        primary_exposure="",
        exposure_kind="none",
        exposure_levels=[],
        reference_level_index=0,
        primary_contrast_level_index=0,
        exposure_is_ordered=False,
        outcome=outcome,
        outcome_levels=outcome_levels,
        event_level_index=len(outcome_levels) - 1,
        observation_window_hours=float(sealed.window_hours[1]),
        sealed_trajectory=sealed,
        adjustment_selection="planner_selectable",
        adjustment_candidates=[],
        measurement_audit_columns=measurement_audit_columns,
        required_reader_label_keys=required_label_keys,
        allowed_literature_citation_keys=list(dict.fromkeys(allowed_literature_citation_keys)),
        direct_comparator_literature_keys=list(dict.fromkeys(direct_comparator_literature_keys)),
        comparison_literature_keys=list(dict.fromkeys(comparison_literature_keys)),
        comparator_titles={
            str(key): " ".join(str(value or "").split())
            for key, value in (comparator_titles or {}).items()
            if str(value or "").strip()
        },
        variable_roster=roster,
    )


def _build_survival_request(
    context: ResearchContext,
    *,
    sealed: SealedSuiteCoordinates,
    variable_roster: Sequence[str],
    allowed_literature_citation_keys: Sequence[str],
    direct_comparator_literature_keys: Sequence[str],
    comparison_literature_keys: Sequence[str],
    comparator_titles: Mapping[str, str] | None,
    required_primary_cohort_selection_mode: str | None,
) -> FamilySpecRequest:
    """Seal the host authority for one landmark survival suite attempt.

    Every scientific coordinate is the sealed suite's; the Planner supplies
    reader labels for the exposure, endpoint and adjustment columns plus the
    comparator application sentences.  The exact roster is published as the
    suite's adjustment columns so the shared spec validator treats it as user
    authority that cannot be edited.
    """

    roster = list(dict.fromkeys(str(value).strip() for value in variable_roster if str(value).strip()))
    variables = {item.name: item for item in context.variables}
    exposure = sealed.exposure_status_column
    outcome = sealed.event_column
    exposure_levels = _levels(context, exposure)
    outcome_levels = _levels(context, outcome)
    if len(exposure_levels) != 2 or len(outcome_levels) != 2:
        raise FamilySpecError(
            "family_spec_survival_levels_unavailable",
            "the sealed survival suite needs a binary exposure status and a 0/1 event column",
            path="primary_exposure",
        )
    dependence = context_dependence_authority(context)
    rationales = {
        name: (
            "Sealed by the reviewed landmark survival suite as a prespecified "
            "baseline adjustment column."
        )
        for name in sealed.adjustment_columns
    }
    def _sealed_coding(name: str) -> list[str]:
        level_count = len(_levels(context, name))
        if level_count == 2:
            return ["binary"]
        if level_count > 2:
            return ["categorical"]
        return ["continuous"]

    candidates = [
        AdjustmentCandidate(
            name=name,
            semantic_role=str(getattr(variables[name].role, "value", variables[name].role)),
            host_temporal_role=None,
            allowed_codings=_sealed_coding(name),
            closed_domain_size=len(_levels(context, name)) or None,
            selectable=False,
            boundary="sealed by the landmark survival suite; not Planner-selectable",
        )
        for name in sealed.adjustment_columns
        if name in variables
    ]
    # Every sealed source column enters the reviewed design, so each needs a
    # reader label (the compiler refuses an unlabeled required variable).
    required_label_keys = [
        name
        for name in dict.fromkeys([exposure, outcome, *sealed.source_columns])
        if name in variables and name != context.cohort.id_columns[0]
    ]
    cohort_fields = _typed_cohort_fields(
        context, required_primary_cohort_selection_mode, require_typed_bound=False
    )
    return FamilySpecRequest(
        family_id=LANDMARK_SURVIVAL_FAMILY_ID,
        analysis_type="survival",
        research_question=str(context.research_question or "").strip() or "(no question text)",
        cohort_name=str(context.cohort.cohort_name),
        **cohort_fields,
        identity_column=context.cohort.id_columns[0],
        cluster_unit="patient" if dependence is not None else None,
        primary_exposure=exposure,
        exposure_kind="categorical",
        exposure_levels=exposure_levels,
        reference_level_index=0,
        primary_contrast_level_index=1,
        exposure_is_ordered=False,
        exposure_companion_columns=[],
        outcome=outcome,
        outcome_levels=outcome_levels,
        event_level_index=1,
        level_label_keys=_binary_level_label_keys(context, exposure),
        sealed_suite=sealed,
        adjustment_selection="exact",
        exact_roster=list(sealed.adjustment_columns),
        exact_rationales=rationales,
        adjustment_candidates=candidates,
        required_reader_label_keys=required_label_keys,
        allowed_literature_citation_keys=list(dict.fromkeys(allowed_literature_citation_keys)),
        direct_comparator_literature_keys=list(dict.fromkeys(direct_comparator_literature_keys)),
        comparison_literature_keys=list(dict.fromkeys(comparison_literature_keys)),
        comparator_titles={
            str(key): " ".join(str(value or "").split())
            for key, value in (comparator_titles or {}).items()
            if str(value or "").strip()
        },
        variable_roster=roster,
    )


def _feature_candidates(
    context: ResearchContext,
    *,
    variable_roster: Sequence[str],
    excluded: frozenset[str],
    include_demographics: bool = False,
    encodes_categories: bool = False,
) -> list[AdjustmentCandidate]:
    """Offer numeric, window-bound, non-outcome measurements as fit features.

    ``include_demographics`` admits owner-declared baseline demographics as
    predictors (a prediction model may use age or sex); a phenotype fit keeps
    them for characterization instead.  ``encodes_categories`` is for a fit
    whose training pipeline one-hot encodes a closed-domain category (the
    prediction model): such a category (sex, admission type) is selectable
    like a number.  A phenotype's distance metric has no such encoding.
    """

    reference = host_outer_feature_window_end_hours(context)
    roles = host_window_bound_roles(
        context, reference_hours=reference, dynamic_roles=_BASELINE_DESCRIPTION_ROLES
    )
    variables = {item.name: item for item in context.variables}
    outcome_lineage = {
        *context.cohort.outcome_columns,
        *([context.target_outcome] if context.target_outcome else []),
    }
    candidates: list[AdjustmentCandidate] = []
    for name in variable_roster:
        variable = variables.get(name)
        if variable is None or name in excluded:
            continue
        role = str(getattr(variable.role, "value", variable.role) or "")
        if role in {"id", "time", "meta", "index", "outcome"}:
            continue
        if role == "demographic" and not include_demographics:
            continue
        lineage = {name, str(getattr(variable, "source_concept", "") or "")}
        if lineage & outcome_lineage:
            continue
        numeric = any(token in str(variable.dtype or "").lower() for token in _NUMERIC_DTYPES)
        timed = roles.get(name) in {"at_or_before_time_zero", "baseline_static"}
        levels = _levels(context, name)
        closed = len(levels) if len(levels) >= 2 else None
        encoded_category = bool(encodes_categories and closed and not numeric)
        candidates.append(
            AdjustmentCandidate(
                name=name,
                semantic_role=role,
                host_temporal_role=roles.get(name),
                allowed_codings=(
                    ["binary"] if closed == 2 else ["categorical"] if closed else ["continuous"]
                ),
                closed_domain_size=closed,
                selectable=bool(timed and (numeric or encoded_category)),
                boundary=(
                    "owner-declared baseline demographic"
                    if roles.get(name) == "baseline_static"
                    else "numeric measurement closed inside the sealed observation window"
                    if numeric and timed
                    else "closed-domain category inside the sealed observation window; "
                    "the training pipeline one-hot encodes it"
                    if encoded_category and timed
                    else "not a numeric window-bound measurement; audit context only"
                ),
            )
        )
    return candidates


def _build_prediction_request(
    context: ResearchContext,
    *,
    variable_roster: Sequence[str],
    allowed_literature_citation_keys: Sequence[str],
    direct_comparator_literature_keys: Sequence[str],
    comparison_literature_keys: Sequence[str],
    comparator_titles: Mapping[str, str] | None,
    required_primary_cohort_selection_mode: str | None,
) -> FamilySpecRequest:
    """Seal the host authority for one static binary prediction attempt."""

    roster = list(dict.fromkeys(str(value).strip() for value in variable_roster if str(value).strip()))
    variables = {item.name: item for item in context.variables}
    outcome = str(context.target_outcome or "").strip()
    outcome_levels = _levels(context, outcome)
    identity = context.cohort.id_columns[0]
    dependence = context_dependence_authority(context)
    optional_roster = _structurally_available_roster(context, roster)
    design_columns = frozenset({outcome, identity, *context.cohort.outcome_columns})
    feature_candidates = _feature_candidates(
        context,
        variable_roster=optional_roster,
        excluded=design_columns,
        include_demographics=True,
        encodes_categories=True,
    )
    feature_names = {item.name for item in feature_candidates}
    measurement_audit_columns = [
        name
        for name in optional_roster
        if name not in design_columns
        and name not in feature_names
        and str(getattr(variables[name].role, "value", variables[name].role)) == "other"
    ]
    cohort_fields = _typed_cohort_fields(
        context, required_primary_cohort_selection_mode, require_typed_bound=False
    )
    return FamilySpecRequest(
        family_id=PREDICTION_FAMILY_ID,
        analysis_type="prediction_model",
        research_question=str(context.research_question or "").strip() or "(no question text)",
        cohort_name=str(context.cohort.cohort_name),
        **cohort_fields,
        identity_column=identity,
        cluster_unit="patient" if dependence is not None else None,
        primary_exposure="",
        exposure_kind="none",
        exposure_levels=[],
        reference_level_index=0,
        primary_contrast_level_index=0,
        exposure_is_ordered=False,
        outcome=outcome,
        outcome_levels=outcome_levels,
        event_level_index=len(outcome_levels) - 1,
        observation_window_hours=host_outer_feature_window_end_hours(context),
        level_label_keys=[],
        counts_only=False,
        feature_candidates=feature_candidates,
        membership_candidates=[],
        adjustment_selection="planner_selectable",
        adjustment_candidates=[],
        measurement_audit_columns=measurement_audit_columns,
        required_reader_label_keys=[name for name in [outcome] if name in variables],
        allowed_literature_citation_keys=list(dict.fromkeys(allowed_literature_citation_keys)),
        direct_comparator_literature_keys=list(dict.fromkeys(direct_comparator_literature_keys)),
        comparison_literature_keys=list(dict.fromkeys(comparison_literature_keys)),
        comparator_titles={
            str(key): " ".join(str(value or "").split())
            for key, value in (comparator_titles or {}).items()
            if str(value or "").strip()
        },
        variable_roster=roster,
    )


def _build_phenotyping_request(
    context: ResearchContext,
    *,
    variable_roster: Sequence[str],
    allowed_literature_citation_keys: Sequence[str],
    direct_comparator_literature_keys: Sequence[str],
    comparison_literature_keys: Sequence[str],
    comparator_titles: Mapping[str, str] | None,
    required_primary_cohort_selection_mode: str | None,
) -> FamilySpecRequest:
    """Seal the host authority for one cross-sectional phenotyping attempt."""

    roster = list(dict.fromkeys(str(value).strip() for value in variable_roster if str(value).strip()))
    variables = {item.name: item for item in context.variables}
    exposure = str(context.primary_exposure or "").strip()
    outcome = str(context.target_outcome or "").strip()
    exposure_levels = _levels(context, exposure)
    outcome_levels = _levels(context, outcome)
    identity = context.cohort.id_columns[0]
    dependence = context_dependence_authority(context)
    optional_roster = _structurally_available_roster(context, roster)
    design_columns = frozenset({exposure, outcome, identity, *context.cohort.outcome_columns})
    # A membership flag defines the population (an inclusion predicate evaluated
    # inside the sealed window), not a covariate, so a binary flag may inherit
    # the outer window like any other materialized measurement.
    membership_roles = host_window_bound_roles(
        context,
        reference_hours=host_outer_feature_window_end_hours(context),
        dynamic_roles=_BASELINE_DESCRIPTION_ROLES,
        outer_window_fallback_roles=_BASELINE_DESCRIPTION_ROLES,
    )
    membership_candidates = [
        name
        for name in optional_roster
        if name not in {outcome, identity}
        and name in variables
        and _levels(context, name) in (["0", "1"], ["false", "true"])
        and membership_roles.get(name) == "at_or_before_time_zero"
    ]
    feature_candidates = _feature_candidates(
        context,
        variable_roster=optional_roster,
        excluded=frozenset({*design_columns, *membership_candidates}),
    )
    baseline_candidates = _baseline_candidates(
        context,
        variable_roster=optional_roster,
        design_columns=frozenset({*design_columns, *membership_candidates}),
    )
    feature_names = {item.name for item in feature_candidates}
    baseline_candidates = [
        item for item in baseline_candidates if item.name not in feature_names or item.semantic_role == "demographic"
    ]
    measurement_audit_columns = [
        name
        for name in optional_roster
        if name not in design_columns
        and name not in feature_names
        and name not in {item.name for item in baseline_candidates}
        and name not in membership_candidates
        and str(getattr(variables[name].role, "value", variables[name].role)) == "other"
    ]
    required_label_keys = [name for name in dict.fromkeys([exposure, outcome]) if name in variables]
    cohort_fields = _typed_cohort_fields(
        context, required_primary_cohort_selection_mode, require_typed_bound=False
    )
    return FamilySpecRequest(
        family_id=PHENOTYPING_FAMILY_ID,
        analysis_type="trajectory_clustering",
        research_question=str(context.research_question or "").strip() or "(no question text)",
        cohort_name=str(context.cohort.cohort_name),
        **cohort_fields,
        identity_column=identity,
        cluster_unit="patient" if dependence is not None else None,
        primary_exposure=exposure,
        exposure_kind="categorical",
        exposure_levels=exposure_levels,
        reference_level_index=0,
        primary_contrast_level_index=len(exposure_levels) - 1,
        exposure_is_ordered=False,
        outcome=outcome,
        outcome_levels=outcome_levels,
        event_level_index=len(outcome_levels) - 1,
        observation_window_hours=host_outer_feature_window_end_hours(context),
        level_label_keys=[],
        counts_only=descriptive_counts_only_required(
            context, analysis_type="trajectory_clustering"
        ),
        feature_candidates=feature_candidates,
        membership_candidates=membership_candidates,
        adjustment_selection="planner_selectable",
        adjustment_candidates=baseline_candidates,
        measurement_audit_columns=measurement_audit_columns,
        required_reader_label_keys=required_label_keys,
        allowed_literature_citation_keys=list(dict.fromkeys(allowed_literature_citation_keys)),
        direct_comparator_literature_keys=list(dict.fromkeys(direct_comparator_literature_keys)),
        comparison_literature_keys=list(dict.fromkeys(comparison_literature_keys)),
        comparator_titles={
            str(key): " ".join(str(value or "").split())
            for key, value in (comparator_titles or {}).items()
            if str(value or "").strip()
        },
        variable_roster=roster,
    )


__all__ = [
    "build_family_spec_request",
    "exposure_companion_columns",
    "family_template_id_for_context",
    "sealed_feasibility_coordinates",
    "sealed_survival_suite_coordinates",
    "sealed_trajectory_suite_coordinates",
]
