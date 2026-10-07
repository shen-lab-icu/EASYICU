"""A static prediction model is made for the stays still in the ICU when it predicts.

The prediction family measures its predictors over the first hours after ICU
admission and predicts at the end of that window.  A stay that died or left
the ICU before then is not one the model predicts for, and its outcome may
come before the prediction: an early death scored with the vitals of a dying
stay makes discrimination look better than it is.  The template keeps the
stays whose ICU length of stay exceeds the prediction time, a typed bound
the time-zero rule passes, and the request is refused before any Provider
call when the bound cannot be applied.  Synthetic contexts and rows only.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.agents.family_spec_planner import _population_authority
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
    candidate_analysis_types,
    select_progressive_variables,
)
from easyicu.research_agent.cohort.schema import build_cohort
from easyicu.research_agent.planning.cohort_eligibility import cohort_predicates_after_time_zero
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.family_spec.contract import (
    FamilySpecError,
    FamilySpecRequest,
    population_required,
    spec_from_mapping,
)
from easyicu.research_agent.planning.family_spec.landmark_categorical_template import (
    typed_bound_predicates,
)
from easyicu.research_agent.planning.family_spec.prediction_template import (
    build_prediction_skeleton,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    TimeWindow,
    VariableRole,
)

from .family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _descriptive_context,
    _phenotyping_context,
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
)

FEATURES = ["age", "sex", "hr_max", "lactate_max", "map_min"]


def _with_icu_stay(context: ResearchContext, unit: str | None = "days") -> ResearchContext:
    """The context with its ICU length of stay recorded in ``unit`` (or none)."""

    variables = [item for item in context.variables if item.name != "los_icu"]
    variables.append(
        ConceptDescriptor(
            name="los_icu", description="ICU length of stay", role=VariableRole.OUTCOME,
            dtype="float64", unit=unit, source_concept="los_icu",
        )
    )
    return context.model_copy(update={"variables": variables})


def _without_icu_stay(context: ResearchContext) -> ResearchContext:
    return context.model_copy(
        update={"variables": [item for item in context.variables if item.name != "los_icu"]}
    )


def _with_minimum_stay(context: ResearchContext, hours: float) -> ResearchContext:
    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["cohort"] = {**constraints.get("cohort", {}), "min_icu_los_hours": hours}
    return context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )


def _with_window(context: ResearchContext, hours: float) -> ResearchContext:
    """The context materialized over the first ``hours`` after ICU admission.

    The host's record and the window it declares carry the same hours, as a
    Web launch writes them.
    """

    window = f"icu_admission[0,{hours:g}]h"
    variables = [
        item.model_copy(update={"analysis_window": window})
        if item.analysis_window == "icu_admission[0,24]h"
        else item
        for item in context.variables
    ]
    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["materialization_window"] = {
        **constraints.get("materialization_window", {}), "hours": hours,
    }
    return context.model_copy(
        update={
            "variables": variables,
            "time_windows": [
                TimeWindow(
                    name=f"first_{hours:g}h", anchor="icu_admission", start_hours=0.0,
                    end_hours=hours, rationale="Outer feature-materialization window bound by the host.",
                )
            ],
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            ),
        }
    )


def _skeleton(request: FamilySpecRequest):
    return build_prediction_skeleton(
        request, spec_from_mapping(_prediction_payload(request, features=FEATURES))
    )


def _inclusion(request: FamilySpecRequest) -> list[tuple[str, str, float, float]]:
    cohort = _skeleton(request).foundation.foundation.cohort
    return [
        (item.concept_id, item.op, item.value.number_value, item.end_offset_hours)
        for item in cohort.inclusion
    ]


def _refusal(context: ResearchContext, mode: str | None) -> str:
    with pytest.raises(FamilySpecError) as caught:
        _request(context, cohort_mode=mode)
    return caught.value.reason_code


def test_the_model_is_made_for_stays_still_in_the_icu_after_its_window() -> None:
    context = _prediction_context()
    request = _request(context, cohort_mode=None)

    assert request.prediction_time_hours == request.observation_window_hours == 24.0
    assert request.cohort_selection_mode == "predicate_filtered"
    assert "los_icu" not in {item.name for item in request.feature_candidates}

    llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=FEATURES))],
        required_primary_cohort_selection_mode=None,
    )

    assert len(llm.calls) == 1
    cohort = result.output.cohort
    assert cohort.selection_mode == "predicate_filtered"
    assert [
        (item.concept_id, item.op, item.value, item.time_window.anchor, item.time_window.end_offset_hours)
        for item in cohort.inclusion
    ] == [("los_icu", ">", 1.0, "icu_admission", 24.0)]


def test_the_bound_is_the_end_of_the_window_the_model_predicts_at() -> None:
    request = _request(_with_window(_prediction_context(), 6.0), cohort_mode=None)

    assert request.prediction_time_hours == 6.0
    assert _inclusion(request) == [("los_icu", ">", 0.25, 6.0)]


def test_the_host_keeps_only_the_stays_still_in_the_icu_at_the_prediction_time() -> None:
    context = _prediction_context()
    request = _request(context, cohort_mode=None)
    _llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=FEATURES))],
        required_primary_cohort_selection_mode=None,
    )
    # Died at 5 h, discharged at about 22 h, left exactly at 24 h, still in
    # the ICU just after 24 h, three days, and an unrecorded stay length.
    rows = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "los_icu": [5 / 24, 0.9, 1.0, 1.01, 3.0, math.nan],
            "death": [True, False, True, False, True, False],
        }
    )

    kept = build_cohort(result.output.cohort, rows)

    assert kept["stay_id"].tolist() == [4, 5]


def test_the_time_zero_rule_passes_the_bound_and_refuses_a_later_one() -> None:
    context = _prediction_context()
    request = _request(context, cohort_mode=None)
    _llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=FEATURES))],
        required_primary_cohort_selection_mode=None,
    )
    bound = result.output.cohort.inclusion[0].to_dict()

    assert cohort_predicates_after_time_zero(
        context, inclusion=[bound], exclusion=[], time_zero_hours=24.0
    ) == ()
    later = {**bound, "value": 25 / 24}
    [found] = cohort_predicates_after_time_zero(
        context, inclusion=[later], exclusion=[], time_zero_hours=24.0
    )
    assert found.reason == "icu_stay_length"


def test_a_prediction_without_the_icu_length_of_stay_is_refused_before_the_planner_call() -> None:
    context = _without_icu_stay(_prediction_context())

    assert _refusal(context, None) == "family_spec_prediction_risk_set_unavailable"
    llm = ScriptedMockLLMClient([])
    with pytest.raises(Exception, match="family_spec_prediction_risk_set_unavailable"):
        ProgressivePlannerAgent(llm).run_attempt(
            context,
            planner_strategy=FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=ALLOWED_CITATIONS,
            direct_comparator_literature_keys=DIRECT_COMPARATORS,
            comparison_literature_keys=DIRECT_COMPARATORS,
            enforce_article_contract=True,
            article_contract_context=context,
            planning_contract_context="",
            required_primary_cohort_selection_mode=None,
        )
    assert llm.calls == []


def test_a_caller_bound_filter_is_bounded_by_the_prediction_time() -> None:
    """A reviewed candidate's filtered prediction cohort keeps its prediction-time bound.

    The next pass is bound to a predicate-filtered cohort.  The prediction
    time filters it, so no stated population is required, and none is
    refused for want of concepts to state one with.
    """

    context = _prediction_context()
    offered = _request(context, cohort_mode="predicate_filtered")
    bare = build_family_spec_request(
        context,
        analysis_types=candidate_analysis_types(context),
        variable_roster=select_progressive_variables(context),
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        required_primary_cohort_selection_mode="predicate_filtered",
    )

    assert offered.population_concepts and not population_required(offered)
    authority = _population_authority(offered)
    assert authority["population_required"] is False
    assert authority["already_applied"] == {"still_in_icu_after_prediction_time_hours": 24.0}
    assert (bare.population_concepts, bare.prediction_time_hours) == ([], 24.0)


def test_a_caller_binding_every_input_row_is_refused_not_unfiltered() -> None:
    assert (
        _refusal(_prediction_context(), "all_input_rows")
        == "family_spec_prediction_risk_set_conflicts_with_population"
    )


@pytest.mark.parametrize(
    ("unit", "stored", "value"),
    [("days", "days", 1.0), (None, "days", 1.0), ("d", "days", 1.0), ("hours", "hours", 24.0), ("h", "hours", 24.0)],
)
def test_the_bound_is_written_in_the_unit_the_roster_records(
    unit: str | None, stored: str, value: float
) -> None:
    context = _with_icu_stay(_prediction_context(), unit)
    request = _request(context, cohort_mode=None)

    assert request.icu_stay_unit == stored
    assert _inclusion(request) == [("los_icu", ">", value, 24.0)]
    # The time-zero rule reads the same unit, so it passes the bound as written.
    predicate = _skeleton(request).foundation.foundation.cohort.inclusion[0]
    assert cohort_predicates_after_time_zero(
        context,
        inclusion=[
            {
                "concept_id": predicate.concept_id,
                "time_window": {
                    "anchor": predicate.anchor,
                    "start_offset_hours": predicate.start_offset_hours,
                    "end_offset_hours": predicate.end_offset_hours,
                },
                "aggregation": predicate.aggregation,
                "op": predicate.op,
                "value": predicate.value.number_value,
            }
        ],
        exclusion=[],
        time_zero_hours=24.0,
    ) == ()


def _with_stay_companions(context: ResearchContext) -> ResearchContext:
    """The ICU length of stay's companions: its observation times (h) and count."""

    companions = [
        ConceptDescriptor(
            name=f"los_icu_{suffix}", role=role, dtype="float64", unit=unit,
            source_concept="los_icu", analysis_window="icu_admission[0,24]h",
        )
        for suffix, role, unit in (
            ("first_time", VariableRole.TIME, "h"),
            ("last_time", VariableRole.TIME, "h"),
            ("n", VariableRole.META, None),
        )
    ]
    return context.model_copy(update={"variables": [*context.variables, *companions]})


def test_the_bound_is_read_from_the_icu_length_of_stay_not_its_companions() -> None:
    context = _with_stay_companions(_prediction_context())
    request = _request(context, cohort_mode=None)

    assert request.icu_stay_unit == "days"
    assert _inclusion(request) == [("los_icu", ">", 1.0, 24.0)]
    # Companions alone are no ICU length of stay: their hours time an
    # observation, and their count is no duration.
    assert (
        _refusal(_with_stay_companions(_without_icu_stay(_prediction_context())), None)
        == "family_spec_prediction_risk_set_unavailable"
    )


@pytest.mark.parametrize("unit", ["min", "weeks", "hours since admission"])
def test_an_icu_stay_in_an_unread_unit_is_refused(unit: str) -> None:
    context = _with_icu_stay(_prediction_context(), unit)

    assert _refusal(context, None) == "family_spec_icu_stay_unit_unread"
    # A minimum stay of another family reads the same unit.
    phenotype = _with_minimum_stay(_with_icu_stay(_phenotyping_context(), unit), 24.0)
    assert _refusal(phenotype, "predicate_filtered") == "family_spec_icu_stay_unit_unread"


def test_a_minimum_stay_is_written_in_the_unit_the_roster_records() -> None:
    context = _with_minimum_stay(_with_icu_stay(_phenotyping_context(), "hours"), 24.0)
    request = _request(context, cohort_mode="predicate_filtered")

    assert (request.minimum_icu_hours, request.icu_stay_unit) == (24.0, "hours")
    assert [
        (item.concept_id, item.op, item.value.number_value)
        for item in typed_bound_predicates(request, end_hours=24.0)
    ] == [("los_icu", ">=", 24.0)]


@pytest.mark.parametrize("hours", [12.0, 24.0])
def test_a_minimum_stay_up_to_the_prediction_time_adds_no_second_bound(hours: float) -> None:
    request = _request(_with_minimum_stay(_prediction_context(), hours), cohort_mode="predicate_filtered")

    assert request.minimum_icu_hours == hours
    assert _inclusion(request) == [("los_icu", ">", 1.0, 24.0)]


def test_a_minimum_stay_beyond_the_prediction_time_is_still_refused() -> None:
    context = _with_minimum_stay(_prediction_context(), 48.0)

    assert _refusal(context, "predicate_filtered") == "family_spec_cohort_eligibility_after_time_zero"


def test_the_planner_is_told_the_bound_is_already_applied() -> None:
    request = _request(_prediction_context(), cohort_mode=None)

    applied = _population_authority(request)["already_applied"]

    assert applied["still_in_icu_after_prediction_time_hours"] == 24.0


def test_the_plan_states_the_prediction_time_and_what_it_leaves_unchecked() -> None:
    request = _request(_prediction_context(), cohort_mode=None)
    selected = next(
        item
        for item in _skeleton(request).outline.design_selection.candidates
        if item.disposition == "selected"
    )

    assert selected.time_zero.startswith("The prediction time, 24 h after ICU admission")
    assert "still in the ICU after 24 h" in selected.observation_window
    # The old wording claimed an ordering nothing enforced.
    assert "after the window" not in selected.observation_window
    assert any(
        "an outcome event that does not end the ICU stay before it is not excluded" in item
        for item in selected.assumptions
    )
    population, _predictors, outcome = selected.reviewable_plan[:3]
    assert population.startswith(
        "Analysis rows of the study cohort that are still in the ICU after the prediction time "
        "(24 h after ICU admission)"
    )
    assert "left the ICU, alive or dead, by the prediction time is not analyzed" in outcome


def test_the_plan_states_the_prediction_time_in_chinese() -> None:
    context = _prediction_context().model_copy(
        update={"research_question": "成人 ICU 入住中，入 ICU 后 24 小时的生命体征与化验能否预测院内死亡？"}
    )
    request = _request(context, cohort_mode=None)
    selected = next(
        item
        for item in _skeleton(request).outline.design_selection.candidates
        if item.disposition == "selected"
    )

    population, _predictors, outcome = selected.reviewable_plan[:3]
    assert population.startswith("研究队列中在预测时点（ICU 入院后 24 h）之后仍在 ICU 内的分析行")
    assert "在预测时点之前离开 ICU（存活或死亡）的入住不纳入分析" in outcome


def test_requests_without_a_prediction_time_keep_their_identity() -> None:
    request = _request(_descriptive_context(), cohort_mode=None)
    payload = request.model_dump(mode="json")

    assert request.prediction_time_hours is None and request.icu_stay_unit == "days"
    assert "prediction_time_hours" not in payload and "icu_stay_unit" not in payload


def test_the_request_contract_binds_the_prediction_time_to_its_family_and_window() -> None:
    prediction = _request(_prediction_context(), cohort_mode=None).model_dump(mode="json")
    descriptive = _request(_descriptive_context(), cohort_mode=None).model_dump(mode="json")

    with pytest.raises(ValidationError, match="end of the observation window"):
        FamilySpecRequest.model_validate({**prediction, "prediction_time_hours": 12.0})
    with pytest.raises(ValidationError, match="end of the observation window"):
        FamilySpecRequest.model_validate({**prediction, "cohort_selection_mode": "all_input_rows"})
    with pytest.raises(ValidationError, match="belongs to the prediction family"):
        FamilySpecRequest.model_validate({**descriptive, "prediction_time_hours": 24.0})
    with pytest.raises(ValidationError, match="belongs to a typed ICU-stay bound"):
        FamilySpecRequest.model_validate({**descriptive, "icu_stay_unit": "hours"})
