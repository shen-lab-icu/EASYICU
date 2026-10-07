"""A prediction over every input row is made for the stays its model predicts for.

A caller that binds every input row (all ICU stays of a study's export) states
the population and leaves the plan no filter of its own.  A static prediction
model still predicts only for the stays in the ICU when it predicts, so its
family refused such a study before any Provider call: a common question,
"among ICU stays, how well do first-24-hour data predict death?", could not be
planned.  The family now keeps the caller's population and adds only its risk
set, the stays still in the ICU after the prediction time; the study's other
typed bounds describe that population and are not applied again.  The check of
a caller-bound population admits exactly that risk set, compared as typed
predicates, and nothing else.  Synthetic contexts and rows only.
"""

from __future__ import annotations

import inspect
import json
import math

import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent import pipeline as pipeline_module
from easyicu.research_agent.agents.family_spec_planner import _population_authority
from easyicu.research_agent.cohort.schema import build_cohort
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.planning.family_spec.contract import FamilySpecRequest
from easyicu.research_agent.reporting.population_selection import analyzed_population
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)

from .family_spec_fixtures import (
    _descriptive_context,
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
)

FEATURES = ["age", "sex", "hr_max", "lactate_max", "map_min"]
CONFLICT = (
    "Planner primary cohort selection mode does not match the caller-bound contract"
)


def _with_bounds(context: ResearchContext) -> ResearchContext:
    """An adult study with a minimum ICU stay, as the Web study's cohort records it."""

    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["cohort"] = {"age_min": 18, "min_icu_los_hours": 12}
    return context.model_copy(
        update={
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            )
        }
    )


def _in_hours(context: ResearchContext) -> ResearchContext:
    variables = [item for item in context.variables if item.name != "los_icu"]
    variables.append(
        ConceptDescriptor(
            name="los_icu",
            description="ICU length of stay",
            role=VariableRole.OUTCOME,
            dtype="float64",
            unit="hours",
            source_concept="los_icu",
        )
    )
    return context.model_copy(update={"variables": variables})


def _plan(context: ResearchContext, mode: str = "all_input_rows"):
    request = _request(context, cohort_mode=mode)
    llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=FEATURES))],
        required_primary_cohort_selection_mode=mode,
    )
    assert len(llm.calls) == 1
    return request, result.output


def _conflict(plan, context: ResearchContext, mode: str = "all_input_rows"):
    from easyicu.research_agent.planning.family_spec import (
        caller_bound_population_conflict,
    )

    return caller_bound_population_conflict(
        plan, context=context, required_selection_mode=mode
    )


def _risk_set(value: float = 1.0, *, end: float = 24.0) -> ConceptPredicate:
    return ConceptPredicate(
        concept_id="los_icu",
        time_window=TimeWindow(
            anchor="icu_admission", start_offset_hours=0.0, end_offset_hours=end
        ),
        aggregation="first",
        op=">",
        value=value,
    )


def _adults() -> ConceptPredicate:
    return ConceptPredicate(
        concept_id="age",
        time_window=TimeWindow(
            anchor="icu_admission", start_offset_hours=0.0, end_offset_hours=24.0
        ),
        aggregation="first",
        op=">=",
        value=18.0,
    )


def _with_cohort(plan, *, inclusion=(), exclusion=()):
    cohort = CohortDefinition(
        name=plan.cohort.name, inclusion=tuple(inclusion), exclusion=tuple(exclusion)
    )
    return plan.model_copy(update={"cohort": cohort})


def test_a_caller_binding_every_input_row_keeps_it_and_adds_only_the_risk_set() -> None:
    context = _with_bounds(_prediction_context())

    request, plan = _plan(context)

    # Before: refused before the Planner call
    # (family_spec_prediction_risk_set_conflicts_with_population).
    assert request.caller_binds_all_input_rows
    assert request.cohort_selection_mode == "predicate_filtered"
    # The adult bound and the minimum stay describe the caller's population.
    assert (request.age_min, request.minimum_icu_hours) == (18.0, 12.0)
    assert plan.cohort.selection_mode == "predicate_filtered"
    assert plan.cohort.inclusion == (_risk_set(),) and plan.cohort.exclusion == ()


def test_the_check_of_a_caller_bound_population_admits_that_risk_set() -> None:
    context = _prediction_context()
    _request_, plan = _plan(context)

    assert _conflict(plan, context) is None


@pytest.mark.parametrize(
    "change",
    [
        {"inclusion": (_risk_set(), _adults())},
        {"inclusion": (_adults(),)},
        {"inclusion": (_risk_set(2.0),)},
        {"inclusion": (_risk_set(24.0),)},
        {"inclusion": (_risk_set(end=48.0),)},
        {"inclusion": (_risk_set(),), "exclusion": (_adults(),)},
    ],
    ids=[
        "an extra predicate",
        "a filter of the Planner's own",
        "another threshold",
        "hours written in a roster of days",
        "another window",
        "an exclusion",
    ],
)
def test_any_other_cohort_is_not_the_population_the_caller_binds(change) -> None:
    context = _prediction_context()
    _request_, plan = _plan(context)

    conflict = _conflict(_with_cohort(plan, **change), context)

    assert conflict is not None and conflict.startswith(CONFLICT)


def test_only_a_prediction_adds_a_risk_set_to_every_input_row() -> None:
    context = _prediction_context()
    _request_, plan = _plan(context)

    conflict = _conflict(
        plan.model_copy(update={"analysis_type": "association_study"}), context
    )

    assert conflict is not None and conflict.startswith(CONFLICT)


def test_the_risk_set_is_compared_in_the_unit_the_roster_records() -> None:
    context = _in_hours(_prediction_context())
    _request_, plan = _plan(context)

    assert plan.cohort.inclusion == (_risk_set(24.0),)
    assert _conflict(plan, context) is None
    # The same stays written in days are not that risk set here.
    days = _with_cohort(plan, inclusion=(_risk_set(1.0),))
    assert _conflict(days, context).startswith(CONFLICT)


def test_a_stated_risk_set_is_not_applied_twice() -> None:
    context = _prediction_context()
    request = _request(context, cohort_mode="predicate_filtered")
    risk_set = {
        "concept_id": "los_icu",
        "anchor": "icu_admission",
        "start_offset_hours": 0,
        "end_offset_hours": 24.0,
        "aggregation": "first",
        "op": ">",
        "value": {
            "mode": "number",
            "string_value": None,
            "number_value": 1.0,
            "boolean_value": None,
            "string_list": [],
            "number_list": [],
        },
    }
    stated = {
        **_prediction_payload(request, features=FEATURES),
        "population": {
            "criteria": [
                {"criterion": "stays longer than a day", "concept_ids": ["los_icu"]}
            ],
            "inclusion": [risk_set],
            "exclusion": [],
        },
    }

    _llm, result = _run(
        context,
        [json.dumps(stated)],
        required_primary_cohort_selection_mode="predicate_filtered",
    )

    assert result.output.cohort.inclusion == (_risk_set(),)


def test_every_step_reads_the_rows_the_model_predicts_for() -> None:
    """Table 1, the model and its validation describe the same risk set."""

    context = _prediction_context()
    _request_, plan = _plan(context)
    # Died at 5 h, discharged at about 22 h, left exactly at 24 h, still in the
    # ICU just after 24 h, three days, and an unrecorded stay length.
    rows = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "los_icu": [5 / 24, 0.9, 1.0, 1.01, 3.0, math.nan],
            "death": [True, False, True, False, True, False],
        }
    )

    assert build_cohort(plan.cohort, rows)["stay_id"].tolist() == [4, 5]
    assert {step.population_scope for step in plan.steps} == {None}
    assert (
        analyzed_population(plan=plan, context=context).source_scope
        == "predicate_selected"
    )


def test_the_plan_names_every_input_row_and_its_risk_set() -> None:
    _request_, plan = _plan(_prediction_context())
    text = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False)

    assert (
        "All input rows of the study cohort that are still in the ICU after the prediction "
        "time (24 h after ICU admission)"
    ) in text
    assert "meet the typed eligibility bound" not in text


def test_the_plan_names_every_input_row_and_its_risk_set_in_chinese() -> None:
    context = _with_bounds(_prediction_context()).model_copy(
        update={
            "research_question": "ICU 入住中，入 ICU 后 24 小时的生命体征与化验能否预测院内死亡？"
        }
    )
    _request_, plan = _plan(context)
    text = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False)

    assert (
        "研究队列全部输入行中在预测时点（ICU 入院后 24 h）之后仍在 ICU 内的分析行"
        in text
    )
    assert "满足类型化纳入界限" not in text


def test_the_planner_is_told_only_the_risk_set_is_applied() -> None:
    request = _request(
        _with_bounds(_prediction_context()), cohort_mode="all_input_rows"
    )

    assert _population_authority(request)["already_applied"] == {
        "still_in_icu_after_prediction_time_hours": 24.0
    }


def test_other_families_and_requests_keep_every_input_row_and_their_identity() -> None:
    descriptive = _request(_descriptive_context(), cohort_mode="all_input_rows")
    prediction = _request(_prediction_context(), cohort_mode="predicate_filtered")

    assert descriptive.cohort_selection_mode == "all_input_rows"
    # Neither carries the new field, so both keep the identity they had.
    for request in (descriptive, prediction):
        assert "caller_binds_all_input_rows" not in request.model_dump(mode="json")


def test_the_request_contract_binds_the_flag_to_a_risk_set_without_a_stated_population() -> (
    None
):
    descriptive = _request(_descriptive_context(), cohort_mode=None).model_dump(
        mode="json"
    )
    prediction = _request(
        _prediction_context(), cohort_mode="all_input_rows"
    ).model_dump(mode="json")

    with pytest.raises(
        ValidationError, match="filtered only by a prediction's risk set"
    ):
        FamilySpecRequest.model_validate(
            {**descriptive, "caller_binds_all_input_rows": True}
        )
    with pytest.raises(ValidationError, match="offers no population to state"):
        FamilySpecRequest.model_validate({**prediction, "population_concepts": ["age"]})


def test_the_pipeline_reads_the_caller_bound_population_from_its_owner() -> None:
    source = inspect.getsource(
        pipeline_module.ResearchAgentPipeline._validate_and_persist_plan
    )

    # The owner's answer decides the refusal; the modes are not compared here.
    call = source.index("conflict = caller_bound_population_conflict(")
    guard = source.index("if conflict is not None:", call)
    assert source.index("raise CohortAuthorityError(conflict)", guard) > guard
    assert "observed_mode" not in source
