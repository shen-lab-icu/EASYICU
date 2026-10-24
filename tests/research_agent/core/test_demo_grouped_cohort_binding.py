"""Saved planning metadata, evaluated only over small synthetic tables."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.cohort.schema import (
    CohortDataError,
    _build_cohort_with_flow,
    _planner_declared_context_column_bindings,
    coerce_cohort_definition,
    cohort_concept_id_scope,
)
from easyicu.research_agent.planning.cohort_predicate_domain import (
    cohort_emptied_exposure_levels,
)
from easyicu.research_agent.planning.exposure_group_compile import (
    ExposureGroupContrastError,
    validate_exposure_group_contrasts,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ConceptDescriptor


@pytest.fixture
def saved_group():
    path = Path(__file__).parents[2] / "fixtures/demo_round_stops/da6_cohort.json"
    raw = json.loads(path.read_text())
    variable = ConceptDescriptor.model_validate(raw["variable"])
    context = SimpleNamespace(
        variables=[variable], target_outcome=None,
        cohort=SimpleNamespace(provenance={}, outcome_columns=[], time_columns=[]),
        user_preferences=None,
    )
    with cohort_concept_id_scope([variable.name]):
        yield raw, context


def test_saved_string_group_codes_are_rejected_before_cohort_materialization(saved_group):
    raw, context = saved_group
    cohort = coerce_cohort_definition(raw["cohort"])
    conflicts = cohort_emptied_exposure_levels(
        context, cohort, targets={raw["variable"]["name"]: [1, 3]},
    )
    assert {item.level for item in conflicts} == {1, 3}
    assert cohort.inclusion[0].value == ["1", "3"]  # No implicit recoding.
    plan = AnalysisPlan(
        research_question="Compare outcomes in the declared groups.",
        analysis_type="association_study", cohort=cohort,
        steps=[AnalysisStep.model_validate(raw["table_step"])],
    )
    with pytest.raises(ExposureGroupContrastError) as caught:
        validate_exposure_group_contrasts(plan, context)
    assert caught.value.reason_code == "declared_exposure_level_empty"


@pytest.mark.parametrize("transform", ["exposure_group", "exposure_group_ordinal"])
@pytest.mark.parametrize("aggregation", ["first", "last", "min", "max"])
def test_numeric_group_predicate_reads_the_named_materialized_group(
    saved_group, transform, aggregation,
):
    raw, context = saved_group
    raw = deepcopy(raw)
    predicate = raw["cohort"]["inclusion"][0]
    predicate.update(value=[1, 3], aggregation=aggregation)
    context.variables = [context.variables[0].model_copy(update={"unit_normalization": transform})]
    cohort = coerce_cohort_definition(raw["cohort"])
    column = raw["variable"]["name"]
    data = pd.DataFrame({column: [1, 2, 3, None]})
    bindings = _planner_declared_context_column_bindings(
        definition=cohort, plan=None, context=context, columns=data.columns,
    )
    selected, flow = _build_cohort_with_flow(cohort, data, column_bindings=bindings)
    assert selected[column].tolist() == [1, 3]
    assert flow[-1]["n_remaining"] == 2
    assert not cohort_emptied_exposure_levels(context, cohort, targets={column: [1, 3]})


@pytest.mark.parametrize("change", ["window", "aggregation", "transform"])
def test_group_binding_still_refuses_a_different_coordinate(saved_group, change):
    raw, context = saved_group
    raw = deepcopy(raw)
    predicate = raw["cohort"]["inclusion"][0]
    predicate["value"] = [1, 3]
    if change == "window":
        predicate["time_window"]["end_offset_hours"] = 48
    elif change == "aggregation":
        predicate["aggregation"] = "mean"
    else:
        context.variables = [context.variables[0].model_copy(update={"unit_normalization": None})]
    cohort = coerce_cohort_definition(raw["cohort"])
    with pytest.raises((CohortDataError, ValueError)):
        _planner_declared_context_column_bindings(
            definition=cohort, plan=None, context=context,
            columns=[raw["variable"]["name"]],
        )


@pytest.mark.parametrize("change", ["window", "aggregation", "transform"])
def test_group_domain_proof_requires_the_same_value_and_window(saved_group, change):
    raw, context = saved_group
    raw = deepcopy(raw)
    predicate = raw["cohort"]["inclusion"][0]
    if change == "window":
        predicate["time_window"]["end_offset_hours"] = 48
    elif change == "aggregation":
        predicate["aggregation"] = "mean"
    else:
        context.variables = [context.variables[0].model_copy(update={"unit_normalization": None})]
    cohort = coerce_cohort_definition(raw["cohort"])
    assert not cohort_emptied_exposure_levels(
        context, cohort, targets={raw["variable"]["name"]: [1, 3]},
    )
