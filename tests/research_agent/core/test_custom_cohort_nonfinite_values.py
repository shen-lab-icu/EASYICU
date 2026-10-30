"""A reviewed missing policy also covers unevaluable infinite values."""

import math

import pandas as pd
import pytest

from easyicu.research_agent.cohort.schema import CohortDataError, _build_cohort_with_flow
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition, ConceptPredicate, TimeWindow, cohort_concept_id_scope,
)


def predicate(concept, policy, *, op="<=", value=4.0):
    return ConceptPredicate(
        concept_id=concept, time_window=TimeWindow("icu_admission", 0, 24),
        aggregation="first", op=op, value=value, missing_policy=policy,
    )


@pytest.mark.parametrize("role", ["inclusion", "exclusion"])
@pytest.mark.parametrize("policy", ["exclude", "retain", "stop"])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_nonfinite_measurement_is_unevaluable_under_reviewed_policy(role, policy, dtype):
    frame = pd.DataFrame({"stay_id": [1, 2, 3, 4, 5], "crea": pd.Series([3, 5, math.inf, -math.inf, None], dtype=dtype)})
    definition = CohortDefinition(name="custom", **{role: (predicate("crea", policy),)})
    if policy == "stop":
        with pytest.raises(CohortDataError, match="missing policy"):
            _build_cohort_with_flow(definition, frame)
        return
    selected, flow = _build_cohort_with_flow(definition, frame)
    observed = {1} if role == "inclusion" else {2}
    expected = observed | ({3, 4, 5} if policy == "retain" else set())
    assert set(selected.stay_id) == expected
    assert flow[1]["n_retained_missing"] == (3 if policy == "retain" else 0)
    assert flow[1]["n_excluded_missing"] == (3 if policy == "exclude" else 0)


@pytest.mark.parametrize("policy", ["exclude", "retain", "stop"])
def test_nonfinite_event_time_does_not_mean_no_event(policy):
    frame = pd.DataFrame({"stay_id": [1, 2, 3, 4], "event": [1, 1, 1, 0],
                          "event_time": [5.0, math.inf, -math.inf, None]})
    with cohort_concept_id_scope(["event"]):
        definition = CohortDefinition(name="custom", exclusion=(predicate("event", policy, op="==", value=1),))
        if policy == "stop":
            with pytest.raises(CohortDataError, match="missing policy"):
                _build_cohort_with_flow(definition, frame)
            return
        selected, flow = _build_cohort_with_flow(definition, frame)
    assert set(selected.stay_id) == ({4} if policy == "exclude" else {2, 3, 4})
    assert flow[1]["n_excluded_missing"] == (2 if policy == "exclude" else 0)
    assert flow[1]["n_retained_missing"] == (2 if policy == "retain" else 0)


def test_legacy_rule_retains_original_nonfinite_behavior():
    frame = pd.DataFrame({"stay_id": [1, 2, 3], "crea": [3.0, math.inf, -math.inf]})
    selected, _ = _build_cohort_with_flow(
        CohortDefinition(name="legacy", inclusion=(predicate("crea", None),)), frame,
    )
    assert set(selected.stay_id) == {1, 3}
