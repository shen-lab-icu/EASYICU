"""A plan whose cohort reads the whole stay reads back from its own JSON.

A cohort predicate over the whole stay has an unbounded end
(``end_offset_hours`` "inf"), as the planning corrections for a whole-stay
criterion ask.  JSON has no infinity: ``AnalysisPlan`` wrote that end as null
and could not read its own cohort back ("invalid time offset: None").  A run
whose plan held one ended with "current analysis plan is not bound to
immutable EvidenceStore authority", which does not name the cause.

A window's offsets are now written as ``TimeWindow.to_dict`` writes them
wherever a model serializes a window: a plan's cohort and a robustness
override.  A NaN offset, which compares false with every bound and so was
accepted as a window, is refused.

Synthetic plans; the concept is a generic event.
"""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    CohortSchemaError,
    ConceptPredicate,
    TimeWindow,
    cohort_concept_id_scope,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

_EVENT = "event_flag"


@pytest.fixture(autouse=True)
def _generic_concepts():
    with cohort_concept_id_scope([_EVENT]):
        yield


def _cohort(end: float, *, name: str = "primary") -> CohortDefinition:
    return CohortDefinition(
        name=name,
        exclusion=(
            ConceptPredicate(
                concept_id=_EVENT,
                time_window=TimeWindow(
                    anchor="icu_admission", start_offset_hours=0.0, end_offset_hours=end
                ),
                aggregation="max",
                op="==",
                value=1,
            ),
        ),
    )


def _plan(cohort: CohortDefinition, override: CohortDefinition) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Is an exposure associated with the outcome?",
        steps=[
            AnalysisStep(
                step_id="01_describe",
                intent="Describe the cohort.",
                expected_outputs=["table:describe"],
            )
        ],
        cohort=cohort,
        robustness_specs=[
            RobustnessSpec(
                spec_id="whole_stay",
                axis="cohort",
                description="Exclude the stays with the event at any time.",
                cohort_override=override,
            )
        ],
    )


def _window(payload: dict) -> dict:
    return payload["exclusion"][0]["time_window"]


def test_a_plan_whose_cohort_reads_the_whole_stay_reads_back() -> None:
    whole_stay = _cohort(math.inf)
    override = _cohort(math.inf, name="whole_stay")
    plan = _plan(whole_stay, override)

    dumped = plan.model_dump_json()
    reread = AnalysisPlan.model_validate_json(dumped)

    payload = json.loads(dumped)
    assert _window(payload["cohort"]) == {
        "anchor": "icu_admission",
        "start_offset_hours": 0.0,
        "end_offset_hours": "inf",
    }
    assert (
        _window(payload["robustness_specs"][0]["cohort_override"])["end_offset_hours"]
        == "inf"
    )
    assert plan.model_dump(mode="json") == payload
    assert reread.cohort == whole_stay
    assert reread.robustness_specs[0].cohort_override == override
    # As the locked cohort writes it.
    assert _window(payload["cohort"]) == _window(whole_stay.to_dict())


def test_a_bounded_window_and_a_python_dump_are_written_as_before() -> None:
    plan = _plan(_cohort(24.0), _cohort(math.inf, name="whole_stay"))

    payload = json.loads(plan.model_dump_json())

    assert _window(payload["cohort"])["end_offset_hours"] == 24.0
    assert (
        _window(plan.model_dump()["robustness_specs"][0]["cohort_override"])[
            "end_offset_hours"
        ]
        == math.inf
    )


def test_the_plan_schema_a_planner_is_given_is_unchanged() -> None:
    schema = AnalysisPlan.model_json_schema()

    (window,) = [
        definition
        for definition in schema["$defs"].values()
        if set(definition.get("properties", {}))
        == {"anchor", "start_offset_hours", "end_offset_hours"}
    ]
    assert window["properties"]["end_offset_hours"] == {
        "title": "End Offset Hours",
        "type": "number",
    }


@pytest.mark.parametrize("offset", ["nan", "NaN", float("nan")])
def test_a_nan_offset_is_no_window(offset) -> None:
    for start, end in ((0, offset), (offset, 24)):
        with pytest.raises(CohortSchemaError):
            TimeWindow.from_dict(
                {
                    "anchor": "icu_admission",
                    "start_offset_hours": start,
                    "end_offset_hours": end,
                }
            )
    with pytest.raises(CohortSchemaError, match="not NaN"):
        TimeWindow(
            anchor="icu_admission", start_offset_hours=0.0, end_offset_hours=math.nan
        )
    with pytest.raises(CohortSchemaError, match="not NaN"):
        TimeWindow(
            anchor="icu_admission", start_offset_hours=math.nan, end_offset_hours=24.0
        )


def test_a_plan_with_a_nan_offset_is_refused() -> None:
    payload = json.loads(_plan(_cohort(24.0), _cohort(48.0)).model_dump_json())
    payload["cohort"]["exclusion"][0]["time_window"]["end_offset_hours"] = math.nan

    with pytest.raises(ValidationError, match="invalid time offset"):
        AnalysisPlan.model_validate(payload)
