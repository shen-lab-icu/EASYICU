"""An accepted, unapplied population criterion is not lost at execution (S1747).

UX on rc_512: the question said 「成人 ICU 患者」; the candidate planned on the
prepared package wrote the adult restriction as a condition on age, which the
compiler could not apply, and the researcher accepted the omission.  The
executed run then re-planned bound to every input row, which forbids any
criterion: the executed plan and its report carried no adult restriction and no
note, although the extraction had applied none.  The execution now leaves the
population to the Planner (it restates the criterion: applied, or shown as not
applied), and the plan card names an adult population that the plan neither
applies nor lists and the saved cohort does not bound.  Synthetic only.
"""

from __future__ import annotations

import inspect

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.pi_copilot.plan_decisions import unselected_question_conditions
from easyicu.webserver.research_launch_population import carried_population_mode

_ADULTS = "成人 ICU 患者中，ICU 入科时年龄（每增加 10 岁）与院内死亡的关联，校正性别。"
_ALL_ROWS = {"name": "source_cohort", "selection_mode": "all_input_rows", "inclusion": [], "exclusion": []}
_AGE = {"concept_id": "age", "anchor": "icu_admission", "start_offset_hours": 0, "end_offset_hours": None,
        "aggregation": "first", "op": ">=", "value": 18}


def test_a_candidate_with_an_unapplied_criterion_leaves_the_mode_open() -> None:
    accepted = {**_ALL_ROWS, "unapplied_population_criteria": ["成人 ICU 患者"]}
    assert carried_population_mode("all_input_rows", candidate_definition=accepted) is None


@pytest.mark.parametrize(
    ("mode", "definition"),
    [
        pytest.param("all_input_rows", _ALL_ROWS, id="nothing-unapplied"),
        pytest.param("all_input_rows", None, id="no-definition"),
        pytest.param("predicate_filtered", {**_ALL_ROWS, "unapplied_population_criteria": ["成人"]}, id="filtered"),
    ],
)
def test_other_candidates_keep_their_mode(mode, definition) -> None:
    assert carried_population_mode(mode, candidate_definition=definition) == mode


def test_the_candidate_execution_asks_this_owner() -> None:
    source = inspect.getsource(agent_pipeline_runs)
    assert ("required_cohort_mode = carried_population_mode(\n"
            "                        required_cohort_mode, candidate_definition=candidate_authority.primary_cohort_definition,") in source


def test_an_adult_population_the_plan_drops_is_named() -> None:
    study = {"question": _ADULTS, "cohort": {"label": "成人 ICU 患者"}}
    assert unselected_question_conditions(study, {"cohort": _ALL_ROWS}) == ["成人"]
    en = {"question": "Among adult ICU patients, is age associated with in-hospital death?"}
    assert unselected_question_conditions(en, {"cohort": _ALL_ROWS}) == ["adult"]


@pytest.mark.parametrize(
    ("study", "plan"),
    [
        pytest.param({"question": _ADULTS},
                     {"cohort": {**_ALL_ROWS, "selection_mode": "predicate_filtered", "inclusion": [_AGE]}},
                     id="applied-by-plan"),
        pytest.param({"question": _ADULTS},
                     {"cohort": {**_ALL_ROWS, "unapplied_population_criteria": ["成人 ICU 患者"]}},
                     id="listed-not-applied"),
        pytest.param({"question": _ADULTS, "cohort": {"label": "成人", "age_min": 18}}, {"cohort": _ALL_ROWS},
                     id="saved-cohort-bound"),
        pytest.param({"question": "ICU 入科时年龄与院内死亡的关联"}, {"cohort": _ALL_ROWS}, id="no-adults"),
        pytest.param({"question": _ADULTS}, {}, id="no-plan"),
    ],
)
def test_an_adult_population_that_is_accounted_for_is_not_named(study, plan) -> None:
    assert unselected_question_conditions(study, plan) == []
