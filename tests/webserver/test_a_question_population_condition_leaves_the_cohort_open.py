"""A question restricted to a recorded condition leaves the population to the plan (leg 3, rc_78a).

UX leg 3 on rc_78a: the study was bound to the official Demo export with a saved
cohort that states no criterion (only its label 「脓毒症 ICU 患者」), so the launch
required every input row.  That mode forbids the Planner from stating any
population criterion, while the question's 「脓毒症」 is a named concept the plan
must select on (S1722): every plan went back, four times, with no way to answer
and no approve path.  Such a launch now leaves the mode to the Planner, as a
metadata-only launch does.  A saved criterion, a researcher's recorded
eligibility decision, a question with no recorded condition, an unsupported
population and a negated one keep the mode.  Metadata only.
"""

from __future__ import annotations

import inspect

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.research_launch_population import question_population_mode

_LEG3 = "脓毒症 ICU 患者中，ICU 入科时首次乳酸与院内死亡的关联。"
_LEG3_EN = "Among ICU patients with sepsis, is the first lactate at ICU admission associated with in-hospital death?"
_UX_STUDY = {"id": "study_leg3", "question": _LEG3,
             "cohort": {"label": "脓毒症 ICU 患者", "review": "脓毒症 ICU 患者"},
             "cohort_eligibility_authority": {}}


def _launch_mode(study: dict, question: str) -> str | None:
    return question_population_mode(
        agent_pipeline_runs._planner_cohort_selection_mode(study),  # noqa: SLF001
        study=study, question=question,
    )


@pytest.mark.parametrize("question", [_LEG3, _LEG3_EN, "AKI 患者中，肌酐与死亡"], ids=["leg3-zh", "en", "aki"])
def test_a_recorded_condition_population_leaves_the_mode_to_the_planner(question: str) -> None:
    assert agent_pipeline_runs._planner_cohort_selection_mode(_UX_STUDY) == "all_input_rows"  # noqa: SLF001
    assert _launch_mode(_UX_STUDY, question) is None


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("成人 ICU 患者中，入科时首次乳酸与院内死亡的关联。", id="no-condition"),
        pytest.param("在心衰患者中，乳酸与院内死亡", id="unsupported-population"),
        pytest.param("不研究脓毒症患者；在成人 ICU 患者中，乳酸与院内死亡", id="negated"),
        pytest.param("ICU 入科后 24 小时最高乳酸与发生脓毒症有什么关联？", id="sepsis-as-outcome"),
    ],
)
def test_without_a_recorded_condition_population_every_input_row_stays(question: str) -> None:
    assert _launch_mode(_UX_STUDY, question) == "all_input_rows"


@pytest.mark.parametrize("state", ["confirmed", "declined"])
def test_a_recorded_eligibility_decision_stands(state: str) -> None:
    study = {**_UX_STUDY, "cohort_eligibility_authority": {
        "selection_state": state, "selection_mode": "all_input_rows", "study_context_revision": 3}}
    assert _launch_mode(study, _LEG3) == "all_input_rows"


def test_a_saved_criterion_keeps_its_mode() -> None:
    assert question_population_mode("predicate_filtered", study=_UX_STUDY, question=_LEG3) == "predicate_filtered"
    assert question_population_mode(None, study=_UX_STUDY, question=_LEG3) is None


def test_a_fresh_export_launch_asks_this_owner() -> None:
    source = inspect.getsource(agent_pipeline_runs)
    assert "question_population_mode(\n                    _planner_cohort_selection_mode(study), study=study, question=question," in source
