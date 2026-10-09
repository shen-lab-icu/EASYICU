"""A prediction question names no exposure, and every concept it names reaches planning.

The first-plan coordinates project the concept a question names as its
exposure.  In a question that builds a prediction model and compares it with
an existing score, that concept is the score: a benchmark, not an exposure.
Projected as the exposure, it framed the plan as an association with the
outcome while the comparison itself was dropped.  A question the reader reads
as a prediction study now names no exposure; every concept it names besides
its outcome is passed to planning with the words that name it and every
concept its name can denote, and the plan states what the question asks of
each.  An association question keeps its exposure, which accounts for itself:
only the other concepts it names are passed.  Generic wording only.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.agent_pipeline_runs import _research_user_preferences
from easyicu.webserver.research_launch_scientific import (
    _metadata_only_planning_coordinates,
)
from easyicu.webserver.study_intent import deterministic_intent, named_study_concepts
from tests.webserver.copilot.research_workflow_fixtures import complete_study

_PREDICTION = "建立并内部验证一个预测成人 ICU 患者院内死亡的模型，并与 SOFA 评分比较区分度和校准度"
_ASSOCIATION = "研究成人 ICU 患者前24小时最高乳酸与院内死亡"
_ADJUSTED = "研究成人 ICU 患者前24小时最高乳酸和 SOFA 评分与院内死亡"


def test_a_prediction_question_projects_no_exposure() -> None:
    coordinates = _metadata_only_planning_coordinates(
        question=_PREDICTION, database="miiv"
    )

    assert coordinates["primary_exposure"] is None
    assert coordinates["primary_exposure_aggregation"] is None
    assert coordinates["target_outcome"] == "death"
    assert [item["concepts"] for item in coordinates["question_named_concepts"]] == [
        ["sofa"]
    ]


@pytest.mark.parametrize(
    ("question", "named"),
    [
        # The sealed exposure accounts for itself: nothing else is named, and
        # the study's planning coordinates are as they were.
        pytest.param(_ASSOCIATION, [], id="only-its-exposure"),
        pytest.param(_ADJUSTED, [["sofa"]], id="and-another-concept"),
    ],
)
def test_an_association_question_keeps_its_exposure(
    question: str, named: list[list[str]]
) -> None:
    coordinates = _metadata_only_planning_coordinates(
        question=question, database="miiv"
    )

    assert coordinates["primary_exposure"] == "lact"
    assert [
        item["concepts"] for item in coordinates["question_named_concepts"]
    ] == named


def test_the_reader_itself_still_reads_what_the_sentence_names() -> None:
    # Only the projection changes: the reader's own slots are as before.
    slots = deterministic_intent(_PREDICTION)["slots"]

    assert slots["analysis_family"]["value"] == "prediction"
    assert slots["exposure"]["value"] == "sofa"


def test_a_score_names_the_risk_it_predicts_too() -> None:
    question = "Build a model that predicts in-hospital mortality and compare it with APACHE IVa"

    assert named_study_concepts(question) == (
        (("apache_iv", "apache_iv_pred_hosp_mort"), "apache"),
    )
    coordinates = _metadata_only_planning_coordinates(
        question=question, database="eicu"
    )
    assert coordinates["primary_exposure"] is None
    (named,) = coordinates["question_named_concepts"]
    assert set(named["concepts"]) == {"apache_iv", "apache_iv_pred_hosp_mort"}


def test_the_named_concepts_reach_planning_in_the_data_constraints() -> None:
    named = [{"concepts": ["sofa"], "evidence": "sofa"}]

    constraints = json.loads(
        _research_user_preferences(complete_study(), question_named_concepts=named)[
            "data_constraints"
        ]
    )

    assert constraints["question_named_concepts"] == named


def test_a_run_declares_the_named_concepts_in_its_data_constraints(
    tmp_path: Path,
) -> None:
    """Every run declares its context through one owner, a trial's run included."""

    named = [{"concepts": ["sofa"], "evidence": "sofa"}]
    study = complete_study()

    def constraints(coordinates: dict[str, Any]) -> dict[str, Any]:
        scientific = SimpleNamespace(
            study=study,
            materialization_study=study,
            patient_grouping=None,
            cohort_window=(0.0, 24.0),
            metadata_planning_coordinates=coordinates,
        )
        declared = agent_pipeline_runs.research_context_declarations(
            scientific, export_path=str(tmp_path / "export")
        )
        return json.loads(declared["user_preferences"].get("data_constraints") or "{}")

    assert constraints({"question_named_concepts": named})[
        "question_named_concepts"
    ] == named
    # Coordinates that name nothing else declare nothing.
    assert "question_named_concepts" not in constraints({})


def test_the_candidate_roster_requires_the_named_concepts() -> None:
    source = inspect.getsource(agent_pipeline_runs)
    acquisition = source.index("acquisition = _metadata_only_planning_acquisition(")
    roster = source.index("required_concepts=(", acquisition)
    closing = source.index("patient_grouping=patient_grouping", roster)
    assert '"question_named_concepts"' in source[roster:closing]
