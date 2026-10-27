"""A question that names a kind the analysis data do not separate stops the launch.

One owner (``webserver.question_named_levels``) reads whether a question names
a level of a categorical concept -- invasive or non-invasive ventilation, a
ventilator mode -- that no analysis quantity distinguishes.  A concept
recorded over time is summarized as its presence in a window, so its level is
lost; such a question is never planned on every kind.  The launch stops with
a typed reason before anything is written or any model is called, the stop
offers an explicit plan edit, and the conversation explains it in a sentence.
A concept recorded once per stay keeps its own value and is not stopped.
Synthetic questions only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver import stop_next_actions
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.question_named_levels import (
    _LEVEL_WORDS,
    NAMED_LEVEL_UNDISTINGUISHED,
    _analysis_data_keep_levels,
    _declaration,
    named_levels,
    stop_undistinguished_named_levels,
    undistinguished_named_levels,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.research_pipeline_run_preparation import (
    ResearchPipelineLaunchRequest,
    prepare_research_pipeline_run,
)
from tests.support.node import run_node


def _levels(question: str) -> list[tuple[str, str]]:
    return [(item.concept, item.level) for item in undistinguished_named_levels(question)]


def test_every_word_names_a_declared_level_of_a_concept_recorded_over_time() -> None:
    for concept, level, _words in _LEVEL_WORDS:
        declaration = _declaration(concept)
        assert declaration is not None, concept
        class_name, target, levels = declaration
        assert class_name == "fct_cncpt", concept
        assert target != "id_tbl", concept
        assert level in levels, (concept, level)
        assert not _analysis_data_keep_levels(concept)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("有创机械通气患者的 28 天死亡率是多少？", [("mech_vent", "invasive")]),
        ("入 ICU 后接受有创通气的患者比例", [("mech_vent", "invasive")]),
        ("无创通气与 ICU 死亡的关联", [("mech_vent", "noninvasive")]),
        (
            "Mortality among patients on invasive mechanical ventilation",
            [("mech_vent", "invasive")],
        ),
        ("Invasively ventilated patients and 28-day death", [("mech_vent", "invasive")]),
        (
            "Compare non-invasive ventilation with invasive ventilation",
            [("mech_vent", "noninvasive"), ("mech_vent", "invasive")],
        ),
        ("Noninvasive ventilation and ICU mortality", [("mech_vent", "noninvasive")]),
        ("Non invasive ventilation and ICU mortality", [("mech_vent", "noninvasive")]),
        ("Does NIV in the first day change mortality?", [("mech_vent", "noninvasive")]),
        ("压力控制通气患者的死亡率", [("vent_mode", "pressure")]),
        ("Volume-controlled ventilation and lactate", [("vent_mode", "volume")]),
        ("SIMV模式与拔管时间", [("vent_breath_seq", "simv")]),
        ("入 ICU 当天接受NIV的患者比例", [("mech_vent", "noninvasive")]),
    ],
)
def test_a_named_kind_of_a_concept_recorded_over_time_is_read(
    question: str, expected: list[tuple[str, str]]
) -> None:
    assert _levels(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "机械通气患者的 28 天死亡率是多少？",
        "Mortality among mechanically ventilated patients",
        "Is ventilation in the first 24 hours associated with ICU death?",
        "有创动脉血压与乳酸的关系",
        "Invasive fungal infection and ICU mortality",
        "Invasive monitoring and ventilation in sepsis",
        "急诊入院与择期手术入院患者的死亡率比较",
        "Female versus male ICU mortality",
        "ICU mortality in a university hospital",
        "",
    ],
)
def test_a_question_without_an_unseparated_kind_is_not_stopped(question: str) -> None:
    assert _levels(question) == []
    stop_undistinguished_named_levels(question)


def test_a_concept_recorded_once_per_stay_keeps_its_levels() -> None:
    assert _analysis_data_keep_levels("adm")
    assert _analysis_data_keep_levels("sex")
    assert not _analysis_data_keep_levels("mech_vent")
    assert not _analysis_data_keep_levels("no_such_concept")


def test_each_kind_is_read_once_with_its_first_words() -> None:
    found = named_levels("有创通气、有创机械通气与无创通气")
    assert [(item.level, item.evidence) for item in found] == [
        ("invasive", "有创通气"),
        ("noninvasive", "无创通气"),
    ]


def test_the_stop_names_the_kind_and_the_question_the_data_answer() -> None:
    with pytest.raises(ResearchPipelineRunError) as stopped:
        stop_undistinguished_named_levels("有创机械通气患者的 28 天死亡率")

    error = stopped.value
    assert error.code == NAMED_LEVEL_UNDISTINGUISHED
    assert error.details == {
        "concept": "mech_vent",
        "named_levels": [
            {"concept": "mech_vent", "level": "invasive", "evidence": "有创机械通气"},
        ],
    }
    assert "invasive, noninvasive" in str(error)
    assert "mechanical ventilation" in str(error)


def _request(tmp_path: Path, question: str, **overrides: Any) -> ResearchPipelineLaunchRequest:
    values: dict[str, Any] = {
        "export_path": str(tmp_path / "export"),
        "study_context": {
            "id": "study-launch",
            "revision": 5,
            "question": question,
            "data_source": {"path": str(tmp_path / "export"), "database": "miiv"},
            "analysis_design": {},
        },
        "project_root": str(tmp_path / "workspace"),
        "provider": {"provider": "openai"},
        "provider_environment": {"OPENAI_API_KEY": "test"},
        "credential_source": "pi_verified",
        "literature_search_authorized": False,
        "plan_revision_source_run_id": "",
        "execution_resume_source_run_id": "",
        "development_resume_source_job_id": "",
        "budget_mode": "planner_canary",
        "runner_image": None,
    }
    values.update(overrides)
    return ResearchPipelineLaunchRequest(**values)


def test_the_launch_stops_before_anything_is_written(tmp_path, monkeypatch) -> None:
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(
        context_store,
        "upsert_context",
        lambda raw, **kwargs: writes.append(dict(raw)) or dict(raw),
    )

    with pytest.raises(ResearchPipelineRunError) as stopped:
        prepare_research_pipeline_run(
            _request(tmp_path, "Among adults on invasive ventilation, 28-day death by sex")
        )

    assert stopped.value.code == NAMED_LEVEL_UNDISTINGUISHED
    assert stopped.value.details["named_levels"][0]["level"] == "invasive"
    assert writes == []
    assert not (tmp_path / "workspace").exists()


def test_the_stop_offers_an_explicit_plan_edit() -> None:
    assert stop_next_actions.STOP_POLICIES[NAMED_LEVEL_UNDISTINGUISHED] == "edit_plan"
    actions = stop_next_actions.next_actions_for_stop(NAMED_LEVEL_UNDISTINGUISHED, detail={})
    assert [action.kind for action in actions] == ["edit_plan"]
    params = actions[0].params
    assert params.requires_explicit_choice is True
    assert params.requires_plan_approval is True


@pytest.mark.parametrize(
    ("language", "expected"),
    [("en", "mechanical ventilation"), ("zh", "机械通气")],
)
def test_the_conversation_explains_the_stop_in_a_sentence(language: str, expected: str) -> None:
    script = r"""
const modules = {};
global.window = { EU_HTML: {esc: x => x}, EasyICU: {guidedPi: {declare: (name, api) => {modules[name] = api;}}}};
require(process.argv[1]);
const pick = process.argv[2] === 'zh' ? (en, zh) => zh : (en, zh) => en;
const owner = modules.errorText.create({tr: pick, staticPreview: false});
console.log(JSON.stringify(owner.runFailureText(process.argv[3], {})));
"""
    path = (
        Path(__file__).resolve().parents[2]
        / "src/easyicu/webserver/static/js/screens-guided-pi-error-text.js"
    )
    result = run_node("node", script, str(path), language, NAMED_LEVEL_UNDISTINGUISHED, check=True)
    text = json.loads(result.stdout)
    assert expected in text
    assert NAMED_LEVEL_UNDISTINGUISHED not in text
