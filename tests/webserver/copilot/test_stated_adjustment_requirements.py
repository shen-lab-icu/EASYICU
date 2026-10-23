"""Structured adjustment readings bind the study, with source gaps preserved."""

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding, PiSessionRecord, ToolExecutionContext,
)
from easyicu.webserver import agent_pipeline_runs, study_contexts


def _update(monkeypatch, *, words="在控制年龄之后", variable="年龄", concepts=("age",),
            statements=None, current=None, question=None):
    question = question or f"{words}，指标与结局的关系如何？"
    current = current or {
        "id": "adjustment-study", "revision": 1, "question": question,
        "data_source": {"path": "/synthetic/source", "database": "miiv"},
    }
    writes = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: dict(current))
    monkeypatch.setattr(tool_module, "_workflow_snapshot", lambda *args, **kwargs: {})
    monkeypatch.setattr(tool_module.study_contexts, "upsert_context",
                        lambda raw, **kwargs: writes.append(raw) or {**current, **raw, "revision": 2})
    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _path: SimpleNamespace(concepts=[SimpleNamespace(concept_id=name) for name in concepts]),
    )
    context = ToolExecutionContext(
        session=PiSessionRecord(session_id="adjustment-session", binding=AuthorityBinding(
            study_context_id=current["id"], study_revision=current["revision"],
        )), user_message=question, allowed_actions={"configure"},
    )
    params = {"question": question, "stated_adjustments": (
        [{"variable": variable, "words": words}] if statements is None else statements
    )}
    return tool_module.execute_tool("easyicu_update_study_context", params, context), writes


@pytest.mark.parametrize("words", ["在控制年龄之后", "以年龄为协变量", "调整年龄"])
def test_a_stated_adjustment_is_verified_and_saved_as_a_required_subset(monkeypatch, words):
    result, writes = _update(monkeypatch, words=words)
    assert result["status"] == "ok"
    assert writes[0]["stated_adjustments"] == [{
        "variable": "年龄", "words": words, "concept": "age", "source_status": "available",
    }]
    assert writes[0].get("covariate_selection", "planner_selectable") == "planner_selectable"
    assert result["details"]["study"]["stated_adjustments"] == writes[0]["stated_adjustments"]


def test_a_source_missing_the_stated_variable_keeps_a_visible_gap(monkeypatch):
    result, writes = _update(monkeypatch, concepts=())
    assert result["status"] == "ok"
    assert writes[0]["stated_adjustments"][0]["source_status"] == "unavailable"
    assert result["details"]["study"]["stated_adjustments"][0]["concept"] == "age"


@pytest.mark.parametrize("statement", [
    {"variable": "年龄", "words": "未出现在问题里的年龄"},
    {"variable": "性别", "words": "在控制年龄之后"},
])
def test_an_ungrounded_statement_is_not_recorded_and_the_question_is_saved(monkeypatch, statement):
    result, writes = _update(monkeypatch, statements=[statement])
    assert result["status"] == "ok"
    assert writes and writes[0]["question"].startswith("在控制年龄之后")
    assert not writes[0].get("stated_adjustments")
    unread, = result["details"]["stated_adjustments_unread"]
    assert unread["variable"] == statement["variable"]
    assert "stated_adjustments not recorded" in result["summary"]


def test_the_wording_rule_and_save_tool_request_adjustment_words():
    main = Path("src/easyicu/webserver/pi_copilot/node_app/src/main.mjs").read_text()
    rule = next(line for line in main.splitlines() if "Question wording rule:" in line)
    assert "stated_adjustments" in rule and "required subset" in rule
    assert "stated_adjustments: Type.Optional(Type.Array(Type.Object({" in main
    assert "variable: Type.String(" in main
    assert "concept: Type.Optional(Type.String(" in main
    assert "easyicu_list_source_concepts" in rule


def test_a_name_nothing_resolves_is_unresolved_not_unavailable(monkeypatch):
    """A score without a dictionary name is no source gap: nothing read it."""

    result, writes = _update(monkeypatch, words="控制试验量表后", variable="试验量表",
                             concepts=("trial_score",))
    assert result["status"] == "ok"
    record, = writes[0]["stated_adjustments"]
    assert record["concept"] is None and record["source_status"] == "unresolved"
    assert record["variable"] == "试验量表"


def test_the_conversations_lookup_names_a_score_the_dictionary_does_not(monkeypatch):
    result, writes = _update(monkeypatch, concepts=("trial_score",), statements=[{
        "variable": "试验量表", "words": "控制试验量表后", "concept": "trial_score",
    }], question="控制试验量表后，指标与结局的关系如何？")
    assert result["status"] == "ok"
    record, = writes[0]["stated_adjustments"]
    assert record["concept"] == "trial_score" and record["source_status"] == "available"


@pytest.mark.parametrize("concept", ["not_in_source", "sex"])
def test_a_lookup_the_source_or_the_dictionary_contradicts_is_not_recorded(monkeypatch, concept):
    result, writes = _update(monkeypatch, concepts=("age", "sex"), statements=[{
        "variable": "年龄", "words": "在控制年龄之后", "concept": concept,
    }])
    assert result["status"] == "ok"
    assert not writes[0].get("stated_adjustments")
    assert result["details"]["stated_adjustments_unread"][0]["variable"] == "年龄"


def test_a_launch_keeps_a_lookup_its_source_lacks_as_unavailable(monkeypatch):
    from easyicu.webserver.research_launch_scientific import checked_stated_adjustments

    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _path: SimpleNamespace(concepts=[SimpleNamespace(concept_id="age")]),
    )
    record, = checked_stated_adjustments(
        [{"variable": "试验量表", "words": "控制试验量表后", "concept": "trial_score"}],
        question="控制试验量表后，指标与结局的关系如何？",
        source={"path": "/synthetic/source"}, recheck=True,
    )
    assert record["concept"] == "trial_score" and record["source_status"] == "unavailable"


def test_an_exact_source_identifier_needs_no_dictionary_alias(monkeypatch):
    result, writes = _update(
        monkeypatch, words="controlling for trial_score", variable="trial_score", concepts=("trial_score",),
    )
    assert result["status"] == "ok"
    record, = writes[0]["stated_adjustments"]
    assert record["concept"] == "trial_score" and record["source_status"] == "available"


def test_saved_requirements_survive_empty_statements_and_agent_roster_changes(monkeypatch):
    _, writes = _update(monkeypatch)
    saved = {**writes[0], "id": "adjustment-study", "revision": 2}
    saved["data_source"] = {"path": "/synthetic/source", "database": "miiv"}
    result, writes = _update(monkeypatch, current=saved, statements=[])
    assert result["status"] == "ok"
    assert writes[0]["stated_adjustments"] == saved["stated_adjustments"]
    # An Agent's exact roster does not overwrite the separate question requirement.
    compiled = agent_pipeline_runs._research_user_preferences({
        **saved, "covariates": [], "covariate_selection": "exact", "covariate_authority": "agent_plan",
    })
    assert json.loads(compiled["data_constraints"])["stated_adjustments"] == saved["stated_adjustments"]


def test_a_requirement_saved_before_source_selection_is_checked_when_compiled(monkeypatch):
    current = {"id": "adjustment-study", "revision": 1, "question": "在控制年龄之后，指标与结局的关系如何？"}
    result, writes = _update(monkeypatch, current=current)
    assert result["status"] == "ok"
    assert writes[0]["stated_adjustments"][0]["source_status"] == "unbound"
    compiled = agent_pipeline_runs._research_user_preferences({
        **writes[0], "data_source": {"path": "/synthetic/source", "database": "miiv"},
    })
    assert json.loads(compiled["data_constraints"])["stated_adjustments"][0]["source_status"] == "available"


def test_the_store_retains_the_requirement_and_binds_its_scientific_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    record = {"variable": "年龄", "words": "调整年龄", "concept": "age", "source_status": "available"}
    raw = {"id": "stored-adjustment", "question": "调整年龄后估计指标与结局的关系。"}
    original = study_contexts.upsert_context(raw)
    with pytest.raises(study_contexts.StudyContextError):
        study_contexts.upsert_context({**raw, "stated_adjustments": [record]})
    saved = study_contexts.upsert_context({**raw, "stated_adjustments": [record]}, _server_question_wording_write=True)
    assert study_contexts.get_context(saved["id"])["stated_adjustments"] == [record]
    assert study_contexts.scientific_configuration_sha256(original) != study_contexts.scientific_configuration_sha256(saved)
    changed = study_contexts.upsert_context({**raw, "question": "只描述指标的分布。"})
    assert changed["stated_adjustments"] == []
    restored = study_contexts.restore_turn_configuration_snapshot(
        saved["id"], {"schema_version": "easyicu.pi-turn-study-snapshot/1",
                      "study_context_id": saved["id"], "source_revision": saved["revision"],
                      "configuration": {"question": saved["question"], "stated_adjustments": [record]}},
        expected_revision=changed["revision"],
    )
    assert restored["stated_adjustments"] == [record]


def test_two_named_adjustments_are_both_required(monkeypatch):
    words = "以年龄、性别为协变量"
    result, writes = _update(monkeypatch, words=words, concepts=("age", "sex"), statements=[
        {"variable": "年龄", "words": words}, {"variable": "性别", "words": words},
    ])
    assert result["status"] == "ok"
    assert [item["concept"] for item in writes[0]["stated_adjustments"]] == ["age", "sex"]


def test_a_score_named_by_the_model_reaches_the_source_and_planning_roster(monkeypatch):
    from easyicu.webserver.research_launch_scientific import _metadata_only_planning_coordinates

    # A synthetic dictionary entry proves this path is independent of role keywords.
    monkeypatch.setattr("easyicu.webserver.study_intent.catalog_named_concepts",
                        lambda text: ("trial_score",) if "试验评分" in text else ())
    words = "在控制试验评分之后"
    result, writes = _update(monkeypatch, words=words, variable="试验评分", concepts=("trial_score",))
    assert result["status"] == "ok"
    saved = writes[0]
    assert saved["stated_adjustments"][0]["concept"] == "trial_score"
    coordinates = _metadata_only_planning_coordinates(
        question=saved["question"], database="miiv", export_path="/synthetic/source",
        stated_adjustments=saved["stated_adjustments"],
    )
    assert coordinates["required_adjustment_concepts"] == ["trial_score"]
    assert coordinates["primary_exposure"] != "trial_score"


def test_a_concept_cannot_be_bound_against_a_negated_question_reading(monkeypatch):
    monkeypatch.setattr("easyicu.webserver.study_intent.catalog_named_concepts",
                        lambda text: ("age",) if text == "年龄" else ())
    result, writes = _update(monkeypatch)
    assert result["status"] == "ok" and not writes[0].get("stated_adjustments")
    assert result["details"]["stated_adjustments_unread"][0]["variable"] == "年龄"


def test_a_variable_named_elsewhere_is_not_grounded_by_another_adjustment_clause(monkeypatch):
    result, writes = _update(
        monkeypatch, question="在控制年龄之后，性别与结局的关系如何？",
        statements=[{"variable": "性别", "words": "在控制年龄之后"}], concepts=("age", "sex"),
    )
    assert result["status"] == "ok" and not writes[0].get("stated_adjustments")
    assert result["details"]["stated_adjustments_unread"][0]["variable"] == "性别"

