"""Conversation subgroup readings bind a source-checked scientific requirement."""

from pathlib import Path
import json

import pytest

from easyicu.webserver import agent_pipeline_runs, study_contexts
from .stated_variable_fixtures import _update


def _save(monkeypatch, **kwargs):
    return _update(monkeypatch, **{
        "field": "stated_subgroups", "variable": "性别", "words": "按性别分层",
        "concepts": ("sex",), **kwargs,
    })


def test_a_stated_subgroup_is_saved_verified_and_passed_to_planning(monkeypatch):
    result, writes = _save(monkeypatch, statements=[{
        "variable": "性别", "words": "按性别分层", "concept": "sex",
    }])
    assert result["status"] == "ok"
    record, = writes[0]["stated_subgroups"]
    assert record == {"variable": "性别", "words": "按性别分层",
                      "concept": "sex", "source_status": "available"}
    assert result["details"]["study"]["stated_subgroups"] == [record]
    compiled = agent_pipeline_runs._research_user_preferences({
        **writes[0], "data_source": {"path": "/synthetic/source", "database": "miiv"},
    })
    assert json.loads(compiled["data_constraints"])["stated_subgroups"] == [record]


@pytest.mark.parametrize("statement", [
    {"variable": "性别", "words": "问题未写出的分层"},
    {"variable": "性别", "words": "按性别分层后做另一分析"},
    {"variable": "性别", "words": "按性别分层", "concept": "age"},
    {"variable": "性别", "words": "按性别分层", "source_status": "available"},
    {"variable": ["性别"], "words": "按性别分层"},
])
def test_an_invalid_reading_returns_a_nonblocking_unread_receipt(monkeypatch, statement):
    result, writes = _save(monkeypatch, statements=[statement], concepts=("sex", "age"))
    assert result["status"] == "ok" and writes[0]["question"]
    assert not writes[0].get("stated_subgroups")
    assert result["details"]["stated_subgroups_unread"]
    assert "stated_subgroups not recorded" in result["summary"]


def test_the_wording_rule_and_save_tool_request_subgroup_words():
    main = Path("src/easyicu/webserver/pi_copilot/node_app/src/main.mjs").read_text()
    rule = next(line for line in main.splitlines() if "Question wording rule:" in line)
    assert "stated_subgroups" in rule and "subgroup" in rule
    assert "full clause" in rule and "easyicu_list_source_concepts" in rule
    assert "before a source is bound" in rule and "stated adjustments and subgroups" in rule
    assert "stated_subgroups: Type.Optional(Type.Array(Type.Object({" in main


def test_subgroups_bind_the_digest_and_restore_without_changing_legacy_digests(tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    record = {"variable": "性别", "words": "按性别分层", "concept": "sex", "source_status": "available"}
    raw = {"id": "stored-subgroup", "question": "按性别分层，指标与结局的关系如何？"}
    original = study_contexts.upsert_context(raw)
    digest = study_contexts.scientific_configuration_sha256(original)
    # The pre-subgroup owner at HEAD produces these bytes for this fixture.
    assert digest == "aaabe112cc25049e003e92dfebc7636f5bee3576a2413566e092fa8f5910b0cc"
    assert digest == study_contexts.scientific_configuration_sha256({**original, "stated_subgroups": []})
    assert digest == study_contexts.scientific_configuration_sha256({
        key: value for key, value in original.items() if key != "stated_subgroups"
    })
    with pytest.raises(study_contexts.StudyContextError):
        study_contexts.upsert_context({**raw, "stated_subgroups": [record]})
    saved = study_contexts.upsert_context({**raw, "stated_subgroups": [record]}, _server_question_wording_write=True)
    assert study_contexts.get_context(saved["id"])["stated_subgroups"] == [record]
    assert digest != study_contexts.scientific_configuration_sha256(saved)
    changed = study_contexts.upsert_context({**raw, "question": "描述指标分布。"})
    assert changed["stated_subgroups"] == []
    restored = study_contexts.restore_turn_configuration_snapshot(
        saved["id"], {"schema_version": "easyicu.pi-turn-study-snapshot/1",
                      "study_context_id": saved["id"], "source_revision": saved["revision"],
                      "configuration": {"question": saved["question"], "stated_subgroups": [record]}},
        expected_revision=changed["revision"],
    )
    assert restored["stated_subgroups"] == [record]


@pytest.mark.parametrize("concepts,status", [((), "unavailable"), (("sex",), "available")])
def test_source_availability_is_preserved_without_dropping_the_requirement(monkeypatch, concepts, status):
    result, writes = _save(monkeypatch, concepts=concepts)
    assert result["status"] == "ok"
    assert writes[0]["stated_subgroups"][0]["source_status"] == status


def test_a_saved_reading_survives_empty_responses_and_a_source_change(monkeypatch):
    _, writes = _save(monkeypatch)
    saved = {**writes[0], "id": "subgroup-study", "revision": 2,
             "data_source": {"path": "/synthetic/source", "database": "miiv"}}
    result, writes = _save(monkeypatch, current=saved, statements=[], concepts=())
    assert result["status"] == "ok"
    assert writes[0]["stated_subgroups"][0]["source_status"] == "unavailable"
    compiled = agent_pipeline_runs._research_user_preferences({**saved, "stated_subgroups": writes[0]["stated_subgroups"]})
    assert json.loads(compiled["data_constraints"])["stated_subgroups"][0]["source_status"] == "unavailable"


def test_a_reading_saved_without_a_source_is_checked_after_binding(monkeypatch):
    question = "按性别分层，指标与结局的关系如何？"
    result, writes = _save(monkeypatch, current={"id": "subgroup-study", "revision": 1, "question": question})
    assert result["status"] == "ok"
    assert writes[0]["stated_subgroups"][0]["source_status"] == "unbound"
    compiled = agent_pipeline_runs._research_user_preferences({
        **writes[0], "data_source": {"path": "/synthetic/source", "database": "miiv"},
    })
    assert json.loads(compiled["data_constraints"])["stated_subgroups"][0]["source_status"] == "available"


def test_a_model_lookup_reaches_the_planning_roster_without_becoming_exposure(monkeypatch):
    from easyicu.webserver.research_launch_scientific import _metadata_only_planning_coordinates

    result, writes = _save(
        monkeypatch, question="按试验分组比较指标与结局的关系。", concepts=("trial_group",),
        statements=[{"variable": "试验分组", "words": "按试验分组比较", "concept": "trial_group"}],
    )
    assert result["status"] == "ok"
    coordinates = _metadata_only_planning_coordinates(
        question=writes[0]["question"], database="miiv", export_path="/synthetic/source",
        stated_subgroups=writes[0]["stated_subgroups"],
    )
    assert coordinates["required_subgroup_concepts"] == ["trial_group"]
    assert coordinates["primary_exposure"] != "trial_group"
    assert not coordinates["required_adjustment_concepts"]


def test_a_model_lookup_not_offered_by_the_source_returns_unread(monkeypatch):
    result, writes = _save(
        monkeypatch, question="按试验分组比较指标与结局的关系。", concepts=("trial_group",),
        statements=[{"variable": "试验分组", "words": "按试验分组比较", "concept": "absent_trial_group"}],
    )
    assert result["status"] == "ok" and not writes[0]["stated_subgroups"]
    assert result["details"]["stated_subgroups_unread"]


def test_a_bad_reading_does_not_discard_an_independent_valid_reading(monkeypatch):
    from easyicu.webserver.research_launch_scientific import checked_stated_subgroups

    _save(monkeypatch)
    words = "分层" + "组" * 241
    rejected = []
    records = checked_stated_subgroups(
        [{"variable": "组", "words": words}, {"variable": "性别", "words": "按性别分层"}],
        question=f"{words}，按性别分层。", source={"path": "/synthetic/source"}, rejected=rejected,
    )
    assert len(rejected) == 1
    assert [item["concept"] for item in records] == ["sex"]


def test_a_subgroup_reading_cannot_become_the_exposure(monkeypatch):
    from easyicu.webserver.research_launch_scientific import _metadata_only_planning_coordinates

    result, writes = _save(monkeypatch, variable="乳酸", words="按乳酸分层", concepts=("lact",))
    assert result["status"] == "ok"
    coordinates = _metadata_only_planning_coordinates(
        question=writes[0]["question"], database="miiv", export_path="/synthetic/source",
        stated_subgroups=writes[0]["stated_subgroups"],
    )
    assert coordinates["required_subgroup_concepts"] == ["lact"]
    assert coordinates["primary_exposure"] is None
