"""Stated scores keep their identity through source binding and preparation."""

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.research_launch_scientific import (
    checked_stated_adjustments,
    _metadata_only_planning_coordinates,
)
from tests.support.official_demo_raw import official_raw as official_raw


def _reading(name):
    return {"variable": name, "words": f"调整{name}后"}


@pytest.mark.parametrize("name,concept", [
    ("APACHE IV 评分", "apache_iv"), ("APACHE-IV score", "apache_iv"),
    ("APACHE IVa", "apache_iv"), ("SOFA 评分", "sofa"),
    ("SOFA-2 评分", "sofa2"), ("SAPS 3 评分", "saps3"),
    ("SAPS3", "saps3"),
])
def test_a_named_score_is_resolved_and_required_for_preparation(monkeypatch, name, concept):
    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _: SimpleNamespace(concepts=[SimpleNamespace(concept_id=concept)]),
    )
    reading = _reading(name)
    question = f"{reading['words']}，指标与结局有何关联？"
    database = "sic" if concept == "saps3" else "eicu"
    record, = checked_stated_adjustments([reading], question=question, source={"path": "/synthetic/export"})
    assert record == {**reading, "concept": concept, "source_status": "available"}
    coordinates = _metadata_only_planning_coordinates(
        question=question, database=database, export_path="/synthetic/export",
        stated_adjustments=[{**reading, "concept": None, "source_status": "unresolved"}],
    )
    assert coordinates["required_adjustment_concepts"] == [concept]


@pytest.mark.parametrize("name,concept,status", [
    ("APACHE II 评分", None, "unresolved"), ("APACHE III 评分", None, "unresolved"),
    ("APACHE V 评分", None, "unresolved"), ("SOFA 3 评分", None, "unresolved"),
    ("SAPS II 评分", None, "unresolved"), ("OASIS 评分", None, "unresolved"),
    ("APACHE IVa Predicted Hospital Mortality", "apache_iv_pred_hosp_mort", "available"),
    ("APACHE IV Predicted Hospital Mortality", "apache_iv_pred_hosp_mort", "available"),
    ("APACHE IV 预测院内死亡率", "apache_iv_pred_hosp_mort", "available"),
])
def test_other_versions_and_predicted_risk_are_not_replaced_by_a_score(monkeypatch, name, concept, status):
    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _: SimpleNamespace(concepts=[SimpleNamespace(concept_id=key)
                                            for key in ("apache_iv", "sofa", "saps3", "apache_iv_pred_hosp_mort")]),
    )
    reading = _reading(name)
    record, = checked_stated_adjustments([reading], question=reading["words"], source={"path": "/synthetic/export"})
    assert (record["concept"], record["source_status"]) == (concept, status)


@pytest.mark.parametrize("database,concept,name,status", [
    ("eicu", "apache_iv", "APACHE IV 评分", "available"),
    ("miiv", "apache_iv", "APACHE IV 评分", "unavailable"),
    ("miiv", "sofa", "SOFA 评分", "available"),
    ("eicu", "saps3", "SAPS 3 评分", "unavailable"),
    ("miiv", "saps3", "SAPS 3 评分", "unavailable"),
    ("sic", "saps3", "SAPS 3 评分", "available"),
])
def test_database_capability_does_not_substitute_an_unavailable_score(monkeypatch, database, concept, name, status):
    catalog = build_database_capability_catalog(database)
    monkeypatch.setattr("easyicu.research_agent.acquisition.catalog.build_available_catalog", lambda _: catalog)
    reading = _reading(name)
    record, = checked_stated_adjustments([reading], question=reading["words"], source={"path": "/synthetic/export"})
    assert record == {**reading, "concept": concept, "source_status": status}


def test_verified_official_raw_source_requests_the_score_for_later_preparation(official_raw):
    paths, source, _ = official_raw
    reading = _reading("APACHE IV 评分")
    coordinates = _metadata_only_planning_coordinates(
        question=f"{reading['words']}，指标与结局有何关联？", database=source.database,
        export_path=str(paths.raw), stated_adjustments=[{**reading, "concept": None, "source_status": "unresolved"}],
    )
    assert coordinates["required_adjustment_concepts"] == ["apache_iv"]
    assert not (paths.raw / "_manifest.json").exists()


def test_mimic_raw_planning_requests_sofa_without_promising_native_apache(monkeypatch):
    monkeypatch.setattr("easyicu.webserver.demo_source_storage.verified_public_demo_raw",
                        lambda _: SimpleNamespace(database="miiv"))
    readings = [_reading(name) for name in ("SOFA 评分", "APACHE IV 评分")]
    question = "，".join(item["words"] for item in readings)
    records = checked_stated_adjustments(readings, question=question, source={"path": "/synthetic/raw"})
    assert [(item["concept"], item["source_status"]) for item in records] == [
        ("sofa", "available"), ("apache_iv", "unavailable"),
    ]


def test_final_prepared_input_cannot_borrow_the_database_capability_menu(monkeypatch):
    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _: SimpleNamespace(concepts=[]),
    )
    reading = _reading("APACHE IV 评分")
    record, = checked_stated_adjustments(
        [reading], question=reading["words"], source={"path": "/synthetic/final", "database": "eicu"},
    )
    assert record == {**reading, "concept": "apache_iv", "source_status": "unavailable"}


def test_launch_preparation_retains_the_rechecked_adjustment(monkeypatch):
    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _: SimpleNamespace(concepts=[SimpleNamespace(concept_id="apache_iv")]),
    )
    reading = _reading("APACHE IV 评分")
    study = {"id": "synthetic", "question": f"{reading['words']}，指标与结局有何关联？",
             "data_source": {"database": "eicu", "path": "/synthetic/export"},
             "stated_adjustments": [{**reading, "concept": None, "source_status": "unresolved"}]}
    prefs = agent_pipeline_runs._research_user_preferences(study)
    assert json.loads(prefs["data_constraints"])["stated_adjustments"][0]["concept"] == "apache_iv"
    assert study["stated_adjustments"][0]["concept"] is None


@pytest.mark.parametrize("words", ["不调整SOFA 评分", "不调整APACHE IV 评分"])
def test_a_negated_score_is_not_saved_as_a_required_adjustment(monkeypatch, words):
    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _: SimpleNamespace(concepts=[SimpleNamespace(concept_id=name) for name in ("sofa", "apache_iv")]),
    )
    with pytest.raises(ValueError, match="non-negated"):
        checked_stated_adjustments(
            [{"variable": words[3:], "words": words}], question=words, source={"path": "/synthetic/export"},
        )


@pytest.mark.parametrize("name", ["", "x" * 161, "score\nprivate", 7, None])
def test_only_a_bounded_variable_name_crosses_the_stop_projection(name):
    from easyicu.webserver.stop_next_actions import safe_stop_coordinates

    assert "stated_variable" not in safe_stop_coordinates({"stated_variable": name})
    assert safe_stop_coordinates({"stated_variable": "试验量表甲"}) == {"stated_variable": "试验量表甲"}
