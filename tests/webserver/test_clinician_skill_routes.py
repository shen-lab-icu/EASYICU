"""ASGI contracts for the metadata-only and scalar clinician skill entries."""

from fastapi.testclient import TestClient
import pytest

from easyicu.research_agent import concept_catalog, skill_packages
from easyicu.webserver.app import app

PREFIX = "/api/capabilities/method-skills"


def test_definition_route_preserves_repeated_concepts_and_demo_mapping():
    params = [("concepts", "sofa2"), ("concepts", "sep3"), ("database", "eicu_demo")]
    response = TestClient(app).get(f"{PREFIX}/clinical-concept-definitions/projection", params=params)
    assert response.status_code == 200
    assert response.json() == concept_catalog.clinical_definitions(["sofa2", "sep3"], database="eicu_demo")


def test_extraction_route_preserves_the_unsubmitted_prefill():
    response = TestClient(app).get(f"{PREFIX}/six-database-cohort-extraction/projection", params={"database": "miiv"})
    assert response.status_code == 200
    assert response.json() == concept_catalog.extraction_projection("miiv")


@pytest.mark.parametrize("skill,params,code", [
    ("clinical-concept-definitions", {"concepts": "unknown_composite"}, "definition.concept_unknown"),
    ("clinical-concept-definitions", {"concepts": ""}, "definition.concept_unknown"),
    ("clinical-concept-definitions", {"concepts": ["sofa2"] * 257}, "definition.concepts_required"),
    ("clinical-concept-definitions", {"concepts": "sofa2", "database": "unknown_database"}, "definition.database_unknown"),
    ("six-database-cohort-extraction", {"database": "unknown_database"}, "definition.database_unknown"),
    ("six-database-cohort-extraction", {"database": ""}, "definition.database_unknown"),
])
def test_definition_routes_preserve_stable_refusal_codes(skill, params, code):
    response = TestClient(app).get(f"{PREFIX}/{skill}/projection", params=params)
    assert response.status_code == 422
    assert response.json() == {"detail": {"error": code}}


@pytest.mark.parametrize("calculation,inputs", [
    ("two_means", {"difference": 5, "standard_deviation": 10}),
    ("two_proportions", {"p1": .5, "p2": .65}),
    ("survival_events", {"hazard_ratio": .7}),
    ("prediction_binary", {"parameters": 24, "cox_snell_r2": .051, "prevalence": .174}),
])
def test_each_calculation_returns_unchanged_owner_result(calculation, inputs):
    response = TestClient(app).post(f"{PREFIX}/sample-size-and-power/calculate", json={"calculation": calculation, **inputs})
    assert response.status_code == 200
    assert response.json() == getattr(skill_packages, calculation)(**inputs)
    assert response.json()["claim_ceiling"] == "analysis_only"


@pytest.mark.parametrize("patch", [
    {"difference": True}, {"difference": "5"}, {"difference": 1e100},
    {"allocation_ratio": 101}, {"alpha": 0},
    {"calculation": "summarize_treatment_patterns"},
    {"standard_deviation": None}, {"patient_rows": []},
    {"basis": "governed_aggregate"}, {"aggregate_receipt": "synthetic-unverified"},
])
def test_request_validation_refuses_before_calculation(patch, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid request reached a planning calculation")

    monkeypatch.setattr(skill_packages, "two_means", forbidden)
    payload = {"calculation": "two_means", "difference": 5, "standard_deviation": 10, **patch}
    response = TestClient(app).post(f"{PREFIX}/sample-size-and-power/calculate", json=payload)
    assert response.status_code == 422


def test_semantic_planning_error_keeps_its_sk4_code():
    response = TestClient(app).post(f"{PREFIX}/sample-size-and-power/calculate", json={"calculation": "survival_events", "hazard_ratio": 1})
    assert response.status_code == 422
    assert response.json() == {"detail": {"error": "planning.null_effect"}}


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e999"])
def test_nonfinite_json_values_receive_serializable_refusals(value):
    response = TestClient(app).post(
        f"{PREFIX}/sample-size-and-power/calculate",
        content='{"calculation":"two_means","standard_deviation":10,"difference":' + value + '}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert all(set(issue) == {"loc", "type", "msg"} for issue in response.json()["detail"])


def test_new_routes_remain_read_only_in_hosted_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(tmp_path / "empty-data-root"))
    client = TestClient(app)
    assert client.get(f"{PREFIX}/clinical-concept-definitions/projection", params={"concepts": "sofa2", "database": "miiv"}).status_code == 200
    assert client.get(f"{PREFIX}/six-database-cohort-extraction/projection", params={"database": "miiv"}).status_code == 200
    response = client.post(f"{PREFIX}/sample-size-and-power/calculate", json={"calculation": "survival_events", "hazard_ratio": .7})
    assert response.status_code == 200
    assert response.json()["patient_data_read"] is False
    assert list(tmp_path.iterdir()) == []
