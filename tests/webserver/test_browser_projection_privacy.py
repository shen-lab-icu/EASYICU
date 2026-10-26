"""Synthetic browser projections distinguish column prose from identifier values."""

from copy import deepcopy
from pathlib import Path

import pytest

from easyicu.webserver.literature_projection import project_run_literature
from easyicu.webserver.run_artifact_disclosure import (
    browser_review_summary,
    privacy_blocked_projection,
    scan_browser_projection,
)


@pytest.fixture
def prepared_projection():
    application = (
        "Use the synthetic measurement, outcome, stratum, and patient_stay_id, "
        "describe the completeness assumptions before comparing estimates."
    )
    binding = {
        "citation_key": "synthetic_method",
        "design_elements": ["missing_data"],
        "application": application,
    }
    plan = {
        "research_question": "Compare a synthetic measurement across strata.",
        "steps": [{
            "step_id": "landmark_stratified_model",
            "intent": "At a 24-hour landmark, fit the association within each stratum.",
            "planned_analysis_role": "primary",
            "inputs": ["patient_stay_id", "measurement", "outcome", "stratum"],
            "dependence": {"group_source": "patient_stay_id"},
            "literature_citation_keys": ["synthetic_method"],
            "literature_design_bindings": [binding],
        }],
    }
    literature = project_run_literature(
        run_id="synthetic-prepared",
        bundle={"citations": [{"key": "synthetic_method", "title": "Synthetic method"}]},
        plan=plan,
    )
    review = {"facts": {"literature_design_bindings": {"steps": [{
        "citations": [deepcopy(binding)],
    }]}}}
    return {
        "agent_plan.json": plan,
        "literature_evidence.json": literature,
        "scientific_plan_review.json": review,
    }


def test_prepared_landmark_stratified_projection_preserves_column_prose(prepared_projection):
    before = deepcopy(prepared_projection)
    application = prepared_projection["agent_plan.json"]["steps"][0]["literature_design_bindings"][0]["application"]
    assert prepared_projection["literature_evidence.json"]["step_citation_map"][0]["citation_bindings"][0]["application"] == application
    scan = scan_browser_projection(prepared_projection)
    assert scan == {
        "passed": True, "scanned_artifacts": 3, "unsafe_value_count": 0, "hits": [],
    }
    assert prepared_projection == before
    assert browser_review_summary(application) == application


@pytest.mark.parametrize("column", ["patient_stay_id", "subject_id", "hadm_id", "patient_id", "mrn"])
def test_column_name_before_a_prose_clause_is_not_a_row_value(column):
    projection = {"plan": {
        "inputs": [column], "dependence": {"group_source": column},
        "application": f"After grouping by {column}, describe the synthetic design.",
    }}
    assert scan_browser_projection(projection)["passed"] is True


@pytest.mark.parametrize(("value", "reason"), [
    ("/home/synthetic/source.csv", "pattern_1"),
    ("/Users/synthetic/source.csv", "pattern_1"),
    ("file:///synthetic/source.csv", "pattern_1"),
    (r"C:\synthetic\source.csv", "pattern_1"),
    (Path("synthetic.csv"), "path_object"),
    ("Bearer synthetic-only-secret", "pattern_2"),
    ("api_key=synthetic-only-secret", "pattern_2"),
    ("token: synthetic-only-secret", "pattern_2"),
    ("sk-synthetic-only-secret", "pattern_2"),
    ("-----BEGIN PRIVATE KEY-----", "pattern_2"),
    ("patient_stay_id: synthetic-row-123", "pattern_3"),
    ("subject_id=987654", "pattern_3"),
    ('"hadm_id": "synthetic"', "pattern_3"),
    ("patient_id: synthetic", "pattern_3"),
    ("mrn=synthetic", "pattern_3"),
    ("stay_id, 987654", "pattern_3"),
    ("patient_stay_id, synthetic-row-123 is present", "pattern_3"),
    ("stay_id, synthetic_row is present", "pattern_3"),
    ('stay_id, "synthetic" is present', "pattern_3"),
    ("stay_id, 'synthetic' is present", "pattern_3"),
    ("stay_id, synthetic", "pattern_3"),
    ("stay_id, synthetic, next_field", "pattern_3"),
    ("stay_id, synthetic; next field", "pattern_3"),
    ("stay_id, synthetic. Next sentence.", "pattern_3"),
    ("stay_id, synthetic\nNext row", "pattern_3"),
])
def test_unsafe_values_still_withhold_a_prepared_projection(prepared_projection, value, reason):
    # A safe mention elsewhere cannot exempt this field, including group_source.
    prepared_projection["agent_plan.json"]["steps"][0]["dependence"]["group_source"] = value
    scan = scan_browser_projection(prepared_projection)
    assert scan["passed"] is False
    assert scan["unsafe_value_count"] == 1
    assert scan["hits"] == [{
        "path": "agent_plan.json.steps[0].dependence.group_source", "reason": reason,
    }]
    blocked = privacy_blocked_projection(
        run_context={"run_id": "synthetic-prepared", "study_id": "synthetic-study"},
        scan=scan,
    )
    assert set(blocked) == {"run_context.json", "quality_gate.json"}
    gate = blocked["quality_gate.json"]
    assert gate["gate"]["reason"] == "research_pipeline_projection_privacy_blocked"
    assert gate["privacy"]["payloads_withheld"] is True
    assert scan_browser_projection(blocked)["passed"] is True


@pytest.mark.parametrize("unsafe", [
    "stay_id: synthetic-row-123", "/tmp/synthetic/source.csv", "password=synthetic-only",
])
def test_prose_exception_does_not_hide_an_unsafe_value_in_the_same_string(unsafe):
    text = f"Group on patient_stay_id, describe the design. {unsafe}"
    assert scan_browser_projection({"application": text})["passed"] is False
