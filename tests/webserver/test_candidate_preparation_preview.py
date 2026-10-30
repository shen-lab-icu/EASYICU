"""A research-input preview is review evidence, never a data grant."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ConceptDescriptor, TimeWindow
from easyicu.webserver import first_observed_preparation, raw_preparation_dependencies as dependencies
from easyicu.webserver.pi_copilot import population_spec_review
from tests.research_agent.planning.test_a_population_spec_compiles_to_the_cohort_it_states import _ctx


def _case(tmp_path, monkeypatch):
    context = _ctx().model_copy(update={
        "variables": [ConceptDescriptor(name="creat_first", dtype="float64", source_concept="creat", analysis_window="baseline")],
        "time_windows": [TimeWindow(name="baseline", anchor="icu_admission", start_hours=-24, end_hours=0)],
    })
    raw = context.model_dump_json().encode()
    (tmp_path / "research_context.json").write_bytes(raw)
    plan = AnalysisPlan(research_question="synthetic", steps=[AnalysisStep(step_id="primary", intent="estimate",
        inputs=["creat_first", "unmapped_input"])])
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={})
    spec = PopulationSpec.model_validate({"criteria": [
        {"id": "c1", "kind": "first_observed_icu_single_hospital_adult", "role": "include", "source": "question", "quote": "first observed"},
        {"id": "c2", "kind": "measurement", "role": "include", "source": "question", "quote": "creatinine at most four", "concept": "creat", "summary": "first", "window": {"start_hours": -24, "end_hours": 0}, "op": "<=", "value": 4},
        {"id": "c3", "kind": "diagnosis_codes", "role": "exclude", "source": "question", "quote": "exclude K92.2", "system": "icd10", "codes": ["K92.2"]},
    ]})
    scope = {"context_sha256": hashlib.sha256(raw).hexdigest(), "scientific_review_sha256": "b" * 64}
    request = SimpleNamespace(payload={"plan_review_authority": authority.model_dump(mode="json"),
                                       first_observed_preparation._SCOPE_KEY: scope})
    pending = SimpleNamespace(requests=[request])
    monkeypatch.setattr(first_observed_preparation, "proposed_first_observed_scope", lambda **_: scope)
    monkeypatch.setattr(population_spec_review, "reviewed_population_spec", lambda **_: {
        "plan_sha256": authority.plan_sha256,
        "criteria": [{"criterion": item.model_dump(mode="json")} for item in spec.criteria],
    })
    return pending, SimpleNamespace(request=request), authority


def test_preview_names_declared_inputs_conditions_and_windows_but_blocks_preparation(tmp_path, monkeypatch):
    from easyicu.webserver.run_artifact_disclosure import scan_browser_projection

    pending, checkpoint, authority = _case(tmp_path, monkeypatch)
    preview = dependencies.first_observed_candidate_preparation_preview(run_dir=tmp_path, pending=pending, checkpoint=checkpoint)
    assert preview["plan_sha256"] == authority.plan_sha256
    assert preview["step_inputs"][0]["source_concept"] == "creat"
    assert preview["step_inputs"][0]["context_window"] == "baseline"
    assert preview["context_windows"][0]["start_hours"] == -24
    assert preview["population_needs"][1]["window"]["end_hours"] == 0
    assert preview["population_needs"][1]["operator"] == "<="
    assert preview["population_needs"][1]["value"] == 4
    assert preview["population_needs"][2]["diagnosis_codes"] == ["K92.2"]
    assert "candidate_step_input_binding_unverified" in preview["blockers"]
    assert preview["complete"] is preview["preparation_authorized"] is preview["analysis_authorized"] is False
    assert scan_browser_projection(preview)["passed"] is True


def test_preview_refuses_changed_context(tmp_path, monkeypatch):
    pending, checkpoint, _ = _case(tmp_path, monkeypatch)
    (tmp_path / "research_context.json").write_text("{}")
    with pytest.raises(ValueError, match="context_changed"):
        dependencies.first_observed_candidate_preparation_preview(run_dir=tmp_path, pending=pending, checkpoint=checkpoint)
