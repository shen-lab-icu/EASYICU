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
from tests.support.population_context import _ctx


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


def test_candidate_dependencies_distinguish_typed_products_and_missing_input(tmp_path, monkeypatch):
    pending, checkpoint, _ = _case(tmp_path, monkeypatch)
    preview = dependencies.first_observed_candidate_preparation_preview(run_dir=tmp_path, pending=pending, checkpoint=checkpoint)
    compiled = preview['declared_plan_dependencies']
    assert compiled['required_source_concepts'] == ['creat']
    assert 'candidate_step_input_binding_unverified' in compiled['blockers']
    assert 'candidate_diagnosis_source_binding_unverified' in compiled['blockers']
    assert compiled['typed_step_dependencies'] == {'primary': []}


def test_candidate_dependencies_bind_typed_producer_and_refuse_missing_producer():
    context = _ctx().model_copy(update={'variables': [
        ConceptDescriptor(name='age', dtype='float64', source_concept='age')]})
    spec = PopulationSpec.model_validate({'criteria': []})
    producer = AnalysisStep(step_id='prepare', intent='prepare', inputs=['age'], expected_outputs=['table:cohort'])
    consumer = AnalysisStep(step_id='estimate', intent='estimate', inputs=['table:cohort'])
    plan = AnalysisPlan(research_question='synthetic', steps=[producer, consumer])
    compiled = dependencies._candidate_plan_input_dependencies(plan=plan, context=context, population_spec=spec)
    assert compiled['typed_step_dependencies']['estimate'] == ['prepare']
    assert compiled['required_source_concepts'] == ['age']
    assert compiled['typed_product_inputs'] == [{'step_id': 'estimate', 'product': 'table:cohort'}]
    assert not compiled['blockers']
    missing = plan.model_copy(update={'steps': [consumer]})
    assert 'candidate_typed_plan_graph_invalid' in dependencies._candidate_plan_input_dependencies(
        plan=missing, context=context, population_spec=spec)['blockers']


def test_typed_export_candidate_uses_metadata_without_opening_data_files(tmp_path):
    import json
    from tests.support.typed_export import typed_export

    export = typed_export(tmp_path / 'export', database='eicu')
    raw = (export / '_manifest.json').read_bytes()
    inventory = dependencies.declared_native_export_inventory(export_path=export,
        expected_manifest_sha256=hashlib.sha256(raw).hexdigest(), expected_database='eicu')
    for item in inventory['files']:
        (export / item['file']).unlink()
    columns = dependencies._declared_typed_export_columns(export_path=export, inventory=inventory)
    assert any(c['source_concept'] == 'death' for f in columns['files'] for c in f['declared_columns'])
    assert any('charttime' in f['time_columns'] for f in columns['files'])
    assert all(f['identity_column'] == 'stay_id' for f in columns['files'])
    manifest = json.loads(raw)
    sidecar = export / manifest['column_metadata']['file']
    sidecar.write_bytes(sidecar.read_bytes() + b' ')
    with pytest.raises(ValueError, match='metadata_changed'):
        dependencies._declared_typed_export_columns(export_path=export, inventory=inventory)


def test_projection_binds_candidate_digest_and_blocks_missing_inputs(tmp_path, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs, raw_preparation_projection, study_contexts
    from tests.support.typed_export import typed_export

    export = typed_export(tmp_path / 'export', database='eicu')
    raw = (export / '_manifest.json').read_bytes()
    scope = {'database': 'eicu', 'export_manifest_sha256': hashlib.sha256(raw).hexdigest()}
    run = {'run_id': 'run', 'project_dir': str(tmp_path)}
    pending = SimpleNamespace(run_dir=str(tmp_path))
    checkpoint = SimpleNamespace(request=object())
    study = {'id': 'study'}
    monkeypatch.setattr(agent_pipeline_runs, 'raw_preparation_review_owner', lambda *_: (pending, study))
    monkeypatch.setattr(study_contexts, 'get_context', lambda *_: study)
    monkeypatch.setattr(study_contexts, 'scientific_configuration_sha256', lambda *_: 'f' * 64)
    monkeypatch.setattr(first_observed_preparation, '_host_inputs', lambda **_: (scope, export))
    declared = {'required_source_concepts': ['age', 'death'], 'typed_step_dependencies': {},
                'typed_product_inputs': [], 'host_derivation_sources': {}, 'blockers': []}
    monkeypatch.setattr(dependencies, 'first_observed_candidate_preparation_preview', lambda **_: {
        'declared_plan_dependencies': declared, 'population_needs': [], 'context_windows': []})
    def project():
        return raw_preparation_projection._candidate_projection(
            run=run, study_context_id='study', checkpoint=checkpoint, scope=scope)
    first = project()
    assert first['candidate_read_scope_sha256'] == project()['candidate_read_scope_sha256']
    assert first['candidate_read_scope']['undeclared_source_concepts'] == []
    assert first['candidate_read_scope']['complete'] is False
    assert first['candidate_read_scope']['preparation_authorized'] is False
    declared['required_source_concepts'] = ['age', 'death', 'absent']
    missing = project()
    assert missing['candidate_read_scope_sha256'] != first['candidate_read_scope_sha256']
    assert missing['candidate_read_scope']['undeclared_source_concepts'] == ['absent']
    assert 'candidate_required_source_concepts_unavailable' in missing['candidate_read_scope']['blockers']
