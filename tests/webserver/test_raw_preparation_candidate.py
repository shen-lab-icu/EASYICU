from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from easyicu.webserver import agent_pipeline_runs, raw_preparation_candidate, sources
from easyicu.webserver.routes.agent import raw_preparation_candidate as candidate_route
from tests.webserver.test_sic_raw_preparation_grant import _case


def test_real_review_owner_builds_candidate_without_approving_or_starting(tmp_path, monkeypatch):
    result, source_root, _, checkpoint = _case(tmp_path, monkeypatch, approved=False)
    (tmp_path / "research_context.json").write_text(result.context.model_dump_json())
    pending = SimpleNamespace(run_id="run", run_dir=str(tmp_path), requests=(checkpoint.request,))
    monkeypatch.setattr(agent_pipeline_runs, "raw_preparation_review_owner", lambda run_id, study_id: (pending, {"id": study_id}))
    monkeypatch.setattr(sources, "load_registry", lambda: {"sources": [{"path": str(source_root)}]})
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    assert outcome["status"] == "blocked"
    assert outcome["blocker_codes"]
    assert not (tmp_path / "raw_preparation_checkpoint.json").exists()
    assert not (tmp_path / "approved_executable_plan.json").exists()


def test_route_requires_bound_conversation_and_is_only_a_preview(monkeypatch):
    from easyicu.webserver.pi_copilot import service

    calls = []
    owner = SimpleNamespace(host_action_session=lambda session, **kwargs: calls.append((session, kwargs)))
    monkeypatch.setattr(service, "get_pi_copilot_service", lambda: owner)
    monkeypatch.setattr(raw_preparation_candidate, "create_candidate_for_review",
                        lambda **kwargs: {"status": "blocked", "blocker_codes": ["dependency_partition_roster_unknown"]})
    body = {"run_id": "run", "study_context_id": "study", "session_id": "session", "project_id": "project"}
    assert candidate_route(body)["status"] == "blocked"
    assert calls == [("session", {"project_id": "project", "study_context_id": "study"})]
    with pytest.raises(HTTPException) as error:
        candidate_route({})
    assert error.value.status_code == 400


def test_candidate_includes_reviewed_predicate_concept_missing_from_variables(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.research_agent.planning.cohort_contract import CohortDefinition, ConceptPredicate, TimeWindow

    result, source_root, _, checkpoint = _case(tmp_path, monkeypatch, approved=False)
    context = result.context.model_copy(update={'variables': [v for v in result.context.variables if v.name != 'age']})
    predicate = ConceptPredicate(concept_id='age', time_window=TimeWindow(
        anchor='icu_admission', start_offset_hours=0, end_offset_hours=24),
        aggregation='first', op='>=', value=18)
    plan = result.plan.model_copy(update={'cohort': CohortDefinition(name='typed', inclusion=(predicate,))})
    (tmp_path / 'research_context.json').write_text(context.model_dump_json())
    pending = SimpleNamespace(run_id='run', run_dir=str(tmp_path), requests=(checkpoint.request,))
    monkeypatch.setattr(agent_pipeline_runs, 'raw_preparation_review_owner', lambda *_: (pending, {'id': 'study'}))
    monkeypatch.setattr(sources, 'load_registry', lambda: {'sources': [{'path': str(source_root)}]})
    monkeypatch.setattr(raw_preparation_candidate, '_plan_authority', lambda _: PlanReviewAuthority.create(plan=plan))
    real_capture = raw_preparation_candidate.capture_data_provenance
    seen = []
    def observe(*args, **kwargs):
        seen.extend(kwargs['concepts'])
        return real_capture(*args, **kwargs)
    monkeypatch.setattr(raw_preparation_candidate, 'capture_data_provenance', observe)
    monkeypatch.setattr(raw_preparation_candidate, 'candidate_from_scientific_review',
        lambda **_: {'pending': None, 'dependencies': {'blocker_codes': ['dependency_cohort_inputs_unknown']}})
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id='run', study_context_id='study')
    assert 'age' in seen
    assert outcome['status'] == 'blocked'
    assert not (tmp_path / 'raw_preparation_checkpoint.json').exists()


def test_reviewed_plan_gets_only_non_approvable_partial_multitable_preview(tmp_path, monkeypatch):
    from easyicu.databases.research_plan_requirements import _constraints
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.webserver import reviewed_multitable_source, source_partition_candidates, study_contexts

    result, source_root, _, checkpoint = _case(tmp_path, monkeypatch, approved=False)
    (tmp_path / 'research_context.json').write_text(result.context.model_dump_json())
    evidence = EvidenceStore(tmp_path)
    evidence.register_text(kind='log', description='context', text=result.context.model_dump_json(),
                           filename='context.json', evidence_id='research_context')
    identity = SourceIdentity.from_record(_constraints(result.context)['source_identity'])
    candidate = {'source_identity_sha256': canonical_sha256(identity.to_dict()),
        'research_context_evidence_sha256': evidence.get('research_context').sha256,
        'files': [{'table': 'cases', 'path': 'cases.parquet', 'columns': ['CaseID']},
                  {'table': 'laboratory', 'path': 'laboratory/1.parquet', 'columns': ['CaseID', 'LaboratoryValue']}],
        'analysis_authorized': False, 'outcome_and_cohort_dependencies_complete': False}
    study = {'id': 'study', 'revision': 1}
    pending = SimpleNamespace(run_id='run', run_dir=str(tmp_path), requests=(checkpoint.request,))
    monkeypatch.setattr(agent_pipeline_runs, 'raw_preparation_review_owner', lambda *_: (pending, study))
    monkeypatch.setattr(sources, 'load_registry', lambda: {'sources': [{'path': str(source_root)}]})
    monkeypatch.setattr(study_contexts, 'get_context', lambda _: study)
    monkeypatch.setattr(study_contexts, 'scientific_configuration_sha256', lambda s: s['revision'])
    monkeypatch.setattr(source_partition_candidates, 'candidate_from_bound_research_context', lambda **_: dict(candidate))
    calls = []
    def build(*, identity, requested_columns, plan_sha256):
        calls.append((identity, requested_columns, plan_sha256))
        inputs = [{k: row[k] for k in ('table', 'path', 'columns')} for row in candidate['files']]
        scope = {'adapter_id': 'registered-parquet-projections/2', 'source_identity': identity.to_dict(),
            'inputs': inputs, 'purpose': 'prepare_multiple_tables_only', 'plan_sha256': plan_sha256,
            'dependencies_sha256': 'a' * 64}
        return scope, {'candidate_sha256': 'a' * 64, 'source_identity_sha256': canonical_sha256(identity.to_dict())}
    monkeypatch.setattr(reviewed_multitable_source, 'build_source_bound_multitable_scope', build)
    existing = tmp_path / 'raw_preparation_checkpoint.json'
    existing.write_text('existing checkpoint remains unchanged')
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id='run', study_context_id='study')
    assert outcome['status'] == 'blocked' and outcome['blocker_codes']
    partial = outcome['partial_multitable_projection']
    assert calls[0][2] == canonical_sha256(result.plan.model_dump(mode='json'))
    assert partial['partial_projection_only'] and partial['would_stage_full_input_files'] and partial['would_project_all_rows']
    assert 'full_input_files_staged' not in partial and 'all_rows_projected' not in partial
    assert not any(partial[k] for k in ('cohort_complete', 'joins_performed', 'derived_features_computed',
                                         'dependency_closure_verified', 'approvable', 'analysis_authorized'))
    assert partial['roster_candidate_sha256'] == 'a' * 64 and 'dependencies_sha256' not in partial
    assert existing.read_text() == 'existing checkpoint remains unchanged'
    assert not (tmp_path / 'approved_executable_plan.json').exists()

    # A source or context candidate detached from this reviewed plan cannot
    # become even a partial multi-table scope.
    candidate['research_context_evidence_sha256'] = 'f' * 64
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id='run', study_context_id='study')
    assert 'partial_multitable_projection' not in outcome
    assert len(calls) == 1
    assert existing.read_text() == 'existing checkpoint remains unchanged'

    candidate['research_context_evidence_sha256'] = evidence.get('research_context').sha256
    candidate['files'] = candidate['files'][:1]
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id='run', study_context_id='study')
    assert outcome['status'] == 'blocked' and 'partial_multitable_projection' not in outcome
    assert len(calls) == 1

    candidate['files'].append({'table': 'laboratory', 'path': 'laboratory/1.parquet', 'columns': ['CaseID', 'LaboratoryValue']})
    def build_then_change(**kwargs):
        scope = build(**kwargs)
        monkeypatch.setattr(study_contexts, 'get_context', lambda _: {'id': 'study', 'revision': 2})
        return scope
    monkeypatch.setattr(reviewed_multitable_source, 'build_source_bound_multitable_scope', build_then_change)
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id='run', study_context_id='study')
    assert outcome['status'] == 'blocked' and 'source_partition_candidate' not in outcome
    assert 'partial_multitable_projection' not in outcome
    assert existing.read_text() == 'existing checkpoint remains unchanged'
