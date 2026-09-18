"""Fresh amendments preserve verified requirements, never old input/approval."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.planning.baseline_requirements import (
    baseline_outline_coverage, baseline_requirement_coverage, bind_baseline_requirements,
)
from easyicu.research_agent.planning.population_requirements import (
    bind_population_requirements, candidate_population_requirements, validate_population_choice,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import ConceptDescriptor
from easyicu.webserver import agent_pipeline_runs, plan_change_requirements as owner, study_contexts
from easyicu.webserver.plan_change_request import PlanChangeRequest, ReferencedPlan, reference_plan_content
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from tests.research_agent.planning.test_baseline_requirements import _context, _plan


@pytest.fixture
def source(tmp_path, monkeypatch):
    study = {"id": "study-synthetic", "question": "Describe a biomarker and hospital death.", "data_source": {"database": "miiv"}}
    digest = study_contexts.scientific_configuration_sha256(study)
    wrapper = tmp_path / study["id"] / 'run-fixture'
    run_id = 'run-source'
    checkpoint_path = wrapper / 'pipeline' / run_id / 'human_review_checkpoint.json'
    checkpoint_path.parent.mkdir(parents=True)

    def build(names=("age", "severity_first"), inherited=None):
        context = _context().model_copy(update={"variables": [
            *_context().variables,
            *[ConceptDescriptor(name=name, source_concept=("severity" if name == "severity_first" else name), role="other", dtype="float64")
              for name in names if name not in {v.name for v in _context().variables}],
        ]})
        population = candidate_population_requirements({"steps": [{
            "step_id": "risk", "expected_outputs": ["table:absolute_risk_context"], "population_scope": "primary_model",
        }]}, "d" * 64)
        context = bind_population_requirements(context, population.model_dump(mode="json"))
        if inherited is not None:
            context = bind_baseline_requirements(context, inherited.model_dump(mode="json"))
        plan = _plan(*names)
        review = build_plan_scientific_review(context=context, plan=plan)
        plan_payload, context_payload = plan.model_dump(mode="json"), context.model_dump(mode="json")
        artifact_sha = hashlib.sha256(json.dumps(plan_payload).encode()).hexdigest()
        checkpoint = HumanReviewCheckpoint.create(
            run_id=run_id, pipeline_config_sha256="b" * 64, environment_identity={},
            llm_signature_sha256="b" * 64, run_input_capsule_sha256="c" * 64,
            capability_activation_sha256="b" * 64, runtime_capabilities=(), runtime_bundle=None,
            requests=(HumanReviewRequest.create(kind='protocol_claim', summary='Review the complete plan.', authority_sha256='b'*64, payload={}),),
            plan_handoff={"plan": plan_payload, "context": context_payload}, execution_coordinates={},
            now=datetime(2020, 1, 1, tzinfo=timezone.utc), ttl=timedelta(hours=1),
        )
        checkpoint_path.write_text(checkpoint.model_dump_json())
        record = SimpleNamespace(
            run_id=run_id, study_id=study['id'], scientific_configuration_sha256=digest,
            artifacts=[SimpleNamespace(name='agent_plan.json', sha256=artifact_sha)],
            artifact_payloads={'agent_plan.json': plan_payload, 'scientific_plan_review.json': review.model_dump(mode='json')},
        )
        row = {'run_id': run_id, 'project_dir': str(wrapper), 'scientific_configuration_sha256': digest}
        monkeypatch.setattr(owner.agent_runs, 'list_run_history', lambda **kwargs: {'runs': [row]})
        monkeypatch.setattr(owner.agent_runs, 'read_run_record', lambda _: record)
        monkeypatch.setattr(owner.agent_runs, 'read_run_review', lambda _: {'artifact_payloads': record.artifact_payloads})
        request = PlanChangeRequest(
            source_run_id=run_id, user_message='Add a function-form sensitivity; retain the rest of the plan.',
            reference_plans=(ReferencedPlan(run_id=run_id, artifact_sha256=artifact_sha, plan=reference_plan_content(plan_payload)),),
        )
        return SimpleNamespace(request=request, context=context, plan=plan, record=record, row=row, checkpoint_path=checkpoint_path,
                               study=study, root=str(tmp_path), digest=digest)
    return build


def bind(fixture, request=None, study=None):
    return owner.bind_plan_change_requirements(request or fixture.request, study=study or fixture.study, project_root=fixture.root)


def compiled_study(f, monkeypatch, tmp_path):
    from easyicu.webserver.pi_copilot import plan_review_progress

    monkeypatch.setenv('EASYICU_HOME', str(tmp_path / 'host-state'))
    before = {**f.study, 'revision': 1}
    after = {
        **before, 'revision': 2, 'covariates': ['age'],
        'covariate_selection': 'exact', 'covariate_authority': 'agent_plan',
        'confirmations': {'agent_plan_configuration_compiled': True},
    }
    plan_review_progress.record_choice(
        before=before, after=after, run=f.row,
        decision_code='agent_plan_configuration', option_id='typed_runtime_projection',
    )
    return after


def test_runtime_projection_retains_full_baseline_without_expanding_adjustment(source, monkeypatch, tmp_path):
    names = ('age', *(f'baseline_{i}' for i in range(14)))
    f = source(names)
    changed = compiled_study(f, monkeypatch, tmp_path)
    request = f.request.model_copy(update={
        'source_scientific_configuration_sha256': f.digest,
        'target_scientific_configuration_sha256': study_contexts.scientific_configuration_sha256(changed),
    })
    bound = bind(f, request, study=changed)
    assert bound.baseline_requirements() is not None
    assert tuple(v.name for v in bound.baseline_requirements().tables[0].variables) == names
    assert set(names) <= set(bound.reference_concepts(set(names)))
    assert changed['covariates'] == ['age']
    assert bound.population_requirements() == bind(f).population_requirements()
    context = bind_baseline_requirements(f.context, bound.baseline_requirements().model_dump(mode='json'))
    assert baseline_outline_coverage(context, [{'step_id': 'small', 'module_id': 'table_one', 'variable_names': ['exposure', 'age']}])['status'] == 'incomplete'
    assert build_plan_scientific_review(context=context, plan=_plan('age')).approval_allowed is False


def test_fresh_runtime_replan_launch_recovers_host_source_without_browser_fields(source, monkeypatch, tmp_path):
    from easyicu.webserver import research_pipeline_run_preparation as preparation
    from tests.webserver.test_research_pipeline_run_preparation import _request, _scientific

    f = source()
    changed = compiled_study(f, monkeypatch, tmp_path)
    request = replace(_request(), study_context=changed, project_root=f.root)
    scientific = replace(_scientific(), study=changed)
    monkeypatch.setattr(preparation.capability_policy, 'capability_settings', lambda: {})
    monkeypatch.setattr(preparation, 'ExtensionRegistry', lambda: SimpleNamespace(
        snapshot=lambda: SimpleNamespace(revision=0, skills=(), mcp_servers=()), pipeline_activation=lambda snapshot: None))
    monkeypatch.setattr(preparation, '_require_profile_dictionaries', lambda **kwargs: None)
    monkeypatch.setattr(preparation, '_require_execution_runtime', lambda **kwargs: None)
    provider = SimpleNamespace(provider={}, provider_environment={}, credential_source='fixture', literature_search_authorized=False)
    _, execution = preparation._prepare_launch_execution(request, scientific, provider)
    assert execution.plan_change_request is not None
    assert execution.plan_change_request.source_run_id == f.request.source_run_id
    assert execution.plan_change_request.baseline_requirements() == bind(f).baseline_requirements()
    assert execution.development_resume_binding is None
    assert execution.plan_revision_source_run_id == execution.execution_resume_source_run_id == ''
    assert request.plan_change_request is None  # No new client field or execution grant.


@pytest.mark.parametrize('drift', ['missing', 'corrupt', 'revision', 'question', 'source', 'user_choice', 'run'])
def test_runtime_continuity_requires_exact_host_receipt(source, monkeypatch, tmp_path, drift):
    from easyicu.webserver.pi_copilot import plan_review_progress

    f = source()
    changed = compiled_study(f, monkeypatch, tmp_path)
    bound = owner.compiled_configuration_plan_change(study=changed, project_root=f.root)
    assert bound.baseline_requirements() is not None
    if drift == 'missing':
        plan_review_progress._path(changed['id']).unlink()
    elif drift == 'corrupt':
        plan_review_progress._path(changed['id']).write_text('{}')
    elif drift == 'revision':
        changed = {**changed, 'revision': 3}
    elif drift == 'question':
        changed = {**changed, 'question': 'New scientific scope.'}
    elif drift == 'source':
        changed = {**changed, 'data_source': {'database': 'eicu'}}
    elif drift == 'user_choice':
        after = {**changed, 'revision': 3}
        plan_review_progress.record_choice(before=changed, after=after, run=f.row,
            decision_code='ADJUSTMENT_SET_NOT_USER_CONFIRMED', option_id='accept_proposed_adjustment')
        changed = after
    else:
        f.row['run_id'] = 'different-run'
    assert owner.compiled_configuration_plan_change(study=changed, project_root=f.root) is None
    with pytest.raises(ResearchPipelineRunError):
        bind(f, bound, study=changed)


@pytest.mark.parametrize('names', [
    ('age', 'severity_first'), ('age', 'severity_first', 'new_marker'),
    tuple(f'baseline_{i}' for i in range(21)),
])
def test_fresh_revision_binds_complete_roster_through_outline_and_final_review(source, names):
    f = source(names)
    request = bind(f)
    required = request.baseline_requirements()
    assert tuple(v.name for v in required.tables[0].variables) == names
    context = bind_baseline_requirements(f.context, required.model_dump(mode='json'))
    missing_plan = _plan(*names[:-1])
    coverage = baseline_requirement_coverage(context, missing_plan)
    assert coverage['status'] == 'incomplete'
    assert coverage['tables'][0]['missing_variables'] == [names[-1]]
    review = build_plan_scientific_review(context=context, plan=missing_plan)
    assert 'ACCEPTED_BASELINE_CONTENT_MISSING' in {finding.code for finding in review.findings}
    assert review.approval_allowed is False
    assert baseline_outline_coverage(context, [{'step_id': 'renamed', 'module_id': 'table_one', 'variable_names': ['exposure', *names[:-1]]}])['status'] == 'incomplete'
    assert baseline_requirement_coverage(context, f.plan)['status'] == 'complete'
    assert baseline_outline_coverage(context, [{'step_id': 'renamed', 'module_id': 'table_one', 'variable_names': ['exposure', *names]}])['status'] == 'complete'
    assert not any(v.name == 'outcome' for v in required.tables[0].variables)


def test_inherited_requirements_survive_even_an_incomplete_source_candidate(source):
    previous = source()
    inherited = bind(previous).baseline_requirements()
    incomplete = source(('age',), inherited=inherited)
    assert bind(incomplete).baseline_requirements() == inherited


def test_preserves_original_population_and_existing_explicit_amendment_rule(source):
    f = source()
    request = bind(f)
    assert request.population_requirements().populations[0].population_scope == 'primary_model'
    with pytest.raises(ValueError, match='Preserve'):
        validate_population_choice(f.context, product='table:absolute_risk_context', scope='analysis_cohort', change_reason=None)
    validate_population_choice(f.context, product='table:absolute_risk_context', scope='analysis_cohort', change_reason='Intentional descriptive estimand amendment for full review.')


@pytest.mark.parametrize('field,value', [('source_run_id','wrong-run'), ('source_scientific_configuration_sha256','f'*64), ('target_scientific_configuration_sha256','f'*64)])
def test_wrong_source_or_configuration_fails_closed(source, field, value):
    f = source()
    bound = bind(f)
    with pytest.raises(ResearchPipelineRunError, match='exact source'):
        bind(f, bound.model_copy(update={field: value}))


@pytest.mark.parametrize('drift', ['reference_hash', 'reference_content', 'review_plan', 'checkpoint', 'record_identity', 'wrong_study', 'bound_roster'])
def test_source_verification_rejects_tampering(source, drift):
    f = source()
    request = bind(f)
    if drift.startswith('reference_'):
        original = request.reference_plans[0]
        updates = {'artifact_sha256': 'f'*64} if drift == 'reference_hash' else {'plan': reference_plan_content(_plan('age').model_dump(mode='json'))}
        request = request.model_copy(update={'reference_plans': (original.model_copy(update=updates),)})
    elif drift == 'review_plan':
        f.record.artifact_payloads['scientific_plan_review.json']['plan_sha256'] = 'f'*64
    elif drift == 'checkpoint':
        payload = json.loads(f.checkpoint_path.read_text())
        payload['plan_handoff']['context']['variables'].pop()
        f.checkpoint_path.write_text(json.dumps(payload))
    elif drift == 'record_identity':
        f.record.run_id = 'other-run'
    elif drift == 'wrong_study':
        f.record.study_id = 'other-study'
    else:
        request = request.model_copy(update={'source_requirements': request.source_requirements.model_copy(update={'baseline': None})})
    with pytest.raises(ResearchPipelineRunError, match='exact source'):
        bind(f, request)


def test_host_explicit_new_scientific_scope_does_not_inherit_old_requirements(source):
    f = source()
    changed = {**f.study, 'question': 'New authorized question with a different baseline roster and population.'}
    digest = study_contexts.scientific_configuration_sha256(changed)
    with pytest.raises(ResearchPipelineRunError):
        bind(f, study=changed)  # Old readability cannot authorize a silent scope transition.
    request = f.request.model_copy(update={'source_scientific_configuration_sha256': f.digest, 'target_scientific_configuration_sha256': digest})
    bound = bind(f, request, study=changed)
    assert bound.baseline_requirements() is None
    assert bound.population_requirements() is None
    assert bound.reference_concepts({'age', 'severity'}) == ()
    # A compiled old launch cannot be reused after current scope changes.
    with pytest.raises(ResearchPipelineRunError):
        bind(f, bind(f), study=changed)


def test_legacy_json_canonical_and_readback_are_unchanged(source):
    f = source()
    payload = f.request.model_dump(mode='json')
    assert set(payload) == {'schema_version', 'source_run_id', 'user_message', 'reference_plans'}
    assert canonical_sha256(PlanChangeRequest.model_validate(payload).model_dump(mode='json')) == canonical_sha256(payload)
    legacy = {'schema_version': 'easyicu.plan-change-request/1', 'source_run_id': 'old', 'user_message': 'Revise.'}
    assert PlanChangeRequest.model_validate(legacy).model_dump(mode='json') == legacy
    assert 'source_requirements' not in f.request.planner_context()
    assert bind(f, bind(f)) == bind(f)


@pytest.mark.parametrize('runtime_projection', [False, True])
@pytest.mark.parametrize('catalog_variant', ['exact', 'existing_unique_alias'])
def test_sealed_operationalized_coordinates_reach_zero_row_menu(source, monkeypatch, tmp_path, catalog_variant, runtime_projection):
    import pyarrow.parquet as pq
    from easyicu.research_agent.acquisition.catalog import AvailableCatalog, CatalogConcept
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    f = source()
    request = (owner.compiled_configuration_plan_change(
        study=compiled_study(f, monkeypatch, tmp_path), project_root=f.root,
    ) if runtime_projection else bind(f))
    assert 'severity' in request.source_requirements.planning_concepts
    assert 'severity_first' in request.source_requirements.operationalized_columns
    catalog = AvailableCatalog(source='fixture', concepts=[CatalogConcept(
        'severity_value' if catalog_variant == 'existing_unique_alias' and name == 'severity' else name,
    ) for name in request.source_requirements.planning_concepts])
    monkeypatch.setattr(agent_pipeline_runs, '_metadata_only_planning_catalog', lambda **kwargs: catalog)
    llm = ScriptedMockLLMClient([json.dumps({'selected_concepts':['age'], 'rationale':'Only age selected by model.', 'inclusion_exclusion':[]})])
    result = agent_pipeline_runs._metadata_only_planning_acquisition(database='miiv', question='Describe baseline.', llm=llm, output_dir=tmp_path/'catalog', plan_change_request=request)
    assert not result.blocked
    assert {'severity', 'severity_first', 'age'} <= set(pq.read_schema(result.universe_path).names)
    assert pq.read_metadata(result.universe_path).num_rows == 0
    # The current catalog cannot support severity: reject before even mock model selection.
    catalog.concepts = [CatalogConcept('age')]
    untouched = ScriptedMockLLMClient([])
    with pytest.raises(ResearchPipelineRunError) as caught:
        agent_pipeline_runs._metadata_only_planning_acquisition(database='miiv', question='Describe baseline.', llm=untouched, output_dir=tmp_path/'missing', plan_change_request=request)
    assert caught.value.code == 'plan_change_required_concepts_unavailable'
    assert untouched.calls == [] and not (tmp_path/'missing').exists()


@pytest.mark.parametrize('feedback_failure', ['changed', 'missing', 'over_budget'])
def test_compiled_review_feedback_fails_before_new_provider_work(source, monkeypatch, tmp_path, feedback_failure):
    f = source()
    changed = compiled_study(f, monkeypatch, tmp_path)
    review = json.loads(json.dumps(f.record.artifact_payloads['scientific_plan_review.json']))
    if feedback_failure == 'changed':
        review['context_sha256'] = 'f' * 64
    elif feedback_failure == 'over_budget':
        review['findings'][0]['message'] = 'x' * 13000
    result = {} if feedback_failure == 'missing' else {'artifact_payloads': {'scientific_plan_review.json': review}}
    monkeypatch.setattr(owner.agent_runs, 'read_run_review', lambda _: result)
    with pytest.raises(ResearchPipelineRunError) as raised:
        owner.compiled_configuration_plan_change(study=changed, project_root=f.root)
    assert raised.value.code == 'plan_change_review_feedback_invalid'


def test_launch_preparation_binds_before_runtime_and_runner_rechecks(source, monkeypatch, tmp_path):
    from easyicu.webserver import research_pipeline_run_preparation as preparation
    from tests.webserver.test_research_pipeline_run_preparation import _request, _scientific
    f = source()
    request = replace(_request(), study_context=f.study, project_root=f.root, plan_change_request=f.request)
    scientific = replace(_scientific(), study=f.study)
    monkeypatch.setattr(preparation.capability_policy, 'capability_settings', lambda: {})
    monkeypatch.setattr(preparation, 'ExtensionRegistry', lambda: SimpleNamespace(
        snapshot=lambda: SimpleNamespace(revision=0, skills=(), mcp_servers=()), pipeline_activation=lambda snapshot: None))
    monkeypatch.setattr(preparation, '_require_profile_dictionaries', lambda **kwargs: None)
    monkeypatch.setattr(preparation, '_require_execution_runtime', lambda **kwargs: None)
    provider = SimpleNamespace(provider={}, provider_environment={}, credential_source='fixture', literature_search_authorized=False)
    authority, execution = preparation._prepare_launch_execution(request, scientific, provider)
    assert execution.plan_change_request.baseline_requirements() == bind(f).baseline_requirements()
    prepared = preparation.PreparedResearchPipelineRun(scientific=scientific, authority=authority, execution=execution)
    monkeypatch.setattr(agent_pipeline_runs, 'prepare_research_pipeline_run', lambda _: prepared)
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path='/unused', study_context=f.study, project_root=f.root, provider={}, plan_change_request=f.request,
    )
    # Mutation between launch preparation and actual runner: fail before any
    # new wrapper/Provider budget/session or data-foundation work.
    f.record.artifacts[0].sha256 = 'f'*64
    monkeypatch.setattr(agent_pipeline_runs.RunDirectory, 'create', lambda *args: pytest.fail('No new wrapper before verification'))
    with pytest.raises(ResearchPipelineRunError, match='exact source'):
        runner(SimpleNamespace(id='new-job'))


@pytest.mark.parametrize('change_scope', [False, True])
def test_copilot_amendment_pins_source_and_current_configuration(source, monkeypatch, change_scope):
    from easyicu.webserver.pi_copilot import tools
    f = source()
    current_study = {**f.study, 'question': 'A newly authorized scientific question.'} if change_scope else f.study
    latest = {**f.row, 'study_id': f.study['id'], 'artifact_names': ['agent_plan.json'], 'run_status': 'failed'}
    context = SimpleNamespace(session=SimpleNamespace(binding=object()), user_message=f.request.user_message)
    captured = {}
    monkeypatch.setattr(tools, '_bound_context', lambda _: current_study)
    monkeypatch.setattr(tools, '_select_run', lambda _: latest)
    monkeypatch.setattr(tools, '_workflow_snapshot', lambda *args, **kwargs: {})
    monkeypatch.setattr(tools, 'infer_explicit_turn_actions', lambda _: {'provider_run'})
    monkeypatch.setattr(tools, '_plan_change_references', lambda *args: f.request.reference_plans)
    monkeypatch.setattr(tools.sources, 'load_registry', lambda: {'sources': []})
    monkeypatch.setattr(tools, 'prepare_user_message', lambda message, **kwargs: SimpleNamespace(provider_message=message))
    monkeypatch.setattr(tools, '_run', lambda *args, **kwargs: captured.update(kwargs) or {'code': 'captured'})
    assert tools._request_replan(context, {'strategy': 'fresh'}) == {'code': 'captured'}
    request = captured['plan_change_request']
    assert request.source_scientific_configuration_sha256 == f.digest
    assert request.target_scientific_configuration_sha256 == study_contexts.scientific_configuration_sha256(current_study)
    assert captured['planner_start_mode'] == 'fresh' and captured['run_intent'] == 'candidate_plan'
    assert 'plan_revision_source_run_id' not in captured
    assert (bind(f, request, study=current_study).baseline_requirements() is None) == change_scope
