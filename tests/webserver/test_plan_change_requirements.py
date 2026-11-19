"""Fresh amendments preserve verified requirements, never old input/approval."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.planning.baseline_requirements import (
    baseline_outline_coverage, baseline_requirement_coverage, bind_baseline_requirements,
)
from easyicu.research_agent.planning.population_requirements import (
    bind_population_requirements, candidate_population_requirements, validate_population_choice,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.typed import RESEARCH_CONTEXT_V3_SCHEMA_VERSION
from easyicu.research_agent.schema import ConceptDescriptor
from easyicu.webserver import agent_pipeline_runs, plan_change_requirements as owner, study_contexts
from easyicu.webserver.plan_change_request import PlanChangeRequest, ReferencedPlan, reference_plan_content
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from tests.research_agent.planning.test_baseline_requirements import _context, _plan
from tests.support.typed_export import typed_export


@pytest.fixture
def source(tmp_path, monkeypatch):
    study = {"id": "study-synthetic", "question": "Describe a biomarker and hospital death.", "data_source": {"database": "miiv"}}
    digest = study_contexts.scientific_configuration_sha256(study)
    wrapper = tmp_path / study["id"] / 'run-fixture'
    run_id = 'run-source'
    checkpoint_path = wrapper / 'pipeline' / run_id / 'human_review_checkpoint.json'
    checkpoint_path.parent.mkdir(parents=True)

    def build(names=("age", "severity_first"), inherited=None, context=None, group="exposure"):
        if context is None:
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
        plan = _plan(*names, group=group)
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
        # The host compile marks extraction complete (plan_decisions).
        'confirmations': {'agent_plan_configuration_compiled': True, 'extraction_completed': True},
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
    # B10: the launch records the compiled candidate this plan continues.
    assert execution.compiled_configuration_source_run_id == f.request.source_run_id
    assert execution.plan_change_request.baseline_requirements() == bind(f).baseline_requirements()
    assert execution.development_resume_binding is None
    assert execution.plan_revision_source_run_id == execution.execution_resume_source_run_id == ''
    assert request.plan_change_request is None  # No new client field or execution grant.


@pytest.mark.parametrize('drift', ['missing', 'corrupt', 'revision', 'older_revision', 'question', 'source', 'user_choice', 'run'])
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
        # A write outside the reconfirmation allowance (here an unscientific
        # title) keeps the digest and still ends the continuity (S1471).
        changed = {**changed, 'revision': 3, 'title': 'Renamed project'}
    elif drift == 'older_revision':
        changed = {**changed, 'revision': 1}
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


@pytest.mark.parametrize('variant', ['reconfirmed', 'unsigned', 'second_write', 'signed_after_another_write', 'scientific_edit', 'extraction_kept'])
def test_runtime_continuity_survives_only_one_signed_reconfirmation(source, monkeypatch, tmp_path, variant):
    # B10 (UX leg 3, 2026-11-19): the raw official-Demo reconfirmation signs
    # its choice one revision later and marks extraction incomplete. The fresh
    # plan stayed a revision of the compiled candidate only at the exact
    # revision; losing it planned from scratch and looped. Exactly that one
    # signed write is allowed (S1471); anything else falls back to the exact match.
    from easyicu.webserver.pi_copilot import demo_planning_authority

    f = source()
    changed = compiled_study(f, monkeypatch, tmp_path)
    reconfirmed = {
        **changed, 'revision': changed['revision'] + 1, 'demo_planning_authority_ref': 'c' * 64,
        'confirmations': {**changed['confirmations'], 'extraction_completed': False},
    }
    signed = reconfirmed['revision']
    if variant == 'unsigned':
        signed = None
    elif variant == 'second_write':
        reconfirmed['revision'] += 1
    elif variant == 'signed_after_another_write':
        # Another write (a title) came between the compile and the signed choice.
        reconfirmed['revision'] += 1
        reconfirmed['title'] = 'Renamed project'
        signed = reconfirmed['revision']
    elif variant == 'scientific_edit':
        reconfirmed['question'] = 'New scientific scope.'
    elif variant == 'extraction_kept':
        reconfirmed['confirmations']['extraction_completed'] = True
    monkeypatch.setattr(demo_planning_authority, 'signed_choice_revision', lambda study: signed)
    bound = owner.compiled_configuration_plan_change(study=reconfirmed, project_root=f.root)
    if variant == 'reconfirmed':
        assert bound is not None and bound.source_run_id == f.request.source_run_id
        assert bind(f, bound, study=reconfirmed).baseline_requirements() == bind(f).baseline_requirements()
    else:
        assert bound is None


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


def test_a_source_that_prepared_its_data_binds_by_its_context_version(source, tmp_path):
    """Preparing data seals a typed context; a fresh plan still inherits its requirements."""
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized", data_path=typed_export(tmp_path / "export"), database="miiv",
        static_concepts=("age",), feature_concepts=("lact",), outcome_concepts=("death",),
    )
    context = build_research_context(
        research_question="Describe age by first lactate and hospital death.", cohort=paths["parquet"],
        cohort_name="prepared", database="miiv", target_outcome="death", primary_exposure="lact_first",
        id_columns=("stay_id",), outcome_columns=("death",),
    )
    f = source(names=("age",), context=context, group="lact_first")

    bound = bind(f).source_requirements

    assert context.schema_version == RESEARCH_CONTEXT_V3_SCHEMA_VERSION
    assert bound.source_context_sha256 == canonical_sha256(context.model_dump(mode="json"))
    [table] = bound.baseline.tables
    assert table.group_by.name == "lact_first"
    assert [variable.name for variable in table.variables] == ["age"]


def test_a_grouped_column_requires_the_measurement_its_groups_are_formed_from(source):
    """A grouped column is formed again in the next run; its measured concept is what the source must hold.

    Hosted pilot: the lactate-group plan's fresh plan stopped at once with
    plan_change_required_concepts_unavailable, requiring ``lact_group_x1``
    from the planning catalog.
    """
    grouped = ConceptDescriptor(
        name="lact_group_x1", source_concept="lact_group_x1", derived_from_concepts=["lact"],
        unit_normalization="exposure_group_ordinal", role="other", dtype="int64",
        is_ordinal=True, ordinal_levels=[1, 2], valid_range=(1.0, 2.0), unit="category",
    )
    context = _context().model_copy(update={"variables": [*_context().variables, grouped]})
    f = source(names=("age",), context=context, group="lact_group_x1")

    bound = bind(f).source_requirements

    assert "lact" in bound.planning_concepts
    assert "lact_group_x1" not in bound.planning_concepts
    [table] = bound.baseline.tables
    assert table.group_by.name == "lact_group_x1"


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
    context = SimpleNamespace(session=SimpleNamespace(binding=object()), user_message=f.request.user_message,
                              expected_replan_source_run_id='')
    captured = {}
    monkeypatch.setattr(tools, '_bound_context', lambda _: current_study)
    monkeypatch.setattr(tools, '_select_run', lambda _: latest)
    monkeypatch.setattr(tools, '_workflow_snapshot', lambda *args, **kwargs: {})
    monkeypatch.setattr(tools, 'infer_explicit_turn_actions', lambda _: {'provider_run'})
    monkeypatch.setattr(tools, '_plan_change_references', lambda *args: f.request.reference_plans)
    monkeypatch.setattr(tools.sources, 'load_registry', lambda **_kwargs: {'sources': []})
    monkeypatch.setattr(tools, 'prepare_user_message', lambda message, **kwargs: SimpleNamespace(provider_message=message))
    monkeypatch.setattr(tools, '_run', lambda *args, **kwargs: captured.update(kwargs) or {'code': 'captured'})
    assert tools._request_replan(context, {'strategy': 'fresh'}) == {'code': 'captured'}
    request = captured['plan_change_request']
    assert request.source_scientific_configuration_sha256 == f.digest
    assert request.target_scientific_configuration_sha256 == study_contexts.scientific_configuration_sha256(current_study)
    assert captured['planner_start_mode'] == 'fresh' and captured['run_intent'] == 'candidate_plan'
    assert 'plan_revision_source_run_id' not in captured
    assert (bind(f, request, study=current_study).baseline_requirements() is None) == change_scope


def _prepare(monkeypatch, f, study, **request_fields):
    from easyicu.webserver import research_pipeline_run_preparation as preparation
    from tests.webserver.test_research_pipeline_run_preparation import _request, _scientific

    request = replace(_request(), study_context=study, project_root=f.root, **request_fields)
    scientific = replace(_scientific(), study=study)
    monkeypatch.setattr(preparation.capability_policy, 'capability_settings', lambda: {})
    monkeypatch.setattr(preparation, 'ExtensionRegistry', lambda: SimpleNamespace(
        snapshot=lambda: SimpleNamespace(revision=0, skills=(), mcp_servers=()), pipeline_activation=lambda snapshot: None))
    monkeypatch.setattr(preparation, '_require_profile_dictionaries', lambda **kwargs: None)
    monkeypatch.setattr(preparation, '_require_execution_runtime', lambda **kwargs: None)
    provider = SimpleNamespace(provider={}, provider_environment={}, credential_source='fixture', literature_search_authorized=False)
    return preparation._prepare_launch_execution(request, scientific, provider)[1]


def test_the_launch_records_the_compiled_candidate_without_a_change_request(source, monkeypatch, tmp_path):
    # Carbon b10r2 F2: the record follows the compiled lineage, not whether
    # the compiled handoff could build a change request.
    f = source()
    changed = compiled_study(f, monkeypatch, tmp_path)
    monkeypatch.setattr(owner, 'compiled_configuration_plan_change', lambda **_: None)
    execution = _prepare(monkeypatch, f, changed)
    assert execution.plan_change_request is None
    assert execution.compiled_configuration_source_run_id == f.request.source_run_id


def test_a_researchers_plan_change_records_no_compiled_candidate(source, monkeypatch, tmp_path):
    # S1466: a plan change the researcher asked for is a user action; that
    # plan may be compiled once more.
    f = source()
    changed = compiled_study(f, monkeypatch, tmp_path)
    request = f.request.model_copy(update={
        'source_scientific_configuration_sha256': f.digest,
        'target_scientific_configuration_sha256': study_contexts.scientific_configuration_sha256(changed),
    })
    execution = _prepare(monkeypatch, f, changed, plan_change_request=request)
    assert execution.plan_change_request is not None
    assert execution.compiled_configuration_source_run_id == ''


def _continued(f, monkeypatch, changed, recorded):
    # The fresh plan that continued the compiled candidate, after its job's
    # two lifecycle writes (the configuration digest is unchanged).
    reviewed = {**changed, 'revision': changed['revision'] + 3}
    fresh = {'run_id': 'run-fresh', 'project_dir': str(f.row['project_dir']) + '-fresh',
             'scientific_configuration_sha256': study_contexts.scientific_configuration_sha256(reviewed)}
    payloads = {fresh['project_dir']: {'artifact_payloads': {'run_context.json': {
        'run_id': 'run-fresh', **({'compiled_configuration_source_run_id': recorded} if recorded else {})}}},
                f.row['project_dir']: {'artifact_payloads': f.record.artifact_payloads}}
    monkeypatch.setattr(owner.agent_runs, 'list_run_history', lambda **kwargs: {'runs': [fresh, f.row]})
    monkeypatch.setattr(owner.agent_runs, 'read_run_review', lambda project_dir: payloads[project_dir])
    return reviewed, fresh


@pytest.mark.parametrize('recorded,carried', [('run-source', True), ('', False), ('another-run', False)])
def test_the_planners_revision_of_a_continuing_plan_keeps_its_record(source, monkeypatch, tmp_path, recorded, carried):
    # Carbon b10r2 F1: the planner's own revision of that plan, on the same
    # configuration, continues the same compiled candidate.
    f = source()
    changed = compiled_study(f, monkeypatch, tmp_path)
    recorded = f.request.source_run_id if recorded == 'run-source' else recorded
    reviewed, fresh = _continued(f, monkeypatch, changed, recorded)
    expected = f.request.source_run_id if carried else ''
    assert owner.compiled_source_run_id_for_launch(
        study=reviewed, project_root=f.root, revision_source_run_id=fresh['run_id'],
    ) == expected
    execution = _prepare(monkeypatch, f, reviewed, plan_revision_source_run_id=fresh['run_id'])
    assert execution.compiled_configuration_source_run_id == expected
    # A changed configuration ends it whatever the record says.
    edited = {**reviewed, 'covariates': ['age', 'sex']}
    assert owner.compiled_source_run_id_for_launch(
        study=edited, project_root=f.root, revision_source_run_id=fresh['run_id'],
    ) == ''
    # Without naming that plan as its source, a launch carries nothing.
    assert owner.compiled_source_run_id_for_launch(study=reviewed, project_root=f.root) == ''


def test_a_compile_after_another_answer_on_the_same_review_still_records(source, monkeypatch, tmp_path):
    # Carbon b10r2 F2(a): the receipt carries the researcher's earlier answer
    # too, so the compiled handoff builds no change request; the launch still
    # records the compiled candidate it continues.
    from easyicu.webserver.pi_copilot import plan_review_progress

    f = source()
    monkeypatch.setenv('EASYICU_HOME', str(tmp_path / 'host-state'))
    start = {**f.study, 'revision': 1}
    answered = {**start, 'revision': 2, 'covariates': ['age']}
    plan_review_progress.record_choice(before=start, after=answered, run=f.row,
                                       decision_code='plan_adjustment_set', option_id='confirm')
    compiled = {**answered, 'revision': 3, 'covariate_selection': 'exact', 'covariate_authority': 'agent_plan',
                'confirmations': {'agent_plan_configuration_compiled': True, 'extraction_completed': True}}
    plan_review_progress.record_choice(before=answered, after=compiled, run=f.row,
                                       decision_code='agent_plan_configuration', option_id='typed_runtime_projection')
    assert set(plan_review_progress.matching_progress(compiled, f.row).choices) == {
        'plan_adjustment_set', 'agent_plan_configuration'}
    execution = _prepare(monkeypatch, f, compiled)
    assert execution.plan_change_request is None
    assert execution.compiled_configuration_source_run_id == f.request.source_run_id


@pytest.mark.parametrize('drift', ['study_edited', 'receipt_run_replanned'])
def test_a_compile_receipt_for_another_configuration_records_nothing(source, monkeypatch, tmp_path, drift):
    # S1488 (2): rule (a) records only when the receipt's configuration is
    # this launch's: a scientific edit since the compile, or a history row
    # for the receipt's run made on another configuration, records nothing.
    f = source()
    changed = compiled_study(f, monkeypatch, tmp_path)
    assert owner.compiled_source_run_id_for_launch(study=changed, project_root=f.root) == f.request.source_run_id
    if drift == 'study_edited':
        study = {**changed, 'covariates': ['age', 'sex']}
    else:
        study = changed
        other = {**f.row, 'scientific_configuration_sha256': 'e' * 64}
        monkeypatch.setattr(owner.agent_runs, 'list_run_history', lambda **kwargs: {'runs': [other]})
    assert owner.compiled_source_run_id_for_launch(study=study, project_root=f.root) == ''
    assert _prepare(monkeypatch, f, study).compiled_configuration_source_run_id == ''


def test_a_revision_of_a_plan_made_on_another_configuration_carries_nothing(source, monkeypatch, tmp_path):
    # S1488 (3): the source plan's record names the compiled candidate, but
    # that plan was made on another configuration; the revision is compiled
    # once instead.
    f = source()
    changed = compiled_study(f, monkeypatch, tmp_path)
    reviewed, fresh = _continued(f, monkeypatch, changed, f.request.source_run_id)
    fresh['scientific_configuration_sha256'] = 'e' * 64
    assert owner.compiled_source_run_id_for_launch(
        study=reviewed, project_root=f.root, revision_source_run_id=fresh['run_id']) == ''
    assert _prepare(monkeypatch, f, reviewed, plan_revision_source_run_id=fresh['run_id']).compiled_configuration_source_run_id == ''
