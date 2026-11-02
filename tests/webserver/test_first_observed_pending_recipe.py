"""Independent negative checks for review-only pending materialization.

Synthetic metadata/proof stubs are used solely to reach rejection paths. No
approval or scientific-review outcome is mocked and no patient data is read.
"""
import json
from types import SimpleNamespace
import hashlib
from pathlib import Path
import pytest
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.webserver import first_observed_deferred_recipe as owner
from tests.support.descriptive_plan import _context, _plan


def args():
    ctx = _context().model_copy(update={'research_question': '描述成年 ICU 首次入住按性别的院内死亡率及 95% 置信区间。'})
    source = {'database': 'eicu_demo', 'registration_id': 'synthetic', 'archive_sha256': 'a' * 64}
    ctx = ctx.model_copy(update={'cohort': ctx.cohort.model_copy(update={'database': 'eicu_demo',
        'provenance': {'evidence_stage': 'metadata_only_planning', 'patient_rows_read': False}}),
        'user_preferences': ctx.user_preferences.model_copy(update={'data_constraints': json.dumps({'source_identity': source})})})
    spec = PopulationSpec.model_validate({'criteria': [{'id': 'c1', 'kind': 'first_observed_icu_single_hospital_adult',
        'role': 'include', 'source': 'question', 'quote': 'first observed ICU'}]})
    return dict(preparation_run_id='run', study_context_id='study', project_id='project', session_id='session',
                context=ctx, plan=_plan().model_copy(update={'research_question': ctx.research_question}), population_spec=spec, skeleton_sha256='b'*64)


def bind_synthetic_checkpoint(case, tmp_path, monkeypatch):
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
    from tests.research_agent.planning.progressive_planner_fixtures import _payload
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
    from easyicu.webserver import agent_pipeline_runs
    raw = _payload()
    raw['cohort']['population_spec'] = case['population_spec'].model_dump(mode='json')
    data = ProgressivePlanSkeleton.model_validate(raw).model_dump_json().encode()
    (tmp_path / 'progressive_plan_skeleton.json').write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    case['skeleton_sha256'] = sha
    authority = PlanReviewAuthority.create(plan=case['plan'], evidence_sha256={'progressive_plan_skeleton':sha})
    payload = {'plan_review_authority':authority.model_dump(mode='json')}
    request = HumanReviewRequest.create(kind='scientific_stop', summary='synthetic', payload=payload,
                                         authority_sha256=canonical_sha256(payload))
    checkpoint = RawPreparationCheckpoint.pending(plan_sha256=authority.plan_sha256, request=request)
    (tmp_path / 'raw_preparation_checkpoint.json').write_text(checkpoint.model_dump_json())
    monkeypatch.setattr(agent_pipeline_runs, 'raw_preparation_review_owner', lambda *a:(SimpleNamespace(run_dir=str(tmp_path)), {}))


def test_extra_criterion_rejected_before_source_read(monkeypatch, tmp_path):
    case = args()
    criteria = [x.model_dump(mode='json') for x in case['population_spec'].criteria]
    criteria.append({'id':'c2','kind':'diagnosis_codes','role':'exclude','source':'question',
                     'quote':'exclude K92.2','system':'icd10','codes':['K92.2']})
    case['population_spec'] = PopulationSpec.model_validate({'criteria': criteria})
    bind_synthetic_checkpoint(case, tmp_path, monkeypatch)
    monkeypatch.setattr(owner, '_metadata_source', lambda *a: pytest.fail('source read before scope rejection'))
    with pytest.raises(ValueError, match='population_invalid'):
        owner.build_pending_first_observed_review(**case)


@pytest.mark.parametrize('level', [None, 0.90])
def test_unrequested_or_wrong_interval_rejected(monkeypatch, level, tmp_path):
    case = args()
    steps=[]
    for step in case['plan'].steps:
        spec=step.exposure_outcome_distribution_spec
        if spec is not None:
            spec=spec.model_copy(update={'interval_method':'wilson' if level else 'none', 'confidence_level':level})
            step=step.model_copy(update={'exposure_outcome_distribution_spec':spec})
        steps.append(step)
    case['plan']=case['plan'].model_copy(update={'steps':steps})
    bind_synthetic_checkpoint(case, tmp_path, monkeypatch)
    monkeypatch.setattr(owner, '_metadata_source', lambda *a:(Path('/synthetic'), {'export_manifest_sha256':'c'*64}))
    monkeypatch.setattr(owner, 'verify_first_observed_preparation_recipe', lambda **k: object())
    monkeypatch.setattr(owner, 'declared_native_export_inventory', lambda **k: pytest.fail('continued beyond invalid interval'))
    with pytest.raises(ValueError, match='requested_interval_missing'):
        owner.build_pending_first_observed_review(**case)


def test_missing_preparation_proof_cannot_become_a_review(monkeypatch, tmp_path):
    case=args()
    question = 'Describe the synthetic population.'
    case['context'] = case['context'].model_copy(update={'research_question': question})
    case['plan'] = case['plan'].model_copy(update={'research_question': question})
    bind_synthetic_checkpoint(case, tmp_path, monkeypatch)
    monkeypatch.setattr(owner, '_metadata_source', lambda *a:(Path('/synthetic'), {'export_manifest_sha256':'c'*64}))
    def no_proof(**kwargs):
        raise ValueError('synthetic preparation proof unavailable')
    monkeypatch.setattr(owner, 'verify_first_observed_preparation_recipe', no_proof)
    monkeypatch.setattr(owner, 'build_plan_scientific_review', lambda **k: pytest.fail('review after missing proof'))
    with pytest.raises(ValueError, match='proof unavailable'):
        owner.build_pending_first_observed_review(**case)


def test_changed_saved_skeleton_cannot_create_pending_review(monkeypatch, tmp_path):
    case = args()
    bind_synthetic_checkpoint(case, tmp_path, monkeypatch)
    with (tmp_path / 'progressive_plan_skeleton.json').open('ab') as stream:
        stream.write(b' ')
    monkeypatch.setattr(owner, '_metadata_source', lambda *a: pytest.fail('source read after skeleton drift'))
    with pytest.raises(ValueError, match='skeleton_changed'):
        owner.build_pending_first_observed_review(**case)


def test_recipe_evidence_is_pending_and_still_blocks_ordinary_approval():
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.orchestration.progressive_planning import population_approval_findings
    from easyicu.research_agent.planning.population_compile import compile_population, host_first_observed_recipe_scope
    from easyicu.research_agent.research_context.export_selection import SourceIdentity

    case = args()
    source = SourceIdentity.from_record(json.loads(case['context'].user_preferences.data_constraints)['source_identity'])
    criterion = case['population_spec'].criteria[0].model_dump(mode='json')
    proof = {'coordinate_sha256': 'a' * 64, 'preparation_checkpoint_sha256': 'b' * 64,
             'source_identity_sha256': canonical_sha256(source.to_dict()),
             'criterion_sha256': canonical_sha256(criterion), 'export_manifest_sha256': 'c' * 64,
             'preparation_run_id': 'run'}
    with host_first_observed_recipe_scope(lambda *_: proof):
        population = compile_population(case['population_spec'], case['context'])
    assert population.criteria[0].disposition == 'pending_materialization'
    assert not population.criteria[0].applied
    findings = population_approval_findings(population)
    assert len(findings) == 1 and findings[0].detail['approval_allowed'] is False
    assert compile_population(case['population_spec'], case['context']).criteria[0].disposition == 'requires_extraction'


def test_first_observed_adult_is_represented_but_unproven_source_still_blocks():
    from easyicu.research_agent.planning.population_compile import compile_population
    from easyicu.research_agent.planning.population_review import adult_population_accounted_for

    case = args()
    assert adult_population_accounted_for(case['context'], case['plan'].cohort, case['population_spec'])
    population = compile_population(case['population_spec'], case['context'])
    assert population.criteria[0].disposition == 'requires_extraction'
    assert population.blocking


def test_unbound_projection_is_not_verified_or_approvable(tmp_path, monkeypatch):
    case = args()
    bind_synthetic_checkpoint(case, tmp_path, monkeypatch)
    from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
    checkpoint = RawPreparationCheckpoint.model_validate_json((tmp_path / 'raw_preparation_checkpoint.json').read_bytes())
    pending = SimpleNamespace(run_id=tmp_path.name, requests=[checkpoint.request])
    monkeypatch.setattr(owner, '_metadata_source', lambda *_: pytest.fail('unbound projection accessed source'))
    projection = owner.project_pending_first_observed_review(
        run_dir=tmp_path, pending=pending, study_context_id='study')
    assert projection == {'schema_version': 'easyicu.first-observed-review-scope/1', 'verified': False}


def test_host_stage_recipe_scope_is_temporary_and_rechecks_proof(monkeypatch):
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.planning.population_compile import compile_population
    from easyicu.research_agent.research_context.export_selection import SourceIdentity
    from easyicu.webserver.first_observed_run_guard import FirstObservedPreparationProof
    from easyicu.research_agent.intake import export_package

    case = args()
    identity = SourceIdentity.from_record(json.loads(case['context'].user_preferences.data_constraints)['source_identity'])
    spec = case['population_spec']
    stage = owner.FirstObservedDeferredReviewStage('run', 'study', 'project', 'session', 'a' * 64)
    monkeypatch.setattr(owner.FirstObservedDeferredReviewStage, '_skeleton_and_source',
        lambda *_: (None, spec, identity, 'c' * 64))
    calls = []
    def verify(**kwargs):
        calls.append(kwargs)
        return FirstObservedPreparationProof('a' * 64, 'b' * 64, 'run',
            canonical_sha256(identity.to_dict()), canonical_sha256(spec.criteria[0].model_dump(mode='json')), 'c' * 64)
    monkeypatch.setattr(owner, 'verify_first_observed_preparation_recipe', verify)
    monkeypatch.setattr(export_package, 'open_export_package', lambda *_: pytest.fail('research data reader before approval'))
    with stage.review_scope(context=case['context']):
        assert compile_population(spec, case['context']).criteria[0].disposition == 'pending_materialization'
    assert len(calls) >= 2
    assert compile_population(spec, case['context']).criteria[0].disposition == 'requires_extraction'


def test_pending_review_finding_requires_bound_recipe_and_live_proof(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.orchestration.progressive_planning import population_pending_review_findings
    from easyicu.research_agent.planning.population_compile import compile_population, host_first_observed_recipe_scope
    from easyicu.research_agent.research_context.export_selection import SourceIdentity

    case = args()
    context, spec, plan = case['context'], case['population_spec'], case['plan']
    source = SourceIdentity.from_record(json.loads(context.user_preferences.data_constraints)['source_identity'])
    criterion = spec.criteria[0].model_dump(mode='json')
    proof = {'coordinate_sha256':'a'*64, 'preparation_checkpoint_sha256':'b'*64,
        'source_identity_sha256':canonical_sha256(source.to_dict()), 'criterion_sha256':canonical_sha256(criterion),
        'export_manifest_sha256':'c'*64, 'preparation_run_id':'run'}
    evidence = EvidenceStore(tmp_path)
    with host_first_observed_recipe_scope(lambda *_: proof):
        population = compile_population(spec, context)
        assert population_pending_review_findings(population, context=context, plan=plan,
                                                  run_dir=tmp_path, evidence=evidence)[0].severity == 'error'
        recipe = {'criterion': criterion, 'schema_version':'easyicu.first-observed-pending-materialization/1',
            'status':'pending_materialization', 'analysis_authorized':False, 'actual_cohort_exists':False,
            'decoded_scope_enforced':True, 'same_skeleton_compilation_verified':True,
            'plan_sha256':canonical_sha256(plan.model_dump(mode='json')),
            'context_sha256':canonical_sha256(context.model_dump(mode='json')),
            'population_status':{'compilation_sha256':population.sha256()}}
        evidence.register_text(kind='log', description='synthetic recipe', filename='recipe.json',
            text=json.dumps(recipe), evidence_id='first_observed_deferred_recipe')
        finding = population_pending_review_findings(population, context=context, plan=plan,
                                                      run_dir=tmp_path, evidence=evidence)[0]
        assert finding.severity == 'info' and finding.detail['analysis_authorized'] is False
    assert population_pending_review_findings(population, context=context, plan=plan,
                                               run_dir=tmp_path, evidence=evidence)[0].severity == 'error'


def test_unknown_hospital_outcomes_never_become_survivors():
    good = SimpleNamespace(missing_outcome_policy='exclude_from_denominator', denominator_policy='observed_outcome_rows')
    owner._validate_first_observed_outcome_policy([good], {'sex', 'death'})
    owner._validate_first_observed_outcome_policy(
        [SimpleNamespace(missing_outcome_policy='fail_closed', denominator_policy='all_declared_rows')], {'sex', 'death'})
    for policy, denominator in [('structural_absence_is_non_event', 'all_declared_rows'),
                                ('exclude_from_denominator', 'all_declared_rows')]:
        with pytest.raises(ValueError, match='unknown_outcome_policy_invalid'):
            owner._validate_first_observed_outcome_policy(
                [SimpleNamespace(missing_outcome_policy=policy, denominator_policy=denominator)], {'sex', 'death'})


def test_stay_base_never_uses_nullable_outcome_as_static_concept():
    files = [{'time_columns': [], 'declared_columns': [
        {'source_concept': 'sex'}, {'source_concept': 'death'}]}]
    assert owner._first_observed_static_base({'sex', 'death'}, files, 'death') == ['sex']
    assert owner._first_observed_static_base({'death'}, files, 'death') == []
    files[0]['declared_columns'].append({'source_concept': 'age'})
    assert owner._first_observed_static_base({'sex', 'death'}, files, 'death') == ['age', 'sex']
    files[0]['time_columns'] = ['charttime']
    assert owner._first_observed_static_base({'sex', 'death'}, files, 'death', {'sex'}) == ['age', 'sex']


def test_planning_only_identity_is_not_allowed_as_actual_plan_input():
    from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

    context = args()['context']
    identity = ConceptDescriptor(name='patientunitstayid', role=VariableRole.ID, dtype='int64')
    derived = ConceptDescriptor(name='icu_stay_is_readmission', role=VariableRole.OTHER,
                                dtype='float64', source_concept='icu_stay_is_readmission')
    context = context.model_copy(update={'cohort': context.cohort.model_copy(
        update={'id_columns': ['patientunitstayid']}), 'variables': [identity, derived]})
    declared = [{'identity_column': 'stay_id'}]
    def plan_with(value):
        return SimpleNamespace(model_dump=lambda **_: {'steps': [{'inputs': value}]})
    assert set(owner._planning_only_structural_columns(context=context, plan=plan_with([]),
                                                        declared_files=declared)) == {
        'stay_id', 'patientunitstayid', 'icu_stay_is_readmission'}
    for name in ('patientunitstayid', 'icu_stay_is_readmission'):
        with pytest.raises(ValueError):
            owner._planning_only_structural_columns(context=context, plan=plan_with([name]),
                                                     declared_files=declared)
    with pytest.raises(ValueError, match='identity_unbound'):
        owner._planning_only_structural_columns(context=context, plan=plan_with([]),
                                                 declared_files=[{'identity_column': 'other'}])
