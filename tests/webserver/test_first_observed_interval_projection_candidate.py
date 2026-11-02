import json
from tests.research_agent.planning.family_spec_fixtures import _descriptive_context, _request
from tests.support.distribution_policies import missing_policies_for
from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping, validate_family_plan_spec
from easyicu.research_agent.planning.family_spec.descriptive_template import build_descriptive_skeleton
from easyicu.research_agent.planning.progressive_resume import assemble_progressive_skeleton
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.dependence_authority import host_independent_units_scope
from easyicu.research_agent.planning.population_compile import host_first_observed_recipe_scope
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
QUESTION='使用 eICU Demo，纳入年龄至少 18 岁、仅有一次可见住院且首次 ICU 入住能够可靠确定的患者，按性别报告住院死亡人数、结局已知人数、死亡率及其 95% 置信区间。'

def test_original_question_family_design_matches_recompiled_interval(monkeypatch, tmp_path):
    base=_descriptive_context()
    sex=ConceptDescriptor(name='sex',role=VariableRole.DEMOGRAPHIC,dtype='str',source_concept='sex',
        observed_domain={'n_unique':2,'is_binary':False,'levels':['Female','Male']})
    death=next(x for x in base.variables if x.name=='death')
    ident=ConceptDescriptor(name='patientunitstayid',role=VariableRole.ID,dtype='int64')
    identity=SourceIdentity(database='eicu_demo',registration_id='synthetic',archive_sha256='a'*64)
    constraints=json.loads(base.user_preferences.data_constraints);constraints['source_identity']=identity.to_dict(); constraints['cohort']={'age_min':18}
    context=base.model_copy(update={'research_question':QUESTION,'primary_exposure':'sex','variables':[ident,sex,death,next(x for x in base.variables if x.name=='age')],
        'user_preferences':base.user_preferences.model_copy(update={'data_constraints':json.dumps(constraints)}),
        'cohort':base.cohort.model_copy(update={'database':'eicu_demo','id_columns':['patientunitstayid']})})
    request=_request(context,cohort_mode='predicate_filtered')
    assert request.counts_only
    labels=list(dict.fromkeys([*request.required_reader_label_keys,*request.level_label_keys]))
    payload={'schema_version':'easyicu.family_plan_spec/1','family_id':request.family_id,
        'request_sha256':request.request_sha256,'adjustment_set':[],
        'distribution_missing_policies':missing_policies_for(request),'baseline_variables':[],
        'reader_display_labels':[{'key':k,'value':'Synthetic '+k} for k in labels],
        'comparator_applications':[{'citation_key':key,'application':'Synthetic comparator for planning.'}
                                   for key in request.direct_comparator_literature_keys],
        'roster_decision_note':'Synthetic descriptive study.',
        'population':{'population_spec':{'criteria':[{'id':'c1','kind':'first_observed_icu_single_hospital_adult',
        'role':'include','source':'question','quote':'年龄至少 18 岁、仅有一次可见住院且首次 ICU 入住能够可靠确定'}, {'id':'c2','kind':'age_years','role':'include','source':'question','quote':'年龄至少 18 岁','min_years':18}]}}}
    spec=spec_from_mapping(payload)
    validate_family_plan_spec(spec,request)
    draft=build_descriptive_skeleton(request,spec)
    skeleton=assemble_progressive_skeleton(outline=draft.outline,foundation=draft.foundation.foundation,
                                           steps=[x.step for x in draft.materializations])
    selected=skeleton.design_selection.selected
    print('SYNTHETIC_PREPARATION_DESIGN',selected.primary_method)
    def proof(candidate, criterion):
        return {'coordinate_sha256':'a'*64,'preparation_checkpoint_sha256':'b'*64,
            'source_identity_sha256':canonical_sha256(identity.to_dict()),
            'criterion_sha256':canonical_sha256(criterion.model_dump(mode='json')),
            'export_manifest_sha256':'c'*64,'preparation_run_id':'synthetic_run'}
    with host_first_observed_recipe_scope(proof):
        counts_plan,_=compile_progressive_plan(skeleton=skeleton, context=context, allowed_literature_citation_keys=request.allowed_literature_citation_keys)
    with host_first_observed_recipe_scope(proof), host_independent_units_scope(lambda candidate: candidate is context):
        plan,_=compile_progressive_plan(skeleton=skeleton,context=context,
            allowed_literature_citation_keys=request.allowed_literature_citation_keys)
    from easyicu.research_agent.planning.population_spec import PopulationSpec
    from easyicu.research_agent.planning.progressive_compiler import _compiled_population
    with host_first_observed_recipe_scope(proof), host_independent_units_scope(lambda candidate: candidate is context):
        population = _compiled_population(PopulationSpec.model_validate(skeleton.cohort.population_spec), context=context, plan=plan)
    assert [(x.criterion.kind, x.disposition) for x in population.criteria] == [
        ('first_observed_icu_single_hospital_adult', 'pending_materialization'), ('age_years', 'applied_by_plan')]
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.orchestration.progressive_planning import population_pending_review_findings
    spec = PopulationSpec.model_validate(skeleton.cohort.population_spec)
    from easyicu.webserver.first_observed_deferred_recipe import _first_observed_review_criterion
    assert _first_observed_review_criterion(spec).kind == 'first_observed_icu_single_hospital_adult'
    recipe = {'criterion': spec.criteria[0].model_dump(mode='json'), 'population_spec': spec.model_dump(mode='json'),
        'schema_version':'easyicu.first-observed-pending-materialization/1', 'status':'pending_materialization',
        'analysis_authorized':False, 'actual_cohort_exists':False, 'decoded_scope_enforced':True,
        'same_skeleton_compilation_verified':True, 'plan_sha256':canonical_sha256(plan.model_dump(mode='json')),
        'context_sha256':canonical_sha256(context.model_dump(mode='json')),
        'population_status':{'compilation_sha256':population.sha256()}}
    evidence=EvidenceStore(tmp_path)
    evidence.register_text(kind='log',description='synthetic recipe',filename='recipe.json',
        text=json.dumps(recipe),evidence_id='first_observed_deferred_recipe')
    with host_first_observed_recipe_scope(proof):
        findings=population_pending_review_findings(population, context=context, plan=plan,run_dir=tmp_path,evidence=evidence)
    assert findings[0].severity == 'info'
    assert findings[0].detail['population_compile']['criteria'][1]['criterion']['kind'] == 'age_years'
    import pytest
    altered=spec.model_dump(mode='json'); altered['criteria'][1]['min_years']=21
    with pytest.raises(ValueError, match='population_invalid'):
        _first_observed_review_criterion(PopulationSpec.model_validate(altered))
    altered=spec.model_dump(mode='json'); altered['criteria'][1]['max_years']=65
    with pytest.raises(ValueError, match='population_invalid'):
        _first_observed_review_criterion(PopulationSpec.model_validate(altered))
    dist=next(step.exposure_outcome_distribution_spec for step in plan.steps if step.exposure_outcome_distribution_spec)
    print('SYNTHETIC_DEFERRED_INTERVAL',dist.interval_method,dist.confidence_level)
    from easyicu.webserver.first_observed_deferred_recipe import _project_review_interval_text
    from easyicu.research_agent.authority.plan_lifecycle import build_normalized_plan_lineage
    count_step = next(x for x in counts_plan.steps if x.step_id == 'exposure_outcome_distribution')
    assert count_step.exposure_outcome_distribution_spec.interval_method == 'none_counts_only'
    assert _project_review_interval_text(original=counts_plan, plan=counts_plan) == counts_plan
    import pytest
    with pytest.raises(ValueError, match='interval_changed'):
        _project_review_interval_text(original=counts_plan, plan=plan)
    with pytest.raises(ValueError, match='interval_changed'):
        _project_review_interval_text(original=plan, plan=counts_plan)
    no_design = plan.model_copy(update={'design_selection':None}, deep=True)
    introduced_design = no_design.model_copy(update={'design_selection':plan.design_selection}, deep=True)
    with pytest.raises(ValueError, match='design_changed'):
        _project_review_interval_text(original=no_design, plan=introduced_design)
    changed = no_design.model_copy(deep=True)
    next(x for x in changed.steps if x.exposure_outcome_distribution_spec).exposure_outcome_distribution_spec.confidence_level = 0.90
    with pytest.raises(ValueError, match='interval_changed'):
        _project_review_interval_text(original=no_design, plan=changed)
    extra = plan.model_copy(deep=True)
    extra.steps.append(next(x for x in extra.steps if x.exposure_outcome_distribution_spec).model_copy(update={'step_id':'extra_distribution'}, deep=True))
    with pytest.raises(ValueError, match='interval_changed'):
        _project_review_interval_text(original=plan, plan=extra)
    attached = plan.model_copy(deep=True)
    supplied = next(x.exposure_outcome_distribution_spec for x in attached.steps if x.exposure_outcome_distribution_spec)
    next(x for x in attached.steps if x.exposure_outcome_distribution_spec is None).exposure_outcome_distribution_spec = supplied.model_copy(deep=True)
    with pytest.raises(ValueError, match='interval_changed'):
        _project_review_interval_text(original=plan, plan=attached)
    duplicate = plan.model_copy(deep=True)
    duplicate.steps.append(next(x for x in duplicate.steps if x.exposure_outcome_distribution_spec).model_copy(deep=True))
    with pytest.raises(ValueError, match='interval_changed'):
        _project_review_interval_text(original=plan, plan=duplicate)
    original = plan.model_copy(deep=True)
    updated = _project_review_interval_text(original=original, plan=plan)
    assert 'no confidence interval' in original.design_selection.selected.primary_method
    assert '95% wilson confidence intervals' in updated.design_selection.selected.primary_method
    source_step = next(x for x in original.steps if x.step_id == 'exposure_outcome_distribution')
    target_step = next(x for x in updated.steps if x.step_id == 'exposure_outcome_distribution')
    assert 'no confidence interval' in source_step.intent
    assert '95% wilson confidence intervals' in target_step.intent
    lineage = build_normalized_plan_lineage(proposed_plan=original, proposed_source='test',
        pre_normalization_plan=original, normalized_plan=updated,
        resume_scientific_semantics_changed=False, host_scientific_semantics_changed=True)
    assert lineage.proposed.plan_sha256 == canonical_sha256(original.model_dump(mode='json'))
    assert lineage.plan_sha256 == canonical_sha256(updated.model_dump(mode='json'))
    assert lineage.transformation_receipts
    unrelated = updated.model_copy(deep=True)
    unrelated.design_selection = unrelated.design_selection.model_copy(update={'candidates':[x.model_copy(update={'primary_method':'tampered'}) if x.disposition=='selected' else x for x in unrelated.design_selection.candidates]})
    with pytest.raises(ValueError, match='design_changed'):
        _project_review_interval_text(original=updated, plan=unrelated)
    altered = updated.model_copy(deep=True)
    altered.design_selection = altered.design_selection.model_copy(update={'candidates':[x.model_copy(update={'primary_method':'tampered'}) if x.disposition=='selected' else x for x in altered.design_selection.candidates]})
    import pytest
    with pytest.raises(ValueError, match='design_changed'):
        _project_review_interval_text(original=original, plan=altered)
    altered = updated.model_copy(deep=True)
    next(x for x in altered.steps if x.step_id == 'exposure_outcome_distribution').intent = 'tampered'
    with pytest.raises(ValueError, match='interval_step_changed'):
        _project_review_interval_text(original=original, plan=altered)

    from contextlib import contextmanager
    from easyicu.webserver.first_observed_deferred_recipe import FirstObservedDeferredReviewStage
    stage = FirstObservedDeferredReviewStage('synthetic', 'synthetic', 'synthetic', 'synthetic', 'a'*64)
    monkeypatch.setattr(FirstObservedDeferredReviewStage, '_skeleton_and_source', lambda self, ctx: (skeleton, None, None, None))
    @contextmanager
    def rejected_scope(self, *, context):
        raise ValueError('proof_invalid')
        yield
    monkeypatch.setattr(FirstObservedDeferredReviewStage, 'review_scope', rejected_scope)
    with pytest.raises(ValueError, match='proof_invalid'):
        stage.normalize_review_interval_text(context=context, plan=original)
    assert original.design_selection.selected.primary_method == selected.primary_method
    @contextmanager
    def counts_scope(self, *, context):
        with host_first_observed_recipe_scope(proof):
            yield
    monkeypatch.setattr(FirstObservedDeferredReviewStage, 'review_scope', counts_scope)
    assert stage.normalize_review_interval_text(context=context, plan=counts_plan) == counts_plan
