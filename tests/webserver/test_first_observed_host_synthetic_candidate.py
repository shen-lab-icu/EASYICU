"""Synthetic signed preparation using real candidate planning evidence, no Provider."""
import subprocess
import sys
import gzip
import hashlib
import zipfile
import json
import textwrap
from dataclasses import replace
import pandas as pd
import pytest
from tests.support.first_observed_handoff import _environment
from tests.support.first_observed_process import _PROCESS
from tests.support.typed_export import typed_export, metadata_binding
from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.concept.metadata_sidecar import ColumnMetadataSidecar, write_content_addressed_sidecar


def _with_service_cleanup(script):
    """Preserve every test assertion while closing the synthetic durable owner."""
    start = "service=durable_research.DurableResearchService(root/'durable',manager=jobs.MANAGER)"
    end = "\nassert service.close()"
    if script.count(start) != 1 or script.count(end) != 1:
        raise ValueError("synthetic_service_cleanup_boundary_changed")
    prefix, rest = script.split(start, 1)
    body, suffix = rest.split(end, 1)
    return (prefix + start + "\ntry:\n" + textwrap.indent(body.lstrip("\n"), "    ")
            + "\nfinally:\n    import sys\n    _active_error = sys.exc_info()[1]\n"
              "    if not service.close():\n        if _active_error is None:\n"
              "            raise AssertionError('synthetic durable service did not close')\n"
              "        _add_note = getattr(_active_error, 'add_note', None)\n"
              "        if callable(_add_note):\n            _add_note('synthetic durable service did not close')\n"
              "        else:\n            print('synthetic durable service did not close', file=sys.stderr)\n"
            + suffix)


def test_synthetic_durable_service_is_closed_on_assertion_failure():
    from pathlib import Path
    from types import SimpleNamespace

    closed = []
    service = SimpleNamespace(close=lambda: closed.append(True) or True)
    namespace = {"root": Path('/synthetic'), "jobs": SimpleNamespace(MANAGER=object()),
                 "durable_research": SimpleNamespace(DurableResearchService=lambda *args, **kwargs: service)}
    script = "service=durable_research.DurableResearchService(root/'durable',manager=jobs.MANAGER)\nraise AssertionError('original')\nassert service.close()\n"
    with pytest.raises(AssertionError, match='original'):
        exec(_with_service_cleanup(script), namespace)
    assert closed == [True]


@pytest.mark.parametrize('legacy_exception', [False, True])
def test_synthetic_durable_service_cleanup_failure_keeps_original_error(legacy_exception, capsys):
    from pathlib import Path
    from types import SimpleNamespace

    closed = []
    service = SimpleNamespace(close=lambda: closed.append(True) or False)
    namespace = {"root": Path('/synthetic'), "jobs": SimpleNamespace(MANAGER=object()),
                 "durable_research": SimpleNamespace(DurableResearchService=lambda *args, **kwargs: service)}
    class LegacyAssertion(AssertionError):
        add_note = None
    namespace['Failure'] = LegacyAssertion if legacy_exception else AssertionError
    script = "service=durable_research.DurableResearchService(root/'durable',manager=jobs.MANAGER)\nraise Failure('original')\nassert service.close()\n"
    with pytest.raises(AssertionError, match='original') as error:
        exec(_with_service_cleanup(script), namespace)
    assert closed == [True]
    assert str(error.value) == 'original'
    if callable(getattr(error.value, 'add_note', None)):
        assert 'synthetic durable service did not close' in getattr(error.value, '__notes__', ())
    else:
        assert 'synthetic durable service did not close' in capsys.readouterr().err


@pytest.mark.parametrize(('extra_age', 'counts_only'), [(False, False), (True, False), (False, True)])
def test_signed_preparation_binds_real_synthetic_candidate_lineage(tmp_path, extra_age, counts_only):
    rows = pd.DataFrame([(1, 'p1', 11, 1, 'MICU', '65'),
                         (2, 'p2', 22, 1, 'MICU', '70'),
                         (3, 'p2', 23, 1, 'MICU', '70')],
                        columns=['patientunitstayid', 'uniquepid', 'patienthealthsystemstayid',
                                 'unitvisitnumber', 'unittype', 'age'])
    rows['gender'] = ['Female', 'Male', 'Male']
    rows['hospitaldischargestatus'] = ['Alive', 'Expired', 'Expired']
    rows.to_parquet(tmp_path / 'patient.parquet', index=False)
    archive = tmp_path / 'synthetic.zip'
    with zipfile.ZipFile(archive, 'w') as package:
        package.writestr('demo/patient.csv.gz', gzip.compress(rows.to_csv(index=False).encode()))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    patient_sha = hashlib.sha256((tmp_path / 'patient.parquet').read_bytes()).hexdigest()
    labs = pd.DataFrame({'stay_id':[1,1,2,3], 'charttime':[1.,2.,1.,1.],
        'age':[65,65,70,70], 'lact':[1.,2.,3.,4.], 'mech_vent':[False,True,False,False],
        'sex':['Female','Female','Male','Male']})
    export = typed_export(tmp_path / 'synthetic-export', database='eicu_demo', labs=labs,
        outcomes=pd.DataFrame({'stay_id':[1,2,3], 'death':[False,True,True]}))
    manifest_path=export/'_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    sidecar=ColumnMetadataSidecar.from_dict(json.loads((export/manifest['column_metadata']['file']).read_text()))
    lab=next(f for f in sidecar.files if f.module=='labs')
    lab=replace(lab,columns={**lab.columns,'sex':metadata_binding('sex','sex',ConceptColumnRole.VALUE,database='eicu_demo')})
    sidecar=replace(sidecar,files=tuple(lab if f.module=='labs' else f for f in sidecar.files))
    manifest['column_metadata']=write_content_addressed_sidecar(export,sidecar).to_dict()
    manifest['concept_selection']['modules']['labs'].append('sex')
    manifest['files'][0]['concept_ids'].append('sex'); manifest['files'][0]['concepts']+=1
    manifest['files'][0]['column_metadata_columns'].append('sex')
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    start = _PROCESS.index('context = _ctx(')
    end = _PROCESS.index('authority = PlanReviewAuthority.create(plan=plan, evidence_sha256=files)')
    candidate = r'''
from tests.support.first_observed_synthetic_review import positive_case
from tests.support.first_observed_synthetic_literature import KEY
from tests.research_agent.planning.progressive_planner_fixtures import _outline_payload, _foundation_payload, _materialization_payloads
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanOutline
from easyicu.research_agent.planning.progressive_artifacts import persist_progressive_planning_artifacts, persist_progressive_planning_authority
from easyicu.research_agent.planning.progressive_artifacts import persist_progressive_planner_checkpoint
from easyicu.research_agent.authority.plan_lifecycle import build_normalized_plan_lineage, persist_normalized_plan
from easyicu.webserver import agent_pipeline_runs
context, skeleton, _, _, _, literature, _ = positive_case(root)
from easyicu.research_agent.schema import ResearchContext
context=ResearchContext.model_validate_json(context.model_dump_json().replace('exposure_flag','sex').replace('outcome_flag','death').replace('exposure and outcome proportions','hospital death proportions by sex'))
context=context.model_copy(update={'research_question':study['question']})
skeleton=ProgressivePlanSkeleton.model_validate_json(skeleton.model_dump_json().replace('exposure_flag','sex').replace('outcome_flag','death'))
literature.research_question=context.research_question
constraints=json.loads(context.user_preferences.data_constraints)
constraints['source_identity']=identity
context=context.model_copy(update={'user_preferences':context.user_preferences.model_copy(update={
    'data_constraints':json.dumps(constraints)})})
context = context.model_copy(update={'cohort':context.cohort.model_copy(update={
    'n_stays':0,'provenance':{**context.cohort.provenance,'patient_rows_read':False}})})
from easyicu.webserver.research_launch_scientific import _metadata_only_planning_coordinates
planning=_metadata_only_planning_coordinates(question=study['question'],database='eicu_demo',
    export_path=str(root/'synthetic-export'),configured_outcome=study['outcome'],
    stated_exposure=study['primary_exposure'])
acquisition=agent_pipeline_runs._metadata_only_planning_acquisition(database='eicu_demo',
    export_path=root/'synthetic-export',question=context.research_question,target_outcome=planning.get('target_outcome'),
    endpoint=planning.get('endpoint'),
    required_coordinates=('sex','death'),output_dir=wrapper/'pipeline_input',
    llm=ScriptedMockLLMClient([json.dumps({'selected_concepts':['sex','death'],
        'rationale':'Synthetic descriptive outcome by sex.','inclusion_exclusion':[]})]))
assert not acquisition.blocked
raw = skeleton.model_dump(mode='json'); raw['steps'].append(_payload()['steps'][3])
raw['steps'][0]['raw_inputs']=[]
raw['steps'][1]['denominator_policy']='observed_outcome_rows'
raw['steps'][1]['missing_outcome_policy']='exclude_from_denominator'
raw['steps'][-1]['raw_inputs']=['sex','death']
outline = json.loads(json.dumps(_outline_payload(raw)).replace('exposure_flag','sex').replace('outcome_flag','death').replace('age_years','age').replace('sex_code','sex'))
for item in outline['steps']: item['variable_names']=['sex','death']
for design in outline['design_selection']['candidates']: design['figure_role']='Show counts and observed proportions.'
typed_outline=ProgressivePlanOutline.model_validate(outline)
foundation=_foundation_payload(raw); foundation['outline_sha256']=canonical_sha256(typed_outline.model_dump(mode='json'))
steps=_materialization_payloads(raw)
for step, item in zip(typed_outline.steps, steps, strict=True): item['outline_step_sha256']=canonical_sha256(step.model_dump(mode='json'))
agent=ProgressivePlannerAgent(ScriptedMockLLMClient([json.dumps(outline), json.dumps(foundation), *(json.dumps(s) for s in steps)]))
checkpoints=[]
plan=agent.run(context, allowed_literature_citation_keys=[KEY],checkpoint_callback=checkpoints.append); facts=agent.last_result.facts
plan_sha=canonical_sha256(plan.model_dump(mode='json')); evidence=EvidenceStore(run)
def reg(key, filename, value):
    path=run/filename; path.write_text(value,encoding='utf-8')
    evidence.register_file(kind='log',description='synthetic candidate',source_path=path,evidence_id=key,producer='test',generation_mode='system')
reg('research_context','research_context.json',context.model_dump_json())
reg('preplan_literature_bundle','preplan_literature_bundle.json',literature.model_dump_json())
for checkpoint in checkpoints:
    persist_progressive_planner_checkpoint(run_dir=run,evidence=evidence,checkpoint=checkpoint,prompt_pack_version='synthetic')
persist_progressive_planning_artifacts(run_dir=run,evidence=evidence,outline=facts.outline,foundation=facts.foundation,
    materializations=facts.materializations,skeleton=facts.skeleton,compile_receipt=facts.compile_receipt,
    prompt_metrics=facts.prompt_metrics,prompt_pack_version='synthetic')
reg('planner_prompt_metrics','planner_prompt_metrics.json',json.dumps(facts.prompt_metrics))
reg('analysis_plan','analysis_plan.json',plan.model_dump_json())
lineage=build_normalized_plan_lineage(proposed_plan=plan,proposed_source='fresh',pre_normalization_plan=plan,
    normalized_plan=plan,resume_scientific_semantics_changed=False,host_scientific_semantics_changed=False)
lineage_path=persist_normalized_plan(run_dir=run,evidence=evidence,normalized=lineage)
persist_progressive_planning_authority(run_dir=run,evidence=evidence,proposed_plan_sha256=lineage.proposed.plan_sha256,
    normalized_plan_sha256=lineage.plan_sha256,normalized_plan_authority_sha256=lineage.authority_sha256,
    normalized_plan_evidence_id=f'plan_lifecycle_revision_{plan.revision}',normalized_plan_filename=lineage_path.name,
    prompt_pack_version='synthetic')
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
review=build_plan_scientific_review(context=context,plan=plan,literature=literature)
reg('scientific_plan_review','scientific_plan_review.json',review.model_dump_json())
files={r.evidence_id:r.sha256 for r in evidence.verified_records()}
'''
    script = (_PROCESS[:start] + candidate + _PROCESS[end:]).replace('"a" * 64', repr(archive_sha))
    script = script.replace('study = study_contexts.get_context(study_id)', '''study_contexts.upsert_context({
    'id':study_id, 'question':'Among adult patients with one visible hospitalization at their first observed ICU admission, report hospital death counts, known outcome counts, mortality rates and 95% confidence intervals by sex. Adult means age at least 18 years. Report exclusions, uncertain first ICU and missing outcomes, and explain that lifetime first ICU cannot be guaranteed.',
    'data_source':{'path':str(root/'synthetic-export'),'database':'eicu_demo'},
    'outcome':'death','primary_exposure':'sex'})
study = study_contexts.get_context(study_id)''')
    script = script.replace('wrapper = root / "wrapper"', '''from easyicu.webserver.pi_copilot.run_authority import research_pipeline_workspace
project_root=research_pipeline_workspace().project_root(study_id)
wrapper = project_root / study_id / "run_candidate"''')
    ledger_setup = r'''
import subprocess
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.webserver import provider_adapter, research_job_recovery
config=PipelineConfig(workdir=str(wrapper/'pipeline'), no_automatic_retries=True)
ledger_path=wrapper.parent/'original-budget.json'
ledger=ProviderHardStopLedger(path=ledger_path,task_ids=('original-task',),
    limits=provider_adapter.web_research_agent_hard_stop_limits('planner_canary'),
    batch_id='original-task',declaration_sha256=study_sha)
task=ledger.start_task('original-task'); task.pause()
from easyicu.webserver.jobs import Job
tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
def synthetic_runner(job):
    research_job_recovery.bind_provider_recovery(job, wrapper_dir=wrapper, task=task, client=object())
    task.pause()
    return {'gate':{'status':'pending'}}
research_job_recovery.run_controlled_candidate(runner=synthetic_runner,
    request=SimpleNamespace(no_automatic_retries=True,allow_environment_resume=False,study_context=study),
    job=Job('candidate','agent-run'),task=task,original_job_id='original',historical_worker_tree=tree)
seed=agent_review_recovery.WebReviewRecoverySeed.create(wrapper_dir=str(wrapper),study=study,
    scientific_configuration_sha256=study_sha,provider_meta={},provider_public={},credential_source='pi_verified',
    budget_mode='planner_canary',prepared_package_binding=None,pipeline_config=config.recovery_payload(),
    pipeline_config_sha256=config.canonical_digest(),acquisition_projection={},hard_stop_ledger_path=str(ledger_path),
    hard_stop_task_id='original-task',hard_stop_declaration_sha256=study_sha,created_at=time.time())
agent_review_recovery.put_recovery_seed(seed)
'''
    script = script.replace('hc = HumanReviewCheckpoint.create(', ledger_setup + '\nhc = HumanReviewCheckpoint.create(')
    script = script.replace('pipeline_config_sha256="1"*64, environment_identity={}',
                            'pipeline_config_sha256=config.canonical_digest(), environment_identity={}')
    script = script.replace('pipeline_config={}, pipeline_config_sha256="1"*64,',
                            'pipeline_config=config.recovery_payload(), pipeline_config_sha256=config.canonical_digest(),')
    script = script.replace('hard_stop_ledger_path=str(root / "ledger"), hard_stop_task_id="job",',
                            'hard_stop_ledger_path=str(ledger_path), hard_stop_task_id="original-task",')
    script = script.replace('hard_stop_declaration_sha256="5"*64', 'hard_stop_declaration_sha256=study_sha')
    script = script.replace('review_authority_sha256=request.authority_sha256, nonce="f"*64)',
                            'review_authority_sha256=request.authority_sha256, nonce=canonical_sha256({"run_id":run_id, "study_context_id":study_id, "checkpoint_sha256":pending.checkpoint_sha256}))')
    script = script.replace('pending = RawPreparationCheckpoint.pending(plan_sha256=plan_sha, request=request)', '''pending = RawPreparationCheckpoint.pending(plan_sha256=plan_sha, request=request)
from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.webserver.raw_preparation_candidate import _publish_once
with AnchoredDirectory.open(run) as directory:
    _publish_once(directory, 'raw_preparation_checkpoint.json', pending.model_dump_json().encode())
    directory.assert_still_selected()''')
    script = script.replace('binding = RawPreparationHostBinding(project_id=project, session_id=session, study_context_id=study_id, run_id=run_id, nonce=choice.nonce)', '''with AnchoredDirectory.open(run) as directory:
    directory.replace_bytes('raw_preparation_checkpoint.json', approved.model_dump_json().encode())
    directory.assert_still_selected()
binding = RawPreparationHostBinding(project_id=project, session_id=session, study_context_id=study_id, run_id=run_id, nonce=choice.nonce)''')
    script = script.split('scope = producer.proposed_first_observed_scope', 1)[0]
    script += r'''
from easyicu.webserver import durable_research, host_action_jobs, raw_preparation_candidate
from easyicu.webserver.deployment_lease import acquire_single_process_lease
from easyicu.webserver.host_action_contracts import HostDecisionOffers
from easyicu.webserver.routes import agent as route
acquire_single_process_lease(); jobs.MANAGER.enable_persistence(root/'jobs.sqlite3')
service=durable_research.DurableResearchService(root/'durable',manager=jobs.MANAGER)
service.launch(); durable_research._SERVICE=service
candidate=raw_preparation_candidate.create_candidate_for_review(run_id=run_id,study_context_id=study_id)
assert candidate['status']=='pending' and candidate['analysis_authorized'] is False
choice, checkpoint, scope, _=raw_preparation_candidate.available_host_review(run_id=run_id,study_context_id=study_id)
action=HostActionRequest(session_id=session,project_id=project,action_code='review_raw_preparation',decision=choice)
def submit(action, body, callback, **key_inputs):
    return host_action_jobs.submit_with_host_action(action,study_context_id=study_id,submit=callback,
        offered=lambda _: HostDecisionOffers(raw_preparation_review=choice),conversations=copilot,**key_inputs)
route._submit_host_decision=submit
answer=route.raw_preparation_review({'study_context_id':study_id,'decision':'approved','host_action':action.model_dump(mode='json')})
job_id=answer['preparation_job_id']
deadline=time.monotonic()+50
while time.monotonic()<deadline:
    job=jobs.MANAGER.get(job_id); command=service.store.get(job_id)
    if job and job.status!='running' and command and command['status']=='terminal': break
    time.sleep(.05)
else: raise AssertionError('guarded preparation did not terminate')
assert job.status=='done', job.error
assert job.result['preparation_status']=='first_observed_coordinate_complete'
from easyicu.webserver import durable_research_store, research_auto_resume
store=service.store
descriptor=research_auto_resume.prepare_descriptor(study=study,project_root=project_root,credential_source='pi_verified',
    budget_mode='planner_canary',external_llm_opt_in=True)
descriptor['provider_binding']={'wrapper_dir':str(wrapper),'hard_stop_ledger_path':str(ledger_path),
    'hard_stop_task_id':'original-task','hard_stop_declaration_sha256':study_sha}
store.prepare('candidate',descriptor); store.activate('candidate'); store.finish('candidate','completed')
binding=research_job_recovery.prepare_deferred_review_stage(service=SimpleNamespace(store=store),
    preparation_run_id=run_id,study_context_id=study_id,project_id=project,session_id=session)
assert binding.candidate_job_id=='candidate' and binding.task.task_id=='original-task'
from easyicu.webserver import research_run_submission
if os.environ.get('EASYICU_SYNTHETIC_REAL_RUNTIME') != '1':
    from easyicu.webserver import research_pipeline_run_preparation
    from easyicu.research_agent.pipeline import ResearchAgentPipeline
    research_pipeline_run_preparation._require_execution_runtime=lambda **_: 'docker'
    def synthetic_runtime(self, **_):
        self._validated_runtime_capabilities=('numpy','pandas','scipy')
        self._validated_runtime_bundle=None
        return self._validated_runtime_capabilities
    ResearchAgentPipeline._preflight_execution_runtime=synthetic_runtime
research_run_submission.provider_environment_for_agent_run=lambda **_: {}
research_run_submission.agent_runs.resolve_agent_provider_config=lambda **_: {}
no_send=ScriptedMockLLMClient([])
provider_adapter.build_research_agent_provider_client=lambda *args, **kwargs: (no_send,{})
submission=research_job_recovery.submit_deferred_review_stage(preparation_run_id=run_id,
    study_context_id=study_id,project_id=project,session_id=session)
assert submission.job_id
deadline=time.monotonic()+90
while time.monotonic()<deadline:
    stage_job=jobs.MANAGER.get(submission.job_id)
    if stage_job and stage_job.status!='running': break
    time.sleep(.05)
else: raise AssertionError('deferred stage did not reach review')
assert stage_job.status=='done', stage_job.error
assert no_send.calls==[]
assert stage_job.result.get('human_review_pending') is True, stage_job.result
assert stage_job.result.get('pending_reviews'), stage_job.result
wrapper_result=Path(stage_job.result['project_dir'])
manifest_result=json.loads((wrapper_result/'source_run_manifest.json').read_text())
assert manifest_result['plan_approval_allowed'] is True
assert len(manifest_result['pending_reviews'])==1
actual_run=wrapper_result/'pipeline'/stage_job.result['run_id']
recipe=json.loads((actual_run/'first_observed_deferred_recipe.json').read_text())
assert recipe['actual_cohort_exists'] is False and recipe['analysis_authorized'] is False
assert recipe['materialization']['analysis_columns']=={'exposure':'sex','outcome':'death'}
assert recipe['source_flow_report']['unknown_outcome_count_required'] is True
assert recipe['source_flow_report']['lifetime_first_icu_not_guaranteed'] is True
assert recipe['budget']['reset_or_increase_authorized'] is False
plan=json.loads((actual_run/'analysis_plan.json').read_text())
distributions=[step['exposure_outcome_distribution_spec'] for step in plan['steps']
               if step.get('exposure_outcome_distribution_spec')]
assert len(distributions)==1 and distributions[0]['confidence_level']==0.95
assert distributions[0]['denominator_policy']=='observed_outcome_rows'
assert distributions[0]['missing_outcome_policy']=='exclude_from_denominator'
(root/'fixture_coordinates.json').write_text(json.dumps({'home':str(root/'home'),'project_id':project,
    'session_id':session,'study_id':study_id,'stage_job_id':submission.job_id,
    'run_id':stage_job.result['run_id'],'wrapper':stage_job.result['project_dir']}))
assert service.close()
'''
    env = _environment(tmp_path)
    site = tmp_path / 'sitecustomize.py'
    site.write_text(site.read_text().replace('"a" * 64', repr(archive_sha)).replace('"b" * 64', repr(manifest_sha)))
    with (tmp_path / 'sitecustomize.py').open('a') as stream:
        stream.write(f'''
    # Synthetic registration replaces only the official registry identity.
    # ZIP and Parquet are still hashed and their six columns streamed by the real owner.
    from easyicu.webserver import eicu_first_observed_source as real_source
    synthetic = SimpleNamespace(database="eicu_demo", archive_sha256={archive_sha!r}, max_uncompressed_bytes=1000000)
    raw_binding = SimpleNamespace(family="eicu", database="eicu_demo", table_paths={{"patient":root / "patient.parquet"}},
        table_sha256={{"patient":{patient_sha!r}}}, export_manifest_sha256={manifest_sha!r})
    owner.resolve_raw_source_identity_binding = lambda **_: raw_binding
    owner.verified_public_demo_raw = lambda _: synthetic
    real_source.resolve_raw_source_identity_binding = lambda **_: raw_binding
    real_source.verified_public_demo_raw = lambda _: synthetic
    real_source.source_paths = lambda _: SimpleNamespace(archive=root / "synthetic.zip")
    owner.sources.registered_source_identity = lambda path: identity if Path(path) == root / 'synthetic-export' else None
    def checked_derive(path):
        with (root / "source-reads.txt").open("a") as handle: handle.write("read\\n")
        return real_source.derive_from_verified_eicu_demo(path)
    owner.derive_from_verified_eicu_demo = checked_derive
''')
    if counts_only:
        script = script.replace('mortality rates and 95% confidence intervals by sex', 'mortality rates by sex')
        script = script.replace("'outcome':'death','primary_exposure':'sex'})", "'outcome':'death','primary_exposure':'sex','analysis_design':{'analysis_family':'descriptive_epidemiology','analysis_unit':'icu_stay','variance_estimator':'none_counts_only'}})")
        script = script.replace("context=context.model_copy(update={'research_question':study['question']})", "context=context.model_copy(update={'research_question':study['question'], 'user_preferences':context.user_preferences.model_copy(update={'data_constraints':json.dumps({**json.loads(context.user_preferences.data_constraints), 'analysis_design':study['analysis_design']})})})")
        script = script.replace("raw = skeleton.model_dump(mode='json'); raw['steps'].append(_payload()['steps'][3])", "raw = skeleton.model_dump(mode='json'); raw['steps'].append(_payload()['steps'][3]); raw['steps'][1]['confidence_level']=None")
        script = script.replace("assert len(distributions)==1 and distributions[0]['confidence_level']==0.95", "assert len(distributions)==1 and distributions[0]['confidence_level'] is None and distributions[0]['interval_method']=='none_counts_only'")
    if extra_age:
        drift_hook = r"""
from easyicu.webserver import first_observed_deferred_recipe as _recipe_owner
from easyicu.research_agent.authority.plan_lifecycle import build_normalized_plan_lineage as _make_lineage
from easyicu.research_agent.schema import AnalysisPlan as _AnalysisPlan
from easyicu.research_agent.authority.plan_lifecycle import cohort_concept_id_scope as _cohort_scope
_original_prepare = _recipe_owner.FirstObservedDeferredReviewStage.prepare_review
_original_requested = _recipe_owner.requested_uncertainty
def _check_drift(self, *, context, agent_context, normalized_lineage, run_dir, evidence):
    with _cohort_scope(normalized_lineage.proposed.cohort_concept_ids):
        proposed = _AnalysisPlan.model_validate(normalized_lineage.proposed.plan_payload)
    altered = normalized_lineage.analysis_plan()
    dist = next(step.exposure_outcome_distribution_spec for step in altered.steps if step.exposure_outcome_distribution_spec)
    dist.confidence_level = 0.90
    forged = _make_lineage(proposed_plan=proposed, proposed_source=normalized_lineage.proposed.source,
        pre_normalization_plan=proposed, normalized_plan=altered, resume_scientific_semantics_changed=False,
        host_scientific_semantics_changed=True, cohort_concept_ids=normalized_lineage.proposed.cohort_concept_ids)
    _recipe_owner.requested_uncertainty = lambda _: SimpleNamespace(unresolved=False, requested=False, confidence_levels=[])
    try:
        _, spec, _, _ = self._skeleton_and_source(context)
        _recipe_owner.build_pending_first_observed_review(preparation_run_id=self.preparation_run_id,
            study_context_id=self.study_context_id, project_id=self.project_id, session_id=self.session_id,
            context=context, plan=forged.analysis_plan(), population_spec=spec,
            skeleton_sha256=self.skeleton_sha256, normalized_lineage=forged, planning_context=agent_context)
    except ValueError as exc:
        assert 'first_observed_recipe_interval_changed' in str(exc), str(exc)
    else:
        raise AssertionError('valid lineage with altered typed CI accepted')
    finally:
        _recipe_owner.requested_uncertainty = _original_requested
    return _original_prepare(self, context=context, agent_context=agent_context, normalized_lineage=normalized_lineage, run_dir=run_dir, evidence=evidence)
_recipe_owner.FirstObservedDeferredReviewStage.prepare_review = _check_drift
"""
        marker_hook = 'submission=research_job_recovery.submit_deferred_review_stage('
        assert marker_hook in script
        script = script.replace(marker_hook, drift_hook + "\n" + marker_hook)
        marker = "raw = skeleton.model_dump(mode='json'); raw['steps'].append(_payload()['steps'][3])"
        assert marker in script
        script = script.replace(marker, marker + "\nraw['cohort']['population_spec']['criteria'].append({'id':'c2','kind':'age_years','role':'include','source':'question','quote':'age at least 18','min_years':18})")
        script = script.replace("assert recipe['actual_cohort_exists'] is False", "assert len(recipe['population_spec']['criteria']) == 2\nassert recipe['population_spec']['criteria'][1]['min_years'] == 18\nassert recipe['actual_cohort_exists'] is False")
    script = _with_service_cleanup(script)
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stderr
