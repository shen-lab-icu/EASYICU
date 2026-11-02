"""First preparation travels through the real host route and guardian.

The patient source and the offered workflow decision are test stubs.  This
does not approve or execute research, or demonstrate the real Demo journey.
"""
import subprocess
import sys
import signal
import pytest

from tests.support.first_observed_handoff import _environment
from tests.support.first_observed_process import _PROCESS


@pytest.mark.parametrize("execution_bound", [False, True])
def test_first_review_starts_same_guarded_preparation_without_analysis(tmp_path, execution_bound):
    script = _PROCESS.split('scope = producer.proposed_first_observed_scope', 1)[0] + r'''
from easyicu.webserver import durable_research, host_action_jobs, raw_preparation_candidate
from easyicu.webserver.deployment_lease import acquire_single_process_lease
from easyicu.webserver.host_action_contracts import HostDecisionOffers
from easyicu.webserver.routes import agent as route
acquire_single_process_lease()
jobs.MANAGER.enable_persistence(root / 'jobs.sqlite3')
service=durable_research.DurableResearchService(root/'durable',manager=jobs.MANAGER)
service.launch(); durable_research._SERVICE=service
candidate=raw_preparation_candidate.create_candidate_for_review(run_id=run_id,study_context_id=study_id)
assert candidate['status']=='pending' and candidate['analysis_authorized'] is False
assert not (root/'source-reads.txt').exists()
from easyicu.webserver.raw_preparation_projection import pending_preparation_projection
preview=pending_preparation_projection({'run_id':run_id,'project_dir':str(run)},study_context_id=study_id)
assert preview['scope']['plan_sha256']==plan_sha
assert preview['scope']['read_scope_version']==2 and 'patient_table_sha256' not in preview['scope']
assert 'candidate_preparation_preview' not in preview and not preview['approvable']
assert not (root/'source-reads.txt').exists()
original_context=(run/'research_context.json').read_bytes()
(run/'research_context.json').write_bytes(original_context+b' ')
changed=pending_preparation_projection({'run_id':run_id,'project_dir':str(run)},study_context_id=study_id)
assert 'candidate_preparation_preview' not in changed
(run/'research_context.json').write_bytes(original_context)
get_current_study=study_contexts.get_context
study_contexts.get_context=lambda _: {**study, 'question': 'different study question'}
changed_study=pending_preparation_projection({'run_id':run_id,'project_dir':str(run)},study_context_id=study_id)
assert 'candidate_preparation_preview' not in changed_study
study_contexts.get_context=get_current_study
assert not (root/'source-reads.txt').exists()
choice, checkpoint, scope, _=raw_preparation_candidate.available_host_review(run_id=run_id,study_context_id=study_id)
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver.first_observed_run_guard import resolve_host_prepared_first_observed_coordinate
host_args = dict(preparation_run_id=run_id, study_context_id=study_id, project_id=project, session_id=session,
    source_identity=SourceIdentity.from_record(identity), criterion=scope['criterion'],
    export_manifest_sha256=scope['export_manifest_sha256'])
try:
    resolve_host_prepared_first_observed_coordinate(**host_args)
except (ValueError, OSError): pass
else: raise AssertionError('unapproved preparation returned a coordinate')
action=HostActionRequest(session_id=session,project_id=project,action_code='review_raw_preparation',decision=choice)
def submit(action, body, callback, **key_inputs):
    return host_action_jobs.submit_with_host_action(action,study_context_id=study_id,submit=callback,
        offered=lambda _: HostDecisionOffers(raw_preparation_review=choice),conversations=copilot,**key_inputs)
route._submit_host_decision=submit
answer=route.raw_preparation_review({'study_context_id':study_id,'decision':'approved',
                                      'host_action':action.model_dump(mode='json')})
job_id=answer['preparation_job_id']
assert answer['preparation_status']=='preparing' and not answer['analysis_authorized']
deadline=time.monotonic()+50
while time.monotonic()<deadline:
    job=jobs.MANAGER.get(job_id); command=service.store.get(job_id)
    if job and job.status!='running' and command and command['status']=='terminal': break
    time.sleep(.05)
else: raise AssertionError('guarded preparation did not terminate')
assert job.status=='done', job.error
assert job.result['preparation_status']=='first_observed_coordinate_complete'
assert not job.result['analysis_authorized']
coordinate_binding = resolve_host_prepared_first_observed_coordinate(**host_args)
assert coordinate_binding.preparation_run_id == run_id
assert coordinate_binding.export_manifest_sha256 == scope['export_manifest_sha256']
for change in ({'project_id': 'wrong-project'}, {'session_id': 'wrong-session'},
               {'study_context_id': 'wrong-study'}, {'export_manifest_sha256': 'd'*64},
               {'source_identity': SourceIdentity(database='eicu_demo', registration_id='other', archive_sha256='a'*64)},
               {'criterion': {**scope['criterion'], 'quote': 'different population'}}):
    try:
        resolve_host_prepared_first_observed_coordinate(**{**host_args, **change})
    except (ValueError, OSError): pass
    else: raise AssertionError('mismatched host binding returned a coordinate: '+str(change))
assert (root/'source-reads.txt').read_text().splitlines()==['read']
from easyicu.webserver.raw_preparation_projection import pending_preparation_projection
projection=pending_preparation_projection({'run_id':run_id,'project_dir':str(run)},study_context_id=study_id)
assert projection['status']=='first_observed_complete' and projection['analysis_authorized'] is False
if 'candidate_preparation_preview' in projection:
    assert projection['candidate_preparation_preview']['complete'] is False
    assert projection['candidate_preparation_preview']['preparation_authorized'] is False
coordinate=root/'home'/'.easyicu'/'private-first-observed'/(choice.nonce+'.parquet')
saved=coordinate.read_bytes(); coordinate.write_bytes(saved+b'changed')
assert pending_preparation_projection({'run_id':run_id,'project_dir':str(run)},study_context_id=study_id)['status']=='first_observed_unavailable'
try:
    resolve_host_prepared_first_observed_coordinate(**host_args)
except (ValueError, OSError, RuntimeError): pass
else: raise AssertionError('modified coordinate returned by host resolver')
coordinate.write_bytes(saved)
assert resolve_host_prepared_first_observed_coordinate(**host_args).sha256 == coordinate_binding.sha256
assert (root/'source-reads.txt').read_text().splitlines()==['read']
assert service.close()
'''
    if execution_bound:
        script = script.replace('spec = PopulationSpec.model_validate',
            "context = context.model_copy(update={'cohort': context.cohort.model_copy(update={'database': 'eicu_demo', 'provenance': {'source_identity': identity}})})\nspec = PopulationSpec.model_validate", 1)
        script = script.replace(
            'authority = PlanReviewAuthority.create(plan=plan, evidence_sha256=files)',
            """from easyicu.research_agent.authority.plan_review import ReviewExecutionAuthority
from easyicu.research_agent.authority.run_input import RunInputCapsule
scientific_identity = {'question': context.research_question, 'database': context.cohort.database,
    'user_preferences': {'data_constraints': context.user_preferences.data_constraints}}
capsule = RunInputCapsule(scientific_identity=scientific_identity, scientific_identity_sha256=canonical_sha256(scientific_identity),
    context_sha256=files['research_context'], cohort_sha256='f'*64, initial_environment={}).model_dump_json().encode()
(run / 'run_input_capsule.json').write_bytes(capsule)
EvidenceStore(run).register_text(kind='log', description='capsule', filename='capsule.json',
    text=capsule.decode(), evidence_id='run_input_capsule')
capsule_sha = hashlib.sha256(capsule).hexdigest()
execution = ReviewExecutionAuthority(pipeline_config_sha256='1'*64, capability_activation_sha256='4'*64,
    run_input_capsule_sha256=capsule_sha)
authority = PlanReviewAuthority.create(plan=plan, evidence_sha256=files, execution=execution)""",
        ).replace('original_payload = {"plan_review_authority": authority.model_dump(mode="json")}',
                  'original_payload = {"plan_review_authority": authority.model_dump(mode="json"), "approval_allowed": False}')
        script = script.replace('run_input_capsule_sha256="3"*64', 'run_input_capsule_sha256=capsule_sha')
        script = script.replace('hc = HumanReviewCheckpoint.create(', """second_payload = {'plan_review_authority': authority.model_dump(mode='json'),
    'approval_allowed': True, 'reason': 'another independent review finding'}
second_request = HumanReviewRequest.create(kind='scientific_stop', summary='Another finding',
    payload=second_payload, authority_sha256=canonical_sha256(second_payload))
hc = HumanReviewCheckpoint.create(""", 1)
        script = script.replace('requests=(original_request,), plan_handoff=',
                                'requests=(original_request, second_request), plan_handoff=', 1)
    env = _environment(tmp_path)
    with (tmp_path / 'sitecustomize.py').open('a') as stream:
        stream.write('''\nreal_binding = owner.resolve_raw_source_identity_binding
def checked_binding(**kwargs):
    decisions = root / "home" / ".easyicu" / "private-preparation-decisions"
    assert decisions.exists() and any(decisions.glob("*.used")), "patient bytes read before consent consumption"
    return real_binding(**kwargs)
owner.resolve_raw_source_identity_binding = checked_binding
''')
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr


def test_user_can_resume_approved_handoff_after_real_restart(tmp_path):
    prefix = _PROCESS.split('scope = producer.proposed_first_observed_scope', 1)[0]
    setup = prefix + r'''
from easyicu.webserver import durable_research, host_action_jobs, raw_preparation_candidate
from easyicu.webserver.deployment_lease import acquire_single_process_lease
from easyicu.webserver.host_action_contracts import HostDecisionOffers
from easyicu.webserver.routes import agent as route
acquire_single_process_lease(); jobs.MANAGER.enable_persistence(root/'jobs.sqlite3')
service=durable_research.DurableResearchService(root/'durable',manager=jobs.MANAGER)
service.launch(); durable_research._SERVICE=service
raw_preparation_candidate.create_candidate_for_review(run_id=run_id,study_context_id=study_id)
choice, checkpoint, scope, _=raw_preparation_candidate.available_host_review(run_id=run_id,study_context_id=study_id)
action=HostActionRequest(session_id=session,project_id=project,action_code='review_raw_preparation',decision=choice)
def submit(action, body, callback, **key_inputs):
    return host_action_jobs.submit_with_host_action(action,study_context_id=study_id,
        submit=lambda: {'job_id':jobs.MANAGER.reserve('raw-preparation-review',job_id='job').id},
        offered=lambda _: HostDecisionOffers(raw_preparation_review=choice),conversations=copilot,**key_inputs)
route._submit_host_decision=submit
binding=RawPreparationHostBinding(project_id=project,session_id=session,study_context_id=study_id,run_id=run_id,nonce=choice.nonce)
(root/'recovery.json').write_text(json.dumps({'binding':binding.model_dump(mode='json'),'scope':scope,
                                              'run':str(run),'action':action.model_dump(mode='json')}))
service.activate_preparation_handoff=lambda *a, **kw: os.kill(os.getpid(),9)
route.raw_preparation_review({'study_context_id':study_id,'decision':'approved','host_action':action.model_dump(mode='json')})
raise AssertionError('expected SIGKILL')
'''
    env = _environment(tmp_path)
    first = subprocess.run([sys.executable, '-c', setup, str(tmp_path)], env=env,
                           capture_output=True, text=True, timeout=60)
    assert first.returncode == -signal.SIGKILL, first.stderr
    from tests.support.first_observed_handoff import _RECOVER
    prefix, tail = _RECOVER.split("accepted=service.recover_approved_preparation_handoff", 1)
    tail = tail[tail.index('deadline=time.monotonic()+50'):]
    restart = prefix + r'''
from easyicu.webserver import raw_preparation_candidate
from easyicu.webserver.routes import agent as route
durable_research._SERVICE=service
from easyicu import state_paths
private=state_paths.state_root()/'private-preparation-decisions'
receipt=private/(binding.nonce+'.json'); original_receipt=receipt.read_bytes()
for condition in ('used','lease','receipt','source'):
    if condition=='used': (private/(binding.nonce+'.used')).write_text('{}')
    if condition=='lease': (run/'.preparation-execution-job').mkdir()
    if condition=='receipt':
        tampered=json.loads(original_receipt); tampered['scientific_configuration_sha256']='0'*64
        receipt.write_text(json.dumps(tampered))
    if condition=='source':
        old_source=producer._metadata_source
        def missing_source(_): raise ValueError('selected source metadata missing')
        producer._metadata_source=missing_source
    try:
        raw_preparation_candidate.recoverable_first_observed_review(
            run_id=binding.run_id,study_context_id=binding.study_context_id)
    except (ValueError,OSError): pass
    else: raise AssertionError('uncertain recovery offer accepted: '+condition)
    if condition=='used': (private/(binding.nonce+'.used')).unlink()
    if condition=='lease': (run/'.preparation-execution-job').rmdir()
    if condition=='receipt': receipt.write_bytes(original_receipt)
    if condition=='source': producer._metadata_source=old_source
    assert service.store.get('job') is None and not (root/'source-reads.txt').exists()
choice, session_id=raw_preparation_candidate.recoverable_first_observed_review(
    run_id=binding.run_id,study_context_id=binding.study_context_id)
assert session_id==binding.session_id and choice.model_dump(mode='json')==saved['action']['decision']
accepted=route.raw_preparation_review({'study_context_id':binding.study_context_id,'decision':'approved',
                                       'host_action':saved['action']})
assert accepted['job_id']=='job' and accepted['reused'] and accepted['preparation_status']=='preparing'
assert not accepted['analysis_authorized']
''' + tail
    second = subprocess.run([sys.executable, '-c', restart, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert second.returncode == 0, second.stderr
    assert (tmp_path/'source-reads.txt').read_text().splitlines() == ['read']
