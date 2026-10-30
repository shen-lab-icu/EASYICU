"""First preparation travels through the real host route and guardian.

The patient source and the offered workflow decision are test stubs.  This
does not approve or execute research, or demonstrate the real Demo journey.
"""
import subprocess
import sys
import signal

from tests.webserver.test_first_observed_approved_handoff_recovery import _environment
from tests.webserver.test_first_observed_guarded_process import _PROCESS


def test_first_review_starts_same_guarded_preparation_without_analysis(tmp_path):
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
choice, checkpoint, scope, _=raw_preparation_candidate.available_host_review(run_id=run_id,study_context_id=study_id)
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
assert (root/'source-reads.txt').read_text().splitlines()==['read']
from easyicu.webserver.raw_preparation_projection import pending_preparation_projection
projection=pending_preparation_projection({'run_id':run_id,'project_dir':str(run)},study_context_id=study_id)
assert projection=={'status':'first_observed_complete','analysis_authorized':False}
coordinate=root/'home'/'.easyicu'/'private-first-observed'/(choice.nonce+'.parquet')
saved=coordinate.read_bytes(); coordinate.write_bytes(saved+b'changed')
assert pending_preparation_projection({'run_id':run_id,'project_dir':str(run)},study_context_id=study_id)['status']=='first_observed_unavailable'
coordinate.write_bytes(saved)
assert (root/'source-reads.txt').read_text().splitlines()==['read']
assert service.close()
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], env=_environment(tmp_path),
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
    from tests.webserver.test_first_observed_approved_handoff_recovery import _RECOVER
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
        old_source=producer.verified_public_demo_raw
        producer.verified_public_demo_raw=lambda _: None
    try:
        raw_preparation_candidate.recoverable_first_observed_review(
            run_id=binding.run_id,study_context_id=binding.study_context_id)
    except (ValueError,OSError): pass
    else: raise AssertionError('uncertain recovery offer accepted: '+condition)
    if condition=='used': (private/(binding.nonce+'.used')).unlink()
    if condition=='lease': (run/'.preparation-execution-job').rmdir()
    if condition=='receipt': receipt.write_bytes(original_receipt)
    if condition=='source': producer.verified_public_demo_raw=old_source
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
