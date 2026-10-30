"""Restart from one real signed host approval with synthetic source inputs."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from tests.webserver.test_first_observed_guarded_process import _PROCESS, _SOURCE_STUB


_OLD_JOB = '''job = jobs.MANAGER.reserve("raw-preparation-review", job_id="job")
tag = HostActionTag(session_id=session, project_id=project, study_context_id=study_id,
    action_code=action.action_code, decision_key=key, action_id=host_action_id(session, action.action_code, key))
assert job.tag_host_action(tag)
copilot.record_job_host_action(session, project_id=project, action_code=action.action_code, action_key=key, child_job_id="job")'''
_HOST_JOB = '''from easyicu.webserver import host_action_jobs
from easyicu.webserver.host_action_contracts import HostDecisionOffers
receipt = host_action_jobs.submit_with_host_action(action, study_context_id=study_id,
    submit=lambda: {"job_id": jobs.MANAGER.reserve("raw-preparation-review", job_id="job").id},
    review_decision="approved", offered=lambda _: HostDecisionOffers(raw_preparation_review=choice), conversations=copilot)
assert receipt["host_action"]["recorded"]
job = jobs.MANAGER.get("job")'''


def _setup_script():
    prefix = _PROCESS.split('durable.admit_preparation_intent(job_id="job"', 1)[0]
    assert _OLD_JOB in prefix
    prefix = prefix.replace(_OLD_JOB, _HOST_JOB)
    marker = 'root = Path(sys.argv[1]); run_id = "run_fixture"; project = "project_fixture"'
    assert marker in prefix
    prefix = prefix.replace(marker, marker + '''
from easyicu.webserver.deployment_lease import acquire_single_process_lease
acquire_single_process_lease()
jobs.MANAGER.enable_persistence(root / "jobs.sqlite3")''')
    return prefix + '''
(run / "raw_preparation_checkpoint.json").write_text(approved.model_dump_json())
(root / "recovery.json").write_text(json.dumps({"binding": binding.model_dump(mode="json"), "scope": scope,
    "run": str(run)}))
os.kill(os.getpid(), 9)
'''


_RECOVER = r'''
import json, sys, time
from pathlib import Path
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.webserver import durable_research, first_observed_preparation as producer, jobs
from easyicu.webserver.deployment_lease import acquire_single_process_lease
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding
root=Path(sys.argv[1]); saved=json.loads((root/'recovery.json').read_text()); run=Path(saved['run'])
binding=RawPreparationHostBinding.model_validate(saved['binding']); scope=saved['scope']
checkpoint=RawPreparationCheckpoint.model_validate_json((run/'raw_preparation_checkpoint.json').read_bytes())
acquire_single_process_lease()
jobs.MANAGER.enable_persistence(root/'jobs.sqlite3')
service=durable_research.DurableResearchService(root/'durable',manager=jobs.MANAGER); service.launch()
accepted=service.recover_approved_preparation_handoff('job',checkpoint=checkpoint,binding=binding,scope=scope,timeout=30)
assert accepted=={'job_id':'job','preparation_status':'preparing','analysis_authorized':False}
again=service.recover_approved_preparation_handoff('job',checkpoint=checkpoint,binding=binding,scope=scope,timeout=30)
assert again['job_id']=='job' and again['job_status'] in {'running','done'}
assert not again['analysis_authorized']
deadline=time.monotonic()+50
while time.monotonic()<deadline:
    job=jobs.MANAGER.get('job'); command=service.store.get('job')
    if job and job.status!='running' and command and command['status']=='terminal': break
    time.sleep(.05)
else: raise AssertionError('recovered preparation did not terminate')
assert job.status=='done', job.error
assert job.result['preparation_status']=='first_observed_coordinate_complete' and not job.result['analysis_authorized']
output=producer.resolve_prepared_first_observed_coordinate(run_dir=run,checkpoint=checkpoint,binding=binding,scope=scope)
assert output.path.is_file()
assert service.close()
'''


def _environment(tmp_path):
    stub = _SOURCE_STUB.replace('    def derive(_):\n', '''    def derive(_):
        with (root / "source-reads.txt").open("a") as handle:
            handle.write("read\\n")
''')
    (tmp_path / 'sitecustomize.py').write_text(stub)
    env = os.environ.copy()
    env.update(EASYICU_HOME=str(tmp_path / 'home'), EASYICU_TEST_FIRST_OBSERVED_SOURCE='1',
               EASYICU_TEST_FIRST_OBSERVED_ROOT=str(tmp_path),
               PYTHONPATH=str(tmp_path) + os.pathsep + env.get('PYTHONPATH', ''))
    return env


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL")
@pytest.mark.parametrize('partial_intent', [False, True])
def test_approved_checkpoint_recovers_same_job_and_nonce_after_sigkill(tmp_path, partial_intent):
    env = _environment(tmp_path)
    setup = _setup_script()
    if partial_intent:
        setup = setup.replace('os.kill(os.getpid(), 9)', '''
durable.admit_preparation_intent(job_id="job", checkpoint=approved, binding=binding, scope=scope)
os.kill(os.getpid(), 9)''')
    first = subprocess.run([sys.executable, '-c', setup, str(tmp_path)], env=env,
                           capture_output=True, text=True, timeout=60)
    assert first.returncode == -signal.SIGKILL, first.stderr
    second = subprocess.run([sys.executable, '-c', _RECOVER, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert second.returncode == 0, second.stderr
    assert (tmp_path / 'source-reads.txt').read_text().splitlines() == ['read']


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL")
@pytest.mark.parametrize('change', ['consumed', 'lease', 'job_claim', 'plan', 'intent', 'study', 'runtime'])
def test_changed_or_uncertain_handoff_never_reads_source(tmp_path, change):
    env = _environment(tmp_path)
    first = subprocess.run([sys.executable, '-c', _setup_script(), str(tmp_path)], env=env,
                           capture_output=True, text=True, timeout=60)
    assert first.returncode == -signal.SIGKILL, first.stderr
    prefix = _RECOVER.split("accepted=service.recover_approved_preparation_handoff", 1)[0]
    script = prefix + '''
from easyicu import state_paths
from easyicu.webserver import raw_preparation_durable
private=state_paths.state_root()/'private-preparation-decisions'
mode=sys.argv[2]
if mode=='consumed': (private/(binding.nonce+'.used')).write_text('{}')
elif mode=='lease': (run/'.preparation-execution-job').mkdir()
elif mode=='job_claim': (private/(binding.nonce+'.host-job.json')).unlink()
elif mode=='plan':
    path=run/'progressive_plan_skeleton.json'; path.write_bytes(path.read_bytes()+b' ')
elif mode=='intent':
    raw_preparation_durable.admit_preparation_intent(job_id='job',checkpoint=checkpoint,binding=binding,scope=scope)
    path=private/'job.preparation-intent.json'; value=json.loads(path.read_text()); value['root']='/different'; path.write_text(json.dumps(value))
elif mode=='study':
    from easyicu.webserver import study_contexts
    study=study_contexts.get_context(binding.study_context_id)
    study_contexts.upsert_context({'id':binding.study_context_id, 'question':'changed research question'},
        active=True, expected_revision=study['revision'], require_revision=True, lifecycle_write=False)
elif mode=='runtime':
    producer.runtime_digest=lambda: '0'*64
try:
    service.recover_approved_preparation_handoff('job',checkpoint=checkpoint,binding=binding,scope=scope,timeout=30)
except Exception as exc:
    expected={'consumed':'nonce_replayed','lease':'lifecycle_present','job_claim':'original_job_unavailable',
              'plan':'population_unverified','intent':'recovery_record_invalid','study':'configuration_changed',
              'runtime':'intent_scope_mismatch'}
    assert expected[mode] in str(exc), (mode, type(exc).__name__, str(exc))
else:
    raise AssertionError('uncertain authority was accepted')
assert service.store.get('job') is None
assert not (root/'source-reads.txt').exists()
assert service.close()
'''
    second = subprocess.run([sys.executable, '-c', script, str(tmp_path), change], env=env,
                            capture_output=True, text=True, timeout=60)
    assert second.returncode == 0, second.stderr


@pytest.mark.skipif(os.name != "posix", reason="requires SIGKILL")
def test_capacity_failure_preserves_recoverable_handoff_without_active_outbox(tmp_path):
    env = _environment(tmp_path)
    first = subprocess.run([sys.executable, '-c', _setup_script(), str(tmp_path)], env=env,
                           capture_output=True, text=True, timeout=60)
    assert first.returncode == -signal.SIGKILL, first.stderr
    script = _RECOVER.replace("jobs.MANAGER.enable_persistence(root/'jobs.sqlite3')", """
jobs.MANAGER=jobs.JobManager(max_running=1)
jobs.MANAGER.enable_persistence(root/'jobs.sqlite3')
blocker=jobs.MANAGER.reserve('agent-run',job_id='capacity-blocker')""")
    script = script.replace("accepted=service.recover_approved_preparation_handoff", """
try:
    service.recover_approved_preparation_handoff('job',checkpoint=checkpoint,binding=binding,scope=scope,timeout=30)
except jobs.JobCapacityError:
    pass
else:
    raise AssertionError('full capacity accepted an interrupted job')
assert service.store.get('job') is None
from easyicu import state_paths
assert (state_paths.state_root()/'private-preparation-decisions'/'job.preparation-handoff.json').is_file()
blocker.complete_from_runner({})
accepted=service.recover_approved_preparation_handoff""")
    second = subprocess.run([sys.executable, '-c', script, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert second.returncode == 0, second.stderr
    assert (tmp_path / 'source-reads.txt').read_text().splitlines() == ['read']
