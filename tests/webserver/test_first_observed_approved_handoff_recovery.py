"""Restart from one real signed host approval with synthetic source inputs."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from tests.support.first_observed_handoff import _RECOVER, _environment, _setup_script










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
