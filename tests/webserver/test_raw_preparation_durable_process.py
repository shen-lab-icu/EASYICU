"""Crash before decision consumption: recover the same synthetic host job."""
from __future__ import annotations

import os
import hashlib
import hmac
import json
from pathlib import Path
import re
import runpy
import signal
import shutil
import subprocess
import sys
import time

import pytest

from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent
from easyicu.research_agent.canonical_json import canonical_json

_BASE = runpy.run_path(str(Path(__file__).with_name('test_multitable_guarded_process.py')))['_SUCCESS']
_SETUP = _BASE.split('output=guarded_multitable_projection(', 1)[0]
_ADMIT = _SETUP + r'''
import os, signal
from easyicu.webserver import raw_preparation_durable as durable
durable._owner=lambda binding, receipt, job_id: root
durable.admit_preparation_intent(job_id='job',checkpoint=approved,binding=binding,scope=scope)
os.kill(os.getpid(),signal.SIGKILL)
'''
_RECOVER = r'''
import sys
from pathlib import Path
import pyarrow.parquet as pq
from easyicu.webserver.jobs import Job
from easyicu.webserver import raw_preparation_durable as durable
root=Path(sys.argv[1]); durable._owner=lambda binding,receipt,job_id:root
job=Job('job','raw-preparation-review')
output=durable.recover_preparation_before_consumption(job_id='job',job=job,timeout=30)
assert pq.read_table(output/'part-0000.parquet').to_pydict()=={'stay_id':[1,2],'age':[65,71]}
assert pq.read_table(output/'part-0001.parquet').to_pydict()=={'stay_id':[1,2],'lact':[1.,2.]}
try:
    durable.recover_preparation_before_consumption(job_id='job',job=job,timeout=30)
    raise AssertionError('decision was replayed')
except ValueError as exc:
    assert 'nonce_replayed' in str(exc)
assert len(list(root.glob('reviewed-multitable-projection-*')))==1
'''


@pytest.mark.skipif(os.name != 'posix', reason='requires SIGKILL')
def test_same_unconsumed_host_decision_recovers_after_process_death(tmp_path):
    env = os.environ.copy(); env['EASYICU_HOME'] = str(tmp_path / 'home')
    first = subprocess.run([sys.executable, '-c', _ADMIT, str(tmp_path)], env=env,
                           capture_output=True, text=True, timeout=60)
    assert first.returncode == -signal.SIGKILL, first.stderr
    second = subprocess.run([sys.executable, '-c', _RECOVER, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert second.returncode == 0, second.stderr

_DBOS_START = _SETUP + r'''
import os, signal, time
from easyicu.webserver import raw_preparation_durable as prep, durable_research, jobs as jobmod
prep._owner=lambda binding, receipt, job_id: root
def pause_before_consumption(**kwargs):
    (root/'before-consumption').write_text('ready')
    while True: time.sleep(.05)
prep.recover_preparation_before_consumption=pause_before_consumption
manager=jobmod.JobManager(); manager.enable_persistence(root/'jobs.sqlite3')
service=durable_research.DurableResearchService(root/'durable',manager=manager); service.launch()
manager.reserve('raw-preparation-review',job_id='job')
service.activate_preparation('job',checkpoint=approved,binding=binding,scope=scope,timeout=30)
deadline=time.monotonic()+30
while not (root/'before-consumption').exists() and time.monotonic()<deadline: time.sleep(.05)
assert (root/'before-consumption').exists()
os.kill(os.getpid(),signal.SIGKILL)
'''
_DBOS_RECOVER = r'''
import sys, time
from pathlib import Path
import pyarrow.parquet as pq
from easyicu.webserver import raw_preparation_durable as prep, durable_research, jobs
root=Path(sys.argv[1]); prep._owner=lambda binding,receipt,job_id:root
manager=jobs.JobManager(); manager.enable_persistence(root/'jobs.sqlite3')
service=durable_research.DurableResearchService(root/'durable',manager=manager); service.launch()
deadline=time.monotonic()+75
while time.monotonic()<deadline:
    job=manager.get('job'); command=service.store.get('job')
    if job and job.status!='running' and command and command['status']=='terminal': break
    time.sleep(.05)
else: raise AssertionError('durable preparation did not reach terminal')
assert job.kind=='raw-preparation-review', job.kind
assert job.status=='done', str(job.error)
assert job.host_action and job.host_action.action_code=='review_raw_preparation'
assert job.result['cohort_complete'] is False and job.result['analysis_authorized'] is False
outputs=list(root.glob('reviewed-multitable-projection-*/output'))
assert len(outputs)==1
assert pq.read_table(outputs[0]/'part-0000.parquet').to_pydict()=={'stay_id':[1,2],'age':[65,71]}
assert service.close()
'''


@pytest.mark.skipif(os.name != 'posix', reason='requires SIGKILL')
def test_real_dbos_reattaches_preparation_job_before_consumption(tmp_path):
    env = os.environ.copy(); env['EASYICU_HOME'] = str(tmp_path / 'home')
    first = subprocess.run([sys.executable, '-c', _DBOS_START, str(tmp_path)], env=env,
                           capture_output=True, text=True, timeout=90)
    assert first.returncode == -signal.SIGKILL, first.stderr
    second = subprocess.run([sys.executable, '-c', _DBOS_RECOVER, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stderr


_SEALED_DBOS_START = _SETUP + r'''
import time
from easyicu.webserver import raw_preparation_durable as prep, durable_research, jobs as jobmod
prep._owner=lambda binding,receipt,job_id:root
manager=jobmod.JobManager(); manager.enable_persistence(root/'jobs.sqlite3')
service=durable_research.DurableResearchService(root/'durable',manager=manager); service.launch()
manager.reserve('raw-preparation-review',job_id='job')
service.activate_preparation('job',checkpoint=approved,binding=binding,scope=scope,timeout=60,sealed_recovery=True)
while True: time.sleep(.05)
'''

_SEALED_DBOS_RECOVER = r'''
import json, sys, time
from pathlib import Path
import pyarrow.parquet as pq
from easyicu import state_paths
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.webserver import raw_preparation_durable as prep, durable_research, jobs
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding
from easyicu.webserver.recoverable_multitable_projection import read_completed_multitable_projection
root=Path(sys.argv[1]); prep._owner=lambda binding,receipt,job_id:root
manager=jobs.JobManager(); manager.enable_persistence(root/'jobs.sqlite3')
service=durable_research.DurableResearchService(root/'durable',manager=manager); service.launch()
deadline=time.monotonic()+75
while time.monotonic()<deadline:
    job=manager.get('job'); command=service.store.get('job')
    if job and job.status!='running' and command and command['status']=='terminal': break
    time.sleep(.05)
else: raise AssertionError('DBOS sealed preparation did not reach terminal')
assert job.id=='job' and job.kind=='raw-preparation-review'
assert job.status=='done', str(job.error)
assert job.host_action and job.host_action.action_code=='review_raw_preparation'
assert job.result['cohort_complete'] is False and job.result['analysis_authorized'] is False
assert command['payload']['schema_version']=='easyicu.durable-preparation/2'
assert command['attempts']==2, command['attempts']
private=state_paths.state_root()/'private-preparation-decisions'
intent=json.loads((private/'job.preparation-intent.json').read_text())
checkpoint=RawPreparationCheckpoint.model_validate(intent['checkpoint'])
binding=RawPreparationHostBinding.model_validate(intent['binding'])
output=read_completed_multitable_projection(job_id='job',checkpoint=checkpoint,binding=binding,
                                            scope=intent['scope'],host_work_root=root)
assert pq.read_table(output/'part-0000.parquet').to_pydict()=={'stay_id':[1,2],'age':[65,71]}
assert pq.read_table(output/'part-0001.parquet').to_pydict()=={'stay_id':[1,2],'lact':[1.,2.]}
assert service.close()
'''


@pytest.mark.skipif(os.name != 'posix', reason='requires SIGKILL')
def test_real_dbos_recovers_same_job_after_sealed_snapshot_and_source_removal(tmp_path):
    # Test-only fault injection holds the real worker after sealing and before
    # projection; it does not inject a different workflow or replace DBOS.
    hook = tmp_path / 'hook'; hook.mkdir()
    holder = runpy.run_path(str(Path(__file__).with_name('test_recoverable_multitable_process.py')))['_HOLD']
    (hook / 'sitecustomize.py').write_text(holder)
    env = os.environ.copy(); env['EASYICU_HOME'] = str(tmp_path / 'home')
    env['EASYICU_TEST_HOLD_PHASE'] = 'after'
    env['EASYICU_TEST_HOLD_FILE'] = str(tmp_path / 'holding')
    env['PYTHONPATH'] = str(hook) + os.pathsep + env.get('PYTHONPATH', '')
    child = subprocess.Popen([sys.executable, '-c', _SEALED_DBOS_START, str(tmp_path)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / 'holding').exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / 'holding').exists(), 'DBOS worker did not reach the sealed boundary'
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
        assert_host_execution_quiescent(tmp_path / '.preparation-execution-job', allow_missing=False, wait_timeout=30)
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=10)
    private = tmp_path / 'home' / '.easyicu' / 'private-preparation-decisions'
    names = ['b' * 64 + '.json', 'b' * 64 + '.used', 'job.preparation-intent.json',
             'job.preparation-started', 'job.preparation-snapshot.json']
    original = {name: hashlib.sha256((private / name).read_bytes()).hexdigest() for name in names}
    decisions_before = sorted(path.name for path in private.iterdir() if re.fullmatch(r'[0-9a-f]{64}\.(json|used)', path.name))
    shutil.rmtree(tmp_path / 'source')
    env.pop('EASYICU_TEST_HOLD_PHASE')
    result = subprocess.run([sys.executable, '-c', _SEALED_DBOS_RECOVER, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert original == {name: hashlib.sha256((private / name).read_bytes()).hexdigest() for name in names}
    assert decisions_before == sorted(path.name for path in private.iterdir() if re.fullmatch(r'[0-9a-f]{64}\.(json|used)', path.name))


_HANDOFF_START = _SETUP + r'''
import os, signal
from easyicu.webserver import raw_preparation_durable as prep, durable_research, jobs as jobmod
from easyicu.webserver.deployment_lease import acquire_single_process_lease
prep._owner=lambda binding,receipt,job_id:root
acquire_single_process_lease()
manager=jobmod.JobManager(); manager.enable_persistence(root/'jobs.sqlite3')
service=durable_research.DurableResearchService(root/'durable',manager=manager); service.launch()
job=manager.reserve('raw-preparation-review',job_id='job')
def crash_before_outbox(*args):
    if sys.argv[2]=='cancel': job.request_cancel('test-cancel')
    os.kill(os.getpid(), signal.SIGKILL)
service.store.prepare_active_handoff=crash_before_outbox
service.activate_preparation_handoff('job',checkpoint=approved,binding=binding,scope=scope,timeout=30)
raise AssertionError('crash did not happen')
'''

_HANDOFF_RESTART = r'''
import sys,time
from pathlib import Path
from easyicu.webserver import raw_preparation_durable as prep, durable_research, jobs
from easyicu.webserver.deployment_lease import acquire_single_process_lease, release_single_process_lease
root=Path(sys.argv[1]); mode=sys.argv[2]
prep._owner=lambda binding,receipt,job_id:(root/'other' if mode=='owner' else root)
if mode=='runtime': prep.runtime_digest=lambda:'f'*64
if mode!='success':
    real_source_check=prep.require_reviewable_multitable_scope
    def observe_source_check(scope):
        (root/'source-metadata-touched').write_text('read')
        return real_source_check(scope)
    prep.require_reviewable_multitable_scope=observe_source_check
if mode!='no_lease': acquire_single_process_lease()
manager=jobs.JobManager(); manager.enable_persistence(root/'jobs.sqlite3')
service=durable_research.DurableResearchService(root/('other-durable' if mode=='otherstore' else 'durable'),manager=manager)
if mode=='no_lease':
    try:
        service.launch()
        raise AssertionError('recovery without deployment lease')
    except durable_research.DurableResearchError as exc:
        assert str(exc)=='preparation_handoff_deployment_lease_missing'
    assert service.store.get('job') is None
    sys.exit(0)
service.launch()
if mode=='success':
    deadline=time.monotonic()+75
    while time.monotonic()<deadline:
        job=manager.get('job'); command=service.store.get('job')
        if job and job.status!='running' and command and command['status']=='terminal': break
        time.sleep(.05)
    else: raise AssertionError('handoff not recovered')
    assert job.id=='job' and job.kind=='raw-preparation-review' and job.status=='done', str(job.error)
    assert command['payload']['schema_version']=='easyicu.durable-preparation/3'
    assert command['attempts']==1
    assert job.result['cohort_complete'] is False and job.result['analysis_authorized'] is False
else:
    time.sleep(.2)
    assert service.store.get('job') is None
    assert not list(root.glob('reviewed-multitable-projection-*'))
assert service.close(); release_single_process_lease()
'''


@pytest.mark.skipif(os.name != 'posix', reason='requires SIGKILL')
@pytest.mark.parametrize('mode', ['success', 'cancel', 'owner', 'runtime', 'used', 'started', 'lease', 'tamper', 'expired', 'no_lease', 'otherstore'])
def test_handoff_recovers_only_unstarted_exact_job_after_sigkill(tmp_path, mode):
    env = os.environ.copy(); env['EASYICU_HOME'] = str(tmp_path / 'home')
    first = subprocess.run([sys.executable, '-c', _HANDOFF_START, str(tmp_path), mode], env=env,
                           capture_output=True, text=True, timeout=60)
    assert first.returncode == -signal.SIGKILL, first.stderr
    private = tmp_path / 'home' / '.easyicu' / 'private-preparation-decisions'
    intent = private / 'job.preparation-intent.json'
    handoff = private / 'job.preparation-handoff.json'
    assert intent.is_file() and handoff.is_file()
    original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [intent, handoff, private / ('b' * 64 + '.json')]}
    if mode == 'used':
        (private / ('b' * 64 + '.used')).write_text('{}')
    elif mode == 'started':
        (private / 'job.preparation-started').write_text('{}')
    elif mode == 'lease':
        (tmp_path / '.preparation-execution-job').mkdir()
    elif mode == 'tamper':
        value = json.loads(handoff.read_text()); value['timeout_seconds'] = 3
        handoff.write_text(json.dumps(value))
    elif mode == 'expired':
        decision = private / ('b' * 64 + '.json')
        value = json.loads(decision.read_text()); value.pop('signature')
        value['expires_at'] = '2000-01-01T00:00:00+00:00'
        signature = hmac.new((private / 'host.key').read_bytes(), canonical_json(value).encode(), hashlib.sha256).hexdigest()
        decision.write_text(canonical_json({**value, 'signature': signature}))
    second = subprocess.run([sys.executable, '-c', _HANDOFF_RESTART, str(tmp_path), mode], env=env,
                            capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stderr
    if mode != 'success':
        assert not (tmp_path / 'source-metadata-touched').exists()
    if mode not in {'tamper', 'expired'}:
        assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [intent, handoff, private / ('b' * 64 + '.json')]}


_HANDOFF_SEALED_START = _SETUP + r'''
import time
from easyicu.webserver import raw_preparation_durable as prep, durable_research, jobs as jobmod
from easyicu.webserver.deployment_lease import acquire_single_process_lease
prep._owner=lambda binding,receipt,job_id:root
acquire_single_process_lease()
manager=jobmod.JobManager(); manager.enable_persistence(root/'jobs.sqlite3')
service=durable_research.DurableResearchService(root/'durable',manager=manager); service.launch()
manager.reserve('raw-preparation-review',job_id='job')
service.activate_preparation_handoff('job',checkpoint=approved,binding=binding,scope=scope,timeout=60)
while True: time.sleep(.05)
'''

_HANDOFF_SEALED_RECOVER = _SEALED_DBOS_RECOVER.replace(
    "manager=jobs.JobManager(); manager.enable_persistence(root/'jobs.sqlite3')",
    "from easyicu.webserver.deployment_lease import acquire_single_process_lease\n"
    "acquire_single_process_lease()\nmanager=jobs.JobManager(); manager.enable_persistence(root/'jobs.sqlite3')",
).replace("easyicu.durable-preparation/2", "easyicu.durable-preparation/3")


@pytest.mark.skipif(os.name != 'posix', reason='requires SIGKILL')
def test_dbos_handoff_job_recovers_after_seal_without_source(tmp_path):
    hook = tmp_path / 'hook'; hook.mkdir()
    holder = runpy.run_path(str(Path(__file__).with_name('test_recoverable_multitable_process.py')))['_HOLD']
    (hook / 'sitecustomize.py').write_text(holder)
    env = os.environ.copy(); env['EASYICU_HOME'] = str(tmp_path / 'home')
    env['EASYICU_TEST_HOLD_PHASE'] = 'after'
    env['EASYICU_TEST_HOLD_FILE'] = str(tmp_path / 'holding')
    env['PYTHONPATH'] = str(hook) + os.pathsep + env.get('PYTHONPATH', '')
    child = subprocess.Popen([sys.executable, '-c', _HANDOFF_SEALED_START, str(tmp_path)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / 'holding').exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / 'holding').exists(), 'handoff worker did not reach the sealed boundary'
        os.kill(child.pid, signal.SIGKILL); child.wait(timeout=10)
        assert_host_execution_quiescent(tmp_path / '.preparation-execution-job', allow_missing=False, wait_timeout=30)
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=10)
    private = tmp_path / 'home' / '.easyicu' / 'private-preparation-decisions'
    names = ['b' * 64 + '.json', 'b' * 64 + '.used', 'job.preparation-intent.json',
             'job.preparation-handoff.json', 'job.preparation-started', 'job.preparation-snapshot.json']
    original = {name: hashlib.sha256((private / name).read_bytes()).hexdigest() for name in names}
    shutil.rmtree(tmp_path / 'source')
    env.pop('EASYICU_TEST_HOLD_PHASE')
    result = subprocess.run([sys.executable, '-c', _HANDOFF_SEALED_RECOVER, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert original == {name: hashlib.sha256((private / name).read_bytes()).hexdigest() for name in names}


def test_completed_handoffs_do_not_exhaust_pending_recovery_limit(tmp_path, monkeypatch):
    from easyicu.webserver import durable_research, jobs, state_paths

    monkeypatch.setattr(state_paths, 'state_root', lambda: tmp_path)
    private = tmp_path / 'private-preparation-decisions'; private.mkdir(mode=0o700)
    service = durable_research.DurableResearchService(tmp_path / 'durable', manager=jobs.JobManager())
    with service.store._connection() as connection:
        connection.executemany("INSERT INTO commands VALUES (?, '{}', 'code', 'terminal', 1, 0, 'done')",
                               [(f'job{i}',) for i in range(1001)])
    for i in range(1001):
        (private / f'job{i}.preparation-handoff.json').write_text('{}')
    # Terminal jobs are never reopened and do not need a live owner lease.
    service._recover_handoffs()
