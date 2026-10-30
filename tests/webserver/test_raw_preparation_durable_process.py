"""Crash before decision consumption: recover the same synthetic host job."""
from __future__ import annotations

import os
import hashlib
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
