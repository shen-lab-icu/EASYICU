"""Crash before decision consumption: recover the same synthetic host job."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys

import pytest

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
