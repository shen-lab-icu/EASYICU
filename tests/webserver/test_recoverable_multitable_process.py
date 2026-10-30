"""Real guardian and worker processes, using a synthetic source only."""
import os
from pathlib import Path
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
_SUCCESS = _SETUP + r'''
import shutil
from easyicu.webserver import raw_preparation_durable as durable
from easyicu.webserver.recoverable_multitable_projection import read_completed_multitable_projection
durable._owner=lambda binding,receipt,job_id:root
durable.admit_preparation_intent(job_id='job',checkpoint=approved,binding=binding,scope=scope)
kwargs=dict(lease_dir=root/'.preparation-execution-job',checkpoint=approved,binding=binding,
            scope=scope,host_work_root=root,timeout=30,job_id='job')
output=guarded_multitable_projection(**kwargs)
assert pq.read_table(output/'part-0000.parquet').to_pydict()=={'stay_id':[1,2],'age':[65,71]}
assert pq.read_table(output/'part-0001.parquet').to_pydict()=={'stay_id':[1,2],'lact':[1.,2.]}
shutil.rmtree(source)  # This synthetic source is not consulted to read completed output.
assert read_completed_multitable_projection(job_id='job',checkpoint=approved,binding=binding,
                                           scope=scope,host_work_root=root)==output
assert guarded_multitable_projection(**kwargs)==output
assert len(list(root.glob('reviewed-multitable-projection-*')))==1
'''


def test_guarded_recoverable_projection_reuses_completed_synthetic_result(tmp_path):
    env = os.environ.copy()
    env['EASYICU_HOME'] = str(tmp_path / 'home')
    result = subprocess.run([sys.executable, '-c', _SUCCESS, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr


_CRASH_START = _SETUP + r'''
from easyicu.webserver import raw_preparation_durable as durable
from easyicu.webserver.jobs import Job
durable._owner=lambda binding,receipt,job_id:root
durable.admit_preparation_intent(job_id='job',checkpoint=approved,binding=binding,scope=scope)
durable.recover_preparation_with_sealed_snapshot(job_id='job',job=Job('job','raw-preparation-review'),timeout=60)
'''

_CRASH_RECOVER = r'''
import sys
from pathlib import Path
import pyarrow.parquet as pq
from easyicu.webserver import raw_preparation_durable as durable
from easyicu.webserver.jobs import Job
root=Path(sys.argv[1]); durable._owner=lambda binding,receipt,job_id:root
if sys.argv[2]=='before':
    try:
        durable.recover_preparation_with_sealed_snapshot(job_id='job',job=Job('job','raw-preparation-review'),timeout=30)
        raise AssertionError('unsealed consumed input was replayed')
    except ValueError as exc:
        assert str(exc)=='guarded_preparation_failed', str(exc)
    assert not list(root.glob('reviewed-multitable-projection-*/output'))
else:
    output=durable.recover_preparation_with_sealed_snapshot(job_id='job',job=Job('job','raw-preparation-review'),timeout=30)
    assert pq.read_table(output/'part-0000.parquet').to_pydict()=={'stay_id':[1,2],'age':[65,71]}
    assert pq.read_table(output/'part-0001.parquet').to_pydict()=={'stay_id':[1,2],'lact':[1.,2.]}
'''

_HOLD = r'''
import os, time
from pathlib import Path
phase=os.environ.get('EASYICU_TEST_HOLD_PHASE')
if phase:
    from easyicu.webserver import recoverable_multitable_projection as projection
    real=projection.stage_reviewed_source_files
    def hold(**kwargs):
        source=Path(kwargs['source_root'])
        if (phase=='before' and source.name=='source') or (phase=='after' and source.name=='snapshot'):
            Path(os.environ['EASYICU_TEST_HOLD_FILE']).write_text('waiting')
            while True: time.sleep(.05)
        return real(**kwargs)
    projection.stage_reviewed_source_files=hold
    if phase=='completion':
        read_completed=projection.read_completed_multitable_projection
        def hold_completed(**kwargs):
            Path(os.environ['EASYICU_TEST_HOLD_FILE']).write_text('waiting')
            while True: time.sleep(.05)
        projection.read_completed_multitable_projection=hold_completed
'''


@pytest.mark.skipif(os.name != 'posix', reason='requires SIGKILL')
@pytest.mark.parametrize('phase', ['before', 'after', 'completion'])
def test_real_guardian_recovery_obeys_seal_boundary(tmp_path, phase):
    hook = tmp_path / 'hook'; hook.mkdir()
    (hook / 'sitecustomize.py').write_text(_HOLD)
    env = os.environ.copy(); env['EASYICU_HOME'] = str(tmp_path / 'home')
    env['EASYICU_TEST_HOLD_PHASE'] = phase
    env['EASYICU_TEST_HOLD_FILE'] = str(tmp_path / 'holding')
    env['PYTHONPATH'] = str(hook) + os.pathsep + env.get('PYTHONPATH', '')
    child = subprocess.Popen([sys.executable, '-c', _CRASH_START, str(tmp_path)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 45
        while not (tmp_path / 'holding').exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / 'holding').exists(), 'worker did not reach the selected crash boundary'
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
        assert_host_execution_quiescent(tmp_path / '.preparation-execution-job', allow_missing=False, wait_timeout=30)
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=10)
    # Only the synthetic original source is removed. A valid sealed retry must
    # still complete; the pre-seal crash must refuse rather than reread it.
    shutil.rmtree(tmp_path / 'source')
    env.pop('EASYICU_TEST_HOLD_PHASE')
    result = subprocess.run([sys.executable, '-c', _CRASH_RECOVER, str(tmp_path), phase], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
