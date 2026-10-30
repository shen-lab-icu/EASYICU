"""Real process cleanup for the fixed preparation worker; no patient data."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent, read_host_execution_receipt

_CHILD = r'''
import subprocess, sys, time
from pathlib import Path
from easyicu.runtime.host_execution_lease import prepare_host_execution
from easyicu.webserver.multitable_guarded_worker import identity
root = Path(sys.argv[1]); lease = prepare_host_execution(root / 'lease', script_sha256=identity())
lease.mark_spawning()
worker = subprocess.Popen([sys.executable, '-m', 'easyicu.webserver.multitable_guarded_worker',
    'supervise', str(lease.path), lease.nonce, lease.script_sha256, '60'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
lease.bind_guardian(worker.pid); lease.release()
# Leave stdin open so the verified worker waits for its request.
time.sleep(2)
(root / 'armed').write_text('ready')
while True: time.sleep(1)
'''


@pytest.mark.skipif(os.name != 'posix', reason='requires SIGKILL')
def test_guardian_confirms_worker_cleanup_after_parent_crash(tmp_path):
    # Invalid, data-free payload ensures this test grants no source access. Kill
    # the parent while the worker is live; the guardian must stop its process tree.
    child = subprocess.Popen([sys.executable, '-c', _CHILD, str(tmp_path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        receipt = None
        while time.monotonic() < deadline and child.poll() is None:
            try:
                receipt = read_host_execution_receipt(tmp_path / 'lease')
            except Exception:
                receipt = None
            if receipt and receipt.get('worker') and (tmp_path / 'armed').exists():
                break
            time.sleep(.005)
        assert receipt and receipt.get('worker'), 'guarded worker did not start'
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=10)
        assert_host_execution_quiescent(tmp_path / 'lease', allow_missing=False, wait_timeout=30)
        final = read_host_execution_receipt(tmp_path / 'lease')
        assert final['state'] == 'quiesced'
        assert final['reason'] in {'parent_lost', 'worker_failed'}
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

_SUCCESS = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
import pyarrow as pa, pyarrow.parquet as pq
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.webserver import sources, jobs, agent_pipeline_runs
from easyicu.webserver.pi_copilot import service
from easyicu.webserver.host_action_contracts import HostActionRequest, HostActionTag, RawPreparationReviewDecision, decision_key, host_action_id
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, record_host_review
from easyicu.webserver.multitable_guarded_worker import guarded_multitable_projection
root=Path(sys.argv[1]); source=root/'source'; source.mkdir()
pq.write_table(pa.table({'stay_id':[1,2],'age':[65,71]}),source/'demographics.parquet')
pq.write_table(pa.table({'stay_id':[1,2],'lact':[1.,2.]}),source/'blood_gas.parquet')
(source/'_manifest.json').write_text(json.dumps({'database':'miiv','format':'parquet',
 'concept_selection':{'mode':'explicit','modules':{'demographics':['age'],'blood_gas':['lact']}},
 'feature_definitions':{'included':False},'files':[{'file':'demographics.parquet','module':'demographics','concepts':1,'concept_ids':['age'],'rows':2},
 {'file':'blood_gas.parquet','module':'blood_gas','concepts':1,'concept_ids':['lact'],'rows':2}]}))
assert sources.register_source(str(source),label='synthetic',crossdb=False)['ok']
identity=sources.registered_source_identity(str(source)); assert identity
scope={'adapter_id':'registered-parquet-projections/1','source_identity':identity.to_dict(),
 'inputs':[{'table':'demographics','path':'demographics.parquet','columns':['stay_id','age']},
           {'table':'blood_gas','path':'blood_gas.parquet','columns':['stay_id','lact']}],
 'purpose':'prepare_multiple_tables_only','plan_sha256':'c'*64,'dependencies_sha256':'d'*64}
payload={'raw_preparation_scope':scope}
request=HumanReviewRequest.create(kind='scientific_stop',summary='Synthetic: read both full files',payload=payload,authority_sha256=canonical_sha256(payload))
pending=RawPreparationCheckpoint.pending(plan_sha256='c'*64,request=request)
choice=RawPreparationReviewDecision(run_id='run',scientific_configuration_sha256='a'*64,checkpoint_sha256=pending.checkpoint_sha256,
 scope_sha256=canonical_sha256(scope),review_authority_sha256=request.authority_sha256,nonce='b'*64)
action=HostActionRequest(session_id='session',project_id='project',action_code='review_raw_preparation',decision=choice)
key=decision_key(choice,review_decision='approved')
tag=HostActionTag(session_id='session',project_id='project',study_context_id='study',action_code=action.action_code,
 decision_key=key,action_id=host_action_id('session',action.action_code,key))
jobs.MANAGER.get=lambda job_id: SimpleNamespace(host_action=tag)
agent_pipeline_runs.server_reviewer_identity=lambda:'fixture-reviewer'
service.get_pi_copilot_service=lambda:SimpleNamespace(host_action_session=lambda *a,**k:None,
 host_action_child_job=lambda *a,**k:{'child_job_id':'job'})
approved=record_host_review(host_action=action,pending=pending,decision='approved',study_context_id='study',job_id='job',run_id='run',scientific_configuration_sha256='a'*64)
binding=RawPreparationHostBinding(project_id='project',session_id='session',study_context_id='study',run_id='run',nonce=choice.nonce)
output=guarded_multitable_projection(lease_dir=root/'lease',checkpoint=approved,binding=binding,scope=scope,host_work_root=root,timeout=30)
assert pq.read_table(output/'part-0000.parquet').to_pydict()=={'stay_id':[1,2],'age':[65,71]}
assert pq.read_table(output/'part-0001.parquet').to_pydict()=={'stay_id':[1,2],'lact':[1.,2.]}
try:
    guarded_multitable_projection(lease_dir=root/'lease',checkpoint=approved,binding=binding,scope=scope,host_work_root=root,timeout=30)
    raise AssertionError('consumed authorization replayed')
except ValueError as exc:
    assert str(exc)=='guarded_preparation_failed'
assert len(list(root.glob('reviewed-multitable-projection-*')))==1
'''


def test_guarded_projection_runs_and_verifies_host_approved_synthetic_source(tmp_path):
    env = os.environ.copy()
    env['EASYICU_HOME'] = str(tmp_path / 'home')
    result = subprocess.run([sys.executable, '-c', _SUCCESS, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
