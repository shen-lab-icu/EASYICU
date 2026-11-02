"""Synthetic host-handoff restart scripts and isolated environment."""
import os

from tests.support.first_observed_process import _PROCESS, _SOURCE_STUB


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
