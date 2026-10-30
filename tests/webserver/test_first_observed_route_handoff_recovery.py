"""A real host route may resume only an existing signed preparation job."""
import os
import signal
import subprocess
import sys

import pytest

from tests.webserver.test_first_observed_approved_handoff_recovery import _environment, _RECOVER, _setup_script


@pytest.mark.skipif(os.name != 'posix', reason='requires SIGKILL')
def test_route_resumes_exact_previously_approved_job_after_restart(tmp_path):
    env = _environment(tmp_path)
    setup = _setup_script().replace('"run": str(run)}))', '"run": str(run), "action": action.model_dump(mode="json")}))')
    assert '"action": action.model_dump' in setup
    first = subprocess.run([sys.executable, '-c', setup, str(tmp_path)], env=env,
                           capture_output=True, text=True, timeout=60)
    assert first.returncode == -signal.SIGKILL, first.stderr
    prefix, tail = _RECOVER.split("accepted=service.recover_approved_preparation_handoff", 1)
    tail = tail[tail.index('deadline=time.monotonic()+50'):]
    script = prefix + '''
from fastapi import HTTPException
from easyicu.webserver.routes import agent as route
durable_research._SERVICE=service
body={'study_context_id':binding.study_context_id,'decision':'approved','host_action':saved['action']}
for field in ('checkpoint_sha256','scope_sha256','nonce'):
    wrong=json.loads(json.dumps(body)); wrong['host_action']['decision'][field]='0'*64
    try:
        route.raw_preparation_review(wrong)
    except HTTPException as exc:
        assert exc.status_code==409
    else:
        raise AssertionError('changed host action was accepted: '+field)
    assert service.store.get('job') is None and not (root/'source-reads.txt').exists()
original_source=producer.verified_public_demo_raw
producer.verified_public_demo_raw=lambda _: None
try:
    route.raw_preparation_review(body)
except HTTPException as exc:
    assert exc.status_code==409
else:
    raise AssertionError('changed source was accepted')
producer.verified_public_demo_raw=original_source
assert service.store.get('job') is None and not (root/'source-reads.txt').exists()
accepted=route.raw_preparation_review(body)
assert accepted['job_id']=='job' and accepted['reused'] and accepted['preparation_status']=='preparing'
assert not accepted['analysis_authorized']
''' + tail
    script = script.replace('assert service.close()', '''
from concurrent.futures import ThreadPoolExecutor
before=service.store.get('job')
service._dispatch=lambda *_: (_ for _ in ()).throw(AssertionError('unexpected redispatch'))
with ThreadPoolExecutor(max_workers=4) as pool:
    repeated=list(pool.map(lambda _: route.raw_preparation_review(body), range(4)))
assert all(item['job_id']=='job' and item['job_status']=='done' and item['preparation_status']=='complete' and item['reused'] for item in repeated)
after=service.store.get('job')
assert before==after
assert (root/'source-reads.txt').read_text().splitlines()==['read']
from easyicu import state_paths
claim=state_paths.state_root()/'private-preparation-decisions'/(binding.nonce+'.host-job.json')
original=claim.read_bytes()
claim.write_bytes(original.replace(b'job',b'bad',1))
try:
    route.raw_preparation_review(body)
except HTTPException as exc:
    assert exc.status_code==409
else:
    raise AssertionError('tampered signed claim accepted')
claim.write_bytes(original)
producer.verified_public_demo_raw=lambda _: None
try:
    route.raw_preparation_review(body)
except HTTPException as exc:
    assert exc.status_code==409
else:
    raise AssertionError('changed source after completion accepted')
producer.verified_public_demo_raw=original_source
coordinate=state_paths.state_root()/'private-first-observed'/(binding.nonce+'.parquet')
original_coordinate=coordinate.read_bytes()
coordinate.write_bytes(original_coordinate+b'bad')
try:
    route.raw_preparation_review(body)
except HTTPException as exc:
    assert exc.status_code==409
else:
    raise AssertionError('changed coordinate accepted')
coordinate.write_bytes(original_coordinate)
from easyicu.webserver.raw_preparation_durable import recover_preparation_with_sealed_snapshot
reused=recover_preparation_with_sealed_snapshot(job_id='job',job=job,timeout=30)
assert reused==coordinate
assert (root/'source-reads.txt').read_text().splitlines()==['read']
assert service.store.get('job')==after
assert service.close()
''')
    second = subprocess.run([sys.executable, '-c', script, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert second.returncode == 0, second.stderr
    assert (tmp_path / 'source-reads.txt').read_text().splitlines() == ['read']
