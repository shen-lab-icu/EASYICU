"""Real, isolated process crashes never overlap host output writers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import pytest


pytestmark = pytest.mark.skipif(os.name != "posix", reason="SIGKILL/setsid acceptance is POSIX-specific")


_PARENT = r'''
import json, sys, threading, time
from pathlib import Path
from easyicu.research_agent.execution.host_runner import HostRunner
from easyicu.research_agent.execution.step_execution import CodeOrigin

root, label = Path(sys.argv[1]), sys.argv[2]
control = root / ".synthetic-control"

if label == "no-group-kill":
    from easyicu.research_agent.execution import runner as shared_runner
    def refuse_group_kill(*args, **kwargs):
        raise AssertionError("a guarded host must not signal a reaped process group")
    shared_runner._terminate_process_group = refuse_group_kill

class SyntheticHostRunner(HostRunner):
    # Exercise the real transport/lease/worker, avoiding unrelated installed
    # scientific-package discovery in each short process.
    def export_validated_runtime_bundle(self):
        return {"schema": "easyicu.host_runtime_preflight/1", "provenance": {"runtime": "host"}}

    @property
    def authority_identity_sha256(self):
        return "1" * 64

    def _execute_command(self, *args, **kwargs):
        result = super()._execute_command(*args, **kwargs)
        if label == "tail" and not getattr(self, "_synthetic_tail_entered", False):
            self._synthetic_tail_entered = True
            def compete():
                try:
                    other = self.run(step_id="working", code=(control / "source.py").read_text(),
                                     code_origin=CodeOrigin.HOST_RENDERED_EXECUTOR)
                    payload = {"succeeded": other.succeeded, "stderr": other.stderr}
                except Exception as error:
                    payload = {"succeeded": False, "error": str(error)}
                (control / "same-instance.result.json").write_text(json.dumps(payload))
            contender = threading.Thread(target=compete, daemon=True)
            contender.start()
            contender.join(3)
            (control / "tail-ready").write_text("worker finished; parent still collecting")
            deadline = time.monotonic() + 15
            while not (control / "finish-tail").exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("synthetic parent tail was not released")
                time.sleep(.02)
        return result

runner = SyntheticHostRunner(workdir=root, cohort_parquet=root / "cohort.parquet", timeout_seconds=30)
try:
    result = runner.run(step_id="working", code=(control / "source.py").read_text(),
                        code_origin=CodeOrigin.HOST_RENDERED_EXECUTOR)
    payload = {"succeeded": result.succeeded, "stderr": result.stderr,
               "failure_code": getattr(result.runner_failure_code, "value", None)}
except Exception as error:
    payload = {"succeeded": False, "error": str(error), "error_type": type(error).__name__}
(control / (label + ".result.json")).write_text(json.dumps(payload))
'''


_CHILD = r'''
import json, os, subprocess, sys, time
from pathlib import Path
import psutil
control = Path(sys.argv[1])
child = psutil.Process()
(control / "child.json").write_text(json.dumps({"pid": child.pid, "created": child.create_time()}))
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
identity = psutil.Process(grandchild.pid)
(control / "grandchild.json").write_text(json.dumps({"pid": identity.pid, "created": identity.create_time()}))
time.sleep(60)
'''


_SOURCE = r'''
import json, os, subprocess, sys, time
from pathlib import Path
import psutil
root = Path(os.environ["EASYICU_RUN_DIR"])
control = root / ".synthetic-control"
out = Path(os.environ["STEP_OUT_DIR"])
with (control / "invocations.txt").open("a") as count:
    count.write("started\n")
if (control / "allow-completion").exists():
    previous = []
    for name in ("worker", "child", "grandchild"):
        marker = control / (name + ".json")
        if marker.exists():
            identity = json.loads(marker.read_text())
            try:
                process = psutil.Process(identity["pid"])
                if process.create_time() == identity["created"] and process.status() != psutil.STATUS_ZOMBIE:
                    previous.append(name)
            except psutil.NoSuchProcess:
                pass
    (control / "overlap.json").write_text(json.dumps(previous))
    (out / "answer.txt").write_text("synthetic completed output")
else:
    worker = psutil.Process()
    guardian = worker.parent()
    for name, process in (("worker", worker), ("guardian", guardian)):
        (control / (name + ".json")).write_text(json.dumps({"pid": process.pid, "created": process.create_time()}))
    (out / "running.txt").write_text("original writer")
    subprocess.Popen([sys.executable, "-c", CHILD_SOURCE, str(control)],
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while not (control / "grandchild.json").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("synthetic descendant did not start")
        time.sleep(.01)
    (control / "ready").write_text("ready")
    time.sleep(60)
'''.replace("CHILD_SOURCE", repr(_CHILD))


def _wait(predicate, *, seconds=15, detail="condition"):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    pytest.fail(f"Timed out waiting for {detail}")


def _identity(path):
    return json.loads(path.read_text())


def _live(identity):
    try:
        process = psutil.Process(identity["pid"])
        return (process.create_time() == identity["created"]
                and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def _kill_owned(identity):
    try:
        process = psutil.Process(identity["pid"])
        if process.create_time() == identity["created"]:
            process.kill()
    except psutil.NoSuchProcess:
        pass


@pytest.fixture
def owned_runs(tmp_path):
    from easyicu.runtime import host_worker

    root = tmp_path / "run_synthetic"
    control = root / ".synthetic-control"
    control.mkdir(parents=True)
    (root / "cohort.parquet").write_bytes(b"synthetic transport-only input")
    (control / "source.py").write_text(_SOURCE)
    environment = {**os.environ, "PYTHONPATH": str(Path(host_worker.__file__).parents[2]),
                   "PYTHONDONTWRITEBYTECODE": "1"}
    processes = []

    def start(label):
        process = subprocess.Popen([sys.executable, "-c", _PARENT, str(root), label],
                                   env=environment, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        processes.append((process, {"pid": process.pid, "created": psutil.Process(process.pid).create_time()}))
        return process

    try:
        yield root, control, start
    finally:
        # Signal only PIDs whose creation time was captured by this fixture or
        # its synthetic worker. A test failure must not leave sleepers behind.
        for name in ("grandchild", "child", "worker", "guardian"):
            marker = control / (name + ".json")
            if marker.exists():
                _kill_owned(_identity(marker))
        for process, identity in processes:
            _kill_owned(identity)
            process.wait(timeout=5)


def _result(control, label):
    path = control / (label + ".result.json")
    _wait(path.exists, detail=f"{label} parent result")
    return json.loads(path.read_text())


def _assert_blocked_without_mutation(root, control, start, label, output_name="running.txt"):
    script = root / "steps" / "working" / "analysis.py"
    output = root / "steps" / "working" / "outputs" / output_name
    before = (script.read_bytes(), output.read_bytes(), (control / "invocations.txt").read_bytes())
    process = start(label)
    result = _result(control, label)
    process.wait(timeout=5)
    assert not result["succeeded"], result
    assert any(code in json.dumps(result) for code in ("host_execution_active", "host_execution_unresolved")), result
    assert (script.read_bytes(), output.read_bytes(), (control / "invocations.txt").read_bytes()) == before


def test_parent_sigkill_reaps_sets_id_descendants_before_reuse(owned_runs):
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent

    root, control, start = owned_runs
    parent = start("first")
    _wait((control / "ready").exists, detail="owned worker and detached grandchild")
    identities = [_identity(control / (name + ".json")) for name in ("worker", "guardian", "child", "grandchild")]
    assert all(_live(identity) for identity in identities)
    assert all(os.getsid(identity["pid"]) == identity["pid"] for identity in identities[2:])
    _assert_blocked_without_mutation(root, control, start, "concurrent")

    (control / "allow-completion").write_text("continue")
    os.kill(parent.pid, signal.SIGKILL)
    parent.wait(timeout=5)
    racer = start("crash-race")
    race_result = _result(control, "crash-race")
    racer.wait(timeout=5)
    _wait(lambda: not any(_live(identity) for identity in identities), detail="orphan cleanup")
    assert_host_execution_quiescent(root / ".host-execution")
    if not race_result["succeeded"]:
        assert any(code in json.dumps(race_result) for code in ("host_execution_active", "host_execution_unresolved")), race_result
        resumed = start("resumed")
        assert _result(control, "resumed")["succeeded"]
        resumed.wait(timeout=5)
    assert json.loads((control / "overlap.json").read_text()) == []
    assert (control / "invocations.txt").read_text().splitlines() == ["started", "started"]
    assert (root / "steps/working/outputs/answer.txt").read_text() == "synthetic completed output"


def test_lost_guardian_never_authorizes_another_writer(owned_runs):
    from easyicu.runtime.host_execution_lease import HostExecutionLeaseError, assert_host_execution_quiescent

    root, control, start = owned_runs
    parent = start("first")
    _wait((control / "ready").exists, detail="owned worker and detached grandchild")
    guardian = _identity(control / "guardian.json")
    worker = _identity(control / "worker.json")
    # Freeze the parent first so its normal cleanup cannot certify the loss;
    # both supervisor deaths model an abrupt local process-group termination.
    os.kill(parent.pid, signal.SIGSTOP)
    os.kill(worker["pid"], signal.SIGSTOP)
    _kill_owned(guardian)
    os.kill(parent.pid, signal.SIGKILL)
    parent.wait(timeout=5)
    _wait(lambda: not _live(guardian), detail="guardian death")
    assert _live(worker)
    with pytest.raises(HostExecutionLeaseError, match="host_execution_(active|unresolved)"):
        assert_host_execution_quiescent(root / ".host-execution")
    _assert_blocked_without_mutation(root, control, start, "unresolved")


def test_worker_sigkill_leaves_unresolved_without_output_collection_or_reuse(owned_runs):
    from easyicu.runtime.host_execution_lease import (
        HostExecutionLeaseError,
        assert_host_execution_quiescent,
        read_host_execution_receipt,
    )

    root, control, start = owned_runs
    parent = start("first")
    _wait((control / "ready").exists, detail="owned worker and detached grandchild")
    worker = _identity(control / "worker.json")
    guardian = _identity(control / "guardian.json")
    descendants = [_identity(control / (name + ".json")) for name in ("child", "grandchild")]
    script = root / "steps/working/analysis.py"
    output = root / "steps/working/outputs/running.txt"
    before = (script.read_bytes(), output.read_bytes(), (control / "invocations.txt").read_bytes())
    assert _live(guardian)
    assert _live(worker)
    _kill_owned(worker)

    result = _result(control, "first")
    parent.wait(timeout=5)
    assert not result["succeeded"], result
    assert "host_execution_unresolved" in json.dumps(result), result
    receipt = read_host_execution_receipt(root / ".host-execution")
    assert receipt["state"] == "unresolved"
    assert receipt["reason"] == "worker_exit_without_cleanup"
    assert (script.read_bytes(), output.read_bytes(), (control / "invocations.txt").read_bytes()) == before
    assert not (root / "steps/working/outputs/answer.txt").exists()
    # These detached children can no longer be discovered from the lost worker.
    # Only this fixture's captured identities authorize their eventual cleanup.
    assert all(_live(identity) for identity in descendants)
    with pytest.raises(HostExecutionLeaseError, match="host_execution_unresolved"):
        assert_host_execution_quiescent(root / ".host-execution")
    _assert_blocked_without_mutation(root, control, start, "worker-lost-contender")


def test_guarded_host_completion_never_signals_reaped_process_group(owned_runs):
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent

    root, control, start = owned_runs
    (control / "allow-completion").write_text("complete worker immediately")
    parent = start("no-group-kill")
    result = _result(control, "no-group-kill")
    parent.wait(timeout=5)
    assert result["succeeded"], result
    assert (root / "steps/working/outputs/answer.txt").read_text() == "synthetic completed output"
    assert_host_execution_quiescent(root / ".host-execution")


def test_parent_collection_tail_blocks_same_instance_and_other_process(owned_runs):
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent

    root, control, start = owned_runs
    (control / "allow-completion").write_text("complete worker immediately")
    parent = start("tail")
    _wait((control / "tail-ready").exists, detail="parent collection tail")
    same_instance = _result(control, "same-instance")
    assert not same_instance["succeeded"], same_instance
    assert "host_execution_active" in json.dumps(same_instance)
    assert (control / "invocations.txt").read_text().splitlines() == ["started"]
    _assert_blocked_without_mutation(root, control, start, "tail-contender", output_name="answer.txt")
    (control / "finish-tail").write_text("release parent")
    assert _result(control, "tail")["succeeded"]
    parent.wait(timeout=5)
    assert_host_execution_quiescent(root / ".host-execution")
