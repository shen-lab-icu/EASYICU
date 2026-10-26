"""Identity and lock-transfer contracts, using only test-created processes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from easyicu.runtime.host_execution_lease import (
    HostExecutionLeaseError, acknowledge_host_execution_parent, assert_host_execution_quiescent,
    claim_host_execution, prepare_host_execution, read_host_execution_receipt,
    request_host_execution_stop, verify_host_worker,
)


def test_parent_fences_workspace_mutation_and_abort_can_only_precede_spawn(tmp_path):
    path = tmp_path.resolve() / "lease"
    lease = prepare_host_execution(path)
    try:
        with pytest.raises(HostExecutionLeaseError, match="host_execution_active"):
            prepare_host_execution(path)
        lease.bind_script("a" * 64)
        assert lease.script_sha256 == "a" * 64
        with pytest.raises(HostExecutionLeaseError, match="script_binding_changed"):
            lease.bind_script("b" * 64)
        lease.abort()
    finally:
        lease.release()
    assert_host_execution_quiescent(path, allow_missing=False)

    lease = prepare_host_execution(path, script_sha256="b" * 64)
    lease.mark_spawning()
    with pytest.raises(HostExecutionLeaseError, match="abort_ambiguous"):
        lease.abort()
    lease.release()
    # The OS lock is free, but a durable pending launch is still a fence.
    with pytest.raises(HostExecutionLeaseError, match="host_execution_unresolved"):
        prepare_host_execution(path)


def test_nonce_mismatch_cannot_clear_or_signal_another_launch(tmp_path):
    path = tmp_path.resolve() / "lease"
    lease = prepare_host_execution(path, script_sha256="a" * 64)
    lease.release()
    for operation in (request_host_execution_stop, claim_host_execution, verify_host_worker):
        with pytest.raises(HostExecutionLeaseError, match="nonce_changed"):
            operation(path, "0" * 32)
    assert not (path / "stop.json").exists()
    assert read_host_execution_receipt(path)["state"] == "prepared"


def test_invalid_nonce_does_not_leak_a_kernel_lock(tmp_path):
    path = tmp_path.resolve() / "lease"
    lease = prepare_host_execution(path, script_sha256="a" * 64)
    lease.abort()
    lease.release()
    with pytest.raises(HostExecutionLeaseError, match="nonce_invalid"):
        claim_host_execution(path, "invalid")
    assert_host_execution_quiescent(path, allow_missing=False)


def test_actual_guardian_holds_lock_across_parent_transfer_and_settles_stop(tmp_path):
    path = tmp_path.resolve() / "lease"
    lease = prepare_host_execution(path, script_sha256="a" * 64)
    program = r'''
import sys,time
from pathlib import Path
from easyicu.runtime.host_execution_lease import claim_host_execution
lease=claim_host_execution(Path(sys.argv[1]),sys.argv[2])
try:
    print("claimed",flush=True)
    deadline=time.monotonic()+5
    while not lease.stop_requested() and time.monotonic()<deadline:
        time.sleep(.01)
    assert lease.stop_requested()
    lease.mark_quiesced("test_stopped_before_worker",cleanup_proven=True)
finally:
    lease.release()
'''
    process = None
    try:
        lease.mark_spawning()
        process = subprocess.Popen(
            [sys.executable, "-c", program, str(path), lease.nonce],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        lease.bind_guardian(process.pid)
        lease.release()
        assert process.stdout.readline().strip() == "claimed"
        with pytest.raises(HostExecutionLeaseError, match="host_execution_active"):
            assert_host_execution_quiescent(path, wait_timeout=.05)
        request_host_execution_stop(path, lease.nonce)
        assert_host_execution_quiescent(path, allow_missing=False, wait_timeout=3, allow_current_parent=True)
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0, stdout + stderr
        # Guardian completion does not release the still-live parent's log and
        # artifact collection tail. Only that exact parent can acknowledge it.
        with pytest.raises(HostExecutionLeaseError, match="host_execution_active"):
            prepare_host_execution(path)
        with pytest.raises(HostExecutionLeaseError, match="host_execution_active"):
            assert_host_execution_quiescent(path)
        assert acknowledge_host_execution_parent(path, lease.nonce)
        assert_host_execution_quiescent(path, allow_missing=False)
        record = read_host_execution_receipt(path)
        assert record["guardian"]["pid"] == process.pid
        assert record["reason"] == "test_stopped_before_worker"
        if os.name != "nt":
            assert (path / "execution.json").stat().st_mode & 0o777 == 0o600
    finally:
        lease.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=3)


def test_receipt_with_dead_parent_can_be_superseded_before_worker_admission(tmp_path):
    path = tmp_path.resolve() / "lease"
    script = r'''
import os,sys
from pathlib import Path
from easyicu.runtime.host_execution_lease import prepare_host_execution
lease=prepare_host_execution(Path(sys.argv[1]),script_sha256="a"*64)
lease.mark_spawning()
os._exit(0)
'''
    subprocess.run([sys.executable, "-c", script, str(path)], check=True, timeout=5)
    prior = read_host_execution_receipt(path)
    assert_host_execution_quiescent(path, allow_missing=False)
    assert read_host_execution_receipt(path) == prior  # Verification never rewrites the old nonce.
    lease = prepare_host_execution(path, script_sha256="b" * 64)
    try:
        assert lease.nonce != prior["nonce"]
        lease.abort()
    finally:
        lease.release()


@pytest.mark.parametrize("mutation", [
    {"state": []}, {"parent": {"pid": 1, "created": float("nan")}},
    {"state": "running", "guardian": None, "worker": None},
])
def test_corrupt_or_unresolved_receipt_never_authorizes_reuse(tmp_path, mutation):
    path = tmp_path.resolve() / "lease"
    lease = prepare_host_execution(path, script_sha256="a" * 64)
    lease.release()
    record = json.loads((path / "execution.json").read_text())
    record.update(mutation)
    (path / "execution.json").write_text(json.dumps(record))
    with pytest.raises(HostExecutionLeaseError):
        assert_host_execution_quiescent(path, allow_missing=False)
    with pytest.raises(HostExecutionLeaseError):
        prepare_host_execution(path)


def test_missing_receipt_is_allowed_only_when_caller_explicitly_allows(tmp_path):
    path = tmp_path.resolve() / "missing"
    assert_host_execution_quiescent(path)
    with pytest.raises(HostExecutionLeaseError, match="receipt_missing"):
        assert_host_execution_quiescent(path, allow_missing=False)


def test_non_host_core_import_does_not_require_optional_psutil(tmp_path):
    script = r'''
import sys
from pathlib import Path
sys.modules["psutil"] = None
from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent, prepare_host_execution, HostExecutionLeaseError
path=Path(sys.argv[1]) / "absent"
assert_host_execution_quiescent(path)
try:
    prepare_host_execution(path)
except HostExecutionLeaseError as exc:
    assert exc.reason_code == "host_execution_identity_unavailable"
else:
    raise AssertionError("Host execution without identity support was admitted")
assert not path.exists()
'''
    subprocess.run([sys.executable, "-c", script, str(tmp_path.resolve())], check=True, timeout=5)


def test_parent_acknowledgement_waits_for_a_transient_verifier_lock(tmp_path):
    from easyicu.runtime.host_execution_lease import _Lease

    path = tmp_path.resolve() / "lease"
    launch = prepare_host_execution(path, script_sha256="a" * 64)
    launch.abort()
    launch.release()
    # Model the guardian's quiesced receipt before the parent has finished
    # collecting artifacts. No worker is executing in this synthetic test.
    record = read_host_execution_receipt(path)
    record["parent_complete"] = False
    (path / "execution.json").write_text(json.dumps(record))
    locked = threading.Event()

    def verifier():
        lease = _Lease(path)
        try:
            locked.set()
            time.sleep(0.1)
        finally:
            lease.release()

    thread = threading.Thread(target=verifier)
    thread.start()
    try:
        assert locked.wait(2)
        assert acknowledge_host_execution_parent(path, launch.nonce)
        assert read_host_execution_receipt(path)["parent_complete"] is True
    finally:
        thread.join(timeout=2)
    assert not thread.is_alive()
    replacement = prepare_host_execution(path, script_sha256="b" * 64)
    replacement.abort()
    replacement.release()
