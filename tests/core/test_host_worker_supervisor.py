"""Guardian admission and cleanup acknowledgement failure boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from easyicu.runtime import host_execution_lease, host_worker


def _arguments(tmp_path: Path, **overrides):
    return argparse.Namespace(**{
        "action": "execute", "script": str(tmp_path / "steps" / "typed" / "analysis.py"),
        "sha256": "a" * 64, "lease_dir": str(tmp_path / ".host-execution"),
        "nonce": "b" * 32, "timeout": 1.0, **overrides,
    })


@pytest.mark.parametrize("changed", ["digest", "workspace"])
def test_registered_worker_never_executes_outside_its_lease(tmp_path, monkeypatch, changed):
    args = _arguments(tmp_path)
    monkeypatch.setattr(host_execution_lease, "verify_host_worker", lambda *a: "c" * 64 if changed == "digest" else args.sha256)
    if changed == "workspace":
        args.lease_dir = str(tmp_path / "another-run" / ".host-execution")
    monkeypatch.setattr(host_worker, "_execute", lambda *a, **kw: pytest.fail("unbound script"))
    with pytest.raises(ValueError, match="does not bind"):
        host_worker._execute_registered(args)


@pytest.mark.parametrize("failed", [False, True])
def test_registered_completion_waits_for_guardian_without_interpreter_exit(tmp_path, monkeypatch, failed):
    events = []
    args = _arguments(tmp_path)
    monkeypatch.setattr(host_execution_lease, "verify_host_worker", lambda *a: events.append("admitted") or args.sha256)

    def execute(*a, **kw):
        events.append("execute")
        if failed:
            raise RuntimeError("synthetic script failure")

    def wait_for_guardian(seconds):
        events.append("waiting_for_guardian")
        raise SystemExit(0)

    monkeypatch.setattr(host_worker, "_execute", execute)
    monkeypatch.setattr(host_worker, "_finish_descendants", lambda: events.append("tree_stopped"))
    monkeypatch.setattr(host_execution_lease, "complete_host_worker_cleanup", lambda *a, exitcode: events.append(("completion", exitcode)))
    monkeypatch.setattr(host_worker.os, "_exit", lambda *a: pytest.fail("worker exited before guardian cleanup"))
    monkeypatch.setattr(host_worker.time, "sleep", wait_for_guardian)
    with pytest.raises(SystemExit):
        host_worker._execute_registered(args)
    assert events == ["admitted", "execute", "tree_stopped", ("completion", int(failed)), "waiting_for_guardian"]


def test_cleanup_failure_never_acknowledges_worker_quiescence(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    monkeypatch.setattr(host_execution_lease, "verify_host_worker", lambda *a: args.sha256)
    monkeypatch.setattr(host_worker, "_execute", lambda *a, **kw: None)
    monkeypatch.setattr(host_worker, "_finish_descendants", lambda: (_ for _ in ()).throw(RuntimeError("unverified descendant")))
    monkeypatch.setattr(host_execution_lease, "complete_host_worker_cleanup", lambda *a: pytest.fail("false cleanup proof"))
    monkeypatch.setattr(host_worker.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit) as stopped:
        host_worker._execute_registered(args)
    assert stopped.value.code == 125


class _Lease:
    script_sha256 = "a" * 64

    def __init__(self, events, *, completion=None, fail_record=False, cleanup_proven=False):
        self.events = events
        self.completion = completion
        self.fail_record = fail_record
        self.cleanup_proven = cleanup_proven

    def parent_alive(self):
        return True

    def stop_requested(self):
        return False

    def record_worker(self, pid):
        self.events.append("record_worker")
        if self.fail_record:
            raise OSError("synthetic durable publication failure")

    def worker_completion_status(self):
        return self.completion

    def terminate_worker_tree(self):
        self.events.append("terminate_tree")
        return self.cleanup_proven

    def mark_quiesced(self, reason, *, cleanup_proven):
        self.events.append(("quiesced", reason, cleanup_proven))

    def mark_unresolved(self, reason):
        self.events.append(("unresolved", reason))

    def release(self):
        self.events.append("release")


def test_failed_worker_registration_keeps_unresolved_receipt(tmp_path, monkeypatch):
    events = []
    args = _arguments(tmp_path, action="supervise")
    lease = _Lease(events, fail_record=True)
    monkeypatch.setattr(host_execution_lease, "claim_host_execution", lambda *a: lease)

    def wait(*, timeout):
        raise subprocess.TimeoutExpired("barriered-worker", timeout)

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, wait=wait, kill=lambda: events.append("kill_unregistered_child")))
    assert host_worker._supervise(args) == 125
    assert events == ["record_worker", "terminate_tree", "kill_unregistered_child", ("unresolved", "guardian_cleanup_unconfirmed"), "release"]


@pytest.mark.parametrize("status,completion", [(0, 0), (1, 1), (-9, None)])
def test_guardian_refuses_independently_exited_worker_even_after_completion(tmp_path, monkeypatch, status, completion):
    events = []
    lease = _Lease(events, completion=completion)
    args = _arguments(tmp_path, action="supervise")
    monkeypatch.setattr(host_execution_lease, "claim_host_execution", lambda *a: lease)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, poll=lambda: status))
    assert host_worker._supervise(args) == 125
    assert events == ["record_worker", ("unresolved", "worker_exit_without_cleanup"), "release"]


@pytest.mark.parametrize("completion", [0, 1, 124])
def test_guardian_stops_live_worker_before_accepting_logical_result(tmp_path, monkeypatch, completion):
    events = []
    lease = _Lease(events, completion=completion, cleanup_proven=True)
    args = _arguments(tmp_path, action="supervise")
    monkeypatch.setattr(host_execution_lease, "claim_host_execution", lambda *a: lease)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, poll=lambda: None, wait=lambda **kw: events.append("worker_dead")))
    assert host_worker._supervise(args) == completion
    assert events == ["record_worker", "terminate_tree", "worker_dead",
                      ("quiesced", "worker_completed" if completion == 0 else "worker_failed", True), "release"]
