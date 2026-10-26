"""Startup diagnostics never replace cleanup proof or expose a lease nonce."""

from types import SimpleNamespace
import subprocess

import pytest

from easyicu.research_agent.execution import host_runner
from easyicu.runtime.host_execution_lease import HostExecutionLeaseError


@pytest.mark.parametrize("receipt_state", ["prepared", "running", "unreadable"])
def test_failed_guardian_still_refuses_collection_and_logs_only_before_admission(
    tmp_path, monkeypatch, caplog, receipt_state,
):
    nonce = "a" * 32
    launch = SimpleNamespace(path=tmp_path, nonce=nonce, mark_spawning=lambda: None)
    runner = object.__new__(host_runner.HostRunner)
    runner._host_launch = launch
    receipt = {
        "nonce": nonce, "state": receipt_state,
        "worker": {"pid": 123} if receipt_state == "running" else None,
    }
    error = HostExecutionLeaseError("host_execution_unresolved")

    def refuse(*args, **kwargs):
        raise error

    def read_receipt(path):
        if receipt_state == "unreadable":
            raise OSError("unreadable receipt")
        return receipt

    monkeypatch.setattr(host_runner, "assert_host_execution_quiescent", refuse)
    monkeypatch.setattr(host_runner, "read_host_execution_receipt", read_receipt)
    monkeypatch.setattr(
        host_runner, "_run_capturing_with_descendant_reaping",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            ["worker"], 125, "", f"HostExecutionLeaseError: startup_failed {nonce}",
        ),
    )
    with pytest.raises(HostExecutionLeaseError) as caught:
        runner._execute_command(["worker"], timeout=1)
    assert caught.value is error
    assert nonce not in caplog.text
    assert ("startup_failed" in caplog.text) is (receipt_state == "prepared")
    assert ("exit=125" in caplog.text) is (receipt_state == "prepared")
    assert not list(tmp_path.iterdir()), "An unsettled run must not be written"
