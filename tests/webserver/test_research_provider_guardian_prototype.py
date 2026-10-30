"""Isolated feasibility test; this does not authorize networked host scripts.

The scientific HostRunner remains an offline, host-rendered-only executor.
This test uses a synthetic request to measure its local lifecycle guarantees;
it does not connect that executor to the production Provider transport.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX process identity and SIGKILL")

_WORK = r'''
import os, time
from pathlib import Path
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.webserver.provider_adapter import web_research_agent_hard_stop_limits

root = Path(os.environ["EASYICU_RUN_DIR"])
book = ProviderHardStopLedger(path=root / "provider_hard_stop_ledger.json", task_ids=("synthetic-task",),
    limits=web_research_agent_hard_stop_limits("planner_canary"), batch_id="synthetic-task",
    declaration_sha256="a" * 64)
book.start_task("synthetic-task")
book.reserve_transport_attempt(task_id="synthetic-task", role="acquisition", model="synthetic",
    messages=[{"role": "user", "content": "synthetic metadata"}], max_tokens=4, prior_attempt_id=None)
# Simulate a request stalled after the durable reservation; no network used.
(root / "synthetic-ready").write_text("ready")
time.sleep(60)
'''

_PARENT = r'''
import sys
from pathlib import Path
from easyicu.research_agent.execution.host_runner import HostRunner
from easyicu.research_agent.execution.step_execution import CodeOrigin

class SyntheticHostRunner(HostRunner):
    def export_validated_runtime_bundle(self):
        return {"schema": "easyicu.host_runtime_preflight/1", "provenance": {"runtime": "host"}}
    @property
    def authority_identity_sha256(self):
        return "1" * 64

root = Path(sys.argv[1])
runner = SyntheticHostRunner(workdir=root, cohort_parquet=root / "cohort.parquet", timeout_seconds=30)
runner.run(step_id="metadata_probe", code=(root / "synthetic-source.py").read_text(),
    code_origin=CodeOrigin.HOST_RENDERED_EXECUTOR)
'''


def test_guardian_cleans_local_fake_provider_but_keeps_unknown_reservation(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import load_provider_hard_stop_ledger
    from easyicu.runtime.host_execution_lease import assert_host_execution_quiescent, read_host_execution_receipt

    root = tmp_path / "run_synthetic"
    root.mkdir()
    (root / "cohort.parquet").write_bytes(b"not patient data")
    (root / "synthetic-source.py").write_text(_WORK)
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, str(root)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        while not (root / "synthetic-ready").exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (root / "synthetic-ready").exists(), "synthetic Provider never reached its reserved request"
        before = load_provider_hard_stop_ledger(root / "provider_hard_stop_ledger.json")
        assert len(before["tasks"][0]["calls"]) == 1
        assert before["tasks"][0]["calls"][0]["state"] == "in_progress"
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        assert parent.returncode == -signal.SIGKILL
        assert_host_execution_quiescent(root / ".host-execution", allow_missing=False, wait_timeout=15)
        receipt = read_host_execution_receipt(root / ".host-execution")
        assert receipt["state"] == "quiesced"
        assert receipt["reason"] == "parent_lost"
        after = load_provider_hard_stop_ledger(root / "provider_hard_stop_ledger.json")
        assert after["tasks"][0]["calls"] == before["tasks"][0]["calls"]
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
