"""Local admission denial never manufactures a remote usage result."""

import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from easyicu.research_agent.authority.provider_hard_stop import load_provider_hard_stop_ledger


_CHILD = r'''
import os
from pathlib import Path
import sys

import easyicu.research_agent.providers.llm  # register reviewed transports
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger, ProviderHardStopLedgerError
from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.webserver import metadata_provider_client as client_owner, metadata_provider_concurrency as slots, provider_adapter

root, phase = Path(sys.argv[1]), sys.argv[2]
environment = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
               "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
               "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
provider = {"provider": "openai", "external": True}
inner, public = provider_adapter.build_research_agent_provider_client(provider, environ=environment, request_timeout=10)
book = ProviderHardStopLedger(path=root / "budget.json", task_ids=("task",),
    limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
    batch_id="task", declaration_sha256="a" * 64)
task = book.start_task("task")
client = client_owner.MetadataProviderClient(inner=inner, task=task, provider=provider,
    environment=environment, public=public, timeout=10)
if phase != "provider_impostor":
    slots.reserve(attempt_id="old", account=slots.account_identity(public), task_id="old-task",
        ledger_path=root / "old.json", guard_path=root / "old-guard", worker_sha256="a" * 64)
    client_owner.guarded_call = lambda *args, **kwargs: os._exit(99)
else:
    def impostor(*args, **kwargs):
        raise slots.MetadataProviderCapacityExhausted("not a local admission result")
    client_owner.guarded_call = impostor
if phase in {"before", "after"}:
    actual = client_owner.record_active_metadata_capacity_rejection
    def interrupt():
        if phase == "after":
            actual()
        os._exit(73)
    client_owner.record_active_metadata_capacity_rejection = interrupt
elif phase in {"write_failure", "write_failure_after_commit"}:
    actual_persist = book._persist_locked
    def persist():
        calls = book._payload["tasks"][0]["calls"]
        if calls and calls[-1].get("state") == "failed_pre_transport_capacity":
            if phase == "write_failure_after_commit":
                actual_persist()
            raise OSError("synthetic admission receipt write failure")
        return actual_persist()
    book._persist_locked = persist
try:
    HardStopClient(client, role="planner", task=task).complete([LLMMessage(role="user", content="synthetic")])
except slots.MetadataProviderCapacityExhausted:
    if phase == "provider_impostor":
        sys.exit(75)
    raise
except ProviderHardStopLedgerError:
    if phase in {"write_failure", "write_failure_after_commit"}:
        sys.exit(74)
    raise
sys.exit(98)
'''


@pytest.mark.parametrize(("phase", "exitcode", "expected_state"), [
    ("before", 73, "in_progress"),
    ("after", 73, "failed_pre_transport_capacity"),
    ("write_failure", 74, "failed_usage_unknown"),
    ("write_failure_after_commit", 74, "failed_pre_transport_capacity"),
])
def test_process_interrupt_or_failed_receipt_never_releases_old_slot(
    tmp_path: Path, phase: str, exitcode: int, expected_state: str,
):
    env = {**os.environ, "EASYICU_HOME": str(tmp_path / "home"),
           "EASYICU_METADATA_PROVIDER_MAX_INFLIGHT": "1"}
    result = subprocess.run([sys.executable, "-c", _CHILD, str(tmp_path), phase], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == exitcode, result.stderr
    payload = load_provider_hard_stop_ledger(tmp_path / "budget.json")
    call = payload["tasks"][0]["calls"][0]
    assert call["state"] == expected_state
    assert call["accounted_tokens"] > 0
    assert call.get("reported_total_tokens") is None
    with sqlite3.connect(tmp_path / "home/.easyicu/metadata-provider-slots.sqlite3") as connection:
        assert connection.execute("select attempt_id from slots").fetchall() == [("old",)]


def test_transport_exception_cannot_claim_the_host_rejected_it_before_send(tmp_path: Path):
    env = {**os.environ, "EASYICU_HOME": str(tmp_path / "home"),
           "EASYICU_METADATA_PROVIDER_MAX_INFLIGHT": "1"}
    result = subprocess.run([sys.executable, "-c", _CHILD, str(tmp_path), "provider_impostor"],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 75, result.stderr
    call = load_provider_hard_stop_ledger(tmp_path / "budget.json")["tasks"][0]["calls"][0]
    assert call["state"] == "failed_usage_unknown"
    assert call["accounted_tokens"] > 0
    with sqlite3.connect(tmp_path / "home/.easyicu/metadata-provider-slots.sqlite3") as connection:
        assert connection.execute("select count(*) from slots").fetchone()[0] == 1
