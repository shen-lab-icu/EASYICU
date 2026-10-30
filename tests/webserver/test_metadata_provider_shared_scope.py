"""Metadata capacity is shared only inside an explicitly enrolled deployment."""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from easyicu.runtime.platform_compat import lock_exclusive, unlock_file
from easyicu.webserver import metadata_provider_concurrency as slots
from easyicu.webserver import metadata_provider_shared_scope as shared


def _dirs(tmp_path):
    coordinator = tmp_path / "coordinator"
    coordinator.mkdir(mode=0o700)
    homes = [tmp_path / "one", tmp_path / "two"]
    for home in homes:
        home.mkdir(mode=0o700)
        (home / ".easyicu").mkdir(mode=0o700)
    return coordinator, homes


def _reserve(name):
    slots.reserve(attempt_id=name, account="same", task_id=name,
                  ledger_path=Path("/private/ledger"), guard_path=Path("/private/guard"),
                  worker_sha256="a" * 64)


def test_migration_keeps_unknown_slot_and_rejects_changed_legacy(tmp_path, monkeypatch):
    coordinator, homes = _dirs(tmp_path)
    monkeypatch.setenv("EASYICU_HOME", str(homes[0]))
    _reserve("unknown-old")
    original = homes[0] / ".easyicu" / shared._LOCAL
    assert shared.initialize_scope(coordinator, homes) == (2, 1)
    monkeypatch.setenv(shared.SHARED_ROOT_ENV, str(coordinator))
    monkeypatch.setenv("EASYICU_HOME", str(homes[1]))
    assert not slots.has_capacity("same")
    assert slots.get("unknown-old")["task_id"] == "unknown-old"
    with pytest.raises(RuntimeError, match="terminal_proof"):
        slots.release("unknown-old")
    with sqlite3.connect(original) as db:
        assert db.execute("select attempt_id from slots").fetchall() == [("unknown-old",)]
        db.execute("delete from slots")  # simulate an old version changing its ledger
    # A legacy mutation in home one also blocks requests from home two.
    monkeypatch.setenv("EASYICU_HOME", str(homes[1]))
    with pytest.raises(shared.SharedScopeError, match="legacy_changed"):
        slots.has_capacity("same")
    with sqlite3.connect(coordinator / shared._DB) as db:
        assert db.execute("select attempt_id from slots").fetchall() == [("unknown-old",)]


def test_changed_slots_after_operator_preview_cannot_be_enrolled(tmp_path, monkeypatch):
    coordinator, homes = _dirs(tmp_path)
    reviewed = shared.preview_scope(coordinator, homes)
    monkeypatch.setenv("EASYICU_HOME", str(homes[0]))
    _reserve("arrived-after-preview")
    with pytest.raises(shared.SharedScopeError, match="changed_since_review"):
        shared.initialize_scope(coordinator, homes, expected_preview=reviewed)
    assert not (coordinator / shared._DB).exists()
    assert slots.get("arrived-after-preview") is not None


def test_unenrolled_home_and_uninitialized_scope_never_fall_back(tmp_path, monkeypatch):
    coordinator, homes = _dirs(tmp_path)
    monkeypatch.setenv("EASYICU_HOME", str(homes[0]))
    monkeypatch.setenv(shared.SHARED_ROOT_ENV, str(coordinator))
    with pytest.raises(shared.SharedScopeError, match="unavailable"):
        _reserve("before-enrollment")
    monkeypatch.delenv(shared.SHARED_ROOT_ENV)
    shared.initialize_scope(coordinator, homes[:1])
    monkeypatch.setenv(shared.SHARED_ROOT_ENV, str(coordinator))
    monkeypatch.setenv("EASYICU_HOME", str(homes[1]))
    with pytest.raises(shared.SharedScopeError, match="unavailable"):
        _reserve("unenrolled")
    assert not (homes[1] / ".easyicu" / shared._LOCAL).exists()


def test_enrolled_home_never_falls_back_when_environment_is_missing_or_wrong(tmp_path, monkeypatch):
    coordinator, homes = _dirs(tmp_path)
    shared.initialize_scope(coordinator, homes)
    monkeypatch.setenv("EASYICU_HOME", str(homes[0]))
    monkeypatch.delenv(shared.SHARED_ROOT_ENV, raising=False)
    with pytest.raises(shared.SharedScopeError, match="configuration_required"):
        _reserve("missing-config")
    monkeypatch.setenv(shared.SHARED_ROOT_ENV, str(homes[1]))
    with pytest.raises(shared.SharedScopeError, match="configuration_required"):
        _reserve("wrong-config")
    assert not (homes[0] / ".easyicu" / shared._LOCAL).exists()
    # A corrupt binding is not interpreted as an unenrolled, local home.
    marker = homes[0] / ".easyicu" / shared._ENROLLMENT
    marker.write_text("{}")
    marker.chmod(0o600)
    with pytest.raises(shared.SharedScopeError, match="enrollment_invalid"):
        _reserve("corrupt-config")


def test_running_instance_blocks_migration_and_crash_cannot_publish_partial_scope(tmp_path, monkeypatch):
    coordinator, homes = _dirs(tmp_path)
    lock = os.open(homes[0] / ".easyicu/webserver-single-process.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        lock_exclusive(lock, blocking=False)
        with pytest.raises(shared.SharedScopeError, match="still_running"):
            shared.initialize_scope(coordinator, homes)
    finally:
        unlock_file(lock)
        os.close(lock)
    assert not (coordinator / shared._DB).exists()
    original_link = os.link
    monkeypatch.setattr(shared.os, "link", lambda *_: (_ for _ in ()).throw(OSError("synthetic crash")))
    with pytest.raises(OSError, match="synthetic crash"):
        shared.initialize_scope(coordinator, homes)
    assert not (coordinator / shared._DB).exists()
    monkeypatch.setattr(shared.os, "link", original_link)
    assert shared.initialize_scope(coordinator, homes) == (2, 0)


def test_separate_processes_in_different_homes_share_one_capacity(tmp_path):
    coordinator, homes = _dirs(tmp_path)
    shared.initialize_scope(coordinator, homes)
    code = r'''
import sys
from pathlib import Path
from easyicu.webserver import metadata_provider_concurrency as slots
print("ready", flush=True)
sys.stdin.readline()
try:
    slots.reserve(attempt_id=sys.argv[1], account="same", task_id="task", ledger_path=Path("/ledger"), guard_path=Path("/guard"), worker_sha256="a" * 64)
    print("admitted", flush=True)
except slots.MetadataProviderCapacityExhausted:
    print("occupied", flush=True)
'''
    workers = []
    try:
        for i, home in enumerate(homes):
            env = {**os.environ, "EASYICU_HOME": str(home), shared.SHARED_ROOT_ENV: str(coordinator),
                   "EASYICU_METADATA_PROVIDER_MAX_INFLIGHT": "1"}
            workers.append(subprocess.Popen([sys.executable, "-c", code, str(i)], env=env,
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, text=True))
        for worker in workers:
            assert worker.stdout.readline().strip() == "ready"
        for worker in workers:
            worker.stdin.write("go\n")
            worker.stdin.flush()
        results = []
        for worker in workers:
            out, err = worker.communicate(timeout=20)
            assert worker.returncode == 0, err
            results.append(out.strip())
        assert sorted(results) == ["admitted", "occupied"]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


def test_cli_refuses_noninteractive_self_attestation(tmp_path):
    coordinator, homes = _dirs(tmp_path)
    result = subprocess.run([sys.executable, "-m", "easyicu.webserver.metadata_provider_shared_scope",
                             str(coordinator), "--home", str(homes[0])], input="ENROLL\n",
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert not (coordinator / shared._DB).exists()


def test_unverified_windows_acl_fails_closed(tmp_path, monkeypatch):
    coordinator, _ = _dirs(tmp_path)
    with monkeypatch.context() as context:
        context.setattr(shared.os, "name", "nt")
        with pytest.raises(shared.SharedScopeError, match="platform_unsupported"):
            shared._private_dir(coordinator)


@pytest.mark.skipif(os.name != "posix", reason="terminal attestation test needs a PTY")
def test_cli_shows_exact_choices_before_local_operator_confirmation(tmp_path):
    import pty

    coordinator, homes = _dirs(tmp_path)
    master, slave = pty.openpty()
    process = None
    try:
        process = subprocess.Popen([sys.executable, "-m", "easyicu.webserver.metadata_provider_shared_scope",
                                    str(coordinator), "--home", str(homes[0]), "--home", str(homes[1])],
                                   stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        os.close(slave)
        slave = -1
        os.write(master, b"ENROLL\n")
        output, error = process.communicate(timeout=20)
        assert process.returncode == 0, error
        assert str(coordinator) in output
        assert all(str(home / ".easyicu") in output for home in homes)
        assert "0 retained legacy request(s)" in output
        assert (coordinator / shared._DB).is_file()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if slave >= 0:
            os.close(slave)
        os.close(master)


def _metadata_client(tmp_path, monkeypatch, coordinator, home):
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.webserver import metadata_provider_client, provider_adapter

    monkeypatch.setenv("EASYICU_HOME", str(home))
    monkeypatch.setenv(shared.SHARED_ROOT_ENV, str(coordinator))
    environment = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
                   "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
                   "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
    provider = {"provider": "openai", "external": True}
    inner, public = provider_adapter.build_research_agent_provider_client(provider, environ=environment, request_timeout=10)
    book = ProviderHardStopLedger(path=tmp_path / "budget.json", task_ids=("task",),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id="task", declaration_sha256="a" * 64)
    task = book.start_task("task")
    client = metadata_provider_client.MetadataProviderClient(inner=inner, task=task, provider=provider,
        environment=environment, public=public, timeout=10)
    return client, book


def test_unverified_shared_scope_fails_before_worker_with_specific_reason(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import agent_pipeline_runs, metadata_provider_client
    from easyicu.webserver.stop_next_actions import next_actions_for_stop

    coordinator, homes = _dirs(tmp_path)
    client, book = _metadata_client(tmp_path, monkeypatch, coordinator, homes[0])
    sent = []
    monkeypatch.setattr(metadata_provider_client, "guarded_call", lambda *a, **k: sent.append(1))
    with pytest.raises(metadata_provider_client.MetadataProviderCoordinationUnavailable) as caught:
        HardStopClient(client, role="planner", task=book.start_task("task")).complete(
            [LLMMessage(role="user", content="synthetic")])
    assert sent == []
    code = agent_pipeline_runs._pipeline_failure_code(caught.value)
    assert code == "research_pipeline_metadata_shared_scope_unavailable"
    assert next_actions_for_stop(code) == []
    call = book.snapshot()["tasks"][0]["calls"][0]
    assert call["state"] == "failed_pre_transport_coordination"
    assert call["accounted_tokens"] > 0
    accounting = book.start_task("task").accounting_summary()
    assert accounting["pre_transport_coordination_rejections"] == 1
    assert accounting["usage_unknown"]["n_calls"] == 0
    assert accounting["conservative_upper_bound"]["total_tokens"] == call["accounted_tokens"]


def test_shared_scope_failure_after_transport_is_not_called_unsent(tmp_path, monkeypatch):
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import agent_pipeline_runs, metadata_provider_client

    coordinator, homes = _dirs(tmp_path)
    shared.initialize_scope(coordinator, homes)
    client, book = _metadata_client(tmp_path, monkeypatch, coordinator, homes[0])
    called = []

    def completed_then_scope_lost(*args, **kwargs):
        kwargs["on_prepared"]("nonce")
        called.append(True)
        (coordinator / shared._DB).unlink()
        return {"ok": True, "text": "synthetic", "usage": {"total_tokens": 1}}

    monkeypatch.setattr(metadata_provider_client, "guarded_call", completed_then_scope_lost)
    with pytest.raises(shared.SharedScopeError) as caught:
        HardStopClient(client, role="planner", task=book.start_task("task")).complete(
            [LLMMessage(role="user", content="synthetic")])
    assert called == [True]
    assert agent_pipeline_runs._pipeline_failure_code(caught.value) == "research_pipeline_execution_failed"
    assert book.snapshot()["tasks"][0]["calls"][0]["state"] == "failed_usage_unknown"


def test_migration_process_crash_before_publication_preserves_legacy_slots(tmp_path, monkeypatch):
    coordinator, homes = _dirs(tmp_path)
    monkeypatch.setenv("EASYICU_HOME", str(homes[0]))
    _reserve("unknown-old")
    code = r'''
import os
from pathlib import Path
import sys
from easyicu.webserver import metadata_provider_shared_scope as shared
shared.os.link = lambda *_: os._exit(77)
shared.initialize_scope(Path(sys.argv[1]), [Path(sys.argv[2]), Path(sys.argv[3])])
'''
    result = subprocess.run([sys.executable, "-c", code, str(coordinator), *(str(h) for h in homes)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 77, result.stderr
    assert not (coordinator / shared._DB).exists()
    with sqlite3.connect(homes[0] / ".easyicu" / shared._LOCAL) as db:
        assert db.execute("select attempt_id from slots").fetchall() == [("unknown-old",)]
    # A partially enrolled home cannot silently revert to its local ledger.
    with pytest.raises(shared.SharedScopeError, match="configuration_required"):
        slots.has_capacity("same")
    # OS locks disappear on process death; a fresh complete migration works.
    assert shared.initialize_scope(coordinator, homes) == (2, 1)
