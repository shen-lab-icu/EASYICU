"""Invalid IPC requests must not create an unknown remote capacity slot."""

import pytest


@pytest.mark.parametrize("failure", ["too_large", "not_json", "frozen"])
def test_preflight_never_reserves_capacity_or_budget(tmp_path, monkeypatch, failure):
    import easyicu.research_agent.providers.llm  # noqa: F401
    from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
    from easyicu.research_agent.providers.hard_stop import HardStopClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.webserver import metadata_provider_client, metadata_provider_concurrency as slots
    from easyicu.webserver import metadata_provider_worker, provider_adapter

    monkeypatch.setattr(slots.state_paths, "state_root", lambda: tmp_path / "state")
    environment = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
                   "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
                   "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
    provider = {"provider": "openai", "external": True}
    inner, public = provider_adapter.build_research_agent_provider_client(provider, environ=environment, request_timeout=10)
    book = ProviderHardStopLedger(path=tmp_path / "budget.json", task_ids=("task",),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id="task", declaration_sha256="a" * 64)
    task = book.start_task("task")
    slots.reserve(attempt_id="old", account=slots.account_identity(public), task_id="old-task",
                  ledger_path=tmp_path / "old.json", guard_path=tmp_path / "old-guard", worker_sha256="a" * 64)
    if failure == "too_large":
        monkeypatch.setattr(metadata_provider_worker, "_MAX_MESSAGE", 512)
    elif failure == "not_json":
        environment["INVALID"] = float("nan")
    else:
        monkeypatch.setattr(metadata_provider_worker.sys, "frozen", True, raising=False)
    sent = []
    monkeypatch.setattr(metadata_provider_client, "guarded_call", lambda *a, **k: sent.append(1))
    client = metadata_provider_client.MetadataProviderClient(
        inner=inner, task=task, provider=provider, environment=environment, public=public, timeout=10,
    )
    with pytest.raises((ValueError, RuntimeError)):
        HardStopClient(client, role="planner", task=task).complete(
            [LLMMessage(role="user", content="synthetic " * 200)]
        )
    assert sent == []
    assert book.snapshot()["tasks"][0]["calls"] == []
    assert slots.get("old") is not None
    with slots._open() as db:
        assert db.execute("SELECT attempt_id FROM slots").fetchall()[0][0] == "old"
        assert db.execute("SELECT COUNT(*) FROM slots").fetchone()[0] == 1


@pytest.mark.parametrize("broken", [None, "task", "ledger", "path", "request", "worker", "nonce", "receipt_nonce", "receipt_worker", "capacity"])
def test_retry_guard_requires_matching_host_receipt_and_capacity(tmp_path, monkeypatch, broken):
    from types import SimpleNamespace

    from easyicu.runtime import host_execution_lease
    from easyicu.webserver import metadata_provider_client as client, metadata_provider_concurrency as slots

    ledger_path = tmp_path / "budget.json"
    task = SimpleNamespace(task_id="task", ledger=SimpleNamespace(path=ledger_path))
    attempt = "attempt"
    path = ledger_path.parent / ".metadata-provider-execution" / attempt
    call = {"attempt_id": attempt, "state": "in_progress"}
    held = {"task_id": "task", "ledger_path": str(ledger_path), "guard_path": str(path),
            "request_sha256": "request", "worker_sha256": "worker", "guard_nonce": "nonce", "account": "account"}
    receipt = {"nonce": "nonce", "script_sha256": "worker"}
    fields = {"task": "task_id", "ledger": "ledger_path", "path": "guard_path",
              "request": "request_sha256", "worker": "worker_sha256", "nonce": "guard_nonce"}
    if broken in fields:
        held[fields[broken]] = "" if broken == "nonce" else "different"
    if broken == "receipt_nonce":
        receipt["nonce"] = "different"
    if broken == "receipt_worker":
        receipt["script_sha256"] = "different"
    monkeypatch.setattr(slots, "get", lambda value: held if value == attempt else None)
    monkeypatch.setattr(slots, "has_capacity", lambda account: account == "account" and broken != "capacity")
    monkeypatch.setattr(client, "identity", lambda: "worker")
    monkeypatch.setattr(host_execution_lease, "assert_host_execution_quiescent", lambda *a, **k: None)
    monkeypatch.setattr(host_execution_lease, "read_host_execution_receipt", lambda _path: receipt)
    if broken is None:
        assert client.verify_outline_retry_guard(task, call, "request") is None
    else:
        with pytest.raises(ValueError, match="retry_journal_guard_unverified"):
            client.verify_outline_retry_guard(task, call, "request")
