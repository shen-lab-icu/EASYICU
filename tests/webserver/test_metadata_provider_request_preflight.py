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
