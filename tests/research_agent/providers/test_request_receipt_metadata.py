import json

from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from tests.support.provider_recovery import _limits


def _ledger(tmp_path, *, task_ids):
    return ProviderHardStopLedger(path=tmp_path / "provider_progress.json", task_ids=task_ids,
                                  limits=_limits(), batch_id="synthetic-batch")


def test_request_metadata_is_durable_without_storing_the_prompt(tmp_path):
    ledger = _ledger(tmp_path, task_ids=("synthetic",))
    client = HardStopClient(ScriptedMockLLMClient(["ok"]), role="planner",
                            task=ledger.start_task("synthetic"))
    content = "Synthetic private text 文"
    client.complete([LLMMessage(role="user", content=content)], max_tokens=16)
    recorded = ledger.snapshot()["tasks"][0]["calls"][0]
    assert recorded["provider"] == "mock"
    assert recorded["data_classes"] == ["text"]
    assert recorded["redaction_status"] == "not_recorded"
    assert recorded["prompt_payload_bytes"] == len(content.encode())
    assert content not in json.dumps(ledger.snapshot(), ensure_ascii=False)


def test_transport_retry_records_the_current_provider_and_schema_bytes(tmp_path):
    from easyicu.research_agent.authority.provider_hard_stop import provider_hard_stop_call_scope
    from easyicu.research_agent.authority.provider_budget import consume_active_transport_attempt
    ledger = _ledger(tmp_path, task_ids=("synthetic",))
    with provider_hard_stop_call_scope(
        task=ledger.start_task("synthetic"), role="writer", model="synthetic",
        provider="openai", messages=[LLMMessage(role="user", content="文")],
        max_tokens=16, additional_prompt_payload_bytes=7,
    ) as scope:
        consume_active_transport_attempt()
        consume_active_transport_attempt(provider="anthropic")
        scope.complete(None)
    first, second = ledger.snapshot()["tasks"][0]["calls"]
    assert first["provider"] == "openai" and second["provider"] == "anthropic"
    assert first["data_classes"] == second["data_classes"] == ["text", "schema"]
    assert first["prompt_payload_bytes"] == second["prompt_payload_bytes"] == 10


def test_provider_manifest_projection_never_discloses_a_custom_endpoint():
    from easyicu.model_request_metadata import provider_name
    assert provider_name({"clients": [{"provider": "openrouter", "base_url": "/private"}]}) == "openrouter"
    assert provider_name({"clients": [{"provider": "custom-secret"}]}) == "unknown"
    assert provider_name({"clients": [{"provider": "openai"}, {"provider": "anthropic"}]}) == "unknown"


def test_fallback_receipts_identify_each_transport_not_the_wrapper(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from easyicu.research_agent.providers.factory import build_provider_client
    from easyicu.research_agent.providers.llm import FallbackLLMClient

    def create(**kwargs):
        if kwargs["model"] == "first":
            raise ConnectionError("Synthetic transport failure")
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok"), finish_reason="stop")], usage=None)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)))))
    clients = [build_provider_client(
        provider=provider, model=model, title="synthetic", request_timeout=1,
        environment={"OPENAI_BASE_URL": "http://127.0.0.1:8787/v1",
                     "CUSTOM_BASE_URL": "http://127.0.0.1:8787/v1",
                     "EASYICU_LLM_API_KEY": "synthetic-key"},
        max_retries=0, stream_enabled=False, allow_environment_overrides=False,
    ) for provider, model in [("openai", "first"), ("custom", "second")]]
    ledger = _ledger(tmp_path, task_ids=("synthetic",))
    wrapped = HardStopClient(FallbackLLMClient(*clients), role="repair", task=ledger.start_task("synthetic"))
    assert wrapped.complete([LLMMessage(role="user", content="Synthetic request")], max_tokens=16) == "ok"
    assert [row["provider"] for row in ledger.snapshot()["tasks"][0]["calls"]] == ["openai", "custom"]
