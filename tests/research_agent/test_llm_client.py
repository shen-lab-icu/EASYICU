from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def _mock_transport_client(
    monkeypatch,
    client_type,
    *,
    model: str,
    completions=None,
    max_retries=0,
):
    # These tests exercise the concrete adapter against an in-memory fake SDK,
    # but the adapter itself must go through its real constructor so provider
    # authority cannot be minted for an ``object.__new__`` pseudo-instance.
    transport = SimpleNamespace(
        chat=SimpleNamespace(completions=completions or SimpleNamespace())
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **_kwargs: transport),
    )
    from easyicu.research_agent.providers.factory import build_provider_client

    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setenv("EASYICU_LLM_MAX_RETRIES", str(max_retries))
    return build_provider_client(
        provider="openai",
        model=model,
        base_url_override="http://127.0.0.1:8787/v1",
        request_timeout=1.0,
        title="EasyICU provider adapter test",
        client_cls=client_type,
    )


def _retry_test_client(monkeypatch, failures):
    from easyicu.research_agent.providers.llm import OpenAIClient

    class _Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if failures:
                raise failures.pop(0)
            message = SimpleNamespace(content="OK")
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=None)

    completions = _Completions()
    client = _mock_transport_client(
        monkeypatch,
        OpenAIClient,
        model="gpt-test",
        completions=completions,
        max_retries=2,
    )
    return client, completions


def test_openai_client_disables_sdk_retries(monkeypatch, ra):
    from easyicu.research_agent.providers.llm import OpenAIClient

    captured = {}

    class _OpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))

    client = OpenAIClient(
        model="gpt-test",
        api_key="test-key",
        base_url="https://provider.invalid/v1",
        max_retries=5,
    )

    assert captured["max_retries"] == 0
    assert client._max_retries == 5


def test_openai_client_can_freeze_transport_environment(monkeypatch, ra):
    from easyicu.research_agent.providers.llm import OpenAIClient

    captured = {}

    class _OpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("EASYICU_LLM_TIMEOUT", "999")
    monkeypatch.setenv("EASYICU_LLM_MAX_RETRIES", "77")
    monkeypatch.setenv("EASYICU_LLM_STREAM", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://mutated.example/v1")

    client = OpenAIClient(
        model="frozen-model",
        api_key="test-key",
        base_url="https://frozen.example/v1",
        request_timeout=13.0,
        max_retries=2,
        stream_enabled=False,
        allow_environment_overrides=False,
    )

    assert client._timeout == 13.0
    assert client._max_retries == 2
    assert client._stream_enabled is False
    assert client._resolved_base_url == "https://frozen.example/v1"
    assert str(captured["base_url"]) == "https://frozen.example/v1"


def test_unmanaged_external_client_is_rejected_before_transport(ra):
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient

    calls = 0

    class _Completions:
        def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("unauthorized messages must never reach transport")

    client = OpenAIClient.__new__(OpenAIClient)
    client._resolved_base_url = "https://provider.invalid/v1"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    client._model = "gpt-test"
    client._timeout = 1.0
    client._extra_body = {}

    with pytest.raises(PermissionError, match="factory-minted"):
        client.complete([LLMMessage(role="user", content="patient free text")])

    assert calls == 0


def test_openai_client_strips_reasoning_blocks_from_content(monkeypatch, ra):
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient

    class _Completions:
        def create(self, **kwargs):
            assert "stream" not in kwargs
            msg = SimpleNamespace(
                content='<think>private chain of thought</think>\n{"ok": true}'
            )
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=None)

    client = _mock_transport_client(
        monkeypatch,
        OpenAIClient,
        model="qwen3-8b",
        completions=_Completions(),
    )

    out, call_usage = client.complete_with_usage(
        [LLMMessage(role="user", content="return json")]
    )

    assert out == '{"ok": true}'
    assert call_usage is None


def test_openai_client_streaming_is_transport_only(monkeypatch, ra):
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient

    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=4,
        total_tokens=16,
    )
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content='{"ok": ', reasoning=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="true}", reasoning=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        ),
        SimpleNamespace(choices=[], usage=usage),
    ]

    class _Stream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return iter(chunks)

        def close(self):
            self.closed = True

    stream = _Stream()

    class _Completions:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            assert "stream_options" not in kwargs
            return stream

    monkeypatch.setenv("EASYICU_LLM_STREAM", "1")
    client = _mock_transport_client(
        monkeypatch,
        OpenAIClient,
        model="gpt-5.6-luna",
        completions=_Completions(),
    )

    out, call_usage = client.complete_with_usage(
        [LLMMessage(role="user", content="return json")]
    )

    assert out == '{"ok": true}'
    assert client.last_finish_reason == "stop"
    assert client.last_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }
    assert call_usage == client.last_usage
    assert stream.closed is True


def test_fallback_client_preserves_call_scoped_usage(ra):
    from easyicu.research_agent.providers.llm import FallbackLLMClient, LLMMessage

    class _UsageClient:
        name = "usage-child"

        def complete_with_usage(
            self, messages, *, max_tokens=2048, temperature=0.2, seed=None
        ):
            return "done", {
                "prompt_tokens": 17,
                "completion_tokens": 4,
                "total_tokens": 21,
            }

    client = FallbackLLMClient(_UsageClient())
    response, usage = client.complete_with_usage(
        [LLMMessage(role="user", content="run")], seed=9
    )

    assert response == "done"
    assert usage == {
        "prompt_tokens": 17,
        "completion_tokens": 4,
        "total_tokens": 21,
    }
    assert client.last_usage == usage


def test_openai_client_stream_closes_on_iteration_error(monkeypatch, ra):
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient

    class _Stream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield SimpleNamespace(choices=[], usage=None)
            raise ValueError("broken stream")

        def close(self):
            self.closed = True

    stream = _Stream()

    class _Completions:
        def create(self, **kwargs):
            return stream

    monkeypatch.setenv("EASYICU_LLM_STREAM", "1")
    client = _mock_transport_client(
        monkeypatch,
        OpenAIClient,
        model="gpt-5.6-luna",
        completions=_Completions(),
    )

    with pytest.raises(ValueError, match="broken stream"):
        client.complete([LLMMessage(role="user", content="return json")])

    assert stream.closed is True


def test_openai_client_recovers_unclosed_reasoning_prefix_for_debuggability(
    monkeypatch, ra
):
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient

    class _Completions:
        def create(self, **kwargs):
            msg = SimpleNamespace(content="<think>still thinking")
            choice = SimpleNamespace(message=msg, finish_reason="length")
            return SimpleNamespace(choices=[choice], usage=None)

    client = _mock_transport_client(
        monkeypatch,
        OpenAIClient,
        model="qwen3-8b",
        completions=_Completions(),
    )

    out = client.complete([LLMMessage(role="user", content="return json")])

    assert out == "still thinking"


def test_openai_client_supports_local_noauth_proxy_mode(monkeypatch, ra):
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient
    import httpx

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "OK", "role": "assistant"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            }

    class _HttpClient:
        def __init__(self):
            self.calls = []

        def post(self, path, json):
            self.calls.append((path, json))
            return _Response()

    http_client = _HttpClient()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: (http_client if kwargs.get("base_url") else SimpleNamespace()),
    )
    client = OpenAIClient(
        model="gpt-5.4",
        base_url="http://127.0.0.1:8787/v1",
        request_timeout=1.0,
        max_retries=0,
    )

    out = client.complete(
        [LLMMessage(role="user", content="Reply with the single word OK.")],
        seed=7,
    )

    assert out == "OK"
    assert http_client.calls[0][0] == "/chat/completions"
    assert http_client.calls[0][1]["model"] == "gpt-5.4"
    assert http_client.calls[0][1]["seed"] == 7
    assert http_client.calls[0][1]["max_completion_tokens"] == 2048
    assert "max_tokens" not in http_client.calls[0][1]
    assert client.last_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }


def test_openai_client_uses_legacy_token_cap_for_non_reasoning_model(
    monkeypatch, ra
):
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="OK")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=message, finish_reason="stop")
                ],
                usage=None,
            )

    client = _mock_transport_client(
        monkeypatch,
        OpenAIClient,
        model="gpt-4o-mini",
        completions=_Completions(),
    )

    assert client.complete(
        [LLMMessage(role="user", content="return OK")],
        max_tokens=77,
    ) == "OK"
    assert captured["max_tokens"] == 77
    assert "max_completion_tokens" not in captured


def test_openai_client_zero_manual_retry_budget_makes_one_attempt(monkeypatch, ra):
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient

    class _Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(choices=[], usage=None)

    completions = _Completions()
    client = _mock_transport_client(
        monkeypatch,
        OpenAIClient,
        model="gpt-5.6-luna",
        completions=completions,
        max_retries=0,
    )
    monkeypatch.delenv("EASYICU_LLM_STREAM", raising=False)
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="LLM_TRANSIENT_NO_CHOICES"):
        client.complete([LLMMessage(role="user", content="return json")])

    assert completions.calls == 1
    assert sleeps == []


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 502, 503, 504])
def test_openai_client_manual_owner_retries_transient_http_status(
    monkeypatch, ra, status_code
):
    from easyicu.research_agent.providers.llm import LLMMessage

    error = RuntimeError(f"provider rejected request with status {status_code}")
    error.status_code = status_code
    client, completions = _retry_test_client(monkeypatch, [error])
    sleeps = []
    monkeypatch.delenv("EASYICU_LLM_STREAM", raising=False)
    monkeypatch.setattr("time.sleep", sleeps.append)

    result = client.complete([LLMMessage(role="user", content="return OK")])

    assert result == "OK"
    assert completions.calls == 2
    assert len(sleeps) == 1


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("connection reset by peer"),
        TimeoutError("connection timed out"),
        RuntimeError("HTTP 502 Bad Gateway"),
        RuntimeError("too many requests"),
    ],
)
def test_openai_client_manual_owner_recognizes_generic_transient_errors(
    monkeypatch, ra, error
):
    from easyicu.research_agent.providers.llm import LLMMessage

    client, completions = _retry_test_client(monkeypatch, [error])
    sleeps = []
    monkeypatch.delenv("EASYICU_LLM_STREAM", raising=False)
    monkeypatch.setattr("time.sleep", sleeps.append)

    assert client.complete([LLMMessage(role="user", content="return OK")]) == "OK"
    assert completions.calls == 2
    assert len(sleeps) == 1


@pytest.mark.parametrize("retry_after", ["nan", "inf", "-1"])
def test_openai_client_ignores_invalid_retry_after_values(monkeypatch, ra, retry_after):
    from easyicu.research_agent.providers.llm import LLMMessage

    error = RuntimeError("provider rejected request with status 503")
    error.status_code = 503
    error.response = SimpleNamespace(headers={"Retry-After": retry_after})
    client, completions = _retry_test_client(monkeypatch, [error])
    sleeps = []
    monkeypatch.delenv("EASYICU_LLM_STREAM", raising=False)
    monkeypatch.setattr("time.sleep", sleeps.append)

    result = client.complete([LLMMessage(role="user", content="return OK")])

    assert result == "OK"
    assert completions.calls == 2
    assert sleeps == [5.0]


def test_openai_client_transport_retry_consumes_provider_budget(monkeypatch, ra):
    from easyicu.research_agent.providers.llm import LLMMessage
    from easyicu.research_agent.authority.provider_budget import (
        StepProviderCallBudget,
        complete_with_provider_budget,
    )

    error = RuntimeError("upstream response")
    error.status_code = 500
    client, completions = _retry_test_client(monkeypatch, [error])
    budget = StepProviderCallBudget(2, step_id="analysis")
    monkeypatch.delenv("EASYICU_LLM_STREAM", raising=False)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = complete_with_provider_budget(
        budget=budget,
        category="coder",
        call=lambda: client.complete([LLMMessage(role="user", content="return OK")]),
    )

    assert result == "OK"
    assert completions.calls == 2
    assert budget.categories == ("coder", "coder")


def test_openai_client_does_not_sleep_or_call_over_provider_budget(monkeypatch, ra):
    from easyicu.research_agent.providers.llm import LLMMessage
    from easyicu.research_agent.authority.provider_budget import (
        ProviderCallBudgetExhausted,
        StepProviderCallBudget,
        complete_with_provider_budget,
    )

    error = RuntimeError("upstream response")
    error.status_code = 504
    client, completions = _retry_test_client(monkeypatch, [error])
    budget = StepProviderCallBudget(1, step_id="analysis")
    sleeps = []
    monkeypatch.delenv("EASYICU_LLM_STREAM", raising=False)
    monkeypatch.setattr("time.sleep", sleeps.append)

    with pytest.raises(ProviderCallBudgetExhausted):
        complete_with_provider_budget(
            budget=budget,
            category="coder",
            call=lambda: client.complete(
                [LLMMessage(role="user", content="return OK")]
            ),
        )

    assert completions.calls == 1
    assert budget.categories == ("coder",)
    assert sleeps == []
