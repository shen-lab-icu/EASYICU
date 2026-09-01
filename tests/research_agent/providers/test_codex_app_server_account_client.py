from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.research_agent.providers.capabilities import user_account_profile
from easyicu.research_agent.providers.factory import (
    authorize_provider_client,
    provider_authorization_manifest,
)
from easyicu.research_agent.providers.llm import CodexAppServerLLMClient
from easyicu.research_agent.providers.protocol import (
    LLMMessage,
    StructuredOutputRequest,
)


SESSION_SHA256 = "a" * 64


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "CODEX_HOME": str(tmp_path / "codex"),
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        "EASYICU_CODEX_SESSION_SHA256": SESSION_SHA256,
        "OPENAI_API_KEY": "must-not-cross-user-account-boundary",
    }


def _authorized_client(
    tmp_path: Path,
    *,
    request_timeout: float = 10,
    turn_hard_timeout: float | None = None,
    reasoning_effort: str | None = None,
) -> CodexAppServerLLMClient:
    environment = _environment(tmp_path)
    client_kwargs: dict[str, Any] = {
        "environment": environment,
        "request_timeout": request_timeout,
    }
    if turn_hard_timeout is not None:
        client_kwargs["turn_hard_timeout"] = turn_hard_timeout
    if reasoning_effort is not None:
        client_kwargs["reasoning_effort"] = reasoning_effort
    client = CodexAppServerLLMClient(
        **client_kwargs,
    )
    profile = user_account_profile("codex")
    assert profile is not None
    return authorize_provider_client(
        client,
        provider=profile.provider_identity,
        model="account-default",
        base_url=f"{profile.endpoint_identity}/session/{SESSION_SHA256}",
        destination="external",
        environment=environment,
    )


class _FakeRuntime:
    instances: list["_FakeRuntime"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.wait_kwargs: dict[str, Any] = {}
        self.__class__.instances.append(self)

    def __enter__(self) -> "_FakeRuntime":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @property
    def notification_count(self) -> int:
        return 0

    def request(
        self,
        method: str,
        params: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "account/read":
            return {
                "account": {
                    "type": "chatgpt",
                    "email": "person@example.org",
                    "planType": "plus",
                }
            }
        if method == "thread/start":
            return {
                "thread": {"id": "thread_1"},
                "model": "gpt-test",
            }
        if method == "turn/start":
            return {"turn": {"id": "turn_1"}}
        raise AssertionError(method)

    def wait_for_notification(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self.wait_kwargs = dict(_kwargs)
        return {
            "method": "turn/completed",
            "params": {
                "threadId": "thread_1",
                "turn": {
                    "id": "turn_1",
                    "status": "completed",
                    "items": [
                        {
                            "id": "item_1",
                            "type": "agentMessage",
                            "text": '{"answer":"ok"}',
                        }
                    ],
                },
            },
        }

    def notifications_since(self, _index: int) -> list[dict[str, Any]]:
        return [
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "thread_1",
                    "turnId": "turn_1",
                    "tokenUsage": {
                        "last": {
                            "inputTokens": 11,
                            "outputTokens": 7,
                            "totalTokens": 18,
                        }
                    },
                },
            }
        ]


def test_codex_app_server_client_uses_chatgpt_account_and_strict_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers import codex_app_server

    _FakeRuntime.instances.clear()
    monkeypatch.setattr(codex_app_server, "CodexAppServerRuntime", _FakeRuntime)
    client = _authorized_client(tmp_path, reasoning_effort="medium")
    structured = StructuredOutputRequest.from_schema(
        name="answer",
        schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    text, usage = client.complete_with_usage(
        [
            LLMMessage(role="system", content="Return JSON."),
            LLMMessage(role="user", content="Answer the question."),
        ],
        structured_output=structured,
    )

    assert json.loads(text) == {"answer": "ok"}
    assert usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    runtime = _FakeRuntime.instances[0]
    assert runtime.kwargs["environment"]["CODEX_HOME"] == str(tmp_path / "codex")
    assert "OPENAI_API_KEY" not in runtime.kwargs["environment"]
    assert runtime.kwargs["experimental_api"] is True
    methods = [method for method, _params in runtime.calls]
    assert methods == ["account/read", "thread/start", "turn/start"]
    thread = runtime.calls[1][1]
    assert thread["ephemeral"] is True
    assert thread["approvalPolicy"] == "never"
    assert thread["sandbox"] == "read-only"
    turn = runtime.calls[2][1]
    assert turn["sandboxPolicy"] == {"type": "readOnly", "networkAccess": False}
    assert turn["effort"] == "medium"
    assert turn["outputSchema"] == json.loads(structured.schema_json)
    assert runtime.wait_kwargs["timeout"] == 10
    assert runtime.wait_kwargs["hard_timeout"] == 10
    progress_predicate = runtime.wait_kwargs["progress_predicate"]
    assert progress_predicate(
        {
            "method": "item/agentMessage/delta",
            "params": {"threadId": "thread_1", "turnId": "turn_1", "delta": "{"},
        }
    )
    assert not progress_predicate(
        {
            "method": "item/agentMessage/delta",
            "params": {"threadId": "thread_1", "turnId": "other", "delta": "{"},
        }
    )


def test_codex_turn_timeouts_are_capped_by_the_task_hard_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers import codex_app_server
    from easyicu.research_agent.providers import llm

    _FakeRuntime.instances.clear()
    monkeypatch.setattr(codex_app_server, "CodexAppServerRuntime", _FakeRuntime)
    monkeypatch.setattr(
        llm,
        "consume_active_transport_attempt",
        lambda: 4.0,
    )
    client = _authorized_client(
        tmp_path,
        request_timeout=10,
        turn_hard_timeout=60,
    )

    assert client.complete([LLMMessage(role="user", content="hello")])

    runtime = _FakeRuntime.instances[0]
    assert client.provider_attempt_budget_aware is True
    assert runtime.wait_kwargs["timeout"] == pytest.approx(4.0, abs=0.01)
    assert runtime.wait_kwargs["hard_timeout"] == pytest.approx(4.0, abs=0.01)


def test_codex_turn_start_time_is_deducted_from_the_task_hard_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers import codex_app_server
    from easyicu.research_agent.providers import llm

    _FakeRuntime.instances.clear()
    monkeypatch.setattr(codex_app_server, "CodexAppServerRuntime", _FakeRuntime)
    monkeypatch.setattr(
        llm,
        "consume_active_transport_attempt",
        lambda: 4.0,
    )
    ticks = iter((100.0, 103.0))
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(ticks))
    client = _authorized_client(
        tmp_path,
        request_timeout=10,
        turn_hard_timeout=60,
    )

    assert client.complete([LLMMessage(role="user", content="hello")])

    runtime = _FakeRuntime.instances[0]
    assert runtime.wait_kwargs["timeout"] == 1.0
    assert runtime.wait_kwargs["hard_timeout"] == 1.0


def test_codex_turn_does_not_wait_after_the_task_hard_stop_expires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers import codex_app_server
    from easyicu.research_agent.providers import llm

    _FakeRuntime.instances.clear()
    monkeypatch.setattr(codex_app_server, "CodexAppServerRuntime", _FakeRuntime)
    monkeypatch.setattr(
        llm,
        "consume_active_transport_attempt",
        lambda: 4.0,
    )
    ticks = iter((100.0, 105.0))
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(ticks))
    client = _authorized_client(
        tmp_path,
        request_timeout=10,
        turn_hard_timeout=60,
    )

    with pytest.raises(
        codex_app_server.CodexAppServerError,
        match="codex_auth_notification_hard_timeout",
    ):
        client.complete([LLMMessage(role="user", content="hello")])

    assert _FakeRuntime.instances[0].wait_kwargs == {}


def test_codex_app_server_authorization_binds_the_user_session(tmp_path: Path) -> None:
    manifest = provider_authorization_manifest(_authorized_client(tmp_path))
    authority = manifest["clients"][0]

    assert authority["provider"] == "codex-app-server"
    assert authority["authorization_mode"] == "account_session"
    assert authority["base_url"].endswith("/session/" + SESSION_SHA256)
    assert str(tmp_path) not in json.dumps(manifest)


def test_codex_app_server_client_rejects_missing_session_binding(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment.pop("EASYICU_CODEX_SESSION_SHA256")

    with pytest.raises(ValueError, match="codex_auth_session_binding_required"):
        CodexAppServerLLMClient(environment=environment)


def test_unmanaged_codex_app_server_client_is_rejected_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.research_agent.providers import codex_app_server

    _FakeRuntime.instances.clear()
    monkeypatch.setattr(codex_app_server, "CodexAppServerRuntime", _FakeRuntime)
    client = CodexAppServerLLMClient(environment=_environment(tmp_path))

    with pytest.raises(PermissionError, match="factory-minted user authorization"):
        client.complete([LLMMessage(role="user", content="hello")])
    assert _FakeRuntime.instances == []


def test_authorized_codex_client_rejects_hard_timeout_mutation(
    tmp_path: Path,
) -> None:
    client = _authorized_client(
        tmp_path,
        request_timeout=10,
        turn_hard_timeout=60,
    )
    client._turn_hard_timeout = 600.0

    with pytest.raises(PermissionError, match="factory-minted user authorization"):
        client.complete([LLMMessage(role="user", content="hello")])


def test_authorized_codex_client_rejects_reasoning_effort_mutation(
    tmp_path: Path,
) -> None:
    client = _authorized_client(tmp_path, reasoning_effort="medium")
    client._reasoning_effort = "high"

    with pytest.raises(PermissionError, match="factory-minted user authorization"):
        client.complete([LLMMessage(role="user", content="hello")])


def test_codex_client_rejects_unknown_reasoning_effort(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reasoning_effort is not supported"):
        CodexAppServerLLMClient(
            environment=_environment(tmp_path),
            reasoning_effort="fastest",
        )
