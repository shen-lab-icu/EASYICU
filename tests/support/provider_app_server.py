"""Shared synthetic fixtures for provider and resume contracts."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from easyicu.research_agent.providers.capabilities import user_account_profile
from easyicu.research_agent.providers.factory import authorize_provider_client
from easyicu.research_agent.providers.llm import CodexAppServerLLMClient
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

