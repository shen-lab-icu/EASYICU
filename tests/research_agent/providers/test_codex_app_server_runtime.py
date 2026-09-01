from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.research_agent.providers import codex_app_server
from easyicu.research_agent.providers.codex_app_server import (
    CodexAppServerError,
    CodexAppServerRuntime,
    resolve_codex_app_server_executable,
)
from easyicu.research_agent.providers.subprocess_env import (
    CODEX_APP_SERVER_EXECUTABLE_ENV,
    build_provider_subprocess_env,
)


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path.resolve()


class _ScriptedClock:
    def __init__(self) -> None:
        self.now = 0.0


class _ScriptedState:
    def __init__(
        self,
        *,
        clock: _ScriptedClock,
        notifications: list[dict[str, Any]],
        events: list[tuple[float, dict[str, Any]]],
    ) -> None:
        self.clock = clock
        self.notifications = notifications
        self.events = list(events)

    def __enter__(self) -> "_ScriptedState":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def wait(self, *, timeout: float) -> None:
        if not self.events:
            self.clock.now += timeout
            return
        delay, notification = self.events[0]
        if delay > timeout:
            self.events[0] = (delay - timeout, notification)
            self.clock.now += timeout
            return
        self.events.pop(0)
        self.clock.now += delay
        self.notifications.append(notification)


def _scripted_runtime(
    *,
    clock: _ScriptedClock,
    events: list[tuple[float, dict[str, Any]]],
) -> CodexAppServerRuntime:
    runtime = object.__new__(CodexAppServerRuntime)
    runtime._notifications = []
    runtime._notification_offset = 0
    runtime._process = SimpleNamespace(poll=lambda: None)
    runtime._state = _ScriptedState(
        clock=clock,
        notifications=runtime._notifications,
        events=events,
    )
    return runtime


def test_explicit_codex_runtime_wins_over_app_bundle_and_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = _executable(tmp_path / "explicit" / "codex")
    bundled = _executable(tmp_path / "ChatGPT.app" / "codex")
    path_codex = _executable(tmp_path / "path" / "codex")
    monkeypatch.setattr(codex_app_server, "_CHATGPT_APP_CODEX", bundled)

    resolved = resolve_codex_app_server_executable(
        {
            CODEX_APP_SERVER_EXECUTABLE_ENV: str(explicit),
            "PATH": str(path_codex.parent),
        }
    )

    assert resolved == str(explicit)


def test_official_chatgpt_bundle_wins_over_stale_path_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = _executable(tmp_path / "ChatGPT.app" / "codex")
    path_codex = _executable(tmp_path / "path" / "codex")
    monkeypatch.setattr(codex_app_server, "_CHATGPT_APP_CODEX", bundled)

    resolved = resolve_codex_app_server_executable({"PATH": str(path_codex.parent)})

    assert resolved == str(bundled)


def test_codex_runtime_falls_back_to_reviewed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_bundle = tmp_path / "missing" / "codex"
    path_codex = _executable(tmp_path / "path" / "codex")
    monkeypatch.setattr(codex_app_server, "_CHATGPT_APP_CODEX", missing_bundle)

    resolved = resolve_codex_app_server_executable({"PATH": str(path_codex.parent)})

    assert resolved == str(path_codex)


def test_invalid_explicit_codex_runtime_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(
        CodexAppServerError,
        match="codex_auth_executable_override_invalid",
    ):
        resolve_codex_app_server_executable(
            {CODEX_APP_SERVER_EXECUTABLE_ENV: str(tmp_path / "missing")}
        )


def test_codex_runtime_override_crosses_only_the_codex_subprocess_boundary(
    tmp_path: Path,
) -> None:
    explicit = _executable(tmp_path / "codex")

    selected = build_provider_subprocess_env(
        "codex",
        environment={
            "HOME": str(tmp_path / "home"),
            "CODEX_HOME": str(tmp_path / "codex-home"),
            CODEX_APP_SERVER_EXECUTABLE_ENV: str(explicit),
            "OPENAI_API_KEY": "must-not-cross-boundary",
        },
    )

    assert selected[CODEX_APP_SERVER_EXECUTABLE_ENV] == str(explicit)
    assert "OPENAI_API_KEY" not in selected


def test_progress_resets_notification_idle_timeout_but_not_the_hard_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ScriptedClock()
    progress = {"method": "item/agentMessage/delta", "params": {"delta": "{"}}
    completed = {"method": "turn/completed", "params": {}}
    runtime = _scripted_runtime(
        clock=clock,
        events=[(4.0, progress), (4.0, completed)],
    )
    monkeypatch.setattr(codex_app_server.time, "monotonic", lambda: clock.now)

    observed = runtime.wait_for_notification(
        lambda item: item.get("method") == "turn/completed",
        timeout=5.0,
        hard_timeout=12.0,
        progress_predicate=lambda item: item.get("method") == "item/agentMessage/delta",
    )

    assert observed == completed
    assert clock.now == 8.0


def test_progress_cannot_extend_notification_wait_beyond_the_hard_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ScriptedClock()
    progress = {"method": "item/reasoning/textDelta", "params": {"delta": "x"}}
    runtime = _scripted_runtime(
        clock=clock,
        events=[(4.0, progress), (4.0, progress), (4.0, progress)],
    )
    monkeypatch.setattr(codex_app_server.time, "monotonic", lambda: clock.now)

    with pytest.raises(
        CodexAppServerError,
        match="codex_auth_notification_hard_timeout",
    ):
        runtime.wait_for_notification(
            lambda item: item.get("method") == "turn/completed",
            timeout=5.0,
            hard_timeout=9.0,
            progress_predicate=lambda item: (
                item.get("method") == "item/reasoning/textDelta"
            ),
        )

    assert clock.now == 9.0


def test_progress_aware_wait_requires_an_explicit_hard_timeout() -> None:
    clock = _ScriptedClock()
    runtime = _scripted_runtime(clock=clock, events=[])

    with pytest.raises(
        ValueError,
        match="progress-aware notification wait requires hard_timeout",
    ):
        runtime.wait_for_notification(
            lambda _item: False,
            timeout=5.0,
            progress_predicate=lambda _item: True,
        )


def test_notification_cursor_survives_bounded_history_rollover() -> None:
    clock = _ScriptedClock()
    runtime = _scripted_runtime(clock=clock, events=[])
    runtime._notification_offset = 512
    runtime._notifications = [
        {"method": "item/agentMessage/delta", "params": {"delta": "x"}}
        for _ in range(512)
    ]
    completed = {"method": "turn/completed", "params": {}}
    runtime._notifications.append(completed)
    runtime._state.notifications = runtime._notifications

    observed = runtime.wait_for_notification(
        lambda item: item.get("method") == "turn/completed",
        after=1_000,
        timeout=5.0,
    )

    assert observed == completed
    assert runtime.notification_count == 1_025
    assert runtime.notifications_since(1_024) == [completed]
