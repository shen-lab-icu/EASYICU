from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.providers import codex_app_server
from easyicu.research_agent.providers.codex_app_server import (
    CodexAppServerError,
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
