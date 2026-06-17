"""Unit tests for the local CLI-agent copilot backend.

These run fully offline: ``subprocess.run`` and ``shutil.which`` are patched so
no real ``claude`` / ``codex`` process is launched.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from easyicu.webapp.copilot import cli_agent


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def test_is_cli_provider():
    assert cli_agent.is_cli_provider("cli_claude")
    assert cli_agent.is_cli_provider("cli_codex")
    assert not cli_agent.is_cli_provider("openai")
    assert not cli_agent.is_cli_provider("")


def test_cli_command_for():
    assert cli_agent.cli_command_for("cli_claude") == "claude"
    assert cli_agent.cli_command_for("cli_codex") == "codex"
    assert cli_agent.cli_command_for("openai") is None


def test_cli_available(monkeypatch):
    monkeypatch.setattr(cli_agent.shutil, "which",
                        lambda cmd: "/usr/bin/" + cmd if cmd == "claude" else None)
    assert cli_agent.cli_available("cli_claude")
    assert not cli_agent.cli_available("cli_codex")


# ---------------------------------------------------------------------------
# Message flattening / argv construction
# ---------------------------------------------------------------------------

def test_flatten_messages_splits_system_and_transcript():
    system, convo = cli_agent._flatten_messages([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
    ])
    assert system == "be brief"
    assert "User:\nhi" in convo
    assert "Assistant:\nhello" in convo
    assert convo.index("hi") < convo.index("hello") < convo.index("more")


def test_resolve_model_sentinels():
    assert cli_agent._resolve_model("default") is None
    assert cli_agent._resolve_model("") is None
    assert cli_agent._resolve_model("local-cli") is None
    assert cli_agent._resolve_model("opus") == "opus"


def test_build_argv_claude_includes_text_mode_and_no_dangerous_flags():
    argv = cli_agent._build_argv("cli_claude", "claude", "opus", "be brief", "/tmp/x")
    assert argv[:2] == ["claude", "-p"]
    assert "--output-format" in argv and "text" in argv
    assert "--model" in argv and "opus" in argv
    assert "--append-system-prompt" in argv
    # Never grant tool/permission bypass — this must stay a text generator.
    assert "--dangerously-skip-permissions" not in argv
    assert "--allow-dangerously-skip-permissions" not in argv


def test_build_argv_codex_is_read_only_sandbox():
    argv = cli_agent._build_argv("cli_codex", "codex", "default", "", "/tmp/x")
    assert argv[:2] == ["codex", "exec"]
    assert "--sandbox" in argv and "read-only" in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    # "default" sentinel -> no -m override.
    assert "-m" not in argv


# ---------------------------------------------------------------------------
# End-to-end client surface (subprocess patched)
# ---------------------------------------------------------------------------

def _fake_run_ok(text="final answer"):
    def _run(argv, **kwargs):
        return SimpleNamespace(returncode=0, stdout=text, stderr="")
    return _run


def test_client_non_stream_returns_openai_shape(monkeypatch):
    monkeypatch.setattr(cli_agent.shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setattr(cli_agent.subprocess, "run", _fake_run_ok("hello world"))
    client = cli_agent.CLIAgentClient(provider="cli_claude")
    resp = client.chat.completions.create(
        model="sonnet",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )
    assert resp.choices[0].message.content == "hello world"


def test_client_stream_yields_delta_chunks(monkeypatch):
    monkeypatch.setattr(cli_agent.shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setattr(cli_agent.subprocess, "run", _fake_run_ok("a b c"))
    client = cli_agent.CLIAgentClient(provider="cli_codex")
    stream = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    tokens = [chunk.choices[0].delta.content for chunk in stream]
    assert "".join(tokens) == "a b c"


def test_with_options_overrides_timeout():
    client = cli_agent.CLIAgentClient(provider="cli_claude", timeout=180.0)
    tuned = client.with_options(timeout=5, max_retries=0)
    assert tuned.timeout == 5
    assert tuned.provider == "cli_claude"
    # original untouched
    assert client.timeout == 180.0


def test_missing_cli_raises(monkeypatch):
    monkeypatch.setattr(cli_agent.shutil, "which", lambda cmd: None)
    client = cli_agent.CLIAgentClient(provider="cli_claude")
    with pytest.raises(cli_agent.CLIAgentError):
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
        )


def test_nonzero_exit_raises(monkeypatch):
    monkeypatch.setattr(cli_agent.shutil, "which", lambda cmd: "/usr/bin/" + cmd)

    def _run(argv, **kwargs):
        return SimpleNamespace(returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(cli_agent.subprocess, "run", _run)
    client = cli_agent.CLIAgentClient(provider="cli_claude")
    with pytest.raises(cli_agent.CLIAgentError):
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
        )


def test_timeout_raises(monkeypatch):
    monkeypatch.setattr(cli_agent.shutil, "which", lambda cmd: "/usr/bin/" + cmd)

    def _run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=1)

    monkeypatch.setattr(cli_agent.subprocess, "run", _run)
    client = cli_agent.CLIAgentClient(provider="cli_claude")
    with pytest.raises(cli_agent.CLIAgentError):
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
        )
