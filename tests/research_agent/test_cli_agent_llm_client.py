"""Unit tests for the local CLI-agent LLM client (altitude-1 integration).

Fully offline: ``shutil.which`` and ``subprocess.run`` are patched so no real
``codex`` / ``claude`` process is launched.
"""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest


def _client(backend="codex", model=None):
    from easyicu.research_agent.providers.llm import CLIAgentLLMClient

    return CLIAgentLLMClient(backend=backend, model=model)


def test_rejects_unknown_backend():
    from easyicu.research_agent.providers.llm import CLIAgentLLMClient

    with pytest.raises(ValueError):
        CLIAgentLLMClient(backend="not-a-cli")


def test_name_and_model():
    c = _client("claude", model="opus")
    assert c.name == "claude-cli"
    assert c._model == "opus"
    # empty model => CLI default
    assert _client("codex")._model == ""


def test_flatten_splits_system_and_transcript():
    from easyicu.research_agent.providers.llm import CLIAgentLLMClient, LLMMessage

    system, convo = CLIAgentLLMClient._flatten([
        LLMMessage(role="system", content="be terse"),
        LLMMessage(role="user", content="hi"),
        LLMMessage(role="assistant", content="hello"),
    ])
    assert system == "be terse"
    assert "User:\nhi" in convo
    assert "Assistant:\nhello" in convo


def test_build_argv_claude_text_mode_no_dangerous_flags():
    c = _client("claude", model="opus")
    argv = c._build_argv("be terse", "/tmp/x")
    assert argv[:2] == ["claude", "-p"]
    assert "--output-format" in argv and "text" in argv
    assert "--model" in argv and "opus" in argv
    assert "--append-system-prompt" in argv
    assert "--dangerously-skip-permissions" not in argv


def test_build_argv_codex_read_only_sandbox():
    c = _client("codex")
    argv = c._build_argv("", "/tmp/x")
    assert argv[:2] == ["codex", "exec"]
    assert "--sandbox" in argv and "read-only" in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert "-m" not in argv  # default model => no override


def test_complete_returns_text_and_strips_reasoning(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)

    def _run(argv, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="<think>secret</think>\nfinal answer",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _run)
    out = _client("codex").complete([LLMMessage(role="user", content="hi")])
    assert out == "final answer"


def test_complete_missing_cli_raises(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    with pytest.raises(RuntimeError):
        _client("claude").complete([LLMMessage(role="user", content="hi")])


def test_complete_nonzero_exit_raises(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError):
        _client("codex").complete([LLMMessage(role="user", content="hi")])


def test_complete_accepts_and_ignores_extra_kwargs(monkeypatch):
    """Protocol kwargs the CLI cannot honour must not raise."""
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    out = _client("codex").complete(
        [LLMMessage(role="user", content="hi")],
        max_tokens=10, temperature=0.7, seed=1, top_p=0.9,
    )
    assert out == "ok"


def test_exported_from_package():
    import easyicu.research_agent as ra

    assert ra.CLIAgentLLMClient is not None


# ---------------------------------------------------------------------------
# build_llm_client capability ladder (concern #1: CLI is optional, never required)
# ---------------------------------------------------------------------------

def _patch_cli(monkeypatch, available):
    """Make only the named CLI backends appear installed."""
    import easyicu.research_agent.providers.llm as llm_mod

    def _which(cmd):
        return "/usr/bin/" + cmd if cmd in available else None

    monkeypatch.setattr(llm_mod.shutil if hasattr(llm_mod, "shutil") else __import__("shutil"),
                        "which", _which)
    # cli_backend_available imports shutil locally, so patch the real module too
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", _which)


def test_ladder_uses_cli_when_available(monkeypatch):
    from easyicu.research_agent.providers.llm import build_llm_client

    _patch_cli(monkeypatch, {"codex"})
    sel = build_llm_client(prefer="codex")
    assert sel.backend == "codex"
    assert sel.fell_back is False
    assert sel.client.name == "codex-cli"


def test_ladder_falls_back_to_mock_when_nothing_available(monkeypatch):
    from easyicu.research_agent.providers.llm import build_llm_client

    _patch_cli(monkeypatch, set())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    sel = build_llm_client(prefer="codex")
    assert sel.backend == "mock"
    assert sel.fell_back is True
    assert "mock" in sel.reason


def test_ladder_falls_back_to_api_when_cli_absent_but_key_present(monkeypatch):
    from easyicu.research_agent.providers.llm import build_llm_client

    _patch_cli(monkeypatch, set())
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    sel = build_llm_client(prefer="claude", api_key="sk-test", model="gpt-4o-mini")
    assert sel.backend == "openai"
    assert sel.fell_back is True
    assert sel.client.name == "openai"


def test_ladder_no_mock_raises_when_nothing_usable(monkeypatch):
    import pytest as _pytest

    from easyicu.research_agent.providers.llm import build_llm_client

    _patch_cli(monkeypatch, set())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with _pytest.raises(RuntimeError):
        build_llm_client(prefer="codex", allow_mock=False)


def test_cli_backend_available_helper(monkeypatch):
    from easyicu.research_agent.providers.llm import cli_backend_available

    _patch_cli(monkeypatch, {"claude"})
    assert cli_backend_available("claude")
    assert not cli_backend_available("codex")
    assert not cli_backend_available("openai")  # not a CLI backend
