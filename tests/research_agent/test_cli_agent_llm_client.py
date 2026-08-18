"""Unit tests for the local CLI-agent LLM client (altitude-1 integration).

Fully offline: ``shutil.which`` and ``subprocess.run`` are patched so no real
``codex`` / ``claude`` process is launched.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest


def _client(backend="codex", model=None):
    from easyicu.research_agent.providers.llm import CLIAgentLLMClient

    return CLIAgentLLMClient(backend=backend, model=model)


def _authorized_client(backend="codex", model=None):
    from easyicu.research_agent.providers.factory import authorize_provider_client

    client = _client(backend, model=model)
    return authorize_provider_client(
        client,
        provider=f"{backend}-cli",
        model=model or "cli-default",
        base_url=f"cli://{backend}",
        destination="external",
        environment={"EASYICU_ALLOW_EXTERNAL_LLM": "1"},
    )


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

    system, convo = CLIAgentLLMClient._flatten(
        [
            LLMMessage(role="system", content="be terse"),
            LLMMessage(role="user", content="hi"),
            LLMMessage(role="assistant", content="hello"),
        ]
    )
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
    assert "--ephemeral" in argv
    assert "--ignore-user-config" in argv
    assert "--ignore-rules" in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert "-m" not in argv  # default model => no override


def test_gemini_is_not_a_registered_cli_backend():
    with pytest.raises(ValueError, match="Unknown CLI backend"):
        _client("gemini", model="flash")


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
    out = _authorized_client("codex").complete([LLMMessage(role="user", content="hi")])
    assert out == "final answer"


def test_direct_constructor_is_denied_before_cli_launch(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unmanaged CLI transport must not launch")
        ),
    )

    with pytest.raises(PermissionError, match="authorization"):
        _client("codex").complete([LLMMessage(role="user", content="hi")])


def test_cli_subprocess_receives_only_reviewed_environment(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setenv("CODEX_HOME", "/private/codex-account")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-switch-to-api-auth")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy-secret.example")
    monkeypatch.setenv("DATABASE_URL", "postgresql://must-not-leak")
    captured = {}

    def _run(argv, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)

    assert (
        _authorized_client("codex").complete([LLMMessage(role="user", content="hi")])
        == "ok"
    )
    assert captured["env"]["CODEX_HOME"] == "/private/codex-account"
    assert "OPENAI_API_KEY" not in captured["env"]
    assert "AWS_SECRET_ACCESS_KEY" not in captured["env"]
    assert "GITHUB_TOKEN" not in captured["env"]
    assert "HTTPS_PROXY" not in captured["env"]
    assert "DATABASE_URL" not in captured["env"]


def test_claude_cli_receives_only_claude_authentication(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-account-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-switch-to-api-auth")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-secret")
    captured = {}

    def _run(argv, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)

    assert (
        _authorized_client("claude").complete([LLMMessage(role="user", content="hi")])
        == "ok"
    )
    assert captured["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "claude-account-token"
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert "OPENAI_API_KEY" not in captured["env"]


def test_codex_cli_materializes_typed_output_schema_in_private_tempdir(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage
    from easyicu.research_agent.providers.protocol import StructuredOutputRequest

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    captured = {}

    def _run(argv, **kwargs):
        schema_path = argv[argv.index("--output-schema") + 1]
        captured["argv"] = list(argv)
        captured["schema"] = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        return SimpleNamespace(
            returncode=0,
            stdout='{"status":"ready","value":7}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _run)
    request = StructuredOutputRequest.from_schema(
        name="codex_account_probe",
        schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "value": {"type": "integer"},
            },
            "required": ["status", "value"],
            "additionalProperties": False,
        },
    )

    result = _authorized_client("codex").complete(
        [LLMMessage(role="user", content="Return JSON")],
        structured_output=request,
    )

    assert result == '{"status":"ready","value":7}'
    assert captured["schema"] == json.loads(request.schema_json)
    assert "--output-schema" in captured["argv"]


def test_claude_account_refuses_unadvertised_strict_schema_before_launch(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage
    from easyicu.research_agent.providers.protocol import (
        StructuredOutputCapabilityError,
        StructuredOutputRequest,
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported schema must fail before launch")
        ),
    )
    request = StructuredOutputRequest.from_schema(
        name="unsupported",
        schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )
    with pytest.raises(StructuredOutputCapabilityError):
        _authorized_client("claude").complete(
            [LLMMessage(role="user", content="Return JSON")],
            structured_output=request,
        )


def test_cli_account_freezes_auth_environment_at_construction(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setenv("CODEX_HOME", "/private/first-account")
    client = _authorized_client("codex")
    monkeypatch.setenv("CODEX_HOME", "/private/second-account")
    captured = {}

    def _run(argv, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    assert client.complete([LLMMessage(role="user", content="hi")]) == "ok"
    assert captured["env"]["CODEX_HOME"] == "/private/first-account"


def test_cli_account_manifest_names_account_session_and_schema_capability(monkeypatch):
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
    )

    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    manifest = provider_authorization_manifest(_authorized_client("codex"))
    client = manifest["clients"][0]
    assert client["provider"] == "codex-cli"
    assert client["base_url"] == "cli://codex"
    assert client["authorization_mode"] == "account_session"
    assert client["transport_policy"]["transport"] == "cli_account"
    assert client["transport_policy"]["strict_json_schema_enabled"] is True


def test_complete_missing_cli_raises(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    with pytest.raises(RuntimeError):
        _authorized_client("claude").complete([LLMMessage(role="user", content="hi")])


def test_complete_nonzero_exit_raises(monkeypatch):
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError):
        _authorized_client("codex").complete([LLMMessage(role="user", content="hi")])


def test_complete_accepts_and_ignores_extra_kwargs(monkeypatch):
    """Protocol kwargs the CLI cannot honour must not raise."""
    from easyicu.research_agent.providers.llm import LLMMessage

    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    out = _authorized_client("codex").complete(
        [LLMMessage(role="user", content="hi")],
        max_tokens=10,
        temperature=0.7,
        seed=1,
        top_p=0.9,
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

    monkeypatch.setattr(
        llm_mod.shutil if hasattr(llm_mod, "shutil") else __import__("shutil"),
        "which",
        _which,
    )
    # cli_backend_available imports shutil locally, so patch the real module too
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", _which)


def test_ladder_uses_cli_when_available(monkeypatch):
    from easyicu.research_agent.providers.llm import build_llm_client

    _patch_cli(monkeypatch, {"codex"})
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    sel = build_llm_client(prefer="codex")
    assert sel.backend == "codex"
    assert sel.fell_back is False
    assert sel.client.name == "codex-cli"


def test_ladder_resolves_codex_model_from_account_specific_environment(monkeypatch):
    from easyicu.research_agent.providers.llm import build_llm_client

    _patch_cli(monkeypatch, {"codex"})
    environment = {
        "PATH": "/usr/bin",
        "HOME": "/private/user",
        "CODEX_HOME": "/private/codex",
        "EASYICU_ALLOW_EXTERNAL_LLM": "1",
        "EASYICU_CODEX_MODEL": "account-configured-model",
    }

    selection = build_llm_client(prefer="codex", environment=environment)

    assert selection.client._model == "account-configured-model"


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


def test_ladder_builds_requested_deepseek_without_cross_provider_env(monkeypatch):
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
    )
    from easyicu.research_agent.providers.llm import build_llm_client

    _patch_cli(monkeypatch, set())
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-select-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong-provider.example/v1")
    monkeypatch.setenv("EASYICU_OPENAI_AUTH_HEADER", "x-api-key")
    monkeypatch.setenv("EASYICU_TRUST_LOOPBACK_PROXY_KEY", "1")

    sel = build_llm_client(
        prefer="deepseek",
        model="deepseek-v4-flash",
        allow_mock=False,
    )

    assert sel.backend == "deepseek"
    assert sel.fell_back is False
    assert sel.client.name == "openai"  # shared wire adapter, not provider identity
    manifest = provider_authorization_manifest(sel.client)
    assert manifest["clients"][0]["provider"] == "deepseek"
    assert manifest["clients"][0]["model"] == "deepseek-v4-flash"
    assert manifest["clients"][0]["base_url"] == "https://api.deepseek.com"


def test_ladder_never_sends_openai_default_model_to_deepseek(monkeypatch):
    from easyicu.research_agent.providers.llm import build_llm_client

    _patch_cli(monkeypatch, set())
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")

    with pytest.raises(ValueError, match="explicit model is required"):
        build_llm_client(prefer="deepseek", allow_mock=False)


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

    _patch_cli(monkeypatch, {"codex", "claude", "gemini"})
    assert cli_backend_available("codex")
    assert not cli_backend_available("claude")
    assert not cli_backend_available("gemini")
    assert not cli_backend_available("openai")  # not a CLI backend


def test_codex_account_readiness_verifies_login_without_returning_output(monkeypatch):
    from easyicu.research_agent.providers.llm import probe_cli_account_readiness

    monkeypatch.setattr(
        shutil,
        "which",
        lambda command, path=None: f"/usr/bin/{command}",
    )
    captured = {}

    def _run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout="Logged in using a private account identifier",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _run)
    result = probe_cli_account_readiness(
        "codex",
        environment={
            "PATH": "/usr/bin",
            "HOME": "/private/user",
            "CODEX_HOME": "/private/codex",
            "OPENAI_API_KEY": "must-not-cross-account-boundary",
        },
    )

    assert result.reason_code == "cli_account_ready"
    assert result.authentication_verified is True
    assert result.launch_ready is True
    assert result.subprocess_calls == 1
    assert captured["argv"] == ["codex", "login", "status"]
    assert "OPENAI_API_KEY" not in captured["env"]
    assert "private account identifier" not in repr(result)


def test_codex_account_readiness_fails_closed_when_login_is_absent(monkeypatch):
    from easyicu.research_agent.providers.llm import probe_cli_account_readiness

    monkeypatch.setattr(
        shutil,
        "which",
        lambda command, path=None: f"/usr/bin/{command}",
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="Not logged in",
            stderr="private diagnostic",
        ),
    )

    result = probe_cli_account_readiness("codex", environment={"PATH": "/usr/bin"})

    assert result.reason_code == "cli_login_required"
    assert result.authentication_verified is False
    assert result.launch_ready is False
    assert "private diagnostic" not in repr(result)


def test_internal_claude_readiness_without_status_command_is_explicitly_unverified(
    monkeypatch,
):
    from easyicu.research_agent.providers.llm import probe_cli_account_readiness

    monkeypatch.setattr(
        shutil,
        "which",
        lambda command, path=None: f"/usr/bin/{command}",
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("This CLI has no reviewed login-status subprocess")
        ),
    )

    result = probe_cli_account_readiness("claude", environment={"PATH": "/usr/bin"})

    assert result.reason_code == "cli_login_status_unavailable"
    assert result.status_check_supported is False
    assert result.authentication_verified is None
    assert result.launch_ready is True
    assert result.subprocess_calls == 0
