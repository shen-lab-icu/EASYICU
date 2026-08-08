"""Protocol and pinned-runtime tests for the local Pi sidecar."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.contracts import (
    PROTOCOL_VERSION,
    PiCopilotError,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.gateway import PiGatewayClient, _PendingRequest
from easyicu.webserver.pi_copilot.provider_config import PiProviderConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = (
    REPO_ROOT
    / "src"
    / "easyicu"
    / "webserver"
    / "pi_copilot"
    / "node_app"
)


def test_pi_packages_and_upstream_commit_are_exactly_pinned() -> None:
    package = json.loads((APP_DIR / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((APP_DIR / "package-lock.json").read_text(encoding="utf-8"))

    assert package["dependencies"]["@earendil-works/pi-coding-agent"] == "0.84.1"
    assert package["dependencies"]["@earendil-works/pi-ai"] == "0.84.1"
    assert package["overrides"]["@earendil-works/pi-agent-core"] == "0.84.1"
    assert lock["packages"][""]["dependencies"]["@earendil-works/pi-coding-agent"] == "0.84.1"
    entrypoint = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    assert "9dd90a49711d088b86fdd9b4aea575913a8328a8" in entrypoint
    assert 'noTools: "builtin"' in entrypoint
    assert 'content: [{ type: "text", text: JSON.stringify(modelVisible) }]' in entrypoint
    assert "details: modelVisible" in entrypoint
    projection = (APP_DIR / "src" / "event-projection.mjs").read_text(
        encoding="utf-8"
    )
    assert "details.summary" in projection


def test_host_reauthorizes_tool_request_and_rejects_unknown_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def execute(name, arguments, context):
        calls.append((name, arguments, context))
        return {"status": "ok", "code": "test_ok", "summary": "ok", "owner": "test", "details": {}, "authority": {}}

    gateway = PiGatewayClient(app_dir=APP_DIR, session_dir=tmp_path, tool_executor=execute)
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-test"))
    gateway._pending["parent"] = _PendingRequest(tool_context=context)
    writes = []
    monkeypatch.setattr(gateway, "_write", lambda payload: writes.append(dict(payload)))

    gateway._handle_tool_request(
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "tool_request",
            "request_id": "tool-1",
            "parent_request_id": "parent",
            "session_id": "pi-test",
            "method": "tool.execute",
            "params": {"name": "easyicu_inspect_context", "arguments": {}},
        }
    )
    assert calls == [("easyicu_inspect_context", {}, context)]
    assert writes[-1]["ok"] is True

    gateway._handle_tool_request(
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "tool_request",
            "request_id": "tool-2",
            "parent_request_id": "parent",
            "session_id": "pi-test",
            "method": "tool.execute",
            "params": {"name": "easyicu_inspect_context", "arguments": {}},
            "unexpected": True,
        }
    )
    assert writes[-1]["ok"] is False
    assert writes[-1]["error"]["code"] == "pi_tool_request_invalid"

    gateway._handle_tool_request(
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "tool_request",
            "request_id": "tool-3",
            "parent_request_id": "parent",
            "session_id": "different-session",
            "method": "tool.execute",
            "params": {"name": "easyicu_inspect_context", "arguments": {}},
        }
    )
    assert writes[-1]["ok"] is False
    assert writes[-1]["error"]["code"] == "pi_tool_request_invalid"


def test_host_rejects_unknown_sidecar_response_fields(tmp_path: Path) -> None:
    gateway = PiGatewayClient(app_dir=APP_DIR, session_dir=tmp_path)
    pending = _PendingRequest()
    gateway._pending["request-1"] = pending

    gateway._handle_payload(
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "response",
            "request_id": "request-1",
            "ok": True,
            "result": {},
            "unexpected": True,
        }
    )

    assert pending.done.is_set()
    assert pending.error is not None
    assert pending.error.code == "pi_protocol_unknown_fields"


def test_host_rejects_cross_session_stream_event(tmp_path: Path) -> None:
    gateway = PiGatewayClient(app_dir=APP_DIR, session_dir=tmp_path)
    context = ToolExecutionContext(session=PiSessionRecord(session_id="pi-test"))
    pending = _PendingRequest(tool_context=context)
    gateway._pending["request-1"] = pending

    gateway._handle_payload(
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "event",
            "request_id": "request-1",
            "session_id": "different-session",
            "event": {"type": "text_delta", "delta": "wrong stream"},
        }
    )

    assert pending.done.is_set()
    assert pending.error is not None
    assert pending.error.code == "pi_protocol_session_mismatch"


def test_sidecar_environment_is_allowlisted_and_workspace_is_private(
    tmp_path: Path,
) -> None:
    gateway = PiGatewayClient(
        app_dir=APP_DIR,
        session_dir=tmp_path / "sessions",
        environ={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "LANG": "en_US.UTF-8",
            "EASYICU_PI_API_KEY": "pi-only-secret",
            "EASYICU_PI_MODEL": "gpt5.6 luna",
            "OPENAI_API_KEY": "scientific-secret",
            "ANTHROPIC_API_KEY": "scientific-secret",
            "TAVILY_API_KEY": "search-secret",
            "AWS_SECRET_ACCESS_KEY": "cloud-secret",
            "DATABASE_PASSWORD": "database-secret",
        },
    )

    assert gateway.environ["EASYICU_PI_API_KEY"] == "pi-only-secret"
    assert gateway.environ["PATH"] == "/usr/bin:/bin"
    assert gateway.environ["HOME"] == str(tmp_path / "home")
    assert "OPENAI_API_KEY" not in gateway.environ
    assert "ANTHROPIC_API_KEY" not in gateway.environ
    assert "TAVILY_API_KEY" not in gateway.environ
    assert "AWS_SECRET_ACCESS_KEY" not in gateway.environ
    assert "DATABASE_PASSWORD" not in gateway.environ
    assert gateway.cwd == (tmp_path / "workspace").resolve()
    assert gateway._child_environment() == {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "LANG": "en_US.UTF-8",
        "EASYICU_PI_API_KEY": "pi-only-secret",
        "EASYICU_PI_MODEL": "gpt5.6 luna",
        "EASYICU_PI_SESSION_DIR": str((tmp_path / "sessions").resolve()),
        "EASYICU_PI_CWD": str((tmp_path / "workspace").resolve()),
    }


def test_reconfigure_preserves_independent_shell_budget_settings(
    tmp_path: Path,
) -> None:
    gateway = PiGatewayClient(
        app_dir=APP_DIR,
        session_dir=tmp_path / "sessions",
        environ={
            "PATH": "/usr/bin:/bin",
            "EASYICU_PI_SESSION_TOKEN_BUDGET": "42000",
            "EASYICU_PI_MAX_TOKENS": "2048",
        },
    )

    gateway.apply_provider_config(
        PiProviderConfig(
            provider="easyicu-local",
            api_key="new-private-key",
            base_url="http://127.0.0.1:8317/v1",
            model="gpt5.6 luna",
            api_transport="openai-completions",
        )
    )

    assert gateway.environ["EASYICU_PI_SESSION_TOKEN_BUDGET"] == "42000"
    assert gateway.environ["EASYICU_PI_MAX_TOKENS"] == "2048"
    assert gateway.environ["EASYICU_PI_API_KEY"] == "new-private-key"


def test_prompt_timeout_aborts_and_refreshes_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = PiGatewayClient(app_dir=APP_DIR, session_dir=tmp_path)
    recovered: list[str] = []
    monkeypatch.setattr(gateway, "_start", lambda: None)
    monkeypatch.setattr(gateway, "_write", lambda payload: None)
    monkeypatch.setattr(
        gateway,
        "_recover_timed_out_prompt",
        lambda session_id: recovered.append(session_id),
    )
    monkeypatch.setattr(
        "easyicu.webserver.pi_copilot.gateway.threading.Event.wait",
        lambda self, timeout: False,
    )

    with pytest.raises(PiCopilotError) as caught:
        gateway.request(
            "session.prompt",
            {"session_id": "pi-timeout", "message": "inspect"},
            timeout=0.1,
        )

    assert caught.value.code == "pi_gateway_timeout"
    assert recovered == ["pi-timeout"]


def test_sidecar_contract_hides_reasoning_and_enforces_token_budget() -> None:
    source = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    projection = (APP_DIR / "src" / "event-projection.mjs").read_text(
        encoding="utf-8"
    )

    assert "EASYICU_PI_SESSION_TOKEN_BUDGET" in source
    assert "pi_shell_token_budget_exhausted" in source
    assert 'update.type === "thinking_delta"' not in source + projection
    assert 'item.type === "thinking"' not in source + projection
    assert "normalizePiEvent" in source
    assert "projectTranscriptMessage" in source


def test_sidecar_projects_safe_agent_activity_and_tool_receipts() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    projection = (APP_DIR / "src" / "event-projection.mjs").as_uri()
    script = f"""
      import {{ normalizePiEvent, projectTranscriptMessage }} from {json.dumps(projection)};
      const events = [
        normalizePiEvent({{ type: 'agent_start' }}),
        normalizePiEvent({{ type: 'turn_start', turnIndex: 1, timestamp: 1720000000000 }}),
        normalizePiEvent({{ type: 'message_start', message: {{ role: 'assistant' }} }}),
        normalizePiEvent({{ type: 'tool_execution_start', toolCallId: 'call-1', toolName: 'easyicu_inspect_context', args: {{ secret: 'must-not-leak' }} }}),
        normalizePiEvent({{ type: 'tool_execution_update', toolCallId: 'call-1', toolName: 'easyicu_inspect_context', partialResult: 'must-not-leak' }}),
        normalizePiEvent({{ type: 'tool_execution_end', toolCallId: 'call-1', toolName: 'easyicu_inspect_context', isError: false, result: {{ content: [{{ type: 'text', text: 'unsafe fallback' }}], details: {{ status: 'ok', code: 'study_context_ready', summary: 'Bounded summary', owner: 'easyicu.study_context' }} }} }}),
        normalizePiEvent({{ type: 'agent_settled' }}),
        normalizePiEvent({{ type: 'message_update', assistantMessageEvent: {{ type: 'thinking_delta', delta: 'private' }} }}),
      ];
      const transcript = projectTranscriptMessage({{
        role: 'toolResult', toolCallId: 'call-1', toolName: 'easyicu_inspect_context',
        content: [{{ type: 'text', text: 'raw result must not leak' }}],
        details: {{ status: 'ok', code: 'study_context_ready', summary: 'Persisted bounded summary', owner: 'easyicu.study_context' }},
      }});
      console.log(JSON.stringify({{ events, transcript }}));
    """
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert [event["type"] if event else None for event in payload["events"]] == [
        "run_start",
        "turn_start",
        "assistant_start",
        "tool_start",
        "tool_progress",
        "tool_end",
        "run_end",
        None,
    ]
    assert "args" not in payload["events"][3]
    assert "partial_result" not in payload["events"][4]
    assert payload["events"][5]["code"] == "study_context_ready"
    assert payload["events"][5]["summary"] == "Bounded summary"
    assert payload["transcript"]["content"][0]["type"] == "tool_result"
    assert payload["transcript"]["content"][0]["summary"] == "Persisted bounded summary"
    assert "raw result must not leak" not in completed.stdout
    assert "must-not-leak" not in completed.stdout


def test_pinned_sidecar_starts_with_only_easyicu_tools(tmp_path: Path) -> None:
    dependency = APP_DIR / "node_modules" / "@earendil-works" / "pi-coding-agent" / "package.json"
    if shutil.which("node") is None or not dependency.is_file():
        pytest.skip("Pinned Node dependencies are not installed in this checkout")

    gateway = PiGatewayClient(
        app_dir=APP_DIR,
        session_dir=tmp_path / "sessions",
        cwd=REPO_ROOT,
        environ={
            "PATH": str(Path(shutil.which("node") or "").parent),
            "EASYICU_PI_API_KEY": "test-only-placeholder",
            "EASYICU_PI_PROVIDER": "easyicu-local",
            "EASYICU_PI_BASE_URL": "http://127.0.0.1:8317/v1",
            "EASYICU_PI_MODEL": "gpt5.6 luna",
            "EASYICU_PI_API": "openai-completions",
        },
    )
    try:
        runtime = gateway.request("runtime.status", {}, timeout=20)
        state = gateway.request(
            "session.create",
            {"session_id": "pi-smoke", "thinking_level": "medium"},
            timeout=30,
        )
    finally:
        gateway.close()

    assert runtime["provider"] == "easyicu-local"
    assert runtime["model"] == "gpt5.6 luna"
    assert runtime["built_in_tools_enabled"] == []
    assert state["enabled_tools"] == runtime["custom_tools"]
    assert len(state["enabled_tools"]) == 15
    assert {"read", "write", "edit", "bash"}.isdisjoint(state["enabled_tools"])
