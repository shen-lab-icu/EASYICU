"""Protocol and pinned-runtime tests for the local Pi sidecar."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.contracts import (
    PROTOCOL_VERSION,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.gateway import PiGatewayClient, _PendingRequest

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
    assert "event.result?.details?.summary" in entrypoint


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
