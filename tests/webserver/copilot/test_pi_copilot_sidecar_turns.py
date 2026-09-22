"""Pinned Pi sidecar sessions: effort level and the reasoning trace.

Moved out of test_pi_copilot_gateway.py so it stays under the large-module
line: the real-sidecar smoke test (tool surface and per-session thinking
level) and the bounded reasoning-summary projection the trace renders.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.gateway import PiGatewayClient

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = REPO_ROOT / "src" / "easyicu" / "webserver" / "pi_copilot" / "node_app"


def test_pinned_sidecar_starts_with_only_easyicu_tools(tmp_path: Path) -> None:
    dependency = (
        APP_DIR
        / "node_modules"
        / "@earendil-works"
        / "pi-coding-agent"
        / "package.json"
    )
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
        workspace_state = gateway.request(
            "session.create",
            {
                "session_id": "pi-workspace-smoke",
                "thinking_level": "off",
                "agent_mode": "workspace",
                "language": "zh",
            },
            timeout=30,
        )
        with pytest.raises(PiCopilotError) as language_error:
            gateway.request(
                "session.create",
                {
                    "session_id": "pi-smoke",
                    "thinking_level": "off",
                    "language": "zh",
                },
                timeout=30,
            )
        # The effort level is per session: requested at creation, changed
        # between turns, clamped by the model, never a forced "off".
        assert state["thinking_level"] in {"medium", "low", "minimal", "off"}
        assert workspace_state["thinking_level"] == "off"
        changed = gateway.request(
            "session.set_thinking_level",
            {"session_id": "pi-smoke", "thinking_level": "high"},
            timeout=10,
        )
        assert changed["requested_thinking_level"] == "high"
        assert changed["thinking_level"] in {"high", "medium", "low", "minimal", "off"}
        assert gateway.request(
            "session.state", {"session_id": "pi-smoke"}, timeout=5
        )["thinking_level"] == changed["thinking_level"]
        with pytest.raises(PiCopilotError) as bad_level:
            gateway.request(
                "session.set_thinking_level",
                {"session_id": "pi-smoke", "thinking_level": "max"},
                timeout=10,
            )
        assert bad_level.value.code == "pi_thinking_level_invalid"
    finally:
        gateway.close()

    # A normal new conversation is durable before its first prompt. This keeps
    # ordinary Web sessions recoverable across a host restart without any
    # benchmark- or feature-specific session type.
    session_file = Path(state["session_file"])
    assert session_file.is_file()
    reopened_gateway = PiGatewayClient(
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
        reopened_state = reopened_gateway.request(
            "session.create",
            {
                "session_id": "pi-smoke",
                "session_file": str(session_file),
                "thinking_level": "off",
                "agent_mode": "research",
            },
            timeout=30,
        )
    finally:
        reopened_gateway.close()

    assert runtime["provider"] == "easyicu-local"
    assert runtime["model"] == "gpt5.6 luna"
    assert runtime["built_in_tools_enabled"] == []
    assert state["enabled_tools"] == runtime["custom_tools"]
    assert state["enabled_tools"]
    assert all(name.startswith("easyicu_") for name in state["enabled_tools"])
    assert {
        "easyicu_list_extensions",
        "easyicu_load_skill",
        "easyicu_call_mcp_tool",
    }.issubset(state["enabled_tools"])
    assert {"read", "write", "edit", "bash"}.isdisjoint(state["enabled_tools"])
    assert workspace_state["agent_mode"] == "workspace"
    assert state["language"] == "en"
    assert workspace_state["language"] == "zh"
    assert language_error.value.code == "pi_session_language_mismatch"
    assert workspace_state["enabled_tools"] == runtime["custom_tools_by_mode"]["workspace"]
    assert workspace_state["enabled_tools"]
    assert all(
        name.startswith("easyicu_") for name in workspace_state["enabled_tools"]
    )
    assert {"read", "write", "edit", "bash"}.isdisjoint(
        workspace_state["enabled_tools"]
    )
    assert reopened_state["session_file"] == str(session_file)
    assert reopened_state["agent_mode"] == "research"
    assert reopened_state["enabled_tools"] == state["enabled_tools"]


def test_sidecar_projects_bounded_reasoning_summaries_for_the_trace() -> None:
    """Reasoning summaries reach the browser bounded; tool arguments do not."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    projection = (APP_DIR / "src" / "event-projection.mjs").as_uri()
    script = f"""
      import {{ normalizePiEvent, projectTranscriptMessage }} from {json.dumps(projection)};
      const long = 'x'.repeat(5000);
      const events = [
        normalizePiEvent({{ type: 'message_update', assistantMessageEvent: {{ type: 'thinking_start' }} }}),
        normalizePiEvent({{ type: 'message_update', assistantMessageEvent: {{ type: 'thinking_delta', delta: long }} }}),
        normalizePiEvent({{ type: 'message_update', assistantMessageEvent: {{ type: 'thinking_end' }} }}),
        normalizePiEvent({{ type: 'message_update', assistantMessageEvent: {{ type: 'toolcall_delta', delta: 'secret-args' }} }}),
      ];
      const transcript = projectTranscriptMessage({{
        role: 'assistant', timestamp: 1720000000000,
        content: [
          {{ type: 'thinking', thinking: '**Listing eICU data sources**\\n\\n', thinkingSignature: 'sig-must-not-leak' }},
          {{ type: 'thinking', thinking: '   ' }},
          {{ type: 'toolCall', id: 'call-1', name: 'easyicu_list_data_sources', arguments: {{ secret: 'must-not-leak' }} }},
        ],
      }});
      console.log(JSON.stringify({{ events, transcript }}));
    """
    completed = subprocess.run(
        [node, "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert [event["type"] if event else None for event in payload["events"]] == [
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        None,
    ]
    assert len(payload["events"][1]["delta"]) == 4000
    assert [part["type"] for part in payload["transcript"]["content"]] == ["thinking", "tool_call"]
    assert payload["transcript"]["content"][0] == {"type": "thinking", "text": "**Listing eICU data sources**\n\n"}
    assert "must-not-leak" not in completed.stdout
