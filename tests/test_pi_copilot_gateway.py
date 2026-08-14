"""Protocol and pinned-runtime tests for the local Pi sidecar."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
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
APP_DIR = REPO_ROOT / "src" / "easyicu" / "webserver" / "pi_copilot" / "node_app"


def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def _tool_request(
    *,
    request_id: str,
    parent_request_id: str,
    session_id: str,
    name: str,
) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "tool_request",
        "request_id": request_id,
        "parent_request_id": parent_request_id,
        "session_id": session_id,
        "method": "tool.execute",
        "params": {"name": name, "arguments": {}},
    }


def _enable_tool_dispatcher(gateway: PiGatewayClient) -> None:
    gateway._tool_dispatcher = gateway._new_tool_dispatcher()


def test_pi_packages_and_upstream_commit_are_exactly_pinned() -> None:
    package = json.loads((APP_DIR / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((APP_DIR / "package-lock.json").read_text(encoding="utf-8"))

    assert package["dependencies"]["@earendil-works/pi-coding-agent"] == "0.84.1"
    assert package["dependencies"]["@earendil-works/pi-ai"] == "0.84.1"
    assert package["overrides"]["@earendil-works/pi-agent-core"] == "0.84.1"
    assert (
        lock["packages"][""]["dependencies"]["@earendil-works/pi-coding-agent"]
        == "0.84.1"
    )
    entrypoint = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    assert "9dd90a49711d088b86fdd9b4aea575913a8328a8" in entrypoint
    assert 'noTools: "builtin"' in entrypoint
    assert (
        'content: [{ type: "text", text: JSON.stringify(modelVisible) }]' in entrypoint
    )
    assert "details: modelVisible" in entrypoint
    assert "isError: true" in entrypoint
    assert 'error?.code || "pi_host_tool_rejected"' in entrypoint
    projection = (APP_DIR / "src" / "event-projection.mjs").read_text(encoding="utf-8")
    assert "details.summary" in projection
    assert 'receipt.status === "blocked"' in projection
    assert 'receipt.status === "failed"' in projection
    for name in (
        "easyicu_update_study_context",
        "easyicu_mine_ideas",
        "easyicu_search_literature",
        "easyicu_prepare_idea_handoff",
        "easyicu_accept_idea_handoff",
        "easyicu_start_extraction",
        "easyicu_run",
        "easyicu_cancel",
        "easyicu_request_replan",
    ):
        declaration = entrypoint.split(f'name: "{name}"', 1)[1].split("}),", 1)[0]
        assert 'executionMode: "sequential"' in declaration


def test_private_runtime_integrity_is_verified_once_per_gateway_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_dir = tmp_path / "private-runtime"
    app_dir.mkdir()
    gateway = PiGatewayClient(app_dir=app_dir, session_dir=tmp_path / "sessions")
    checks: list[Path] = []

    def verify(path: Path) -> bool:
        checks.append(Path(path))
        return True

    monkeypatch.setattr(
        "easyicu.webserver.pi_copilot.gateway.runtime_is_installed",
        verify,
    )
    monkeypatch.setattr(gateway, "_node_binary", lambda: "/usr/local/bin/node")
    monkeypatch.setattr(gateway, "_node_version", lambda node: (24, 11, 0))

    assert gateway.installation_status()["runtime_integrity_verified"] is True
    assert gateway.installation_status()["runtime_integrity_verified"] is True
    assert checks == [app_dir.resolve()]

    gateway.close()

    assert gateway.installation_status()["runtime_integrity_verified"] is True
    assert checks == [app_dir.resolve(), app_dir.resolve()]


def test_upstream_multi_tool_batch_serializes_authority_mutations() -> None:
    node = shutil.which("node")
    if not node or not (APP_DIR / "node_modules").is_dir():
        pytest.skip("Pinned Pi Node runtime is unavailable")
    script = r"""
import { runAgentLoop } from "./node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-agent-core/dist/agent-loop.js";
import { Type } from "typebox";

const outcomes = [];
let authorityRevision = 1;
const result = (code) => ({ content: [{ type: "text", text: code }], details: { code } });
const mutate = (name) => ({
  name,
  label: name,
  description: name,
  parameters: Type.Object({}, { additionalProperties: false }),
  executionMode: "sequential",
  execute: async () => {
    const observed = authorityRevision;
    await new Promise((resolve) => setTimeout(resolve, 5));
    if (observed !== authorityRevision) {
      outcomes.push("pi_session_authority_stale");
      throw new Error("pi_session_authority_stale");
    }
    if (name === "easyicu_run" && authorityRevision !== 1) {
      outcomes.push("pi_session_authority_stale");
      throw new Error("pi_session_authority_stale");
    }
    authorityRevision += 1;
    outcomes.push(name + ":ok");
    return result(name + "_ok");
  },
});
const tools = [mutate("easyicu_update_study_context"), mutate("easyicu_run")];
const user = { role: "user", content: "configure then run", timestamp: Date.now() };
const assistant = (content, stopReason) => ({
  role: "assistant", content, stopReason, timestamp: Date.now(),
  api: "openai-completions", provider: "test", model: "test",
  usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
});
const responses = [
  assistant([
    { type: "toolCall", id: "configure-1", name: "easyicu_update_study_context", arguments: {} },
    { type: "toolCall", id: "run-1", name: "easyicu_run", arguments: {} },
  ], "toolUse"),
  assistant([{ type: "text", text: "done" }], "stop"),
];
const streamFn = async () => {
  const message = responses.shift();
  return {
    async *[Symbol.asyncIterator]() {
      yield { type: "start", partial: message };
      yield { type: "done", reason: message.stopReason, message };
    },
    result: async () => message,
  };
};
await runAgentLoop(
  [user],
  { systemPrompt: "test", messages: [], tools },
  {
    model: { api: "openai-completions", provider: "test", id: "test" },
    convertToLlm: async (messages) => messages,
    toolExecution: "parallel",
  },
  async () => {},
  undefined,
  streamFn,
);
if (outcomes.filter((item) => item.endsWith(":ok")).length !== 1) throw new Error(JSON.stringify(outcomes));
if (outcomes.filter((item) => item === "pi_session_authority_stale").length !== 1) throw new Error(JSON.stringify(outcomes));
"""
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        cwd=APP_DIR,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_host_reauthorizes_tool_request_and_rejects_unknown_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def execute(name, arguments, context):
        calls.append((name, arguments, context))
        return {
            "status": "ok",
            "code": "test_ok",
            "summary": "ok",
            "owner": "test",
            "details": {},
            "authority": {},
        }

    gateway = PiGatewayClient(
        app_dir=APP_DIR, session_dir=tmp_path, tool_executor=execute
    )
    _enable_tool_dispatcher(gateway)
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
    assert gateway._tool_dispatcher is not None
    assert gateway._tool_dispatcher.wait_until_idle(2)
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
    gateway.close()


def test_slow_tool_in_one_session_does_not_block_other_session_or_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slow_started = threading.Event()
    release_slow = threading.Event()
    writes: list[dict] = []
    events: list[dict] = []

    def execute(name, arguments, context):
        if name == "slow":
            slow_started.set()
            assert release_slow.wait(2)
        return {"code": f"{name}_ok"}

    gateway = PiGatewayClient(
        app_dir=APP_DIR,
        session_dir=tmp_path,
        tool_executor=execute,
        tool_dispatch_max_workers=2,
    )
    _enable_tool_dispatcher(gateway)
    monkeypatch.setattr(gateway, "_write", lambda payload: writes.append(dict(payload)))
    context_a = ToolExecutionContext(session=PiSessionRecord(session_id="session-a"))
    context_b = ToolExecutionContext(session=PiSessionRecord(session_id="session-b"))
    gateway._pending["parent-a"] = _PendingRequest(tool_context=context_a)
    gateway._pending["parent-b"] = _PendingRequest(
        tool_context=context_b,
        event_sink=lambda event: events.append(dict(event)),
    )

    gateway._handle_tool_request(
        _tool_request(
            request_id="tool-a",
            parent_request_id="parent-a",
            session_id="session-a",
            name="slow",
        )
    )
    assert slow_started.wait(1)
    gateway._handle_tool_request(
        _tool_request(
            request_id="tool-b",
            parent_request_id="parent-b",
            session_id="session-b",
            name="fast",
        )
    )
    gateway._handle_payload(
        {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "event",
            "request_id": "parent-b",
            "session_id": "session-b",
            "event": {"type": "progress", "summary": "still responsive"},
        }
    )

    _wait_for(lambda: any(row.get("request_id") == "tool-b" for row in writes))
    assert not any(row.get("request_id") == "tool-a" for row in writes)
    assert events == [{"type": "progress", "summary": "still responsive"}]

    release_slow.set()
    assert gateway._tool_dispatcher is not None
    assert gateway._tool_dispatcher.wait_until_idle(2)
    gateway.close()


def test_host_tools_are_strictly_ordered_within_one_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    order: list[str] = []
    writes: list[dict] = []
    write_threads: list[str] = []

    def execute(name, arguments, context):
        order.append(f"{name}:start")
        if name == "first":
            first_started.set()
            assert release_first.wait(2)
        order.append(f"{name}:end")
        return {"code": f"{name}_ok"}

    gateway = PiGatewayClient(
        app_dir=APP_DIR,
        session_dir=tmp_path,
        tool_executor=execute,
        tool_dispatch_max_workers=2,
    )
    _enable_tool_dispatcher(gateway)
    def record_write(payload):
        writes.append(dict(payload))
        write_threads.append(threading.current_thread().name)

    monkeypatch.setattr(gateway, "_write", record_write)
    context = ToolExecutionContext(session=PiSessionRecord(session_id="same-session"))
    gateway._pending["parent-1"] = _PendingRequest(tool_context=context)
    gateway._pending["parent-2"] = _PendingRequest(tool_context=context)

    gateway._handle_tool_request(
        _tool_request(
            request_id="tool-1",
            parent_request_id="parent-1",
            session_id="same-session",
            name="first",
        )
    )
    assert first_started.wait(1)
    gateway._handle_tool_request(
        _tool_request(
            request_id="tool-2",
            parent_request_id="parent-2",
            session_id="same-session",
            name="second",
        )
    )
    time.sleep(0.03)
    assert order == ["first:start"]

    release_first.set()
    assert gateway._tool_dispatcher is not None
    assert gateway._tool_dispatcher.wait_until_idle(2)
    assert order == ["first:start", "first:end", "second:start", "second:end"]
    assert [row["request_id"] for row in writes] == ["tool-1", "tool-2"]
    assert set(write_threads) == {"easyicu-pi-host-tool-writer"}
    gateway.close()


def test_host_tool_queue_capacity_and_shutdown_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    writes: list[dict] = []

    def execute(name, arguments, context):
        started.set()
        assert release.wait(2)
        return {"code": "late_success"}

    gateway = PiGatewayClient(
        app_dir=APP_DIR,
        session_dir=tmp_path,
        tool_executor=execute,
        tool_dispatch_max_workers=1,
        tool_dispatch_max_pending=1,
    )
    _enable_tool_dispatcher(gateway)
    monkeypatch.setattr(gateway, "_write", lambda payload: writes.append(dict(payload)))
    for suffix in ("a", "b"):
        gateway._pending[f"parent-{suffix}"] = _PendingRequest(
            tool_context=ToolExecutionContext(
                session=PiSessionRecord(session_id=f"session-{suffix}")
            )
        )

    gateway._handle_tool_request(
        _tool_request(
            request_id="tool-a",
            parent_request_id="parent-a",
            session_id="session-a",
            name="slow",
        )
    )
    assert started.wait(1)
    gateway._handle_tool_request(
        _tool_request(
            request_id="tool-b",
            parent_request_id="parent-b",
            session_id="session-b",
            name="never-run",
        )
    )
    assert writes[-1]["error"]["code"] == "pi_host_tool_dispatcher_full"

    gateway.close()
    _wait_for(lambda: any(row.get("request_id") == "tool-a" for row in writes))
    closed = next(row for row in writes if row.get("request_id") == "tool-a")
    assert closed["ok"] is False
    assert closed["error"]["code"] == "pi_host_tool_dispatcher_closed"
    release.set()
    time.sleep(0.03)
    assert len([row for row in writes if row.get("request_id") == "tool-a"]) == 1


def test_host_tool_exception_is_sanitized_by_dispatcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[dict] = []

    def execute(name, arguments, context):
        raise RuntimeError("secret traceback detail")

    gateway = PiGatewayClient(
        app_dir=APP_DIR,
        session_dir=tmp_path,
        tool_executor=execute,
    )
    _enable_tool_dispatcher(gateway)
    monkeypatch.setattr(gateway, "_write", lambda payload: writes.append(dict(payload)))
    gateway._pending["parent"] = _PendingRequest(
        tool_context=ToolExecutionContext(
            session=PiSessionRecord(session_id="session")
        )
    )
    gateway._handle_tool_request(
        _tool_request(
            request_id="tool",
            parent_request_id="parent",
            session_id="session",
            name="boom",
        )
    )
    assert gateway._tool_dispatcher is not None
    assert gateway._tool_dispatcher.wait_until_idle(2)
    assert writes[-1]["error"]["code"] == "pi_host_tool_failed"
    assert "secret" not in writes[-1]["error"]["message"]
    gateway.close()


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
            "EASYICU_PI_INPUT_PRICE_USD_PER_1M_TOKENS": "1.25",
            "EASYICU_PI_OUTPUT_PRICE_USD_PER_1M_TOKENS": "5",
            "EASYICU_PI_MAX_COST_USD_PER_MESSAGE": "0.5",
            "EASYICU_PI_MAX_COST_USD_PER_SESSION": "5",
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
    assert gateway.environ["EASYICU_PI_INPUT_PRICE_USD_PER_1M_TOKENS"] == "1.25"
    assert gateway.environ["EASYICU_PI_MAX_COST_USD_PER_SESSION"] == "5"
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
    budget = (APP_DIR / "src" / "shell-budget.mjs").read_text(encoding="utf-8")
    projection = (APP_DIR / "src" / "event-projection.mjs").read_text(encoding="utf-8")

    assert "EASYICU_PI_SESSION_TOKEN_BUDGET" in source
    assert "pi_shell_token_budget_exhausted" in budget
    assert "EASYICU_PI_MAX_COST_USD_PER_MESSAGE" in source
    assert "pi_shell_session_cost_budget_exhausted" in budget
    assert "record.budgetGuard.authorize(context, options)" in source
    assert "maxRetries: 0" in source
    assert 'update.type === "thinking_delta"' not in source + projection
    assert 'item.type === "thinking"' not in source + projection
    assert "normalizePiEvent" in source
    assert "projectTranscriptMessage" in source


def test_shell_budget_guard_blocks_each_provider_boundary() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    module = APP_DIR / "src" / "shell-budget.mjs"
    script = f"""
      import {{ providerCallReceipt, restoredProviderCallCount, ShellBudgetGuard }} from {json.dumps(module.as_uri())};
      const tokenGuard = new ShellBudgetGuard({{
        tokenBudget: 5000,
        maxOutputTokens: 1000,
        maxProviderCallsPerMessage: 2,
        maxProviderCallsPerSession: 3,
        consumedTokens: () => 100,
      }});
      tokenGuard.beginMessage();
      try {{ tokenGuard.authorize({{systemPrompt: 'x', messages: []}}, {{}}); }}
      catch (error) {{ console.log(error.code); }}

      const callGuard = new ShellBudgetGuard({{
        tokenBudget: 50000,
        maxOutputTokens: 1000,
        maxProviderCallsPerMessage: 1,
        maxProviderCallsPerSession: 3,
        consumedTokens: () => 0,
      }});
      callGuard.beginMessage();
      callGuard.authorize({{messages: []}}, {{}});
      try {{ callGuard.authorize({{messages: []}}, {{}}); }}
      catch (error) {{ console.log(error.code); }}

      const entries = [
        {{ type: 'message', role: 'assistant' }},
        {{ type: 'custom', customType: 'easyicu.shell-budget/1', data: providerCallReceipt(9) }},
        {{ type: 'compaction', usage: {{ totalTokens: 200 }} }},
      ];
      console.log(restoredProviderCallCount(entries, 1));

      const priced = new ShellBudgetGuard({{
        tokenBudget: 50000,
        maxOutputTokens: 1000,
        maxProviderCallsPerMessage: 2,
        maxProviderCallsPerSession: 4,
        consumedTokens: () => 0,
        pricing: {{
          inputPriceUsdPerMillionTokens: 10,
          outputPriceUsdPerMillionTokens: 30,
          maxCostUsdPerMessage: 0.08,
          maxCostUsdPerSession: 0.12,
        }},
      }});
      priced.beginMessage();
      priced.authorize({{messages: []}}, {{maxTokens: 1000}});
      const receipt = priced.receipt();
      console.log(receipt.schema_version, receipt.reserved_cost_micro_usd, priced.state().pricing_available);
      priced.endMessage();

      const restored = new ShellBudgetGuard({{
        tokenBudget: 50000,
        maxOutputTokens: 1000,
        maxProviderCallsPerMessage: 2,
        maxProviderCallsPerSession: 4,
        consumedTokens: () => 0,
        persistedEntries: [{{ type: 'custom', customType: receipt.schema_version, data: receipt }}],
        pricing: {{
          inputPriceUsdPerMillionTokens: 10,
          outputPriceUsdPerMillionTokens: 30,
          maxCostUsdPerMessage: 0.08,
          maxCostUsdPerSession: 0.12,
        }},
      }});
      restored.beginMessage();
      try {{ restored.authorize({{messages: []}}, {{maxTokens: 1000}}); }}
      catch (error) {{ console.log(error.code); }}

      try {{
        new ShellBudgetGuard({{
          tokenBudget: 50000,
          maxOutputTokens: 1000,
          maxProviderCallsPerMessage: 2,
          maxProviderCallsPerSession: 4,
          consumedTokens: () => 0,
          persistedEntries: entries,
          pricing: {{
            inputPriceUsdPerMillionTokens: 10,
            outputPriceUsdPerMillionTokens: 30,
            maxCostUsdPerMessage: 0.08,
            maxCostUsdPerSession: 0.12,
          }},
        }});
      }} catch (error) {{ console.log(error.code); }}

      try {{
        new ShellBudgetGuard({{
          tokenBudget: 50000,
          maxOutputTokens: 1000,
          maxProviderCallsPerMessage: 2,
          maxProviderCallsPerSession: 4,
          consumedTokens: () => 0,
          persistedEntries: [{{ type: 'custom', customType: receipt.schema_version, data: receipt }}],
        }});
      }} catch (error) {{ console.log(error.code); }}
    """
    completed = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines() == [
        "pi_shell_token_budget_exhausted",
        "pi_shell_message_provider_call_budget_exhausted",
        "9",
        "easyicu.shell-budget/2 71110 true",
        "pi_shell_session_cost_budget_exhausted",
        "pi_shell_cost_history_unavailable",
        "pi_shell_pricing_binding_mismatch",
    ]


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
      const blockedEvent = normalizePiEvent({{
        type: 'tool_execution_end', toolCallId: 'call-2', toolName: 'easyicu_run',
        isError: false,
        result: {{ details: {{ status: 'blocked', code: 'pi_session_authority_stale', summary: 'Blocked', owner: 'easyicu.pi' }} }},
      }});
      const blockedTranscript = projectTranscriptMessage({{
        role: 'toolResult', toolCallId: 'call-2', toolName: 'easyicu_run', isError: false,
        details: {{ status: 'blocked', code: 'pi_session_authority_stale', summary: 'Blocked', owner: 'easyicu.pi' }},
      }});
      const workspaceStart = normalizePiEvent({{
        type: 'tool_execution_start', toolCallId: 'call-3',
        toolName: 'easyicu_read_project_file',
        args: {{ file: 'prototype/index.html', secret: 'must-not-leak' }},
      }});
      const workspaceEnd = normalizePiEvent({{
        type: 'tool_execution_end', toolCallId: 'call-3',
        toolName: 'easyicu_read_project_file',
        result: {{ details: {{ status: 'ok', code: 'pi_workspace_file_read',
          summary: 'Read index.html', owner: 'easyicu.workspace',
          details: {{ resource: {{ kind: 'file', file: 'prototype/index.html', label: 'index.html', media_type: 'text/html' }},
            text: 'must-not-leak' }} }} }},
      }});
      const unsafeWorkspace = normalizePiEvent({{
        type: 'tool_execution_start', toolCallId: 'call-4',
        toolName: 'easyicu_read_project_file', args: {{ file: '../secret.txt' }},
      }});
      const researchArtifacts = normalizePiEvent({{
        type: 'tool_execution_end', toolCallId: 'call-5', toolName: 'easyicu_list_artifacts',
        result: {{ details: {{ status: 'ok', code: 'easyicu_artifacts_projected',
          summary: 'Listed artifacts', owner: 'easyicu.agent_runs', details: {{
            project_dir: 'must-not-leak', resources: [
              {{ kind: 'research_artifact', run_id: 'run_20260808', artifact: 'table1_summary.json', label: 'Table 1', media_type: 'application/json' }},
              {{ kind: 'research_artifact', run_id: '../unsafe', artifact: '../secret.json' }},
            ]
          }} }} }},
      }});
      const dataPackageReview = normalizePiEvent({{
        type: 'tool_execution_end', toolCallId: 'call-data-package',
        toolName: 'easyicu_inspect_data_package', result: {{ details: {{
          status: 'ok', code: 'easyicu_data_package_review_ready',
          summary: 'Data package ready', owner: 'easyicu.data_package_review',
          details: {{ resource: {{ kind: 'data_package_review',
            study_context_id: 'study_review', study_revision: 7,
            review_sha256: '{'d' * 64}', label: 'Data package review',
            media_type: 'application/json', source_path: 'must-not-leak' }} }}
        }} }} }});
      const submittedRun = normalizePiEvent({{
        type: 'tool_execution_end', toolCallId: 'call-6', toolName: 'easyicu_run',
        result: {{ details: {{ status: 'ok', code: 'easyicu_full_run_submitted',
          summary: 'Submitted full run', owner: 'easyicu.agent', details: {{
            job_id: '6a2bf5684685', project_dir: 'must-not-leak'
          }} }} }},
      }});
      const unsafeJob = normalizePiEvent({{
        type: 'tool_execution_end', toolCallId: 'call-7', toolName: 'easyicu_run',
        result: {{ details: {{ status: 'ok', code: 'easyicu_full_run_submitted',
          summary: 'Submitted full run', owner: 'easyicu.agent', details: {{
            job_id: '../unsafe'
          }} }} }},
      }});
      const providerErrorEvent = normalizePiEvent({{
        type: 'message_end', message: {{ role: 'assistant', stopReason: 'error',
          errorMessage: 'dial tcp 203.0.113.1:443: connect: operation timed out' }}
      }});
      const providerErrorTranscript = projectTranscriptMessage({{
        role: 'assistant', content: [], stopReason: 'error',
        errorMessage: 'dial tcp 203.0.113.1:443: connect: operation timed out'
      }});
      const shellBudgetEvent = normalizePiEvent({{
        type: 'message_end', message: {{ role: 'assistant', stopReason: 'error',
          errorMessage: 'pi_shell_token_budget_exhausted: bounded session budget reached' }}
      }});
      const shellBudgetTranscript = projectTranscriptMessage({{
        role: 'assistant', content: [], stopReason: 'error',
        errorMessage: 'pi_shell_token_budget_exhausted: bounded session budget reached'
      }});
      console.log(JSON.stringify({{ events, transcript, blockedEvent, blockedTranscript, workspaceStart, workspaceEnd, unsafeWorkspace, researchArtifacts, dataPackageReview, submittedRun, unsafeJob, providerErrorEvent, providerErrorTranscript, shellBudgetEvent, shellBudgetTranscript }}));
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
    assert payload["blockedEvent"]["is_error"] is True
    assert payload["blockedEvent"]["code"] == "pi_session_authority_stale"
    assert payload["blockedTranscript"]["content"][0]["is_error"] is True
    assert payload["workspaceStart"]["resource"] == {
        "kind": "file",
        "file": "prototype/index.html",
        "label": "index.html",
        "media_type": "text/plain",
    }
    assert payload["workspaceEnd"]["resource"]["media_type"] == "text/html"
    assert "resource" not in payload["unsafeWorkspace"]
    assert payload["researchArtifacts"]["resources"] == [{
        "kind": "research_artifact",
        "run_id": "run_20260808",
        "artifact": "table1_summary.json",
        "label": "Table 1",
        "media_type": "application/json",
    }]
    assert payload["dataPackageReview"]["resource"] == {
        "kind": "data_package_review",
        "study_context_id": "study_review",
        "study_revision": 7,
        "review_sha256": "d" * 64,
        "label": "Data package review",
        "media_type": "application/json",
    }
    assert payload["submittedRun"]["job_id"] == "6a2bf5684685"
    assert "job_id" not in payload["unsafeJob"]
    assert payload["providerErrorEvent"]["error_code"] == "pi_model_provider_unavailable"
    assert payload["providerErrorTranscript"]["error_code"] == "pi_model_provider_unavailable"
    assert payload["shellBudgetEvent"]["error_code"] == "pi_shell_token_budget_exhausted"
    assert payload["shellBudgetTranscript"]["error_code"] == "pi_shell_token_budget_exhausted"
    assert "raw result must not leak" not in completed.stdout
    assert "project_dir" not in completed.stdout
    assert "must-not-leak" not in completed.stdout
    assert "203.0.113.1" not in completed.stdout


def test_sidecar_projects_only_verified_literature_click_targets() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    projection = (APP_DIR / "src" / "event-projection.mjs").as_uri()
    script = f"""
      import {{ normalizePiEvent }} from {json.dumps(projection)};
      const result = normalizePiEvent({{
        type: 'tool_execution_end', toolCallId: 'lit-1',
        toolName: 'easyicu_search_literature', result: {{ details: {{
          status: 'ok', code: 'easyicu_literature_search_completed',
          summary: 'Search complete', owner: 'easyicu.ideas', details: {{
            host_rebind_after_turn: true, resources: [
            {{ kind: 'literature_source', title: 'Source-backed article',
               url: 'https://pubmed.ncbi.nlm.nih.gov/12345/', pmid: '12345',
               venue: 'Critical Care', year: '2025' }},
            {{ kind: 'literature_source', title: '<img src=x onerror=alert(1)>',
               url: 'javascript:alert(1)', pmid: '999' }}
          ] }}
        }} }} }});
      console.log(JSON.stringify(result));
    """
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["resources"] == [
        {
            "kind": "literature_source",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
            "label": "Source-backed article",
            "title": "Source-backed article",
            "year": "2025",
            "venue": "Critical Care",
            "relevance": "",
            "doi": "",
            "pmid": "12345",
            "media_type": "text/html",
            "authority_class": "literature_retrieval_candidate",
        }
    ]
    assert payload["host_rebind_after_turn"] is True


def test_research_system_prompt_routes_short_execution_intent_to_run_owner() -> None:
    entrypoint = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    assert "treat that as execution intent rather than a request to inspect an older run" in entrypoint
    assert "then call easyicu_run" in entrypoint
    assert "Use easyicu_inspect_run only when the user asks for status" in entrypoint
    assert "A persisted run_id is historical evidence, not proof of an active job" in entrypoint
    assert "run_id_status=pending_pipeline_start" in entrypoint
    assert "save that commitment in typed analysis_design" in entrypoint
    assert "typed analysis_design.analysis_family" in entrypoint
    assert "Never upgrade a descriptive unadjusted noncausal contrast" in entrypoint
    assert "Never call easyicu_resume without an approved/rejected decision" in entrypoint
    assert "an explicit user rerun request must call easyicu_run" in entrypoint
    assert "Treat every scientific question as one ordinary research project" in entrypoint
    assert "Evaluation orchestration and scoring stay outside the Copilot product surface" in entrypoint
    assert "When the workflow reports failed_pipeline_requires_fresh_plan" in entrypoint
    assert "treat the terminal failed run as immutable history" in entrypoint
    assert "covariate_rationales" in entrypoint
    assert "covariate_temporal_roles" in entrypoint
    assert "save only an explicit positive choice" in entrypoint
    assert "sensitivity_spec" in entrypoint


def test_system_prompt_keeps_declined_optional_sensitivity_out_of_study_context() -> None:
    entrypoint = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    assert "the user declines, that is not a sensitivity spec or a StudyContext change" in entrypoint
    assert "call easyicu_resume with decision='approved'" in entrypoint


def test_research_system_prompt_requires_tool_first_idea_mining() -> None:
    entrypoint = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    assert "Tool-first Idea Mining rule" in entrypoint
    assert "do not author a candidate from general model knowledge" in entrypoint
    assert "call easyicu_mine_ideas before writing the answer" in entrypoint
    assert "use easyicu_accept_idea_handoff with its exact run_id and idea_id" in entrypoint


def test_research_system_prompt_does_not_guess_literature_grant_state() -> None:
    entrypoint = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    assert "call easyicu_search_literature" in entrypoint
    assert "let the host-held one-turn gate authoritatively allow or block it" in entrypoint
    assert "never infer or claim that it is absent before the tool returns" in entrypoint


def test_system_prompt_keeps_copilot_replies_concise_while_preserving_blockers() -> None:
    entrypoint = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    assert "Match the user's language and brevity" in entrypoint
    assert "use at most two short sentences around tool calls" in entrypoint
    assert "ask one direct question and stop" in entrypoint
    assert "independently answerable scientific decision" in entrypoint
    assert "Do not bundle cohort, estimand, timing, adjustment" in entrypoint
    assert "do not recommend model-based or heteroskedasticity-robust variance" in entrypoint
    assert "offer only the safe_alternatives returned by the host" in entrypoint
    assert "instead of inventing an executable cohort" in entrypoint
    assert "Typed time-window rule" in entrypoint
    assert "It is not a phenotype's clinical definition anchor" in entrypoint
    assert "never save suspected-infection onset as its physical anchor" in entrypoint
    assert "never send a questionnaire or numbered list of confirmations" in entrypoint
    assert "Never hide a blocker or weaken its exact stable code" in entrypoint


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
            },
            timeout=30,
        )
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
    assert len(state["enabled_tools"]) == 30
    assert {
        "easyicu_list_extensions",
        "easyicu_load_skill",
        "easyicu_call_mcp_tool",
    }.issubset(state["enabled_tools"])
    assert {"read", "write", "edit", "bash"}.isdisjoint(state["enabled_tools"])
    assert workspace_state["agent_mode"] == "workspace"
    assert workspace_state["enabled_tools"] == runtime["custom_tools_by_mode"]["workspace"]
    assert len(workspace_state["enabled_tools"]) == 36
    assert {"read", "write", "edit", "bash"}.isdisjoint(
        workspace_state["enabled_tools"]
    )
    assert reopened_state["session_file"] == str(session_file)
    assert reopened_state["agent_mode"] == "research"
    assert reopened_state["enabled_tools"] == state["enabled_tools"]
