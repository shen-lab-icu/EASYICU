"""A conversation turn's deadline does not charge the time its own host tools take.

The sidecar bounds every host tool itself (``pi_host_tool_timeout``); the host's
turn deadline bounds the Pi/model side.  A slow submission tool used to exhaust
the turn deadline while its job was still being created, so the conversation
reported a failure for work that went on to run.  Timings here are real but
coarse: every margin is at least 150 ms.
"""

from __future__ import annotations

import math
import re
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

from easyicu.webserver.pi_copilot import gateway as gateway_module
from easyicu.webserver.pi_copilot.contracts import (
    PROTOCOL_VERSION,
    PiCopilotError,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.gateway import (
    PiGatewayClient,
    _HostToolClock,
    _PendingRequest,
)
from easyicu.webserver.pi_copilot.host_tool_dispatcher import HostToolDispatchRejected

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = REPO_ROOT / "src" / "easyicu" / "webserver" / "pi_copilot" / "node_app"
SESSION_ID = "session-budget"


def _tool_request(*, parent_request_id: str, request_id: str = "tool-1") -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "tool_request",
        "request_id": request_id,
        "parent_request_id": parent_request_id,
        "session_id": SESSION_ID,
        "method": "tool.execute",
        "params": {"name": "easyicu_request_replan", "arguments": {}},
    }


class _Sidecar:
    """Plays the Node sidecar for one prompt: think, call a host tool, think, answer."""

    def __init__(
        self,
        gateway: PiGatewayClient,
        *,
        think_before: float,
        think_after: Optional[float],
        calls_tool: bool = True,
    ) -> None:
        self.gateway = gateway
        self.think_before = think_before
        self.think_after = think_after
        self.calls_tool = calls_tool
        self.writes: list[dict] = []
        self.tool_answered = threading.Event()

    def write(self, payload: dict, **_transport: object) -> None:
        self.writes.append(dict(payload))
        if payload.get("kind") == "request":
            threading.Thread(
                target=self._turn, args=(payload["request_id"],), daemon=True
            ).start()
        elif payload.get("kind") == "tool_response":
            self.tool_answered.set()

    def _turn(self, request_id: str) -> None:
        time.sleep(self.think_before)
        if self.calls_tool:
            self.gateway._handle_tool_request(
                _tool_request(parent_request_id=request_id)
            )
            if not self.tool_answered.wait(10):
                return
        if self.think_after is None:
            return
        time.sleep(self.think_after)
        self.gateway._handle_payload(
            {
                "protocol_version": PROTOCOL_VERSION,
                "kind": "response",
                "request_id": request_id,
                "ok": True,
                "result": {"state": "done"},
            }
        )


def _gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tool_seconds: float,
    think_before: float,
    think_after: Optional[float],
    calls_tool: bool = True,
) -> tuple[PiGatewayClient, _Sidecar, list[str]]:
    def execute(name, arguments, context):
        time.sleep(tool_seconds)
        return {"code": "easyicu_full_run_submitted"}

    gateway = PiGatewayClient(app_dir=APP_DIR, session_dir=tmp_path, tool_executor=execute)
    gateway._tool_dispatcher = gateway._new_tool_dispatcher()
    sidecar = _Sidecar(
        gateway,
        think_before=think_before,
        think_after=think_after,
        calls_tool=calls_tool,
    )
    monkeypatch.setattr(gateway, "_write", sidecar.write)
    recovered: list[str] = []
    monkeypatch.setattr(gateway, "_recover_timed_out_prompt", recovered.append)
    return gateway, sidecar, recovered


def _prompt(gateway: PiGatewayClient, *, timeout: float) -> dict:
    return gateway._request_started(
        "session.prompt",
        {"session_id": SESSION_ID, "message": "regenerate the plan"},
        timeout=timeout,
        tool_context=ToolExecutionContext(session=PiSessionRecord(session_id=SESSION_ID)),
    )


def _close(gateway: PiGatewayClient) -> None:
    assert gateway._tool_dispatcher is not None
    assert gateway._tool_dispatcher.wait_until_idle(5)
    gateway.close()


def test_a_host_tool_longer_than_the_turn_budget_does_not_fail_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, sidecar, recovered = _gateway(
        tmp_path, monkeypatch, tool_seconds=1.0, think_before=0.05, think_after=0.05
    )

    assert _prompt(gateway, timeout=0.4) == {"state": "done"}

    assert recovered == []
    answer = next(row for row in sidecar.writes if row.get("kind") == "tool_response")
    assert answer["ok"] is True
    assert answer["result"] == {"code": "easyicu_full_run_submitted"}
    _close(gateway)


def test_model_time_alone_still_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, _sidecar, recovered = _gateway(
        tmp_path,
        monkeypatch,
        tool_seconds=0.0,
        think_before=0.0,
        think_after=None,
        calls_tool=False,
    )
    started = time.monotonic()

    with pytest.raises(PiCopilotError) as caught:
        _prompt(gateway, timeout=0.3)

    assert caught.value.code == "pi_gateway_timeout"
    assert recovered == [SESSION_ID]
    assert time.monotonic() - started < 2.0
    _close(gateway)


def test_model_time_on_both_sides_of_a_tool_adds_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 0.35 s + 0.35 s of model time exceeds the 0.5 s budget; the 0.3 s tool
    # in between is not charged, so the deadline falls after the tool.
    gateway, _sidecar, recovered = _gateway(
        tmp_path, monkeypatch, tool_seconds=0.3, think_before=0.35, think_after=0.35
    )
    started = time.monotonic()

    with pytest.raises(PiCopilotError) as caught:
        _prompt(gateway, timeout=0.5)

    elapsed = time.monotonic() - started
    assert caught.value.code == "pi_gateway_timeout"
    assert recovered == [SESSION_ID]
    assert elapsed >= 0.7
    _close(gateway)


def test_a_tool_past_the_sidecar_deadline_is_charged_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sidecar gives up on a tool at its own deadline; past that point the
    # wait is charged to the turn again instead of waiting on the tool.
    monkeypatch.setattr(gateway_module, "HOST_TOOL_SIDECAR_TIMEOUT_SECONDS", 0.3)
    gateway, _sidecar, recovered = _gateway(
        tmp_path, monkeypatch, tool_seconds=2.0, think_before=0.05, think_after=0.05
    )
    started = time.monotonic()

    with pytest.raises(PiCopilotError) as caught:
        _prompt(gateway, timeout=0.5)

    elapsed = time.monotonic() - started
    assert caught.value.code == "pi_gateway_timeout"
    assert recovered == [SESSION_ID]
    assert 0.7 <= elapsed < 1.6
    _close(gateway)


def test_the_host_limit_is_the_sidecar_tool_timer() -> None:
    source = (APP_DIR / "src" / "main.mjs").read_text(encoding="utf-8")
    block = source[source.index("async function requestHostTool") :]
    block = block[: block.index("\nfunction ")]
    timer = re.search(
        r'code: "pi_host_tool_timeout" \}\)\);\s*\},\s*([\d\s*]+)\);', block
    )

    assert timer is not None
    milliseconds = math.prod(int(value) for value in re.findall(r"\d+", timer.group(1)))
    assert milliseconds == gateway_module.HOST_TOOL_SIDECAR_TIMEOUT_SECONDS * 1000


def test_the_clock_counts_queued_and_overlapping_tools_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    clock = _HostToolClock(clock=lambda: now[0])

    clock.start("a")
    now[0] = 1.0
    clock.start("b")  # queued behind "a" in the same session
    now[0] = 3.0
    clock.finish("a")
    now[0] = 4.0
    clock.finish("b")
    assert clock.seconds(10.0) == pytest.approx(4.0)

    now[0] = 6.0
    clock.start("c")
    now[0] = 7.0
    clock.finish("c")
    now[0] = 7.5
    clock.finish("c")  # a second answer does not extend the span
    now[0] = 8.0
    clock.start("d")  # still running
    assert clock.seconds(9.5) == pytest.approx(6.5)

    monkeypatch.setattr(gateway_module, "HOST_TOOL_SIDECAR_TIMEOUT_SECONDS", 2.0)
    # a: 0-2, b: 1-3, c: 6-7, d: 8-10 -> 3 + 1 + 2
    assert clock.seconds(20.0) == pytest.approx(6.0)


def test_a_rejected_tool_does_not_pause_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FullDispatcher:
        def submit(self, **_task: object) -> None:
            raise HostToolDispatchRejected(
                "pi_host_tool_dispatcher_full", "The host-tool queue is full."
            )

    now = [0.0]
    writes: list[dict] = []
    gateway = PiGatewayClient(app_dir=APP_DIR, session_dir=tmp_path)
    gateway._tool_dispatcher = _FullDispatcher()  # type: ignore[assignment]
    monkeypatch.setattr(gateway, "_write", lambda payload, **_t: writes.append(dict(payload)))
    pending = _PendingRequest(
        tool_context=ToolExecutionContext(session=PiSessionRecord(session_id=SESSION_ID)),
        host_tools=_HostToolClock(clock=lambda: now[0]),
    )
    gateway._pending["parent"] = pending

    gateway._handle_tool_request(_tool_request(parent_request_id="parent"))
    now[0] = 5.0

    assert writes[-1]["error"]["code"] == "pi_host_tool_dispatcher_full"
    assert pending.host_tools.seconds(now[0]) == 0.0
    gateway._tool_dispatcher = None
    gateway.close()
