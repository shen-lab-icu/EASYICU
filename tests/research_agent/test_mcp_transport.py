"""Official MCP SDK protocol and transport integration tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
import mcp.types as mcp_types
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session
from starlette.applications import Starlette

from easyicu.research_agent.mcp_policy import (
    MCP_ALLOWED_ROOTS_ENV,
    MCP_PATIENT_DATA_TOKEN_ENV,
    MCP_SCOPES_ENV,
    granted_scopes,
)
from easyicu.research_agent.mcp_server import (
    SERVER_INFO,
    TOOL_SCHEMAS,
    dispatch as application_dispatch,
)
from easyicu.research_agent.mcp_transport import (
    create_mcp_server,
    create_streamable_http_app,
    validate_http_server_config,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _mcp_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MCP_ALLOWED_ROOTS_ENV, str(tmp_path))
    monkeypatch.setenv(
        MCP_SCOPES_ENV,
        "metadata,run_pipeline,write_artifacts,bind_evidence",
    )


def _server(*, dispatcher=application_dispatch, **kwargs):
    return create_mcp_server(
        dispatcher=dispatcher,
        server_info=SERVER_INFO,
        tool_schemas=TOOL_SCHEMAS,
        **kwargs,
    )


@pytest.mark.anyio
async def test_official_client_initializes_lists_and_calls_tools() -> None:
    server = _server()

    async with create_connected_server_and_client_session(
        server,
        raise_exceptions=True,
    ) as session:
        initialized = await session.initialize()
        listed = await session.list_tools()
        result = await session.call_tool("research_agent.list_skills", {})

    assert initialized.serverInfo.name == "easyicu-research-agent"
    names = {tool.name for tool in listed.tools}
    assert {
        "research_agent.run",
        "research_agent.list_skills",
        "research_agent.read_manifest",
        "research_agent.list_export_concepts",
        "research_agent.assess_export_coverage",
        "research_agent.bind_evidence",
    } <= names
    assert result.isError is False
    assert result.structuredContent is not None
    keys = {item["key"] for item in result.structuredContent["skills"]}
    assert {"association_analysis", "prediction_model", "data_quality_audit"} <= keys


@pytest.mark.anyio
async def test_sdk_rejects_empty_export_coverage_before_dispatch() -> None:
    calls: list[tuple[str, dict]] = []
    server = _server(
        dispatcher=lambda name, arguments: (
            calls.append((name, arguments or {})) or {"unexpected": True}
        )
    )

    async with create_connected_server_and_client_session(
        server,
        raise_exceptions=True,
    ) as session:
        await session.initialize()
        result = await session.call_tool(
            "research_agent.assess_export_coverage",
            {"export_dir": "/not/read", "concepts": []},
        )

    assert result.isError is True
    assert "Input validation error" in result.content[0].text
    assert calls == []


@pytest.mark.anyio
async def test_sdk_validates_input_schema_before_dispatch() -> None:
    calls: list[tuple[str, dict]] = []
    server = _server(
        dispatcher=lambda name, arguments: (
            calls.append((name, arguments or {})) or {"unexpected": True}
        )
    )

    async with create_connected_server_and_client_session(
        server,
        raise_exceptions=True,
    ) as session:
        await session.initialize()
        result = await session.call_tool("research_agent.run", {})

    assert result.isError is True
    assert "Input validation error" in result.content[0].text
    assert calls == []


@pytest.mark.anyio
async def test_sdk_preserves_structured_error_contract() -> None:
    server = _server(
        dispatcher=lambda _name, _arguments: {
            "error": "scope not granted",
            "error_code": "scope_not_granted",
        }
    )

    async with create_connected_server_and_client_session(
        server,
        raise_exceptions=True,
    ) as session:
        await session.initialize()
        result = await session.call_tool("research_agent.list_skills", {})

    assert result.isError is True
    assert result.structuredContent == {
        "error": "scope not granted",
        "error_code": "scope_not_granted",
    }
    assert json.loads(result.content[0].text) == result.structuredContent


@pytest.mark.anyio
async def test_tool_timeout_returns_error_and_server_remains_usable() -> None:
    release = threading.Event()
    calls = 0

    def dispatcher(_name: str, _arguments: dict | None) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            release.wait(timeout=2)
        return {"call": calls}

    server = _server(
        dispatcher=dispatcher,
        tool_timeout_seconds=0.05,
    )
    async with create_connected_server_and_client_session(
        server,
        raise_exceptions=True,
    ) as session:
        await session.initialize()
        timed_out = await session.call_tool("research_agent.list_skills", {})
        release.set()
        await asyncio.sleep(0.05)
        completed = await session.call_tool("research_agent.list_skills", {})

    assert timed_out.isError is True
    assert timed_out.structuredContent["error_code"] == "tool_timeout"
    assert timed_out.structuredContent["dispatch_started"] is True
    assert timed_out.structuredContent["execution_may_continue"] is True
    assert completed.isError is False


@pytest.mark.anyio
async def test_timed_out_dispatchers_hold_slots_until_workers_finish() -> None:
    """Sequential timeouts must not turn a capacity of two into many threads.

    A purely concurrent burst did not expose the bug because requests waiting
    for a slot timed out together. Repeated calls did: each timeout released
    AnyIO's limiter token while its abandoned worker kept running.
    """

    lock = threading.Lock()
    release = threading.Event()
    active = 0
    maximum = 0
    calls = 0

    def dispatcher(_name: str, _arguments: dict | None) -> dict:
        nonlocal active, maximum, calls
        with lock:
            calls += 1
            active += 1
            maximum = max(maximum, active)
        try:
            release.wait(timeout=2)
            return {"ok": True}
        finally:
            with lock:
                active -= 1

    server = _server(
        dispatcher=dispatcher,
        max_concurrent_tool_calls=2,
        tool_timeout_seconds=0.03,
    )
    try:
        async with create_connected_server_and_client_session(
            server,
            raise_exceptions=True,
        ) as session:
            await session.initialize()
            timed_out = [
                await session.call_tool("research_agent.list_skills", {})
                for _ in range(8)
            ]

            assert all(result.isError for result in timed_out)
            assert maximum == 2
            assert calls == 2
            assert sum(
                bool(result.structuredContent["dispatch_started"])
                for result in timed_out
            ) == 2

            release.set()
            await asyncio.sleep(0.1)
            completed = await session.call_tool(
                "research_agent.list_skills",
                {},
            )
    finally:
        release.set()

    assert completed.isError is False
    assert calls == 3
    assert maximum == 2


@pytest.mark.anyio
async def test_tool_limiter_bounds_concurrent_dispatch() -> None:
    lock = threading.Lock()
    active = 0
    maximum = 0

    def dispatcher(_name: str, _arguments: dict | None) -> dict:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.08)
            return {"ok": True}
        finally:
            with lock:
                active -= 1

    server = _server(
        dispatcher=dispatcher,
        max_concurrent_tool_calls=2,
    )
    async with create_connected_server_and_client_session(
        server,
        raise_exceptions=True,
    ) as session:
        await session.initialize()
        results = await asyncio.gather(
            *[
                session.call_tool("research_agent.list_skills", {})
                for _ in range(6)
            ]
        )

    assert all(result.isError is False for result in results)
    assert maximum == 2


@pytest.mark.anyio
async def test_client_cancellation_retains_slot_until_worker_finishes() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def dispatcher(_name: str, _arguments: dict | None) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            release.wait(timeout=2)
        return {"call": calls}

    server = _server(dispatcher=dispatcher, max_concurrent_tool_calls=1)
    try:
        async with create_connected_server_and_client_session(
            server,
            raise_exceptions=True,
        ) as session:
            await session.initialize()
            pending = asyncio.create_task(
                session.call_tool("research_agent.list_skills", {})
            )
            assert await asyncio.to_thread(started.wait, 1)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

            waiting = asyncio.create_task(
                session.call_tool("research_agent.list_skills", {})
            )
            await asyncio.sleep(0.05)
            assert calls == 1
            assert waiting.done() is False

            release.set()
            completed = await waiting
    finally:
        release.set()

    assert completed.isError is False
    assert calls == 2


@pytest.mark.anyio
async def test_real_stdio_transport_uses_official_client(tmp_path: Path) -> None:
    env = os.environ.copy()
    env[MCP_ALLOWED_ROOTS_ENV] = str(tmp_path)
    env[MCP_SCOPES_ENV] = "metadata"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "easyicu.research_agent.mcp_server",
            "--transport",
            "stdio",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
    )

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            initialized = await session.initialize()
            listed = await session.list_tools()

    assert initialized.serverInfo.name == "easyicu-research-agent"
    assert "research_agent.list_skills" in {tool.name for tool in listed.tools}


def _initialize_payload() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": mcp_types.LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "easyicu-test", "version": "1"},
        },
    }


def _http_client(app: Starlette) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )


def _http_app(
    *,
    server=None,
    bearer_token: str | None = None,
    max_body_bytes: int = 1024 * 1024,
):
    server = server or _server()
    return create_streamable_http_app(
        server=server,
        host="127.0.0.1",
        port=80,
        bearer_token=bearer_token,
        allowed_hosts=["testserver"],
        allowed_origins=["http://trusted.example"],
        max_body_bytes=max_body_bytes,
    )


@pytest.mark.anyio
async def test_streamable_http_initializes_without_custom_jsonrpc_parser() -> None:
    app = _http_app()
    async with app.router.lifespan_context(app):
        async with _http_client(app) as client:
            response = await client.post(
                "/mcp",
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                json=_initialize_payload(),
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["serverInfo"]["name"] == "easyicu-research-agent"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ({"Origin": "https://evil.example"}, 403),
        ({"Origin": "null"}, 403),
        ({"Host": "evil.example"}, 421),
        ({"Content-Type": "text/plain"}, 400),
    ],
)
async def test_streamable_http_sdk_rejects_unsafe_headers(
    headers: dict[str, str],
    expected_status: int,
) -> None:
    app = _http_app()
    request_headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        **headers,
    }
    async with app.router.lifespan_context(app):
        async with _http_client(app) as client:
            response = await client.post(
                "/mcp",
                headers=request_headers,
                content=json.dumps(_initialize_payload()),
            )

    assert response.status_code == expected_status


@pytest.mark.anyio
async def test_streamable_http_rejects_malformed_and_oversized_messages() -> None:
    app = _http_app(max_body_bytes=8)
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    async with app.router.lifespan_context(app):
        async with _http_client(app) as client:
            oversized = await client.post("/mcp", headers=headers, content=b"x" * 9)

    assert oversized.status_code == 413

    app = _http_app()
    async with app.router.lifespan_context(app):
        async with _http_client(app) as client:
            malformed = await client.post("/mcp", headers=headers, content=b"{")

    assert malformed.status_code == 400


@pytest.mark.anyio
async def test_streamable_http_enforces_bearer_and_patient_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MCP_SCOPES_ENV, "metadata,read_patient_data")
    monkeypatch.setenv(MCP_PATIENT_DATA_TOKEN_ENV, "patient-token")

    def projected_scopes(_name: str, _arguments: dict | None) -> dict:
        return {"scopes": sorted(granted_scopes())}

    server = _server(dispatcher=projected_scopes)
    app = _http_app(server=server, bearer_token="mcp-token")
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    call = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "research_agent.list_skills",
            "arguments": {},
        },
    }
    async with app.router.lifespan_context(app):
        async with _http_client(app) as client:
            missing = await client.post("/mcp", headers=headers, json=call)
            authorized = await client.post(
                "/mcp",
                headers={**headers, "Authorization": "Bearer mcp-token"},
                json=call,
            )
            patient_authorized = await client.post(
                "/mcp",
                headers={
                    **headers,
                    "Authorization": "Bearer mcp-token",
                    "X-EasyICU-Patient-Data": "patient-token",
                },
                json=call,
            )

    assert missing.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["result"]["structuredContent"]["scopes"] == ["metadata"]
    assert patient_authorized.json()["result"]["structuredContent"]["scopes"] == [
        "metadata",
        "read_patient_data",
    ]


@pytest.mark.anyio
async def test_anonymous_loopback_http_cannot_start_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.research_agent.mcp_server as tool_server

    started: list[bool] = []

    class _NeverRuns:
        def __init__(self, **_kwargs):
            started.append(True)

    monkeypatch.setattr(tool_server, "ResearchAgentPipeline", _NeverRuns)
    app = _http_app()
    call = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "research_agent.run",
            "arguments": {
                "question": "q",
                "cohort_path": "cohort.parquet",
                "model": "local-model",
            },
        },
    }
    async with app.router.lifespan_context(app):
        async with _http_client(app) as client:
            response = await client.post(
                "/mcp",
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                json=call,
            )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is True
    assert result["structuredContent"]["error_code"] == "scope_not_granted"
    assert started == []


def test_remote_http_bind_requires_an_independent_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")

    with pytest.raises(ValueError, match="non-loopback"):
        validate_http_server_config("0.0.0.0", None)
    with pytest.raises(ValueError, match="must not reuse OPENAI_API_KEY"):
        validate_http_server_config("0.0.0.0", "provider-secret")
    assert (
        validate_http_server_config("0.0.0.0", "independent-mcp-token")
        == "independent-mcp-token"
    )
