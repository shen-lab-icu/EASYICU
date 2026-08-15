"""Official MCP SDK transport adapter for EasyICU research-agent tools.

The application-owned contracts remain in :mod:`easyicu.research_agent.mcp_server`:
tool schemas, disclosure policy, authorization scopes, audit records and the
plain-Python ``dispatch`` seam. This module delegates protocol negotiation,
JSON-RPC validation, stdio framing and Streamable HTTP to the official
``modelcontextprotocol/python-sdk`` package.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import os
import secrets
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, Optional

import mcp.server.stdio
import mcp.types as mcp_types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .mcp_policy import (
    MCP_PATIENT_DATA_TOKEN_ENV,
    SCOPE_METADATA,
    SCOPE_READ_PATIENT_DATA,
    process_scopes,
    scope_override,
)

MCP_BEARER_TOKEN_ENV = "EASYICU_MCP_BEARER_TOKEN"
MCP_TRUSTED_REVERSE_PROXY_ENV = "EASYICU_MCP_TRUSTED_REVERSE_PROXY"
DEFAULT_HTTP_MAX_BODY_BYTES = 1024 * 1024
DEFAULT_MAX_CONCURRENT_TOOL_CALLS = 4
LOOPBACK_ANONYMOUS_SCOPES = frozenset({SCOPE_METADATA})

Dispatcher = Callable[[str, Optional[dict[str, Any]]], dict[str, Any]]
ServerFactory = Callable[..., Server]


class _BoundedDispatcher:
    """Keep a concurrency slot until the real synchronous call returns.

    AnyIO's ``abandon_on_cancel=True`` releases its capacity-limiter token when
    the *waiting request* is cancelled, even though the worker thread continues.
    Repeated request timeouts could therefore accumulate unbounded live
    dispatchers. This supervisor separates request lifetime from worker
    lifetime: cancellation abandons the result, not the occupied slot.
    """

    def __init__(self, capacity: int) -> None:
        self._slots = asyncio.Semaphore(capacity)

    async def run(
        self,
        invoke: Callable[[], dict[str, Any]],
        *,
        state: dict[str, bool],
        queue_timeout_seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        if queue_timeout_seconds is None:
            await self._slots.acquire()
        else:
            await asyncio.wait_for(
                self._slots.acquire(),
                timeout=queue_timeout_seconds,
            )
        state["started"] = True
        try:
            # asyncio.to_thread preserves the request ContextVars carrying MCP
            # scopes and patient-data authority.
            task = asyncio.create_task(asyncio.to_thread(invoke))
        except BaseException:
            self._slots.release()
            raise
        task.add_done_callback(self._worker_finished)
        # A running synchronous dispatcher cannot be killed safely. Do not let
        # protocol cancellation detach it from request lifecycle: retain the
        # slot and wait for the actual operation to converge before propagating
        # cancellation.
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancelled:
            while True:
                if task.done():
                    break
                try:
                    await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    # Repeated protocol cancellation still cannot detach the
                    # synchronous worker from this request lifecycle.
                    continue
                except BaseException:
                    # The caller cancelled first. Converge the worker and keep
                    # that cancellation authoritative over a later worker error.
                    break
            raise cancelled

    def _worker_finished(self, task: asyncio.Task[dict[str, Any]]) -> None:
        self._slots.release()
        if not task.cancelled():
            # Retrieve an abandoned worker exception so asyncio does not emit
            # "Task exception was never retrieved". An active waiter still
            # receives the same exception when awaiting the task.
            task.exception()


def _json_safe(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible copy for both MCP result surfaces."""

    return json.loads(json.dumps(dict(payload), ensure_ascii=False, default=str))


def create_mcp_server(
    *,
    dispatcher: Dispatcher,
    server_info: Mapping[str, str],
    tool_schemas: Sequence[Mapping[str, Any]],
    max_concurrent_tool_calls: int = DEFAULT_MAX_CONCURRENT_TOOL_CALLS,
    tool_timeout_seconds: Optional[float] = None,
) -> Server:
    """Build the official low-level SDK server around EasyICU contracts.

    ``tool_timeout_seconds`` bounds queue wait before dispatch starts. Once a
    synchronous dispatcher starts, the request waits for its real completion;
    returning a timeout while paid or mutating work continues would create an
    unsafe detached operation.
    """

    if max_concurrent_tool_calls <= 0:
        raise ValueError("max_concurrent_tool_calls must be positive")
    if tool_timeout_seconds is not None and tool_timeout_seconds <= 0:
        raise ValueError("tool_timeout_seconds must be positive when configured")

    server = Server(
        server_info["name"],
        version=server_info["version"],
        instructions=(
            "EasyICU research tools. Tool results remain governed by EasyICU "
            "scope, disclosure, evidence and patient-data audit policies."
        ),
    )
    dispatcher_supervisor = _BoundedDispatcher(max_concurrent_tool_calls)

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return [mcp_types.Tool.model_validate(schema) for schema in tool_schemas]

    @server.call_tool(validate_input=True)
    async def _call_tool(
        name: str,
        arguments: dict[str, Any],
    ) -> mcp_types.CallToolResult:
        invoke = partial(dispatcher, name, dict(arguments))
        dispatch_state = {"started": False}
        try:
            result = await dispatcher_supervisor.run(
                invoke,
                state=dispatch_state,
                queue_timeout_seconds=tool_timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = {
                "error": (
                    f"tool {name!r} exceeded the configured MCP queue timeout"
                ),
                "error_code": "tool_timeout",
                "dispatch_started": dispatch_state["started"],
                "execution_may_continue": False,
            }

        safe_result = _json_safe(result)
        return mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps(
                        safe_result,
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
            ],
            structuredContent=safe_result,
            isError="error" in safe_result,
        )

    return server


def _is_loopback_host(host: str) -> bool:
    value = str(host or "").strip().rstrip(".").lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def validate_http_server_config(
    host: str,
    bearer_token: Optional[str],
    *,
    trusted_reverse_proxy: bool = False,
    tls_certfile: Optional[str] = None,
    tls_keyfile: Optional[str] = None,
) -> Optional[str]:
    """Fail closed before binding an externally reachable MCP socket."""

    token = str(bearer_token or "").strip() or None
    cert = str(tls_certfile or "").strip() or None
    key = str(tls_keyfile or "").strip() or None
    if bool(cert) != bool(key):
        raise ValueError("direct MCP TLS requires both a certificate and private key")
    if cert is not None and key is not None:
        for label, value in (("TLS certificate", cert), ("TLS private key", key)):
            path = os.path.expanduser(value)
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"{label} must be an existing regular file")
    if not _is_loopback_host(host):
        if token is None:
            raise ValueError(
                "non-loopback MCP HTTP binding requires an independent bearer token "
                f"in {MCP_BEARER_TOKEN_ENV}"
            )
        if not trusted_reverse_proxy and (cert is None or key is None):
            raise ValueError(
                "direct non-loopback MCP Streamable HTTP requires --tls-certfile "
                "and --tls-keyfile, or explicit --trusted-reverse-proxy mode"
            )
    if token is not None:
        for secret_name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            provider_secret = os.environ.get(secret_name)
            if provider_secret and secrets.compare_digest(token, provider_secret):
                raise ValueError(f"{MCP_BEARER_TOKEN_ENV} must not reuse {secret_name}")
    return token


def _default_allowed_hosts(host: str, port: int) -> list[str]:
    if _is_loopback_host(host):
        return [
            f"127.0.0.1:{port}",
            f"localhost:{port}",
            f"[::1]:{port}",
        ]
    rendered = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return [f"{rendered}:{port}"]


def _default_allowed_origins(host: str, port: int) -> list[str]:
    if _is_loopback_host(host):
        hosts = ("127.0.0.1", "localhost", "[::1]")
    else:
        hosts = (f"[{host}]" if ":" in host and not host.startswith("[") else host,)
    return [f"{scheme}://{item}:{port}" for item in hosts for scheme in ("http", "https")]


def _headers(scope: Scope) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for raw_name, raw_value in scope.get("headers", []):
        name = raw_name.decode("latin-1").lower()
        values.setdefault(name, []).append(raw_value.decode("latin-1"))
    return values


class _RequestAuthorityMiddleware:
    """Apply EasyICU credentials/scopes around the official transport."""

    def __init__(self, app: ASGIApp, *, bearer_token: Optional[str]) -> None:
        self.app = app
        self.bearer_token = bearer_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = _headers(scope)
        authorization = headers.get("authorization", [])
        patient_tokens = headers.get("x-easyicu-patient-data", [])
        if len(authorization) > 1 or len(patient_tokens) > 1:
            await PlainTextResponse("duplicate credential header", status_code=400)(
                scope, receive, send
            )
            return

        if self.bearer_token is not None:
            supplied = authorization[0] if authorization else ""
            prefix = "Bearer "
            candidate = (
                supplied[len(prefix) :] if supplied.startswith(prefix) else ""
            )
            if not candidate or not secrets.compare_digest(
                candidate, self.bearer_token
            ):
                await PlainTextResponse(
                    "missing or invalid bearer token",
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )(scope, receive, send)
                return

        request_scopes = process_scopes()
        expected_patient_token = str(
            os.environ.get(MCP_PATIENT_DATA_TOKEN_ENV, "") or ""
        ).strip()
        supplied_patient_token = patient_tokens[0].strip() if patient_tokens else ""
        patient_authorized = bool(
            expected_patient_token
            and supplied_patient_token
            and secrets.compare_digest(
                supplied_patient_token,
                expected_patient_token,
            )
        )
        if not patient_authorized:
            request_scopes = request_scopes - {SCOPE_READ_PATIENT_DATA}
        if self.bearer_token is None:
            request_scopes = request_scopes & LOOPBACK_ANONYMOUS_SCOPES

        with scope_override(request_scopes):
            await self.app(scope, receive, send)


class _RequestBodyLimitMiddleware:
    """Bound POST bodies before the SDK parses or creates request state."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        messages: list[Message] = []
        size = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                continue
            size += len(message.get("body", b""))
            if size > self.max_body_bytes:
                await PlainTextResponse("request body too large", status_code=413)(
                    scope, receive, send
                )
                return
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)


def create_streamable_http_app(
    *,
    server: Server,
    host: str = "127.0.0.1",
    port: int = 8765,
    bearer_token: Optional[str] = None,
    trusted_reverse_proxy: bool = False,
    tls_certfile: Optional[str] = None,
    tls_keyfile: Optional[str] = None,
    allowed_hosts: Optional[Sequence[str]] = None,
    allowed_origins: Optional[Sequence[str]] = None,
    max_body_bytes: int = DEFAULT_HTTP_MAX_BODY_BYTES,
) -> Starlette:
    """Create a stateless JSON Streamable HTTP app using the official SDK."""

    token = validate_http_server_config(
        host,
        bearer_token,
        trusted_reverse_proxy=trusted_reverse_proxy,
        tls_certfile=tls_certfile,
        tls_keyfile=tls_keyfile,
    )
    exact_hosts = list(dict.fromkeys([*_default_allowed_hosts(host, port), *(allowed_hosts or ())]))
    exact_origins = list(
        dict.fromkeys(
            [*_default_allowed_origins(host, port), *(allowed_origins or ())]
        )
    )
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=exact_hosts,
        allowed_origins=exact_origins,
    )
    manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=True,
        stateless=True,
        security_settings=security,
    )

    async def handle_streamable_http(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        await manager.handle_request(scope, receive, send)

    endpoint: ASGIApp = _RequestBodyLimitMiddleware(
        handle_streamable_http,
        max_body_bytes=max_body_bytes,
    )
    endpoint = _RequestAuthorityMiddleware(endpoint, bearer_token=token)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette):
        async with manager.run():
            yield

    app = Starlette(
        routes=[Route("/mcp", endpoint=endpoint)],
        lifespan=lifespan,
    )
    app.state.easyicu_mcp_server = server
    app.state.easyicu_mcp_session_manager = manager
    return app


async def _serve_stdio(server: Server) -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def _run_streamable_http(
    *,
    server: Server,
    host: str,
    port: int,
    bearer_token: Optional[str],
    trusted_reverse_proxy: bool,
    tls_certfile: Optional[str],
    tls_keyfile: Optional[str],
    allowed_hosts: Sequence[str],
    allowed_origins: Sequence[str],
    max_body_bytes: int,
) -> int:
    import uvicorn

    app = create_streamable_http_app(
        server=server,
        host=host,
        port=port,
        bearer_token=bearer_token,
        trusted_reverse_proxy=trusted_reverse_proxy,
        tls_certfile=tls_certfile,
        tls_keyfile=tls_keyfile,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        max_body_bytes=max_body_bytes,
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_certfile=tls_certfile,
        ssl_keyfile=tls_keyfile,
    )
    return 0


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    server_factory: ServerFactory,
) -> int:
    """Run an injected application server over official MCP transports."""

    parser = argparse.ArgumentParser(description="EasyICU research-agent MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allowed-host", action="append", default=[])
    parser.add_argument("--allowed-origin", action="append", default=[])
    parser.add_argument(
        "--trusted-reverse-proxy",
        action="store_true",
        help=(
            "assert that a non-loopback bind is behind a configured trusted "
            "TLS-terminating reverse proxy"
        ),
    )
    parser.add_argument("--tls-certfile")
    parser.add_argument("--tls-keyfile")
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=DEFAULT_HTTP_MAX_BODY_BYTES,
    )
    parser.add_argument(
        "--tool-timeout-seconds",
        type=float,
        help=(
            "maximum seconds an MCP request waits for a dispatcher slot; once "
            "a synchronous worker starts, the request waits for its real "
            "in-process completion"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    server = server_factory(tool_timeout_seconds=args.tool_timeout_seconds)

    if args.transport == "streamable-http":
        try:
            return _run_streamable_http(
                server=server,
                host=args.host,
                port=args.port,
                bearer_token=os.environ.get(MCP_BEARER_TOKEN_ENV),
                trusted_reverse_proxy=(
                    args.trusted_reverse_proxy
                    or str(os.environ.get(MCP_TRUSTED_REVERSE_PROXY_ENV, ""))
                    .strip()
                    .lower()
                    in {"1", "true", "yes", "on"}
                ),
                tls_certfile=args.tls_certfile,
                tls_keyfile=args.tls_keyfile,
                allowed_hosts=args.allowed_host,
                allowed_origins=args.allowed_origin,
                max_body_bytes=args.max_request_bytes,
            )
        except ValueError as exc:
            parser.error(str(exc))

    asyncio.run(_serve_stdio(server))
    return 0


__all__ = [
    "DEFAULT_HTTP_MAX_BODY_BYTES",
    "DEFAULT_MAX_CONCURRENT_TOOL_CALLS",
    "LOOPBACK_ANONYMOUS_SCOPES",
    "MCP_BEARER_TOKEN_ENV",
    "MCP_TRUSTED_REVERSE_PROXY_ENV",
    "create_mcp_server",
    "create_streamable_http_app",
    "main",
    "validate_http_server_config",
]
