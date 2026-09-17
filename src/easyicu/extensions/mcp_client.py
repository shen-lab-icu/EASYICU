"""Bounded read-only client for user-configured Streamable HTTP MCP servers."""

from __future__ import annotations

import json
import re
from functools import partial
from typing import Any, Dict, Mapping, Sequence

import anyio

from easyicu.outbound_url_security import (
    OutboundUrlSecurityError,
    validate_outbound_http_endpoint,
    validated_http_endpoint_with_pin,
)

from .contracts import ExtensionRegistryError, McpServerActivation

MAX_MCP_ARGUMENT_BYTES = 16_000
MAX_MCP_RESULT_BYTES = 30_000
MAX_MCP_TOOLS = 100
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "cookie",
        "set_cookie",
        "path",
        "file_path",
        "patient_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "mrn",
    }
)
_HOST_PATH = re.compile(
    r"(?:file://|/(?:Users|home|private|tmp|var|etc|opt|Volumes)/|[A-Za-z]:\\)",
    flags=re.IGNORECASE,
)
_SENSITIVE_TEXT = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-[A-Za-z0-9_-]{8,}|"
    r"\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+|"
    r"[\"']?(?:subject_id|stay_id|hadm_id|patient_id|mrn)[\"']?\s*[:,=]\s*[\"']?[A-Za-z0-9-]+)",
    flags=re.IGNORECASE,
)


class McpClientError(ExtensionRegistryError):
    """MCP transport or projection failure owned by the extension boundary."""


def _validate_arguments(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise McpClientError(
            "extension_mcp_arguments_too_deep",
            "MCP tool arguments exceed the bounded nesting depth.",
        )
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _HOST_PATH.search(value):
            raise McpClientError(
                "extension_mcp_host_path_rejected",
                "MCP tool arguments must not contain a host filesystem path.",
            )
        if _SENSITIVE_TEXT.search(value):
            raise McpClientError(
                "extension_mcp_sensitive_argument_rejected",
                "MCP tool arguments must not contain credentials or patient identifiers.",
            )
        return value[:8_000]
    if isinstance(value, Mapping):
        clean: Dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:100]:
            key = str(raw_key)[:160]
            if key.casefold().replace("-", "_") in _SENSITIVE_KEYS:
                raise McpClientError(
                    "extension_mcp_sensitive_argument_rejected",
                    "MCP tool arguments must not contain credentials, host paths, or patient identifiers.",
                    details={"field": key},
                )
            clean[key] = _validate_arguments(raw_value, depth=depth + 1)
        return clean
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_validate_arguments(item, depth=depth + 1) for item in list(value)[:100]]
    raise McpClientError(
        "extension_mcp_arguments_invalid",
        "MCP tool arguments must be bounded JSON data.",
    )


def _project_result(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _HOST_PATH.search(value):
            return "[host path redacted]"
        if _SENSITIVE_TEXT.search(value):
            return "[sensitive text redacted]"
        return value[:8_000]
    if isinstance(value, Mapping):
        projected: Dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:100]:
            key = str(raw_key)[:160]
            normalized = key.casefold().replace("-", "_")
            if normalized in _SENSITIVE_KEYS:
                projected[f"redacted_field_{len(projected) + 1}"] = "[redacted]"
            else:
                projected[key] = _project_result(raw_value, depth=depth + 1)
        return projected
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_project_result(item, depth=depth + 1) for item in list(value)[:100]]
    if hasattr(value, "model_dump"):
        return _project_result(value.model_dump(mode="json"), depth=depth + 1)
    return str(value)[:2_000]


def _bounded_payload(value: Any) -> Any:
    projected = _project_result(value)
    encoded = json.dumps(projected, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) > MAX_MCP_RESULT_BYTES:
        raise McpClientError(
            "extension_mcp_result_too_large",
            "The MCP tool result exceeds the bounded projection limit.",
            details={"bytes": len(encoded), "max_bytes": MAX_MCP_RESULT_BYTES},
        )
    return projected


def _validated_url(url: str, *, pinned_ip: str | None = None) -> str:
    """Validate an MCP endpoint, optionally pinning the validated IP.

    ``pinned_ip`` is per-call only and safe to pass through threads; each
    thread validates its own connection (see
    :mod:`easyicu.outbound_url_security` for the TOCTOU limitation).
    """
    try:
        return validate_outbound_http_endpoint(url, pinned_ip=pinned_ip)
    except OutboundUrlSecurityError as exc:
        raise McpClientError(
            "extension_mcp_url_rejected",
            "The MCP endpoint violates the outbound network policy.",
            details={"reason": exc.reason},
        ) from exc


class _PinnedHostTransport:
    """httpx transport that connects to a validated IP with the real Host.

    Closes the DNS-rebinding window for loopback-HTTP endpoints: the request
    URL already carries the validated literal IP (no re-resolution happens),
    and this transport forces the original ``Host`` header so virtual-hosted
    servers still route correctly. No shared state; one instance per call.
    """

    def __init__(self, host_header: str) -> None:
        import httpx

        self._transport = httpx.AsyncHTTPTransport()
        self._host_header = host_header

    async def handle_async_request(self, request):  # type: ignore[no-untyped-def]
        import httpx

        headers = httpx.Headers(request.headers)
        headers["host"] = self._host_header
        pinned = httpx.Request(
            request.method,
            request.url,
            headers=headers,
            stream=request.stream,
            extensions=request.extensions,
        )
        return await self._transport.handle_async_request(pinned)

    async def __aenter__(self):  # type: ignore[no-untyped-def]
        await self._transport.__aenter__()
        return self

    async def __aexit__(self, *args):  # type: ignore[no-untyped-def]
        await self._transport.__aexit__(*args)

    async def aclose(self) -> None:
        await self._transport.aclose()


def _validated_endpoint(url: str, *, pinned_ip: str | None = None) -> tuple[str, Any | None]:
    """Validate an MCP endpoint and return ``(connect_url, transport)``.

    For loopback-HTTP DNS names the connect URL carries the validated literal
    IP and ``transport`` forces the original Host header, so no DNS
    re-resolution happens between validation and connect. All other URLs
    return ``(url, None)`` (literal IPs need no pinning; https keeps its
    hostname for TLS SNI/certificate verification — see
    :func:`validated_http_endpoint_with_pin`).
    """

    try:
        connect_url, host_header = validated_http_endpoint_with_pin(
            url, pinned_ip=pinned_ip
        )
    except OutboundUrlSecurityError as exc:
        raise McpClientError(
            "extension_mcp_url_rejected",
            "The MCP endpoint violates the outbound network policy.",
            details={"reason": exc.reason},
        ) from exc
    if host_header is None:
        return connect_url, None
    return connect_url, _PinnedHostTransport(host_header)


async def _open_and_list(
    url: str, timeout_seconds: float, *, pinned_ip: str | None = None
) -> Dict[str, Any]:
    try:
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise McpClientError(
            "extension_mcp_runtime_unavailable",
            "Install EasyICU with the mcp extra before using MCP servers.",
        ) from exc
    endpoint, transport = _validated_endpoint(url, pinned_ip=pinned_ip)
    try:
        with anyio.fail_after(timeout_seconds):
            async with httpx.AsyncClient(
                transport=transport,
                follow_redirects=False,
                timeout=httpx.Timeout(timeout_seconds),
            ) as client:
                async with streamable_http_client(
                    endpoint,
                    http_client=client,
                    terminate_on_close=True,
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        listed = await session.list_tools()
    except McpClientError:
        raise
    except TimeoutError as exc:
        raise McpClientError(
            "extension_mcp_timeout", "The MCP server did not respond within the time limit."
        ) from exc
    except Exception as exc:
        raise McpClientError(
            "extension_mcp_connection_failed",
            "The MCP server connection or protocol handshake failed.",
            details={"failure_type": type(exc).__name__},
        ) from exc
    tools = []
    for item in list(getattr(listed, "tools", ()) or ())[:MAX_MCP_TOOLS]:
        tools.append(
            {
                "name": str(getattr(item, "name", ""))[:128],
                "description": re.sub(
                    r"\s+", " ", str(getattr(item, "description", "") or "")
                ).strip()[:1_000],
            }
        )
    return {
        "ok": True,
        "transport": "streamable-http",
        "tools": tools,
        "tool_count": len(tools),
    }


async def _open_and_call(
    server: McpServerActivation,
    tool_name: str,
    arguments: Mapping[str, Any],
    timeout_seconds: float,
) -> Dict[str, Any]:
    try:
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:  # pragma: no cover
        raise McpClientError(
            "extension_mcp_runtime_unavailable",
            "Install EasyICU with the mcp extra before using MCP servers.",
        ) from exc
    name = str(tool_name or "").strip()
    if name not in server.allowed_tools:
        raise McpClientError(
            "extension_mcp_tool_not_allowed",
            "The requested MCP tool is not in this server's frozen allowlist.",
            details={"server": server.name, "tool": name},
        )
    clean_arguments = _validate_arguments(arguments)
    argument_bytes = len(
        json.dumps(clean_arguments, ensure_ascii=False).encode("utf-8")
    )
    if argument_bytes > MAX_MCP_ARGUMENT_BYTES:
        raise McpClientError(
            "extension_mcp_arguments_too_large",
            "MCP tool arguments exceed the bounded JSON limit.",
        )
    endpoint, transport = _validated_endpoint(server.url)
    try:
        with anyio.fail_after(timeout_seconds):
            async with httpx.AsyncClient(
                transport=transport,
                follow_redirects=False,
                timeout=httpx.Timeout(timeout_seconds),
            ) as client:
                async with streamable_http_client(
                    endpoint,
                    http_client=client,
                    terminate_on_close=True,
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await session.call_tool(name, clean_arguments)
    except McpClientError:
        raise
    except TimeoutError as exc:
        raise McpClientError(
            "extension_mcp_timeout", "The MCP tool did not respond within the time limit."
        ) from exc
    except Exception as exc:
        raise McpClientError(
            "extension_mcp_call_failed",
            "The MCP tool call failed at the external server boundary.",
            details={"failure_type": type(exc).__name__},
        ) from exc
    raw = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
    return {
        "ok": not bool(getattr(result, "isError", False)),
        "server": server.name,
        "tool": name,
        "trust": "untrusted_external_metadata",
        "result": _bounded_payload(raw),
    }


def list_mcp_tools(url: str, *, timeout_seconds: float = 10.0) -> Dict[str, Any]:
    return anyio.run(partial(_open_and_list, url, float(timeout_seconds)))


def call_mcp_tool(
    server: McpServerActivation,
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    timeout_seconds: float = 15.0,
) -> Dict[str, Any]:
    return anyio.run(
        partial(
            _open_and_call,
            server,
            tool_name,
            arguments,
            float(timeout_seconds),
        )
    )


__all__ = ["McpClientError", "call_mcp_tool", "list_mcp_tools"]
