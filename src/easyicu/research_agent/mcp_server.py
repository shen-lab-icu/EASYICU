"""MCP-compatible tool server.

Model Context Protocol (MCP) lets Claude Desktop, Continue and other
clients invoke server-side tools as if they were local functions.

This module exposes three EasyICU research-agent tools:

* ``research_agent.run`` — start a pipeline run from a JSON request;
* ``research_agent.list_skills`` — enumerate the registered
  :class:`ClinicalSkill` objects;
* ``research_agent.read_manifest`` — return a past run's manifest by
  ``run_id``.

The dispatch table remains a plain dict for tests and Python callers,
but ``main()`` now speaks the MCP JSON-RPC methods used by desktop
clients: ``initialize``, ``tools/list`` and ``tools/call`` over stdio.
An optional no-dependency SSE transport is provided for clients that
still expect the older MCP SSE shape.
"""

from __future__ import annotations

import argparse
import http.server
import json
import queue
import sys
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .pipeline import ResearchAgentPipeline
from .skills import list_skills


PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "easyicu-research-agent", "version": "0.1.0"}


def _tool_run(args: Dict[str, Any]) -> Dict[str, Any]:
    workdir = args.pop("workdir", "./research_output")
    pipeline = ResearchAgentPipeline(workdir=workdir)
    cohort = args.pop("cohort_path", None)
    if cohort is None:
        return {"error": "cohort_path is required"}
    result = pipeline.run(cohort=cohort, **args)
    return result.model_dump()


def _tool_list_skills(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "skills": [
            {
                "key": s.key,
                "name": s.name,
                "description": s.description,
                "target_outcome": s.target_outcome,
                "primary_predictor": s.primary_predictor,
                "expected_variables": s.expected_variables,
            }
            for s in list_skills()
        ]
    }


def _tool_read_manifest(args: Dict[str, Any]) -> Dict[str, Any]:
    workdir = Path(args.get("workdir", "./research_output"))
    run_id = args.get("run_id")
    if not run_id:
        return {"error": "run_id is required"}
    path = workdir / run_id / "manifest.json"
    if not path.exists():
        return {"error": f"manifest not found at {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


# Public dispatch table — map tool names to handlers.
TOOLS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "research_agent.run": _tool_run,
    "research_agent.list_skills": _tool_list_skills,
    "research_agent.read_manifest": _tool_read_manifest,
}


# Tool schemas — mirrored after MCP's tool descriptor format.
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "research_agent.run",
        "description": "Run an ICU-aware research-agent pipeline against a cohort parquet.",
        "inputSchema": {
            "type": "object",
            "required": ["question", "cohort_path"],
            "properties": {
                "question": {"type": "string"},
                "cohort_path": {"type": "string"},
                "workdir": {"type": "string"},
                "cohort_name": {"type": "string"},
                "database": {"type": "string"},
                "target_outcome": {"type": "string"},
                "cross_database_validation": {"type": "array", "items": {"type": "string"}},
                "inclusion_criteria": {"type": "array", "items": {"type": "string"}},
                "exclusion_criteria": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "research_agent.list_skills",
        "description": "Enumerate the registered ClinicalSkill recipes.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "research_agent.read_manifest",
        "description": "Read a past run's manifest by run_id.",
        "inputSchema": {
            "type": "object",
            "required": ["run_id"],
            "properties": {
                "run_id": {"type": "string"},
                "workdir": {"type": "string"},
            },
        },
    },
]


def dispatch(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a tool by name. Returns either the tool's result dict or an error."""
    arguments = dict(arguments or {})
    fn = TOOLS.get(tool_name)
    if fn is None:
        return {"error": f"unknown tool '{tool_name}'", "known": list(TOOLS)}
    try:
        return fn(arguments)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# MCP JSON-RPC
# ---------------------------------------------------------------------------


def _response(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }
    if data is not None:
        payload["error"]["data"] = data
    return payload


def _tool_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    is_error = "error" in result
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(result, indent=2, ensure_ascii=False, default=str),
            }
        ],
        "isError": is_error,
    }


def _handle_single_jsonrpc(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle one MCP JSON-RPC request.

    Notifications have no ``id`` and therefore intentionally return
    ``None``. This matters for MCP clients: replying to
    ``notifications/initialized`` is a protocol violation.
    """
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    is_notification = "id" not in request

    # Backwards-compatible ad-hoc shape used by the original stub:
    # {"tool": "research_agent.list_skills", "arguments": {...}}
    if method is None and request.get("tool"):
        result = dispatch(str(request.get("tool")), request.get("arguments"))
        if is_notification:
            return result
        return _response(req_id, result)

    if method == "initialize":
        return _response(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })

    if method in {"notifications/initialized", "initialized"}:
        return None if is_notification else _response(req_id, {})

    if method == "ping":
        return None if is_notification else _response(req_id, {})

    if method == "tools/list":
        return _response(req_id, {"tools": TOOL_SCHEMAS})

    if method == "tools/call":
        if not isinstance(params, dict):
            return _error(req_id, -32602, "Invalid params: expected object")
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            return _error(req_id, -32602, "Invalid params: tools/call requires string 'name'")
        if not isinstance(arguments, dict):
            return _error(req_id, -32602, "Invalid params: 'arguments' must be an object")
        return _response(req_id, _tool_result_payload(dispatch(name, arguments)))

    # These are optional MCP surfaces. Returning empty lists lets clients
    # complete capability discovery without assuming resources/prompts exist.
    if method == "resources/list":
        return _response(req_id, {"resources": []})
    if method == "prompts/list":
        return _response(req_id, {"prompts": []})

    if is_notification:
        return None
    return _error(req_id, -32601, f"Method not found: {method!r}")


def handle_jsonrpc(payload: Any) -> Optional[Any]:
    """Handle a JSON-RPC request or batch payload.

    Returns a JSON-serialisable response, a list of responses for a
    batch, or ``None`` for notifications.
    """
    if isinstance(payload, list):
        responses = [r for item in payload if (r := handle_jsonrpc(item)) is not None]
        return responses or None
    if not isinstance(payload, dict):
        return _error(None, -32600, "Invalid Request: expected JSON object")
    return _handle_single_jsonrpc(payload)


def _run_stdio() -> int:  # pragma: no cover - covered through handle_jsonrpc
    """Run MCP JSON-RPC over stdin/stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps(_error(None, -32700, "Parse error")), flush=True)
            continue
        response = handle_jsonrpc(req)
        if response is not None:
            print(json.dumps(response, ensure_ascii=False, default=str), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Minimal SSE transport
# ---------------------------------------------------------------------------


class _SSESession:
    def __init__(self) -> None:
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()


_SSE_SESSIONS: Dict[str, _SSESession] = {}


def _make_sse_handler() -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "EasyICUMCP/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Keep stdout clean for JSON-RPC clients; diagnostics go to stderr.
            print(format % args, file=sys.stderr)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path not in {"/sse", "/events"}:
                self.send_error(404, "not found")
                return
            session_id = uuid.uuid4().hex
            session = _SSESession()
            _SSE_SESSIONS[session_id] = session

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            endpoint = f"/messages?session_id={session_id}"
            self._write_sse("endpoint", endpoint)
            try:
                while True:
                    item = session.queue.get()
                    self._write_sse("message", json.dumps(item, ensure_ascii=False, default=str))
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                _SSE_SESSIONS.pop(session_id, None)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            try:
                req = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._write_json(_error(None, -32700, "Parse error"), status=400)
                return

            response = handle_jsonrpc(req)
            if parsed.path in {"/messages", "/message"}:
                session_id = (qs.get("session_id") or qs.get("sessionId") or [""])[0]
                session = _SSE_SESSIONS.get(session_id)
                if session is None:
                    self._write_json({"error": "unknown SSE session"}, status=404)
                    return
                if response is not None:
                    session.queue.put(response)
                self._write_json({"accepted": True})
                return

            if parsed.path in {"/jsonrpc", "/rpc"}:
                self._write_json(response or {})
                return

            self.send_error(404, "not found")

        def _write_sse(self, event: str, data: str) -> None:
            payload = f"event: {event}\ndata: {data}\n\n".encode("utf-8")
            self.wfile.write(payload)
            self.wfile.flush()

        def _write_json(self, payload: Any, *, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _run_sse(host: str, port: int) -> int:  # pragma: no cover
    server = http.server.ThreadingHTTPServer((host, port), _make_sse_handler())
    print(f"easyicu research-agent MCP SSE listening on http://{host}:{port}/sse", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    """Run the MCP server.

    Default transport is stdio, which is what Claude Desktop, Continue
    and Cursor typically mount for local MCP servers. ``--transport
    sse`` starts a tiny stdlib HTTP/SSE bridge for clients that still
    use the legacy SSE transport.
    """
    parser = argparse.ArgumentParser(description="EasyICU research-agent MCP server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.transport == "sse":
        return _run_sse(args.host, args.port)
    return _run_stdio()


__all__ = [
    "TOOLS",
    "TOOL_SCHEMAS",
    "dispatch",
    "handle_jsonrpc",
    "main",
]
