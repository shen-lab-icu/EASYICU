"""Long-lived strict JSON-lines client for the pinned Pi Node sidecar."""

from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Mapping, Optional

from .contracts import (
    PROTOCOL_VERSION,
    PiCopilotError,
    ToolExecutionContext,
)
from .install import preferred_app_dir
from .provider_config import PiProviderConfig, PiProviderConfigStore
from .tools import execute_tool

MAX_PROTOCOL_LINE_BYTES = 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15 * 60
MIN_NODE_VERSION = (22, 19, 0)
_CHILD_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "EASYICU_PI_API_KEY",
        "EASYICU_PI_PROVIDER",
        "EASYICU_PI_MODEL",
        "EASYICU_PI_BASE_URL",
        "EASYICU_PI_API",
        "EASYICU_PI_CONTEXT_WINDOW",
        "EASYICU_PI_MAX_TOKENS",
        "EASYICU_PI_SESSION_TOKEN_BUDGET",
    }
)

EventSink = Callable[[Dict[str, Any]], None]


@dataclass
class _PendingRequest:
    done: threading.Event = field(default_factory=threading.Event)
    event_sink: Optional[EventSink] = None
    tool_context: Optional[ToolExecutionContext] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[PiCopilotError] = None


class PiGatewayClient:
    """Own one Node sidecar and re-authorize every host-tool request."""

    def __init__(
        self,
        *,
        app_dir: Optional[Path] = None,
        session_dir: Optional[Path] = None,
        cwd: Optional[Path] = None,
        environ: Optional[Mapping[str, str]] = None,
        provider_store: Optional[PiProviderConfigStore] = None,
        tool_executor: Callable[
            [str, Mapping[str, Any], ToolExecutionContext], Dict[str, Any]
        ] = execute_tool,
    ) -> None:
        self.app_dir = (
            Path(app_dir)
            if app_dir is not None
            else preferred_app_dir()
        ).resolve()
        self.entrypoint = self.app_dir / "src" / "main.mjs"
        self.session_dir = (
            Path(session_dir)
            if session_dir is not None
            else Path.home() / ".easyicu" / "pi-agent" / "sessions"
        ).resolve()
        self.cwd = (
            Path(cwd)
            if cwd is not None
            else self.session_dir.parent / "workspace"
        ).resolve()
        self.provider_store = provider_store or PiProviderConfigStore()
        self._provider_file_enabled = environ is None
        source_environment = os.environ if environ is None else environ
        provider_environment = self.provider_store.environment(
            environ=source_environment,
            include_file=self._provider_file_enabled,
        )
        source_environment = {
            **source_environment,
            **provider_environment,
        }
        self.environ = {
            key: str(value)
            for key, value in source_environment.items()
            if key in _CHILD_ENV_KEYS or key.startswith("LC_")
        }
        self._tool_executor = tool_executor
        self._process: Optional[subprocess.Popen[str]] = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._pending: Dict[str, _PendingRequest] = {}
        self._stderr_tail: Deque[str] = collections.deque(maxlen=20)
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

    def _node_binary(self) -> Optional[str]:
        direct = shutil.which("node", path=self.environ.get("PATH"))
        if direct:
            return direct
        home = Path(self.environ.get("HOME") or Path.home())
        candidates = sorted(
            (home / ".nvm" / "versions" / "node").glob("*/bin/node")
        )
        return str(candidates[-1]) if candidates else None

    def _node_version(self, node: Optional[str]) -> Optional[tuple[int, int, int]]:
        if not node:
            return None
        try:
            completed = subprocess.run(
                [node, "--version"],
                env=self.environ,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        raw = completed.stdout.strip().removeprefix("v")
        parts = raw.split(".")
        if len(parts) < 3 or not all(part.isdigit() for part in parts[:3]):
            return None
        return tuple(int(part) for part in parts[:3])

    def _child_environment(self) -> Dict[str, str]:
        return {
            **self.environ,
            "EASYICU_PI_SESSION_DIR": str(self.session_dir),
            "EASYICU_PI_CWD": str(self.cwd),
        }

    def installation_status(self) -> Dict[str, Any]:
        node = self._node_binary()
        node_version = self._node_version(node)
        provider_status = self.provider_store.public_status(
            environ=self.environ,
            include_file=self._provider_file_enabled,
        )
        dependency = (
            self.app_dir
            / "node_modules"
            / "@earendil-works"
            / "pi-coding-agent"
            / "package.json"
        )
        return {
            "node_available": bool(node),
            "node_version": (
                ".".join(str(part) for part in node_version)
                if node_version is not None
                else None
            ),
            "node_version_supported": bool(
                node_version is not None and node_version >= MIN_NODE_VERSION
            ),
            "entrypoint_available": self.entrypoint.is_file(),
            "dependency_installed": dependency.is_file(),
            "lockfile_present": (self.app_dir / "package-lock.json").is_file(),
            "api_key_configured": bool(
                str(self.environ.get("EASYICU_PI_API_KEY") or "").strip()
            ),
            "provider_connection_verified": bool(
                provider_status.get("connection_verified")
            ),
            "provider": str(
                self.environ.get("EASYICU_PI_PROVIDER") or "easyicu-local"
            ),
            "model": str(self.environ.get("EASYICU_PI_MODEL") or "gpt-5.6-luna"),
            "base_url_configured": bool(
                str(
                    self.environ.get("EASYICU_PI_BASE_URL")
                    or "http://127.0.0.1:8317/v1"
                ).strip()
            ),
            "api_transport": str(
                self.environ.get("EASYICU_PI_API") or "openai-completions"
            ),
            "provider_configuration": provider_status,
        }

    def apply_provider_config(self, config: PiProviderConfig) -> None:
        """Restart the sidecar boundary with a newly verified configuration."""

        self.close()
        with self._state_lock:
            provider_environment = config.as_environment()
            for key in provider_environment:
                self.environ.pop(key, None)
            self.environ.update(provider_environment)
            self._provider_file_enabled = True

    def _start(self) -> None:
        with self._state_lock:
            if self._process and self._process.poll() is None:
                return
            status = self.installation_status()
            missing = [
                key
                for key in (
                    "node_available",
                    "node_version_supported",
                    "entrypoint_available",
                    "dependency_installed",
                    "lockfile_present",
                )
                if not status[key]
            ]
            if missing:
                raise PiCopilotError(
                    "pi_gateway_not_installed",
                    "The pinned Pi Copilot sidecar is not installed.",
                    status_code=503,
                    details={"missing": missing},
                )
            node = self._node_binary()
            assert node is not None
            self.session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.cwd.mkdir(parents=True, exist_ok=True, mode=0o700)
            env = self._child_environment()
            self._process = subprocess.Popen(
                [node, str(self.entrypoint)],
                cwd=str(self.app_dir),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self._reader_thread = threading.Thread(
                target=self._read_stdout,
                name="easyicu-pi-gateway-reader",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                name="easyicu-pi-gateway-stderr",
                daemon=True,
            )
            self._reader_thread.start()
            self._stderr_thread.start()

    def _write(self, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_PROTOCOL_LINE_BYTES:
            raise PiCopilotError(
                "pi_protocol_line_too_large",
                "The Pi gateway request exceeds the protocol size limit.",
                status_code=500,
            )
        with self._write_lock:
            process = self._process
            if (
                process is None
                or process.poll() is not None
                or process.stdin is None
            ):
                raise PiCopilotError(
                    "pi_gateway_unavailable",
                    "The Pi gateway process is not running.",
                    status_code=503,
                )
            try:
                process.stdin.write(encoded + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise PiCopilotError(
                    "pi_gateway_pipe_closed",
                    "The Pi gateway process closed its input channel.",
                    status_code=503,
                ) from exc

    def request(
        self,
        method: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        event_sink: Optional[EventSink] = None,
        tool_context: Optional[ToolExecutionContext] = None,
    ) -> Dict[str, Any]:
        self._start()
        request_id = uuid.uuid4().hex
        pending = _PendingRequest(
            event_sink=event_sink,
            tool_context=tool_context,
        )
        with self._state_lock:
            self._pending[request_id] = pending
        try:
            self._write(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "request",
                    "request_id": request_id,
                    "method": str(method),
                    "params": dict(params or {}),
                }
            )
            if not pending.done.wait(timeout=max(0.1, float(timeout))):
                if method == "session.prompt":
                    session_id = str((params or {}).get("session_id") or "")
                    if session_id:
                        self._recover_timed_out_prompt(session_id)
                raise PiCopilotError(
                    "pi_gateway_timeout",
                    f"The Pi gateway did not finish {method!r} before the host deadline.",
                    status_code=504,
                    details={"method": method},
                )
            if pending.error is not None:
                raise pending.error
            return dict(pending.result or {})
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)

    def _recover_timed_out_prompt(self, session_id: str) -> None:
        """Best-effort stop and state refresh after the host prompt deadline."""

        try:
            self.request(
                "session.abort",
                {"session_id": session_id},
                timeout=5,
            )
        except PiCopilotError:
            pass
        try:
            self.request(
                "session.state",
                {"session_id": session_id},
                timeout=5,
            )
        except PiCopilotError:
            pass

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                if len(line.encode("utf-8")) > MAX_PROTOCOL_LINE_BYTES:
                    self._fail_all(
                        PiCopilotError(
                            "pi_protocol_line_too_large",
                            "The Pi gateway emitted an oversized protocol line.",
                            status_code=502,
                        )
                    )
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    self._fail_all(
                        PiCopilotError(
                            "pi_protocol_invalid_json",
                            "The Pi gateway emitted invalid JSON.",
                            status_code=502,
                        )
                    )
                    continue
                if not isinstance(payload, dict):
                    continue
                self._handle_payload(payload)
        finally:
            return_code = process.poll()
            self._fail_all(
                PiCopilotError(
                    "pi_gateway_exited",
                    "The Pi gateway process exited unexpectedly.",
                    status_code=503,
                    details={"return_code": return_code},
                )
            )

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            # Stderr is diagnostic-only and never crosses the browser/model
            # boundary. Keep a bounded in-memory tail for local debugging.
            self._stderr_tail.append(line.strip()[:1000])

    def _handle_payload(self, payload: Dict[str, Any]) -> None:
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            self._fail_all(
                PiCopilotError(
                    "pi_protocol_version_unsupported",
                    "The Pi gateway emitted an unsupported protocol version.",
                    status_code=502,
                )
            )
            return
        kind = payload.get("kind")
        if kind == "tool_request":
            self._handle_tool_request(payload)
            return
        request_id = str(payload.get("request_id") or "")
        with self._state_lock:
            pending = self._pending.get(request_id)
        if pending is None:
            return
        allowed_fields = {
            "event": {
                "protocol_version",
                "kind",
                "request_id",
                "session_id",
                "event",
            },
            "response": {
                "protocol_version",
                "kind",
                "request_id",
                "ok",
                "result",
                "error",
            },
        }.get(str(kind))
        if allowed_fields is None or set(payload) - allowed_fields:
            pending.error = PiCopilotError(
                "pi_protocol_unknown_fields",
                "The Pi gateway emitted an envelope with unknown fields.",
                status_code=502,
            )
            pending.done.set()
            return
        if kind == "event":
            event = payload.get("event")
            if not isinstance(event, dict):
                pending.error = PiCopilotError(
                    "pi_protocol_event_invalid",
                    "The Pi gateway emitted an invalid event payload.",
                    status_code=502,
                )
                pending.done.set()
                return
            if (
                pending.tool_context is not None
                and str(payload.get("session_id") or "")
                != pending.tool_context.session.session_id
            ):
                pending.error = PiCopilotError(
                    "pi_protocol_session_mismatch",
                    "The Pi gateway emitted an event for a different session.",
                    status_code=502,
                )
                pending.done.set()
                return
            if pending.event_sink:
                try:
                    pending.event_sink(dict(event))
                except Exception:
                    # UI progress rendering cannot alter the Pi/scientific
                    # authority result. The final response still arrives.
                    pass
            return
        if not isinstance(payload.get("ok"), bool):
            pending.error = PiCopilotError(
                "pi_protocol_response_invalid",
                "The Pi gateway emitted an invalid response payload.",
                status_code=502,
            )
            pending.done.set()
            return
        if payload.get("ok"):
            result = payload.get("result")
            if not isinstance(result, dict):
                pending.error = PiCopilotError(
                    "pi_protocol_response_invalid",
                    "The Pi gateway response result must be an object.",
                    status_code=502,
                )
            else:
                pending.result = dict(result)
        else:
            error = payload.get("error")
            if not isinstance(error, dict):
                pending.error = PiCopilotError(
                    "pi_protocol_response_invalid",
                    "The Pi gateway error response must be an object.",
                    status_code=502,
                )
            else:
                pending.error = PiCopilotError(
                    str(error.get("code") or "pi_gateway_error"),
                    str(error.get("message") or "The Pi gateway request failed."),
                    status_code=502,
                    details=(
                        dict(error.get("details"))
                        if isinstance(error.get("details"), dict)
                        else {}
                    ),
                )
        pending.done.set()

    def _handle_tool_request(self, payload: Dict[str, Any]) -> None:
        allowed = {
            "protocol_version",
            "kind",
            "request_id",
            "parent_request_id",
            "session_id",
            "method",
            "params",
        }
        unknown = sorted(set(payload) - allowed)
        request_id = str(payload.get("request_id") or "")
        parent_request_id = str(payload.get("parent_request_id") or "")
        with self._state_lock:
            pending = self._pending.get(parent_request_id)
        if (
            unknown
            or not request_id
            or payload.get("method") != "tool.execute"
            or pending is None
            or pending.tool_context is None
            or str(payload.get("session_id") or "")
            != pending.tool_context.session.session_id
        ):
            self._send_tool_error(
                request_id or "unknown",
                "pi_tool_request_invalid",
                "The Pi host rejected an invalid or unbound tool request.",
            )
            return
        params = payload.get("params")
        params = params if isinstance(params, dict) else {}
        if set(params) - {"name", "arguments"}:
            self._send_tool_error(
                request_id,
                "pi_tool_request_invalid",
                "The Pi host rejected unknown tool request fields.",
            )
            return
        try:
            result = self._tool_executor(
                str(params.get("name") or ""),
                (
                    params.get("arguments")
                    if isinstance(params.get("arguments"), dict)
                    else {}
                ),
                pending.tool_context,
            )
            self._write(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "tool_response",
                    "request_id": request_id,
                    "ok": True,
                    "result": result,
                }
            )
        except PiCopilotError as exc:
            self._send_tool_error(request_id, exc.code, exc.message)
        except Exception:
            self._send_tool_error(
                request_id,
                "pi_host_tool_failed",
                "The EasyICU host tool failed without exposing traceback text.",
            )

    def _send_tool_error(self, request_id: str, code: str, message: str) -> None:
        try:
            self._write(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "tool_response",
                    "request_id": request_id,
                    "ok": False,
                    "error": {"code": code, "message": message},
                }
            )
        except PiCopilotError:
            pass

    def _fail_all(self, error: PiCopilotError) -> None:
        with self._state_lock:
            pending = list(self._pending.values())
        for row in pending:
            if not row.done.is_set():
                row.error = error
                row.done.set()

    def close(self) -> None:
        with self._state_lock:
            process = self._process
            self._process = None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


__all__ = ["PiGatewayClient"]
