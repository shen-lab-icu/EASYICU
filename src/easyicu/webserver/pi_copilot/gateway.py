"""Long-lived strict JSON-lines client for the pinned Pi Node sidecar."""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import select
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Mapping, Optional

import psutil

from .contracts import (
    PROTOCOL_VERSION,
    PiCopilotError,
    ToolExecutionContext,
)
from .install import (
    packaged_app_dir,
    packaged_runtime_is_complete,
    preferred_app_dir,
    runtime_is_installed,
)
from .host_tool_dispatcher import (
    HostToolDispatchRejected,
    HostToolDispatcher,
    HostToolOutcome,
)
from .provider_config import PiProviderConfig, PiProviderConfigStore
from .tools import MUTATING_HOST_TOOLS, execute_tool

MAX_PROTOCOL_LINE_BYTES = 1024 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15 * 60
STARTUP_READY_TIMEOUT_SECONDS = 15.0
_LOG = logging.getLogger(__name__)
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
        "EASYICU_PI_CODEX_AUTH_FILE",
        "EASYICU_PI_CODEX_SESSION_SHA256",
        "EASYICU_PI_CONTEXT_WINDOW",
        "EASYICU_PI_MAX_TOKENS",
        "EASYICU_PI_SESSION_TOKEN_BUDGET",
        "EASYICU_PI_MAX_PROVIDER_CALLS_PER_MESSAGE",
        "EASYICU_PI_MAX_PROVIDER_CALLS_PER_SESSION",
        "EASYICU_PI_INPUT_PRICE_USD_PER_1M_TOKENS",
        "EASYICU_PI_OUTPUT_PRICE_USD_PER_1M_TOKENS",
        "EASYICU_PI_MAX_COST_USD_PER_MESSAGE",
        "EASYICU_PI_MAX_COST_USD_PER_SESSION",
        "EASYICU_PI_MAX_OPEN_SESSIONS",
        "EASYICU_PI_SESSION_IDLE_SECONDS",
        "EASYICU_PI_SOFT_RSS_MB",
        "EASYICU_PI_EMERGENCY_RSS_MB",
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
    process: Optional[subprocess.Popen[str]] = None


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
        account_binding_sha256: Optional[str] = None,
        tool_executor: Callable[
            [str, Mapping[str, Any], ToolExecutionContext], Dict[str, Any]
        ] = execute_tool,
        tool_dispatch_max_workers: int = 4,
        tool_dispatch_max_pending: int = 64,
    ) -> None:
        self.app_dir = (
            Path(app_dir)
            if app_dir is not None
            else preferred_app_dir()
        ).resolve()
        self.entrypoint = self.app_dir / "src" / "main.mjs"
        self.declared_session_dir = (
            Path(session_dir)
            if session_dir is not None
            else Path.home() / ".easyicu" / "pi-agent" / "sessions"
        ).expanduser().absolute()
        self.session_dir = self.declared_session_dir.resolve()
        self.declared_cwd = (
            Path(cwd)
            if cwd is not None
            else self.declared_session_dir.parent / "workspace"
        ).expanduser().absolute()
        self.cwd = self.declared_cwd.resolve()
        self.provider_store = provider_store or PiProviderConfigStore()
        self.account_binding_sha256 = str(account_binding_sha256 or "").strip()
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
        if self.account_binding_sha256:
            # One immutable credential authority per child: account-backed Pi
            # must never inherit a previously configured API secret.
            self.environ.pop("EASYICU_PI_API_KEY", None)
            auth_file = Path(
                str(self.environ.get("EASYICU_PI_CODEX_AUTH_FILE") or "")
            )
            if (
                re.fullmatch(r"[a-f0-9]{64}", self.account_binding_sha256)
                is None
                or self.environ.get("EASYICU_PI_CODEX_SESSION_SHA256")
                != self.account_binding_sha256
                or self.environ.get("EASYICU_PI_PROVIDER") != "openai-codex"
                or self.environ.get("EASYICU_PI_API")
                != "openai-codex-responses"
                or not auth_file.is_absolute()
                or auth_file.is_symlink()
                or not auth_file.is_file()
                or auth_file.resolve() != auth_file
            ):
                raise PiCopilotError(
                    "pi_codex_account_authority_invalid",
                    "The Codex account conversation boundary is invalid.",
                    status_code=409,
                )
        self._tool_executor = tool_executor
        self._tool_dispatch_max_workers = int(tool_dispatch_max_workers)
        self._tool_dispatch_max_pending = int(tool_dispatch_max_pending)
        self._tool_dispatch_generation = 0
        self._tool_dispatcher: Optional[HostToolDispatcher] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._startup_lock = threading.Lock()
        self._ready_process: Optional[subprocess.Popen[str]] = None
        self._startup_diagnostic: Dict[str, Any] = {}
        # Bookkeeping never shares a lock with process startup or pipe I/O:
        # timed-out requests must be removed without waiting on either.
        self._pending_lock = threading.Lock()
        self._pending: Dict[str, _PendingRequest] = {}
        self._stderr_tail: Deque[str] = collections.deque(maxlen=20)
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        # A private installed runtime contains thousands of executable files.
        # Verify it once per gateway/sidecar lifetime, then invalidate the
        # receipt whenever that sidecar is closed or exits.
        self._installed_runtime_integrity: Optional[bool] = None

    def _new_tool_dispatcher(self) -> HostToolDispatcher:
        self._tool_dispatch_generation += 1
        generation = self._tool_dispatch_generation
        return HostToolDispatcher(
            max_workers=self._tool_dispatch_max_workers,
            max_pending=self._tool_dispatch_max_pending,
            on_response_error=lambda exc: self._handle_tool_response_error(
                generation, exc
            ),
        )

    def _handle_tool_response_error(
        self, generation: int, error: Exception
    ) -> None:
        with self._state_lock:
            if generation != self._tool_dispatch_generation:
                return
        if isinstance(error, PiCopilotError):
            failure = error
        else:
            failure = PiCopilotError(
                "pi_host_tool_response_failed",
                "The EasyICU host could not return a tool response to Pi.",
                status_code=503,
            )
        self._fail_all(failure)

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

    def _runtime_integrity_status(self, *, packaged: bool) -> bool:
        if packaged:
            return packaged_runtime_is_complete(self.app_dir)
        with self._state_lock:
            if self._installed_runtime_integrity is None:
                self._installed_runtime_integrity = runtime_is_installed(self.app_dir)
            return self._installed_runtime_integrity

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
        packaged = self.app_dir == packaged_app_dir().resolve()
        runtime_integrity_verified = self._runtime_integrity_status(packaged=packaged)
        account_configured = bool(self.account_binding_sha256)
        if account_configured:
            provider_configuration: Dict[str, Any] = {
                "provider": "openai-codex",
                "model": str(
                    self.environ.get("EASYICU_PI_MODEL") or "gpt-5.6-luna"
                ),
                "api_transport": "openai-codex-responses",
                "credential_present": True,
                "connection_verified": True,
                "model_available": True,
                # Account/model validation is not itself a generation receipt.
                # A real prompt may still fail at the provider boundary.
                "inference_verified": False,
                "credential_storage": "browser_isolated_codex_home",
                "secrets_returned": False,
            }
        else:
            provider_configuration = provider_status
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
            "runtime_integrity_verified": runtime_integrity_verified,
            "api_key_configured": account_configured
            or bool(str(self.environ.get("EASYICU_PI_API_KEY") or "").strip()),
            "provider_connection_verified": account_configured
            or bool(provider_status.get("connection_verified")),
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
            "provider_configuration": provider_configuration,
        }

    def apply_provider_config(self, config: PiProviderConfig) -> None:
        """Restart the sidecar boundary with a newly verified configuration."""

        if self.account_binding_sha256:
            raise PiCopilotError(
                "pi_codex_account_reconfigure_forbidden",
                "A browser-bound Codex conversation cannot be changed to an API key.",
                status_code=409,
            )

        self.close()
        with self._state_lock:
            provider_environment = config.as_environment()
            for key in provider_environment:
                self.environ.pop(key, None)
            self.environ.update(provider_environment)
            self._provider_file_enabled = True

    @staticmethod
    def _remaining_before_deadline(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PiCopilotError(
                "pi_gateway_timeout", "The Pi readiness deadline expired.", status_code=504,
            )
        return remaining

    def _acquire_before_deadline(self, lock: Any, deadline: Optional[float]) -> None:
        if deadline is None:
            lock.acquire()
        elif not lock.acquire(timeout=self._remaining_before_deadline(deadline)):
            raise PiCopilotError(
                "pi_gateway_timeout", "The Pi readiness lock deadline expired.", status_code=504,
            )

    def _start(self) -> subprocess.Popen[str]:
        started = time.monotonic()
        deadline = started + STARTUP_READY_TIMEOUT_SECONDS
        if not self._startup_lock.acquire(timeout=STARTUP_READY_TIMEOUT_SECONDS):
            raise PiCopilotError(
                "pi_gateway_startup_timeout",
                "The Pi runtime did not become ready within its startup budget.",
                status_code=504,
                details={"phase": "startup_lock", "budget_seconds": STARTUP_READY_TIMEOUT_SECONDS,
                         "elapsed_seconds": round(time.monotonic() - started, 3)},
            )
        process = None
        try:
            self._acquire_before_deadline(self._state_lock, deadline)
            try:
                self._spawn_process()
                process = self._process
                self._remaining_before_deadline(deadline)
                if process is not None and self._ready_process is process:
                    return process
            finally:
                self._state_lock.release()
            ready = self._request_started(
                "runtime.status", {}, deadline=deadline, expected_process=process,
            )
            if ready.get("gateway") != "ready":
                raise PiCopilotError(
                    "pi_gateway_not_ready", "The Pi runtime did not confirm readiness.", status_code=502,
                )
            self._acquire_before_deadline(self._state_lock, deadline)
            try:
                self._remaining_before_deadline(deadline)
                if self._process is not process or process is None or process.poll() is not None:
                    raise PiCopilotError("pi_gateway_exited", "The Pi runtime exited during startup.", status_code=503)
                self._ready_process = process
                self._startup_diagnostic = {
                    "status": "ready", "elapsed_seconds": round(time.monotonic() - started, 3),
                    "budget_seconds": STARTUP_READY_TIMEOUT_SECONDS,
                }
            finally:
                self._state_lock.release()
            _LOG.info("Pi startup readiness pid=%s elapsed_seconds=%.3f", process.pid, time.monotonic() - started)
            return process
        except PiCopilotError as exc:
            elapsed = round(time.monotonic() - started, 3)
            code = "pi_gateway_startup_timeout" if exc.code == "pi_gateway_timeout" else exc.code
            diagnostic = {"status": "failed", "code": code, "elapsed_seconds": elapsed,
                          "budget_seconds": STARTUP_READY_TIMEOUT_SECONDS}
            # Publish a replacement snapshot without waiting again on a lock
            # whose acquisition may be the reason readiness already timed out.
            self._startup_diagnostic = diagnostic
            _LOG.warning("Pi startup readiness failed code=%s elapsed_seconds=%.3f", code, elapsed)
            raise PiCopilotError(
                code,
                "The Pi runtime did not become ready within its startup budget."
                if code == "pi_gateway_startup_timeout" else str(exc),
                status_code=exc.status_code,
                details={**exc.details, "phase": "startup_readiness", **diagnostic},
            ) from exc
        finally:
            self._startup_lock.release()

    def _spawn_process(self) -> None:
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
                    "runtime_integrity_verified",
                )
                if not status[key]
            ]
            if missing:
                if missing == ["runtime_integrity_verified"]:
                    raise PiCopilotError(
                        "pi_runtime_integrity_mismatch",
                        "The installed Copilot runtime does not match the packaged content manifest.",
                        status_code=503,
                    )
                raise PiCopilotError(
                    "pi_gateway_not_installed",
                    "The pinned Copilot runtime is not installed.",
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
            self._ready_process = None
            if self._tool_dispatcher is None or self._tool_dispatcher.closed:
                self._tool_dispatcher = self._new_tool_dispatcher()
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

    def _write_before_deadline(
        self, process: subprocess.Popen[str], encoded: str, deadline: float,
    ) -> None:
        """Only the read-only readiness probe uses nonblocking pipe I/O.

        The caller holds the write lock. Restore the original descriptor mode
        before ordinary requests can write; no background sender survives timeout.
        """
        assert process.stdin is not None
        fd = process.stdin.fileno()
        blocking = os.get_blocking(fd)
        os.set_blocking(fd, False)
        try:
            pending = memoryview((encoded + "\n").encode("utf-8"))
            while pending:
                self._remaining_before_deadline(deadline)
                try:
                    written = os.write(fd, pending)
                except BlockingIOError:
                    select.select([], [fd], [], self._remaining_before_deadline(deadline))
                    continue
                if written <= 0:
                    raise BrokenPipeError("Pi readiness pipe closed")
                pending = pending[written:]
            self._remaining_before_deadline(deadline)
        finally:
            os.set_blocking(fd, blocking)

    def _write(
        self, payload: Mapping[str, Any], *,
        expected_process: Optional[subprocess.Popen[str]] = None,
        require_ready: bool = False,
        deadline: Optional[float] = None,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_PROTOCOL_LINE_BYTES:
            raise PiCopilotError(
                "pi_protocol_line_too_large",
                "The Pi gateway request exceeds the protocol size limit.",
                status_code=500,
            )
        self._acquire_before_deadline(self._write_lock, deadline)
        try:
            self._acquire_before_deadline(self._state_lock, deadline)
            try:
                process = self._process
                if (
                    expected_process is not None
                    and (process is not expected_process
                         or (require_ready and self._ready_process is not expected_process))
                ):
                    raise PiCopilotError(
                        "pi_gateway_process_changed",
                        "The ready Pi process changed before the request could be sent.",
                        status_code=503,
                    )
                if process is None or process.poll() is not None or process.stdin is None:
                    raise PiCopilotError(
                        "pi_gateway_unavailable", "The Pi gateway process is not running.",
                        status_code=503,
                    )
            finally:
                self._state_lock.release()
            # Pin this object through I/O; close may detach/stop it, but the
            # request must never look up or write to a replacement mid-send.
            # Keep the state lock free during pipe I/O so readiness queueing can
            # time out even while an ordinary synchronous writer is blocked.
            try:
                if deadline is None:
                    process.stdin.write(encoded + "\n")
                    process.stdin.flush()
                else:
                    self._write_before_deadline(process, encoded, deadline)
            except (OSError, ValueError) as exc:
                raise PiCopilotError(
                    "pi_gateway_pipe_closed",
                    "The Pi gateway process closed its input channel.",
                    status_code=503,
                ) from exc
        finally:
            self._write_lock.release()

    def request(
        self,
        method: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        event_sink: Optional[EventSink] = None,
        tool_context: Optional[ToolExecutionContext] = None,
    ) -> Dict[str, Any]:
        process = self._start()
        return self._request_started(
            method, params, timeout=timeout, event_sink=event_sink, tool_context=tool_context,
            expected_process=process, require_ready=True,
        )

    def _request_started(
        self,
        method: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        event_sink: Optional[EventSink] = None,
        tool_context: Optional[ToolExecutionContext] = None,
        expected_process: Optional[subprocess.Popen[str]] = None,
        require_ready: bool = False,
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One transport attempt; readiness calls this without recursively starting."""
        request_id = uuid.uuid4().hex
        pending = _PendingRequest(
            event_sink=event_sink,
            tool_context=tool_context,
            process=expected_process,
        )
        with self._pending_lock:
            self._pending[request_id] = pending
        try:
            self._write(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "request",
                    "request_id": request_id,
                    "method": str(method),
                    "params": dict(params or {}),
                },
                expected_process=expected_process, require_ready=require_ready,
                deadline=deadline,
            )
            wait_timeout = (
                self._remaining_before_deadline(deadline)
                if deadline is not None else max(0.1, float(timeout))
            )
            if not pending.done.wait(timeout=wait_timeout):
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
            if deadline is not None:
                self._remaining_before_deadline(deadline)
            return dict(pending.result or {})
        finally:
            with self._pending_lock:
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
                        ), process=process,
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
                        ), process=process,
                    )
                    continue
                if not isinstance(payload, dict):
                    continue
                self._handle_payload(payload)
        finally:
            return_code = process.poll()
            dispatcher: Optional[HostToolDispatcher] = None
            with self._state_lock:
                if self._process is process:
                    self._ready_process = None
                    self._installed_runtime_integrity = None
                    dispatcher = self._tool_dispatcher
                    self._tool_dispatcher = None
            gateway_exit = PiCopilotError(
                "pi_gateway_exited",
                "The Pi gateway process exited unexpectedly.",
                status_code=503,
                details={"return_code": return_code},
            )
            # Preserve the precise process-exit cause before dispatcher
            # shutdown attempts any final response writes to the dead pipe.
            self._fail_all(gateway_exit, process=process)
            if dispatcher is not None:
                dispatcher.shutdown()

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
        with self._pending_lock:
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
        with self._pending_lock:
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
        tool_name = str(params.get("name") or "")
        tool_arguments = (
            dict(params.get("arguments"))
            if isinstance(params.get("arguments"), dict)
            else {}
        )
        tool_context = pending.tool_context
        session_id = tool_context.session.session_id
        with self._state_lock:
            dispatcher = self._tool_dispatcher
        if dispatcher is None:
            self._send_tool_error(
                request_id,
                "pi_host_tool_dispatcher_closed",
                "The EasyICU host-tool dispatcher is closed.",
            )
            return

        def execute() -> Dict[str, Any]:
            with self._pending_lock:
                current_parent = self._pending.get(parent_request_id)
            if current_parent is not pending or pending.done.is_set():
                raise PiCopilotError(
                    "pi_tool_parent_request_stale",
                    "The Pi host rejected a tool whose parent request is no longer active.",
                    status_code=409,
                )
            return self._tool_executor(tool_name, tool_arguments, tool_context)

        def respond(outcome: HostToolOutcome) -> None:
            self._write_tool_outcome(request_id, outcome)

        try:
            dispatcher.submit(
                session_id=session_id,
                operation_id=request_id,
                mutating=tool_name in MUTATING_HOST_TOOLS,
                execute=execute,
                respond=respond,
            )
        except HostToolDispatchRejected as exc:
            self._send_tool_error(request_id, exc.code, exc.message)

    def _write_tool_outcome(
        self, request_id: str, outcome: HostToolOutcome
    ) -> None:
        if outcome.ok:
            payload: Dict[str, Any] = {
                "protocol_version": PROTOCOL_VERSION,
                "kind": "tool_response",
                "request_id": request_id,
                "ok": True,
                "result": dict(outcome.result or {}),
            }
        else:
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "kind": "tool_response",
                "request_id": request_id,
                "ok": False,
                "error": {
                    "code": str(outcome.error_code or "pi_host_tool_failed"),
                    "message": str(
                        outcome.error_message or "The EasyICU host tool failed."
                    ),
                    **(
                        {
                            "details": {
                                "operation_id": outcome.operation_id,
                                "operation_state": outcome.operation_state,
                            }
                        }
                        if outcome.operation_id and outcome.operation_state
                        else {}
                    ),
                },
            }
        try:
            self._write(payload)
        except (TypeError, ValueError):
            if not outcome.ok:
                raise
            self._write(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "tool_response",
                    "request_id": request_id,
                    "ok": False,
                    "error": {
                        "code": "pi_host_tool_result_invalid",
                        "message": "The EasyICU host tool returned a non-serializable result.",
                    },
                }
            )
        except PiCopilotError as exc:
            if not outcome.ok or exc.code != "pi_protocol_line_too_large":
                raise
            self._write(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "tool_response",
                    "request_id": request_id,
                    "ok": False,
                    "error": {
                        "code": "pi_host_tool_result_too_large",
                        "message": "The EasyICU host tool result exceeded the protocol limit.",
                    },
                }
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

    def _fail_all(
        self, error: PiCopilotError, *, process: Optional[subprocess.Popen[str]] = None,
    ) -> None:
        with self._pending_lock:
            # A late EOF/close from an old child must not fail its replacement.
            pending = [
                row for row in self._pending.values()
                if process is None or row.process is process
            ]
        for row in pending:
            if not row.done.is_set():
                row.error = error
                row.done.set()

    def memory_status(self) -> Dict[str, Any]:
        """Read sidecar RSS and bounded hot-session state without starting it."""

        with self._state_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return {"running": False, "pid": None, "rss_mb": 0.0}
        pid = int(process.pid)
        try:
            rss_mb = round(psutil.Process(pid).memory_info().rss / (1024**2), 1)
        except (psutil.Error, OSError):
            rss_mb = 0.0
        status: Dict[str, Any] = {
            "running": True,
            "pid": pid,
            "rss_mb": rss_mb,
        }
        try:
            runtime = self.request("runtime.memory", {}, timeout=5)
        except (OSError, PiCopilotError) as exc:
            status["diagnostic_error"] = getattr(exc, "code", "pi_gateway_io_error")
        else:
            status.update(runtime)
            status["running"] = True
        with self._state_lock:
            status["startup_readiness"] = dict(self._startup_diagnostic)
        return status

    def maintain_sessions(self, *, exclude_session_id: str = "") -> Dict[str, Any]:
        """Ask an already-running sidecar to unload safe idle sessions."""

        with self._state_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return {"running": False, "maintained": False}
        try:
            status = self.request(
                "runtime.maintain",
                {"exclude_session_id": str(exclude_session_id or "")},
                timeout=5,
            )
        except (OSError, PiCopilotError) as exc:
            return {
                "running": process.poll() is None,
                "maintained": False,
                "diagnostic_error": getattr(exc, "code", "pi_gateway_io_error"),
            }
        return {"running": True, "maintained": True, **status}

    def close(self) -> None:
        with self._state_lock:
            process = self._process
            dispatcher = self._tool_dispatcher
            self._process = None
            self._ready_process = None
            self._tool_dispatcher = None
            self._installed_runtime_integrity = None
        if dispatcher is not None:
            dispatcher.shutdown()
        if process is None:
            return
        self._fail_all(
            PiCopilotError(
                "pi_gateway_closed",
                "The Pi gateway was closed before the request completed.",
                status_code=503,
            ), process=process,
        )
        # Never wait on a writer blocked in the child's full stdin pipe.
        # Closing its TextIOWrapper can also wait on that active write, so send
        # EOF only when idle; otherwise stop the pinned child before closing it.
        writer_idle = self._write_lock.acquire(blocking=False)
        if writer_idle:
            try:
                if process.stdin:
                    process.stdin.close()
            except OSError:
                pass
            finally:
                self._write_lock.release()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        finally:
            if not writer_idle and process.poll() is not None:
                try:
                    if process.stdin:
                        process.stdin.close()
                except OSError:
                    pass


__all__ = ["PiGatewayClient"]
