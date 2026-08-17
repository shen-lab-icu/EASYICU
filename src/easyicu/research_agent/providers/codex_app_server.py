"""Minimal stdio client for the official Codex App Server protocol.

This module owns JSON-RPC process transport only. Browser cookies and the
filesystem location of each user's managed ChatGPT login belong to the Web
session owner; Research Agent policy and evidence authority remain unchanged.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..authority.secret_redaction import redact_text_secrets
from .subprocess_env import CODEX_APP_SERVER_EXECUTABLE_ENV


_CHATGPT_APP_CODEX = Path("/Applications/ChatGPT.app/Contents/Resources/codex")


class CodexAppServerError(RuntimeError):
    """Typed App Server boundary failure with a stable reason code."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code)
        super().__init__(message or self.code)


def _resolved_executable(path: Path, *, error_code: str) -> str:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CodexAppServerError(error_code) from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise CodexAppServerError(error_code)
    return str(resolved)


def resolve_codex_app_server_executable(
    environment: Mapping[str, str],
    *,
    configured: str = "codex",
) -> str:
    """Resolve one reviewed App Server binary without mutating global tools."""

    override = str(environment.get(CODEX_APP_SERVER_EXECUTABLE_ENV) or "").strip()
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            raise CodexAppServerError("codex_auth_executable_override_invalid")
        return _resolved_executable(
            candidate,
            error_code="codex_auth_executable_override_invalid",
        )

    configured_text = str(configured or "codex").strip() or "codex"
    configured_path = Path(configured_text)
    if configured_path.is_absolute():
        return _resolved_executable(
            configured_path,
            error_code="codex_auth_executable_missing",
        )
    if configured_path.name != configured_text:
        raise CodexAppServerError("codex_auth_executable_missing")

    if _CHATGPT_APP_CODEX.is_file() and os.access(_CHATGPT_APP_CODEX, os.X_OK):
        return str(_CHATGPT_APP_CODEX.resolve())

    executable = shutil.which(
        configured_text,
        path=environment.get("PATH"),
    )
    if executable is None:
        raise CodexAppServerError("codex_auth_executable_missing")
    return str(Path(executable).resolve())


class CodexAppServerRuntime:
    """One initialized Codex App Server stdio process."""

    def __init__(
        self,
        *,
        environment: Mapping[str, str],
        cwd: Path,
        executable: str = "codex",
        request_timeout: float = 30.0,
        experimental_api: bool = False,
    ) -> None:
        self._environment = {str(key): str(value) for key, value in environment.items()}
        self._cwd = Path(cwd).resolve()
        self._executable = str(executable or "codex")
        self._request_timeout = max(0.1, float(request_timeout))
        self._experimental_api = bool(experimental_api)
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._notifications: list[dict[str, Any]] = []
        self._notification_offset = 0
        self._stderr: list[str] = []
        self._lifecycle_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._state = threading.Condition(threading.RLock())
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None

    @property
    def notification_count(self) -> int:
        with self._state:
            return self._notification_offset + len(self._notifications)

    def start(self) -> None:
        """Spawn, initialize, and prove the exact isolated ``CODEX_HOME``."""

        with self._lifecycle_lock:
            if self._process is not None and self._process.poll() is None:
                return
            codex_home_raw = str(self._environment.get("CODEX_HOME") or "").strip()
            home_raw = str(self._environment.get("HOME") or "").strip()
            if not codex_home_raw or not home_raw:
                raise CodexAppServerError("codex_auth_isolated_home_required")
            codex_home = Path(codex_home_raw).resolve()
            home = Path(home_raw).resolve()
            if not codex_home.is_absolute() or not home.is_absolute():
                raise CodexAppServerError("codex_auth_isolated_home_invalid")
            self._cwd.mkdir(parents=True, exist_ok=True, mode=0o700)
            home.mkdir(parents=True, exist_ok=True, mode=0o700)
            codex_home.mkdir(parents=True, exist_ok=True, mode=0o700)
            executable = resolve_codex_app_server_executable(
                self._environment,
                configured=self._executable,
            )
            try:
                process = subprocess.Popen(
                    [executable, "app-server", "--stdio"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    cwd=self._cwd,
                    env=dict(self._environment),
                )
            except OSError as exc:
                raise CodexAppServerError(
                    "codex_auth_app_server_start_failed",
                    type(exc).__name__,
                ) from exc
            self._process = process
            self._reader = threading.Thread(
                target=self._read_stdout,
                name="easyicu-codex-app-server-stdout",
                daemon=True,
            )
            self._stderr_reader = threading.Thread(
                target=self._read_stderr,
                name="easyicu-codex-app-server-stderr",
                daemon=True,
            )
            self._reader.start()
            self._stderr_reader.start()
            try:
                initialized = self._request_started(
                    "initialize",
                    {
                        "clientInfo": {
                            "name": "easyicu-research-agent",
                            "version": "1",
                        },
                        "capabilities": {"experimentalApi": self._experimental_api},
                    },
                    timeout=self._request_timeout,
                )
                observed_home = Path(
                    str(initialized.get("codexHome") or "")
                ).resolve()
                if observed_home != codex_home:
                    raise CodexAppServerError("codex_auth_home_mismatch")
                self.notify("initialized")
            except Exception:
                self.close()
                raise

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.start()
        return self._request_started(
            method,
            params,
            timeout=self._request_timeout if timeout is None else float(timeout),
        )

    def _request_started(
        self,
        method: str,
        params: Mapping[str, Any] | None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        with self._state:
            self._request_id += 1
            request_id = self._request_id
            mailbox: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._pending[request_id] = mailbox
        payload: dict[str, Any] = {"id": request_id, "method": str(method)}
        if params is not None:
            payload["params"] = dict(params)
        try:
            self._write(payload)
            try:
                response = mailbox.get(timeout=max(0.1, float(timeout)))
            except queue.Empty as exc:
                raise CodexAppServerError(
                    "codex_auth_app_server_timeout",
                    f"App Server request timed out: {method}",
                ) from exc
        finally:
            with self._state:
                self._pending.pop(request_id, None)
        error = response.get("error")
        if error is not None:
            code = "codex_auth_app_server_request_failed"
            if isinstance(error, Mapping) and int(error.get("code") or 0) == -32601:
                code = "codex_auth_app_server_method_unsupported"
            raise CodexAppServerError(code, f"App Server request failed: {method}")
        result = response.get("result")
        if result is None:
            return {}
        if not isinstance(result, Mapping):
            raise CodexAppServerError("codex_auth_app_server_result_invalid")
        return dict(result)

    def notify(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"method": str(method)}
        if params is not None:
            payload["params"] = dict(params)
        self._write(payload)

    def wait_for_notification(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        after: int = 0,
        timeout: float,
        hard_timeout: float | None = None,
        progress_predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        if progress_predicate is not None and hard_timeout is None:
            raise ValueError("progress-aware notification wait requires hard_timeout")
        started = time.monotonic()
        idle_timeout = max(0.1, float(timeout))
        idle_deadline = started + idle_timeout
        hard_deadline = started + max(
            0.1,
            float(timeout if hard_timeout is None else hard_timeout),
        )
        index = max(0, int(after))
        with self._state:
            while True:
                start = max(0, index - self._notification_offset)
                for notification in self._notifications[start:]:
                    if predicate(notification):
                        return dict(notification)
                    if progress_predicate is not None and progress_predicate(
                        notification
                    ):
                        idle_deadline = time.monotonic() + idle_timeout
                index = self._notification_offset + len(self._notifications)
                now = time.monotonic()
                if now >= hard_deadline:
                    raise CodexAppServerError("codex_auth_notification_hard_timeout")
                remaining = min(idle_deadline, hard_deadline) - now
                if remaining <= 0:
                    raise CodexAppServerError("codex_auth_notification_timeout")
                process = self._process
                if process is None or process.poll() is not None:
                    raise CodexAppServerError(
                        "codex_auth_app_server_exited",
                        self._safe_stderr(),
                    )
                self._state.wait(timeout=remaining)

    def notifications_since(self, index: int = 0) -> list[dict[str, Any]]:
        with self._state:
            start = max(0, int(index) - self._notification_offset)
            return [dict(item) for item in self._notifications[start:]]

    def close(self) -> None:
        with self._lifecycle_lock:
            process = self._process
            self._process = None
            if process is None:
                return
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.0)
            with self._state:
                self._state.notify_all()

    def _write(self, payload: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise CodexAppServerError(
                "codex_auth_app_server_exited",
                self._safe_stderr(),
            )
        encoded = json.dumps(
            dict(payload),
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        try:
            with self._write_lock:
                process.stdin.write(encoded + "\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise CodexAppServerError("codex_auth_app_server_write_failed") from exc

    def _read_stdout(self) -> None:
        process = self._process
        stream = process.stdout if process is not None else None
        if stream is None:
            return
        for raw_line in stream:
            try:
                message = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            request_id = message.get("id")
            if isinstance(request_id, int) and (
                "result" in message or "error" in message
            ):
                with self._state:
                    mailbox = self._pending.get(request_id)
                if mailbox is not None:
                    try:
                        mailbox.put_nowait(message)
                    except queue.Full:
                        pass
                continue
            if "method" in message and "id" not in message:
                with self._state:
                    self._notifications.append(message)
                    if len(self._notifications) > 1_024:
                        del self._notifications[:512]
                        self._notification_offset += 512
                    self._state.notify_all()
        with self._state:
            self._state.notify_all()

    def _read_stderr(self) -> None:
        process = self._process
        stream = process.stderr if process is not None else None
        if stream is None:
            return
        for raw_line in stream:
            line = redact_text_secrets(raw_line.strip())[:500]
            if not line:
                continue
            with self._state:
                self._stderr.append(line)
                if len(self._stderr) > 20:
                    del self._stderr[:-20]

    def _safe_stderr(self) -> str:
        with self._state:
            return " | ".join(self._stderr[-3:])[:1_000]

    def __enter__(self) -> "CodexAppServerRuntime":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "CodexAppServerError",
    "CodexAppServerRuntime",
    "resolve_codex_app_server_executable",
]
