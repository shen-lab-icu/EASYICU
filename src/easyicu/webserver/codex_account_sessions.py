"""Per-browser Codex/ChatGPT authentication for the native Web product.

The browser receives only an opaque HttpOnly session cookie. Codex-managed
tokens stay inside an isolated server-side ``CODEX_HOME`` and are never read,
serialized, or copied by EasyICU.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from fastapi import Request, Response

from easyicu.research_agent.providers.codex_app_server import (
    CodexAppServerError,
    CodexAppServerRuntime,
)
from easyicu.research_agent.providers.subprocess_env import (
    build_provider_subprocess_env,
)
from easyicu.webserver import state_paths

COOKIE_NAME = "easyicu_codex_user_session"
SESSION_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{40,100}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_USER_CODE_RE = re.compile(r"[A-Za-z0-9-]{4,32}")


class CodexAccountSessionError(ValueError):
    """Stable, sanitized user-session boundary failure."""

    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


@dataclass(frozen=True)
class CodexSessionCoordinates:
    binding_sha256: str
    root: Path
    home: Path
    codex_home: Path
    runtime_cwd: Path


@dataclass
class _ManagedRuntime:
    runtime: CodexAppServerRuntime
    login_id: str = ""
    verification_url: str = ""
    user_code: str = ""
    lock: Any = field(default_factory=threading.RLock, repr=False)


_RUNTIMES: dict[str, _ManagedRuntime] = {}
_RUNTIMES_LOCK = threading.RLock()


def _sessions_root() -> Path:
    return state_paths.state_root() / "codex-user-sessions"


def _binding(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _coordinates(token: str, *, create: bool) -> CodexSessionCoordinates:
    if _TOKEN_RE.fullmatch(token) is None:
        raise CodexAccountSessionError("codex_auth_session_invalid")
    binding = _binding(token)
    if _DIGEST_RE.fullmatch(binding) is None:  # pragma: no cover - hashlib invariant
        raise CodexAccountSessionError("codex_auth_session_invalid")
    root = (_sessions_root() / binding).resolve()
    sessions_root = _sessions_root().resolve()
    if root.parent != sessions_root:
        raise CodexAccountSessionError("codex_auth_session_invalid")
    coordinates = CodexSessionCoordinates(
        binding_sha256=binding,
        root=root,
        home=root / "home",
        codex_home=root / "codex",
        runtime_cwd=root / "runtime",
    )
    if create:
        for path in (
            sessions_root,
            coordinates.root,
            coordinates.home,
            coordinates.codex_home,
            coordinates.runtime_cwd,
        ):
            if path.is_symlink():
                raise CodexAccountSessionError("codex_auth_session_invalid")
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            if (
                path.is_symlink()
                or not path.is_dir()
                or path.resolve() != path
            ):
                raise CodexAccountSessionError("codex_auth_session_invalid")
            path.chmod(0o700)
    else:
        for path in (
            coordinates.root,
            coordinates.home,
            coordinates.codex_home,
            coordinates.runtime_cwd,
        ):
            if not path.exists():
                raise CodexAccountSessionError("codex_auth_login_required")
            if (
                path.is_symlink()
                or not path.is_dir()
                or path.resolve() != path
            ):
                raise CodexAccountSessionError("codex_auth_session_invalid")
    return coordinates


def _request_token(request: Request) -> str:
    token = str(request.cookies.get(COOKIE_NAME) or "").strip()
    if not token:
        raise CodexAccountSessionError("codex_auth_login_required")
    return token


def _ensure_session(
    request: Request,
    response: Response,
) -> CodexSessionCoordinates:
    token = str(request.cookies.get(COOKIE_NAME) or "").strip()
    if _TOKEN_RE.fullmatch(token) is None:
        token = secrets.token_urlsafe(32)
    coordinates = _coordinates(token, create=True)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )
    return coordinates


def environment_for_coordinates(
    coordinates: CodexSessionCoordinates,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = dict(os.environ if environ is None else environ)
    source["HOME"] = str(coordinates.home)
    source["CODEX_HOME"] = str(coordinates.codex_home)
    source["EASYICU_ALLOW_EXTERNAL_LLM"] = "1"
    source["EASYICU_CODEX_SESSION_SHA256"] = coordinates.binding_sha256
    return build_provider_subprocess_env(
        "codex",
        environment=source,
        required_keys=(
            "EASYICU_ALLOW_EXTERNAL_LLM",
            "EASYICU_CODEX_SESSION_SHA256",
            "EASYICU_CODEX_MODEL",
            "CODEX_MODEL",
        ),
    )


def environment_for_request(request: Request) -> dict[str, str]:
    """Return only the current browser user's isolated account environment."""

    coordinates = _coordinates(_request_token(request), create=False)
    return environment_for_coordinates(coordinates)


def _runtime_for(coordinates: CodexSessionCoordinates) -> _ManagedRuntime:
    with _RUNTIMES_LOCK:
        managed = _RUNTIMES.get(coordinates.binding_sha256)
        if managed is None:
            managed = _ManagedRuntime(
                runtime=CodexAppServerRuntime(
                    environment=environment_for_coordinates(coordinates),
                    cwd=coordinates.runtime_cwd,
                    request_timeout=30.0,
                )
            )
            _RUNTIMES[coordinates.binding_sha256] = managed
        return managed


def _masked_email(value: object) -> str | None:
    text = str(value or "").strip()
    if "@" not in text:
        return None
    local, domain = text.rsplit("@", 1)
    if not local or not domain:
        return None
    return f"{local[:1]}***@{domain}"


def _base_status(*, session_present: bool) -> dict[str, Any]:
    return {
        "schema_version": "easyicu.codex-user-auth-status/1",
        "provider": "codex",
        "provider_identity": "codex-app-server",
        "authentication_mode": "chatgpt_account",
        "session_present": bool(session_present),
        "authentication_verified": False,
        "account_session_present": False,
        "account_session_status": "codex_auth_login_required",
        "plan_type": None,
        "account_label": None,
        "secrets_returned": False,
    }


def status(request: Request) -> dict[str, Any]:
    """Read sanitized account state for this browser only."""

    try:
        coordinates = _coordinates(_request_token(request), create=False)
    except CodexAccountSessionError:
        return _base_status(session_present=False)
    managed = _runtime_for(coordinates)
    with managed.lock:
        return _status_for_managed(managed)


def _status_for_managed(managed: _ManagedRuntime) -> dict[str, Any]:
    public = _base_status(session_present=True)
    try:
        result = managed.runtime.request(
            "account/read",
            {"refreshToken": False},
            timeout=15.0,
        )
    except CodexAppServerError as exc:
        public["account_session_status"] = exc.code
        return public
    account = result.get("account")
    if isinstance(account, Mapping) and account.get("type") == "chatgpt":
        public.update(
            {
                "authentication_verified": True,
                "account_session_present": True,
                "account_session_status": "codex_auth_ready",
                "plan_type": str(account.get("planType") or "unknown"),
                "account_label": _masked_email(account.get("email")),
            }
        )
        managed.login_id = ""
        managed.verification_url = ""
        managed.user_code = ""
        return public
    if managed.login_id:
        failure = next(
            (
                item
                for item in reversed(managed.runtime.notifications_since())
                if item.get("method") == "account/login/completed"
                and (item.get("params") or {}).get("loginId") == managed.login_id
                and not bool((item.get("params") or {}).get("success"))
            ),
            None,
        )
        if failure is not None:
            managed.login_id = ""
            managed.verification_url = ""
            managed.user_code = ""
            public["account_session_status"] = "codex_auth_login_failed"
        else:
            public["account_session_status"] = "codex_auth_login_pending"
    return public


def _validated_verification_url(value: object) -> str:
    text = str(value or "").strip()
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "auth.openai.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path.rstrip("/") != "/codex/device"
        or parsed.fragment
    ):
        raise CodexAccountSessionError("codex_auth_verification_url_invalid")
    return text


def start_login(request: Request, response: Response) -> dict[str, Any]:
    """Start the official managed ChatGPT device-code ceremony."""

    coordinates = _ensure_session(request, response)
    managed = _runtime_for(coordinates)
    with managed.lock:
        current = _status_for_managed(managed)
        if current["authentication_verified"]:
            return {"ok": True, "auth": current, "login_started": False}
        if managed.login_id and managed.verification_url and managed.user_code:
            return {
                "ok": True,
                "auth": current,
                "login_started": True,
                "verification_url": managed.verification_url,
                "user_code": managed.user_code,
            }
        try:
            result = managed.runtime.request(
                "account/login/start",
                {"type": "chatgptDeviceCode"},
                timeout=30.0,
            )
        except CodexAppServerError as exc:
            raise CodexAccountSessionError(exc.code) from exc
        login_id = str(result.get("loginId") or "").strip()
        verification_url = _validated_verification_url(result.get("verificationUrl"))
        user_code = str(result.get("userCode") or "").strip()
        if not login_id or _USER_CODE_RE.fullmatch(user_code) is None:
            raise CodexAccountSessionError("codex_auth_device_code_invalid")
        managed.login_id = login_id
        managed.verification_url = verification_url
        managed.user_code = user_code
        return {
            "ok": True,
            "auth": _status_for_managed(managed),
            "login_started": True,
            "verification_url": verification_url,
            "user_code": user_code,
        }


def status_for_coordinates(coordinates: CodexSessionCoordinates) -> dict[str, Any]:
    """Internal status variant used after the response cookie is created."""

    managed = _runtime_for(coordinates)
    with managed.lock:
        return _status_for_managed(managed)


def cancel_login(request: Request) -> dict[str, Any]:
    coordinates = _coordinates(_request_token(request), create=False)
    managed = _runtime_for(coordinates)
    with managed.lock:
        if managed.login_id:
            try:
                managed.runtime.request(
                    "account/login/cancel",
                    {"loginId": managed.login_id},
                    timeout=15.0,
                )
            except CodexAppServerError as exc:
                raise CodexAccountSessionError(exc.code) from exc
        managed.login_id = ""
        managed.verification_url = ""
        managed.user_code = ""
        return {"ok": True, "auth": _status_for_managed(managed)}


def logout(request: Request) -> dict[str, Any]:
    coordinates = _coordinates(_request_token(request), create=False)
    managed = _runtime_for(coordinates)
    with managed.lock:
        try:
            managed.runtime.request(
                "account/logout",
                {},
                timeout=15.0,
            )
        except CodexAppServerError as exc:
            raise CodexAccountSessionError(exc.code) from exc
        managed.login_id = ""
        managed.verification_url = ""
        managed.user_code = ""
        return {"ok": True, "auth": _status_for_managed(managed)}


def shutdown_all() -> None:
    """Stop only App Server children owned by this EasyICU process."""

    with _RUNTIMES_LOCK:
        managed = list(_RUNTIMES.values())
        _RUNTIMES.clear()
    for entry in managed:
        entry.runtime.close()


__all__ = [
    "COOKIE_NAME",
    "CodexAccountSessionError",
    "CodexSessionCoordinates",
    "cancel_login",
    "environment_for_coordinates",
    "environment_for_request",
    "logout",
    "shutdown_all",
    "start_login",
    "status",
]
