"""Per-browser Codex/ChatGPT authentication for the native Web product.

The browser receives only an opaque HttpOnly session cookie. Codex-managed
tokens stay inside an isolated server-side ``CODEX_HOME``. They are never
returned to the browser or copied into a project; the isolated Pi conversation
process may read the current access credential in memory from that exact home.
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
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_SUBSCRIPTION_MODEL_CATALOG_AUTHORITY = "easyicu.codex-subscription-model-catalog/1"
_MAX_CODEX_AUTH_FILE_BYTES = 256 * 1024
# Codex App Server releases can lag the subscription backend catalog. These
# account models are documented by OpenAI and independently shipped by the
# OpenCode/Pi Codex OAuth providers; the real App Server turn remains the final
# capability check.
_DOCUMENTED_SUBSCRIPTION_MODELS = (
    (
        "gpt-5.6-luna",
        "GPT-5.6 Luna",
        "Cost-sensitive, high-volume GPT-5.6 model.",
    ),
    (
        "gpt-5.6-sol",
        "GPT-5.6 Sol",
        "Flagship GPT-5.6 model for complex professional work.",
    ),
    (
        "gpt-5.6-terra",
        "GPT-5.6 Terra",
        "Balanced GPT-5.6 model for intelligence and cost.",
    ),
)


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
    login_flow: str = ""
    auth_url: str = ""
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


def _coordinates_from_binding(
    binding_sha256: str,
) -> CodexSessionCoordinates:
    binding = str(binding_sha256 or "").strip().lower()
    if _DIGEST_RE.fullmatch(binding) is None:
        raise CodexAccountSessionError("codex_auth_session_invalid")
    sessions_root = _sessions_root().resolve()
    root = (sessions_root / binding).resolve()
    if root.parent != sessions_root:
        raise CodexAccountSessionError("codex_auth_session_invalid")
    coordinates = CodexSessionCoordinates(
        binding_sha256=binding,
        root=root,
        home=root / "home",
        codex_home=root / "codex",
        runtime_cwd=root / "runtime",
    )
    for path in (
        coordinates.root,
        coordinates.home,
        coordinates.codex_home,
        coordinates.runtime_cwd,
    ):
        if (
            not path.exists()
            or path.is_symlink()
            or not path.is_dir()
            or path.resolve() != path
        ):
            raise CodexAccountSessionError("codex_auth_login_required")
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
    managed = _runtime_for(coordinates)
    with managed.lock:
        if not _status_for_managed(managed).get("authentication_verified"):
            raise CodexAccountSessionError("codex_auth_login_required")
    return environment_for_coordinates(coordinates)


def binding_for_request(request: Request) -> str:
    """Return the current verified browser binding, never its opaque cookie."""

    coordinates = _coordinates(_request_token(request), create=False)
    managed = _runtime_for(coordinates)
    with managed.lock:
        if not _status_for_managed(managed).get("authentication_verified"):
            raise CodexAccountSessionError("codex_auth_login_required")
    return coordinates.binding_sha256


def environment_for_binding(
    binding_sha256: str,
    *,
    model: str,
) -> dict[str, str]:
    """Resolve one immutable Copilot-session binding for a scientific run."""

    coordinates = _coordinates_from_binding(binding_sha256)
    managed = _runtime_for(coordinates)
    with managed.lock:
        if not _status_for_managed(managed).get("authentication_verified"):
            raise CodexAccountSessionError("codex_auth_login_required")
        selected = _validated_model(managed, model)
    environment = environment_for_coordinates(coordinates)
    environment["EASYICU_CODEX_MODEL"] = selected
    return environment


def _private_auth_file(coordinates: CodexSessionCoordinates) -> Path:
    candidate = coordinates.codex_home / "auth.json"
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise CodexAccountSessionError("codex_auth_login_required") from exc
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.resolve() != candidate
        or candidate.parent.resolve() != coordinates.codex_home
        or metadata.st_size > _MAX_CODEX_AUTH_FILE_BYTES
        or metadata.st_mode & 0o077
    ):
        raise CodexAccountSessionError("codex_auth_file_invalid")
    return candidate


def pi_conversation_environment_for_binding(
    binding_sha256: str,
    *,
    model: str,
) -> dict[str, str]:
    """Project one verified account into an isolated Pi conversation process.

    App Server remains the sole login and refresh owner.  Pi receives only the
    exact private auth-file coordinate and rereads the current access token for
    each provider call; no credential value crosses this Python contract.
    """

    coordinates = _coordinates_from_binding(binding_sha256)
    managed = _runtime_for(coordinates)
    with managed.lock:
        try:
            account = managed.runtime.request(
                "account/read",
                {"refreshToken": True},
                timeout=30.0,
            ).get("account")
        except CodexAppServerError as exc:
            raise CodexAccountSessionError(exc.code) from exc
        if not isinstance(account, Mapping) or account.get("type") != "chatgpt":
            raise CodexAccountSessionError("codex_auth_login_required")
        selected = _validated_model(managed, model)
        auth_file = _private_auth_file(coordinates)
    environment = environment_for_coordinates(coordinates)
    environment.update(
        {
            "EASYICU_PI_PROVIDER": "openai-codex",
            "EASYICU_PI_MODEL": selected,
            "EASYICU_PI_BASE_URL": "https://chatgpt.com/backend-api",
            "EASYICU_PI_API": "openai-codex-responses",
            "EASYICU_PI_CODEX_AUTH_FILE": str(auth_file),
            "EASYICU_PI_CODEX_SESSION_SHA256": coordinates.binding_sha256,
        }
    )
    return environment


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


def _discard_runtime(
    binding_sha256: str,
    managed: _ManagedRuntime,
) -> None:
    with _RUNTIMES_LOCK:
        if _RUNTIMES.get(binding_sha256) is managed:
            _RUNTIMES.pop(binding_sha256, None)
    managed.runtime.close()


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
        managed.login_flow = ""
        managed.auth_url = ""
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
            managed.login_flow = ""
            managed.auth_url = ""
            managed.verification_url = ""
            managed.user_code = ""
            public["account_session_status"] = "codex_auth_login_failed"
        else:
            public["account_session_status"] = "codex_auth_login_pending"
    return public


def _validated_openai_url(value: object, *, device_code: bool) -> str:
    text = str(value or "").strip()
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "auth.openai.com"
        or parsed.username is not None
        or parsed.password is not None
        or (device_code and parsed.path.rstrip("/") != "/codex/device")
        or (not device_code and not parsed.path.startswith("/"))
        or parsed.fragment
    ):
        raise CodexAccountSessionError("codex_auth_url_invalid")
    return text


def start_login(
    request: Request,
    response: Response,
    *,
    flow: str = "browser",
) -> dict[str, Any]:
    """Start Codex-managed browser OAuth, with device code as fallback."""

    coordinates = _ensure_session(request, response)
    managed = _runtime_for(coordinates)
    selected_flow = "device_code" if flow == "device_code" else "browser"
    with managed.lock:
        current = _status_for_managed(managed)
        if current["authentication_verified"]:
            return {"ok": True, "auth": current, "login_started": False}
        if managed.login_id and managed.login_flow == selected_flow:
            return {
                "ok": True,
                "auth": current,
                "login_started": True,
                "flow": managed.login_flow,
                **({"auth_url": managed.auth_url} if managed.auth_url else {}),
                **(
                    {
                        "verification_url": managed.verification_url,
                        "user_code": managed.user_code,
                    }
                    if managed.verification_url and managed.user_code
                    else {}
                ),
            }
        if managed.login_id:
            try:
                managed.runtime.request(
                    "account/login/cancel",
                    {"loginId": managed.login_id},
                    timeout=15.0,
                )
            except CodexAppServerError as exc:
                raise CodexAccountSessionError(exc.code) from exc
        try:
            result = managed.runtime.request(
                "account/login/start",
                (
                    {"type": "chatgptDeviceCode"}
                    if selected_flow == "device_code"
                    else {"type": "chatgpt", "codexStreamlinedLogin": True}
                ),
                timeout=30.0,
            )
        except CodexAppServerError as exc:
            raise CodexAccountSessionError(exc.code) from exc
        login_id = str(result.get("loginId") or "").strip()
        if not login_id:
            raise CodexAccountSessionError("codex_auth_login_response_invalid")
        if selected_flow == "browser":
            auth_url = _validated_openai_url(
                result.get("authUrl"),
                device_code=False,
            )
            verification_url = ""
            user_code = ""
        else:
            auth_url = ""
            verification_url = _validated_openai_url(
                result.get("verificationUrl"),
                device_code=True,
            )
            user_code = str(result.get("userCode") or "").strip()
            if _USER_CODE_RE.fullmatch(user_code) is None:
                raise CodexAccountSessionError("codex_auth_device_code_invalid")
        managed.login_id = login_id
        managed.login_flow = selected_flow
        managed.auth_url = auth_url
        managed.verification_url = verification_url
        managed.user_code = user_code
        return {
            "ok": True,
            "auth": _status_for_managed(managed),
            "login_started": True,
            "flow": selected_flow,
            **({"auth_url": managed.auth_url} if managed.auth_url else {}),
            **(
                {
                    "verification_url": managed.verification_url,
                    "user_code": managed.user_code,
                }
                if managed.verification_url and managed.user_code
                else {}
            ),
        }


def _model_rows(
    managed: _ManagedRuntime,
) -> tuple[list[dict[str, Any]], str]:
    app_server_failure_code = ""
    try:
        result = managed.runtime.request(
            "model/list",
            {"includeHidden": False, "limit": 100},
            timeout=20.0,
        )
    except CodexAppServerError as exc:
        # The authenticated App Server can transiently fail its catalog lookup
        # even though account turns remain available. Keep the documented
        # subscription catalog usable and expose the typed degraded reason;
        # the first real turn remains the final model capability check.
        result = {}
        app_server_failure_code = exc.code
    rows: list[dict[str, Any]] = []
    for raw in result.get("data") or []:
        if not isinstance(raw, Mapping) or bool(raw.get("hidden")):
            continue
        model = str(raw.get("model") or raw.get("id") or "").strip()
        if _MODEL_RE.fullmatch(model) is None:
            continue
        rows.append(
            {
                "id": model,
                "label": str(raw.get("displayName") or model).strip()[:160],
                "description": str(raw.get("description") or "").strip()[:400],
                "is_default": bool(raw.get("isDefault")),
                "catalog_source": "codex_app_server",
            }
        )
    observed = {row["id"] for row in rows}
    for model, label, description in _DOCUMENTED_SUBSCRIPTION_MODELS:
        if model in observed:
            continue
        rows.append(
            {
                "id": model,
                "label": label,
                "description": description,
                "is_default": False,
                "catalog_source": "openai_documented_subscription",
            }
        )
    if not rows:
        raise CodexAccountSessionError("codex_auth_model_catalog_empty")
    return rows, app_server_failure_code


def _validated_model(managed: _ManagedRuntime, model: str) -> str:
    requested = str(model or "").strip()
    rows, _app_server_failure_code = _model_rows(managed)
    if not requested:
        default = next((row["id"] for row in rows if row["is_default"]), None)
        return str(default or rows[0]["id"])
    if requested not in {row["id"] for row in rows}:
        raise CodexAccountSessionError("codex_auth_model_unavailable")
    return requested


def models(request: Request) -> dict[str, Any]:
    coordinates = _coordinates(_request_token(request), create=False)
    managed = _runtime_for(coordinates)
    with managed.lock:
        if not _status_for_managed(managed).get("authentication_verified"):
            raise CodexAccountSessionError("codex_auth_login_required")
        rows, app_server_failure_code = _model_rows(managed)
    return {
        "schema_version": "easyicu.codex-user-model-catalog/1",
        "provider": "codex",
        "catalog_authority": _SUBSCRIPTION_MODEL_CATALOG_AUTHORITY,
        "catalog_status": (
            "documented_fallback"
            if app_server_failure_code
            else "app_server_verified"
        ),
        **(
            {"catalog_failure_reason_code": app_server_failure_code}
            if app_server_failure_code
            else {}
        ),
        "models": rows,
        "secrets_returned": False,
    }


def validated_model_for_request(request: Request, model: str) -> str:
    """Compile a browser selection against the signed-in account catalog."""

    coordinates = _coordinates(_request_token(request), create=False)
    managed = _runtime_for(coordinates)
    with managed.lock:
        if not _status_for_managed(managed).get("authentication_verified"):
            raise CodexAccountSessionError("codex_auth_login_required")
        return _validated_model(managed, model)


def status_for_coordinates(coordinates: CodexSessionCoordinates) -> dict[str, Any]:
    """Internal status variant used after the response cookie is created."""

    managed = _runtime_for(coordinates)
    with managed.lock:
        return _status_for_managed(managed)


def cancel_login(request: Request) -> dict[str, Any]:
    coordinates = _coordinates(_request_token(request), create=False)
    managed = _runtime_for(coordinates)
    failure: CodexAccountSessionError | None = None
    with managed.lock:
        if managed.login_id:
            try:
                managed.runtime.request(
                    "account/login/cancel",
                    {"loginId": managed.login_id},
                    timeout=15.0,
                )
            except CodexAppServerError as exc:
                failure = CodexAccountSessionError(exc.code)
        managed.login_id = ""
        managed.login_flow = ""
        managed.auth_url = ""
        managed.verification_url = ""
        managed.user_code = ""
    # Codex keeps the loopback callback listener alive after cancellation.
    # Closing only this browser-owned child releases the fixed callback port.
    _discard_runtime(coordinates.binding_sha256, managed)
    if failure is not None:
        raise failure
    return {"ok": True, "auth": _base_status(session_present=True)}


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
        managed.login_flow = ""
        managed.auth_url = ""
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
    "binding_for_request",
    "cancel_login",
    "environment_for_binding",
    "environment_for_coordinates",
    "environment_for_request",
    "logout",
    "models",
    "pi_conversation_environment_for_binding",
    "shutdown_all",
    "start_login",
    "status",
    "validated_model_for_request",
]
