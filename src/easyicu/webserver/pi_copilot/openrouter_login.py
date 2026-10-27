"""OpenRouter account login (OAuth PKCE) for the API connection form.

OpenRouter's PKCE flow hands a third-party app a fresh API key after the
researcher signs in on openrouter.ai. Here the key stays on this host: the
browser holds only an opaque HttpOnly cookie naming a pending login, and the
connection routes resolve that login to the key for exactly one destination,
OpenRouter's own API. The ordinary verify-and-save path then stores it like a
typed credential, so verification, catalog and receipt rules do not fork.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import urlencode

from easyicu.webserver.pi_copilot.contracts import PiCopilotError

OPENROUTER_PROVIDER = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
AUTHORIZE_URL = "https://openrouter.ai/auth"
KEY_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
CALLBACK_PATH = "/api/copilot/pi/provider-config/openrouter/callback"
COOKIE_NAME = "easyicu_openrouter_login"
COOKIE_PATH = "/api/copilot/pi/provider-config"
CREDENTIAL_SOURCE = "openrouter_login"
LOGIN_TTL_SECONDS = 15 * 60
_MAX_PENDING = 16
_MAX_EXCHANGE_BYTES = 64 * 1024
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{32,128}")
_CODE_RE = re.compile(r"[A-Za-z0-9._~-]{1,512}")
_KEY_RE = re.compile(r"[A-Za-z0-9._~:-]{16,512}")

# (code, code_verifier) -> (HTTP status, decoded JSON body)
Exchange = Callable[[str, str], tuple[int, Any]]


@dataclass
class _Login:
    verifier: str
    started: float
    key: str = ""


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _default_exchange(code: str, verifier: str) -> tuple[int, Any]:
    import requests

    try:
        response = requests.post(
            KEY_EXCHANGE_URL,
            json={"code": code, "code_verifier": verifier, "code_challenge_method": "S256"},
            headers={"Accept": "application/json"},
            timeout=15,
            allow_redirects=False,
            stream=True,
        )
        try:
            raw = response.raw.read(_MAX_EXCHANGE_BYTES + 1, decode_content=True)
        finally:
            response.close()
    except requests.RequestException as exc:
        raise PiCopilotError(
            "openrouter_login_unreachable",
            "EasyICU could not reach OpenRouter to finish the sign-in.",
        ) from exc
    if len(raw) > _MAX_EXCHANGE_BYTES:
        return response.status_code, {}
    try:
        return response.status_code, json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return response.status_code, {}


class OpenRouterLoginStore:
    """Pending sign-ins, then their keys, in process memory only."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._logins: Dict[str, _Login] = {}

    def _prune(self) -> None:
        now = self._clock()
        for token in [t for t, row in self._logins.items() if now - row.started > LOGIN_TTL_SECONDS]:
            del self._logins[token]

    def _live(self, token: str) -> Optional[_Login]:
        if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
            return None
        self._prune()
        return self._logins.get(token)

    def start(self, callback_url: str) -> tuple[str, str]:
        """Return (cookie token, OpenRouter authorization URL)."""

        token = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        with self._lock:
            self._prune()
            while len(self._logins) >= _MAX_PENDING:
                del self._logins[min(self._logins, key=lambda t: self._logins[t].started)]
            self._logins[token] = _Login(verifier=verifier, started=self._clock())
        query = urlencode({
            "callback_url": callback_url,
            "code_challenge": _challenge(verifier),
            "code_challenge_method": "S256",
        })
        return token, f"{AUTHORIZE_URL}?{query}"

    def exchange_authorization_code(self, token: str, code: str, *, exchange: Optional[Exchange] = None) -> None:
        """Trade the returned code for a key, once, for this browser's login."""

        with self._lock:
            login = self._live(token)
            if login is None or login.key:
                raise PiCopilotError(
                    "openrouter_login_not_started",
                    "Start the OpenRouter sign-in from EasyICU again.",
                )
            verifier = login.verifier
        if not isinstance(code, str) or not _CODE_RE.fullmatch(code):
            raise PiCopilotError("openrouter_login_denied", "OpenRouter did not complete the sign-in.")
        status, payload = (exchange or _default_exchange)(code, verifier)
        key = payload.get("key") if isinstance(payload, Mapping) else None
        if not 200 <= int(status) < 300 or not isinstance(key, str) or not _KEY_RE.fullmatch(key):
            with self._lock:
                self._logins.pop(token, None)
            raise PiCopilotError(
                "openrouter_login_exchange_failed",
                "OpenRouter did not issue a key for this sign-in.",
                details={"status_code": int(status)},
            )
        with self._lock:
            login = self._live(token)
            if login is None:
                raise PiCopilotError(
                    "openrouter_login_not_started",
                    "Start the OpenRouter sign-in from EasyICU again.",
                )
            login.key = key

    def resolve_api_key(
        self,
        token: str,
        *,
        credential_source: str,
        provider: str,
        base_url: str,
        api_key: str,
    ) -> str:
        """The typed key, or this browser's OpenRouter key for OpenRouter only."""

        if credential_source != CREDENTIAL_SOURCE:
            return api_key
        if (
            api_key.strip()
            or provider.strip() != OPENROUTER_PROVIDER
            or base_url.strip().rstrip("/") != OPENROUTER_BASE_URL
        ):
            raise PiCopilotError(
                "openrouter_login_wrong_service",
                "An OpenRouter sign-in can only connect OpenRouter.",
                status_code=422,
            )
        with self._lock:
            login = self._live(token)
            key = login.key if login else ""
        if not key:
            raise PiCopilotError(
                "openrouter_login_not_signed_in",
                "Sign in with OpenRouter again; the earlier sign-in expired.",
                status_code=409,
            )
        return key

    def finish(self, token: str) -> None:
        """Forget the key once the connection store holds it."""

        with self._lock:
            self._logins.pop(token, None)


_STORE = OpenRouterLoginStore()


def login_store() -> OpenRouterLoginStore:
    return _STORE


__all__ = [
    "CALLBACK_PATH",
    "COOKIE_NAME",
    "COOKIE_PATH",
    "CREDENTIAL_SOURCE",
    "LOGIN_TTL_SECONDS",
    "OPENROUTER_BASE_URL",
    "OpenRouterLoginStore",
    "login_store",
]
