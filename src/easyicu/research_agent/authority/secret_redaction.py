"""Shared secret classification for provenance and bounded debug diagnostics."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit


REDACTED = "[REDACTED]"

_SENSITIVE_KEY_SUFFIXES = frozenset(
    {
        "api_key",
        "apikey",
        "key",
        "token",
        "secret",
        "password",
        "passwd",
        "credential",
        "authorization",
        "proxy_authorization",
        "cookie",
        "cookies",
        "set_cookie",
        "session",
        "session_id",
        "dsn",
        "connection_string",
        "database_url",
        "database_uri",
        "private_key",
        "client_secret",
        "access_key",
        "access_key_id",
        "refresh_token",
    }
)
_AUTH_VALUE_RE = re.compile(r"(?i)^(?:bearer|basic|token|apikey|api-key)\s+[^\s]{8,}$")
_TOKEN_VALUE_RE = re.compile(
    r"(?:"
    r"\bsk-[A-Za-z0-9_-]{16,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    r")"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?"
    r"-----END [^-]*PRIVATE KEY-----",
    flags=re.IGNORECASE | re.DOTALL,
)
_URL_CREDENTIAL_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^/\s@]+)@")
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization|proxy[-_]authorization)\s*[:=]\s*"
    r"(?:(?:bearer|basic|token)\s+)?[A-Za-z0-9._~+/\-=]{8,}"
)
_COOKIE_HEADER_RE = re.compile(r"(?i)\b(cookie|set[-_]cookie)\s*[:=]\s*[^\r\n]+")
_NAMED_SECRET_RE = re.compile(
    r"(?i)\b("
    r"api[_-]?key|password|passwd|token|secret|session[_-]?id|dsn|"
    r"connection[_-]?string|database[_-]?(?:url|uri)"
    r")\s*[:=]\s*[^\s,;}\]\"']{4,}"
)


def _normalise_key(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).casefold()).strip("_")


def is_sensitive_key(name: Any) -> bool:
    """Classify exact credential keys and underscore-delimited suffixes."""

    normalized = _normalise_key(name)
    return any(
        normalized == suffix or normalized.endswith(f"_{suffix}")
        for suffix in _SENSITIVE_KEY_SUFFIXES
    )


def _has_url_credentials(value: str) -> bool:
    if "://" not in value:
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return bool(_URL_CREDENTIAL_RE.search(value))
    return bool(parsed.netloc and (parsed.username is not None or parsed.password))


def string_contains_secret(value: str) -> bool:
    """Detect common credential-bearing values even under an unfamiliar key."""

    text = str(value or "").strip()
    if not text:
        return False
    return bool(
        _has_url_credentials(text)
        or _AUTH_VALUE_RE.fullmatch(text)
        or _TOKEN_VALUE_RE.search(text)
        or _PRIVATE_KEY_RE.search(text)
        or _AUTH_HEADER_RE.search(text)
        or _COOKIE_HEADER_RE.search(text)
        or _NAMED_SECRET_RE.search(text)
    )


def redact_text_secrets(value: Any) -> str:
    """Remove recognized credentials from otherwise useful diagnostic text."""

    text = value if isinstance(value, str) else str(value)
    text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)
    text = _URL_CREDENTIAL_RE.sub(r"\1[REDACTED]@", text)
    text = _AUTH_HEADER_RE.sub(lambda match: f"{match.group(1)}: {REDACTED}", text)
    text = _COOKIE_HEADER_RE.sub(lambda match: f"{match.group(1)}: {REDACTED}", text)
    text = _NAMED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}={REDACTED}",
        text,
    )
    text = _AUTH_VALUE_RE.sub(REDACTED, text)
    return _TOKEN_VALUE_RE.sub(REDACTED, text)


def redact_debug_value(value: Any, *, key: str = "") -> Any:
    """Recursively redact a JSON-like debug envelope."""

    if is_sensitive_key(key) and value is not None:
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_debug_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_debug_value(item, key=key) for item in value]
    if isinstance(value, str):
        return redact_text_secrets(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def debug_capture_enabled(value: str | None) -> bool:
    """Require an explicit truthy value; ``EASYICU_LLM_DEBUG=0`` stays off."""

    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


__all__ = [
    "REDACTED",
    "debug_capture_enabled",
    "is_sensitive_key",
    "redact_debug_value",
    "redact_text_secrets",
    "string_contains_secret",
]
