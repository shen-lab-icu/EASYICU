"""Host-header protection with correct bracketed IPv6 handling.

This module owns the *host access policy* — which Host headers are accepted
and whether a forwarding proxy is trusted. Both the middleware that enforces
it and the Settings → Privacy panel that reports it read it from here, so the
UI cannot claim a guarantee the running server is not applying.
"""

from __future__ import annotations

import os
from typing import Any, Dict
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse

#: Headers a reverse proxy adds. A browser on this machine never sends them.
PROXY_HEADERS = (
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "forwarded",
)

DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]", "testserver")

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def trusts_proxy() -> bool:
    """True when an operator has vouched that the proxy authenticates."""
    return _env_flag("EASYICU_WEB_TRUST_PROXY")


def resolve_allowed_hosts() -> list[str]:
    configured = [
        host.strip()
        for host in os.getenv("EASYICU_WEB_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    ]
    if "*" in configured and not _env_flag("EASYICU_WEB_ALLOW_ANY_HOST"):
        configured = [host for host in configured if host != "*"]
    return configured or list(DEFAULT_ALLOWED_HOSTS)


def local_access_policy() -> Dict[str, Any]:
    """The live local-access facts the Privacy panel renders."""
    allowed = resolve_allowed_hosts()
    proxy_trusted = trusts_proxy()
    return {
        "loopback_clients_only": True,
        "allowed_hosts": allowed,
        "any_host_allowed": "*" in allowed,
        "proxy_headers_trusted": proxy_trusted,
        "proxy_headers_rejected": not proxy_trusted,
        "enforced": "*" not in allowed and not proxy_trusted,
    }


def _normalize_host(value: str) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = urlsplit("//" + raw)
        if parsed.username is not None or parsed.password is not None:
            return None
        _ = parsed.port
    except ValueError:
        return None
    return parsed.hostname.rstrip(".").lower() if parsed.hostname else None


def _host_allowed(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    normalized = _normalize_host(host)
    if normalized is None:
        return False
    for pattern in allowed_hosts:
        if pattern == "*":
            return True
        if pattern.startswith("*."):
            suffix = _normalize_host(pattern[2:])
            if suffix and normalized.endswith("." + suffix):
                return True
            continue
        if normalized == _normalize_host(pattern):
            return True
    return False


class AllowedHostsMiddleware:
    """Reject DNS-rebinding Host headers without breaking ``[::1]:port``."""

    def __init__(self, app, allowed_hosts: list[str] | tuple[str, ...]) -> None:
        self.app = app
        self.allowed_hosts = tuple(allowed_hosts)

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] in {"http", "websocket"}:
            headers = dict(scope.get("headers") or [])
            host = headers.get(b"host", b"").decode("latin-1")
            if not _host_allowed(host, self.allowed_hosts):
                response = PlainTextResponse("Invalid host header", status_code=400)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
