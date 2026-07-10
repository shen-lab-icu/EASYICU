"""Host-header protection with correct bracketed IPv6 handling."""

from __future__ import annotations

from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse


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
