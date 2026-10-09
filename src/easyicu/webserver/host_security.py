"""Host-header protection with correct bracketed IPv6 handling.

This module owns the *host access policy* — which Host headers are accepted,
whether a forwarding proxy is trusted, and when a Unix-socket peer counts as
local. Both the middleware that enforces
it and the Settings → Privacy panel that reports it read it from here, so the
UI cannot claim a guarantee the running server is not applying.
"""

from __future__ import annotations

import os
import stat
from typing import Any, Dict, Mapping
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
    """True when an operator has vouched that the proxy authenticates.

    D-P2-7: ``EASYICU_WEB_TRUST_PROXY=1`` must only be set when an
    authenticating reverse proxy in front of EasyICU verifies every request
    itself (identity, TLS, and host). With the flag on, a loopback peer is no
    longer proof of a local user — any remote client the proxy forwards
    reaches the filesystem and job APIs, including the extension-management
    routes whose only remaining barrier is the ``expected_sha256``
    exact-edit confirmation. Never enable this flag for a proxy that only
    forwards.
    """

    return _env_flag("EASYICU_WEB_TRUST_PROXY")


#: The one Unix socket this server listens on, set by
#: ``python -m easyicu.webserver run --uds PATH`` (or by an operator who runs
#: ``uvicorn --uds PATH`` with the same absolute PATH).
UNIX_SOCKET_ENV = "EASYICU_WEB_UNIX_SOCKET"


class UnixSocketDirectoryError(RuntimeError):
    """The socket's directory would let another account connect."""


def configured_unix_socket() -> str | None:
    raw = os.getenv(UNIX_SOCKET_ENV, "").strip()
    return os.path.abspath(os.path.expanduser(raw)) if raw else None


def unix_socket_directory_problem(socket_path: str) -> str | None:
    """Why the socket's directory would admit another account, or None.

    uvicorn makes the socket itself world-writable (0666), so its directory is
    what keeps other accounts on a shared host out: it must be a real
    directory (not a symbolic link) that this server's account owns, with no
    group or other access.
    """

    directory = os.path.dirname(os.path.abspath(socket_path))
    try:
        info = os.lstat(directory)
    except OSError as exc:
        return f"its directory {directory} cannot be read ({exc.strerror or exc})"
    if not stat.S_ISDIR(info.st_mode):
        return f"{directory} is not a directory (a symbolic link is not accepted)"
    if info.st_uid != os.geteuid():
        return f"{directory} belongs to uid {info.st_uid}, not to this server's uid {os.geteuid()}"
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        return f"{directory} has mode {mode:04o}, which lets group or other users in (chmod 700 it)"
    return None


def require_private_unix_socket_directory(socket_path: str) -> None:
    problem = unix_socket_directory_problem(socket_path)
    if problem:
        raise UnixSocketDirectoryError(
            f"EasyICU WebApp will not listen on the Unix socket {socket_path}: {problem}."
        )


def is_local_unix_socket_peer(scope: Mapping[str, Any]) -> bool:
    """A connection on the private Unix socket this server was configured for.

    uvicorn reports a Unix listener as ``server == (path, None)`` and gives
    its peers no address. Both must hold, the path must be the configured
    socket, and its directory must still be private: a directory opened up
    after start makes every request on it fail closed.
    """

    configured = configured_unix_socket()
    server = scope.get("server")
    if not configured or scope.get("client") is not None:
        return False
    if not isinstance(server, (tuple, list)) or len(server) != 2 or server[1] is not None:
        return False
    if not isinstance(server[0], str) or os.path.abspath(server[0]) != configured:
        return False
    return unix_socket_directory_problem(configured) is None


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
