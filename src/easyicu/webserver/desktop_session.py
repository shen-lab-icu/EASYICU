"""Desktop-only authentication for the loopback WebApp boundary.

Binding FastAPI to loopback prevents remote network clients from reaching the
filesystem APIs, but it does not identify which local process opened the UI.
The desktop shell therefore supplies one random token per launch. A one-time
bootstrap URL exchanges that token for an HttpOnly cookie; API probes may use
the same token in a private request header.

The middleware is installed only when ``EASYICU_DESKTOP_SESSION_TOKEN`` is
set. Ordinary browser launches keep their existing behavior.
"""

from __future__ import annotations

import os
import secrets
from urllib.parse import parse_qsl, urlencode

from fastapi import FastAPI
from starlette.datastructures import MutableHeaders, QueryParams
from starlette.responses import JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

DESKTOP_SESSION_ENV = "EASYICU_DESKTOP_SESSION_TOKEN"
DESKTOP_HEADER = "x-easyicu-desktop-token"
DESKTOP_COOKIE = "easyicu_desktop_session"
DESKTOP_BOOTSTRAP_QUERY = "desktop_token"


class DesktopSessionMiddleware:
    """Require the desktop shell's launch-scoped token for every request."""

    def __init__(self, app: ASGIApp, *, token: str) -> None:
        selected = str(token or "").strip()
        if not selected:
            raise ValueError("Desktop session token must not be empty")
        self.app = app
        self.token = selected

    def _matches(self, candidate: str | None) -> bool:
        return bool(candidate) and secrets.compare_digest(str(candidate), self.token)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = MutableHeaders(scope=scope)
        if self._matches(headers.get(DESKTOP_HEADER)):
            await self.app(scope, receive, send)
            return

        cookies = _parse_cookie_header(headers.get("cookie", ""))
        if self._matches(cookies.get(DESKTOP_COOKIE)):
            await self.app(scope, receive, send)
            return

        query_string = scope.get("query_string", b"").decode("utf-8")
        bootstrap_token = QueryParams(query_string).get(DESKTOP_BOOTSTRAP_QUERY)
        if (
            scope.get("method") == "GET"
            and scope.get("path") in {"/", "/index.html"}
            and self._matches(bootstrap_token)
        ):
            clean_query = urlencode(
                [
                    (key, value)
                    for key, value in parse_qsl(query_string, keep_blank_values=True)
                    if key != DESKTOP_BOOTSTRAP_QUERY
                ]
            )
            location = str(scope.get("path") or "/")
            if clean_query:
                location = f"{location}?{clean_query}"
            response = RedirectResponse(location, status_code=303)
            response.set_cookie(
                DESKTOP_COOKIE,
                self.token,
                httponly=True,
                samesite="strict",
                secure=False,
                path="/",
            )
            await response(scope, receive, send)
            return

        response = JSONResponse(
            status_code=403,
            content={"detail": "EasyICU desktop session authentication failed."},
        )
        await response(scope, receive, send)


def install_desktop_session(app: FastAPI) -> bool:
    """Install the middleware when a desktop shell launch token is present."""

    token = str(os.environ.get(DESKTOP_SESSION_ENV) or "").strip()
    if not token:
        return False
    app.add_middleware(DesktopSessionMiddleware, token=token)
    return True


def _parse_cookie_header(raw: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for item in str(raw or "").split(";"):
        name, separator, value = item.strip().partition("=")
        if separator and name:
            cookies[name] = value
    return cookies
