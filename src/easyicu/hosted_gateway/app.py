"""Small ASGI authentication boundary; the instance still owns every API route."""

from __future__ import annotations

import asyncio
import hmac
import html
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route

from .auth import (
    CSRF_COOKIE, LOGIN_WINDOW, SESSION_COOKIE, SESSION_SECONDS, LoginLimiter, Signer, verify_password,
)
from .config import GatewayConfig
from .proxy import InstanceProxy, unavailable
from .storage import read_secret, read_users

ACCESS_LOG = logging.getLogger("easyicu.hosted_gateway.access")
CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; form-action 'self'; frame-ancestors 'none'"
LOGIN_ERROR = "用户名或密码错误，请重试。"
STATE_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError)


class RequestBoundary:
    """Validate the public endpoint and emit one redacted access record."""

    def __init__(self, app, public_origin: str):
        self.app = app
        self.host = urlsplit(public_origin).netloc

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started, status = time.monotonic(), 500

        async def capture(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            hosts = [value.decode("latin-1").lower() for key, value in scope["headers"] if key == b"host"]
            if hosts != [self.host] or scope["scheme"] != "https":
                await Response(status_code=400)(scope, receive, capture)
            else:
                await self.app(scope, receive, capture)
        finally:
            ACCESS_LOG.info(json.dumps({
                "time": datetime.now(timezone.utc).isoformat(),
                "user": scope.get("gateway.user", "-"), "method": scope["method"],
                "path": scope["path"], "status": status,
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
            }, ensure_ascii=True))


def create_app(config: GatewayConfig, *, clock=time.time) -> Starlette:
    read_users(config.users_file)  # Fail at startup if private state is invalid.
    signer = Signer(read_secret(config.secret_file), clock)
    limiter = LoginLimiter(clock)
    auth_lock = asyncio.Lock()
    proxy = InstanceProxy()
    template = (Path(__file__).parent / "templates/login.html").read_text(encoding="utf-8")

    def login_page(*, error: str = "", status: int = 200, csrf: str = ""):
        csrf = csrf or signer.issue("csrf", {"nonce": secrets.token_urlsafe(32)}, LOGIN_WINDOW)
        response = HTMLResponse(template.replace("{{csrf}}", html.escape(csrf, quote=True)).replace(
            "{{error}}", html.escape(error, quote=True)), status_code=status,
            headers={"cache-control": "no-store", "content-security-policy": CSP})
        response.set_cookie(CSRF_COOKIE, csrf, max_age=LOGIN_WINDOW, secure=True,
                            httponly=True, samesite="lax", path="/")
        return response

    def same_origin(request: Request) -> bool:
        return request.headers.getlist("origin") == [config.public_origin]

    async def login(request: Request):
        if request.method == "GET":
            return login_page()
        if not same_origin(request):
            return Response(status_code=403, headers={"cache-control": "no-store"})
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/x-www-form-urlencoded":
            return Response(status_code=400)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 16384:
                return Response(status_code=413)
        try:
            fields = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=8)
            if any(len(values) != 1 for values in fields.values()):
                raise ValueError("duplicate form fields")
        except (ValueError, UnicodeError):
            return Response(status_code=400)
        csrf = fields.get("csrf", [""])[0]
        cookie = request.cookies.get(CSRF_COOKIE, "")
        if not (csrf and hmac.compare_digest(csrf.encode(), cookie.encode()) and signer.verify("csrf", csrf)):
            return login_page(error="登录页面已过期，请重试。", status=403)
        name = fields.get("username", [""])[0]
        password = fields.get("password", [""])[0]
        ip = request.client.host if request.client else "unknown"
        async with auth_lock:
            if limiter.blocked(name, ip):
                return login_page(error="登录尝试过多，请稍后再试。", status=429, csrf=csrf)
            try:
                users = await run_in_threadpool(read_users, config.users_file)
            except STATE_ERRORS:
                return unavailable()
            user = users.get(name)
            valid = False
            if len(password.encode("utf-8")) <= 4096:
                valid = await run_in_threadpool(verify_password, password, user["password"] if user else None)
            if not valid:
                limiter.failure(name, ip)
                return login_page(error=LOGIN_ERROR, status=401, csrf=csrf)
        request.scope["gateway.user"] = name
        response = RedirectResponse("/", status_code=303, headers={"cache-control": "no-store"})
        response.set_cookie(SESSION_COOKIE, signer.issue("session", {
            "user": name, "version": user["version"],
        }, SESSION_SECONDS), max_age=SESSION_SECONDS, secure=True, httponly=True, samesite="lax", path="/")
        response.delete_cookie(CSRF_COOKIE, secure=True, httponly=True, samesite="lax")
        return response

    async def logout(request: Request):
        if not same_origin(request):
            return Response(status_code=403, headers={"cache-control": "no-store"})
        response = RedirectResponse("/login", status_code=303, headers={"cache-control": "no-store"})
        response.delete_cookie(SESSION_COOKIE, secure=True, httponly=True, samesite="lax")
        return response

    class AuthenticatedInstance:
        async def __call__(self, scope, receive, send):
            request = Request(scope, receive)
            claims = signer.verify("session", request.cookies.get(SESSION_COOKIE, ""))
            user = None
            if claims and isinstance(claims.get("user"), str):
                try:
                    users = await run_in_threadpool(read_users, config.users_file)
                except STATE_ERRORS:
                    await unavailable()(scope, receive, send)
                    return
                candidate = users.get(claims["user"])
                if candidate and candidate["version"] == claims.get("version"):
                    user = candidate
            if user is None:
                headers = {"cache-control": "no-store"}
                if scope["path"] == "/api" or scope["path"].startswith("/api/"):
                    response = JSONResponse({"detail": {"error": "login_required"}}, status_code=401,
                                            headers=headers)
                else:
                    response = RedirectResponse("/login", status_code=302, headers=headers)
            else:
                scope["gateway.user"] = claims["user"]
                response = await proxy.forward(request, user["socket"])
            await response(scope, receive, send)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await proxy.close()

    return Starlette(routes=[Route("/login", login, methods=["GET", "POST"]),
                             Route("/logout", logout, methods=["POST"]),
                             Mount("/", app=AuthenticatedInstance())], lifespan=lifespan,
                     middleware=[Middleware(RequestBoundary, public_origin=config.public_origin)])
