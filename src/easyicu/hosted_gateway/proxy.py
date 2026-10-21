"""Byte-stream forwarding to the authenticated account's Unix socket only."""

from __future__ import annotations

import anyio
import httpx
from starlette.responses import HTMLResponse, StreamingResponse

from .auth import CSRF_COOKIE, SESSION_COOKIE

HOP_HEADERS = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
               b"te", b"trailer", b"transfer-encoding", b"upgrade"}


def unavailable() -> HTMLResponse:
    return HTMLResponse("你的 EasyICU 暂时不可用，请联系管理员。", status_code=503,
                        headers={"cache-control": "no-store"})


def end_to_end(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    nominated = {part.strip().lower() for key, value in headers if key.lower() == b"connection"
                 for part in value.split(b",")}
    removed = HOP_HEADERS | (nominated - {b"host", b"origin"})
    return [(key, value) for key, value in headers if key.lower() not in removed]


class UpstreamResponse(StreamingResponse):
    def __init__(self, upstream: httpx.Response):
        super().__init__(upstream.aiter_raw(), status_code=upstream.status_code)
        self.upstream = upstream
        self.raw_headers = [
            (key, value) for key, value in end_to_end(upstream.headers.raw)
            if not (key.lower() == b"set-cookie" and value.split(b"=", 1)[0].strip().decode(
                "latin-1") in {SESSION_COOKIE, CSRF_COOKIE})
        ]

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Cleanup also runs when the browser disconnects or cancels an SSE stream.
            with anyio.CancelScope(shield=True):
                await self.upstream.aclose()


class InstanceProxy:
    def __init__(self):
        self.clients: dict[str, httpx.AsyncClient] = {}

    async def close(self) -> None:
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()

    async def forward(self, request, socket: str):
        if socket not in self.clients:
            self.clients[socket] = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(uds=socket), trust_env=False,
                timeout=httpx.Timeout(10, read=None), follow_redirects=False,
            )
        headers = [(key, value) for key, value in end_to_end(request.headers.raw)
                   if key.lower() not in {b"cookie", b"forwarded", b"x-real-ip"}
                   and not key.lower().startswith(b"x-forwarded-")]
        headers += [(b"x-forwarded-for", (request.client.host if request.client else "unknown").encode()),
                    (b"x-forwarded-proto", b"https"),
                    (b"x-forwarded-host", request.headers["host"].encode("latin-1"))]
        raw_path = request.scope.get("raw_path", request.url.path.encode())
        query = request.scope.get("query_string", b"")
        url = httpx.URL("http://instance").copy_with(raw_path=raw_path + (b"?" + query if query else b""))
        # A direct Request avoids client cookie-jar/header merging across requests.
        outbound = httpx.Request(request.method, url, headers=headers, content=request.stream())
        try:
            upstream = await self.clients[socket].send(outbound, stream=True)
        except httpx.HTTPError:
            return unavailable()
        return UpstreamResponse(upstream)
