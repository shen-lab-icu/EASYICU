"""Byte-stream forwarding to the authenticated account's Unix socket only."""

from __future__ import annotations

import logging

import anyio
import httpx
from starlette.responses import StreamingResponse

from .auth import CSRF_COOKIE, SESSION_COOKIE
from .pages import unavailable

# The instance's uvicorn closes an idle keep-alive connection after 5 s (its
# default timeout_keep_alive). Expiring pooled connections sooner avoids
# reusing one the instance is closing; a request that still meets a closed
# connection before any response byte is retried once when it is idempotent.
KEEPALIVE_EXPIRY_SECONDS = 2.0
# A GET/HEAD whose response has not started within this deadline is treated as
# a stalled connection: cancelling the wait discards that connection, and the
# request is sent once more on a fresh one. Streams and long polls already
# start their response (SSE headers at once, /api/attention within ~20 s);
# only the wait for the response start is bounded, never the body.
RESPONSE_START_DEADLINE_SECONDS = 90.0
PROXY_LOG = logging.getLogger("easyicu.hosted_gateway.proxy")
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD"})
STALE_CONNECTION_ERRORS = (httpx.RemoteProtocolError, httpx.ReadError,
                           httpx.WriteError, httpx.ConnectError)

HOP_HEADERS = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
               b"te", b"trailer", b"transfer-encoding", b"upgrade"}


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
                limits=httpx.Limits(keepalive_expiry=KEEPALIVE_EXPIRY_SECONDS),
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
        idempotent = request.method.upper() in IDEMPOTENT_METHODS
        # A retried request must resend the same bytes, so an idempotent body is read once.
        body = await request.body() if idempotent else None
        for attempt in (1, 2):
            # A direct Request avoids client cookie-jar/header merging across requests.
            outbound = httpx.Request(request.method, url, headers=headers,
                                     content=body if idempotent else request.stream())
            try:
                if idempotent:
                    with anyio.fail_after(RESPONSE_START_DEADLINE_SECONDS):
                        upstream = await self.clients[socket].send(outbound, stream=True)
                else:
                    upstream = await self.clients[socket].send(outbound, stream=True)
            except TimeoutError:
                self._note(request, "response_start_deadline", attempt)
                if attempt == 1:
                    continue
                return unavailable()
            except STALE_CONNECTION_ERRORS as exc:
                if idempotent and attempt == 1:
                    self._note(request, f"stale_connection:{type(exc).__name__}", attempt)
                    continue
                return unavailable()
            except httpx.HTTPError:
                return unavailable()
            return UpstreamResponse(upstream)
        return unavailable()

    @staticmethod
    def _note(request, event: str, attempt: int) -> None:
        # The same redacted fields as the access record: no query, body or cookie.
        PROXY_LOG.warning("instance forward %s (attempt %d): %s %s",
                          event, attempt, request.method, request.url.path)
