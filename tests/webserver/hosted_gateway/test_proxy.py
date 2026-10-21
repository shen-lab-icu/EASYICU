import asyncio
import gzip
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tests.support.gateway_uds import install_pair_connector


@pytest.fixture
def instances(gateway, client, monkeypatch):
    """Two ASGI apps over Unix streams; optionally use named UDS listeners."""
    _, config, _, users = gateway
    servers, threads, seen = [], [], []
    gate = threading.Event()

    def instance(name):
        async def endpoint(request: Request):
            seen.append(name)
            if request.url.path == "/encoded":
                return Response(gzip.compress(b"synthetic download"), status_code=206, headers={
                    "content-encoding": "gzip", "content-disposition": 'attachment; filename="demo.txt"',
                    "set-cookie": "__Host-easyicu_session=upstream-forgery; Path=/; Secure",
                })
            if request.url.path == "/redirect":
                return Response(status_code=307, headers={"location": "http://unreachable.invalid/"})
            if request.url.path in {"/sse", "/download"}:
                async def chunks():
                    yield b"data: first\n\n" if request.url.path == "/sse" else b"x" * 262144
                    await asyncio.to_thread(gate.wait, 5)
                    yield b"data: second\n\n" if request.url.path == "/sse" else b"y" * 262144
                return StreamingResponse(chunks(), media_type=("text/event-stream"
                                         if request.url.path == "/sse" else "application/octet-stream"))
            body = await request.body()
            if request.method == "POST" and request.headers.get("origin") != config.public_origin:
                return JSONResponse({"blocked_by_instance": True}, status_code=403)
            return JSONResponse({"instance": name, "method": request.method,
                                 "raw_path": request.scope["raw_path"].decode(),
                                 "query": request.scope["query_string"].decode(),
                                 "body": body.decode(), "headers": dict(request.headers)})
        return Starlette(routes=[Route("/{path:path}", endpoint,
                                       methods=["GET", "POST", "PATCH", "HEAD"])])

    with tempfile.TemporaryDirectory(prefix="b3-", dir="/tmp") as temp:
        close_pairs = None
        try:
            apps = {}
            for name in ("alice", "bob"):
                socket = str(Path(temp) / f"{name}.sock")
                users[name]["socket"] = socket
                apps[socket] = instance(name)
                if os.environ.get("EASYICU_GATEWAY_NAMED_UDS_TEST") != "1":
                    continue
                server = uvicorn.Server(uvicorn.Config(apps[socket], uds=socket, log_config=None,
                                                       access_log=False, lifespan="off"))
                thread = threading.Thread(target=server.run, daemon=True)
                servers.append(server)
                threads.append(thread)
                thread.start()
                deadline = time.monotonic() + 5
                while not server.started and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert server.started
            if not servers:
                close_pairs = install_pair_connector(monkeypatch, apps)
            config.users_file.write_text(json.dumps(users))
            yield seen, gate
        finally:
            gate.set()
            if close_pairs:
                client.portal.call(close_pairs)
            for server in servers:
                server.should_exit = True
            for thread in threads:
                thread.join(timeout=6)


def test_each_user_routes_only_to_own_socket(client, login, instances):
    seen, _ = instances
    assert login(client).status_code == 303
    reply = client.get("/bob?instance=bob", headers={"x-forwarded-host": "bob"})
    assert reply.json()["instance"] == "alice"
    assert seen == ["alice"]
    assert login(client, username="bob").status_code == 303
    assert client.get("/alice").json()["instance"] == "bob"
    assert seen == ["alice", "bob"]


def test_forward_method_raw_path_query_body_headers_but_no_cookie(client, login, instances):
    login(client)
    response = client.patch("/files/a%2Fb?q=private%20query&q=two", content=b"synthetic body",
                            headers={"origin": "https://icu.example:8443", "x-demo": "keep",
                                     "x-forwarded-for": "forged", "x-forwarded-proto": "http",
                                     "x-forwarded-host": "forged", "forwarded": "forged",
                                     "x-real-ip": "forged", "connection": "x-hop",
                                     "x-hop": "remove"})
    result = response.json()
    assert result["method"] == "PATCH" and result["body"] == "synthetic body"
    assert result["raw_path"] == "/files/a%2Fb"
    assert result["query"] == "q=private%20query&q=two"
    headers = result["headers"]
    assert "cookie" not in headers and "forwarded" not in headers and "x-real-ip" not in headers
    assert "x-hop" not in headers
    assert headers["host"] == "icu.example:8443"
    assert headers["origin"] == "https://icu.example:8443" and headers["x-demo"] == "keep"
    assert headers["x-forwarded-for"] == "testclient"
    assert headers["x-forwarded-proto"] == "https"
    assert headers["x-forwarded-host"] == "icu.example:8443"


def test_cross_site_origin_is_not_rewritten_and_instance_refuses(client, login, instances):
    login(client)
    reply = client.post("/api/action", headers={"origin": "https://attacker.example"})
    assert reply.status_code == 403 and reply.json() == {"blocked_by_instance": True}


def test_down_socket_is_generic_chinese_503(client, login):
    login(client)
    reply = client.get("/api/data")
    assert reply.status_code == 503
    assert "你的 EasyICU 暂时不可用，请联系管理员。" in reply.text
    assert "sock" not in reply.text and "Traceback" not in reply.text
    assert reply.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("path,first,second", [
    ("/sse", b"data: first\n\n", b"data: second\n\n"),
    ("/download", b"x" * 262144, b"y" * 262144),
], ids=["sse", "download"])
def test_response_arrives_before_upstream_finishes(gateway, client, login, instances, path, first, second):
    from easyicu.hosted_gateway.auth import SESSION_COOKIE

    login(client)
    cookie = client.cookies.get(SESSION_COOKIE)
    _, gate = instances
    received = bytearray()

    async def exercise():
        arrived = asyncio.Event()
        disconnected = asyncio.Event()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                assert message["status"] == 200
            if message["type"] == "http.response.body":
                received.extend(message.get("body", b""))
                if len(received) >= len(first):
                    arrived.set()

        scope = {"type": "http", "http_version": "1.1", "asgi": {"version": "3.0"},
                 "scheme": "https", "method": "GET", "path": path, "raw_path": path.encode(),
                 "query_string": b"", "root_path": "", "server": ("icu.example", 8443),
                 "client": ("192.0.2.1", 42), "headers": [
                     (b"host", b"icu.example:8443"),
                     (b"cookie", f"{SESSION_COOKIE}={cookie}".encode())]}
        task = asyncio.create_task(gateway[0](scope, receive, send))
        try:
            await asyncio.wait_for(arrived.wait(), timeout=2)
            assert not task.done() and not gate.is_set()
            assert bytes(received) == first
            gate.set()
            await asyncio.wait_for(task, timeout=3)
        finally:
            gate.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    client.portal.call(exercise)
    assert bytes(received) == first + second


def test_access_log_has_only_safe_fields(client, login, instances, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="easyicu.hosted_gateway.access")
    login(client)
    client.post("/echo?secret=private-query", content=b"private-body",
                headers={"origin": "https://icu.example:8443"})
    records = [json.loads(record.message) for record in caplog.records
               if record.name == "easyicu.hosted_gateway.access"]
    last = records[-1]
    assert set(last) == {"time", "user", "method", "path", "status", "duration_ms"}
    assert last["user"] == "alice" and last["method"] == "POST"
    assert last["path"] == "/echo" and last["status"] == 200 and last["duration_ms"] >= 0
    output = json.dumps(records)
    assert all(value not in output for value in ("private-query", "private-body", "synthetic-pass", "Cookie"))


def test_compressed_download_headers_and_gateway_cookie_are_preserved(client, login, instances):
    from easyicu.hosted_gateway.auth import SESSION_COOKIE

    login(client)
    original = client.cookies.get(SESSION_COOKIE)
    reply = client.get("/encoded")
    assert reply.status_code == 206 and reply.content == b"synthetic download"
    assert reply.headers["content-encoding"] == "gzip"
    assert reply.headers["content-disposition"] == 'attachment; filename="demo.txt"'
    assert "set-cookie" not in reply.headers
    assert client.cookies.get(SESSION_COOKIE) == original


def test_redirect_is_returned_without_following_it(client, login, instances):
    login(client)
    reply = client.get("/redirect")
    assert reply.status_code == 307 and reply.headers["location"] == "http://unreachable.invalid/"


def test_disconnect_closes_upstream(client, login, gateway, instances, monkeypatch):
    import httpx
    from easyicu.hosted_gateway.auth import SESSION_COOKIE

    login(client)
    closed = []
    aclose = httpx.Response.aclose

    async def observe(response):
        if response.request.url.host == "instance":
            closed.append(response)
        await aclose(response)

    monkeypatch.setattr(httpx.Response, "aclose", observe)

    async def exercise():
        received = asyncio.Event()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b""}
            await received.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                received.set()

        scope = {"type": "http", "http_version": "1.1", "asgi": {"version": "3.0"},
                 "scheme": "https", "method": "GET", "path": "/sse", "raw_path": b"/sse",
                 "query_string": b"", "root_path": "", "server": ("icu.example", 8443),
                 "client": ("192.0.2.1", 42), "headers": [
                     (b"host", b"icu.example:8443"),
                     (b"cookie", f"{SESSION_COOKIE}={client.cookies.get(SESSION_COOKIE)}".encode())]}
        await asyncio.wait_for(gateway[0](scope, receive, send), timeout=2)
        assert closed and all(response.is_closed for response in closed)
    client.portal.call(exercise)
