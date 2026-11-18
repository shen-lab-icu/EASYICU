"""Synthetic accounts only; no existing instance or credential is opened."""

import asyncio
import gzip
import json
import os
import re
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
from starlette.testclient import TestClient

from tests.support.gateway_uds import install_pair_connector


@pytest.fixture
def gateway(tmp_path):
    from easyicu.hosted_gateway.app import create_app
    from easyicu.hosted_gateway.auth import hash_password
    from easyicu.hosted_gateway.config import GatewayConfig

    users = {
        name: {"socket": str(tmp_path / name / "web.sock"),
               "password": hash_password("synthetic-pass"), "version": name + "-v1"}
        for name in ("alice", "bob")
    }
    users_file = tmp_path / "users.json"
    users_file.write_text(json.dumps(users))
    users_file.chmod(0o600)
    secret_file = tmp_path / "secret"
    secret_file.write_bytes(b"synthetic-secret-for-tests-only-32")
    secret_file.chmod(0o600)
    config = GatewayConfig(public_origin="https://icu.example:8443", users_file=users_file,
                           secret_file=secret_file, cert_file=tmp_path / "cert.pem",
                           key_file=tmp_path / "key.pem")
    clock = [1_800_000_000.0]
    app = create_app(config, clock=lambda: clock[0])
    return app, config, clock, users


@pytest.fixture
def client(gateway):
    with TestClient(gateway[0], base_url="https://icu.example:8443",
                    follow_redirects=False) as client:
        yield client


@pytest.fixture
def login():
    def perform(client, username="alice", password="synthetic-pass", **kwargs):
        page = client.get("/login")
        csrf = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
        return client.post("/login", data={"username": username, "password": password,
                                          "csrf": csrf},
                           headers={"origin": "https://icu.example:8443"}, **kwargs)
    return perform


@pytest.fixture
def admin_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"public_origin": "https://icu.example:8443",
                                 "users_file": "users.json", "secret_file": "secret",
                                 "cert_file": "cert.pem", "key_file": "key.pem"}))
    users_root = tmp_path / "instances"
    for name in ("a", "b"):
        instance = users_root / name
        (instance / "run").mkdir(parents=True, mode=0o700)
        instance.chmod(0o700)
        env = instance / "instance.env"
        env.write_text('EASYICU_PUBLIC_ORIGIN="https://icu.example:8443"\n'
                       'EASYICU_HOSTED="1"\nEASYICU_WEB_TRUST_PROXY="1"\n')
        env.chmod(0o600)
    return config, users_root


@pytest.fixture
def run_admin(admin_config, monkeypatch):
    from easyicu.hosted_gateway.cli import main

    monkeypatch.setattr("easyicu.hosted_gateway.cli.getpass.getpass",
                        lambda _: "synthetic-pass")

    def run(*args):
        return main([*args, "--config", str(admin_config[0])])
    return run


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[3]


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

    with tempfile.TemporaryDirectory(prefix="b3-") as temp:
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
