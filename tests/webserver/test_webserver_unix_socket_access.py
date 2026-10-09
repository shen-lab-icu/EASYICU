"""Who counts as local when the WebApp listens on a Unix socket.

On a shared server a loopback port admits every account on the host, so the
WebApp can instead listen on a Unix socket in a directory only its own
account may enter (``run --uds``; a browser reaches it through
``ssh -L 18770:<socket>``). uvicorn gives such peers no address, which the
loopback check refused outright. The socket counts as local only when it is
the declared one and its directory is private; the TCP, proxy and same-origin
checks stay as they were.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import uvicorn

from easyicu.webserver import __main__ as web_cli
from easyicu.webserver import host_security, settings
import easyicu.webserver.app as app_module
from easyicu.webserver.app import app

# What a browser sends through ``ssh -L 18770:<socket>``.
FORWARDED = {"Host": "127.0.0.1:18770"}


@pytest.fixture
def private_dir():
    # A socket path must stay short (104 bytes on macOS), so not tmp_path.
    directory = Path(tempfile.mkdtemp(prefix="eu-uds-", dir="/tmp"))
    directory.chmod(0o700)
    yield directory
    directory.chmod(0o700)
    shutil.rmtree(directory, ignore_errors=True)


def _serve(socket_path: Path, *, lifespan: str):
    server = uvicorn.Server(
        uvicorn.Config(app, uds=str(socket_path), lifespan=lifespan, log_level="critical")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def served(private_dir, monkeypatch):
    """The real app on a real socket; the reset route writes nothing."""

    socket_path = private_dir / "web.sock"
    monkeypatch.setenv(host_security.UNIX_SOCKET_ENV, str(socket_path))
    monkeypatch.setattr(settings, "reset_settings", lambda: {})
    monkeypatch.setattr(settings, "about", lambda: {})
    server, thread = _serve(socket_path, lifespan="off")
    deadline = time.monotonic() + 20
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    yield socket_path
    server.should_exit = True
    thread.join(timeout=10)


def _request(socket_path: Path, method: str, target: str, headers: dict[str, str]):
    lines = [f"{method} {target} HTTP/1.1", "Connection: close", "Content-Length: 0"]
    lines += [f"{name}: {value}" for name, value in headers.items()]
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(10)
        connection.connect(str(socket_path))
        connection.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        raw = b""
        while chunk := connection.recv(65536):
            raw += chunk
    head, _, body = raw.partition(b"\r\n\r\n")
    return int(head.split(b" ", 2)[1]), body


def test_a_request_on_the_private_unix_socket_is_local(served):
    assert _request(served, "GET", "/api/health", FORWARDED)[0] == 200
    same_origin = {**FORWARDED, "Origin": "http://127.0.0.1:18770"}
    assert _request(served, "POST", "/api/settings/reset", same_origin)[0] == 200


def test_the_unix_socket_still_refuses_cross_origin_writes_and_proxies(served):
    status, body = _request(
        served, "POST", "/api/settings/reset", {**FORWARDED, "Origin": "http://evil.example"}
    )
    assert (status, b"Cross-origin" in body) == (403, True)
    status, body = _request(
        served, "GET", "/api/health", {**FORWARDED, "X-Forwarded-For": "203.0.113.9"}
    )
    assert (status, b"forwarded by a proxy" in body) == (403, True)


def test_only_the_declared_socket_in_a_private_directory_counts(served, monkeypatch):
    monkeypatch.delenv(host_security.UNIX_SOCKET_ENV)
    assert _request(served, "GET", "/api/health", FORWARDED)[0] == 403
    monkeypatch.setenv(host_security.UNIX_SOCKET_ENV, str(served.parent / "other.sock"))
    assert _request(served, "GET", "/api/health", FORWARDED)[0] == 403

    # A directory opened after start fails every request closed.
    monkeypatch.setenv(host_security.UNIX_SOCKET_ENV, str(served))
    served.parent.chmod(0o750)
    assert _request(served, "GET", "/api/health", FORWARDED)[0] == 403
    served.parent.chmod(0o700)
    assert _request(served, "GET", "/api/health", FORWARDED)[0] == 200


def test_only_a_unix_listener_without_a_peer_address_counts(private_dir, monkeypatch):
    socket_path = str(private_dir / "web.sock")
    monkeypatch.setenv(host_security.UNIX_SOCKET_ENV, socket_path)

    assert host_security.is_local_unix_socket_peer({"server": (socket_path, None), "client": None})
    for scope in (
        {"server": (socket_path, 8765), "client": None},  # a TCP listener
        {"server": (socket_path, None), "client": ("127.0.0.1", 50000)},  # a peer address
        {"server": None, "client": None},
    ):
        assert not host_security.is_local_unix_socket_peer(scope), scope


def test_declaring_a_socket_does_not_widen_tcp(private_dir, monkeypatch):
    monkeypatch.setenv(host_security.UNIX_SOCKET_ENV, str(private_dir / "web.sock"))

    remote = TestClient(app, client=("192.168.1.20", 50000)).get("/api/health")
    proxied = TestClient(app, client=("127.0.0.1", 50000)).get(
        "/api/health", headers={"X-Forwarded-For": "203.0.113.9"}
    )

    assert (remote.status_code, proxied.status_code) == (403, 403)


def _never(*args, **kwargs):  # noqa: ANN002, ANN003
    raise AssertionError("the server must not start")


@pytest.mark.parametrize("mode", [0o750, 0o705, 0o701, 0o755])
def test_a_socket_in_a_shared_directory_refuses_to_start(private_dir, monkeypatch, capsys, mode):
    socket_path = private_dir / "web.sock"
    private_dir.chmod(mode)
    monkeypatch.setattr(web_cli.subprocess, "run", _never)
    monkeypatch.setattr(web_cli.subprocess, "Popen", _never)

    assert web_cli.main(["run", "--uds", str(socket_path)]) == 2
    assert f"mode {mode:04o}" in capsys.readouterr().err

    # Started without the launcher, the app refuses before uvicorn binds.
    monkeypatch.setenv(host_security.UNIX_SOCKET_ENV, str(socket_path))
    with pytest.raises(host_security.UnixSocketDirectoryError, match="chmod 700"):
        app_module._refuse_a_shared_unix_socket()
    assert app.router.on_startup[0] is app_module._refuse_a_shared_unix_socket
    server, thread = _serve(socket_path, lifespan="on")
    thread.join(timeout=20)
    assert (thread.is_alive(), server.started, socket_path.exists()) == (False, False, False)


def test_a_linked_or_foreign_directory_refuses_to_start(private_dir, monkeypatch):
    link = private_dir.with_name(private_dir.name + "-link")
    link.symlink_to(private_dir, target_is_directory=True)
    try:
        problem = host_security.unix_socket_directory_problem(str(link / "web.sock"))
    finally:
        link.unlink()
    assert "symbolic link" in problem

    uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: uid + 1)
    problem = host_security.unix_socket_directory_problem(str(private_dir / "web.sock"))
    assert f"belongs to uid {uid}" in problem


def test_run_uds_starts_uvicorn_on_the_socket_it_declares(private_dir, monkeypatch):
    seen = {}

    def fake_run(cmd, env):  # noqa: ANN001
        seen.update(cmd=cmd, env=env)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(web_cli.subprocess, "run", fake_run)
    socket_path = str(private_dir / "web.sock")

    assert web_cli.main(["run", "--uds", socket_path]) == 0
    assert seen["cmd"][-2:] == ["--uds", socket_path]
    assert "--host" not in seen["cmd"]
    assert seen["env"][host_security.UNIX_SOCKET_ENV] == socket_path
