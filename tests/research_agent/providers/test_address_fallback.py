"""A blackholed address family must not stall outbound literature requests.

Regression for the pilot1 plan-generation hang (2026-11-16): IPv6 to NCBI and
OpenAlex timed out silently on the server, so each urlopen waited its full
15–20 s before IPv4 was tried.
"""

import http.server
import socket
import threading
import time
import urllib.request

import pytest

from easyicu.utils import address_fallback as af

BLACKHOLE = ("2001:db8::1", 0)  # Documentation prefix; never actually dialled.


class _FakeSocket(socket.socket):
    """A real socket, except that connecting to BLACKHOLE waits out its timeout."""

    timeouts: list = []

    def connect(self, address):
        if address[0] == BLACKHOLE[0]:
            _FakeSocket.timeouts.append(self.gettimeout())
            time.sleep(self.gettimeout())
            raise TimeoutError("timed out")
        _FakeSocket.timeouts.append(self.gettimeout())
        return super().connect(address)


@pytest.fixture
def local_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


@pytest.fixture
def blackholed_v6_first(monkeypatch, local_server):
    port = local_server

    def getaddrinfo(host, p, *args, **kwargs):
        assert host == "lit.example"
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (BLACKHOLE[0], p, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", p)),
        ]

    monkeypatch.setattr(af.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(af.socket, "socket", _FakeSocket)
    monkeypatch.setattr(af, "CONNECT_ATTEMPT_SECONDS", 0.2)
    monkeypatch.setattr(af, "_preferred_family", {})
    _FakeSocket.timeouts = []
    return port


def test_blackholed_first_address_costs_one_bounded_attempt(blackholed_v6_first):
    started = time.monotonic()
    sock = af.create_connection(("lit.example", blackholed_v6_first), timeout=4.0)
    elapsed = time.monotonic() - started
    try:
        assert sock.family == socket.AF_INET
        assert sock.gettimeout() == 4.0  # Reads keep the caller's timeout.
    finally:
        sock.close()
    assert _FakeSocket.timeouts == [0.2, 4.0]
    assert elapsed < 2.0


def test_the_working_family_is_tried_first_next_time(blackholed_v6_first):
    af.create_connection(("lit.example", blackholed_v6_first), timeout=4.0).close()
    _FakeSocket.timeouts = []
    af.create_connection(("lit.example", blackholed_v6_first), timeout=4.0).close()
    assert _FakeSocket.timeouts == [0.2]  # IPv4 first; no blackhole wait.


def test_a_single_address_keeps_the_whole_timeout(monkeypatch, local_server):
    monkeypatch.setattr(af.socket, "socket", _FakeSocket)
    monkeypatch.setattr(af, "_preferred_family", {})
    _FakeSocket.timeouts = []
    af.create_connection(("127.0.0.1", local_server), timeout=4.0).close()
    assert _FakeSocket.timeouts == [4.0]


def test_all_addresses_failing_raises_the_last_error(monkeypatch):
    monkeypatch.setattr(af.socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (BLACKHOLE[0], 1, 0, 0)),
    ])
    monkeypatch.setattr(af.socket, "socket", _FakeSocket)
    with pytest.raises(TimeoutError):
        af.create_connection(("lit.example", 1), timeout=0.1)


def test_installed_urlopen_reaches_the_working_address_quickly(monkeypatch, blackholed_v6_first):
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})  # Dial directly, not via a host proxy.
    monkeypatch.setattr(urllib.request, "_opener", None)
    monkeypatch.setattr(af, "_installed", False)
    af.install()
    started = time.monotonic()
    with urllib.request.urlopen(f"http://lit.example:{blackholed_v6_first}/", timeout=4.0) as resp:
        assert resp.read() == b"ok"
    assert time.monotonic() - started < 2.0


def test_literature_clients_install_the_connector(monkeypatch):
    from easyicu.research_agent.bibliographic_metadata import OpenAlexLiteratureClient
    from easyicu.research_agent.literature import PubMedLiteratureClient, TavilyLiteratureClient

    calls = []
    import easyicu.research_agent.bibliographic_metadata as bib
    import easyicu.research_agent.literature as lit

    monkeypatch.setattr(bib, "install_address_fallback", lambda: calls.append("openalex"))
    monkeypatch.setattr(lit, "install_address_fallback", lambda: calls.append("lit"))
    PubMedLiteratureClient()
    TavilyLiteratureClient(api_key="x")
    OpenAlexLiteratureClient()
    assert calls == ["lit", "lit", "openalex"]
