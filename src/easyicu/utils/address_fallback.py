"""Outbound HTTP(S) connections that do not stall on an unreachable address.

``socket.create_connection`` tries the resolver's addresses one by one and
gives each the caller's full timeout. Where one address family is silently
dropped (an IPv6 route that is advertised but blackholed, common on hospital
and server networks) every request then waits out that timeout before the
working family is tried; curl and browsers hide this with Happy Eyeballs.
A literature search issues many requests, so a 15 s stall per request turned
plan generation into a 40-minute hang.

This owner bounds each non-final connect attempt, alternates address
families (RFC 8305 section 4), and remembers per host which family last
connected so later requests start there. The final attempt keeps the
caller's whole timeout, so a slow but working network is not cut short.
Read timeouts are unchanged.

``install()`` puts the connector behind ``urllib.request.urlopen`` for this
process; callers keep calling ``urlopen`` (and tests keep stubbing it).
"""

from __future__ import annotations

import http.client
import socket
import threading
import urllib.request
from typing import Any, Optional

CONNECT_ATTEMPT_SECONDS = 3.0

_lock = threading.Lock()
_installed = False
_preferred_family: dict[tuple[str, int], int] = {}


def _ordered(host: str, port: int, infos: list[tuple]) -> list[tuple]:
    by_family: dict[int, list[tuple]] = {}
    for info in infos:
        by_family.setdefault(info[0], []).append(info)
    families = list(by_family)
    preferred = _preferred_family.get((host, port))
    if preferred in by_family:
        families.remove(preferred)
        families.insert(0, preferred)
    ordered: list[tuple] = []
    while any(by_family.values()):
        for family in families:
            if by_family[family]:
                ordered.append(by_family[family].pop(0))
    return ordered


def create_connection(address: tuple[str, int], timeout: Any = socket._GLOBAL_DEFAULT_TIMEOUT,
                      source_address: Optional[tuple[str, int]] = None, **_: Any) -> socket.socket:
    """Drop-in for ``socket.create_connection`` with bounded per-address attempts."""
    host, port = address
    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    if not infos:
        raise OSError(f"getaddrinfo returned no addresses for {host}")
    default = timeout is socket._GLOBAL_DEFAULT_TIMEOUT
    full = socket.getdefaulttimeout() if default else timeout
    candidates = _ordered(host, port, infos)
    error: Optional[OSError] = None
    for index, (family, kind, proto, _canon, sockaddr) in enumerate(candidates):
        final = index == len(candidates) - 1
        attempt = full if final or full is None else min(float(full), CONNECT_ATTEMPT_SECONDS)
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(attempt)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            sock.settimeout(full)
        except OSError as exc:
            error = exc
            sock.close()
            continue
        with _lock:
            _preferred_family[(host, port)] = family
        return sock
    assert error is not None
    raise error


class _HTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = create_connection


class _HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = create_connection


class _HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_HTTPConnection, req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_HTTPSConnection, req, context=self._context)


def install() -> None:
    """Route this process's ``urllib.request.urlopen`` through the connector (idempotent)."""
    global _installed
    with _lock:
        if _installed:
            return
        urllib.request.install_opener(urllib.request.build_opener(_HTTPHandler(), _HTTPSHandler()))
        _installed = True


__all__ = ["CONNECT_ATTEMPT_SECONDS", "create_connection", "install"]
