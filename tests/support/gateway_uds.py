"""Real Unix byte streams for fake upstream ASGI apps, without network binds.

Socket pairs let sandboxed tests exercise HTTPX/httpcore's UDS transport and
Uvicorn's HTTP protocol. Only pathname connection establishment is substituted.
The integration fixture also has an explicit named-listener mode for operators.
"""

import asyncio
import socket

import anyio
import uvicorn
from anyio.abc import ByteStream, SocketAttribute
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import ServerState


class UnixPairStream(ByteStream):
    def __init__(self, sock):
        self.sock = sock
        self.sock.setblocking(False)

    @property
    def extra_attributes(self):
        return {SocketAttribute.raw_socket: lambda: self.sock}

    async def receive(self, max_bytes=65536):
        value = await asyncio.get_running_loop().sock_recv(self.sock, max_bytes)
        if not value:
            raise anyio.EndOfStream
        return value

    async def send(self, item):
        await asyncio.get_running_loop().sock_sendall(self.sock, item)

    async def send_eof(self):
        self.sock.shutdown(socket.SHUT_WR)

    async def aclose(self):
        self.sock.close()


def install_pair_connector(monkeypatch, apps):
    transports = []

    async def connect_unix(path):
        app = apps[str(path)]  # A wrong account socket cannot silently resolve.
        client_socket, server_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        config = uvicorn.Config(app, log_config=None, access_log=False, lifespan="off")
        config.load()
        try:
            transport, _ = await asyncio.get_running_loop().create_connection(
                lambda: H11Protocol(config, ServerState(), {}), sock=server_socket)
        except BaseException:
            client_socket.close()
            server_socket.close()
            raise
        transports.append(transport)
        return UnixPairStream(client_socket)

    monkeypatch.setattr(anyio, "connect_unix", connect_unix)

    async def close():
        for transport in transports:
            transport.close()
        await asyncio.sleep(0)
    return close
