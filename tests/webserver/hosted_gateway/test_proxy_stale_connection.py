"""A pooled instance connection the instance just closed must not become a 503.

The instance's uvicorn closes an idle keep-alive connection after 5 s. When the
gateway reuses that connection at the same moment, httpx fails before any
response byte. One read of a run's result table met exactly this, so the
conversation showed no result summary for a finished analysis.
"""

import httpx
import pytest

from easyicu.hosted_gateway import proxy as proxy_module
from tests.webserver.hosted_gateway.test_proxy import instances  # noqa: F401  (fixture)


def _flaky_send(monkeypatch, *, failures, error=httpx.RemoteProtocolError):
    real_send = httpx.AsyncClient.send
    attempts = []

    async def send(self, request, **kwargs):
        attempts.append(request.method)
        if len(attempts) <= failures:
            raise error("Server disconnected without sending a response.", request=request)
        return await real_send(self, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return attempts


@pytest.mark.parametrize("error", [httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError])
def test_a_get_meeting_a_closed_pooled_connection_is_retried_once(client, login, instances, monkeypatch, error):  # noqa: F811
    login(client)
    attempts = _flaky_send(monkeypatch, failures=1, error=error)
    reply = client.get("/runs/r1/artifacts/result_tables.json?expected_sha256=ab")
    assert reply.status_code == 200
    assert reply.json()["raw_path"] == "/runs/r1/artifacts/result_tables.json"
    assert reply.json()["query"] == "expected_sha256=ab"
    assert attempts == ["GET", "GET"]


def test_a_get_that_fails_twice_is_the_generic_503(client, login, instances, monkeypatch):  # noqa: F811
    login(client)
    attempts = _flaky_send(monkeypatch, failures=2)
    reply = client.get("/api/data")
    assert reply.status_code == 503
    assert "你的 EasyICU 暂时不可用，请联系管理员。" in reply.text
    assert attempts == ["GET", "GET"]


def test_a_non_idempotent_request_is_never_resent(client, login, instances, monkeypatch):  # noqa: F811
    login(client)
    attempts = _flaky_send(monkeypatch, failures=1)
    reply = client.post("/api/action", content=b"once", headers={"origin": "https://icu.example:8443"})
    assert reply.status_code == 503
    assert attempts == ["POST"]


def test_other_transport_errors_are_not_retried(client, login, instances, monkeypatch):  # noqa: F811
    login(client)
    attempts = _flaky_send(monkeypatch, failures=1, error=httpx.ConnectTimeout)
    assert client.get("/api/data").status_code == 503
    assert attempts == ["GET"]


def test_pooled_connections_expire_before_the_instance_closes_them():
    # uvicorn's default timeout_keep_alive is 5 s; expiring sooner avoids the race.
    assert proxy_module.KEEPALIVE_EXPIRY_SECONDS < 5


def _stalling_send(monkeypatch, *, stalls):
    import anyio

    real_send = httpx.AsyncClient.send
    attempts = []

    async def send(self, request, **kwargs):
        attempts.append(request.method)
        if len(attempts) <= stalls:
            await anyio.sleep(30)  # The instance never starts its response.
        return await real_send(self, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(proxy_module, "RESPONSE_START_DEADLINE_SECONDS", 0.2)
    return attempts


def test_a_get_whose_response_never_starts_is_resent_once(client, login, instances, monkeypatch, caplog):  # noqa: F811
    import logging

    caplog.set_level(logging.WARNING, logger="easyicu.hosted_gateway.proxy")
    login(client)
    attempts = _stalling_send(monkeypatch, stalls=1)
    reply = client.get("/js/app.js?v=private-query")
    assert reply.status_code == 200 and reply.json()["raw_path"] == "/js/app.js"
    assert attempts == ["GET", "GET"]
    notes = [record.getMessage() for record in caplog.records if record.name == "easyicu.hosted_gateway.proxy"]
    assert notes == ["instance forward response_start_deadline (attempt 1): GET /js/app.js"]
    assert "private-query" not in " ".join(notes)


def test_a_get_that_stalls_twice_ends_as_the_503_page_not_a_hang(client, login, instances, monkeypatch):  # noqa: F811
    login(client)
    attempts = _stalling_send(monkeypatch, stalls=2)
    reply = client.get("/api/settings")
    assert reply.status_code == 503 and attempts == ["GET", "GET"]


def test_the_deadline_covers_the_response_start_only_and_allows_long_polls():
    # /api/attention long-polls ~20 s before answering; SSE starts at once.
    assert proxy_module.RESPONSE_START_DEADLINE_SECONDS >= 60
