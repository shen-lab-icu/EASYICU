"""A slow streamed upstream does not stall the relay's event loop.

11-03 external review: ``_bounded_streaming_response`` iterated the blocking
``requests`` stream inside an async generator, so a chunk that took 0.2 s
held up a 0.02 s heartbeat (and health checks) on the same worker for the
whole 0.2 s. Synthetic upstream only; nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import time

from easyicu import hosted_llm_server as hosted


class _SlowUpstream:
    status_code = 200
    closed = False

    def iter_content(self, chunk_size=1024):
        for index in range(3):
            time.sleep(0.2)
            yield f"data: {index}\n\n".encode()

    def close(self):
        self.closed = True


def test_a_heartbeat_runs_on_time_while_the_stream_waits(monkeypatch):
    upstream = _SlowUpstream()
    monkeypatch.setattr(hosted, "_post_upstream", lambda *_args, **_kwargs: upstream)

    async def scenario():
        response = await hosted._bounded_streaming_response(object(), {"stream": True})
        delays = []

        async def heartbeat():
            for _ in range(10):
                started = time.perf_counter()
                await asyncio.sleep(0.02)
                delays.append(time.perf_counter() - started)

        chunks = []

        async def consume():
            async for chunk in response.body_iterator:
                chunks.append(chunk)

        beating = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)  # the heartbeat is waiting before the stream starts
        await consume()
        await beating
        return chunks, delays

    chunks, delays = asyncio.run(scenario())
    assert len(chunks) == 3
    assert max(delays) < 0.15, delays
    assert upstream.closed


class _DelayedErrorUpstream:
    """A 401 whose body trickles in (or fails) only when it is read."""

    status_code = 401
    closed = False

    def __init__(self, delay: float, fail: bool = False):
        self.delay = delay
        self.fail = fail

    def json(self):
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("body read failed")
        return {"error": {"message": "bad key"}}

    def close(self):
        self.closed = True


def _slow_error_scenario(monkeypatch, upstream):
    import httpx

    monkeypatch.setattr(hosted, "OPENROUTER_API_KEY", "test-upstream-key")
    monkeypatch.setattr(hosted, "HOSTED_SERVER_TOKEN", "shared-test-token")
    monkeypatch.setattr(hosted, "HOSTED_ALLOWED_MODELS", set())
    monkeypatch.setattr(hosted, "_UPSTREAM_SEMAPHORE", None)
    monkeypatch.setattr(hosted, "_post_upstream", lambda *_a, **_k: upstream)
    headers = {"Authorization": "Bearer shared-test-token"}

    async def scenario():
        transport = httpx.ASGITransport(app=hosted.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            slow = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": next(iter(hosted.MODEL_ALIASES)),
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            )
            # Measure from before the wait: a blocked loop would also delay the
            # wake-up of this sleep, not just the health request.
            started = time.perf_counter()
            await asyncio.sleep(0.2)  # the slow error body is now being read
            health = await client.get("/health")
            latency = time.perf_counter() - started
            result = await slow
        return health, latency, result

    health, latency, result = asyncio.run(scenario())
    assert health.status_code == 200
    return latency, result


def test_a_slow_upstream_error_body_does_not_stall_other_requests(monkeypatch):
    upstream = _DelayedErrorUpstream(delay=1.0)
    latency, result = _slow_error_scenario(monkeypatch, upstream)
    assert latency < 0.6, latency
    assert result.status_code == 401
    assert result.json() == {"error": {"message": "bad key"}}
    assert upstream.closed
    assert hosted._UPSTREAM_SEMAPHORE._value == hosted.HOSTED_MAX_CONCURRENT_UPSTREAM


def test_a_failing_error_body_read_releases_slot_and_response(monkeypatch):
    upstream = _DelayedErrorUpstream(delay=0.3, fail=True)
    latency, result = _slow_error_scenario(monkeypatch, upstream)
    assert latency < 0.28, latency
    assert result.status_code == 500
    assert upstream.closed
    assert hosted._UPSTREAM_SEMAPHORE._value == hosted.HOSTED_MAX_CONCURRENT_UPSTREAM
