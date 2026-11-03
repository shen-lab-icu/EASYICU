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
