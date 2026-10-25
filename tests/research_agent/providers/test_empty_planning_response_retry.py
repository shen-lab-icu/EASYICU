"""Synthetic stalled transports: no network, real attempt and hard-stop owners."""

import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopExceeded, ProviderHardStopLedger,
)
from easyicu.research_agent.providers.clients import current_provider_call_receipt
from easyicu.research_agent.providers.factory import build_provider_client
from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.research_agent.providers.llm import OpenAIClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.providers.structured_retry import call_llm_with_structured_retry
from easyicu.research_agent.providers.transport_retry import TransportRetrySchedule
from tests.support.provider_recovery import _limits


@pytest.fixture
def transport(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    state = SimpleNamespace(rebuild_fails=False)
    monkeypatch.setattr("time.monotonic", lambda: clock.now)
    sleeps, calls, pools, outcomes = [], [], [], []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr("time.sleep", sleep)

    def sdk(**kwargs):
        if pools and state.rebuild_fails:
            raise RuntimeError("synthetic construction failure")
        pool = len(pools)
        pools.append(pool)

        def create(**kwargs):
            calls.append((pool, kwargs))
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                clock.now += float(kwargs["timeout"])
                raise outcome
            if not isinstance(outcome, str):
                return outcome
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=outcome), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=sdk))

    def build(*, limit=6, retries=2, stream=False, wall=1800):
        ledger = ProviderHardStopLedger(
            path=tmp_path / "ledger.json", task_ids=("synthetic",), batch_id="synthetic",
            limits=_limits(max_provider_attempts_per_run=limit,
                           max_provider_attempts_per_batch=limit,
                           max_wall_clock_seconds_per_task=wall,
                           max_total_tokens_per_run=20_000_000,
                           max_total_tokens_per_batch=20_000_000,
                           max_estimated_cost_usd_per_batch=100),
        )
        client = build_provider_client(
            provider="openai", model="synthetic", request_timeout=480,
            title="Synthetic empty planning response", client_cls=OpenAIClient,
            base_url_override="http://127.0.0.1:8787/v1",
            environment={"OPENAI_API_KEY": "synthetic", "EASYICU_ALLOW_EXTERNAL_LLM": "1"},
            max_retries=retries, retryable_http_status_codes=(500, 502, 503, 504),
            retry_schedule=TransportRetrySchedule((20, 180)) if retries == 2 else None,
            allow_environment_overrides=False,
            stream_enabled=stream,
        )
        return HardStopClient(client, role="planner", task=ledger.start_task("synthetic")), ledger

    return SimpleNamespace(build=build, outcomes=outcomes, calls=calls, pools=pools, sleeps=sleeps, state=state)


def ask(client, role="progressive_planner_step_materialization", events=None):
    return call_llm_with_structured_retry(
        client, [LLMMessage(role="user", content="Describe a synthetic measurement.")],
        json.loads, role=role, max_retries=0, max_tokens=32,
        progress_callback=events.append if events is not None else None,
    )


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("synthetic"), httpx.ConnectError("synthetic")])
def test_empty_stall_retries_on_a_fresh_pool_and_accounts_both_attempts(transport, failure):
    transport.outcomes[:] = [failure, '{"ready":true}']
    client, ledger = transport.build()
    events = []
    assert ask(client, events=events) == {"ready": True}
    assert events[-1].reason_code == "planning_empty_response_retry"
    assert [pool for pool, _ in transport.calls] == [0, 1]
    assert transport.calls[0][1]["timeout"] == 480
    assert transport.calls[1][1]["timeout"] <= 120
    assert len(transport.sleeps) == 1
    assert 0 < transport.sleeps[0] <= 3
    rows = ledger.snapshot()["tasks"][0]["calls"]
    assert [row["state"] for row in rows] == ["failed_usage_unknown", "completed"]
    receipt = current_provider_call_receipt()
    assert receipt.transport_attempts == 2
    assert receipt.transport_retry_reasons == ("planning_empty_response_retry",)


def test_persistent_stall_stops_after_one_retry_with_a_typed_receipt(transport):
    transport.outcomes[:] = [httpx.ReadTimeout("synthetic") for _ in range(3)]
    client, ledger = transport.build()
    with pytest.raises(httpx.ReadTimeout) as stopped:
        ask(client)
    assert len(transport.calls) == 2
    assert len(transport.sleeps) == 1
    assert len(ledger.snapshot()["tasks"][0]["calls"]) == 2
    row = stopped.value.easyicu_structured_attempt_metadata[-1]
    assert row["transport_attempts"] == 2
    assert row["reason_code"] == "planning_empty_response_retry"
    assert row["raw_chars"] == 0


def test_retry_cannot_cross_the_original_attempt_budget(transport):
    transport.outcomes[:] = [httpx.ReadTimeout("synthetic"), '{"ready":true}']
    client, ledger = transport.build(limit=1)
    with pytest.raises(ProviderHardStopExceeded) as stopped:
        ask(client)
    assert len(transport.calls) == len(ledger.snapshot()["tasks"][0]["calls"]) == 1
    assert stopped.value.easyicu_structured_attempt_metadata[-1]["transport_attempts"] == 1


@pytest.mark.parametrize("role", ["writer", "coder"])
def test_nonplanning_calls_keep_the_status_allowlist(transport, role):
    transport.outcomes[:] = [httpx.ReadTimeout("synthetic"), "{}"]
    client, _ = transport.build()
    with pytest.raises(httpx.ReadTimeout):
        ask(client, role)
    assert len(transport.calls) == 1 and not transport.sleeps


def test_retry_wait_cannot_use_the_remaining_task_time(transport):
    transport.outcomes[:] = [httpx.ReadTimeout("synthetic"), "{}"]
    client, _ = transport.build(wall=481)
    with pytest.raises(httpx.ReadTimeout) as stopped:
        ask(client)
    assert stopped.value.easyicu_transport_retry_exhausted == "transport_retry_wall_clock_exhausted"
    assert len(transport.calls) == len(transport.pools) == 1 and not transport.sleeps


def test_failed_pool_replacement_never_reuses_the_stalled_connection(transport):
    transport.outcomes[:] = [httpx.ReadTimeout("synthetic"), "{}"]
    client, _ = transport.build()
    transport.state.rebuild_fails = True
    with pytest.raises(httpx.ReadTimeout):
        ask(client)
    assert len(transport.calls) == 1 and not transport.sleeps


def test_planning_retry_permission_does_not_escape_the_structured_call(transport):
    transport.outcomes[:] = ["{}", httpx.ReadTimeout("synthetic"), "{}"]
    client, _ = transport.build()
    assert ask(client) == {}
    with pytest.raises(httpx.ReadTimeout):
        client.complete([LLMMessage(role="user", content="An ordinary synthetic call.")])
    assert len(transport.calls) == 2 and not transport.sleeps


@pytest.mark.parametrize("partial", [False, True])
def test_stream_chunks_prevent_retry_and_streams_are_closed(transport, partial):
    closed = []

    class Stream:
        def __iter__(self):
            if partial:
                yield SimpleNamespace(choices=[])
            raise httpx.ReadTimeout("synthetic")

        def close(self):
            closed.append(True)

    transport.outcomes[:] = [Stream(), Stream()]
    client, _ = transport.build(stream=True)
    with pytest.raises(httpx.ReadTimeout):
        ask(client)
    assert len(transport.calls) == len(closed) == (1 if partial else 2)


@pytest.mark.parametrize("cause", [httpx.WriteTimeout("synthetic"), httpx.PoolTimeout("synthetic")])
def test_sdk_wrapping_does_not_broaden_the_read_timeout_rule(transport, cause):
    failure = type("APITimeoutError", (Exception,), {"__module__": "openai"})("synthetic")
    failure.__cause__ = cause
    transport.outcomes[:] = [failure, "{}"]
    client, _ = transport.build()
    with pytest.raises(type(failure)):
        ask(client)
    assert len(transport.calls) == 1 and not transport.sleeps


@pytest.mark.parametrize("kind", ["partial", "status", "text", "disabled"])
def test_partial_responses_untyped_errors_and_disabled_retries_do_not_resend(transport, kind):
    failure = httpx.ReadTimeout("synthetic")
    if kind == "partial":
        failure.easyicu_response_started = True
    elif kind == "status":
        failure.status_code = 429
    elif kind == "text":
        failure = RuntimeError("ReadTimeout: synthetic")
    transport.outcomes[:] = [failure, "{}"]
    client, _ = transport.build(retries=0 if kind == "disabled" else 2)
    with pytest.raises(type(failure)):
        ask(client)
    assert len(transport.calls) == 1 and not transport.sleeps
