"""Only observed transport time can release an unknown completion hold.

Synthetic prompts, in-memory transports and a fake clock; no provider calls.
"""

from datetime import datetime, timedelta, timezone
import json
import math
import sys
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority import provider_hard_stop as owner
from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.providers.transport_retry import TransportRetrySchedule


def _limits(**overrides):
    return owner.ProviderHardStopLimits(**{
        "max_provider_attempts_per_run": 3, "max_provider_attempts_per_batch": 6,
        "max_total_tokens_per_run": 1_000_000, "max_total_tokens_per_batch": 2_000_000,
        "max_estimated_cost_usd_per_batch": 10.0, "max_wall_clock_seconds_per_task": 60.0,
        "input_cost_usd_per_million_tokens": 1.0, "output_cost_usd_per_million_tokens": 2.0,
        **overrides,
    })


def _ledger(tmp_path, *, limits=None):
    return owner.ProviderHardStopLedger(
        path=tmp_path / "ledger.json", task_ids=("E1",),
        limits=limits or _limits(), batch_id="test-batch",
    )


def _message(text="Describe a synthetic experiment."):
    return [LLMMessage(role="user", content=text)]


def _failure(status=500, *, text="Synthetic transport failure"):
    failure = RuntimeError(text)
    if status is not None:
        failure.status_code = status
    return failure


def _ok():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=1, total_tokens=5),
    )


def _client(monkeypatch, outcomes, *, schedule, max_retries=None):
    from easyicu.research_agent.providers.factory import build_provider_client
    from easyicu.research_agent.providers.llm import OpenAIClient

    pending = iter(outcomes)

    def create(**kwargs):
        outcome = next(pending)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    transport = SimpleNamespace(create=create)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=transport)),
    ))
    client = build_provider_client(
        provider="openai", model="gpt-5.6-luna", request_timeout=1,
        title="Synthetic transport accounting", client_cls=OpenAIClient,
        base_url_override="http://127.0.0.1:8787/v1",
        environment={"OPENAI_API_KEY": "synthetic-key", "EASYICU_ALLOW_EXTERNAL_LLM": "1"},
        max_retries=max_retries if max_retries is not None else schedule.max_retries,
        retryable_http_status_codes=(500, 502, 503, 504),
        allow_environment_overrides=False, retry_schedule=schedule,
    )
    client._retry_random = lambda: 0.5
    sleeps = []
    monkeypatch.setattr(owner.time, "sleep", sleeps.append)
    return client, transport, sleeps


@pytest.fixture
def clock(monkeypatch):
    seconds = [0.0]
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(owner.time, "monotonic", lambda: seconds[0])
    monkeypatch.setattr(
        owner, "_utc_now", lambda: (epoch + timedelta(seconds=seconds[0])).isoformat()
    )
    return seconds


def _scope(task, *, max_tokens=4000, messages=None):
    return owner.provider_hard_stop_call_scope(
        task=task, role="planner", model="synthetic-model",
        messages=messages or _message("Describe a small synthetic experiment."),
        max_tokens=max_tokens,
    )


def test_failure_is_durable_before_retry_wait_and_does_not_charge_the_wait(
    tmp_path, monkeypatch, clock, ra,
):
    ledger = _ledger(tmp_path, limits=_limits(max_wall_clock_seconds_per_task=600))
    inner, transport, _ = _client(
        monkeypatch, [_failure(), _failure(), _ok()],
        schedule=TransportRetrySchedule((20.0, 180.0), window_seconds=300.0),
    )
    create = transport.create

    def request(**kwargs):
        clock[0] += 0.125
        return create(**kwargs)

    monkeypatch.setattr(transport, "create", request)
    slept = []

    def sleep(seconds):
        call = owner.load_provider_hard_stop_ledger(ledger.path)["tasks"][0]["calls"][-1]
        assert call["state"] == "failed_usage_unknown"
        assert call["transport_elapsed_seconds"] == 0.125
        assert call["transport_finished_at"] == owner._utc_now()
        assert call["finished_at"] == owner._utc_now()
        slept.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(owner.time, "sleep", sleep)
    client = HardStopClient(inner, role="planner", task=ledger.start_task("E1"))
    assert client.complete(_message()) == "OK"
    assert slept == [20.0, 180.0]
    calls = ledger.snapshot()["tasks"][0]["calls"]
    assert len(calls) == 3
    for call in calls[:2]:
        assert call["transport_elapsed_seconds"] == 0.125
        assert call["accounted_tokens"] == (
            call["prompt_token_reservation"]
            + math.ceil(0.125 * owner.MAX_COMPLETION_TOKENS_PER_SECOND)
        )
    assert calls[2]["state"] == "completed"
    assert calls[2]["accounted_tokens"] == 5


@pytest.mark.parametrize("seconds", [0.0, 0.000015, 0.125, 60.0])
@pytest.mark.parametrize("max_tokens", [4000, 160_000])
def test_observed_failure_bounds_completion_but_keeps_prompt_and_reservations(
    tmp_path, clock, seconds, max_tokens,
):
    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with _scope(task, max_tokens=max_tokens) as active:
        owner.consume_active_provider_hard_stop_attempt()
        before = ledger.snapshot()["tasks"][0]["calls"][0]
        clock[0] += seconds
        ended_at = owner._utc_now()
        owner.record_active_provider_transport_failure("SyntheticFailure")
        # An outer wrapper sees the same failure; it must not finalize twice.
        active.fail("SyntheticFailure")
    call = owner.load_provider_hard_stop_ledger(ledger.path)["tasks"][0]["calls"][0]
    completion = min(
        before["completion_token_reservation"],
        math.ceil(seconds * owner.MAX_COMPLETION_TOKENS_PER_SECOND),
    )
    for field in (
        "prompt_token_reservation", "completion_token_reservation",
        "requested_completion_tokens", "provider_completion_token_reservation_floor",
    ):
        assert call[field] == before[field]
    assert call["state"] == "failed_usage_unknown"
    assert call["transport_finished_at"] == ended_at
    assert call["transport_elapsed_seconds"] == seconds
    assert call["failed_completion_token_bound"] == completion
    assert call["max_completion_tokens_per_second"] == 100_000
    assert call["accounted_tokens"] == call["prompt_token_reservation"] + completion
    assert call["reported_prompt_tokens"] is None
    assert call["reported_completion_tokens"] is None
    assert call["accounted_estimated_cost_usd"] == pytest.approx(
        (call["prompt_token_reservation"] * ledger.limits.input_cost_usd_per_million_tokens
         + completion * ledger.limits.output_cost_usd_per_million_tokens) / 1_000_000
    )
    summary = task.accounting_summary()
    assert summary["provider_reported"]["n_calls"] == 0
    assert summary["usage_unknown"]["n_calls"] == 1
    assert summary["conservative_upper_bound"]["total_tokens"] == call["accounted_tokens"]


@pytest.mark.parametrize("ending", ["unknown", "retry", "crash", "success_without_usage"])
def test_without_an_observed_failure_end_the_entire_reservation_remains(
    tmp_path, clock, ending,
):
    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with _scope(task) as active:
        owner.consume_active_provider_hard_stop_attempt()
        before = ledger.snapshot()["tasks"][0]["calls"][0]
        clock[0] += 0.125
        if ending == "unknown":
            active.fail("LostTransport")
        elif ending == "retry":
            owner.consume_active_provider_hard_stop_attempt()
        elif ending == "success_without_usage":
            active.complete(None)
        # A crash never invokes a transport-end hook.
    call = ledger.snapshot()["tasks"][0]["calls"][0]
    assert call["accounted_tokens"] == before["accounted_tokens"]
    assert call["accounted_estimated_cost_usd"] == before["accounted_estimated_cost_usd"]
    assert call.get("transport_finished_at") is None
    assert call.get("failed_completion_token_bound") is None


def test_nine_small_successes_and_six_fast_failures_leave_room_for_the_next_call(
    tmp_path, clock,
):
    ledger = _ledger(tmp_path, limits=_limits(
        max_provider_attempts_per_run=30, max_provider_attempts_per_batch=30,
        max_total_tokens_per_run=1_200_000, max_estimated_cost_usd_per_batch=30,
        input_cost_usd_per_million_tokens=2.5, output_cost_usd_per_million_tokens=33,
    ))
    task = ledger.start_task("E1")
    messages = _message("Explain a synthetic experiment. " * 1517)
    for index in range(15):
        with _scope(task, messages=messages) as active:
            owner.consume_active_provider_hard_stop_attempt()
            clock[0] += 0.125
            if index < 9:
                active.complete({"prompt_tokens": 8000, "completion_tokens": 1200})
            else:
                owner.record_active_provider_transport_failure("SyntheticFailure")
    totals = ledger.snapshot()["totals"]
    assert totals["provider_attempts"] == 15
    assert totals["accounted_tokens"] < 1_200_000
    assert totals["accounted_estimated_cost_usd"] < 30
    with _scope(task, messages=messages) as active:
        owner.consume_active_provider_hard_stop_attempt()
        active.complete({"prompt_tokens": 8000, "completion_tokens": 1200})


def test_legacy_ledger_without_transport_fields_loads_and_keeps_full_accounting(tmp_path):
    ledger = _ledger(tmp_path)
    with _scope(ledger.start_task("E1")) as active:
        owner.consume_active_provider_hard_stop_attempt()
        active.fail("LostProcess")
    payload = ledger.snapshot()
    call = payload["tasks"][0]["calls"][0]
    for field in (
        "transport_finished_at", "transport_elapsed_seconds",
        "failed_completion_token_bound", "max_completion_tokens_per_second",
    ):
        call.pop(field, None)
    payload["sha256"] = owner._payload_digest(payload)
    ledger.path.write_text(json.dumps(payload))
    loaded = owner.load_provider_hard_stop_ledger(ledger.path)
    resumed = owner.ProviderHardStopLedger(
        path=ledger.path, limits=ledger.limits, task_ids=("E1",),
        batch_id="test-batch", resume_existing=True,
    )
    assert loaded["schema_version"] == "easyicu.provider_hard_stop_ledger/1"
    assert resumed.snapshot()["tasks"][0]["calls"] == [call]
    assert call["accounted_tokens"] == (
        call["prompt_token_reservation"] + call["completion_token_reservation"]
    )


@pytest.mark.parametrize("kind", ["http", "opaque", "json", "vision"])
def test_terminal_transport_failures_are_timed_without_reading_error_bodies(
    tmp_path, monkeypatch, clock, ra, kind,
):
    failure = (
        json.JSONDecodeError("synthetic", "", 0) if kind == "json"
        else _failure(None if kind == "opaque" else 500, text="synthetic error")
    )
    inner, transport, sleeps = _client(
        monkeypatch, [failure], schedule=None, max_retries=0,
    )

    def request(**kwargs):
        clock[0] += 0.125
        raise failure

    monkeypatch.setattr(transport, "create", request)
    ledger = _ledger(tmp_path)
    client = HardStopClient(inner, role="planner", task=ledger.start_task("E1"))
    with pytest.raises(type(failure)) as raised:
        if kind == "vision":
            client.complete_with_images(prompt="Synthetic figure.", image_paths=[])
        else:
            client.complete(_message())
    assert raised.value is failure
    assert sleeps == []
    call = ledger.snapshot()["tasks"][0]["calls"][0]
    assert call["state"] == "failed_usage_unknown"
    assert call["transport_elapsed_seconds"] == 0.125
    assert call["failed_completion_token_bound"] == 12_500


def test_anthropic_failure_is_settled_before_backoff_and_again_at_exhaustion(
    tmp_path, monkeypatch, clock, ra,
):
    from easyicu.research_agent.providers.factory import build_provider_client

    def request(**kwargs):
        clock[0] += 0.125
        raise _failure(text="synthetic error")

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(
        Anthropic=lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=request)),
    ))
    inner = build_provider_client(
        provider="anthropic", model="synthetic-model", request_timeout=1,
        title="Synthetic transport accounting",
        environment={"ANTHROPIC_API_KEY": "synthetic-key", "EASYICU_ALLOW_EXTERNAL_LLM": "1"},
        max_retries=1, retryable_http_status_codes=(500,),
        allow_environment_overrides=False,
        retry_schedule=TransportRetrySchedule((20.0,), jitter_fraction=0),
    )
    ledger = _ledger(tmp_path)
    client = HardStopClient(inner, role="planner", task=ledger.start_task("E1"))

    def sleep(seconds):
        call = owner.load_provider_hard_stop_ledger(ledger.path)["tasks"][0]["calls"][-1]
        assert call["state"] == "failed_usage_unknown"
        assert call["transport_elapsed_seconds"] == 0.125
        clock[0] += seconds

    monkeypatch.setattr(owner.time, "sleep", sleep)
    with pytest.raises(RuntimeError):
        client.complete(_message())
    calls = ledger.snapshot()["tasks"][0]["calls"]
    assert len(calls) == 2
    assert all(call["failed_completion_token_bound"] == 12_500 for call in calls)


@pytest.mark.parametrize("elapsed", [None, -1.0, math.nan, math.inf, True])
def test_missing_or_invalid_duration_cannot_release_a_hold(tmp_path, elapsed):
    ledger = _ledger(tmp_path)
    ledger.start_task("E1")
    attempt = ledger.reserve_transport_attempt(
        task_id="E1", role="planner", model="synthetic-model", messages=_message(),
        max_tokens=4000, prior_attempt_id=None,
    )
    before = ledger.snapshot()["tasks"][0]["calls"][0]
    ledger.finish_transport_attempt(
        task_id="E1", attempt_id=attempt, usage=None, error_type="SyntheticFailure",
        transport_elapsed_seconds=elapsed,
    )
    call = ledger.snapshot()["tasks"][0]["calls"][0]
    assert call["accounted_tokens"] == before["accounted_tokens"]
    assert call["accounted_estimated_cost_usd"] == before["accounted_estimated_cost_usd"]
    assert call.get("transport_finished_at") is None


def test_observed_end_does_not_override_reported_success_or_refund_unreported_success(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_task("E1")
    for usage in ({"prompt_tokens": 100, "completion_tokens": 40}, None):
        attempt = ledger.reserve_transport_attempt(
            task_id="E1", role="planner", model="synthetic-model", messages=_message(),
            max_tokens=4000, prior_attempt_id=None,
        )
        ledger.finish_transport_attempt(
            task_id="E1", attempt_id=attempt, usage=usage, error_type=None,
            transport_elapsed_seconds=0.0,
        )
    reported, unreported = ledger.snapshot()["tasks"][0]["calls"]
    assert reported["accounted_tokens"] == 140
    assert unreported["accounted_tokens"] == (
        unreported["prompt_token_reservation"] + unreported["completion_token_reservation"]
    )


@pytest.mark.parametrize(("overrides", "code"), [
    ({"max_provider_attempts_per_run": 1}, "RUN_PROVIDER_ATTEMPT_LIMIT"),
    ({"max_total_tokens_per_run": 140_000}, "RUN_TOKEN_LIMIT"),
    ({"max_estimated_cost_usd_per_batch": 0.28}, "BATCH_COST_LIMIT"),
])
def test_a_bounded_failure_still_requires_the_full_next_reservation(
    tmp_path, clock, overrides, code,
):
    ledger = _ledger(tmp_path, limits=_limits(**overrides))
    with _scope(ledger.start_task("E1")) as active:
        owner.consume_active_provider_hard_stop_attempt()
        clock[0] += 0.125
        owner.record_active_provider_transport_failure("SyntheticFailure")
        settled = ledger.snapshot()["tasks"][0]["calls"][0]
        with pytest.raises(owner.ProviderHardStopExceeded, match=code):
            owner.consume_active_provider_hard_stop_attempt()
        active.fail("ProviderHardStopExceeded")
    assert ledger.snapshot()["tasks"][0]["calls"] == [settled]
