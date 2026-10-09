"""A transport retry waits the schedule its caller states, and says why it stopped.

A client given a ``TransportRetrySchedule`` waits one drawn delay before each
retry its status allowlist allows; a provider's Retry-After lengthens a wait
and never shortens it.  It does not retry when even its shortest wait would
end past the schedule's window (a drawn wait is cut to end at the window, so
jitter moves when a retry begins, never whether), or when its wait would use
the hard-stop wall clock that remained at the last reservation; the raised
exception then names why
(``easyicu_transport_retry_exhausted``).  The allowlist still decides which
failures are retried, from typed status fields only.  A client given no
schedule keeps its historical backoff.  The schedule is part of the reviewed
transport policy and of the client's dispatch identity.

Synthetic exceptions from an in-memory fake transport; waits are recorded,
never slept.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from easyicu.research_agent.providers import clients
from easyicu.research_agent.providers.transport_retry import (
    RETRY_ATTEMPTS_EXHAUSTED,
    RETRY_WALL_CLOCK_EXHAUSTED,
    RETRY_WINDOW_EXHAUSTED,
    TransportRetrySchedule,
)

_WEB_STATUSES = (500, 502, 503, 504)
_SCHEDULE = TransportRetrySchedule((180.0,), jitter_fraction=0.2, window_seconds=300.0)


def _ok() -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="OK"), finish_reason="stop")
        ],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=1, total_tokens=5),
    )


def _failure(status=500, *, text="proxy CONNECT returned status 503", retry_after=None):
    failure = RuntimeError(text)
    if status is not None:
        failure.status_code = status
    if retry_after is not None:
        failure.response = SimpleNamespace(headers={"Retry-After": str(retry_after)})
    return failure


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **_kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(monkeypatch, outcomes, *, schedule=_SCHEDULE, max_retries=None, draw=0.5):
    completions = _Completions(outcomes)
    transport = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_kwargs: transport)
    )
    monkeypatch.setenv("EASYICU_ALLOW_EXTERNAL_LLM", "1")
    monkeypatch.delenv("EASYICU_LLM_STREAM", raising=False)
    from easyicu.research_agent.providers.factory import build_provider_client
    from easyicu.research_agent.providers.llm import OpenAIClient

    retries = max_retries
    if retries is None:
        retries = schedule.max_retries if schedule is not None else 1
    client = build_provider_client(
        provider="openai",
        model="gpt-5.6-luna",
        base_url_override="http://127.0.0.1:8787/v1",
        request_timeout=1.0,
        title="EasyICU transport retry test",
        client_cls=OpenAIClient,
        max_retries=retries,
        retryable_http_status_codes=_WEB_STATUSES,
        allow_environment_overrides=False,
        retry_schedule=schedule,
    )
    client._retry_random = lambda: draw
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    return client, completions, sleeps


def _ask(client):
    from easyicu.research_agent.providers.llm import LLMMessage

    return client.complete([LLMMessage(role="user", content="return OK")])


@pytest.mark.parametrize(("draw", "wait"), [(0.0, 144.0), (0.5, 180.0), (1.0, 216.0)])
def test_a_scheduled_retry_waits_its_drawn_delay_then_succeeds(
    monkeypatch, ra, draw, wait
):
    client, completions, sleeps = _client(monkeypatch, [_failure(), _ok()], draw=draw)

    assert _ask(client) == "OK"

    assert completions.calls == 2
    assert client.last_transport_attempts == 2
    assert sleeps == [pytest.approx(wait)]


def test_a_retry_after_lengthens_a_wait_and_never_shortens_it(monkeypatch, ra):
    client, _completions, sleeps = _client(
        monkeypatch, [_failure(retry_after=240), _ok()]
    )
    assert _ask(client) == "OK"
    assert sleeps == [pytest.approx(240.0)]

    client, _completions, sleeps = _client(
        monkeypatch, [_failure(retry_after=10), _ok()]
    )
    assert _ask(client) == "OK"
    assert sleeps == [pytest.approx(180.0)]


def test_the_last_failure_names_the_spent_attempts(monkeypatch, ra):
    client, completions, sleeps = _client(monkeypatch, [_failure(), _failure(502)])

    with pytest.raises(RuntimeError) as raised:
        _ask(client)

    assert completions.calls == 2
    assert sleeps == [pytest.approx(180.0)]
    assert raised.value.status_code == 502
    assert raised.value.easyicu_transport_attempts == 2
    assert raised.value.easyicu_transport_retry_exhausted == RETRY_ATTEMPTS_EXHAUSTED


@pytest.mark.parametrize(
    ("schedule", "retry_after"),
    [
        # The planned wait alone ends past the window.
        (TransportRetrySchedule((180.0,), window_seconds=100.0), None),
        # The provider asks for a wait that ends past the window.
        (_SCHEDULE, 400),
    ],
)
def test_a_wait_that_ends_past_the_window_is_not_taken(
    monkeypatch, ra, schedule, retry_after
):
    client, completions, sleeps = _client(
        monkeypatch, [_failure(retry_after=retry_after), _ok()], schedule=schedule
    )

    with pytest.raises(RuntimeError) as raised:
        _ask(client)

    assert completions.calls == 1
    assert sleeps == []
    assert raised.value.easyicu_transport_retry_exhausted == RETRY_WINDOW_EXHAUSTED


def test_a_wait_that_would_use_the_remaining_wall_clock_is_not_taken(monkeypatch, ra):
    monkeypatch.setattr(clients, "consume_active_transport_attempt", lambda: 150.0)
    client, completions, sleeps = _client(monkeypatch, [_failure(), _ok()])

    with pytest.raises(RuntimeError) as raised:
        _ask(client)

    assert completions.calls == 1
    assert sleeps == []
    assert raised.value.easyicu_transport_retry_exhausted == RETRY_WALL_CLOCK_EXHAUSTED

    monkeypatch.setattr(clients, "consume_active_transport_attempt", lambda: 1_000.0)
    client, completions, sleeps = _client(monkeypatch, [_failure(), _ok()])
    assert _ask(client) == "OK"
    assert sleeps == [pytest.approx(180.0)]


@pytest.mark.parametrize(
    "failure",
    [
        # The text names a 5xx status; no typed field does.
        _failure(None, text="Error code: 500 - proxy CONNECT returned status 503"),
        _failure("500"),
        _failure(429),
    ],
)
def test_the_allowlist_still_decides_what_is_retried(monkeypatch, ra, failure):
    client, completions, sleeps = _client(monkeypatch, [failure, _ok()])

    with pytest.raises(RuntimeError):
        _ask(client)

    assert completions.calls == 1
    assert sleeps == []
    assert not hasattr(failure, "easyicu_transport_retry_exhausted")


def test_a_client_without_a_schedule_keeps_its_backoff(monkeypatch, ra):
    client, completions, sleeps = _client(
        monkeypatch, [_failure(), _failure(), _ok()], schedule=None, max_retries=2
    )

    assert _ask(client) == "OK"

    assert completions.calls == 3
    assert sleeps == [5.0, 20.0]


def test_a_schedule_states_one_wait_per_retry(monkeypatch, ra):
    with pytest.raises(ValueError, match="one wait per retry"):
        _client(monkeypatch, [_ok()], max_retries=2)


def test_the_schedule_is_part_of_the_reviewed_transport(monkeypatch, ra):
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
    )

    client, completions, _sleeps = _client(monkeypatch, [_ok()])
    policy = provider_authorization_manifest(client)["clients"][0]["transport_policy"]

    assert policy["schema_version"] == "easyicu.provider_transport_policy/4"
    assert policy["transport_max_attempts"] == 2
    assert policy["transport_retry_delays_seconds"] == [180.0]
    assert policy["transport_retry_jitter_fraction"] == 0.2
    assert policy["transport_retry_window_seconds"] == 300.0

    # A schedule changed after the client was minted is not the reviewed one.
    client._retry_schedule = TransportRetrySchedule((1.0,))
    with pytest.raises(PermissionError, match="factory-minted"):
        _ask(client)
    assert completions.calls == 0


def test_a_client_without_a_schedule_keeps_its_policy(monkeypatch, ra):
    from easyicu.research_agent.providers.factory import (
        provider_authorization_manifest,
    )

    client, _completions, _sleeps = _client(monkeypatch, [_ok()], schedule=None)
    policy = provider_authorization_manifest(client)["clients"][0]["transport_policy"]

    assert policy["schema_version"] == "easyicu.provider_transport_policy/2"
    assert not any(key.startswith("transport_retry_") for key in policy)


def test_configured_authorization_states_the_schedule_and_refuses_it_for_cli(
    monkeypatch, ra
):
    from easyicu.research_agent.providers.client_trust import ProviderConfigurationError
    from easyicu.research_agent.providers.factory import (
        provider_authorization_for_configuration,
    )

    def authorize(provider, attempts):
        return provider_authorization_for_configuration(
            provider=provider,
            model="gpt-5.6-luna",
            environment={"OPENAI_BASE_URL": "http://127.0.0.1:8317/v1"},
            request_timeout=480.0,
            transport_max_attempts=attempts,
            retryable_http_status_codes=_WEB_STATUSES,
            retry_schedule=_SCHEDULE,
        )

    policy = authorize("openai", 2)["clients"][0]["transport_policy"]
    assert policy["schema_version"] == "easyicu.provider_transport_policy/4"
    assert policy["transport_retry_delays_seconds"] == [180.0]

    with pytest.raises(ValueError, match="does not match its attempts"):
        authorize("openai", 3)
    with pytest.raises(ProviderConfigurationError):
        authorize("codex", 2)


@pytest.mark.parametrize(
    "arguments",
    [
        {"delays_seconds": ()},
        {"delays_seconds": (0.0,)},
        {"delays_seconds": (math.inf,)},
        {"delays_seconds": (1.0,), "jitter_fraction": 1.0},
        {"delays_seconds": (1.0,), "jitter_fraction": -0.1},
        {"delays_seconds": (1.0,), "window_seconds": 0.0},
        {"delays_seconds": (1.0,), "window_seconds": math.nan},
    ],
)
def test_a_schedule_rejects_waits_it_cannot_keep(arguments):
    with pytest.raises(ValueError):
        TransportRetrySchedule(**arguments)


@pytest.mark.parametrize(("draw", "wait"), [(5.0, 216.0), (-3.0, 144.0)])
def test_a_draw_outside_the_unit_interval_is_clamped(draw, wait):
    planned, reason = _SCHEDULE.next_wait(
        retry_index=0,
        since_start=0.0,
        since_reservation=0.0,
        wall_clock_remaining=None,
        retry_after=None,
        draw=lambda: draw,
    )

    assert reason == ""
    assert planned == pytest.approx(wait)


@pytest.mark.parametrize("draw", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(("since_start", "retried"), [(150.0, True), (157.0, False)])
def test_jitter_moves_when_a_retry_begins_never_whether(draw, since_start, retried):
    wait, reason = _SCHEDULE.next_wait(
        retry_index=0,
        since_start=since_start,
        since_reservation=0.0,
        wall_clock_remaining=None,
        retry_after=None,
        draw=lambda: draw,
    )

    if retried:
        assert reason == ""
        assert wait >= 144.0 - 1e-9
        assert since_start + wait <= 300.0
    else:
        assert (wait, reason) == (None, RETRY_WINDOW_EXHAUSTED)


def test_a_drawn_wait_is_cut_to_end_at_the_window():
    # A gateway answered 504 after 120 s; the upper draw (216 s) would end at
    # 336 s, past the 300 s window, so the retry begins at the window instead.
    wait, reason = _SCHEDULE.next_wait(
        retry_index=0,
        since_start=120.0,
        since_reservation=0.0,
        wall_clock_remaining=None,
        retry_after=None,
        draw=lambda: 1.0,
    )

    assert reason == ""
    assert wait == pytest.approx(180.0)
