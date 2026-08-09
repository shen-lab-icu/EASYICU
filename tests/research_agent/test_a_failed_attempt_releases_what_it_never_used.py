"""A reservation is a hold, and every hold needs a release path.

Before transport the ledger reserves ``max(requested_completion, 128_000)``
completion tokens, because a gateway may return more than it was asked for and
the run must be able to afford the worst case.  A call that REPORTS usage then
releases that hold down to the provider's own numbers.  A call that reports
nothing -- because the connection died -- had no release path at all, and stayed
charged at the floor for the rest of the run.

MEASURED on h1_ventilation_survival, 2026-08-03 (``..._7c6bac6_verify07``).
The local gateway was answering HTTP 500 in 0.98 s with
``Post ".../responses": EOF``, a quarter of the 3.5-7.9 s a successful call
needs.  All 14 attempts asked for 4,096 completion tokens (2,048 for repair);
the 10 that died were charged the 128,000 floor each -- 1,848,481 of the run's
2,000,000 tokens and $45.39 of the batch's $100, for output that never existed.
The run stopped at step 3 of 9 on its own accounting, with no analysis defect.

The rule: release the completion hold to what the caller authorized, keep the
prompt reservation in full (those bytes may have been transmitted and billed).
Retry storms remain bounded by ``max_provider_attempts_per_{run,batch}``.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.authority.provider_hard_stop import (
    PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR,
    consume_active_provider_hard_stop_attempt,
    provider_hard_stop_call_scope,
)

from .test_provider_hard_stop import _ledger, _message


def _one_failed_attempt(tmp_path, *, max_tokens: int, error: str = "EOF"):
    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with provider_hard_stop_call_scope(
        task=task,
        role="planner",
        model="test-model",
        messages=_message(),
        max_tokens=max_tokens,
    ) as call:
        consume_active_provider_hard_stop_attempt()
        call.fail(error)
    return ledger.snapshot()["tasks"][0]["calls"][0]


def test_the_floor_is_released_down_to_what_the_caller_asked_for(tmp_path):
    call = _one_failed_attempt(tmp_path, max_tokens=4096)

    assert call["state"] == "failed_usage_unknown"
    assert call["requested_completion_tokens"] == 4096
    assert call["completion_token_reservation"] == 4096
    assert call["unreported_completion_hold_released"] == (
        PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR - 4096
    )


def test_the_prompt_reservation_is_not_released(tmp_path):
    """Those bytes may have reached the provider, so they stay charged."""

    call = _one_failed_attempt(tmp_path, max_tokens=4096)

    prompt_reserve = int(call["prompt_token_reservation"])
    assert prompt_reserve > 0
    assert call["accounted_tokens"] == prompt_reserve + 4096


def test_a_caller_asking_for_more_than_the_floor_keeps_the_larger_hold(tmp_path):
    """The release only ever shrinks a hold toward the caller's own ceiling."""

    over = PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR + 5_000
    call = _one_failed_attempt(tmp_path, max_tokens=over)

    assert call["completion_token_reservation"] == over
    assert "unreported_completion_hold_released" not in call


def test_the_cost_is_released_with_the_tokens(tmp_path):
    """The batch stops on dollars too, so both ledgers must agree."""

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with provider_hard_stop_call_scope(
        task=task,
        role="planner",
        model="test-model",
        messages=_message(),
        max_tokens=4096,
    ) as scoped:
        consume_active_provider_hard_stop_attempt()
        scoped.fail("EOF")

    call = ledger.snapshot()["tasks"][0]["calls"][0]
    prompt_reserve = int(call["prompt_token_reservation"])
    expected = (
        prompt_reserve * ledger.limits.input_cost_usd_per_million_tokens
        + 4096 * ledger.limits.output_cost_usd_per_million_tokens
    ) / 1_000_000.0
    assert call["accounted_estimated_cost_usd"] == pytest.approx(expected)


def test_a_reported_call_is_still_released_to_the_provider_numbers(tmp_path):
    """The path that already worked must keep working."""

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with provider_hard_stop_call_scope(
        task=task,
        role="planner",
        model="test-model",
        messages=_message(),
        max_tokens=4096,
    ) as call:
        consume_active_provider_hard_stop_attempt()
        call.complete(
            {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        )

    recorded = ledger.snapshot()["tasks"][0]["calls"][0]
    assert recorded["state"] == "completed"
    assert recorded["accounted_tokens"] == 120


def test_the_retry_that_replaced_an_attempt_is_released_too(tmp_path):
    """The h1 shape: our own retry loop closes the prior attempt.

    Those are the calls that consumed the run.  They are closed by
    ``reserve_transport_attempt`` rather than by ``fail()``, so the release has
    to live where BOTH paths reach it.
    """

    ledger = _ledger(tmp_path)
    task = ledger.start_task("E1")
    with provider_hard_stop_call_scope(
        task=task,
        role="planner",
        model="test-model",
        messages=_message(),
        max_tokens=4096,
    ):
        consume_active_provider_hard_stop_attempt()
        consume_active_provider_hard_stop_attempt()

    calls = ledger.snapshot()["tasks"][0]["calls"]
    retried = calls[0]
    assert retried["error_type"] == "TransportRetry"
    assert retried["completion_token_reservation"] == 4096


_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def test_the_recorded_run_would_not_have_hit_its_ceiling(tmp_path):
    """Replays the exact ledger of the run this was measured on.

    Not a restatement of the rule: it reads the 14 recorded attempts and applies
    the release to each, so it stops being meaningful only if the corpus stops
    containing the failure.
    """

    recorded = sorted(
        _CORPUS.glob("batch_*_verify07/ehrflowbench_progress.json")
    )
    if not recorded:
        pytest.skip("the run that recorded this ceiling is not on disk")
    ledger = json.loads(recorded[-1].read_text(encoding="utf-8"))
    ceiling = int(ledger["limits"]["max_total_tokens_per_run"])

    before = after = 0
    failures = 0
    for task in ledger["tasks"]:
        for call in task["calls"]:
            charged = int(call.get("accounted_tokens") or 0)
            before += charged
            if str(call.get("state")) == "completed":
                after += charged
                continue
            failures += 1
            after += int(call.get("prompt_token_reservation") or 0) + int(
                call.get("requested_completion_tokens") or 0
            )

    assert failures == 10, "the corpus no longer contains the transport storm"
    assert before > ceiling * 0.97, before  # it really did consume the run
    assert after < ceiling * 0.40, after  # and would not have
