"""Unknown provider usage preserves the pre-transport worst-case reserve."""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.provider_hard_stop import (
    PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR,
    consume_active_provider_hard_stop_attempt,
    provider_hard_stop_call_scope,
)

from tests.research_agent.providers.test_provider_hard_stop import _ledger, _message


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


def test_unknown_usage_keeps_the_worst_case_completion_reserve(tmp_path):
    call = _one_failed_attempt(tmp_path, max_tokens=4096)

    assert call["state"] == "failed_usage_unknown"
    assert call["requested_completion_tokens"] == 4096
    assert call["completion_token_reservation"] == (
        PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
    )
    assert "unreported_completion_hold_released" not in call


def test_unknown_usage_keeps_prompt_and_completion_accounting(tmp_path):
    call = _one_failed_attempt(tmp_path, max_tokens=4096)

    prompt_reserve = int(call["prompt_token_reservation"])
    assert prompt_reserve > 0
    assert call["accounted_tokens"] == (
        prompt_reserve + PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
    )


def test_a_caller_asking_for_more_than_the_floor_keeps_the_larger_hold(tmp_path):
    over = PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR + 5_000
    call = _one_failed_attempt(tmp_path, max_tokens=over)

    assert call["completion_token_reservation"] == over
    assert "unreported_completion_hold_released" not in call


def test_unknown_usage_cost_keeps_the_worst_case_completion_reserve(tmp_path):
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
        + PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
        * ledger.limits.output_cost_usd_per_million_tokens
    ) / 1_000_000.0
    assert call["accounted_estimated_cost_usd"] == pytest.approx(expected)


def test_a_reported_call_is_released_to_the_provider_numbers(tmp_path):
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


def test_retry_with_unknown_usage_keeps_the_worst_case_reserve(tmp_path):
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

    retried = ledger.snapshot()["tasks"][0]["calls"][0]
    assert retried["error_type"] == "TransportRetry"
    assert (
        retried["completion_token_reservation"]
        == PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
    )
