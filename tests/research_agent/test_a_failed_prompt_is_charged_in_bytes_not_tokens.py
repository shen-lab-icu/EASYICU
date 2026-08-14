"""Unknown provider usage retains the original pre-transport prompt bound."""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.authority.provider_hard_stop import (
    PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR,
    PROVIDER_PROMPT_OVERHEAD_TOKEN_RESERVATION,
    ProviderHardStopLedger,
    ProviderHardStopLimits,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


class _Message:
    def __init__(self, content: str) -> None:
        self.role = "user"
        self.content = content


def _ledger(tmp_path) -> ProviderHardStopLedger:
    return ProviderHardStopLedger(
        path=(tmp_path / "ledger.json").resolve(),
        task_ids=("m1",),
        batch_id="test-batch",
        limits=ProviderHardStopLimits(
            max_provider_attempts_per_run=200,
            max_provider_attempts_per_batch=900,
            max_total_tokens_per_run=2_000_000,
            max_total_tokens_per_batch=8_000_000,
            max_estimated_cost_usd_per_batch=100.0,
            max_wall_clock_seconds_per_task=3600.0,
            input_cost_usd_per_million_tokens=1.25,
            output_cost_usd_per_million_tokens=10.0,
        ),
    )


def _one_failed_call(tmp_path, *, prompt: str, max_tokens: int = 4096) -> dict:
    ledger = _ledger(tmp_path)
    ledger.start_task("m1")
    attempt = ledger.reserve_transport_attempt(
        task_id="m1",
        role="writer",
        model="gpt-test",
        messages=[_Message(prompt)],
        max_tokens=max_tokens,
        prior_attempt_id=None,
    )
    ledger.finish_transport_attempt(
        task_id="m1",
        attempt_id=attempt,
        usage=None,
        error_type="TransportRetry",
    )
    document = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    calls = document["tasks"][0]["calls"]
    assert len(calls) == 1
    return calls[0]


def test_a_failed_call_keeps_a_token_for_every_prompt_byte(tmp_path):
    prompt = "x" * 82_000
    call = _one_failed_call(tmp_path, prompt=prompt)

    assert call["state"] == "failed_usage_unknown"
    charged_prompt = int(call["prompt_token_reservation"])
    assert charged_prompt >= len(prompt.encode("utf-8"))
    assert (
        int(call["completion_token_reservation"])
        == PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
    )
    assert "unreported_prompt_hold_released" not in call


def test_unknown_usage_keeps_the_exact_original_prompt_hold(tmp_path):
    prompt = "x" * 82_000
    ledger = _ledger(tmp_path)
    ledger.start_task("m1")
    attempt = ledger.reserve_transport_attempt(
        task_id="m1",
        role="writer",
        model="gpt-test",
        messages=[_Message(prompt)],
        max_tokens=4096,
        prior_attempt_id=None,
    )
    before = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))[
        "tasks"
    ][0]["calls"][0]
    ledger.finish_transport_attempt(
        task_id="m1", attempt_id=attempt, usage=None, error_type="TransportRetry"
    )
    after = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))[
        "tasks"
    ][0]["calls"][0]

    assert after["prompt_token_reservation"] == before["prompt_token_reservation"]
    assert after["accounted_tokens"] == before["accounted_tokens"]
    assert after["accounted_estimated_cost_usd"] == pytest.approx(
        before["accounted_estimated_cost_usd"]
    )


def test_completed_but_unreported_usage_keeps_both_original_holds(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_task("m1")
    attempt = ledger.reserve_transport_attempt(
        task_id="m1",
        role="writer",
        model="gpt-test",
        messages=[_Message("x" * 82_000)],
        max_tokens=4096,
        prior_attempt_id=None,
    )
    before = ledger.snapshot()["tasks"][0]["calls"][0]

    ledger.finish_transport_attempt(
        task_id="m1", attempt_id=attempt, usage=None, error_type=None
    )
    after = ledger.snapshot()["tasks"][0]["calls"][0]

    assert after["state"] == "completed_usage_unreported"
    assert after["prompt_token_reservation"] == before["prompt_token_reservation"]
    assert after["completion_token_reservation"] == before[
        "completion_token_reservation"
    ]
    assert after["accounted_tokens"] == before["accounted_tokens"]


def test_the_reservation_before_the_call_is_unchanged(tmp_path):
    """A call must still be refused if its worst case would not fit.

    The byte bound is right while the answer is unknown; only the release
    after a known failure changes.
    """

    prompt = "x" * 82_000
    ledger = _ledger(tmp_path)
    ledger.start_task("m1")
    attempt = ledger.reserve_transport_attempt(
        task_id="m1",
        role="writer",
        model="gpt-test",
        messages=[_Message(prompt)],
        max_tokens=4096,
        prior_attempt_id=None,
    )
    document = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    call = document["tasks"][0]["calls"][0]

    assert call["state"] == "in_progress"
    assert int(call["prompt_token_reservation"]) >= len(prompt)
    ledger.finish_transport_attempt(
        task_id="m1", attempt_id=attempt, usage=None, error_type="TransportRetry"
    )


def test_a_reported_call_still_lands_on_the_providers_own_numbers(tmp_path):
    """The release path that already worked must not move."""

    ledger = _ledger(tmp_path)
    ledger.start_task("m1")
    attempt = ledger.reserve_transport_attempt(
        task_id="m1",
        role="writer",
        model="gpt-test",
        messages=[_Message("x" * 82_000)],
        max_tokens=4096,
        prior_attempt_id=None,
    )
    ledger.finish_transport_attempt(
        task_id="m1",
        attempt_id=attempt,
        usage={"prompt_tokens": 20_100, "completion_tokens": 3_336},
        error_type=None,
    )
    call = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))["tasks"][
        0
    ]["calls"][0]

    assert int(call["accounted_tokens"]) == 23_436


def test_a_failed_call_still_costs_more_than_nothing(tmp_path):
    """The prompt did leave this machine; it is not free."""

    call = _one_failed_call(tmp_path, prompt="x" * 82_000)

    assert int(call["accounted_tokens"]) > 0
    assert float(call["accounted_estimated_cost_usd"]) > 0.0


def test_the_recorded_ledger_shows_a_failure_costing_more_than_an_answer():
    """Re-measures the run that motivated this, from its own ledger."""

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    worst_failed = 0
    best_reported = 0
    failed_total = 0
    for path in _CORPUS.glob("batch_*/ehrflowbench_progress.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for task in document.get("tasks", []) or []:
            for call in task.get("calls", []) or []:
                if not isinstance(call, dict):
                    continue
                tokens = int(call.get("accounted_tokens") or 0)
                if call.get("error_type"):
                    worst_failed = max(worst_failed, tokens)
                    failed_total += tokens
                elif call.get("reported_prompt_tokens") is not None:
                    best_reported = max(best_reported, tokens)

    if not worst_failed or not best_reported:
        pytest.skip("no recorded ledger carries both call kinds")
    assert worst_failed > best_reported, (worst_failed, best_reported)
    assert failed_total > 100_000, failed_total
