"""Unknown usage releases only the conservative prompt over-reservation.

``_prompt_token_reservation`` bounds a prompt by its UTF-8 byte count::

    prompt_bytes + 16 * len(messages) + 64 + PROVIDER_PROMPT_OVERHEAD_...

and says why that is safe: "Every tokenizer token consumes at least one encoded
byte."  True, and about four times the truth -- the same codebase measured the
ratio on real receipts and wrote it down::

    providers/prompt_budget.py
      bytes/token over the 2026-07-23 E1 replay (8 real calls, all roles)
        min 3.7685   max 4.3812   mean 3.99
      CONSERVATIVE_BYTES_PER_TOKEN = 3.0     # deliberately below every sample

The prompt's byte-denominated hold can safely release to the calibrated
conservative estimator after a failed call. The completion hold cannot: a
reviewed gateway may ignore or strip the requested cap, so unknown completion
usage must keep the provider-maximum reserve.

MEASURED on verify12, 2026-08-04, from the batch's own durable ledger:

    a successful call   accounted_tokens =  23,436   (provider-reported)
    a failed  call      accounted_tokens =  90,542   (86,446 prompt + 4,096)

A call that returned nothing cost 3.9x one that returned an answer.  Over the
m1 run: 19 of 39 calls failed and were charged 707,014 tokens -- 35% of the
2,000,000 run ceiling, and 2.75x the 256,708 the 20 successful calls actually
used.  The manuscript writer was then refused a 150,931-token reservation, so
all nine analysis steps passed and the run produced no manuscript.

The reservation itself is not the problem and is unchanged: before a call, a
byte bound is the honest worst case.  What changes is the release, and it
releases to the estimator this codebase already calibrated for this exact
question rather than to a new number.
"""

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
from easyicu.research_agent.providers.prompt_budget import (
    CONSERVATIVE_BYTES_PER_TOKEN,
    OBSERVED_BYTES_PER_TOKEN,
    estimate_prompt_tokens,
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


def test_a_failed_call_is_not_charged_a_token_for_every_byte(tmp_path):
    """The defect, at the size the recorded run hit it."""

    prompt = "x" * 82_000
    call = _one_failed_call(tmp_path, prompt=prompt)

    assert call["state"] == "failed_usage_unknown"
    charged_prompt = int(call["prompt_token_reservation"])
    assert charged_prompt < len(prompt), (
        f"a failed call was charged {charged_prompt} prompt tokens for a "
        f"{len(prompt)}-byte "
        "prompt -- at least one token per byte"
    )
    assert (
        int(call["completion_token_reservation"])
        == PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
    )


def test_the_release_uses_the_estimator_this_codebase_calibrated(tmp_path):
    """Not a new constant: the one the prompt-budget guard already uses."""

    prompt = "x" * 82_000
    call = _one_failed_call(tmp_path, prompt=prompt, max_tokens=4096)

    expected_prompt = (
        estimate_prompt_tokens(len(prompt.encode("utf-8")))
        + PROVIDER_PROMPT_OVERHEAD_TOKEN_RESERVATION
    )
    assert int(call["accounted_tokens"]) == (
        expected_prompt + PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
    ), call


def test_the_released_estimate_still_over_counts_every_observed_ratio(tmp_path):
    """Failing closed is preserved: the divisor is below all real receipts.

    3.0 against a measured minimum of 3.7685 means the release is still an
    over-charge, just not a 4x one.
    """

    assert CONSERVATIVE_BYTES_PER_TOKEN < OBSERVED_BYTES_PER_TOKEN

    prompt_bytes = 82_000
    call = _one_failed_call(tmp_path, prompt="x" * prompt_bytes)
    truthful = prompt_bytes / OBSERVED_BYTES_PER_TOKEN

    assert int(call["prompt_token_reservation"]) > truthful


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
