"""A failed retry loop must report the shape of its history, not one message.

Both shapes below cost a real diagnosis on 2026-07-29:

* Five Planner attempts were rejected by the *same* host check (a required
  field the payload projection deleted before the check ran). The traceback
  showed one message, which reads like a single bad response rather than a
  loop that could never converge.
* A transport 408 aborted the loop from outside the parser guard, discarding
  two already-recorded parse failures. The operator saw only the 408 and had
  no way to know two responses had already been rejected.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.providers.structured_retry import (
    StructuredAttempt,
    StructuredResponseFailure,
    annotate_with_attempt_history,
    call_llm_with_structured_retry,
    summarise_attempt_history,
)


def _failed(index: int, error_class: str, message: str) -> StructuredAttempt:
    return StructuredAttempt(
        attempt=index,
        raw_head="{...}",
        raw_chars=5,
        error_class=error_class,
        error_message=message,
    )


def test_identical_failures_are_reported_as_a_loop_that_changed_nothing() -> None:
    """The signature of an unwinnable retry loop."""

    message = (
        "Planner exposure/outcome distribution steps must declare "
        "exposure_outcome_distribution_spec"
    )
    attempts = [_failed(index, "ValueError", message) for index in range(5)]

    summary = summarise_attempt_history(attempts, role="planner")

    assert "all 5 planner attempts failed identically" in summary
    assert "the retry feedback did not change the outcome" in summary
    assert message in summary


def test_distinct_failures_are_each_reported() -> None:
    """A thrashing model is a different problem and must read differently."""

    attempts = [
        _failed(0, "JSONDecodeError", "Expecting value: line 1 column 1"),
        _failed(1, "ValidationError", "outcome_levels: field required"),
        _failed(2, "ValidationError", "outcome_levels: field required"),
    ]

    summary = summarise_attempt_history(attempts, role="planner")

    assert "3 planner attempt(s) failed in 2 distinct ways" in summary
    assert "JSONDecodeError" in summary
    assert "ValidationError" in summary
    assert "identically" not in summary


def test_the_failure_message_carries_the_history() -> None:
    attempts = [_failed(index, "ValueError", "same complaint") for index in range(4)]

    failure = StructuredResponseFailure(attempts, role="planner")

    assert "n_attempts=4" in str(failure)
    assert "all 4 planner attempts failed identically" in str(failure)


def test_a_transport_failure_carries_out_the_parse_failures_it_aborted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tonight's exact shape: two rejected responses, then a 408.

    ``authorized_complete`` is replaced rather than driven through a real
    client: provider authorization is a separate boundary with its own
    tests, and what is under test here is that the loop carries its
    history out through an exception raised at the transport call.
    """

    class _Boom(RuntimeError):
        pass

    calls = {"n": 0}

    def _fake_authorized_complete(_llm, _messages, **_kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            return "not json"
        raise _Boom("Error code: 408 - stream disconnected before completion")

    monkeypatch.setattr(
        "easyicu.research_agent.providers.structured_retry.authorized_complete",
        _fake_authorized_complete,
    )

    def _parser(raw: str):
        raise ValueError(f"Planner LLM did not return parseable JSON (len={len(raw)})")

    with pytest.raises(_Boom) as excinfo:
        call_llm_with_structured_retry(
            object(),
            [],
            parser=_parser,
            role="planner",
            max_retries=5,
        )

    notes = getattr(excinfo.value, "__notes__", [])
    assert notes, "the transport error carried no retry history"
    joined = " ".join(notes)
    assert "all 2 planner attempts failed identically" in joined
    assert "did not return parseable JSON" in joined


def test_a_clean_first_call_is_not_described_as_a_failure() -> None:
    exc = RuntimeError("boom")
    annotate_with_attempt_history(
        exc,
        [
            StructuredAttempt(
                attempt=0,
                raw_head="{}",
                raw_chars=2,
                error_class=None,
                error_message=None,
            )
        ],
        role="planner",
    )

    joined = " ".join(getattr(exc, "__notes__", []))
    assert "parsed cleanly" in joined


def test_no_history_adds_no_note() -> None:
    """Nothing to say is said by saying nothing, not by an empty note."""

    exc = RuntimeError("boom")
    annotate_with_attempt_history(exc, [], role="planner")

    assert not getattr(exc, "__notes__", [])
