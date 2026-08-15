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
    safe_structured_attempt_metadata,
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


def test_feedback_carries_forward_the_constraints_already_violated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thrashing shape: fix the newest complaint, re-break an older one.

    Three consecutive real Planner runs spent all five attempts on three to
    five *different* violations. Showing only the newest rejection lets the
    model treat the constraint set as one item long.
    """

    prompts: list = []
    complaints = [
        "spec columns must be explicit step inputs",
        "robustness axis must be one of the closed set",
        "plan is missing required role: data_quality",
        "spec columns must be explicit step inputs",
    ]

    def _fake_authorized_complete(_llm, messages, **_kwargs):
        prompts.append(messages)
        return "not json"

    monkeypatch.setattr(
        "easyicu.research_agent.providers.structured_retry.authorized_complete",
        _fake_authorized_complete,
    )

    calls = {"n": 0}

    def _parser(_raw: str):
        index = min(calls["n"], len(complaints) - 1)
        calls["n"] += 1
        raise ValueError(complaints[index])

    with pytest.raises(StructuredResponseFailure):
        call_llm_with_structured_retry(
            object(),
            [],
            parser=_parser,
            role="planner",
            max_retries=3,
        )

    # The final request must restate every distinct constraint seen so far,
    # not merely the one that rejected the previous response.
    final = "\n".join(str(message.content) for message in prompts[-1])
    assert "missing required role: data_quality" in final
    assert "robustness axis must be one of the closed set" in final
    assert "spec columns must be explicit step inputs" in final

    # The newest complaint is stated once, in the preamble -- not duplicated
    # into the carried-forward list.
    assert final.count("plan is missing required role: data_quality") == 1


def test_a_repeated_complaint_is_not_listed_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical rejections stay one line; the note must not grow per attempt."""

    prompts: list = []

    def _fake_authorized_complete(_llm, messages, **_kwargs):
        prompts.append(messages)
        return "not json"

    monkeypatch.setattr(
        "easyicu.research_agent.providers.structured_retry.authorized_complete",
        _fake_authorized_complete,
    )

    def _parser(_raw: str):
        raise ValueError("the one and only complaint")

    with pytest.raises(StructuredResponseFailure):
        call_llm_with_structured_retry(
            object(), [], parser=_parser, role="planner", max_retries=3
        )

    final = "\n".join(str(message.content) for message in prompts[-1])
    assert final.count("the one and only complaint") == 1
    assert "Earlier attempts were rejected" not in final


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


def test_python_310_compatibility_keeps_retry_notes_without_add_note() -> None:
    """The supported 3.10 runtime must expose the same audit contract."""

    class _LegacyRuntimeError(RuntimeError):
        add_note = None

    exc = _LegacyRuntimeError("boom")
    annotate_with_attempt_history(
        exc,
        [_failed(0, "ValueError", "bad structured response")],
        role="planner",
    )

    joined = " ".join(getattr(exc, "__notes__", []))
    assert "1 planner attempt failed" in joined
    assert "bad structured response" in joined


def test_no_history_adds_no_note() -> None:
    """Nothing to say is said by saying nothing, not by an empty note."""

    exc = RuntimeError("boom")
    annotate_with_attempt_history(exc, [], role="planner")

    assert not getattr(exc, "__notes__", [])


def test_safe_attempt_projection_excludes_response_message_and_extra_usage() -> None:
    secret = "sk-test-secret-response"
    projected = safe_structured_attempt_metadata(
        [
            StructuredAttempt(
                attempt=0,
                raw_head=f'{{"prompt": "{secret}"}}',
                raw_chars=31,
                error_class="ValidationError",
                error_message=f"input contained {secret}",
                finish_reason="stop",
                usage_summary={
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "actual_model": secret,
                },
                transport_attempts=2,
            )
        ]
    )

    assert projected == [
        {
            "attempt": 1,
            "raw_chars": 31,
            "error_class": "validation",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
            "transport_attempts": 2,
        }
    ]
    assert secret not in str(projected)


def test_structured_failure_captures_safe_provider_metadata_per_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Client:
        last_finish_reason = "length"
        last_usage = {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "total_tokens": 28,
            "actual_model": "provider-private-routing-name",
        }
        last_transport_attempts = 2

    monkeypatch.setattr(
        "easyicu.research_agent.providers.structured_retry.authorized_complete",
        lambda *_args, **_kwargs: "not json",
    )

    with pytest.raises(StructuredResponseFailure) as raised:
        call_llm_with_structured_retry(
            _Client(),
            [],
            parser=lambda _raw: (_ for _ in ()).throw(ValueError("bad contract")),
            role="planner",
            max_retries=0,
        )

    assert safe_structured_attempt_metadata(raised.value.attempts) == [
        {
            "attempt": 1,
            "raw_chars": 8,
            "error_class": "validation",
            "finish_reason": "length",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
            },
            "transport_attempts": 2,
            "violation_sha256": (
                "b7b6859180b747b49401775dda829994d1f7c8ad3044110b38c51b503105785b"
            ),
        }
    ]


def test_validation_projection_records_only_closed_stage_field_paths_and_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-private-field-and-message"

    class _ValidationFailure(ValueError):
        easyicu_structured_validation_stage = "schema_validation"

        def errors(self):
            return [
                {
                    "loc": ("steps", 2, "figure_panels", 0, "chart_type"),
                    "type": "literal_error",
                    "msg": secret,
                    "input": secret,
                },
                {
                    "loc": (secret,),
                    "type": secret,
                    "msg": secret,
                },
            ]

    monkeypatch.setattr(
        "easyicu.research_agent.providers.structured_retry.authorized_complete",
        lambda *_args, **_kwargs: '{"plan": "private"}',
    )

    with pytest.raises(StructuredResponseFailure) as raised:
        call_llm_with_structured_retry(
            object(),
            [],
            parser=lambda _raw: (_ for _ in ()).throw(_ValidationFailure(secret)),
            role="planner",
            max_retries=0,
        )

    projected = safe_structured_attempt_metadata(raised.value.attempts)
    assert projected[0]["validation_stage"] == "schema_validation"
    assert projected[0]["validation_issues"] == [
        {
            "location": ["steps", 2, "figure_panels", 0, "chart_type"],
            "issue_type": "literal_error",
        },
        {"location": ["<other>"], "issue_type": "other"},
    ]
    assert len(projected[0]["violation_sha256"]) == 64
    assert secret not in str(projected)


def test_validation_stage_is_inferred_from_the_contract_owner_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def validate_literature_citation_bindings() -> None:
        raise ValueError("private citation payload")

    monkeypatch.setattr(
        "easyicu.research_agent.providers.structured_retry.authorized_complete",
        lambda *_args, **_kwargs: "{}",
    )

    with pytest.raises(StructuredResponseFailure) as raised:
        call_llm_with_structured_retry(
            object(),
            [],
            parser=lambda _raw: validate_literature_citation_bindings(),
            role="planner",
            max_retries=0,
        )

    projected = safe_structured_attempt_metadata(raised.value.attempts)
    assert projected[0]["validation_stage"] == "literature_authority"
    assert "private citation payload" not in str(projected)


def test_transport_failure_attaches_response_free_terminal_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("provider echoed sk-private-key")
    failure.easyicu_transport_attempts = 2

    def _raise(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        "easyicu.research_agent.providers.structured_retry.authorized_complete",
        _raise,
    )

    with pytest.raises(RuntimeError) as raised:
        call_llm_with_structured_retry(
            object(), [], parser=lambda raw: raw, role="planner", max_retries=4
        )

    assert raised.value is failure
    assert raised.value.easyicu_structured_attempt_metadata == [
        {
            "attempt": 1,
            "raw_chars": 0,
            "error_class": "error",
            "finish_reason": None,
            "usage": {},
            "transport_attempts": 2,
        }
    ]
    assert "sk-private-key" not in str(raised.value.easyicu_structured_attempt_metadata)


def test_safe_attempt_projection_closes_secret_shaped_categories() -> None:
    projected = safe_structured_attempt_metadata(
        [
            StructuredAttempt(
                attempt=0,
                raw_head="",
                raw_chars=0,
                error_class="sk-secret-shaped-error-class",
                error_message=None,
                finish_reason="sk-secret-shaped-finish-reason",
            )
        ]
    )

    assert projected[0]["error_class"] == "error"
    assert projected[0]["finish_reason"] == "other"
    assert "sk-secret" not in str(projected)


#: The exact ValidationError rendering that killed ``e3_kdigo_gradient`` at
#: planning on the 2026-08-02 nine-task run: one short header line plus one
#: long violation line. Kept verbatim because the header/violation shape is
#: what makes a naive line-boundary clip drop the only useful part.
_REAL_E3_REJECTION = (
    "1 problem(s), all of which must be fixed together in one corrected "
    "response:\n"
    "    - steps.8: Value error, model_requirements are currently supported "
    "only on method='adjusted_association_models' steps that declare expected "
    "output 'table:adjusted_association_estimates'; other analysis families "
    "must use their family-specific planning and validation contracts"
)


def test_a_clipped_failure_never_reads_as_a_complete_constraint() -> None:
    """The post-mortem is what a human reads when planning produced nothing.

    Rendered with a bare slice, E3's rejection ended at "other analysis
    families must use" -- a sentence that stops right before it says what they
    must use. Nothing marked the cut, so the host's own guidance reads as
    incomplete and the wrong thing gets investigated first.
    """

    summary = summarise_attempt_history(
        [
            StructuredAttempt(
                attempt=0,
                raw_head="",
                raw_chars=0,
                error_class="ValidationError",
                error_message=_REAL_E3_REJECTION,
            ),
            StructuredAttempt(
                attempt=1,
                raw_head="",
                raw_chars=0,
                error_class="JSONDecodeError",
                error_message="Expecting ',' delimiter: line 1 column 6508",
            ),
        ],
        role="planner",
    )

    assert "[...truncated]" in summary, (
        "a clipped failure must say it was clipped; without the marker the "
        "remaining text reads as the whole constraint"
    )
    # The clip must still carry the violation, not just the header line that
    # says a violation exists.
    assert "model_requirements are currently supported only" in summary
    assert "steps.8" in summary


def test_a_failure_that_fits_is_not_marked_as_clipped() -> None:
    """The marker has to mean something, so it may not appear by default."""

    short = "Expecting ',' delimiter: line 1 column 6508 (char 6507)"
    summary = summarise_attempt_history(
        [
            StructuredAttempt(
                attempt=0,
                raw_head="",
                raw_chars=0,
                error_class="JSONDecodeError",
                error_message=short,
            ),
            StructuredAttempt(
                attempt=1,
                raw_head="",
                raw_chars=0,
                error_class="ValueError",
                error_message="two distinct ways are needed to reach the list",
            ),
        ],
        role="planner",
    )

    assert short in summary
    assert "[...truncated]" not in summary
