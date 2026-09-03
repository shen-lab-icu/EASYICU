"""A contract rejection must not be reported as a formatting problem.

MEASURED: the canonical-9 task m2 (mortality prediction) lost its entire run to
``StructuredResponseFailure[role=planner, n_attempts=5]: 5 planner attempt(s)
failed in 5 distinct ways``, and 27 such exhaustions are recorded across the
corpus. The task dies before its run directory exists, so nothing of it is
written except the batch result.

One of m2's rejections was "analysis plan may declare at most one step with
planned_analysis_role='primary'" -- a perfectly well-formed JSON document
describing an invalid study. The retry told it "Your previous response could
not be parsed into the required structured output" and then, in the most
salient position of the message, instructed it not to use trailing commas,
comments or Markdown fences.

The retry already carries every distinct earlier rejection forward (that was
fixed after three consecutive Planner runs burned all five attempts on
different violations). What it did not do was say which kind of thing was
wrong.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.providers.structured_retry import (
    call_llm_with_structured_retry,
)


# The built-in mock, not a hand-rolled double: `authorized_complete` refuses a
# client the provider graph has not authorized, so a bare stub is never called
# at all and every assertion below would pass vacuously.
_ScriptedLLM = ScriptedMockLLMClient


def _messages_of(llm, index: int):
    return llm.calls[index][0]


def _feedback_text(llm, index: int = 1) -> str:
    """Everything the given call was told, as one string."""

    assert len(llm.calls) > index, "no retry happened"
    return "\n".join(str(m.content or "") for m in _messages_of(llm, index))


def _run(responses, parser):
    llm = _ScriptedLLM(responses)
    try:
        call_llm_with_structured_retry(
            llm=llm,
            messages=[LLMMessage(role="user", content="plan it")],
            parser=parser,
            role="planner",
            max_retries=1,
        )
    except Exception:
        pass
    return llm


def _semantic_parser(raw):
    payload = json.loads(raw)  # succeeds: the response IS valid JSON
    if len(payload.get("primary_steps", [])) > 1:
        raise ValueError(
            "analysis plan may declare at most one step with "
            "planned_analysis_role='primary'"
        )
    return payload


def test_a_well_formed_json_rejection_says_the_formatting_was_fine() -> None:
    llm = _run(
        ['{"primary_steps": ["a", "b"]}', '{"primary_steps": ["a", "b"]}'],
        _semantic_parser,
    )
    feedback = _feedback_text(llm)
    assert "well-formed JSON" in feedback
    assert "Nothing is wrong with the formatting" in feedback
    # The rejection itself must still be there.
    assert "at most one step" in feedback


def test_a_well_formed_json_rejection_drops_the_json_syntax_advice() -> None:
    llm = _run(
        ['{"primary_steps": ["a", "b"]}', '{"primary_steps": ["a", "b"]}'],
        _semantic_parser,
    )
    feedback = _feedback_text(llm)
    for noise in ("trailing commas", "Markdown code fence", "Do not include comments"):
        assert noise not in feedback, f"formatting advice survived: {noise}"


def test_a_genuinely_unparseable_response_still_gets_the_syntax_advice() -> None:
    llm = _run(["not json at all", "still not json"], json.loads)
    feedback = _feedback_text(llm)
    assert "could not be parsed" in feedback
    assert "trailing commas" in feedback


def test_the_framing_is_decided_per_attempt_not_once() -> None:
    """Attempt 2 can be malformed where attempt 1 was not.

    An earlier version of this fix assigned over the caller's parameters, so
    the first well-formed attempt pinned "the formatting was fine" for the
    whole retry loop.
    """

    calls = {"n": 0}

    def parser(raw):
        calls["n"] += 1
        payload = json.loads(raw)
        raise ValueError("contract says no")

    llm = _ScriptedLLM(
        ['{"ok": 1}', "not json at all", '{"ok": 1}']
    )
    with pytest.raises(Exception):
        call_llm_with_structured_retry(
            llm=llm,
            messages=[LLMMessage(role="user", content="plan it")],
            parser=parser,
            role="planner",
            max_retries=2,
        )
    # Attempt 1 was valid JSON -> validation framing on the 2nd call.
    second = _feedback_text(llm, 1)
    assert "well-formed JSON" in second
    # Attempt 2 was NOT valid JSON -> syntax framing must come back.
    third = _feedback_text(llm, 2)
    assert "could not be parsed" in third
    assert "trailing commas" in third


def test_a_caller_supplied_framing_is_never_overridden() -> None:
    llm = _ScriptedLLM(['{"primary_steps": ["a", "b"]}'] * 2)
    try:
        call_llm_with_structured_retry(
            llm=llm,
            messages=[LLMMessage(role="user", content="plan it")],
            parser=_semantic_parser,
            role="planner",
            max_retries=1,
            feedback_preamble="CUSTOM PREAMBLE:",
            feedback_instructions="CUSTOM INSTRUCTIONS.",
        )
    except Exception:
        pass
    feedback = _feedback_text(llm)
    assert "CUSTOM PREAMBLE:" in feedback and "CUSTOM INSTRUCTIONS." in feedback
    assert "well-formed JSON" not in feedback
