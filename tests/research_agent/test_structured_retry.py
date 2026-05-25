"""Tests for the structured-response retry wrapper.

This is the general fix for the planner-JSON-parse crash observed in
the 3-rep reproducibility experiment (rep3 vasopressor): on parse
failure, feed the error back to the LLM as a feedback message and let
it try again. The wrapper is role-agnostic so the same retry policy
applies to planner / replanner / writer / any future structured-output
agent.
"""

from __future__ import annotations

import json
from typing import List

import pytest

from easyicu.research_agent.llm import LLMMessage
from easyicu.research_agent.structured_retry import (
    StructuredResponseFailure,
    call_llm_with_structured_retry,
)


class _ScriptedClient:
    """Returns a scripted sequence of strings and records each call."""

    name = "scripted"

    def __init__(self, replies: List[str]) -> None:
        self.replies = list(replies)
        self.calls: List[List[LLMMessage]] = []

    def complete(self, messages, *, max_tokens=2048, temperature=0.2):
        # Deep-copy the messages so the test can inspect what the
        # wrapper actually sent on each retry.
        self.calls.append(
            [LLMMessage(role=m.role, content=m.content) for m in messages]
        )
        if not self.replies:
            raise RuntimeError("scripted client ran out of replies")
        return self.replies.pop(0)


def test_structured_retry_returns_first_success_without_retrying():
    client = _ScriptedClient(['{"value": 42}'])
    out = call_llm_with_structured_retry(
        client,
        [LLMMessage(role="user", content="give json")],
        parser=lambda raw: json.loads(raw),
        role="probe",
        max_retries=2,
    )
    assert out == {"value": 42}
    assert len(client.calls) == 1, "no retry should have happened"


def test_structured_retry_feeds_error_back_and_succeeds_on_second_attempt():
    bad = "{ not valid json"
    good = '{"value": 7}'
    client = _ScriptedClient([bad, good])

    out = call_llm_with_structured_retry(
        client,
        [LLMMessage(role="user", content="give json")],
        parser=lambda raw: json.loads(raw),
        role="planner",
        max_retries=2,
        format_reminder="The JSON must include the key 'value'.",
    )
    assert out == {"value": 7}
    # Two calls: the failed one and the retry that succeeded.
    assert len(client.calls) == 2
    # The retry conversation must include the original user message, the
    # failed assistant turn (verbatim), and a new user-feedback message.
    retry_msgs = client.calls[1]
    roles = [m.role for m in retry_msgs]
    assert roles[0] == "user"
    assert roles[-2] == "assistant"
    assert retry_msgs[-2].content == bad
    assert roles[-1] == "user"
    feedback_content = retry_msgs[-1].content
    assert "could not be parsed" in feedback_content
    assert "JSONDecodeError" in feedback_content
    # format_reminder must be included
    assert "must include the key 'value'" in feedback_content


def test_structured_retry_raises_after_exhausting_retries():
    client = _ScriptedClient(["bad-1", "bad-2", "bad-3"])
    with pytest.raises(StructuredResponseFailure) as ctx:
        call_llm_with_structured_retry(
            client,
            [LLMMessage(role="user", content="x")],
            parser=lambda raw: json.loads(raw),
            role="planner",
            max_retries=2,
        )
    err = ctx.value
    assert err.role == "planner"
    assert len(err.attempts) == 3, "should record every attempt"
    # All attempts should have error fields set.
    assert all(a.error_class == "JSONDecodeError" for a in err.attempts)
    # Three LLM calls
    assert len(client.calls) == 3


def test_structured_retry_max_retries_zero_means_single_call():
    client = _ScriptedClient(["nope"])
    with pytest.raises(StructuredResponseFailure):
        call_llm_with_structured_retry(
            client,
            [LLMMessage(role="user", content="x")],
            parser=lambda raw: json.loads(raw),
            role="solo",
            max_retries=0,
        )
    assert len(client.calls) == 1, "max_retries=0 should call exactly once"


def test_structured_retry_handles_value_error_from_parser():
    """Parser may raise any exception — wrapper catches them all and feeds back."""

    def picky_parser(raw: str) -> dict:
        data = json.loads(raw)  # may raise JSONDecodeError
        if "required_key" not in data:
            raise ValueError("missing required_key")
        return data

    client = _ScriptedClient(['{"other_key": 1}', '{"required_key": "ok"}'])
    out = call_llm_with_structured_retry(
        client,
        [LLMMessage(role="user", content="x")],
        parser=picky_parser,
        role="planner",
        max_retries=2,
    )
    assert out == {"required_key": "ok"}
    assert len(client.calls) == 2
    feedback = client.calls[1][-1].content
    assert "ValueError" in feedback
    assert "missing required_key" in feedback


def test_structured_retry_does_not_mutate_original_messages():
    """The caller's messages list must be untouched by the retry loop."""
    original = [LLMMessage(role="user", content="x")]
    client = _ScriptedClient(["bad", '{"ok": true}'])
    call_llm_with_structured_retry(
        client,
        original,
        parser=lambda raw: json.loads(raw),
        role="x",
        max_retries=1,
    )
    assert original == [LLMMessage(role="user", content="x")]
    assert len(original) == 1
