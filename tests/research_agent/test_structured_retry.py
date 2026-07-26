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
from typing import Literal

import pytest
from pydantic import BaseModel

from easyicu.research_agent.providers.llm import LLMMessage
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
    call_llm_with_structured_retry,
)


def test_structured_retry_returns_first_success_without_retrying():
    client = ScriptedMockLLMClient(['{"value": 42}'])
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
    client = ScriptedMockLLMClient([bad, good])

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
    retry_msgs = client.calls[1][0]
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
    client = ScriptedMockLLMClient(["bad-1", "bad-2", "bad-3"])
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
    client = ScriptedMockLLMClient(["nope"])
    with pytest.raises(StructuredResponseFailure):
        call_llm_with_structured_retry(
            client,
            [LLMMessage(role="user", content="x")],
            parser=lambda raw: json.loads(raw),
            role="solo",
            max_retries=0,
        )
    assert len(client.calls) == 1, "max_retries=0 should call exactly once"


def test_structured_retry_four_retries_means_five_total_attempts():
    client = ScriptedMockLLMClient(["bad-1", "bad-2", "bad-3", "bad-4", "bad-5"])
    with pytest.raises(StructuredResponseFailure):
        call_llm_with_structured_retry(
            client,
            [LLMMessage(role="user", content="x")],
            parser=lambda raw: json.loads(raw),
            role="planner",
            max_retries=4,
        )
    assert len(client.calls) == 5


def test_structured_retry_handles_value_error_from_parser():
    """Parser may raise any exception — wrapper catches them all and feeds back."""

    def picky_parser(raw: str) -> dict:
        data = json.loads(raw)  # may raise JSONDecodeError
        if "required_key" not in data:
            raise ValueError("missing required_key")
        return data

    client = ScriptedMockLLMClient(['{"other_key": 1}', '{"required_key": "ok"}'])
    out = call_llm_with_structured_retry(
        client,
        [LLMMessage(role="user", content="x")],
        parser=picky_parser,
        role="planner",
        max_retries=2,
    )
    assert out == {"required_key": "ok"}
    assert len(client.calls) == 2
    feedback = client.calls[1][0][-1].content
    assert "ValueError" in feedback
    assert "missing required_key" in feedback


def test_structured_retry_feedback_includes_validation_error_detail():
    """Schema errors fed to retry should include field and bad value details."""

    class Payload(BaseModel):
        concept_id: Literal["sofa"]

    client = ScriptedMockLLMClient(
        ['{"concept_id": "sofa2_admission"}', '{"concept_id": "sofa"}']
    )
    out = call_llm_with_structured_retry(
        client,
        [LLMMessage(role="user", content="give json")],
        parser=lambda raw: Payload.model_validate_json(raw),
        role="planner",
        max_retries=1,
    )
    assert out.concept_id == "sofa"
    assert len(client.calls) == 2
    feedback = client.calls[1][0][-1].content
    assert "ValidationError" in feedback
    assert "concept_id" in feedback
    assert "sofa2_admission" in feedback


def test_structured_retry_does_not_mutate_original_messages():
    """The caller's messages list must be untouched by the retry loop."""
    original = [LLMMessage(role="user", content="x")]
    client = ScriptedMockLLMClient(["bad", '{"ok": true}'])
    call_llm_with_structured_retry(
        client,
        original,
        parser=lambda raw: json.loads(raw),
        role="x",
        max_retries=1,
    )
    assert original == [LLMMessage(role="user", content="x")]
    assert len(original) == 1


def test_structured_retry_keeps_only_latest_failed_response() -> None:
    client = ScriptedMockLLMClient(["bad-1", "bad-2", '{"ok": true}'])

    result = call_llm_with_structured_retry(
        client,
        [LLMMessage(role="user", content="base")],
        parser=lambda raw: json.loads(raw),
        role="planner",
        max_retries=2,
    )

    assert result == {"ok": True}
    third_messages = client.calls[2][0]
    assert [message.role for message in third_messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert third_messages[-2].content == "bad-2"
    assert all(message.content != "bad-1" for message in third_messages)


def test_structured_retry_can_regenerate_without_replaying_large_failed_response() -> None:
    failed = '{"large_payload":"' + ("x" * 20_000) + '"}'
    client = ScriptedMockLLMClient([failed, '{"ok": true}'])

    def require_ok(raw: str) -> dict:
        parsed = json.loads(raw)
        if "ok" not in parsed:
            raise ValueError("replace the invalid field")
        return parsed

    result = call_llm_with_structured_retry(
        client,
        [LLMMessage(role="user", content="immutable base")],
        parser=require_ok,
        role="planner",
        max_retries=1,
        include_failed_response_on_retry=False,
    )

    assert result == {"ok": True}
    retry_messages = client.calls[1][0]
    assert [message.role for message in retry_messages] == ["user", "user"]
    assert retry_messages[0].content == "immutable base"
    assert failed not in {message.content for message in retry_messages}
    assert "replace the invalid field" in retry_messages[-1].content


def test_structured_retry_can_project_failed_response_for_bounded_memory() -> None:
    failed = '{"large_prose":"' + ("x" * 20_000) + '","keep":"coordinate"}'
    client = ScriptedMockLLMClient([failed, '{"ok": true}'])

    def require_ok(raw: str) -> dict:
        parsed = json.loads(raw)
        if "ok" not in parsed:
            raise ValueError("repair the coordinate")
        return parsed

    result = call_llm_with_structured_retry(
        client,
        [LLMMessage(role="user", content="immutable base")],
        parser=require_ok,
        role="planner",
        max_retries=1,
        failed_response_transform=lambda _raw: '{"keep":"coordinate"}',
    )

    assert result == {"ok": True}
    retry_messages = client.calls[1][0]
    assert retry_messages[-2].content == '{"keep":"coordinate"}'
    assert failed not in {message.content for message in retry_messages}
