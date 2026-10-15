"""A response that stopped inside its open JSON is answered by the same request, once.

Three real Planner calls returned about 950 tokens of JSON that stopped with
one object open, finish "stop", far below the output limit: the relay had
lost the response's end.  Each spent one of the call's contract attempts,
and one call then ran out of attempts on its next, ordinary rejection.

Such a response is now answered once per call by the same request, outside
the attempt budget.  The cut response is never sent back, completed or
repaired, and the model is told nothing about it.  A second truncated
response in the same call, a response cut at the output limit, and every
other rejection spend the budget as before.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
    call_llm_with_structured_retry,
    safe_structured_attempt_metadata,
)

TRUNCATED_RESPONSE_RESEND_REASON = "json_truncated_transport_retry"

_CUT = '{"steps": 1, "cohort": {"rule": "adults"}'
_CUT_IN_STRING = '{"steps": 1, "cohort": {"rule": "adu'
_OK = '{"steps": 1, "cohort": {"rule": "adults"}}'
_TRUNCATED = [{"location": ["<root>"], "issue_type": "json_truncated"}]
_BASE = [LLMMessage(role="user", content="plan it")]


def _parse(raw: str) -> dict:
    payload = json.loads(raw.strip())
    if payload.get("steps") != 1:
        raise ValueError("a plan has exactly one step")
    return payload


def _call(client: ScriptedMockLLMClient, *, max_retries: int, events: list):
    return call_llm_with_structured_retry(
        llm=client,
        messages=_BASE,
        parser=_parse,
        role="planner",
        max_retries=max_retries,
        progress_callback=events.append,
    )


def _failed_rows(
    responses: list[str], *, max_retries: int, finish: str | None = None
) -> tuple[list, ScriptedMockLLMClient]:
    client = ScriptedMockLLMClient(responses)
    if finish is not None:
        client.last_finish_reason = finish
    with pytest.raises(StructuredResponseFailure) as failed:
        _call(client, max_retries=max_retries, events=[])
    return safe_structured_attempt_metadata(failed.value.attempts), client


@pytest.mark.parametrize("cut", [_CUT, _CUT_IN_STRING])
def test_a_cut_response_is_answered_by_the_same_request(cut: str) -> None:
    client = ScriptedMockLLMClient([cut, _OK])
    events: list = []

    assert _call(client, max_retries=0, events=events) == json.loads(_OK)

    assert [messages for messages, _kwargs in client.calls] == [_BASE, _BASE]
    assert [(e.phase, e.attempt, e.total_attempts) for e in events] == [
        ("started", 1, 1),
        ("rejected", 1, 1),
        ("started", 2, 2),
        ("accepted", 2, 2),
    ]
    assert events[1].reason_code == TRUNCATED_RESPONSE_RESEND_REASON
    assert list(events[1].validation_issues) == _TRUNCATED


def test_the_resent_attempt_is_its_own_row() -> None:
    rows, client = _failed_rows([_CUT, '{"steps": 2}'], max_retries=0)

    assert len(client.calls) == 2
    assert [row["attempt"] for row in rows] == [1, 2]
    assert rows[0]["reason_code"] == TRUNCATED_RESPONSE_RESEND_REASON
    assert rows[0]["validation_issues"] == _TRUNCATED
    assert rows[0]["json_eof"] == {"open_depth": 1, "in_string": False}
    assert "reason_code" not in rows[1]
    assert "validation_issues" not in rows[1]


def test_a_second_cut_response_spends_the_budget() -> None:
    rows, client = _failed_rows([_CUT, _CUT], max_retries=0)

    assert len(client.calls) == 2
    assert rows[0]["reason_code"] == TRUNCATED_RESPONSE_RESEND_REASON
    assert "reason_code" not in rows[1]
    assert rows[1]["validation_issues"] == _TRUNCATED

    client = ScriptedMockLLMClient([_CUT, _CUT, _OK])
    assert _call(client, max_retries=1, events=[]) == json.loads(_OK)
    first, second, third = (messages for messages, _kwargs in client.calls)
    assert first == second == _BASE
    assert third[: len(_BASE)] == _BASE
    assert third[len(_BASE)] == LLMMessage(role="assistant", content=_CUT)
    assert "could not be parsed" in third[-1].content


@pytest.mark.parametrize(
    "response",
    ['{"steps": 2}', '{"steps": 1} and more', "Here is the plan:", '{"steps": x'],
)
def test_any_other_rejection_spends_the_budget(response: str) -> None:
    rows, client = _failed_rows([response, _OK], max_retries=0)

    assert len(client.calls) == 1
    assert [row["attempt"] for row in rows] == [1]
    assert "reason_code" not in rows[0]


def test_a_response_cut_at_the_output_limit_spends_the_budget() -> None:
    rows, client = _failed_rows([_CUT, _OK], max_retries=0, finish="length")

    assert len(client.calls) == 1
    assert rows[0]["finish_reason"] == "length"
    assert rows[0]["validation_issues"] == _TRUNCATED
    assert "reason_code" not in rows[0]


def test_the_model_is_never_told_of_the_cut_response() -> None:
    client = ScriptedMockLLMClient([_CUT, '{"steps": 2}', _OK])

    assert _call(client, max_retries=1, events=[]) == json.loads(_OK)

    sent = [messages for messages, _kwargs in client.calls]
    assert sent[0] == sent[1] == _BASE
    assert all(_CUT not in message.content for messages in sent for message in messages)
    feedback = sent[2][-1].content
    assert "a plan has exactly one step" in feedback
    assert "JSONDecodeError" not in feedback
    assert "Earlier attempts" not in feedback


def test_a_resend_after_contract_feedback_preserves_the_entire_request() -> None:
    from easyicu.research_agent.providers.protocol import StructuredOutputRequest

    schema = StructuredOutputRequest.from_schema(name="plan", schema={"type": "object"})
    client = ScriptedMockLLMClient(['{"steps": 2}', _CUT, _OK])
    assert call_llm_with_structured_retry(
        client,
        _BASE,
        _parse,
        max_retries=1,
        max_tokens=777,
        temperature=0.0,
        structured_output=schema,
    ) == json.loads(_OK)
    assert len(client.calls) == 3
    assert client.calls[1] == client.calls[2]
    assert client.calls[0][1] == {
        "max_tokens": 777,
        "temperature": 0.0,
        "structured_output": schema,
    }


def test_host_control_can_stop_before_the_resend() -> None:
    from easyicu.research_agent.contracts.control_signals import ProgressControlSignal

    client = ScriptedMockLLMClient([_CUT, _OK])
    signal = ProgressControlSignal("cancelled by host")

    def stop(event):
        if event.phase == "rejected":
            raise signal

    with pytest.raises(ProgressControlSignal) as raised:
        call_llm_with_structured_retry(
            client, _BASE, _parse, max_retries=0, progress_callback=stop
        )
    assert raised.value is signal
    assert len(client.calls) == 1


def test_the_resend_uses_authorized_complete_and_preserves_a_budget_stop(
    monkeypatch,
) -> None:
    from easyicu.research_agent.contracts.control_signals import ProgressControlSignal
    from easyicu.research_agent.providers import structured_retry as owner

    client = ScriptedMockLLMClient([_CUT, _OK])
    authorized = owner.authorized_complete
    signal = ProgressControlSignal("provider budget exhausted")
    calls = []

    def guarded(llm, messages, **kwargs):
        calls.append((list(messages), kwargs))
        if len(calls) == 2:
            raise signal
        return authorized(llm, messages, **kwargs)

    monkeypatch.setattr(owner, "authorized_complete", guarded)
    with pytest.raises(ProgressControlSignal) as raised:
        _call(client, max_retries=0, events=[])
    assert raised.value is signal
    assert len(calls) == 2 and calls[0] == calls[1]
    rows = signal.easyicu_structured_attempt_metadata
    assert [row["attempt"] for row in rows] == [1, 2]
    assert rows[0]["reason_code"] == TRUNCATED_RESPONSE_RESEND_REASON
    assert rows[1]["error_class"] == "error"  # private exception names stay redacted
    assert len(client.calls) == 1
