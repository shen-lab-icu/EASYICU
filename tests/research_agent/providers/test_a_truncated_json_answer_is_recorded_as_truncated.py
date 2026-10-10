"""A structured answer that stops before its JSON closes is recorded as truncated.

A real Planner call returned twice a response the decoder rejected at its
last character, an object still open: the response had stopped before its
JSON closed.  The record said only ``json_shape``, the stage any malformed
answer shares, and kept no text, so a truncated answer could not be told
from one of the wrong shape.

Each attempt that fails to decode now records where its JSON stood at its
end -- how many objects and arrays were still open, and whether it ended
inside a string -- and a response that ended inside an open value carries
the issue ``json_truncated``.  Neither says anything of what the response
wrote, and neither changes the retry or what it is told.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.providers.structured_diagnostics import (
    json_eof_state,
    json_truncation_issues,
    safe_json_eof_state,
    safe_projected_validation_issues,
)
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
    call_llm_with_structured_retry,
    safe_structured_attempt_metadata,
)

_TRUNCATED = [{"location": ["<root>"], "issue_type": "json_truncated"}]


def _decode_failure(raw: str) -> json.JSONDecodeError:
    with pytest.raises(json.JSONDecodeError) as failed:
        json.loads(raw.strip())
    return failed.value


@pytest.mark.parametrize(
    ("raw", "state"),
    [
        ('{"a": {"b": 1}', {"open_depth": 1, "in_string": False}),
        ('  {"a": [1, 2', {"open_depth": 2, "in_string": False}),
        ('{"a": "x', {"open_depth": 1, "in_string": True}),
        ('{"a": "} ] {"', {"open_depth": 1, "in_string": False}),
        ('{"a": "\\"}', {"open_depth": 1, "in_string": True}),
        ('{"a": "\\\\"}', {"open_depth": 0, "in_string": False}),
        ('{"a": 1}', {"open_depth": 0, "in_string": False}),
        ("", {"open_depth": 0, "in_string": False}),
        ("[" * 80, {"open_depth": 64, "in_string": False}),
    ],
)
def test_the_end_state_counts_open_values_outside_strings(
    raw: str, state: dict
) -> None:
    assert json_eof_state(raw) == state


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version": "x", "foundation": {"cohort": {}}',
        '{"a": [1, 2',
        '{"a": "an unfinished sentence',
        '{"a": tru',
        '{"a": [fals',
        '{"a": nul',
    ],
)
def test_a_response_that_ends_inside_an_open_value_is_truncated(raw: str) -> None:
    assert json_truncation_issues(_decode_failure(raw), raw) == _TRUNCATED


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "Here is the plan:",
        '{"a": 1} trailing words',
        '{"a": 1,}',
        '{"a": x}',
        '{"a": x',
        '{"a": "\\\\"}"',
        "tru",
    ],
)
def test_a_response_of_the_wrong_shape_is_not_truncated(raw: str) -> None:
    assert json_truncation_issues(_decode_failure(raw), raw) == []


def test_a_failure_that_is_no_decode_failure_is_not_truncated() -> None:
    assert json_truncation_issues(ValueError("too many primary steps"), '{"a": 1') == []


@pytest.mark.parametrize(
    "value",
    [
        {"open_depth": True, "in_string": False},
        {"open_depth": -1, "in_string": False},
        {"open_depth": 65, "in_string": False},
        {"open_depth": 1, "in_string": "no"},
        {"open_depth": 1, "in_string": False, "text": "x"},
        {"open_depth": 1},
        [1, False],
    ],
)
def test_a_recorded_end_state_is_revalidated(value: Any) -> None:
    assert safe_json_eof_state(value) is None
    assert safe_json_eof_state({"open_depth": 3, "in_string": True}) == {
        "open_depth": 3,
        "in_string": True,
    }


def test_the_issue_survives_its_revalidation() -> None:
    assert safe_projected_validation_issues(_TRUNCATED) == _TRUNCATED


def _parse(raw: str) -> dict:
    payload = json.loads(raw.strip())
    if payload.get("steps") != 1:
        raise ValueError("a plan has exactly one step")
    return payload


def _run(responses: list[str]) -> tuple[list, list]:
    events: list = []
    with pytest.raises(StructuredResponseFailure) as failed:
        call_llm_with_structured_retry(
            llm=ScriptedMockLLMClient(responses),
            messages=[LLMMessage(role="user", content="plan it")],
            parser=_parse,
            role="planner",
            max_retries=1,
            progress_callback=events.append,
        )
    return safe_structured_attempt_metadata(failed.value.attempts), events


def test_each_failed_attempt_records_its_end_state_and_no_text() -> None:
    secret = "a sentence the record must never hold"
    rows, events = _run(
        [
            '{"steps": 1, "note": "' + secret + '", "cohort": {"rule": 1}',
            '{"steps": 1, "note": "' + secret,
        ]
    )

    assert [row["attempt"] for row in rows] == [1, 2]
    assert [row["json_eof"] for row in rows] == [
        {"open_depth": 1, "in_string": False},
        {"open_depth": 1, "in_string": True},
    ]
    assert [row["validation_issues"] for row in rows] == [_TRUNCATED, _TRUNCATED]
    assert all(row["validation_stage"] == "json_shape" for row in rows)
    assert secret not in json.dumps(rows)
    rejected = [event for event in events if event.phase == "rejected"]
    assert [list(event.validation_issues) for event in rejected] == [
        _TRUNCATED,
        _TRUNCATED,
    ]


def test_an_answer_of_the_wrong_shape_records_no_truncation() -> None:
    rows, _events = _run(['{"steps": 1} and more', '{"steps": 2}'])

    assert rows[0]["json_eof"] == {"open_depth": 0, "in_string": False}
    assert "validation_issues" not in rows[0]
    assert "json_eof" not in rows[1]
    assert "validation_issues" not in rows[1]
    assert len(rows) == 2
