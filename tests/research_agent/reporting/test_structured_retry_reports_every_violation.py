"""The retry must be told every violation, not the first one that fit.

A validation error prints roughly 240 characters per violation, a third of
it a documentation URL the model cannot visit. The feedback message used to
be a fixed 400-character slice of that prose, which states the first
violation and silently drops the rest.

A real Planner run recorded the consequence. Attempt 0 was rejected with 20
violations and told about one -- a missing field. Attempt 4 supplied that
field and died on six forbidden fields that had been present, and
unreported, since attempt 0. Five attempts were spent, the task never
planned, and every rejection was legible in full to the host.

The unit these tests group by is one rejection's *whole* violation set,
because that is the unit the defect spans: a check satisfied by "some
violation was mentioned" passes on the broken renderer.
"""

from __future__ import annotations

from typing import List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field

from easyicu.research_agent.providers.llm import LLMMessage
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
    call_llm_with_structured_retry,
    render_parse_failure,
)


class _ClosedSpec(BaseModel):
    """A spec that forbids unknown fields, like the Planner-owned specs."""

    model_config = ConfigDict(extra="forbid")

    n_resamples: int = Field(ge=2, le=500)
    sample_fraction: Optional[float] = None


# The exact payload the Planner sent in the recorded rejection: a
# representation-role design written into the stability spec.
_MISFILED_SPEC = {
    "time_anchor": "icu_admission",
    "grid_start_hours": 0.0,
    "grid_end_hours": 72.0,
    "grid_step_hours": 1.0,
    "features": ["sofa2_resp_max", "sofa2_coag_max", "lact_max"],
    "aggregation": "median",
}

_MISFILED_LOCATIONS = (
    "n_resamples",
    "time_anchor",
    "grid_start_hours",
    "grid_end_hours",
    "grid_step_hours",
    "features",
    "aggregation",
)


def _reject(payload: object) -> Exception:
    with pytest.raises(
        Exception
    ) as caught:  # noqa: PT011 — the exception is the subject
        _ClosedSpec.model_validate(payload)
    return caught.value


def test_every_violation_is_named_not_just_the_ones_that_fit_a_character_budget():
    rendered = render_parse_failure(_reject(_MISFILED_SPEC))
    missing = [name for name in _MISFILED_LOCATIONS if name not in rendered]
    assert not missing, (
        "the retry is told about a strict subset of what it was rejected for; "
        f"unreported: {missing}"
    )


def test_the_violation_count_is_stated_so_a_short_reply_is_visibly_incomplete():
    exc = _reject(_MISFILED_SPEC)
    assert str(len(exc.errors())) in render_parse_failure(exc)


def test_the_documentation_url_is_not_spent_on_a_link_the_model_cannot_open():
    assert "errors.pydantic.dev" not in render_parse_failure(_reject(_MISFILED_SPEC))


def test_a_capped_rendering_says_how_many_violations_it_withheld():
    """A cap that reads as completeness is how a partial report gets trusted."""

    exc = _reject({f"unknown_field_{index}": index for index in range(60)})
    rendered = render_parse_failure(exc, max_chars=400)
    assert "further problem(s) not listed" in rendered
    assert str(len(exc.errors())) in rendered


def test_a_validator_without_enumerable_violations_still_renders_its_message():
    assert "no such key" in render_parse_failure(KeyError("no such key"))


def test_the_input_echo_is_omitted_rather_than_clipped_into_a_wrong_value():
    """A missing field's reported input is the enclosing object, not a value."""

    rendered = render_parse_failure(_reject(_MISFILED_SPEC))
    for line in rendered.splitlines():
        if line.strip().startswith("- n_resamples:"):
            assert (
                "you sent" not in line
            ), "the whole payload was attributed to the one field absent from it"
            break
    else:  # pragma: no cover - the location is asserted above
        pytest.fail("n_resamples violation was not rendered at all")


def _feedback_of_failed_run(payloads: List[str], max_retries: int) -> List[str]:
    client = ScriptedMockLLMClient(payloads)
    with pytest.raises(StructuredResponseFailure):
        call_llm_with_structured_retry(
            client,
            [LLMMessage(role="user", content="declare the spec")],
            parser=lambda raw: _ClosedSpec.model_validate_json(raw),
            role="planner",
            max_retries=max_retries,
        )
    return [call[0][-1].content for call in client.calls[1:]]


def test_the_live_retry_receives_the_whole_violation_set():
    """End to end: the message the client is actually handed, not the helper."""

    import json

    feedback = _feedback_of_failed_run([json.dumps(_MISFILED_SPEC)] * 2, max_retries=1)
    assert feedback, "no retry was issued"
    missing = [name for name in _MISFILED_LOCATIONS if name not in feedback[0]]
    assert not missing, f"the live retry was not told about: {missing}"


def test_an_earlier_rejection_is_carried_forward_whole():
    """The carry-forward exists to state the whole constraint set at once."""

    import json

    second = {"n_resamples": 3, "sample_fraction": 0.5, "unexpected_extra": 1}
    feedback = _feedback_of_failed_run(
        [json.dumps(_MISFILED_SPEC), json.dumps(second), json.dumps(second)],
        max_retries=2,
    )
    assert len(feedback) >= 2, "expected a second retry carrying the earlier reason"
    carried = feedback[-1]
    missing = [name for name in _MISFILED_LOCATIONS if name not in carried]
    assert not missing, (
        "the earlier rejection was carried forward in part, so a response "
        f"satisfying it can still be rejected for: {missing}"
    )
