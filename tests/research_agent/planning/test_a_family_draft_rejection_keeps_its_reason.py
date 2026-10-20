"""Family draft diagnostics keep typed reasons and schema paths, never values."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.agents import progressive_planner
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
)
from easyicu.research_agent.orchestration.progress import (
    ResumableProgressChannel,
    planner_retry_progress_callback,
)
from easyicu.research_agent.planning.family_spec import FamilySpecError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
    call_llm_with_structured_retry,
)
from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _context,
    _request,
    _spec_payload,
)


def test_a_family_draft_records_the_same_reason_in_attempts_and_the_rejected_audit(monkeypatch):
    monkeypatch.setattr(progressive_planner, "_MAX_STEP_PARSE_RETRIES", 0)
    context = _context(exact=True)
    payload = _spec_payload(_request(context))
    payload["request_sha256"] = "0" * 64
    llm = ScriptedMockLLMClient([json.dumps(payload)])
    audit = []
    channel = ResumableProgressChannel()
    channel.bind_audit_logger(SimpleNamespace(emit=lambda **event: audit.append(event)))
    with pytest.raises(StructuredResponseFailure) as stopped:
        ProgressivePlannerAgent(llm).run_attempt(
            context, planner_strategy=FAMILY_SPEC_STRATEGY,
            allowed_literature_citation_keys=ALLOWED_CITATIONS,
            direct_comparator_literature_keys=DIRECT_COMPARATORS,
            comparison_literature_keys=DIRECT_COMPARATORS,
            required_primary_cohort_selection_mode="predicate_filtered",
            progress_callback=planner_retry_progress_callback(channel.emit, run_id="synthetic"),
        )
    cause = stopped.value.__cause__
    assert isinstance(cause, FamilySpecError)
    record = stopped.value.easyicu_structured_attempt_metadata[0]
    rejected = [event["detail"] for event in audit if event["detail"]["retry_phase"] == "rejected"]
    assert len(rejected) == 1
    for diagnostic in [record, rejected[0]]:
        assert diagnostic["reason_code"] == cause.reason_code == "family_spec_request_digest_mismatch"
        assert diagnostic["validation_issues"] == [
            {"location": ["request_sha256"], "issue_type": "value_error"}
        ]
        assert str(cause) not in json.dumps(diagnostic)
        assert payload["request_sha256"] not in json.dumps(diagnostic)
        assert "roster_decision_note" not in json.dumps(diagnostic)


@pytest.mark.parametrize(
    ("path", "location"),
    [
        ("adjustment_set[1].coding", ["adjustment_set", 1, "coding"]),
        ("reader_display_labels.private_patient_value", ["reader_display_labels", "<other>"]),
        ("", None),
    ],
)
def test_typed_path_projection_redacts_values_and_preserves_retry_wording(path, location):
    error = FamilySpecError("family_spec_synthetic_refusal", "private_patient_value", path=path)
    original = "family_spec_synthetic_refusal: private_patient_value"
    assert str(error) == original
    llm = ScriptedMockLLMClient(["{}", "{}"])
    events = []

    def parser(raw):
        raise error

    with pytest.raises(StructuredResponseFailure) as stopped:
        call_llm_with_structured_retry(
            llm, [], parser=parser, role="family_spec_planner", max_retries=1,
            progress_callback=events.append,
        )
    for diagnostic in [*stopped.value.attempts, *[e for e in events if e.phase == "rejected"]]:
        assert diagnostic.reason_code == error.reason_code
        assert "private_patient_value" not in json.dumps(diagnostic.validation_issues)
    for record in stopped.value.easyicu_structured_attempt_metadata:
        assert record["reason_code"] == error.reason_code
        if location is not None:
            assert record["validation_issues"] == [{"location": location, "issue_type": "value_error"}]
        else:
            assert not record.get("validation_issues")
        assert "private_patient_value" not in json.dumps(record)
    # Diagnostics must not change the feedback the Planner already sees.
    assert original in str(llm.calls[1])
