"""Focused fail-closed checks for the pre-Provider clinical-time owner."""

from __future__ import annotations

from easyicu.research_agent.gates.preplan import (
    clinical_time_authority_findings,
    preplan_data_failure_reason,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
)


def _context(*, materialized_window: str) -> ResearchContext:
    return ResearchContext(
        research_question="Assess an exposure relative to a clinical event.",
        cohort=CohortDescriptor(
            cohort_name="ICU stays", database="synthetic", n_stays=25
        ),
        variables=[
            ConceptDescriptor(
                name="x",
                role="other",
                dtype="int64",
                analysis_window=materialized_window,
                analysis_window_role="exposure_definition",
            )
        ],
        primary_exposure="x",
        user_preferences=UserPreferences(
            timing_and_design='{"anchor":"event onset"}'
        ),
    )


def test_anchor_mismatch_is_a_structured_pre_provider_error() -> None:
    findings = clinical_time_authority_findings(
        _context(materialized_window="icu_admission[0,24]h")
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.validator == "clinical_time_authority_gate"
    assert finding.severity == "error"
    assert finding.detail == {
        "kind": "primary_exposure_time_anchor_mismatch",
        "status": "mismatch",
        "primary_exposure": "x",
        "declared_anchor": "event_onset",
        "definition_anchor": "icu_admission",
        "observation_window_anchor": "icu_admission",
        "observation_window_role": "exposure_definition",
        "declared_source": "user_preferences.timing_and_design.anchor",
        "definition_source": "variables.x.analysis_window",
        "observation_window_source": "variables.x.analysis_window",
        "required_action": (
            "create_new_study_or_materialization_authority_with_matching_anchor"
        ),
        "provider_called": False,
    }
    assert preplan_data_failure_reason(findings) == "clinical_time_authority_failed"


def test_matching_anchor_does_not_create_a_finding() -> None:
    assert (
        clinical_time_authority_findings(
            _context(materialized_window="event_onset[0,24]h")
        )
        == []
    )
