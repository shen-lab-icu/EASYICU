"""Authority rules for legacy findings that predate explicit step scope."""

from __future__ import annotations

import pytest

from easyicu.research_agent.pipeline_report import (
    _legacy_unscoped_finding_owner_step_id,
    _partition_findings_by_supersession,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ValidationFinding,
)


def _robustness_plan(*, duplicate_result_owner: bool = False) -> AnalysisPlan:
    steps = [
        AnalysisStep(
            step_id="07_sensitivity",
            intent="Execute the locked robustness variants.",
            method="cohort_definition_sensitivity",
            expected_outputs=["table:robustness_summary"],
        ),
        AnalysisStep(
            step_id="07_sensitivity_figure",
            intent="Render the locked robustness summary.",
            method="cohort_definition_sensitivity",
            expected_outputs=["figure:robustness_plot"],
        ),
    ]
    if duplicate_result_owner:
        steps.append(
            AnalysisStep(
                step_id="08_second_sensitivity",
                intent="Execute another locked robustness result.",
                method="cohort_definition_sensitivity",
                expected_outputs=["table:second_robustness_summary"],
            )
        )
    return AnalysisPlan(research_question="Test legacy authority.", steps=steps)


def _legacy_membership_error() -> ValidationFinding:
    return ValidationFinding(
        validator="robustness_cohort_membership",
        severity="error",
        message="Legacy membership output omitted replay fields.",
        detail={"issues": ["missing_membership_field"]},
    )


def test_unique_nonfigure_owner_supersedes_legacy_unscoped_finding() -> None:
    plan = _robustness_plan()
    finding = _legacy_membership_error()

    active, superseded = _partition_findings_by_supersession(
        [finding],
        success_step_ids={"07_sensitivity", "07_sensitivity_figure"},
        known_step_ids={"07_sensitivity", "07_sensitivity_figure"},
        plan=plan,
    )

    assert active == []
    assert superseded == [finding]


def test_legacy_owner_inference_fails_closed_when_result_owner_is_ambiguous() -> None:
    plan = _robustness_plan(duplicate_result_owner=True)
    finding = _legacy_membership_error()

    assert _legacy_unscoped_finding_owner_step_id(finding, plan=plan) is None
    active, superseded = _partition_findings_by_supersession(
        [finding],
        success_step_ids={
            "07_sensitivity",
            "07_sensitivity_figure",
            "08_second_sensitivity",
        },
        plan=plan,
    )

    assert active == [finding]
    assert superseded == []


def test_mixed_table_and_figure_step_is_not_a_legacy_result_owner() -> None:
    plan = AnalysisPlan(
        research_question="Test mixed ownership.",
        steps=[
            AnalysisStep(
                step_id="07_mixed_sensitivity",
                intent="Produce a mixed robustness bundle.",
                method="cohort_definition_sensitivity",
                expected_outputs=[
                    "table:robustness_summary",
                    "figure:robustness_plot",
                ],
            )
        ],
    )

    assert (
        _legacy_unscoped_finding_owner_step_id(
            _legacy_membership_error(),
            plan=plan,
        )
        is None
    )


def test_legacy_unscoped_finding_stays_active_until_owner_succeeds() -> None:
    plan = _robustness_plan()
    finding = _legacy_membership_error()

    active, superseded = _partition_findings_by_supersession(
        [finding],
        success_step_ids={"07_sensitivity_figure"},
        known_step_ids={"07_sensitivity", "07_sensitivity_figure"},
        plan=plan,
    )

    assert active == [finding]
    assert superseded == []


def test_legacy_inference_cannot_claim_replanned_away_supersession() -> None:
    plan = _robustness_plan()
    finding = _legacy_membership_error()

    active, superseded = _partition_findings_by_supersession(
        [finding],
        success_step_ids={"07_sensitivity_figure"},
        known_step_ids={"07_sensitivity_figure"},
        plan=plan,
    )

    assert active == [finding]
    assert superseded == []


@pytest.mark.parametrize("legacy_first", [True, False])
def test_current_scoped_error_blocks_legacy_retirement_regardless_of_order(
    legacy_first: bool,
) -> None:
    plan = _robustness_plan()
    legacy = _legacy_membership_error()
    current = ValidationFinding(
        validator="robustness_cohort_membership",
        severity="error",
        message="Current owner still fails membership replay.",
        detail={"step_id": "08_current"},
    )
    findings = [legacy, current] if legacy_first else [current, legacy]

    active, superseded = _partition_findings_by_supersession(
        findings,
        success_step_ids={"07_sensitivity"},
        known_step_ids={"07_sensitivity", "08_current"},
        plan=plan,
    )

    assert active == findings
    assert superseded == []


def test_run_level_robustness_finding_is_never_inferred_as_step_owned() -> None:
    plan = _robustness_plan()
    finding = ValidationFinding(
        validator="robustness_panel",
        severity="error",
        message="Current run-level panel is incomplete.",
    )

    active, superseded = _partition_findings_by_supersession(
        [finding],
        success_step_ids={"07_sensitivity", "07_sensitivity_figure"},
        plan=plan,
    )

    assert active == [finding]
    assert superseded == []
