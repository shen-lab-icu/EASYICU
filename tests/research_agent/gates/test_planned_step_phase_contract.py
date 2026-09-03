"""The planning layer owns which run phase a planned step belongs to.

Consumers used to re-derive study semantics from the free-text ``method`` and
``step_id``; where that failed, a reader fell back to canned prose that
asserted one case's exposure, outcome, and time zero for every plan. The
derivation therefore lives once, here, in the layer that owns plan semantics.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.step_phase import (
    PLANNED_STEP_PHASES,
    compile_plan_step_phases,
    compile_step_phase,
    design_analysis_family,
)


def test_declared_role_decides_whether_a_step_is_a_result() -> None:
    """`planned_analysis_role` is Planner-owned and the pipeline gates on it,
    so a method string can never overrule it in either direction."""

    # the method text reads like a data check, but the plan calls it the result
    assert (
        compile_step_phase(
            {
                "step_id": "q",
                "method": "table_one_primary_result",
                "planned_analysis_role": "primary",
            }
        )
        == "analysis"
    )
    # a secondary result is still a result, not a check
    assert (
        compile_step_phase(
            {
                "step_id": "s",
                "method": "absolute_risk_context",
                "planned_analysis_role": "secondary",
            }
        )
        == "analysis"
    )
    # a sensitivity step whose wording never says "sensitivity"
    assert (
        compile_step_phase(
            {
                "step_id": "z",
                "method": "alternative_exposure_coding",
                "planned_analysis_role": "sensitivity",
            }
        )
        == "robustness"
    )


def test_auxiliary_steps_are_never_promoted_into_results() -> None:
    """The role says this step carries no claim. Where no supporting shape
    matches, say exactly that rather than inferring a result from the method."""

    assert (
        compile_step_phase(
            {
                "step_id": "x",
                "method": "descriptive_counts_of_exposure_outcome",
                "planned_analysis_role": "auxiliary",
            }
        )
        == "support"
    )
    # supporting shapes the role is silent about are still resolved
    for method, expected in (
        ("cohort_definition_and_attrition", "cohort"),
        ("table_one", "data_check"),
        ("missing_data", "data_check"),
        ("visualization", "reporting"),
    ):
        assert (
            compile_step_phase(
                {
                    "step_id": "s",
                    "method": method,
                    "planned_analysis_role": "auxiliary",
                }
            )
            == expected
        ), method


def test_a_plan_that_declares_no_role_still_compiles_by_method() -> None:
    """Historical plans and host fixtures predate the declared role."""

    phases = compile_plan_step_phases(
        [
            {"step_id": "c", "method": "cohort_definition_and_attrition"},
            {"step_id": "m", "method": "missing_data"},
            {"step_id": "p", "method": "descriptive_counts"},
            {"step_id": "r", "method": "robustness_sensitivity"},
            {"step_id": "f", "method": "visualization"},
        ]
    )
    assert phases == ("cohort", "data_check", "analysis", "robustness", "reporting")


def test_figure_only_outputs_are_reporting_even_without_a_method() -> None:
    assert (
        compile_step_phase({"step_id": "f", "expected_outputs": ["figure:overview"]})
        == "reporting"
    )
    # a step that also produces a table is not merely a rendering step
    assert (
        compile_step_phase(
            {
                "step_id": "f",
                "expected_outputs": ["figure:overview", "table:estimates"],
            }
        )
        != "reporting"
    )


def test_compile_accepts_a_typed_step_as_well_as_a_mapping() -> None:
    """A consumer reading a persisted plan must not have to rebuild the whole
    typed plan just to read the phase."""

    from easyicu.research_agent.schema import AnalysisStep

    step = AnalysisStep(
        step_id="primary_association",
        intent="Estimate the prespecified adjusted association.",
        planned_analysis_role="primary",
        method="signed_landmark_restricted_cubic_spline",
    )
    assert compile_step_phase(step) == "analysis"
    assert compile_step_phase(step.model_dump(mode="json")) == "analysis"


def test_every_compiled_phase_is_a_declared_phase() -> None:
    shapes = [
        {},
        {"method": "unknown_widget"},
        {"planned_analysis_role": "auxiliary"},
        {"planned_analysis_role": "primary"},
        {"expected_outputs": ["figure:x"]},
        {"method": "cohort_definition_and_attrition"},
    ]
    for shape in shapes:
        assert compile_step_phase(shape) in PLANNED_STEP_PHASES, shape
    assert compile_plan_step_phases("not-a-list") == ()
    assert compile_plan_step_phases(None) == ()


def test_the_phase_contract_names_no_case_variable_or_database() -> None:
    """Prompt hygiene: this is shared, case-neutral policy."""

    from pathlib import Path

    import easyicu.research_agent.planning.step_phase as module

    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "lact",
        "sofa",
        "sepsis",
        "mimic",
        "eicu",
        "hirid",
        "amsterdam",
        "sicdb",
        "mortality",
    ):
        assert forbidden not in source, forbidden


def test_design_analysis_family_resolves_or_declines() -> None:
    assert design_analysis_family("association_study") == "association"
    assert design_analysis_family("not_a_real_analysis_type") is None
    assert design_analysis_family("") is None
    assert design_analysis_family(None) is None


@pytest.mark.parametrize("role", ["primary", "secondary", "sensitivity", "auxiliary"])
def test_declared_roles_are_all_accepted(role: str) -> None:
    """Every value of the Planner-owned enum must compile to a phase."""

    assert (
        compile_step_phase({"step_id": "s", "method": "x", "planned_analysis_role": role})
        in PLANNED_STEP_PHASES
    )
