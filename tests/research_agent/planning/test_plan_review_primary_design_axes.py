"""A signed primary's own design is executed, but it is not a robustness axis.

A landmark, an exposure spline or cluster-robust variance that the primary
itself runs is part of the primary analysis. Only a specification that varies
the primary -- here the linear refit of a spline primary -- is evidence that
the result is robust.
"""

from __future__ import annotations

from easyicu.research_agent.planning.scientific_review import _sensitivity_facts
from easyicu.research_agent.schema import AnalysisStep, UserPreferences

from .scientific_review_fixtures import _context, _plan

_LANDMARK = {
    "spec_id": "landmark_24h_primary",
    "axis": "timing",
    "strategy": "landmark",
    "landmark_hours": 24,
    "require_alive_at_landmark": True,
    "exclude_negative_event_times": True,
    "event_time_variable": "death_time",
    "observation_duration_variable": "los_icu",
    "observation_duration_unit": "days",
}
_SPLINE = {
    "spec_id": "peak_lactate_rcs_primary",
    "axis": "functional_form",
    "strategy": "restricted_cubic_spline",
    "execution_variables": ["exposure"],
}
_LINEAR = {
    "spec_id": "linear_per_unit_sensitivity",
    "axis": "functional_form",
    "strategy": "linear_per_unit",
    "execution_variables": ["exposure"],
}
_CLUSTERED = {
    "spec_id": "repeated_stays_cluster_robust",
    "axis": "repeated_stays",
    "strategy": "cluster_robust",
}


def _signed_primary_facts(specs: list[dict], **primary: object) -> dict:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                covariates=["age"], sensitivity_specs=specs
            )
        }
    )
    base = _plan()
    signed = next(
        step for step in base.steps if step.planned_analysis_role == "primary"
    ).model_copy(update={"model_requirements": [], "sensitivity_spec_ids": [], **primary})
    steps = [
        signed if step.planned_analysis_role == "primary" else step
        for step in base.steps
    ]
    if primary.get("method") == "signed_landmark_categorical_association":
        steps.insert(
            0,
            AnalysisStep(
                step_id="landmark_cohort",
                planned_analysis_role="auxiliary",
                intent="Build the signed landmark analysis cohort.",
                method="signed_landmark_analysis_cohort",
                inputs=["exposure", "death", "death_time", "los_icu", "age"],
                expected_outputs=["artifact:analysis_cohort"],
                icu_rule_refs=primary["icu_rule_refs"],
            ),
        )
    return _sensitivity_facts(context, base.model_copy(update={"steps": steps}))


def _spline_primary_facts(specs: list[dict]) -> dict:
    return _signed_primary_facts(
        specs,
        method="signed_landmark_restricted_cubic_spline",
        inputs=[
            "artifact:analysis_cohort",
            "exposure",
            "death",
            "death_time",
            "los_icu",
            "age",
        ],
        expected_outputs=[
            "table:landmark_rcs_curve",
            "table:landmark_rcs_contrasts",
            "table:landmark_linear_sensitivity",
        ],
    )


def test_a_spline_primary_varies_only_through_its_linear_refit() -> None:
    facts = _spline_primary_facts([_LANDMARK, _SPLINE, _LINEAR])
    assert facts["executed_spec_ids"] == [
        "landmark_24h_primary",
        "linear_per_unit_sensitivity",
        "peak_lactate_rcs_primary",
    ]
    assert facts["typed_executable"] == ["functional_form"]

    # Without the linear refit nothing varies the primary.
    assert _spline_primary_facts([_LANDMARK, _SPLINE])["typed_executable"] == []

    # Its cluster-robust variance is the primary's own inference as well.
    clustered = _spline_primary_facts([_LANDMARK, _SPLINE, _LINEAR, _CLUSTERED])
    assert "repeated_stays_cluster_robust" in clustered["executed_spec_ids"]
    assert clustered["typed_executable"] == ["functional_form"]


def test_a_categorical_landmark_primary_executes_its_landmark_without_a_timing_axis() -> None:
    contract = "scientific_runtime_contract:" + "a" * 64
    facts = _signed_primary_facts(
        [_LANDMARK],
        method="signed_landmark_categorical_association",
        inputs=["artifact:analysis_cohort", "exposure", "death", "age"],
        icu_rule_refs=[contract],
    )

    # The landmark obligation is met, but it is no robustness axis.
    assert facts["executed_spec_ids"] == ["landmark_24h_primary"]
    assert "timing" not in facts["typed_executable"]
