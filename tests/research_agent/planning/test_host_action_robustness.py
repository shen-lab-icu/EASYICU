"""Host actions with a published robustness design are credited only in their claimed shape."""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.host_action_robustness import (
    HOST_ACTION_ROBUSTNESS_AXES,
    host_action_prespecified_axes,
)
from easyicu.research_agent.contracts.phenotyping_execution import (
    cross_sectional_phenotyping_owns_step,
)
from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_executor import (
    cross_sectional_phenotyping_executor_owns_step,
)
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.research_agent.schema import AnalysisStep


def _primary(**updates) -> AnalysisStep:
    return AnalysisStep(
        step_id="primary_cluster_solution",
        planned_analysis_role="primary",
        intent="Fit the prespecified cross-sectional clusters.",
        inputs=["hr_max", "lactate_max", "map_min", "artifact:analysis_cohort"],
        expected_outputs=["table:phenotype_profiles", "table:phenotype_assignments"],
        method="cross_sectional_phenotyping",
        scientific_action_id="phenotyping.cluster_solution",
        phenotyping_feature_columns=["hr_max", "lactate_max", "map_min"],
    ).model_copy(update=updates)


def _downstream(action: str, product: str, **updates) -> AnalysisStep:
    return AnalysisStep(
        step_id=action.split(".", 1)[1],
        planned_analysis_role="sensitivity",
        intent="Replay the published host design on the sealed primary matrix.",
        inputs=["table:phenotype_assignments"],
        expected_outputs=[product],
        method=action,
        scientific_action_id=action,
    ).model_copy(update=updates)


def _selection(**updates) -> AnalysisStep:
    return _downstream("phenotyping.k_selection", "table:cluster_selection", **updates)


def _stability(**updates) -> AnalysisStep:
    return _downstream("phenotyping.cluster_stability", "table:cluster_stability", **updates)


def test_executor_claim_and_review_credit_ask_the_same_owner() -> None:
    steps = [
        _primary(),
        _selection(),
        _stability(),
        _primary(planned_analysis_role="secondary"),
        _selection(inputs=["artifact:analysis_cohort"]),
        _stability(expected_outputs=["table:cluster_stability", "table:extra"]),
        _selection(planned_analysis_role="primary"),
    ]
    assert [cross_sectional_phenotyping_owns_step(step) for step in steps] == [
        True, True, True, False, False, False, False,
    ]
    assert [cross_sectional_phenotyping_executor_owns_step(step) for step in steps] == [
        cross_sectional_phenotyping_owns_step(step) for step in steps
    ]


def test_host_owned_selection_and_stability_contribute_two_axes() -> None:
    assert host_action_prespecified_axes([_primary(), _selection(), _stability()]) == (
        "model_specification",
        "resampling_stability",
    )
    assert host_action_prespecified_axes([_primary(), _stability()]) == ("resampling_stability",)


@pytest.mark.parametrize(
    "downstream",
    [
        # Reopening cohort bytes is not the replay of the sealed primary matrix.
        _selection(inputs=["table:phenotype_assignments", "artifact:analysis_cohort"]),
        # Wrong product: the owner writes exactly one table for this action.
        _selection(expected_outputs=["table:k_curve"]),
        # A model requirement riding on the step takes it away from the owner.
        _stability(model_requirements=["adjusted_model"]),
    ],
)
def test_a_step_the_owner_would_not_claim_earns_nothing(downstream) -> None:
    assert not cross_sectional_phenotyping_owns_step(downstream)
    assert host_action_prespecified_axes([_primary(), downstream]) == ()


def test_the_action_id_alone_is_never_enough() -> None:
    # Same action ids, but no host-owned primary for them to replay.
    assert host_action_prespecified_axes([_selection(), _stability()]) == ()
    agent_primary = _primary(scientific_action_id=None, method="kmeans_custom")
    assert host_action_prespecified_axes([agent_primary, _selection(), _stability()]) == ()
    # Two host primaries leave the replayed matrix ambiguous.
    second = _primary(step_id="second_solution")
    assert host_action_prespecified_axes([_primary(), second, _selection(), _stability()]) == ()


def test_host_action_axes_are_review_vocabulary_not_user_declarable() -> None:
    declarable = set(
        PrespecifiedSensitivitySpec.model_fields["axis"].annotation.__args__
    )
    for axes in HOST_ACTION_ROBUSTNESS_AXES.values():
        assert not set(axes) & declarable


# --- static prediction -------------------------------------------------------

from easyicu.research_agent.contracts.prediction_execution import (  # noqa: E402
    static_prediction_executes_robustness_spec,
    static_prediction_features,
    static_prediction_owns_step,
)
from easyicu.research_agent.execution.runners.prediction_model_executor import (  # noqa: E402
    prediction_model_executor_owns_step,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec  # noqa: E402


def _prediction_primary(**updates) -> AnalysisStep:
    return AnalysisStep(
        step_id="primary_performance",
        planned_analysis_role="primary",
        intent="Fit the prespecified static model and report held-out performance.",
        inputs=["death", "age", "lactate", "artifact:analysis_cohort"],
        expected_outputs=["table:prediction_scores", "table:model_performance"],
        method="prespecified_prediction_model_discrimination_calibration",
        scientific_action_id="prediction.discrimination_calibration",
    ).model_copy(update=updates)


def _decision_curve(**updates) -> AnalysisStep:
    return AnalysisStep(
        step_id="clinical_utility",
        planned_analysis_role="secondary",
        intent="Net benefit across the host threshold grid on held-out predictions.",
        inputs=["table:prediction_scores"],
        expected_outputs=["table:clinical_utility"],
        method="prespecified_decision_curve_analysis",
        scientific_action_id="prediction.decision_curve",
    ).model_copy(update=updates)


def test_prediction_claim_is_shared_by_executor_and_reviewer() -> None:
    steps = [
        _prediction_primary(),
        _decision_curve(),
        _decision_curve(planned_analysis_role="sensitivity"),
        _decision_curve(inputs=["artifact:analysis_cohort"]),
        _decision_curve(expected_outputs=["table:net_benefit"]),
    ]
    assert [static_prediction_owns_step(step) for step in steps] == [True, True, False, False, False]
    assert [prediction_model_executor_owns_step(step) for step in steps] == [
        static_prediction_owns_step(step) for step in steps
    ]


def test_decision_curve_axis_needs_the_host_owned_primary() -> None:
    assert host_action_prespecified_axes([_prediction_primary(), _decision_curve()]) == (
        "decision_threshold",
    )
    assert host_action_prespecified_axes([_decision_curve()]) == ()
    assert host_action_prespecified_axes(
        [_prediction_primary(inputs=["artifact:analysis_cohort"]), _decision_curve()]
    ) == ()


def _complete_case(variables, **updates) -> RobustnessSpec:
    fields = {
        "spec_id": "complete_case_model_roster",
        "axis": "missing",
        "description": "Complete-case refit of the exact model roster.",
        "missing_override": {"strategy": "complete_case", "variables": list(variables)},
        **updates,
    }
    return RobustnessSpec(**fields)


def test_owner_executes_only_the_complete_case_refit_of_its_exact_roster() -> None:
    features = static_prediction_features(
        ("death", "age", "lactate", "stay_group"), outcome="death", group_source="stay_group"
    )
    assert features == ("age", "lactate")
    run = lambda spec: static_prediction_executes_robustness_spec(  # noqa: E731
        spec, features=features, outcome="death"
    )
    assert run(_complete_case(["lactate", "death", "age"]))
    assert not run(_complete_case(["death", "age"]))  # narrower set: another analysis
    assert not run(_complete_case(["death", "age", "lactate", "sex"]))  # wider set
    assert not run(_complete_case(["death", "age", "lactate", "age"]))  # repeated variable
    assert not run(
        _complete_case(["death", "age", "lactate"], outcome_override={"concept_id": "mort_28d"})
    )
    assert not run(
        _complete_case(
            ["death", "age", "lactate"],
            missing_override={"strategy": "multiple_imputation", "variables": ["death", "age", "lactate"]},
        )
    )
