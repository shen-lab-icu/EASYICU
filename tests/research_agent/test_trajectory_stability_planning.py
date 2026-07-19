"""Planner ownership gates for the standard trajectory-stability executor."""

from __future__ import annotations

from easyicu.research_agent.pipeline_execute import _plan_signature
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    TrajectoryStabilitySpec,
)
from easyicu.research_agent.trajectory.plan_contract import (
    STABILITY_EXECUTOR_INPUTS,
    STABILITY_EXECUTOR_OUTPUTS,
)
from easyicu.research_agent.execution.runners.trajectory_stability_executor import (
    trajectory_stability_executor_owns_step,
)


def _stability_spec(*, base_seed: int = 1729) -> TrajectoryStabilitySpec:
    return TrajectoryStabilitySpec(
        resampling_method="subsample_without_replacement",
        n_resamples=3,
        sample_fraction=0.8,
        sample_fraction_rounding="floor",
        base_seed=base_seed,
        seed_derivation="numpy_seedsequence_spawn_uint32_v1",
        cross_resample_membership="distinct_membership_required",
        stability_metric="adjusted_rand_index",
        stability_aggregation="mean",
        metric_label_source="raw_refit_labels_label_invariant",
        evaluation_scope="sampled_overlap",
        label_alignment="hungarian_maximum_overlap",
        label_alignment_reference="frozen_candidate_assignments",
        label_alignment_tie_break=("minimum_rank_distance_then_lexicographic_v1"),
        final_assignment_policy="copy_selected_candidate_labels",
        minimum_successful_resamples=3,
        failed_refit_policy="record_once_no_retry",
        refit_engine="easyicu_observed_data_diag_gmm_v1",
        refit_initialization="random_balanced_assignments",
        refit_max_iter=60,
        refit_tolerance=1e-4,
        refit_regularization=1e-6,
        decision_mode="report_only",
        threshold_failure_action="fail_closed_require_planner_revision",
    )


def _candidate_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="candidate",
        intent="Compare the agent-planned candidate clustering solutions.",
        method="model_based_clustering",
        inputs=[
            "artifact:trajectory_representation",
            "manifest:trajectory_representation_schema",
        ],
        expected_outputs=[
            "artifact:candidate_cluster_models",
            "artifact:candidate_cluster_assignments",
            "manifest:cluster_selection",
            "manifest:candidate_cluster_solution_schema",
        ],
    )


def _stability_step(
    *,
    spec: TrajectoryStabilitySpec | None = None,
    exact_contract: bool = False,
    extra_outputs: tuple[str, ...] = (),
) -> AnalysisStep:
    inputs = (
        sorted(STABILITY_EXECUTOR_INPUTS)
        if exact_contract
        else [
            "artifact:trajectory_representation",
            "artifact:candidate_cluster_models",
            "artifact:candidate_cluster_assignments",
            "manifest:cluster_selection",
            "manifest:trajectory_representation_schema",
            "manifest:candidate_cluster_solution_schema",
        ]
    )
    outputs = (
        sorted(STABILITY_EXECUTOR_OUTPUTS)
        if exact_contract
        else [
            "artifact:stability_freeze",
            "artifact:cluster_assignments",
            "manifest:trajectory_missingness_policy",
            "table:cluster_assignments",
            "table:cluster_stability",
            "table:cluster_stability_assignments",
            "table:cluster_assignment_provenance",
        ]
    )
    return AnalysisStep(
        step_id="stability",
        intent="Assess stability and freeze only the selected candidate solution.",
        method=(
            "trajectory_cluster_stability"
            if spec is not None
            else "model_based_clustering_with_bootstrap_stability"
        ),
        inputs=inputs,
        expected_outputs=[*outputs, *extra_outputs],
        trajectory_stability_spec=spec,
    )


def _plan(
    *,
    spec: TrajectoryStabilitySpec | None = None,
    exact_contract: bool = False,
    extra_stability_outputs: tuple[str, ...] = (),
) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Discover and validate longitudinal ICU phenotypes.",
        analysis_type="trajectory_clustering",
        revision=1,
        steps=[
            _candidate_step(),
            _stability_step(
                spec=spec,
                exact_contract=exact_contract,
                extra_outputs=extra_stability_outputs,
            ),
        ],
    )


def test_standard_stability_route_requires_closed_independent_owner() -> None:
    valid = _plan(spec=_stability_spec(), exact_contract=True)
    assert trajectory_stability_executor_owns_step(valid.steps[1], plan=valid)

    no_spec = _plan(exact_contract=True)
    assert not trajectory_stability_executor_owns_step(no_spec.steps[1], plan=no_spec)

    extra_outcome = _plan(
        spec=_stability_spec(),
        exact_contract=True,
        extra_stability_outputs=("table:outcome_by_cluster",),
    )
    assert not trajectory_stability_executor_owns_step(
        extra_outcome.steps[1], plan=extra_outcome
    )

    wrong_method = _plan(spec=_stability_spec(), exact_contract=True)
    wrong_method.steps[1].method = "cluster_robust_regression"
    assert not trajectory_stability_executor_owns_step(
        wrong_method.steps[1], plan=wrong_method
    )

    monolithic_step = AnalysisStep(
        step_id="candidate_and_stability",
        intent="Select candidates and assess stability in one scientific step.",
        method="model_based_clustering_with_bootstrap_stability",
        inputs=sorted(STABILITY_EXECUTOR_INPUTS),
        expected_outputs=sorted(
            STABILITY_EXECUTOR_OUTPUTS
            | {
                "artifact:candidate_cluster_models",
                "artifact:candidate_cluster_assignments",
                "manifest:cluster_selection",
                "manifest:candidate_cluster_solution_schema",
            }
        ),
        trajectory_stability_spec=_stability_spec(),
    )
    monolithic = AnalysisPlan(
        research_question="Discover and validate longitudinal ICU phenotypes.",
        analysis_type="trajectory_clustering",
        steps=[monolithic_step],
    )
    assert not trajectory_stability_executor_owns_step(monolithic_step, plan=monolithic)


def test_plan_signature_changes_when_stability_spec_changes() -> None:
    first = _plan(spec=_stability_spec(base_seed=1729), exact_contract=True)
    same = _plan(spec=_stability_spec(base_seed=1729), exact_contract=True)
    changed = _plan(spec=_stability_spec(base_seed=1730), exact_contract=True)

    assert _plan_signature(first) == _plan_signature(same)
    assert _plan_signature(first) != _plan_signature(changed)
