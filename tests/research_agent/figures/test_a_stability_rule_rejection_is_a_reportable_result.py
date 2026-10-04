"""A stability rule that rejects the selected solution is a reportable result.

The stability owner froze the selected class solution only when every planned
subsample refit reproduced it and their mean agreement reached the planner's
minimum.  When the rule rejected the solution, the owner reported an
execution failure: the run stopped before the class description, the figure
and the manuscript.  Because every planned refit must succeed, one refit that
did not converge was enough.  A rejection by the prespecified rule is the
study's result, so the owner now completes, freezes nothing, leaves every
class-describing table empty, and states the rule's outcome.  A refit that
failed for any reason other than the model itself is still an engine failure.
Synthetic, opaque bundles only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.prespecified_rule_outcomes import (
    validate_rule_outcome,
)
from easyicu.research_agent.execution.runners import (
    trajectory_stability_executor as owner,
)
from easyicu.research_agent.execution.runners.trajectory_stability_executor import (
    run_trajectory_stability,
)
from easyicu.research_agent.trajectory.mixed_mode_latent_class import (
    ClassModelFitNotRealized,
    fit_observed_data_mixed_mode_lca,
)
from easyicu.research_agent.trajectory.plan_contract import (
    DIAG_GMM_BEST_OF_10_ENGINE,
)

from tests.support.trajectory_stability_bundle import (
    stability_spec,
    write_upstream_bundle,
)

_CLASS_TABLES = ("cluster_assignments.csv", "cluster_assignment_provenance.csv")
_CHARACTERIZATION_TABLES = ("trajectory_profiles.csv", "cluster_sizes.csv")


def _bundle(tmp_path: Path, n_clusters: int = 3):
    return write_upstream_bundle(
        tmp_path,
        n_clusters=n_clusters,
        id_column="opaque_unit",
        representation_columns=("coordinate_a", "coordinate_b", "coordinate_c"),
        assignment_column="candidate_label",
    )


def _unstable_refit(x, *, n_components, seed, max_iter, tolerance, regularization):
    del seed, max_iter, tolerance, regularization
    labels = np.arange(len(x), dtype=int) % n_components
    return labels, {
        "converged": True,
        "n_iter": 1,
        "final_log_likelihood": 0.0,
        "parameter_sha256": "synthetic-instability-probe",
    }


def _outcome(summary):
    [payload] = summary["reportable_rule_outcomes"]
    return validate_rule_outcome(payload)


def _rows(path: Path) -> int:
    return len(pd.read_csv(path))


@pytest.mark.parametrize("include_characterization", [False, True])
def test_a_mean_below_the_minimum_is_a_reported_no_solution(
    tmp_path: Path, monkeypatch, include_characterization: bool
) -> None:
    resolved, _representation, _assignments = _bundle(tmp_path)
    monkeypatch.setattr(owner, "_fit_observed_data_diag_gmm", _unstable_refit)
    out_dir = tmp_path / "step_outputs"

    summary = run_trajectory_stability(
        spec=stability_spec(
            decision_mode="minimum_mean_threshold", minimum_mean_stability=0.99
        ),
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs=resolved,
        include_characterization=include_characterization,
    )

    assert summary["status"] == "ok"
    assert summary["errors"] == [] and summary["failure_class"] is None
    assert summary["scientific_status"] == "failed_closed"
    assert summary["reason_code"] == "TRAJECTORY_STABILITY_BELOW_THRESHOLD"
    assert summary["freeze_status"] == "not_frozen_stability_threshold_failed"
    assert summary["reportable_result"] == "no_stable_phenotype_solution"
    # The rule does not change the selected class count.
    assert summary["selected_n_clusters"] == 3
    assert summary["stability_threshold_passed"] is False
    outcome = _outcome(summary)
    assert outcome.disposition == "stability_below_threshold"
    assert outcome.selected_class_count == 3
    assert outcome.successful_resamples == outcome.planned_resamples == 2
    assert outcome.mean_stability == summary["mean_adjusted_rand_index"] < 0.99
    # The refits that measured the instability stay; no class is described.
    assert _rows(out_dir / "cluster_stability.csv") == 2
    tables = _CLASS_TABLES + (_CHARACTERIZATION_TABLES if include_characterization else ())
    assert all(_rows(out_dir / table) == 0 for table in tables)
    freeze = json.loads((out_dir / "stability_freeze.json").read_text(encoding="utf-8"))
    assert freeze["freeze_status"] == summary["freeze_status"]
    assert freeze["reportable_result"] == "no_stable_phenotype_solution"
    policy = json.loads(
        (out_dir / "trajectory_missingness_policy.json").read_text(encoding="utf-8")
    )
    assert policy["n_clusters"] is None
    assert policy["reason_code"] == summary["reason_code"]


@pytest.mark.parametrize(
    "decision_mode, minimum", [("minimum_mean_threshold", 0.5), ("report_only", None)]
)
def test_refits_that_do_not_reproduce_the_solution_leave_stability_unestablished(
    tmp_path: Path, monkeypatch, decision_mode: str, minimum: float | None
) -> None:
    resolved, _representation, _assignments = _bundle(tmp_path)
    fit = owner._fit_observed_data_diag_gmm
    calls = []

    def second_refit_does_not_converge(x, **kwargs):
        calls.append(len(x))
        if len(calls) == 2:
            raise ClassModelFitNotRealized("observed-data refit did not converge")
        return fit(x, **kwargs)

    monkeypatch.setattr(owner, "_fit_observed_data_diag_gmm", second_refit_does_not_converge)
    out_dir = tmp_path / "step_outputs"

    summary = run_trajectory_stability(
        spec=stability_spec(decision_mode=decision_mode, minimum_mean_stability=minimum),
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs=resolved,
    )

    assert summary["status"] == "ok"
    assert summary["scientific_status"] == "failed_closed"
    assert summary["reason_code"] == "TRAJECTORY_STABILITY_REFITS_BELOW_MINIMUM"
    assert summary["freeze_status"] == "not_frozen_stability_refits_below_minimum"
    assert summary["reportable_result"] == "no_stable_phenotype_solution"
    # Too few refits: the prespecified mean does not exist.
    assert summary["mean_adjusted_rand_index"] is None
    assert summary["stability_threshold_passed"] is None
    outcome = _outcome(summary)
    assert outcome.disposition == "too_few_successful_refits"
    assert (outcome.successful_resamples, outcome.planned_resamples) == (1, 2)
    assert outcome.minimum_mean_stability == minimum
    for table in ("cluster_stability.csv", "cluster_stability_assignments.csv", *_CLASS_TABLES):
        assert _rows(out_dir / table) == 0
    [failure] = json.loads(
        (out_dir / "cluster_stability_refit_failures.json").read_text(encoding="utf-8")
    )["failures"]
    assert failure["solution_not_realized"] is True


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("engine defect"),
        ValueError("a refit coordinate has no observed values"),
    ],
    ids=["runtime_error", "input_error"],
)
def test_a_refit_that_fails_for_another_reason_is_still_an_engine_failure(
    tmp_path: Path, monkeypatch, error: Exception
) -> None:
    resolved, _representation, _assignments = _bundle(tmp_path)
    fit = owner._fit_observed_data_diag_gmm
    calls = []

    def second_refit_fails(x, **kwargs):
        calls.append(len(x))
        if len(calls) == 2:
            raise error
        return fit(x, **kwargs)

    monkeypatch.setattr(owner, "_fit_observed_data_diag_gmm", second_refit_fails)

    summary = run_trajectory_stability(
        spec=stability_spec(
            decision_mode="minimum_mean_threshold", minimum_mean_stability=0.5
        ),
        out_dir=tmp_path / "step_outputs",
        run_dir=tmp_path,
        resolved_inputs=resolved,
    )

    assert summary["status"] == "failed_closed"
    assert summary["failure_class"] == "numerical_engine_failure"
    assert summary["reason_code"] == "TRAJECTORY_REFIT_ENGINE_FAILURE"
    assert "reportable_rule_outcomes" not in summary


def test_a_stable_solution_states_that_it_met_the_rule(tmp_path: Path) -> None:
    resolved, representation, _assignments = _bundle(tmp_path)
    out_dir = tmp_path / "step_outputs"

    summary = run_trajectory_stability(
        spec=stability_spec(
            decision_mode="minimum_mean_threshold", minimum_mean_stability=0.5
        ),
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs=resolved,
    )

    assert summary["status"] == "ok"
    assert summary["freeze_status"] == "candidate_labels_frozen_stability_threshold_passed"
    assert "scientific_status" not in summary
    assert _outcome(summary).disposition == "stability_threshold_met"
    assert _rows(out_dir / "cluster_assignments.csv") == len(representation)


def test_a_report_only_design_that_completes_its_refits_states_no_rule(
    tmp_path: Path,
) -> None:
    resolved, _representation, _assignments = _bundle(tmp_path)

    summary = run_trajectory_stability(
        spec=stability_spec(),
        out_dir=tmp_path / "step_outputs",
        run_dir=tmp_path,
        resolved_inputs=resolved,
    )

    assert summary["status"] == "ok"
    assert summary["errors"] == [] and summary["failure_class"] is None
    assert summary["stability_threshold_passed"] is None
    assert "reportable_rule_outcomes" not in summary


def _starts_raising(errors):
    raised = iter(errors)

    def start(x, **kwargs):
        raise next(raised)

    return start


def test_a_multi_start_fit_fails_as_the_model_only_when_every_start_did(
    monkeypatch,
) -> None:
    x = np.zeros((12, 2))
    fit = dict(
        engine=DIAG_GMM_BEST_OF_10_ENGINE,
        n_components=2,
        seed=7,
        max_iter=10,
        tolerance=1e-6,
        regularization=1e-6,
    )
    unrealized = [ClassModelFitNotRealized("did not converge")] * 10
    monkeypatch.setattr(owner, "_fit_observed_data_diag_gmm", _starts_raising(unrealized))
    with pytest.raises(ClassModelFitNotRealized):
        owner._fit_with_engine(x, **fit)

    mixed = [ValueError("ordinal levels are malformed"), *unrealized[1:]]
    monkeypatch.setattr(owner, "_fit_observed_data_diag_gmm", _starts_raising(mixed))
    with pytest.raises(ValueError) as raised:
        owner._fit_with_engine(x, **fit)
    assert not isinstance(raised.value, ClassModelFitNotRealized)


@pytest.mark.parametrize("engine", ["diagonal_gaussian", "mixed_mode"])
def test_each_engine_types_a_fit_that_does_not_converge(engine: str) -> None:
    x = np.random.default_rng(3).normal(size=(40, 2))
    fit = dict(n_components=2, seed=5, max_iter=1, tolerance=1e-12, regularization=1e-6)

    with pytest.raises(ClassModelFitNotRealized):
        if engine == "diagonal_gaussian":
            owner._fit_observed_data_diag_gmm(x, **fit)
        else:
            fit_observed_data_mixed_mode_lca(x, column_levels=(None, None), **fit)
