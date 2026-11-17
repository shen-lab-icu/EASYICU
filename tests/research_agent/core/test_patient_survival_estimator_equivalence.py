"""Numerical contracts on synthetic, correlated copies of patient stays."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.execution.runners.survival_patient_intervals import (
    clustered_rmst_table,
)
from easyicu.research_agent.methods.patient_cluster_survival import (
    bind_groups,
    clustered_ph_test,
    cox_fit_arguments,
)
from easyicu.research_agent.methods.ph_schoenfeld import ph_test
from easyicu.research_agent.methods.rmst import rmst, rmst_difference
from easyicu.research_agent.methods.time_varying_cox import (
    fit_piecewise_time_varying_cox,
)


def _frames(copies):
    rng = np.random.default_rng(732)
    n = 100
    exposure = rng.binomial(1, 0.5, n)
    event_time = rng.exponential(12, n)
    # A modest change of exposure effect over time makes the PH direction
    # visible when correlated copies are falsely counted as independent.
    event_time = np.where(exposure == 1, event_time**1.3, event_time)
    base = pd.DataFrame(
        {
            "time": np.minimum(event_time, 27),
            "event": (event_time <= 27).astype(int),
            "exposure": exposure,
        }
    )
    independent = base.loc[base.index.repeat(copies)].reset_index(drop=True)
    clustered = independent.copy()
    bind_groups(
        clustered,
        clustered.assign(patient=np.repeat(np.arange(n), copies)),
        PlannedDependenceRequirement(
            group_source="patient", group_derivation="identity"
        ),
    )
    return independent, clustered


@pytest.mark.parametrize("copies", [1, 4])
def test_binary_cox_preserves_single_stays_and_widens_for_correlated_stays(copies):
    independent, clustered = _frames(copies)
    old = CoxPHFitter().fit(independent, duration_col="time", event_col="event")
    frame, kwargs = cox_fit_arguments(clustered)
    actual = CoxPHFitter().fit(frame, duration_col="time", event_col="event", **kwargs)
    np.testing.assert_allclose(actual.params_, old.params_, rtol=1e-10)
    if copies == 1:
        pd.testing.assert_frame_equal(actual.summary, old.summary)
    else:
        assert (
            actual.standard_errors_["exposure"] > 1.5 * old.standard_errors_["exposure"]
        )


@pytest.mark.parametrize("copies", [1, 4])
def test_ph_trend_preserves_single_stays_and_reduces_duplicate_evidence(copies):
    independent, clustered = _frames(copies)
    arguments = dict(duration_col="time", event_col="event", covariates=["exposure"])
    old = ph_test(independent, **arguments)
    actual = clustered_ph_test(clustered, **arguments)
    if copies == 1:
        pd.testing.assert_frame_equal(actual, old)
    else:
        assert actual.loc[0, "p_value"] > old.loc[0, "p_value"]
        assert actual.loc[0, "test_statistic"] < old.loc[0, "test_statistic"]


def _rmst_table(frame):
    exposed = frame.loc[frame.exposure.eq(1)]
    comparator = frame.loc[frame.exposure.eq(0)]
    a = rmst(exposed.time, exposed.event, 27)
    b = rmst(comparator.time, comparator.event, 27)
    contrast = rmst_difference(frame.time, frame.event, frame.exposure, 27)
    return pd.DataFrame(
        [
            {
                "exposed_rmst_days": a.rmst,
                "comparator_rmst_days": b.rmst,
                "rmst_difference_days": -contrast["diff"],
                "exposed_rmst_ci_low": a.ci_low,
                "exposed_rmst_ci_high": a.ci_high,
                "comparator_rmst_ci_low": b.ci_low,
                "comparator_rmst_ci_high": b.ci_high,
                "ci_low": -contrast["ci"][1],
                "ci_high": -contrast["ci"][0],
                "standard_error": contrast["se_diff"],
                "p_value": contrast["p_value"],
            }
        ]
    )


@pytest.mark.parametrize("copies", [1, 4])
def test_rmst_preserves_single_stays_and_deletes_whole_repeated_patients(copies):
    independent, clustered = _frames(copies)
    old = _rmst_table(independent)
    actual = clustered_rmst_table(
        clustered,
        sealed=SimpleNamespace(
            derived_exposure_column="exposure",
            derived_time_column="time",
            derived_event_column="event",
        ),
        table=old,
        tau=27,
    )
    if copies == 1:
        pd.testing.assert_frame_equal(actual, old)
    else:
        assert actual.loc[0, "standard_error"] > 1.5 * old.loc[0, "standard_error"]
        np.testing.assert_allclose(
            actual[
                ["exposed_rmst_days", "comparator_rmst_days", "rmst_difference_days"]
            ],
            old[["exposed_rmst_days", "comparator_rmst_days", "rmst_difference_days"]],
        )


@pytest.mark.parametrize("copies", [1, 4])
def test_piecewise_cox_preserves_single_stays_and_widens_patient_intervals(copies):
    independent, clustered = _frames(copies)
    arguments = dict(
        duration_col="time",
        event_col="event",
        covariates=["exposure"],
        interval_cutpoints=[7],
        exposure_col="exposure",
    )
    old = fit_piecewise_time_varying_cox(independent, **arguments)
    actual = fit_piecewise_time_varying_cox(clustered, **arguments)
    np.testing.assert_allclose(actual.coefficient, old.coefficient, rtol=1e-10)
    if copies == 1:
        pd.testing.assert_frame_equal(actual, old)
    else:
        assert (actual.standard_error > 1.5 * old.standard_error).all()
        assert (actual.ci_low < old.ci_low).all()
        assert (actual.ci_high > old.ci_high).all()
