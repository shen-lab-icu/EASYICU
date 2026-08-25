from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.methods.time_varying_cox import (
    TimeVaryingCoxError,
    fit_piecewise_time_varying_cox,
)


def _survival_frame() -> pd.DataFrame:
    pytest.importorskip("lifelines")
    rng = np.random.default_rng(20260825)
    n = 1_200
    exposure = rng.binomial(1, 0.4, n)
    age = rng.normal(0.0, 1.0, n)
    rate = np.exp(-3.0 + 0.55 * exposure + 0.2 * age)
    event_time = rng.exponential(1.0 / rate)
    return pd.DataFrame(
        {
            "time": np.minimum(event_time, 27.0),
            "event": (event_time <= 27.0).astype(int),
            "exposure": exposure,
            "age": age,
        }
    )


def test_piecewise_time_varying_cox_reports_every_sealed_interval() -> None:
    result = fit_piecewise_time_varying_cox(
        _survival_frame(),
        duration_col="time",
        event_col="event",
        covariates=["exposure", "age"],
        interval_cutpoints=[7.0, 14.0],
        exposure_col="exposure",
    )

    assert len(result) == 6
    exposure = result.loc[result["is_exposure"]].sort_values("interval_index")
    assert exposure["interval_index"].tolist() == [1, 2, 3]
    assert exposure["interval_start_days"].tolist() == [0.0, 7.0, 14.0]
    assert exposure["interval_end_days"].tolist() == [7.0, 14.0, 27.0]
    assert (exposure["ci_low"] < exposure["hazard_ratio"]).all()
    assert (exposure["hazard_ratio"] < exposure["ci_high"]).all()
    assert np.isfinite(
        result[["hazard_ratio", "ci_low", "ci_high", "p_value"]].to_numpy()
    ).all()


def test_piecewise_time_varying_cox_rejects_post_followup_cutpoint() -> None:
    with pytest.raises(TimeVaryingCoxError, match="precede observed follow-up"):
        fit_piecewise_time_varying_cox(
            _survival_frame(),
            duration_col="time",
            event_col="event",
            covariates=["exposure", "age"],
            interval_cutpoints=[7.0, 27.0],
            exposure_col="exposure",
        )
