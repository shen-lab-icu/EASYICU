"""Tests for the Grambsch-Therneau proportional-hazards diagnostic.

The directional self-checks (PH-holds -> large global p; PH-violated -> small
global p) require ``lifelines``. If it is not importable in this venv they are
skipped with a clear reason rather than asserting on a broken fallback. The
import-guard behaviour is tested unconditionally, since the module must import
and raise a clear error with only pandas+numpy+scipy present.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.ph_schoenfeld import (
    PHTestResult,
    PHTestUnavailableError,
    ph_test,
    run_ph_test,
)

_HAS_LIFELINES = importlib.util.find_spec("lifelines") is not None
_needs_lifelines = pytest.mark.skipif(
    not _HAS_LIFELINES,
    reason="lifelines is not installed; PH (Schoenfeld) test cannot run",
)


# ---------------------------------------------------------------------------
# Simulated survival datasets
# ---------------------------------------------------------------------------


def _make_ph_holds(rng: np.random.Generator, n: int = 400) -> pd.DataFrame:
    """Exposure with a CONSTANT log-hazard effect -> PH holds.

    Exponential survival times with hazard ``exp(beta * exposure)``. The
    proportional-hazards assumption is true by construction, so the global PH
    test should NOT reject.
    """

    exposure = rng.integers(0, 2, size=n).astype(float)
    beta = np.log(2.0)  # constant HR = 2 for exposure
    base_rate = 0.1
    rate = base_rate * np.exp(beta * exposure)
    event_time = rng.exponential(1.0 / rate)
    censor_time = rng.exponential(1.0 / base_rate * 3.0)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return pd.DataFrame(
        {"duration": duration, "event": event, "exposure": exposure}
    )


def _make_ph_violated(rng: np.random.Generator, n: int = 600) -> pd.DataFrame:
    """Exposure whose effect FLIPS over follow-up -> PH violated.

    Early follow-up: exposed subjects have a much higher hazard (they fail
    fast). Late follow-up: the surviving exposed subjects have a much lower
    hazard. This is an early-crossing-hazards pattern; a single HR averages two
    opposite regimes, so the global PH test should reject.
    """

    exposure = rng.integers(0, 2, size=n).astype(float)
    duration = np.empty(n)
    event = np.ones(n, dtype=int)
    crossover = 1.0
    for i in range(n):
        if exposure[i] > 0.5:
            # Exposed: high early hazard, then low late hazard.
            early = rng.exponential(1.0 / 3.0)  # fast early failures
            if early <= crossover:
                duration[i] = early
            else:
                # Survived the early phase -> now very low hazard.
                duration[i] = crossover + rng.exponential(1.0 / 0.05)
        else:
            # Unexposed: steady moderate hazard throughout.
            duration[i] = rng.exponential(1.0 / 0.6)
    # Light administrative censoring so not every row is an event.
    admin = np.quantile(duration, 0.9)
    censored = duration > admin
    duration[censored] = admin
    event[censored] = 0
    return pd.DataFrame(
        {"duration": duration, "event": event, "exposure": exposure}
    )


# ---------------------------------------------------------------------------
# Import-guard behaviour (runs with or without lifelines)
# ---------------------------------------------------------------------------


def test_module_imports_without_lifelines():
    """The module must import on a pandas+numpy+scipy-only install."""

    import easyicu.research_agent.ph_schoenfeld as mod

    assert hasattr(mod, "ph_test")
    assert hasattr(mod, "PHTestResult")
    assert issubclass(PHTestUnavailableError, RuntimeError)


def test_raises_clear_error_when_lifelines_absent():
    """Without lifelines, ph_test raises an actionable RuntimeError-derived error."""

    if _HAS_LIFELINES:
        pytest.skip("lifelines is installed; guard path not exercised here")
    df = pd.DataFrame(
        {"duration": [1.0, 2.0, 3.0], "event": [1, 0, 1], "exposure": [0.0, 1.0, 1.0]}
    )
    with pytest.raises(PHTestUnavailableError) as excinfo:
        ph_test(df, "duration", "event", ["exposure"])
    assert "lifelines" in str(excinfo.value).lower()
    assert "pip install" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Input validation (runs without lifelines only after the guard clears; these
# guard-independent checks need lifelines to reach the validation branch, so we
# gate them too where the guard fires first).
# ---------------------------------------------------------------------------


@_needs_lifelines
def test_unknown_time_transform_raises():
    df = _make_ph_holds(np.random.default_rng(0))
    with pytest.raises(ValueError, match="time_transform"):
        ph_test(df, "duration", "event", ["exposure"], time_transform="bogus")


@_needs_lifelines
def test_missing_columns_raise():
    df = _make_ph_holds(np.random.default_rng(0))
    with pytest.raises(ValueError, match="missing required columns"):
        ph_test(df, "duration", "event", ["not_a_column"])


@_needs_lifelines
def test_empty_covariates_raise():
    df = _make_ph_holds(np.random.default_rng(0))
    with pytest.raises(ValueError, match="non-empty"):
        ph_test(df, "duration", "event", [])


# ---------------------------------------------------------------------------
# Directional reference-value self-checks (require lifelines)
# ---------------------------------------------------------------------------


@_needs_lifelines
def test_ph_holds_global_p_is_large():
    """Constant-effect data -> global PH p LARGE (no violation)."""

    rng = np.random.default_rng(20260706)
    df = _make_ph_holds(rng)
    table = ph_test(df, "duration", "event", ["exposure"], time_transform="km")

    assert list(table.columns) == ["covariate", "test_statistic", "p_value"]
    assert set(table["covariate"]) == {"exposure", "global"}
    global_p = float(table.loc[table["covariate"] == "global", "p_value"].iloc[0])
    assert global_p > 0.05, f"expected no PH violation, got global p={global_p}"


@_needs_lifelines
def test_ph_violated_global_p_is_small():
    """Time-varying (crossing) effect -> global PH p SMALL (violation)."""

    rng = np.random.default_rng(20260706)
    df = _make_ph_violated(rng)
    table = ph_test(df, "duration", "event", ["exposure"], time_transform="km")

    global_p = float(table.loc[table["covariate"] == "global", "p_value"].iloc[0])
    assert global_p < 0.05, f"expected PH violation, got global p={global_p}"


@_needs_lifelines
def test_result_wrapper_violated_and_accessors():
    """PHTestResult.violated() flags the time-varying covariate."""

    rng = np.random.default_rng(20260706)
    df = _make_ph_violated(rng)
    result = run_ph_test(df, "duration", "event", ["exposure"], time_transform="km")

    assert isinstance(result, PHTestResult)
    assert result.violated(alpha=0.05) == ["exposure"]
    assert result.is_violated(alpha=0.05) is True
    assert result.global_p_value() < 0.05
    # "global" is the joint row, never returned as a violated covariate.
    assert "global" not in result.violated()


@_needs_lifelines
def test_result_wrapper_no_violation_when_ph_holds():
    rng = np.random.default_rng(20260706)
    df = _make_ph_holds(rng)
    result = run_ph_test(df, "duration", "event", ["exposure"], time_transform="km")

    assert result.violated(alpha=0.05) == []
    assert result.is_violated(alpha=0.05) is False
