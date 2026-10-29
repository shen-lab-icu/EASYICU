"""A Cox result pools coefficients, retains PH checks and receipts its MI policy."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.model_terms import DataPreparationPlan
from easyicu.research_agent.execution.runners import survival_primary_executor as owner
from tests.support.survival_primary import _requirement, _runtime_kwargs


def test_cox_pools_imputations_with_event_and_cumulative_hazard_in_model(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(81)
    n = 200
    frame = pd.DataFrame(
        {
            "follow_up_days": rng.uniform(1, 40, n),
            "death": rng.integers(0, 2, n),
            "treatment": rng.integers(0, 2, n),
            "age": rng.normal(60, 10, n),
        }
    )
    frame.loc[::4, "age"] = np.nan
    source = tmp_path / "synthetic.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / "out"
    monkeypatch.setenv("STEP_OUT_DIR", str(output))
    choice = DataPreparationPlan(
        missing_data={
            "strategy": "multiple_imputation",
            "m": 4,
            "iterations": 3,
            "variables": [{"name": "age", "method": "pmm"}],
        }
    )
    summary = owner.run_survival_primary(
        **_runtime_kwargs(
            _requirement(),
            input_path=source,
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
        data_preparation=choice,
    )
    assert summary["n_analysis"] == n
    receipt = json.loads((output / "survival_analysis_receipt.json").read_text())
    assert receipt["missing_data_plan"]["strategy"] == "multiple_imputation"
    assert "multiple_imputation" in receipt["applied_filter"]
    diagnostics = summary["data_preparation"]["missing_data"]
    assert diagnostics["outcome_in_model"] == "death"
    assert "Nelson–Aalen" in diagnostics["survival_predictors"]
    assert diagnostics["pooling"]["between_variance"] > 0
    assert summary["data_preparation"]["sensitivity"]["strategy"] == "complete_case"
    assert summary["data_preparation"]["sensitivity"]["n"] < n


def test_cox_pooling_uses_log_hazard_coefficients():
    from types import SimpleNamespace

    requirement = _requirement()
    frame = pd.DataFrame(
        {
            "follow_up_days": np.linspace(1, 20, 120),
            "death": [0, 1] * 60,
            "treatment": [0, 1, 0, 1] * 30,
            "age": np.linspace(30, 80, 120),
        }
    )
    plan = DataPreparationPlan(
        missing_data={
            "strategy": "multiple_imputation",
            "m": 3,
            "iterations": 2,
            "variables": [{"name": "age", "method": "pmm"}],
        }
    )

    def estimate(beta):
        return owner._CoxEstimate(float(np.exp(beta)), 0.5, 2.0, 0.2, beta, 0.3)

    betas = iter([0.2, 0.3, 0.15])  # the other imputations, then the CC comparison
    ph = pd.DataFrame(
        {
            "covariate": ["treatment__is_1", "age", "global"],
            "test_statistic": [1.0, 1.0, 1.0],
            "p_value": [0.5, 0.5, 1.0],
        }
    )
    pooled, _, diagnostics, _ = owner.pool_survival_imputations(
        source=frame,
        imputed=SimpleNamespace(
            datasets=(frame.copy(), frame.copy(), frame.copy()), diagnostics={}
        ),
        plan=plan,
        terms=requirement.model_terms,
        exposure="treatment",
        time_column="follow_up_days",
        event_column="death",
        event_value=1,
        horizon=28.0,
        fit_cox=lambda *args, **kwargs: estimate(next(betas)),
        ph_test=lambda *args, **kwargs: ph,
        primary_estimate=estimate(0.1),
        primary_ph=ph,
    )
    assert pooled.coefficient == pytest.approx(0.2)
    assert pooled.hazard_ratio == pytest.approx(np.exp(0.2))
    assert diagnostics["pooling"]["within_variance"] == pytest.approx(0.04)
    assert diagnostics["pooling"]["between_variance"] == pytest.approx(0.01)
