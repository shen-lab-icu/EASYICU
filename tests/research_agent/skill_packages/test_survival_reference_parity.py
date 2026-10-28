"""Cox scripts preserve host ties, horizon, complete cases, coding, and diagnostics."""

import json

import numpy as np
import pandas as pd
import pytest
from statsmodels.duration.hazard_regression import PHReg

from easyicu.research_agent.execution.model_matrix import compile_model_terms
from easyicu.research_agent.execution.runners.survival_primary_executor import survival_primary_executor_code
from easyicu.research_agent.skill_packages.survival_time_to_event.scripts.example_data import example_cohort, example_spec
from easyicu.research_agent.skill_packages.survival_time_to_event.scripts.load_cohort import load_cohort
from easyicu.research_agent.skill_packages.survival_time_to_event.scripts.run_analysis import run_analysis
from tests.support.reference_scripts import assert_numeric_parity, run_host_executor

VALUES = ["coefficient", "standard_error", "ci_low", "ci_high", "p_value", "hazard_ratio"]


def test_survival_matches_host_fit_counts_and_diagnostics(tmp_path, monkeypatch):
    frame, spec = example_cohort(), example_spec()
    host = run_host_executor(tmp_path, monkeypatch, frame, spec.host_step(), survival_primary_executor_code)
    out = tmp_path / "standalone"
    run_analysis(load_cohort(frame, spec), spec, out)
    expected = pd.read_csv(host / "survival_effect_estimates.csv")
    actual = pd.read_csv(out / "survival_effect_estimates.csv")
    assert_numeric_parity(actual, expected, counts=["n_analysis_rows", "n_events"], values=VALUES)
    assert int(actual.n_analysis_rows.iloc[0]) == 233
    complete = frame.dropna(subset=["follow_up_days", "death", "treatment", "age", "sex"])
    events = int((complete.death.eq(5) & complete.follow_up_days.le(28)).sum())
    assert int(actual.n_events.iloc[0]) == events
    for name in ("survival_ph_diagnostic.csv", "survival_logrank.csv"):
        # All numeric diagnostics, counts and textual decisions retain parity.
        pd.testing.assert_frame_equal(pd.read_csv(out / name), pd.read_csv(host / name),
                                      check_exact=False, atol=1e-8, rtol=1e-8)
    receipt = json.loads((out / "survival_analysis_receipt.json").read_text())
    assert receipt["n_complete_case_dropped"] == 7
    assert receipt["n_censored_at_horizon"] == int(complete.follow_up_days.gt(28).sum())
    assert receipt["exposure_design_column"] == "treatment__is_0"


def test_breslow_ties_fail_survival_parity(tmp_path, monkeypatch):
    frame, spec = example_cohort(), example_spec()
    host = run_host_executor(tmp_path, monkeypatch, frame, spec.host_step(), survival_primary_executor_code)
    expected = pd.read_csv(host / "survival_effect_estimates.csv")
    primary = spec.primary
    compiled = compile_model_terms(frame, terms=primary.model_terms, exposure=primary.exposure_source)
    analysis = frame[["follow_up_days", "death"]].join(compiled.design).dropna()
    event = (analysis.death.eq(5) & analysis.follow_up_days.le(28)).astype(int)
    fit = PHReg(analysis.follow_up_days.clip(upper=28), analysis[list(compiled.design.columns)],
                status=event, ties="breslow").fit()
    focal = list(compiled.design.columns).index("treatment__is_0")
    actual = expected.copy()
    actual["coefficient"] = fit.params[focal]
    actual["standard_error"] = fit.bse[focal]
    actual["hazard_ratio"] = np.exp(fit.params[focal])
    actual["p_value"] = fit.pvalues[focal]
    actual[["ci_low", "ci_high"]] = np.exp(fit.conf_int()[focal])
    assert abs(actual.coefficient.iloc[0] - expected.coefficient.iloc[0]) > 1e-8
    with pytest.raises(AssertionError):
        assert_numeric_parity(actual, expected, counts=["n_analysis_rows", "n_events"], values=VALUES)
