"""Declared risk ratios use a binary Poisson mean with sandwich covariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    AdjustedAssociationError,
    adjusted_association_executor_owns_step,
    run_adjusted_association_from_env,
)
from easyicu.research_agent.schema import AnalysisStep


def frame():
    rng = np.random.default_rng(835)
    n = 4000
    x = rng.integers(0, 2, n)
    z = rng.uniform(-1, 1, n)
    y = rng.binomial(1, 0.2 * 2 ** x * np.exp(0.1 * z))
    return pd.DataFrame({"x": x, "z": z, "y": y, "patient": np.arange(n) // 2})


def declaration():
    return dict(
        requirement_id="primary_rr", exposure="x", outcome="y", covariates=["z"],
        model_terms=[
            dict(name="x", role="exposure", coding="binary", levels=["0", "1"],
                 reference_level="0", transform="treatment_contrast"),
            dict(name="z", role="covariate", coding="continuous", transform="identity"),
        ],
        estimator_kind="glm_poisson", analysis_set="complete_case", analysis_role="primary",
        method_family="statsmodels_glm_poisson_log_robust",
    )


def rr_step():
    requirement = declaration()
    for key in ("exposure", "estimator_kind"):
        requirement.pop(key)
    requirement.update(exposure_source="x", outcome_type="binary", analysis_set="source_aware")
    step = AnalysisStep(
        step_id="rr", intent="Estimate the adjusted risk ratio", inputs=["table:analysis_cohort"],
        method="adjusted_association_models", planned_analysis_role="primary",
        expected_outputs=["table:adjusted_association_estimates"], model_requirements=[requirement],
    )
    return step


def test_modified_poisson_is_claimed_only_for_a_declared_binary_rr():
    from easyicu.research_agent.execution.runners.selection import select_standard_executor
    from easyicu.research_agent.schema import AnalysisPlan
    step = rr_step()
    assert adjusted_association_executor_owns_step(step)
    selected = select_standard_executor(step, plan=AnalysisPlan(
        research_question="Estimate the adjusted risk ratio", analysis_type="association_study", steps=[step],
    ))
    assert selected is not None
    assert "statsmodels_glm_poisson_log_robust" in selected.code
    step.model_requirements[0].outcome_type = "continuous"
    assert not adjusted_association_executor_owns_step(step)


@pytest.mark.parametrize("clustered", [False, True])
def test_modified_poisson_matches_statsmodels_and_recovers_rr(tmp_path, clustered):
    data = frame()
    data.loc[0, "z"] = np.nan
    kwargs = declaration()
    if clustered:
        kwargs["dependence"] = dict(group_source="patient", group_derivation="identity")
    summary = run_adjusted_association_from_env(frame=data, output_dir=tmp_path,
                                               emit_step_summary=False, **kwargs)
    used = data.dropna()
    design = sm.add_constant(used[["x", "z"]].astype(float))
    cov = {"cov_type": "cluster", "cov_kwds": {"groups": used.patient}} if clustered else {"cov_type": "HC0"}
    reference = sm.GLM(used.y, design, family=sm.families.Poisson()).fit(**cov)
    table = pd.read_csv(tmp_path / "adjusted_association_estimates.csv").iloc[0]
    assert table.effect_scale == summary["effect_scale"] == "risk_ratio"
    assert table.estimate == pytest.approx(np.exp(reference.params.x), rel=1e-9)
    assert [table.ci_low, table.ci_high] == pytest.approx(np.exp(reference.conf_int().loc["x"]))
    assert table.standard_error == pytest.approx(reference.bse.x)
    assert table.estimate == pytest.approx(2.0, abs=0.15)
    odds_ratio = np.exp(sm.Logit(used.y, design).fit(disp=False).params.x)
    assert odds_ratio - table.estimate > 0.5
    assert table.variance_estimator == ("cluster_robust" if clustered else "sandwich_robust")
    assert summary["outcome_proportion"] == pytest.approx(used.y.mean())
    assert summary["risk_ratio_recommendation"]["recommended"] is True
    assert summary["risk_ratio_recommendation"]["threshold"] == 0.1
    assert summary["model_contracts"][0]["outcome_type"] == "binary"
    assert table.n == len(used)
    assert table.n_events == used.y.sum()
    assert table.outcome_proportion == pytest.approx(used.y.mean())
    from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
    step = rr_step()
    step.model_requirements[0].analysis_set = "complete_case"
    if clustered:
        from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
        step.model_requirements[0].dependence = PlannedDependenceRequirement.model_validate(kwargs["dependence"])
    issues, _ = PrimaryModelContractValidator._planned_model_requirement_issues(
        step=step, contracts=summary["model_contracts"],
    )
    assert issues == []
    from easyicu.research_agent.robustness.primary_effect import _extract_primary_effect_payload_from_summary
    effect = _extract_primary_effect_payload_from_summary(summary, path=None, preferred_predictor="x")
    assert effect["effect_measure"] == "RR"
    assert effect["primary_or"] == pytest.approx(table.estimate)


@pytest.mark.parametrize("events,recommended", [(20, False), (22, True)])
def test_common_outcome_recommendation_does_not_change_declared_logistic(tmp_path, events, recommended):
    data = frame().iloc[:200].copy()
    data["y"] = (np.arange(200) < events).astype(int)
    kwargs = {**declaration(), "estimator_kind": "logistic", "method_family": "statsmodels_logit_mle"}
    summary = run_adjusted_association_from_env(frame=data, output_dir=tmp_path,
                                               emit_step_summary=False, **kwargs)
    assert summary["effect_scale"] == "odds_ratio"
    assert summary["outcome_proportion"] == events / 200
    assert summary["risk_ratio_recommendation"]["recommended"] is recommended


def test_rr_refuses_a_nonbinary_outcome(tmp_path):
    data = frame()
    data.loc[0, "y"] = 2
    with pytest.raises(AdjustedAssociationError, match="binary 0/1 outcome"):
        run_adjusted_association_from_env(frame=data, output_dir=tmp_path, **declaration())
    assert not (tmp_path / "adjusted_association_estimates.csv").exists()


def test_rr_robustness_cannot_silently_refit_an_odds_ratio(tmp_path):
    from easyicu.research_agent.robustness.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness.panel import default_robustness_specs
    data = frame().iloc[:200]
    summary = run_adjusted_association_from_env(frame=data, output_dir=tmp_path,
                                               emit_step_summary=False, **declaration())
    summary["estimator_adapter"] = dict(data=data.to_dict("records"), exposure="x", outcome="y",
                                       estimator_kind="logistic", missing_strategy="complete_case")
    records = [dict(step_id="rr", status="ok", planned_analysis_role="primary", step_summary=summary,
                    analysis_request={"step": {"step_id": "rr", "planned_analysis_role": "primary"}})]
    rows, warnings = fit_robustness_rows_from_records(specs=default_robustness_specs(), per_step_records=records,
                                                      allow_implicit_cohort_refit=True)
    assert len(rows) == 1
    assert rows[0].point_estimate == pytest.approx(summary["primary_rr"])
    assert any("risk-ratio" in warning for warning in warnings)
