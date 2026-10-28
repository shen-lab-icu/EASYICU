"""A deterministic owner runs exactly the estimator its contract names."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from easyicu.research_agent.contracts.model_tokens import (
    ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_OLS_ESTIMATOR,
    ASSOCIATION_RR_ESTIMATOR,
)
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    _FIT_METHODS,
    _estimator_kind,
    run_adjusted_association_from_env,
)
from easyicu.research_agent.schema import (
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES,
    PlannedModelRequirement,
)


def _requirement(family: str, outcome_type: str) -> PlannedModelRequirement:
    coding = "binary" if outcome_type == "binary" else "continuous"
    term = {
        "name": "exposure",
        "role": "exposure",
        "coding": coding,
        "transform": "treatment_contrast" if coding == "binary" else "identity",
    }
    if coding == "binary":
        term.update(
            {
                "levels": ["0", "1"],
                "reference_level": "0",
            }
        )
    return PlannedModelRequirement(
        requirement_id="r1",
        outcome="death",
        outcome_type=outcome_type,
        method_family=family,
        exposure_source="exposure",
        covariates=[],
        model_terms=[term],
        analysis_role="primary",
        analysis_set="complete_case",
        required_for_step_success=True,
    )


def test_the_fitted_method_is_the_exact_declared_estimator_token():
    assert _FIT_METHODS == {
        "logistic": ASSOCIATION_LOGIT_ESTIMATOR,
        "linear": ASSOCIATION_OLS_ESTIMATOR,
        "glm_poisson": ASSOCIATION_RR_ESTIMATOR,
        "proportional_odds": "statsmodels_ordered_logit",
        "count_poisson": "statsmodels_count_poisson",
        "negative_binomial": "statsmodels_negative_binomial",
    }


def test_only_the_exact_binary_estimator_resolves_to_the_fit_that_runs():
    assert ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES == {
        ASSOCIATION_LOGIT_ESTIMATOR,
        "statsmodels_glm_binomial",
        ASSOCIATION_RR_ESTIMATOR,
    }
    assert _estimator_kind(_requirement(ASSOCIATION_LOGIT_ESTIMATOR, "binary")) == (
        "logistic"
    )
    assert _estimator_kind(_requirement("statsmodels_glm_binomial", "binary")) == ""
    for family in ("modified_poisson", ASSOCIATION_RR_ESTIMATOR):
        assert _estimator_kind(_requirement(family, "binary")) == "glm_poisson"


def test_quantile_regression_is_still_refused_because_its_estimand_differs():
    """A close statistical relationship never authorizes substitution."""

    for family in ("quantile_regression", "median_quantile_regression"):
        requirement = _requirement(family, "continuous")
        assert (
            requirement.method_family in ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES
        )
        assert _estimator_kind(requirement) == ""

    for family in ("linear_regression", "ordinary_least_squares", "ols"):
        requirement = _requirement(family, "continuous")
        assert requirement.method_family == ASSOCIATION_OLS_ESTIMATOR
        assert _estimator_kind(requirement) == "linear"


@pytest.mark.parametrize("family,kind,scale,estimate", [
    (ASSOCIATION_LOGIT_ESTIMATOR, "logistic", "odds_ratio", 3.0),
    (ASSOCIATION_RR_ESTIMATOR, "glm_poisson", "risk_ratio", 2.0),
    ("modified_poisson", "glm_poisson", "risk_ratio", 2.0),
])
def test_reported_binary_fit_names_the_estimator_that_produced_it(tmp_path, family, kind, scale, estimate):
    requirement = _requirement(family, "binary")
    data = pd.DataFrame([
        {"exposure": exposure, "death": int(row < events)}
        for exposure, events in ((0, 10), (1, 20)) for row in range(40)
    ])
    run_adjusted_association_from_env(
        frame=data, output_dir=tmp_path, requirement_id=requirement.requirement_id,
        exposure="exposure", outcome="death", covariates=[],
        model_terms=[term.model_dump(mode="json") for term in requirement.model_terms],
        estimator_kind=_estimator_kind(requirement), method_family=requirement.method_family,
        analysis_set="complete_case", analysis_role="primary",
    )
    report = json.loads((tmp_path / "step_summary.json").read_text())
    contract = report["model_contracts"][0]
    assert contract["fit_method"] == contract["method_family"] == requirement.method_family
    assert contract["outcome_type"] == "binary"
    assert report["estimator_kind"] == kind
    assert report["effect_scale"] == scale
    table = pd.read_csv(tmp_path / "adjusted_association_estimates.csv").iloc[0]
    assert table.estimator_kind == kind and table.effect_scale == scale
    assert table.estimate == pytest.approx(estimate)
    assert table.n == 80 and table.n_events == 30
    if kind == "glm_poisson":
        assert report["primary_rr"] == pytest.approx(estimate)
        assert "primary_or" not in report
