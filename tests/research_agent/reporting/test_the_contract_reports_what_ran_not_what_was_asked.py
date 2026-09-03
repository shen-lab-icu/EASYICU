"""A deterministic owner runs exactly the estimator its contract names."""

from __future__ import annotations

from easyicu.research_agent.contracts.model_tokens import (
    ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_OLS_ESTIMATOR,
)
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    _FIT_METHODS,
    _estimator_kind,
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
    }


def test_only_the_exact_binary_estimator_resolves_to_the_fit_that_runs():
    assert ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES == {
        ASSOCIATION_LOGIT_ESTIMATOR,
        "statsmodels_glm_binomial",
    }
    assert _estimator_kind(_requirement(ASSOCIATION_LOGIT_ESTIMATOR, "binary")) == (
        "logistic"
    )
    assert _estimator_kind(_requirement("statsmodels_glm_binomial", "binary")) == ""


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
