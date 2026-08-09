"""Canonical spelling for typed statistical-model contract tokens."""

from __future__ import annotations

import re
from typing import Any


ADJUSTED_ASSOCIATION_ANALYSIS_KIND = "adjusted_association_estimates"
ASSOCIATION_LOGIT_ESTIMATOR = "statsmodels_logit_mle"
ASSOCIATION_GLM_BINOMIAL_ESTIMATOR = "statsmodels_glm_binomial"
ASSOCIATION_OLS_ESTIMATOR = "statsmodels_ols"
SURVIVAL_COX_ESTIMATOR = "cox_proportional_hazards"
SURVIVAL_PH_DIAGNOSTIC = "schoenfeld_per_covariate_with_bonferroni_summary"

#: The exact step method and product key that name the host-owned primary
#: adjusted-association contract.  They live beside the estimator tokens rather
#: than in ``schema`` so that the ownership predicate reading them
#: (:mod:`.association_execution`) stays dependency-neutral; ``schema``
#: re-exports both, so every existing caller is unchanged.
PLANNED_MODEL_REQUIREMENTS_STEP_METHOD = "adjusted_association_models"
PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND = "table"
ADJUSTED_ASSOCIATION_OUTPUT = (
    f"{PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND}:{ADJUSTED_ASSOCIATION_ANALYSIS_KIND}"
)

_ASSOCIATION_METHOD_ALIASES = {
    "binary_logistic_regression": ASSOCIATION_LOGIT_ESTIMATOR,
    "binomial_logistic_regression": ASSOCIATION_LOGIT_ESTIMATOR,
    "logistic_regression": ASSOCIATION_LOGIT_ESTIMATOR,
    "logit": ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_LOGIT_ESTIMATOR: ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_GLM_BINOMIAL_ESTIMATOR: ASSOCIATION_GLM_BINOMIAL_ESTIMATOR,
    "linear_regression": ASSOCIATION_OLS_ESTIMATOR,
    "ordinary_least_squares": ASSOCIATION_OLS_ESTIMATOR,
    "ols": ASSOCIATION_OLS_ESTIMATOR,
    ASSOCIATION_OLS_ESTIMATOR: ASSOCIATION_OLS_ESTIMATOR,
    "quantile_regression": "statsmodels_quantreg",
    "median_quantile_regression": "statsmodels_quantreg_median_vcov_robust",
    "statsmodels_quantreg": "statsmodels_quantreg",
    "statsmodels_quantreg_median_vcov_robust": (
        "statsmodels_quantreg_median_vcov_robust"
    ),
}

_SURVIVAL_ESTIMATOR_ALIASES = {
    "cox": SURVIVAL_COX_ESTIMATOR,
    "cox_ph": SURVIVAL_COX_ESTIMATOR,
    "cox_regression": SURVIVAL_COX_ESTIMATOR,
    SURVIVAL_COX_ESTIMATOR: SURVIVAL_COX_ESTIMATOR,
}

_SURVIVAL_PH_ALIASES = {
    "schoenfeld": SURVIVAL_PH_DIAGNOSTIC,
    # Historical plans called the whole contract a "global" test. Keep them
    # readable, but canonicalize to the statistic actually implemented: one
    # Schoenfeld test per covariate plus a Bonferroni family-wise summary.
    "schoenfeld_global_test": SURVIVAL_PH_DIAGNOSTIC,
    "schoenfeld_global": SURVIVAL_PH_DIAGNOSTIC,
    "global_schoenfeld": SURVIVAL_PH_DIAGNOSTIC,
    "global_schoenfeld_test": SURVIVAL_PH_DIAGNOSTIC,
    "global_schoenfeld_residual_test": SURVIVAL_PH_DIAGNOSTIC,
    SURVIVAL_PH_DIAGNOSTIC: SURVIVAL_PH_DIAGNOSTIC,
}


def normalise_model_contract_token(value: Any) -> str:
    """Collapse presentation spelling without changing contract semantics."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def canonical_association_method(value: Any) -> str:
    """Canonicalise presentation aliases without substituting an estimator."""

    token = normalise_model_contract_token(value)
    return _ASSOCIATION_METHOD_ALIASES.get(token, token)


def canonical_survival_estimator(value: Any) -> str:
    """Canonicalise only aliases that mean the exact unpenalized Cox model."""

    token = normalise_model_contract_token(value)
    return _SURVIVAL_ESTIMATOR_ALIASES.get(token, token)


def canonical_survival_ph_diagnostic(value: Any) -> str:
    """Canonicalise aliases for the implemented per-covariate PH diagnostic."""

    token = normalise_model_contract_token(value)
    return _SURVIVAL_PH_ALIASES.get(token, token)


__all__ = [
    "ADJUSTED_ASSOCIATION_ANALYSIS_KIND",
    "ADJUSTED_ASSOCIATION_OUTPUT",
    "PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND",
    "PLANNED_MODEL_REQUIREMENTS_STEP_METHOD",
    "ASSOCIATION_GLM_BINOMIAL_ESTIMATOR",
    "ASSOCIATION_LOGIT_ESTIMATOR",
    "ASSOCIATION_OLS_ESTIMATOR",
    "SURVIVAL_COX_ESTIMATOR",
    "SURVIVAL_PH_DIAGNOSTIC",
    "canonical_association_method",
    "canonical_survival_estimator",
    "canonical_survival_ph_diagnostic",
    "normalise_model_contract_token",
]
