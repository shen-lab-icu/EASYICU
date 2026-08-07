"""The model contract names the fit that happened, beside the one declared.

An external review (P1-2) read the binary method-family roster --

    binary_logistic_regression, binomial_logistic_regression,
    logistic_regression, logit, statsmodels_logit_mle, statsmodels_glm_binomial

-- saw all six map to one ``sm.Logit`` fit, and concluded the host "accepts a
specific method name, executes another implementation, then echoes the original
method name in the model contract".

The last clause is not what happens.  MEASURED on the recorded corpus: 177
emitted model contracts, every one carrying BOTH keys, and the six that declared
``statsmodels_glm_binomial`` read

    method_family = statsmodels_glm_binomial
    fit_method    = statsmodels_logit_maximum_likelihood

so a reader sees exactly what was asked for and exactly what ran.

That transparency is what makes the mapping acceptable, and it is the reason
this file exists rather than a change.  Two further facts decided it:

* Logit MLE and GLM-Binomial under a logit link maximize the SAME likelihood,
  so the point estimate is identical.  This is unlike the continuous roster,
  where ``_estimator_kind`` deliberately refuses quantile regression because
  fitting OLS instead "would answer a different question under the declared
  method's name" -- there the estimand changes, here it does not.
* Declining the family would send those plans to the Coder, which is the
  opposite of the direction this whole layer moved in.

What was accidental is now guaranteed: a future simplification that set
``fit_method`` from the declaration would erase the only record of the
substitution, and this refuses it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    _FIT_METHODS,
    _estimator_kind,
)
from easyicu.research_agent.schema import (
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES,
    PlannedModelRequirement,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _requirement(family: str, outcome_type: str) -> PlannedModelRequirement:
    return PlannedModelRequirement(
        requirement_id="r1",
        outcome="death",
        outcome_type=outcome_type,
        method_family=family,
        exposure_source="exposure",
        analysis_role="primary",
        analysis_set="complete_case",
        required_for_step_success=True,
    )


def test_the_fitted_method_is_never_the_declared_family():
    """The two keys must not be able to collapse into one.

    ``_FIT_METHODS`` is keyed by what the host RUNS, so no declaration can
    select its own name as the answer.
    """

    fitted = set(_FIT_METHODS.values())
    declared = {
        *ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
        *ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES,
    }

    assert fitted.isdisjoint(declared), fitted & declared


def test_every_accepted_binary_family_resolves_to_the_one_fit_that_runs():
    for family in ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES:
        kind = _estimator_kind(_requirement(family, "binary"))
        assert kind == "logistic", (family, kind)
        assert _FIT_METHODS[kind] == "statsmodels_logit_maximum_likelihood"


def test_quantile_regression_is_still_refused_because_its_estimand_differs():
    """The asymmetry that decides P1-2.

    Fitting OLS where quantile regression was declared answers a different
    question; fitting Logit where GLM-Binomial was declared maximizes the same
    likelihood. The host refuses the first and accepts the second, and this
    holds that line in place.
    """

    for family in ("quantile_regression", "median_quantile_regression"):
        assert family in ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES
        assert _estimator_kind(_requirement(family, "continuous")) == ""

    for family in ("linear_regression", "ordinary_least_squares", "ols"):
        assert _estimator_kind(_requirement(family, "continuous")) == "linear"


def test_the_recorded_contracts_carry_both_names():
    """Re-measures the corpus rather than restating it."""

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    def _contracts(node):
        if isinstance(node, dict):
            if "fit_method" in node and "method_family" in node:
                yield node
            for value in node.values():
                yield from _contracts(value)
        elif isinstance(node, list):
            for value in node:
                yield from _contracts(value)

    seen = substituted = 0
    for path in _CORPUS.glob(
        "batch_*/*/aware/run_*/steps/*/outputs/step_summary.json"
    ):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for contract in _contracts(document):
            seen += 1
            family = str(contract.get("method_family") or "")
            fitted = str(contract.get("fit_method") or "")
            # The record must never present the declaration as the fit.
            assert fitted != family, (family, fitted, path.name)
            assert fitted in set(_FIT_METHODS.values()), (fitted, path.name)
            if family and family != "logistic_regression":
                substituted += 1

    if not seen:
        pytest.skip("no recorded contract carries both keys")
    assert seen > 100, seen
    # The case the review was about must still be in the corpus.
    assert substituted > 0, "no recorded contract declares a non-default spelling"
