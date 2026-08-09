"""Which coefficient the estimator reports must be asked for, not positioned.

``fit_estimator`` reported the first non-constant column of the design.  That
made column order part of its contract, and the only thing enforcing it was a
comment inside the single caller that knew::

    # fit_estimator reports the first non-const column, so the exposure must
    # lead the design matrix.

An adjusted-association owner is a second caller.  It has no way to discover
that rule except by reading the estimator, and getting it wrong reports a
covariate's effect under the exposure's name -- a wrong number that looks
entirely reasonable.

``test_a_covariate_first_design_reports_the_named_exposure`` is the load-bearing
one: it builds the design in the order that used to be silently wrong.
"""

from __future__ import annotations

import math

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("statsmodels.api")

from easyicu.research_agent.robustness.estimators import (  # noqa: E402
    fit_estimator,
)

_N = 4000


def _frame() -> pd.DataFrame:
    """A cohort where exposure and the covariate push the outcome opposite ways.

    The two effects have different signs and very different magnitudes, so a
    fit that reports the wrong column cannot accidentally agree with one that
    reports the right one.
    """

    rng = np.random.default_rng(20260729)
    exposure = rng.integers(0, 2, _N).astype(float)
    age = rng.normal(0.0, 1.0, _N)
    logit = -0.5 + 1.6 * exposure - 1.1 * age
    probability = 1.0 / (1.0 + np.exp(-logit))
    outcome = (rng.random(_N) < probability).astype(float)
    return pd.DataFrame({"exposure": exposure, "age": age, "death": outcome})


def _fit(columns, **kwargs):
    frame = _frame()
    return fit_estimator(
        cohort=None,
        X=frame[list(columns)],
        y=frame["death"],
        kind="logistic",
        **kwargs,
    )


def test_the_default_still_reports_the_first_non_constant_column() -> None:
    """Existing callers must be unchanged by the new parameter."""

    named = _fit(["exposure", "age"], term="exposure")
    positional = _fit(["exposure", "age"])

    assert positional.point_estimate == pytest.approx(named.point_estimate)
    assert positional.converged and named.converged


def test_a_covariate_first_design_reports_the_named_exposure() -> None:
    """The failure the parameter exists to prevent.

    With ``age`` leading the design, the positional answer is age's odds ratio
    (below 1 by construction) while the exposure's is above 1.  Both fits
    converge and neither looks anomalous on its own.
    """

    positional = _fit(["age", "exposure"])
    named = _fit(["age", "exposure"], term="exposure")

    assert positional.converged and named.converged
    assert positional.point_estimate < 1.0 < named.point_estimate
    # ...and the named answer is the same one a correctly ordered design gives.
    assert named.point_estimate == pytest.approx(
        _fit(["exposure", "age"], term="exposure").point_estimate, rel=1e-9
    )


def test_the_named_term_is_the_adjusted_effect_not_the_crude_one() -> None:
    """Naming the term must not quietly drop the covariates from the fit."""

    adjusted = _fit(["exposure", "age"], term="exposure")
    crude = _fit(["exposure"], term="exposure")

    assert adjusted.converged and crude.converged
    assert not math.isclose(adjusted.point_estimate, crude.point_estimate, rel_tol=1e-3)
    assert "adjust" not in adjusted.notes  # the note is the caller's job, not this


def test_a_term_absent_from_the_design_fails_closed() -> None:
    """Answering with another column's number would hide the disagreement."""

    result = _fit(["exposure", "age"], term="charlson_max")

    assert result.converged is False
    assert result.point_estimate is None
    assert "charlson_max" in result.notes
    assert "not a predictor" in result.notes


def test_the_constant_cannot_be_requested_as_the_term() -> None:
    result = _fit(["exposure", "age"], term="const")

    assert result.converged is False
    assert result.point_estimate is None


def test_the_caller_that_knew_the_rule_now_states_it() -> None:
    """The comment is replaced by an argument, so a reader cannot miss it."""

    from pathlib import Path

    from easyicu.research_agent.robustness import estimators

    source = Path(estimators.__file__).read_text(encoding="utf-8")

    assert "term=exposure," in source
    assert "so the exposure must lead the design matrix" not in source
