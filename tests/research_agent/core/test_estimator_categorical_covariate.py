"""A declared adjustment set may name a column that holds words.

fresh23 got the primary adjusted association a deterministic owner and the
step still died -- this time inside the estimator::

    AdjustedAssociationError: declared model 'primary_full_cohort_logistic'
    could not be fitted as declared: could not convert string to float: 'Male'

The plan declared ``covariates: ['age', 'sex', 'charlson_first']`` and the
cohort stores ``sex`` as ``Male``/``Female``.  ``fit_estimator`` cast the whole
design to float, so any adjustment set naming a categorical column could not
be fitted at all.

The encoding belongs in ``fit_estimator`` and not in the association owner,
because the robustness replay fits its variants through the same function so
that a disagreement between the primary estimate and its sensitivity variants
is a real disagreement.  Fixing one caller would have left the other with the
same failure and made the two estimates incomparable.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("statsmodels.api")

from easyicu.research_agent.robustness.estimators import (  # noqa: E402
    fit_estimator,
)

_N = 3000


def _frame() -> "pd.DataFrame":
    """The real shape: numeric exposure, one word-valued covariate."""

    rng = np.random.default_rng(20260729)
    frame = pd.DataFrame(
        {
            "exposure": rng.integers(0, 2, _N).astype(float),
            "age": rng.normal(65.0, 12.0, _N),
            "sex": rng.choice(["Male", "Female"], _N),
            "comorbidity": rng.integers(0, 8, _N).astype(float),
        }
    )
    return frame


def _outcome(frame: "pd.DataFrame", *, log_or: float) -> "pd.Series":
    rng = np.random.default_rng(11)
    linear = -1.5 + log_or * frame["exposure"] + 0.02 * (frame["age"] - 65.0)
    probability = 1.0 / (1.0 + np.exp(-linear))
    return pd.Series((rng.random(_N) < probability).astype(int))


def test_a_word_valued_covariate_no_longer_stops_the_declared_model() -> None:
    """The load-bearing one: this exact design raised before."""

    frame = _frame()
    result = fit_estimator(
        cohort=None,
        X=frame,
        y=_outcome(frame, log_or=0.9),
        kind="logistic",
        term="exposure",
    )

    assert result.converged is True
    assert result.point_estimate is not None
    # exp(0.9) = 2.46; a wide band, because this asserts that the declared
    # exposure was fitted, not that a random draw hit a particular value.
    assert 1.8 < result.point_estimate < 3.3
    assert result.n == _N


def test_the_reference_level_is_recorded_and_does_not_depend_on_row_order() -> None:
    """A contrast nobody can name is a number nobody can read.

    Row order is data, not design: shuffling the cohort must not silently
    change which level the coefficient is measured against.
    """

    frame = _frame()
    outcome = _outcome(frame, log_or=0.9)

    forward = fit_estimator(
        cohort=None, X=frame, y=outcome, kind="logistic", term="exposure"
    )
    shuffled_index = frame.sample(frac=1.0, random_state=3).index
    backward = fit_estimator(
        cohort=None,
        X=frame.loc[shuffled_index],
        y=outcome.loc[shuffled_index],
        kind="logistic",
        term="exposure",
    )

    assert "sex treatment-coded against 'Female'" in forward.notes
    assert forward.notes == backward.notes
    assert forward.point_estimate == pytest.approx(backward.point_estimate, rel=1e-9)


def test_asking_for_the_categorical_itself_fails_closed_naming_its_contrasts() -> None:
    """Picking a contrast for the caller is the guess ``term`` exists to remove."""

    frame = _frame()
    result = fit_estimator(
        cohort=None,
        X=frame,
        y=_outcome(frame, log_or=0.9),
        kind="logistic",
        term="sex",
    )

    assert result.converged is False
    assert result.point_estimate is None
    assert "sex=Male" in result.notes


def test_the_named_contrast_can_be_asked_for_directly() -> None:
    """Fail-closed must leave a way through, or it is just a wall."""

    frame = _frame()
    result = fit_estimator(
        cohort=None,
        X=frame,
        y=_outcome(frame, log_or=0.9),
        kind="logistic",
        term="sex=Male",
    )

    assert result.converged is True
    assert result.point_estimate is not None


def test_a_covariate_with_one_observed_level_does_not_silently_leave_the_model() -> (
    None
):
    """An adjustment set that shrinks by itself is a different study.

    One level yields no contrast columns, so the declared predictor would
    simply vanish from the design.  The existing rank guard refuses to drop a
    declared predictor, and this holds it to that.
    """

    frame = _frame()
    frame["site"] = "only_site"
    result = fit_estimator(
        cohort=None,
        X=frame,
        y=_outcome(frame, log_or=0.9),
        kind="logistic",
        term="exposure",
    )

    # Measured on the first draft: this converged, reported a plausible odds
    # ratio, and had quietly dropped `site` from the declared adjustment set --
    # the rank guard never saw it, because a column that was never built cannot
    # be dropped. An unconditional assertion, because a conditional one passed
    # vacuously against exactly that behaviour.
    assert result.converged is False
    assert result.point_estimate is None
    assert "site" in result.notes
    assert "one observed level" in result.notes


def test_a_numeric_only_design_is_untouched() -> None:
    """No categorical, no note, no behaviour change for existing callers."""

    frame = _frame().drop(columns=["sex"])
    result = fit_estimator(
        cohort=None,
        X=frame,
        y=_outcome(_frame(), log_or=0.9),
        kind="logistic",
        term="exposure",
    )

    assert result.converged is True
    assert "treatment-coded" not in (result.notes or "")
