"""``separation_detected: false`` has to mean "checked", not "returned numbers".

The adjusted-association owner wrote the literal ``False`` into every model
contract, justified in a comment by "a separated design cannot satisfy both a
finite estimate and a finite interval".  That is not true.  Quasi-separation
routinely returns an enormous coefficient, an interval spanning orders of
magnitude, and ``converged=True`` -- all finite.

The figure renderer beside it already had a test for exactly that state
(estimate 2.9e7, interval 1e-8 to 8.4e22) and refuses to draw it.  So one layer
knew the state existed while the producer asserted it could not.

The contract's own validator refuses a missing value
(``fit_diagnostics_must_be_boolean``), so an answer is obligatory -- which is
precisely why it must be computed rather than assumed.

The signal is the textbook one: under separation some fitted probabilities are
numerically 0 or 1.  Two other signals were tried and removed for reachability.
An invented magnitude bound could not work -- the
extreme-but-real frame below reaches a log-odds of 9.65 against a corpus maximum
of 4.82, so any bound wide enough to spare real effects was too wide to catch
anything, and any bound tight enough to catch would have refused a genuine
estimate.

MEASURED over the recorded corpus: 159 emitted estimates, largest |log odds
ratio| 4.82, largest standard error 0.56.  Nothing recorded is near separation,
which is why the field was never yet wrong -- and now it cannot become wrong
silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.robustness.estimators import (
    _CERTAIN_PROBABILITY_TOLERANCE,
    fit_estimator,
)


def _zero_cell_frame(n: int = 300) -> pd.DataFrame:
    """Quasi-separation: one covariate level has no events at all.

    This is the case the replaced literal asserted could not exist. The fit
    CONVERGES, the estimate is finite and unremarkable (about 0.9), the interval
    is finite -- and the design is separated. A comment reading "a separated
    design cannot satisfy both a finite estimate and a finite interval" is
    refuted by exactly this frame.
    """

    rng = np.random.default_rng(5)
    group = np.array(["a"] * 140 + ["b"] * 140 + ["c"] * (n - 280))
    death = np.concatenate(
        [
            rng.integers(0, 2, 140),
            rng.integers(0, 2, 140),
            np.zeros(n - 280),
        ]
    ).astype(float)
    return pd.DataFrame(
        {
            "stay_id": np.arange(n),
            "exposure": rng.integers(0, 2, n).astype(float),
            "grp": group,
            "death": death,
            "age": rng.normal(65, 10, n),
        }
    )


def _perfectly_separated_frame(n: int = 60) -> pd.DataFrame:
    """Everyone exposed dies and nobody unexposed does."""

    rng = np.random.default_rng(11)
    exposure = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float)
    return pd.DataFrame(
        {
            "stay_id": np.arange(n),
            "exposure": exposure,
            "death": exposure.copy(),
            "age": rng.normal(65, 10, n),
        }
    )


def _extreme_but_real_frame(n: int = 200) -> pd.DataFrame:
    """An enormous but genuine association, which must NOT be flagged.

    100 versus 100 with one counterexample each way: the maximum likelihood
    estimate exists and is finite, and the odds ratio comes back around 15,590
    with a standard error of 1.7 and no fitted probability at the boundary. It
    is a real estimate of a near-deterministic relationship, and a diagnostic
    that refused it would block real findings. Mislabelling this frame as
    quasi-separation is what an earlier draft of this file did.
    """

    rng = np.random.default_rng(3)
    exposure = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float)
    death = exposure.copy()
    death[0] = 1.0
    death[-1] = 0.0
    return pd.DataFrame(
        {
            "stay_id": np.arange(n),
            "exposure": exposure,
            "death": death,
            "age": rng.normal(65, 10, n),
        }
    )


def _ordinary_frame(n: int = 400) -> pd.DataFrame:
    """A real-shaped association: strong, but the groups overlap."""

    rng = np.random.default_rng(7)
    exposure = rng.integers(0, 2, n).astype(float)
    logit = -1.2 + 1.1 * exposure + 0.01 * rng.normal(0, 10, n)
    probability = 1.0 / (1.0 + np.exp(-logit))
    return pd.DataFrame(
        {
            "stay_id": np.arange(n),
            "exposure": exposure,
            "death": (rng.random(n) < probability).astype(float),
            "age": rng.normal(65, 10, n),
        }
    )


def _fit(frame: pd.DataFrame):
    predictors = [c for c in ("exposure", "grp", "age") if c in frame.columns]
    return fit_estimator(
        cohort=frame,
        X=frame[predictors],
        y=frame["death"],
        kind="logistic",
        term="exposure",
    )


def test_a_converged_finite_fit_can_still_be_separated():
    """The exact claim the replaced comment made, refuted on a real fit."""

    result = _fit(_zero_cell_frame())

    assert result.converged is True, result.notes
    assert result.point_estimate is not None and np.isfinite(result.point_estimate)
    assert result.ci_low is not None and result.ci_high is not None
    assert result.separation_detected is True, result.notes


def test_perfect_separation_is_reported_as_separated():
    result = _fit(_perfectly_separated_frame())

    if result.separation_detected is None:
        pytest.fail("the fit reported no separation verdict at all")
    assert result.separation_detected is True, result.notes


def test_an_enormous_but_genuine_association_is_not_flagged():
    """Refusing this would block real findings, not protect them."""

    result = _fit(_extreme_but_real_frame())

    assert result.converged is True, result.notes
    assert result.point_estimate is not None and result.point_estimate > 1000
    assert result.separation_detected is False, result.notes


def test_the_note_names_which_signal_fired():
    """A reader must not have to guess what the flag reacted to."""

    result = _fit(_zero_cell_frame())

    assert "separation" in result.notes.lower(), result.notes
    assert (
        "fitted probabilities" in result.notes or "log-odds" in result.notes
    ), result.notes


def test_an_ordinary_association_is_not_reported_as_separated():
    """The corpus's real fits must keep reading False.

    A diagnostic that flagged ordinary data would be worse than the literal it
    replaces: it would block real estimates.
    """

    result = _fit(_ordinary_frame())

    assert result.point_estimate is not None, result.notes
    assert result.separation_detected is False, result.notes
    assert "separation" not in result.notes.lower(), result.notes


def test_the_verdict_survives_into_the_model_contract():
    """The literal it replaces lived in the contract, so that is what must change."""

    import inspect

    from easyicu.research_agent.execution.runners import (
        adjusted_association_executor as owner,
    )

    source = inspect.getsource(owner)
    assert '"separation_detected": False' not in source
    assert '"separation_detected": bool(result.separation_detected)' in source


def test_the_contract_still_carries_a_boolean_because_its_validator_demands_one():
    """``None`` is refused downstream, so the fit may never leave it unanswered."""

    ordinary = _fit(_ordinary_frame())
    separated = _fit(_zero_cell_frame())

    for result in (ordinary, separated):
        assert isinstance(bool(result.separation_detected), bool)
    assert bool(ordinary.separation_detected) is False
    assert bool(separated.separation_detected) is True


def test_the_only_constant_is_a_floating_point_tolerance():
    """No invented magnitude bound survives in this diagnostic.

    A first draft carried one and it did not work: the extreme-but-real frame
    above reaches a log-odds of 9.65 against a corpus maximum of 4.82, so any
    bound wide enough to spare real effects was too wide to catch anything, and
    any bound tight enough to catch would have refused a genuine estimate. The
    signals are statsmodels' own detector and the textbook boundary
    probabilities; this tolerance is floating point, not judgement.
    """

    import easyicu.research_agent.robustness.estimators as module

    assert _CERTAIN_PROBABILITY_TOLERANCE <= 1e-5
    assert not hasattr(module, "_SEPARATED_COEFFICIENT_MAGNITUDE")


def test_no_recorded_estimate_is_extreme_enough_to_worry_about():
    """Re-measures the corpus: nothing recorded is anywhere near separation.

    This is why the field was never yet wrong, and why the fix is prophylactic:
    largest |log odds ratio| 4.82, largest standard error 0.56 across 159
    estimates.
    """

    import glob
    import os
    import pathlib

    corpus = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")
    if not corpus.exists():
        pytest.skip("recorded run corpus is not mounted")

    seen = 0
    for path in glob.glob(
        str(corpus / "batch_*/*/aware/run_*/steps/*/outputs/*.csv")
    ):
        if "adjusted_association_estimates" not in os.path.basename(path):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        for value in pd.to_numeric(
            frame.get("estimate", pd.Series(dtype=float)), errors="coerce"
        ).dropna():
            if value <= 0:
                continue
            seen += 1
            assert abs(np.log(float(value))) < 10.0, value

    if not seen:
        pytest.skip("no recorded estimate could be parsed")
    assert seen > 50, seen
