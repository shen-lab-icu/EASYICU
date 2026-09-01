"""The events must be counted on the rows the model actually fitted.

MEASURED 2026-07-30 in ``e2_lactate_mortality`` of batch
``..._88d3983_canonical9_full02``.  Step ``05_primary_adjusted_association`` --
the study's primary result -- was claimed by the deterministic owner, generated,
approved by the concept audit, and **executed successfully**: ``returncode=0``,
``runner_provenance.json`` written, ``primary_or = 1.3572170056325177``.

Then the host's own ``primary_model_contract`` validator recomputed the model's
denominator from the bound cohort and refused the step over one issue::

    model_denominator_or_event_mismatch
      expected_n = 515   reported_n = 515          <- agree
      expected_event_n = 78   reported_event_n = 102   <- disagree

The bound cohort (``cohort_analysis_development_sample.parquet``, 1,000 rows)
holds 102 deaths in total and 515 rows with ``lact_max`` observed, and those 515
rows hold **78** deaths.  So the owner reported the analysis set's denominator
with the whole cohort's numerator -- a 19.8% event rate where the truth was
15.1%, a 31% relative overstatement -- because ``n`` came from
``result.n`` (post ``dropna``) while ``n_events`` was recounted on
``model_frame`` (pre ``dropna``), one line apart.

The gate was right.  The number went into three places at once -- the estimates
row, ``model_contracts[0].event_n`` and ``step_summary.n_events`` -- so a
manuscript would have carried it, and refusing the step was the correct
outcome.  What it cost was the whole step: a correct primary estimate was
computed and thrown away, and the same shape was live in ``m1`` too.

The fix moves the count to the only layer that knows which rows were used.
``fit_estimator`` performs the ``dropna``; it now reports ``n_events`` from the
surviving rows, and the owner reads it rather than deriving its own.  Fixing it
there rather than in the owner also keeps the robustness replay -- which goes
through the same function -- counting the same way, so a sensitivity estimate
stays comparable with the primary one.

Verified against the real cohort: ``n=515``, ``n_events=78``, and the odds ratio
is bit-for-bit the recorded ``1.3572170056325177`` -- the estimate was never
wrong, only the count reported beside it.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("statsmodels.api")

from easyicu.research_agent.execution.runners.adjusted_association_executor import (  # noqa: E402
    AdjustedAssociationError,
    run_adjusted_association_from_env,
)
from easyicu.research_agent.robustness.estimators import (  # noqa: E402
    EstimatorResult,
    fit_estimator,
)


def _frame_with_missing_exposure() -> "pd.DataFrame":
    """The measured shape: events concentrated in the rows the fit will drop.

    120 rows.  The 40 rows with a missing exposure are all deaths, so a
    whole-frame count says 60 events and a complete-case count says 20.  Any
    implementation that recounts on its own frame reports the wrong one, and
    the two differ far enough that no rounding can hide it.
    """

    rng = np.random.default_rng(20260730)
    n_complete, n_dropped = 80, 40
    exposure = np.concatenate(
        [rng.normal(4.0, 1.5, n_complete), np.full(n_dropped, np.nan)]
    )
    outcome = np.concatenate(
        [
            np.array([1] * 20 + [0] * (n_complete - 20)),
            np.ones(n_dropped, dtype=int),
        ]
    )
    return pd.DataFrame(
        {
            "lact_max": exposure,
            "death": outcome,
            "age": rng.normal(65.0, 12.0, n_complete + n_dropped),
            "sex": rng.choice(["Male", "Female"], n_complete + n_dropped),
        }
    )


_WHOLE_FRAME_EVENTS = 60
_COMPLETE_CASE_EVENTS = 20
_COMPLETE_CASE_ROWS = 80


def test_the_fixture_really_separates_the_two_counts():
    """Otherwise every assertion below passes for the wrong reason."""

    frame = _frame_with_missing_exposure()
    assert int(frame["death"].sum()) == _WHOLE_FRAME_EVENTS
    observed = frame[frame["lact_max"].notna()]
    assert len(observed) == _COMPLETE_CASE_ROWS
    assert int(observed["death"].sum()) == _COMPLETE_CASE_EVENTS


# ---------------------------------------------------------------------------
# The estimator owns the count, because it owns the row set
# ---------------------------------------------------------------------------


def test_the_estimator_counts_events_on_the_rows_it_kept():
    frame = _frame_with_missing_exposure()
    result = fit_estimator(
        cohort=None,
        X=frame[["lact_max", "age", "sex"]],
        y=frame["death"],
        kind="logistic",
        term="lact_max",
    )
    assert result.converged
    assert result.n == _COMPLETE_CASE_ROWS
    assert result.n_events == _COMPLETE_CASE_EVENTS
    assert result.n_events != _WHOLE_FRAME_EVENTS


def test_the_numerator_never_exceeds_its_own_denominator():
    """The property the host's contract checks, stated directly."""

    frame = _frame_with_missing_exposure()
    result = fit_estimator(
        cohort=None,
        X=frame[["lact_max", "age", "sex"]],
        y=frame["death"],
        kind="logistic",
        term="lact_max",
    )
    assert result.n_events is not None
    assert 0 <= result.n_events <= result.n


def test_a_continuous_outcome_reports_no_event_count():
    """``continuous_outcome_event_n_must_be_null`` is a contract clause."""

    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "lact_max": rng.normal(4.0, 1.5, 90),
            "los": rng.normal(6.0, 2.0, 90),
            "age": rng.normal(65.0, 12.0, 90),
        }
    )
    result = fit_estimator(
        cohort=None,
        X=frame[["lact_max", "age"]],
        y=frame["los"],
        kind="linear",
        term="lact_max",
    )
    assert result.converged
    assert result.n_events is None


# ---------------------------------------------------------------------------
# The owner reports what the estimator returned, and refuses to invent it
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _step_out_dir(tmp_path, monkeypatch):
    """The owner writes its estimates table where the host tells it to."""

    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path))


def _run(frame: "pd.DataFrame") -> dict:
    return run_adjusted_association_from_env(
        requirement_id="primary_logistic_lact_max_death",
        exposure="lact_max",
        outcome="death",
        covariates=["age", "sex"],
        estimator_kind="logistic",
        analysis_set="complete_case",
        analysis_role="primary",
        method_family="logistic_regression",
        model_terms=[
            {
                "name": "lact_max",
                "role": "exposure",
                "coding": "continuous",
                "transform": "identity",
            },
            {
                "name": "age",
                "role": "covariate",
                "coding": "continuous",
                "transform": "identity",
            },
            {
                "name": "sex",
                "role": "covariate",
                "coding": "categorical",
                "levels": ["Female", "Male"],
                "reference_level": "Female",
                "transform": "treatment_contrast",
            },
        ],
        frame=frame,
        emit_step_summary=False,
    )


def test_the_owner_reports_the_complete_case_event_count():
    summary = _run(_frame_with_missing_exposure())
    assert summary["n_total"] == _COMPLETE_CASE_ROWS
    assert summary["n_events"] == _COMPLETE_CASE_EVENTS


def test_every_place_the_count_appears_agrees():
    """It travels to three artifacts; two of them feed the manuscript."""

    summary = _run(_frame_with_missing_exposure())
    contract = summary["model_contracts"][0]
    assert contract["n"] == summary["n_total"] == _COMPLETE_CASE_ROWS
    assert contract["event_n"] == summary["n_events"] == _COMPLETE_CASE_EVENTS


def test_the_owner_refuses_a_denominator_without_its_numerator(monkeypatch):
    """Fail closed rather than emit a null the contract will reject anyway."""

    import easyicu.research_agent.execution.runners.adjusted_association_executor as owner

    real = owner.fit_estimator

    def _without_event_count(*args, **kwargs):
        result = real(*args, **kwargs)
        return EstimatorResult(
            result.point_estimate,
            result.ci_low,
            result.ci_high,
            result.se,
            result.n,
            result.converged,
            result.notes,
            result.terms,
            n_events=None,
        )

    monkeypatch.setattr(owner, "fit_estimator", _without_event_count)
    with pytest.raises(AdjustedAssociationError, match="without reporting the events"):
        _run(_frame_with_missing_exposure())


def test_the_estimate_itself_is_unchanged_by_this_fix():
    """The fit was always right; only the count beside it was not.

    A change to the reported denominator/numerator that also moved the estimate
    would mean the row set changed, which is a different -- and much worse --
    kind of fix.
    """

    frame = _frame_with_missing_exposure()
    summary = _run(frame)
    direct = fit_estimator(
        cohort=None,
        X=frame[["lact_max", "age", "sex"]],
        y=frame["death"],
        kind="logistic",
        term="lact_max",
    )
    assert summary["adjusted_effect"] == direct.point_estimate
    assert summary["n_total"] == direct.n
