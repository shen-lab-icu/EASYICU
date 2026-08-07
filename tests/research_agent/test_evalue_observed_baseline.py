"""An E-value must be anchored to this run's observed event rate, or not reported.

Converting an odds ratio to a risk ratio needs the cohort's baseline event
rate. The host used to seed ``baseline_prev = 0.1`` and let every failure mode
fall through to it -- no outcome-rate product, an unreadable file, an
unparseable cell -- so a reported E-value could be computed at an invented 10%
rate. It carried an explanatory note, but a note does not make an invented
scientific input defensible, and nothing in the value itself reveals the
substitution.

The guessed rate is not a rounding detail. It moves the reported number:

    OR = 2.0, assumed rate 0.10  ->  E = 3.04
    OR = 2.0, observed rate 0.214 ->  E = 2.68

Both are "the E-value for OR=2.0". Only one of them is about this cohort.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.methods.sensitivity import (
    BaselinePrevalenceRequiredError,
    compute_e_value,
)
from easyicu.research_agent.orchestration.finalize import resolve_observed_event_rate


# ---------------------------------------------------------------------------
# The kernel must refuse, so no caller can reintroduce the guess
# ---------------------------------------------------------------------------


def test_an_odds_ratio_without_a_baseline_rate_is_refused():
    with pytest.raises(BaselinePrevalenceRequiredError):
        compute_e_value(estimate=2.0, estimate_type="or")


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_a_baseline_rate_outside_the_unit_interval_is_refused(bad: float):
    """0 and 1 are refused too: neither has a risk ratio to convert into."""

    with pytest.raises(BaselinePrevalenceRequiredError):
        compute_e_value(estimate=2.0, estimate_type="or", baseline_prevalence=bad)


def test_rr_and_hr_need_no_baseline_rate():
    """Only the OR conversion needs it; refusing the others would be wrong."""

    assert compute_e_value(estimate=2.0, estimate_type="rr").e_value > 1.0
    assert compute_e_value(estimate=2.0, estimate_type="hr").e_value > 1.0


def test_the_assumed_rate_really_did_move_the_reported_number():
    """The measurement that justifies refusing rather than noting.

    If the old default had been numerically harmless, a note would have been a
    defensible design. It was not.
    """

    assumed = compute_e_value(
        estimate=2.0, estimate_type="or", baseline_prevalence=0.1
    ).e_value
    observed = compute_e_value(
        estimate=2.0, estimate_type="or", baseline_prevalence=0.214
    ).e_value

    assert round(assumed, 2) == 3.04
    assert round(observed, 2) == 2.68
    # 0.36 apart. Not a rounding artefact, and in the direction that
    # OVERSTATES robustness to unmeasured confounding.
    assert abs(assumed - observed) == pytest.approx(0.358, abs=0.01)


def test_the_note_records_the_rate_the_conversion_actually_used():
    result = compute_e_value(
        estimate=2.0, estimate_type="or", baseline_prevalence=0.214
    )
    assert result.note is not None
    assert "0.2140" in result.note


# ---------------------------------------------------------------------------
# The resolver reads the run's own product, and refuses rather than reduces
# ---------------------------------------------------------------------------


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_single_declared_rate_resolves(tmp_path):
    path = _write(tmp_path, "outcome_rate.csv", "outcome_rate\n0.214\n")
    resolved = resolve_observed_event_rate(path)

    assert resolved.value == pytest.approx(0.214)
    assert resolved.cause == ""
    assert resolved.source_column == "outcome_rate"


def test_a_missing_product_resolves_to_nothing_not_to_a_default(tmp_path):
    resolved = resolve_observed_event_rate(None)
    assert resolved.value is None
    assert resolved.cause == "no_outcome_rate_product"


def test_an_unreadable_product_resolves_to_nothing(tmp_path):
    resolved = resolve_observed_event_rate(tmp_path / "does_not_exist.csv")
    assert resolved.value is None
    assert resolved.cause == "outcome_rate_unreadable"


def test_a_product_with_no_usable_rate_resolves_to_nothing(tmp_path):
    # Present, parseable, and carries nothing that is a rate in (0, 1).
    path = _write(tmp_path, "outcome_rate.csv", "n_events,n_total\n2021,9445\n")
    resolved = resolve_observed_event_rate(path)

    assert resolved.value is None
    assert resolved.cause == "outcome_rate_has_no_usable_rate"


def test_disagreeing_rates_are_refused_not_reduced(tmp_path):
    """The failure the old code could not even see.

    It kept the LAST matching cell, so a per-group product silently anchored
    the E-value to whichever row sorted last. First, last, and mean are three
    different scientific choices; none of them is stated to the reader.
    """

    path = _write(
        tmp_path,
        "outcome_rate.csv",
        "group,outcome_rate\nexposed,0.31\nunexposed,0.18\n",
    )
    resolved = resolve_observed_event_rate(path)

    assert resolved.value is None
    assert resolved.cause == "outcome_rate_ambiguous"
    assert resolved.candidates == (0.18, 0.31)
    assert "0.1800" in resolved.reason and "0.3100" in resolved.reason


def test_the_same_rate_repeated_across_rows_is_not_ambiguous(tmp_path):
    """Refusing agreement would be a false block, not a safe one."""

    path = _write(
        tmp_path,
        "outcome_rate.csv",
        "stratum,outcome_rate\na,0.214\nb,0.214\n",
    )
    resolved = resolve_observed_event_rate(path)

    assert resolved.value == pytest.approx(0.214)
    assert resolved.cause == ""


def test_every_unresolved_cause_is_distinct(tmp_path):
    """A shared cause string would make the finding unactionable.

    "E-values were not computed" is only useful if it says which of the four
    things went wrong -- a reader has to know whether to add a product, fix a
    file, or declare which rate is the baseline.
    """

    causes = {
        resolve_observed_event_rate(None).cause,
        resolve_observed_event_rate(tmp_path / "nope.csv").cause,
        resolve_observed_event_rate(_write(tmp_path, "a.csv", "n\n5\n")).cause,
        resolve_observed_event_rate(
            _write(tmp_path, "b.csv", "outcome_rate\n0.1\n0.9\n")
        ).cause,
    }
    assert len(causes) == 4, causes
