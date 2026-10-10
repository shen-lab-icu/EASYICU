"""An AUROC's interval follows how its validation rows depend.

One stay per patient keeps the DeLong interval.  When a patient contributes
several stays the patients are resampled, all of a patient's stays together,
so repeat stays cannot narrow the interval.  Synthetic data only.
"""

from __future__ import annotations

import numpy as np
import pytest

from easyicu.research_agent.methods.auc_interval import (
    BOOTSTRAP_DRAWS,
    CLUSTER_BOOTSTRAP_METHOD,
    DELONG_METHOD,
    DELONG_PAIRED_METHOD,
    AUCIntervalError,
    auc_interval,
    paired_auc_difference,
)
from easyicu.research_agent.methods.delong_auc import (
    delong_auc_ci,
    delong_difference,
    delong_test,
)


def _rows(n: int = 400, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    outcome = (rng.random(n) < 0.3).astype(int)
    model = rng.normal(size=n) + 1.2 * outcome
    comparator = rng.normal(size=n) + 0.6 * outcome
    return outcome, model, comparator


def test_one_stay_per_patient_keeps_the_delong_interval() -> None:
    outcome, model, _ = _rows()
    interval = auc_interval(outcome, model, np.arange(outcome.size))
    reference = delong_auc_ci(outcome, model)

    assert interval.method == DELONG_METHOD
    assert (interval.auc, interval.se, interval.ci_low, interval.ci_high) == (
        reference.auc,
        reference.se,
        reference.ci_low,
        reference.ci_high,
    )
    assert interval.bootstrap_n == interval.bootstrap_skipped_n == 0


def test_repeat_stays_are_resampled_by_patient() -> None:
    outcome, model, _ = _rows()
    # Every patient's stay recorded twice: the copies add no information.
    doubled_outcome = np.repeat(outcome, 2)
    doubled_model = np.repeat(model, 2)
    patients = np.repeat(np.arange(outcome.size), 2)

    interval = auc_interval(doubled_outcome, doubled_model, patients)

    assert interval.method == CLUSTER_BOOTSTRAP_METHOD
    assert interval.bootstrap_n == BOOTSTRAP_DRAWS
    assert interval.subject_n == outcome.size and interval.row_n == 2 * outcome.size
    assert interval.auc == pytest.approx(delong_auc_ci(outcome, model).auc, abs=1e-12)
    assert interval.ci_low <= interval.auc <= interval.ci_high
    # Resampling patients keeps the width of the undoubled rows ...
    once = delong_auc_ci(outcome, model).se
    assert 0.75 * once < interval.se < 1.33 * once
    # ... which reading the copies as independent stays would understate.
    assert interval.se > 1.2 * delong_auc_ci(doubled_outcome, doubled_model).se
    # Seeded: the same rows give the same interval.
    assert auc_interval(doubled_outcome, doubled_model, patients) == interval


def test_few_events_still_give_every_draw_both_classes() -> None:
    # One patient holds the only events.  Patients with and without an event
    # are drawn apart, so no draw loses them.
    patients = np.concatenate([np.arange(300), [0]])
    outcome = np.zeros(patients.size, dtype=int)
    outcome[0] = outcome[-1] = 1
    score = np.linspace(0.0, 1.0, patients.size)

    interval = auc_interval(outcome, score, patients)

    assert interval.method == CLUSTER_BOOTSTRAP_METHOD
    assert interval.bootstrap_skipped_n == 0


def test_too_many_draws_without_both_classes_stop_the_interval() -> None:
    # Every patient has an event; only patient 0 also has a stay without one,
    # so about a third of the draws hold no non-event stay.
    patients = np.concatenate([np.arange(20), [0]])
    outcome = np.ones(patients.size, dtype=int)
    outcome[-1] = 0
    score = np.linspace(0.0, 1.0, patients.size)

    with pytest.raises(AUCIntervalError) as stopped:
        auc_interval(outcome, score, patients)
    assert stopped.value.code == "auc_bootstrap_degenerate"


def test_a_paired_difference_on_independent_rows_is_delong() -> None:
    outcome, model, comparator = _rows()
    difference = paired_auc_difference(
        outcome, model, comparator, np.arange(outcome.size)
    )
    auc_a, auc_b, z, p_value = delong_test(outcome, model, comparator)

    assert difference.method == DELONG_PAIRED_METHOD
    assert (difference.auc_a, difference.auc_b) == (auc_a, auc_b)
    assert difference.z == pytest.approx(z, rel=1e-12)
    assert difference.p_value == pytest.approx(p_value, rel=1e-12)
    assert difference.ci_low < difference.difference < difference.ci_high
    assert difference.ci_high - difference.difference == pytest.approx(
        1.959963984540054 * difference.se
    )


def test_a_paired_difference_on_repeat_stays_is_a_patient_bootstrap() -> None:
    outcome, model, comparator = _rows()
    patients = np.repeat(np.arange(outcome.size), 2)

    difference = paired_auc_difference(
        np.repeat(outcome, 2), np.repeat(model, 2), np.repeat(comparator, 2), patients
    )

    assert difference.method == CLUSTER_BOOTSTRAP_METHOD
    assert difference.z is None and difference.p_value is None
    assert difference.difference == pytest.approx(
        difference.auc_a - difference.auc_b, abs=1e-15
    )
    assert difference.ci_low < difference.difference < difference.ci_high
    independent = paired_auc_difference(
        outcome, model, comparator, np.arange(outcome.size)
    )
    assert 0.75 * independent.se < difference.se < 1.33 * independent.se


def test_the_delong_test_reads_the_difference_it_reports() -> None:
    outcome, model, comparator = _rows(seed=11)
    auc_a, auc_b, difference, variance = delong_difference(outcome, model, comparator)
    tested_a, tested_b, z, _p = delong_test(outcome, model, comparator)

    assert (auc_a, auc_b) == (tested_a, tested_b)
    assert difference == pytest.approx(auc_a - auc_b, abs=1e-15)
    assert z == pytest.approx(difference / np.sqrt(variance), rel=1e-12)
    assert delong_test(outcome, model, model)[2:] == (0.0, 1.0)


def test_incomplete_subject_ids_are_refused() -> None:
    outcome, model, _ = _rows(n=50)
    patients = np.arange(outcome.size).astype(object)
    patients[3] = None
    with pytest.raises(ValueError):
        auc_interval(outcome, model, patients)
    with pytest.raises(ValueError):
        auc_interval(outcome, model, np.arange(outcome.size - 1))
