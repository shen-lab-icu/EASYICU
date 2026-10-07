"""A hospital death is timed by its recorded time, not by its calendar day.

MIMIC-IV records the date of death (``patients.dod``, a DATE) and, for a death
in hospital, its time (``admissions.deathtime``).  The fixed-horizon mortality
loader counted every death in whole calendar days from the ICU admission date:

- a death 25 hours after admission was one day, inside a 24-hour landmark, so
  the landmark suites (alive at the landmark = follow-up beyond it) dropped it
  as a death before the landmark;
- a death 2 hours after a late-evening admission was one day too, alive at a
  12-hour landmark;
- a death 28 days and 6 hours after admission was a death within 28 days.

A death that the stay's own admission records is now timed by that record.  A
death after discharge keeps its calendar day, the resolution its date
supports, and a death recorded before the ICU admission time on the day of
admission stays a death at admission.  Synthetic tables only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.scores import outcomes


def _tables(*, intime, dod=None, deathtime=None):
    """One MIMIC-IV stay in hospital admission 10."""

    return {
        "icustays": pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [10],
                "stay_id": [100],
                "intime": [intime],
                "los": [2.0],
            }
        ),
        "patients": pd.DataFrame({"subject_id": [1], "dod": [dod]}),
        "admissions": pd.DataFrame({"hadm_id": [10], "deathtime": [deathtime]}),
    }


def _load(monkeypatch, tables):
    monkeypatch.setattr(
        outcomes,
        "_raw_table",
        lambda database, data_path, table: tables[table].copy(),
    )
    return outcomes.load_outcomes("miiv").set_index("stay_id").loc[100]


def test_a_death_25_hours_after_admission_outlives_a_24_hour_landmark(
    monkeypatch,
) -> None:
    stay = _load(
        monkeypatch,
        _tables(
            intime="2180-01-01 10:00:00",
            dod="2180-01-02",
            deathtime="2180-01-02 11:00:00",
        ),
    )

    assert bool(stay["mort_28d"]) is True
    assert stay["followup_days_28d"] == pytest.approx(25 / 24)
    # Alive at a 24-hour landmark: follow-up beyond it.
    assert stay["followup_days_28d"] > 24 / 24


def test_a_death_2_hours_after_a_late_admission_precedes_a_12_hour_landmark(
    monkeypatch,
) -> None:
    stay = _load(
        monkeypatch,
        _tables(
            intime="2180-01-01 23:00:00",
            dod="2180-01-02",
            deathtime="2180-01-02 01:00:00",
        ),
    )

    assert stay["followup_days_28d"] == pytest.approx(2 / 24)
    assert not stay["followup_days_28d"] > 12 / 24


def test_a_death_6_hours_past_28_days_is_no_death_within_28_days(
    monkeypatch,
) -> None:
    stay = _load(
        monkeypatch,
        _tables(
            intime="2180-01-01 12:00:00",
            dod="2180-01-29",
            deathtime="2180-01-29 18:00:00",
        ),
    )

    assert bool(stay["mort_28d"]) is False
    assert stay["followup_days_28d"] == 28.0
    assert bool(stay["mort_90d"]) is True
    assert stay["followup_days_90d"] == pytest.approx(28.25)


def test_a_recorded_death_time_without_a_date_of_death_is_still_a_death(
    monkeypatch,
) -> None:
    stay = _load(
        monkeypatch,
        _tables(intime="2180-01-01 12:00:00", deathtime="2180-01-03 12:00:00"),
    )

    # Not a survivor followed for a year: the admission records the death.
    assert bool(stay["mort_28d"]) is True
    assert stay["followup_days_28d"] == pytest.approx(2.0)


def test_a_death_after_discharge_keeps_its_calendar_day(monkeypatch) -> None:
    stay = _load(monkeypatch, _tables(intime="2180-01-01 12:00:00", dod="2180-01-06"))

    assert bool(stay["mort_28d"]) is True
    assert stay["followup_days_28d"] == 5.0


def test_a_death_recorded_before_admission_on_its_day_is_a_death_at_admission(
    monkeypatch,
) -> None:
    stay = _load(
        monkeypatch,
        _tables(
            intime="2180-01-01 12:00:00",
            dod="2180-01-01",
            deathtime="2180-01-01 11:30:00",
        ),
    )

    assert bool(stay["mort_28d"]) is True
    assert stay["followup_days_28d"] == 0.0


def test_a_death_recorded_before_admission_without_a_date_is_still_a_death(
    monkeypatch,
) -> None:
    stay = _load(
        monkeypatch,
        _tables(intime="2180-01-01 12:00:00", deathtime="2180-01-01 11:30:00"),
    )

    # Its day is the recorded time's; it is no survivor followed for a year.
    assert bool(stay["mort_28d"]) is True
    assert stay["followup_days_28d"] == 0.0


def test_an_unparseable_death_time_is_refused(monkeypatch) -> None:
    tables = _tables(
        intime="2180-01-01 12:00:00", dod="2180-01-03", deathtime="not-a-time"
    )

    with pytest.raises(ValueError, match="deathtime"):
        _load(monkeypatch, tables)


def test_an_admission_listed_twice_is_refused(monkeypatch) -> None:
    tables = _tables(intime="2180-01-01 12:00:00", dod="2180-01-03")
    tables["admissions"] = pd.concat([tables["admissions"]] * 2, ignore_index=True)

    with pytest.raises(ValueError, match="admissions: identity keys"):
        _load(monkeypatch, tables)
