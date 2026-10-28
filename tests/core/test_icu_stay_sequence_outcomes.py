"""ICU readmission, the readmitted-stay attribute and ICU-free days.

Both readmission columns read the source's whole ICU stay table: a stay's
answer does not depend on which stays an extract asked for.  Stays of one
hospitalisation chain into one ICU episode when the next begins within an
hour of the previous discharge.
"""

from __future__ import annotations

import pandas as pd
import pytest

import easyicu.scores.outcomes as outcomes

READMISSION = ["icu_readmission", "icu_stay_is_readmission"]


def _tables(tables):
    return lambda database, data_path, table: tables[table].copy()


def _mimic_iv(monkeypatch):
    icu = pd.DataFrame(
        [
            # subject 1: a 30-minute transfer, then a return six days later.
            (1, 10, 11, "2020-01-01 00:00", "2020-01-03 00:00"),
            (1, 10, 12, "2020-01-03 00:30", "2020-01-04 00:00"),
            (1, 10, 13, "2020-01-08 00:00", "2020-01-09 00:00"),
            # subject 2: dies in the ICU.
            (2, 20, 21, "2020-01-01 00:00", "2020-01-02 00:00"),
            # subject 3: two stays sharing an admission time, then a later
            # hospitalisation outside the first window.
            (3, 30, 31, "2020-01-01 00:00", "2020-01-02 00:00"),
            (3, 30, 32, "2020-01-01 00:00", "2020-01-02 00:00"),
            (3, 31, 33, "2020-02-01 00:00", "2020-02-03 00:00"),
            # subject 4: a second ICU stay in a different hospitalisation.
            (4, 40, 41, "2020-01-01 00:00", "2020-01-02 00:00"),
            (4, 41, 42, "2020-01-05 00:00", "2020-01-06 00:00"),
        ],
        columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
    )
    icu["los"] = (pd.to_datetime(icu.outtime) - pd.to_datetime(icu.intime)).dt.total_seconds() / 86400
    patients = pd.DataFrame({"subject_id": [1, 2, 3, 4], "dod": [None, "2020-01-01", None, None]})
    admissions = pd.DataFrame(
        {
            "hadm_id": [10, 20, 30, 31, 40, 41],
            "deathtime": [None, "2020-01-01 12:00", None, None, None, None],
            "hospital_expire_flag": [0, 1, 0, 0, 0, 0],
        }
    )
    monkeypatch.setattr(
        outcomes, "_raw_table", _tables({"icustays": icu, "patients": patients, "admissions": admissions})
    )


def _by_stay(frame, identifier):
    return frame.set_index(identifier)


def _read(column, *stays):
    """True/False, or None where the column has no answer."""
    values = [None if pd.isna(column[stay]) else bool(column[stay]) for stay in stays]
    return values[0] if len(values) == 1 else values


def test_mimic_iv_readmission_reads_episodes_not_stays(monkeypatch):
    _mimic_iv(monkeypatch)
    out = _by_stay(outcomes.load_outcomes("miiv", output_concepts=READMISSION), "stay_id")

    returned = out["icu_readmission"]
    readmitted = out["icu_stay_is_readmission"]
    # The transfer is the same episode; the return six days later is not.
    assert _read(returned, 11, 12, 13) == [True, True, False]
    assert _read(readmitted, 11, 12, 13) == [False, False, True]
    # Death in the ICU: not at risk of a return, and the stay is a first one.
    assert _read(returned, 21) is None and _read(readmitted, 21) is False
    # Two stays sharing an admission time cannot be ordered: no answer for
    # that hospitalisation, while the patient's next one is read on its own.
    assert _read(returned, 31, 32) == [None, None] and _read(readmitted, 31, 32) == [None, None]
    assert _read(returned, 33) is False and _read(readmitted, 33) is False
    # Another hospitalisation's ICU stay is not a readmission.
    assert _read(returned, 41) is False and _read(readmitted, 42) is False


def test_readmission_reads_the_whole_stay_table_whatever_the_extract(monkeypatch):
    _mimic_iv(monkeypatch)
    out = outcomes.load_outcomes("miiv", patient_ids=[13], output_concepts=READMISSION)

    assert out["stay_id"].tolist() == [13]
    assert bool(out.loc[0, "icu_stay_is_readmission"]) is True


def test_mimic_iv_icu_free_days_count_every_icu_stay_in_the_window(monkeypatch):
    _mimic_iv(monkeypatch)
    out = _by_stay(outcomes.load_outcomes("miiv"), "stay_id")
    free = out["icu_free_days_28"]

    # 2 days, 23.5 hours after a transfer, and the 1-day return.
    assert free[11] == pytest.approx(28 - (2 + 23.5 / 24 + 1))
    assert free[13] == pytest.approx(27.0)
    assert free[21] == 0.0
    # Overlapping stays count once; a stay after day 28 does not count.
    assert free[31] == pytest.approx(27.0) and free[32] == pytest.approx(27.0)
    # ICU time in a later hospitalisation inside the window counts.
    assert free[41] == pytest.approx(26.0)


def test_icu_free_days_need_day_28_survival(monkeypatch):
    _mimic_iv(monkeypatch)
    real_followup = outcomes._mimic_stay_death_days

    def without_followup(database, data_path):
        frame = real_followup(database, data_path)
        frame.loc[frame["_stay"] == 13, "followup_days"] = float("nan")
        return frame

    monkeypatch.setattr(outcomes, "_mimic_stay_death_days", without_followup)
    out = _by_stay(outcomes.load_outcomes("miiv"), "stay_id")

    assert pd.isna(out.loc[13, "mort_28d"])
    assert pd.isna(out.loc[13, "icu_free_days_28"])


def test_named_outcomes_merge_on_the_stay(monkeypatch):
    _mimic_iv(monkeypatch)
    out = outcomes.load_outcomes(
        "miiv", output_concepts=["icu_death", "mort_28d", "icu_free_days_28", *READMISSION]
    )

    assert len(out) == 9 and out["stay_id"].is_unique
    assert {"icu_death", "mort_28d", "icu_free_days_28", *READMISSION} <= set(out.columns)
    assert out.attrs["source_identifier"] == "stay_id"


def _eicu(monkeypatch):
    rows = [
        # hospitalisation 100: a 30-minute transfer, then a return.
        (1, 100, 1, 0, 600, "Alive"),
        (2, 100, 2, -630, 300, "Alive"),
        (3, 100, 3, -2000, 100, "Alive"),
        # 200: a return in which the patient dies.
        (4, 200, 1, 0, 100, "Alive"),
        (5, 200, 2, -500, 50, "Expired"),
        # 300: death after a direct transfer ends the first episode.
        (6, 300, 1, 0, 100, "Alive"),
        (7, 300, 2, -110, 60, "Expired"),
        # 400: two stays carrying the same visit number.
        (8, 400, 1, 0, 100, "Alive"),
        (9, 400, 1, -500, 100, "Alive"),
    ]
    patient = pd.DataFrame(
        rows,
        columns=[
            "patientunitstayid", "patienthealthsystemstayid", "unitvisitnumber",
            "hospitaladmitoffset", "unitdischargeoffset", "unitdischargestatus",
        ],
    )
    monkeypatch.setattr(outcomes, "_raw_table", _tables({"patient": patient}))


def test_eicu_readmission_reads_unit_stays_on_the_hospital_clock(monkeypatch):
    _eicu(monkeypatch)
    out = _by_stay(outcomes.load_outcomes("eicu", output_concepts=READMISSION), "patientunitstayid")

    returned = out["icu_readmission"]
    readmitted = out["icu_stay_is_readmission"]
    assert list(out.columns) == READMISSION
    assert _read(returned, 1, 2, 3) == [True, True, False]
    assert _read(readmitted, 1, 2, 3) == [False, False, True]
    assert _read(returned, 4, 5) == [True, None] and _read(readmitted, 5) is True
    assert _read(returned, 6, 7) == [None, None]
    assert _read(readmitted, 6, 7) == [False, False]
    assert _read(returned, 8, 9) == [None, None] and _read(readmitted, 8, 9) == [None, None]


def test_eicu_has_no_free_days_or_followup(monkeypatch):
    _eicu(monkeypatch)

    assert outcomes.load_outcomes("eicu").empty
    out = outcomes.load_outcomes("eicu", output_concepts=["icu_free_days_28", *READMISSION])
    assert "icu_free_days_28" not in out.columns


def test_databases_without_a_stay_sequence_return_nothing(monkeypatch):
    monkeypatch.setattr(outcomes, "_raw_table", _tables({}))

    assert outcomes.load_outcomes("sic", output_concepts=READMISSION).empty
    assert outcomes.load_outcomes("hirid", output_concepts=READMISSION).empty
