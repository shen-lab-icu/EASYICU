"""MIMIC-III ICU readmission reads icustays like MIMIC-IV (upper-case headers).

2026-11-03 extraction audit 2: ``icu_readmission`` was a structural
placeholder for MIMIC-III although its icustays table carries HADM_ID,
INTIME and OUTTIME for every stay.
"""

from __future__ import annotations

import pandas as pd

import easyicu.scores.outcomes as outcomes

READMISSION = ["icu_readmission", "icu_stay_is_readmission"]


def _mimic_iii(monkeypatch):
    icu = pd.DataFrame(
        [
            # admission 10: a 30-minute transfer, then a return days later.
            (1, 10, 11, "2101-01-01 00:00:00", "2101-01-03 00:00:00"),
            (1, 10, 12, "2101-01-03 00:30:00", "2101-01-04 00:00:00"),
            (1, 10, 13, "2101-01-08 00:00:00", "2101-01-09 00:00:00"),
            # admission 20: one stay, death in the ICU.
            (2, 20, 21, "2101-01-01 00:00:00", "2101-01-02 00:00:00"),
        ],
        columns=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
    )
    admissions = pd.DataFrame(
        {
            "HADM_ID": [10, 20],
            "DEATHTIME": [None, "2101-01-01 12:00:00"],
            "HOSPITAL_EXPIRE_FLAG": [0, 1],
        }
    )
    tables = {"icustays": icu, "admissions": admissions}
    monkeypatch.setattr(outcomes, "_raw_table", lambda database, data_path, table: tables[table].copy())


def test_mimic_iii_readmission_reads_episodes_of_one_admission(monkeypatch):
    _mimic_iii(monkeypatch)

    out = outcomes.load_outcomes("mimic", output_concepts=READMISSION).set_index("icustay_id")

    assert list(out.columns) == READMISSION
    # 11 and 12 are one episode (30-minute transfer) followed by a return (13).
    assert out.loc[[11, 12, 13], "icu_readmission"].tolist() == [True, True, False]
    assert out.loc[[11, 12, 13], "icu_stay_is_readmission"].tolist() == [False, False, True]
    # Died in the ICU: not at risk of a readmission.
    assert pd.isna(out.loc[21, "icu_readmission"])
    assert not out.loc[21, "icu_stay_is_readmission"]
