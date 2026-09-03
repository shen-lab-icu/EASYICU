"""Contracts for the source-bound MIMIC-IV hospital follow-up owner."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.hospital_mortality_followup import (
    HospitalMortalityFollowupError,
    derive_mimic_iv_hospital_mortality_followup,
)


def _icustays() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [101, 102, 103, 104, 105, 106, 107, 108],
            "hadm_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "intime": [
                "2026-01-01 00:00:00",
                "2026-01-02 00:00:00",
                "2026-01-03 00:00:00",
                "2026-01-04 00:00:00",
                "2026-01-05 00:00:00",
                "2026-01-06 00:00:00",
                "2026-01-07 00:00:00",
                "not-a-timestamp",
            ],
        }
    )


def _admissions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hadm_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "hospital_expire_flag": [1, 0, 1, 0, 1, 0, 2, 0],
            "deathtime": [
                "2026-01-01 00:00:00",  # zero-time event is valid
                None,
                "2026-01-02 23:00:00",  # before ICU admission
                "2026-01-04 12:00:00",  # survivor contradiction
                None,  # event time missing
                None,
                None,
                None,
            ],
            "dischtime": [
                "2026-01-03 00:00:00",
                "2026-01-04 00:00:00",
                "2026-01-04 00:00:00",
                "2026-01-05 00:00:00",
                "2026-01-07 00:00:00",
                "2026-01-05 23:00:00",  # before ICU admission
                "2026-01-08 00:00:00",
                "2026-01-09 00:00:00",
            ],
        }
    )


def test_derives_source_bound_event_and_hospital_censoring_axis() -> None:
    result = derive_mimic_iv_hospital_mortality_followup(_icustays(), _admissions())

    assert result.frame[["stay_id", "hospital_death", "hospital_followup_time_hours"]].to_dict(
        orient="records"
    ) == [
        {"stay_id": 101, "hospital_death": 1, "hospital_followup_time_hours": 0.0},
        {"stay_id": 102, "hospital_death": 0, "hospital_followup_time_hours": 48.0},
    ]
    assert result.frame.loc[0, "death_time_hours"] == 0.0
    # NaN is structural non-applicability for a censored stay, not a missing
    # event timestamp, so assert it separately rather than comparing dicts.
    assert pd.isna(result.frame.loc[1, "death_time_hours"])
    assert result.exclusions.to_dict(orient="records") == [
        {"stay_id": 103, "reason_code": "hospital_death_before_icu_admission"},
        {"stay_id": 104, "reason_code": "hospital_survivor_death_time_inconsistent"},
        {"stay_id": 105, "reason_code": "hospital_death_time_missing"},
        {"stay_id": 106, "reason_code": "hospital_discharge_before_icu_admission"},
        {"stay_id": 107, "reason_code": "hospital_mortality_flag_invalid"},
        {"stay_id": 108, "reason_code": "icu_intime_unavailable"},
    ]
    assert result.receipt["zero_time_event_stays"] == 1
    assert result.receipt["event_stays"] == 1
    assert result.receipt["censored_stays"] == 1
    assert result.receipt["exclusion_counts"] == {
        "hospital_death_time_missing": 1,
        "hospital_death_before_icu_admission": 1,
        "hospital_discharge_before_icu_admission": 1,
        "hospital_mortality_flag_invalid": 1,
        "hospital_survivor_death_time_inconsistent": 1,
        "icu_intime_unavailable": 1,
    }
    assert "/Volumes/" not in str(result.receipt)
    assert result.receipt["privacy"]["source_paths_returned"] is False


def test_missing_admission_is_an_explicit_row_exclusion() -> None:
    admissions = _admissions().loc[lambda frame: frame["hadm_id"].ne(8)].copy()
    result = derive_mimic_iv_hospital_mortality_followup(_icustays(), admissions)

    assert result.exclusions.iloc[-1].to_dict() == {
        "stay_id": 108,
        "reason_code": "hospital_admission_missing",
    }


@pytest.mark.parametrize(
    ("table", "column", "code"),
    [
        ("icustays", "stay_id", "hospital_followup_icustays_key_nonunique"),
        ("admissions", "hadm_id", "hospital_followup_admissions_key_nonunique"),
    ],
)
def test_ambiguous_raw_join_fails_closed(table: str, column: str, code: str) -> None:
    icustays = _icustays()
    admissions = _admissions()
    if table == "icustays":
        icustays.loc[1, column] = icustays.loc[0, column]
    else:
        admissions.loc[1, column] = admissions.loc[0, column]

    with pytest.raises(HospitalMortalityFollowupError, match=code):
        derive_mimic_iv_hospital_mortality_followup(icustays, admissions)
