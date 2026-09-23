"""Contracts for the source-bound hospital follow-up owner (MIMIC-IV and eICU)."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.hospital_mortality_followup import (
    HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS,
    HospitalMortalityFollowupError,
    derive_eicu_hospital_mortality_followup,
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


def _patient() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patientunitstayid": [201, 202, 203, 204, 205, 206, 207, 208],
            "hospitaldischargestatus": [
                "Expired",  # event at 0 h is a valid zero-time event
                "Alive",
                "Alive",  # hospital discharge recorded before unit discharge
                None,  # status missing
                "Unknown",  # status outside the eICU domain
                "Expired",  # offset missing
                "Expired",  # death before ICU admission
                "Alive",  # discharge before ICU admission
            ],
            "hospitaldischargeoffset": [0, 2880, 600, 1440, 1440, None, -30, -60],
            "unitdischargeoffset": [0, 2000, 660, 1400, 1400, 100, 10, 10],
        }
    )


def test_eicu_patient_table_derives_the_same_hospital_axis() -> None:
    result = derive_eicu_hospital_mortality_followup(_patient(), database="eicu_demo")

    assert tuple(result.frame.columns) == HOSPITAL_MORTALITY_FOLLOWUP_COLUMNS
    assert result.frame[["stay_id", "hospital_death", "hospital_followup_time_hours"]].to_dict(
        orient="records"
    ) == [
        {"stay_id": 201, "hospital_death": 1, "hospital_followup_time_hours": 0.0},
        {"stay_id": 202, "hospital_death": 0, "hospital_followup_time_hours": 48.0},
        {"stay_id": 203, "hospital_death": 0, "hospital_followup_time_hours": 10.0},
    ]
    assert result.frame.loc[0, "death_time_hours"] == 0.0
    assert pd.isna(result.frame.loc[1, "death_time_hours"])
    assert result.exclusions.to_dict(orient="records") == [
        {"stay_id": 204, "reason_code": "hospital_mortality_status_missing"},
        {"stay_id": 205, "reason_code": "hospital_mortality_status_invalid"},
        {"stay_id": 206, "reason_code": "hospital_discharge_offset_missing"},
        {"stay_id": 207, "reason_code": "hospital_death_before_icu_admission"},
        {"stay_id": 208, "reason_code": "hospital_discharge_before_icu_admission"},
    ]
    receipt = result.receipt
    assert receipt["schema_version"] == "easyicu.eicu_hospital_mortality_followup/1"
    assert receipt["database"] == "eicu_demo"
    assert receipt["time_origin"] == "icu_admission"
    assert receipt["event"]["definition"] == "patient.hospitaldischargestatus == 'Expired'"
    assert receipt["zero_time_event_stays"] == 1
    assert receipt["event_stays"] == 1
    assert receipt["censored_stays"] == 2
    assert receipt["chronology_notes"] == {
        "hospital_discharge_before_icu_discharge_stays": 1
    }
    assert receipt["exclusion_counts"] == {
        "hospital_death_before_icu_admission": 1,
        "hospital_discharge_before_icu_admission": 1,
        "hospital_discharge_offset_missing": 1,
        "hospital_mortality_status_invalid": 1,
        "hospital_mortality_status_missing": 1,
    }
    assert receipt["privacy"]["identifier_values_returned"] is False
    # The receipt vocabulary is shared with the MIMIC-IV derivation.
    mimic = derive_mimic_iv_hospital_mortality_followup(_icustays(), _admissions())
    assert set(mimic.receipt) - {"chronology_notes"} == set(receipt) - {"chronology_notes"}


def test_eicu_derivation_fails_closed_on_ambiguous_identity_or_database() -> None:
    duplicated = _patient()
    duplicated.loc[1, "patientunitstayid"] = duplicated.loc[0, "patientunitstayid"]
    with pytest.raises(
        HospitalMortalityFollowupError, match="hospital_followup_patient_key_nonunique"
    ):
        derive_eicu_hospital_mortality_followup(duplicated)
    with pytest.raises(
        HospitalMortalityFollowupError, match="hospital_followup_database_unsupported"
    ):
        derive_eicu_hospital_mortality_followup(_patient(), database="miiv")
    with pytest.raises(
        HospitalMortalityFollowupError, match="hospital_followup_patient_columns_missing"
    ):
        derive_eicu_hospital_mortality_followup(
            _patient().drop(columns=["hospitaldischargeoffset"])
        )
