from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.base import (
    BaseICULoader,
    DatabaseDetectionError,
    detect_database_type,
)
from easyicu.config import load_src_cfg
from easyicu.database_config import SUPPORTED_DATABASES
from easyicu.databases.profiles import public_database_keys
from easyicu.patient_filter import (
    PatientFilter,
    PatientFilterCriterionError,
)


def _filter_with(frame: pd.DataFrame, *, database: str = "miiv") -> PatientFilter:
    patient_filter = PatientFilter(database=database, data_path="/unused")
    patient_filter._demographics = frame
    return patient_filter


def test_detect_database_type_passes_the_requested_data_path(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeLoader:
        database = "miiv"

        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("easyicu.base.BaseICULoader", FakeLoader)

    assert detect_database_type(tmp_path) == "miiv"
    assert captured["data_path"] == tmp_path


def test_flat_mimic_iv_is_detected_by_stay_id_schema_before_admissions_marker(
    tmp_path,
) -> None:
    pd.DataFrame({"stay_id": [1]}).to_parquet(tmp_path / "icustays.parquet")
    pd.DataFrame({"hadm_id": [10]}).to_parquet(tmp_path / "admissions.parquet")
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False

    assert loader._detect_database(None, tmp_path) == "miiv"


def test_unknown_database_path_fails_closed_instead_of_defaulting_to_miiv(
    tmp_path,
) -> None:
    with pytest.raises(DatabaseDetectionError) as caught:
        detect_database_type(tmp_path)

    assert caught.value.code == "database_detection_unavailable"
    assert caught.value.data_path == tmp_path


def test_conflicting_database_markers_fail_closed(tmp_path) -> None:
    (tmp_path / "numericitems").mkdir()
    pd.DataFrame({"CaseID": [1]}).to_parquet(tmp_path / "cases.parquet")
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False

    with pytest.raises(DatabaseDetectionError) as caught:
        loader._detect_database(None, tmp_path)

    assert caught.value.code == "database_detection_ambiguous"
    assert caught.value.candidates == ("aumc", "sic")


def test_explicit_database_rejects_an_unrecognized_prepared_path(tmp_path) -> None:
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False

    with pytest.raises(DatabaseDetectionError) as caught:
        loader._setup_data_path(tmp_path, "miiv")

    assert caught.value.code == "database_path_unrecognized"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("miii", "mimic"), ("sicdb", "sic"), ("mimic-iv", "miiv")],
)
def test_patient_filter_normalizes_public_database_aliases(
    alias: str,
    canonical: str,
) -> None:
    assert PatientFilter(database=alias).database == canonical


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("miii", "mimic"), ("sicdb", "sic"), ("mimic-iv", "miiv")],
)
def test_public_source_config_resolves_registered_aliases(
    alias: str,
    canonical: str,
) -> None:
    assert load_src_cfg(alias).name == canonical


def test_legacy_database_constants_cover_every_public_canonical_key() -> None:
    assert set(SUPPORTED_DATABASES) == set(public_database_keys())


@pytest.mark.parametrize(
    ("kwargs", "criterion"),
    [
        ({"age_min": 18}, "age"),
        ({"first_icu_stay": True}, "first_icu_stay"),
        ({"los_min": 24}, "los"),
        ({"gender": "F"}, "gender"),
        ({"survived": True}, "survived"),
    ],
)
def test_requested_criterion_missing_from_source_fails_closed(
    kwargs: dict[str, object],
    criterion: str,
) -> None:
    patient_filter = _filter_with(pd.DataFrame({"patient_id": [1, 2]}))

    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(**kwargs)

    assert caught.value.code == "patient_filter_criterion_unavailable"
    assert caught.value.criterion == criterion


def test_sepsis_derivation_failure_is_not_converted_to_no_sepsis(
    monkeypatch,
) -> None:
    patient_filter = _filter_with(pd.DataFrame({"patient_id": [1, 2]}))

    def fail(**_kwargs):
        raise RuntimeError("broken concept mapping")

    monkeypatch.setattr("easyicu.api.load_sepsis3", fail)

    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(has_sepsis=False)

    assert caught.value.code == "patient_filter_sepsis_unascertainable"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_sepsis_result_without_a_stay_identifier_fails_closed(monkeypatch) -> None:
    patient_filter = _filter_with(pd.DataFrame({"patient_id": [1, 2]}))
    monkeypatch.setattr(
        "easyicu.api.load_sepsis3",
        lambda **_kwargs: pd.DataFrame({"sep3": [1]}),
    )

    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(has_sepsis=True)

    assert caught.value.code == "patient_filter_sepsis_identifier_unavailable"


def test_eicu_deidentified_over_89_age_maps_to_90(monkeypatch) -> None:
    patient_filter = PatientFilter(database="eicu", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "patientunitstayid": [1, 2, 3],
                "age": ["> 89", ">89", "72"],
            }
        ),
    )

    result = patient_filter._load_eicu_demographics()

    assert result["age"].tolist() == [90, 90, 72]


def test_eicu_missing_discharge_status_remains_unknown(monkeypatch) -> None:
    patient_filter = PatientFilter(database="eicu", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "patientunitstayid": [1, 2, 3],
                "hospitaldischargestatus": ["Alive", "Expired", None],
            }
        ),
    )

    result = patient_filter._load_eicu_demographics()
    patient_filter._demographics = result

    assert result["survived"].tolist() == [True, False, pd.NA]
    assert patient_filter.filter(survived=True) == [1]


def test_eicu_unit_visit_number_does_not_claim_patient_global_first_stay(
    monkeypatch,
) -> None:
    patient_filter = PatientFilter(database="eicu", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "patientunitstayid": [1, 2],
                "uniquepid": ["patient-a", "patient-a"],
                "patienthealthsystemstayid": [10, 20],
                "unitvisitnumber": [1, 1],
            }
        ),
    )

    result = patient_filter._load_eicu_demographics()
    patient_filter._demographics = result

    assert result["first_unit_stay_within_hospitalization"].tolist() == [True, True]
    assert "first_icu_stay" not in result.columns
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(first_icu_stay=True)
    assert caught.value.code == "patient_filter_criterion_unavailable"


def test_eicu_unit_discharge_status_does_not_substitute_for_hospital_survival(
    monkeypatch,
) -> None:
    patient_filter = PatientFilter(database="eicu", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "patientunitstayid": [1, 2],
                "unitdischargestatus": ["Alive", "Expired"],
            }
        ),
    )

    result = patient_filter._load_eicu_demographics()
    patient_filter._demographics = result

    assert result["unit_survived"].tolist() == [True, False]
    assert "survived" not in result.columns
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(survived=True)
    assert caught.value.code == "patient_filter_criterion_unavailable"


def test_aumc_destination_is_icu_survival_not_hospital_survival(monkeypatch) -> None:
    patient_filter = PatientFilter(database="aumc", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "admissionid": [1, 2, 3, 4],
                "destination": ["Home", "Deceased", "Overleden", None],
            }
        ),
    )

    result = patient_filter._load_aumc_demographics()
    patient_filter._demographics = result

    assert result["icu_survived"].tolist() == [True, False, False, pd.NA]
    assert "survived" not in result.columns
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(survived=True)
    assert caught.value.code == "patient_filter_criterion_unavailable"


def test_sicdb_demographics_use_official_age_los_and_hospital_survival(
    monkeypatch,
) -> None:
    patient_filter = PatientFilter(database="sicdb", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "CaseID": [1, 2, 3],
                "PatientID": [10, 10, 20],
                "OffsetAfterFirstAdmission": [0, 86_400, 0],
                "AgeOnAdmission": [65, 70, 75],
                "TimeOfStay": [90_000, 86_400, 172_800],
                "ICUOffset": [3_600, 0, 0],
                "HospitalDischargeType": [2026, 2028, None],
                "OffsetOfDeath": [10_000_000, 80_000, None],
                "Sex": [1, 0, 1],
            }
        ),
    )

    result = patient_filter._load_sic_demographics()
    patient_filter._demographics = result

    assert result["age"].tolist() == [65, 70, 75]
    assert result["los_hours"].tolist() == [24.0, 24.0, 48.0]
    assert result["first_icu_stay"].tolist() == [True, False, True]
    assert result["survived"].tolist() == [True, False, pd.NA]
    assert patient_filter.filter(first_icu_stay=True) == [1, 3]


def test_sicdb_first_stay_is_unknown_without_patient_level_evidence(
    monkeypatch,
) -> None:
    patient_filter = PatientFilter(database="sicdb", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame({"CaseID": [1, 2]}),
    )

    demographics = patient_filter._load_sic_demographics()
    patient_filter._demographics = demographics

    assert "first_icu_stay" not in demographics.columns
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(first_icu_stay=True)
    assert caught.value.code == "patient_filter_criterion_unavailable"


def test_sicdb_tied_first_admission_offsets_remain_unknown(monkeypatch) -> None:
    patient_filter = PatientFilter(database="sicdb", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "CaseID": [1, 2, 3],
                "PatientID": [10, 10, 10],
                "OffsetAfterFirstAdmission": [0, 0, 500],
            }
        ),
    )

    result = patient_filter._load_sic_demographics()

    assert result["first_icu_stay"].tolist() == [pd.NA, pd.NA, False]


def test_sicdb_positive_offsets_are_readmissions_when_first_case_is_missing(
    monkeypatch,
) -> None:
    patient_filter = PatientFilter(database="sicdb", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "CaseID": [1, 2, 3, 4],
                "PatientID": [10, 10, 20, 20],
                "OffsetAfterFirstAdmission": [86_400, 172_800, -1, None],
            }
        ),
    )

    result = patient_filter._load_sic_demographics()

    assert result["first_icu_stay"].tolist() == [False, False, pd.NA, pd.NA]


def test_aumc_age_threshold_that_splits_a_published_band_is_unavailable() -> None:
    patient_filter = _filter_with(
        pd.DataFrame(
            {
                "patient_id": [1, 2],
                "age": [64.5, 74.5],
                "age_lower": [60.0, 70.0],
                "age_upper": [69.0, 79.0],
            }
        ),
        database="aumc",
    )

    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(age_min=65)

    assert caught.value.code == "patient_filter_grouped_age_indeterminate"


def test_aumc_age_threshold_on_a_band_boundary_is_applied() -> None:
    patient_filter = _filter_with(
        pd.DataFrame(
            {
                "patient_id": [1, 2],
                "age": [64.5, 74.5],
                "age_lower": [60.0, 70.0],
                "age_upper": [69.0, 79.0],
            }
        ),
        database="aumc",
    )

    assert patient_filter.filter(age_min=70) == [2]


def test_aumc_loader_preserves_age_band_bounds(monkeypatch) -> None:
    patient_filter = PatientFilter(database="aumc", data_path=Path("/unused"))
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {"admissionid": [1, 2], "agegroup": ["60-69", "80+"]}
        ),
    )

    result = patient_filter._load_aumc_demographics()

    assert result["age_lower"].tolist() == [60.0, 80.0]
    assert result.loc[0, "age_upper"] == 69.0
    assert pd.isna(result.loc[1, "age_upper"])
