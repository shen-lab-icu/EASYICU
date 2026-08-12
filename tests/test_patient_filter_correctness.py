from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.base import BaseICULoader, detect_database_type
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

    assert result["age"].tolist() == [65, 70, 75]
    assert result["los_hours"].tolist() == [24.0, 24.0, 48.0]
    assert result["survived"].tolist() == [True, False, pd.NA]


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
