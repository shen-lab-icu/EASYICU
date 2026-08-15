from __future__ import annotations

from inspect import signature
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
from easyicu.io.data_converter import DataConverter
from easyicu.patient_filter import (
    FilterCriteria,
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
    ("directory_name", "column", "canonical", "web_alias"),
    [
        ("mimic-iv", "icustay_id", "mimic", "miii"),
        ("mimic-iii", "stay_id", "miiv", "miiv"),
    ],
)
def test_database_schema_overrides_conflicting_directory_name_across_layers(
    tmp_path, directory_name, column, canonical, web_alias
) -> None:
    from easyicu.webserver.dataio import _detect_database

    prepared = tmp_path / directory_name
    prepared.mkdir()
    pd.DataFrame({column: [1]}).to_csv(prepared / "icustays.csv", index=False)
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False

    assert loader._detect_database(None, prepared) == canonical
    assert DataConverter(prepared, verbose=False).database == canonical
    assert _detect_database(prepared) == web_alias


def test_unrelated_eicu_ancestor_does_not_override_mimic_schema(tmp_path) -> None:
    from easyicu.webserver.dataio import _detect_database

    prepared = tmp_path / "eicu_archive" / "selected_export"
    prepared.mkdir(parents=True)
    pd.DataFrame({"stay_id": [1]}).to_csv(prepared / "icustays.csv", index=False)
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False

    assert loader._detect_database(None, prepared) == "miiv"
    assert DataConverter(prepared, verbose=False).database == "miiv"
    assert _detect_database(prepared) == "miiv"


def test_mimic_bucket_layout_does_not_override_the_stay_id_generation(
    tmp_path,
) -> None:
    """MIMIC-III and IV may both use converter bucket directories."""

    from easyicu.webserver.dataio import _detect_database

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "chartevents_bucket").mkdir()
    pd.DataFrame({"stay_id": [1]}).to_csv(prepared / "icustays.csv", index=False)
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False

    assert loader._detect_database(None, prepared) == "miiv"
    assert DataConverter(prepared, verbose=False).database == "miiv"
    assert _detect_database(prepared) == "miiv"


def test_unreadable_identity_table_does_not_fall_back_to_the_directory_label(
    tmp_path,
) -> None:
    from easyicu.webserver.dataio import _detect_database

    prepared = tmp_path / "mimic-iv"
    prepared.mkdir()
    (prepared / "icustays.parquet").write_bytes(b"not parquet")
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False

    for detect in (
        lambda: loader._detect_database(None, prepared),
        lambda: DataConverter(prepared, verbose=False).database,
        lambda: _detect_database(prepared),
    ):
        with pytest.raises(DatabaseDetectionError) as caught:
            detect()
        assert caught.value.code == "database_detection_schema_unreadable"


def test_schema_detection_accepts_official_uppercase_table_names_on_case_sensitive_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    pd.DataFrame({"stay_id": [1]}).to_csv(
        prepared / "ICUSTAYS.csv.gz", index=False, compression="gzip"
    )
    real_is_file = Path.is_file

    def case_sensitive_is_file(path: Path) -> bool:
        if path.parent == prepared:
            actual_names = {entry.name for entry in prepared.iterdir()}
            if path.name not in actual_names:
                return False
        return real_is_file(path)

    # APFS is normally case-insensitive, so emulate Linux's exact-name lookup.
    # The detector must enumerate children case-insensitively rather than rely
    # on the host filesystem accepting a lower-case spelling.
    monkeypatch.setattr(Path, "is_file", case_sensitive_is_file)
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False

    assert loader._detect_database(None, prepared) == "miiv"
    assert DataConverter(prepared, verbose=False).database == "miiv"


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


def test_mixed_prepared_database_schemas_fail_closed(tmp_path) -> None:
    """One directory must never be silently claimed by the first schema seen."""

    pd.DataFrame({"stay_id": [1]}).to_csv(tmp_path / "icustays.csv", index=False)
    pd.DataFrame({"patientunitstayid": [1]}).to_csv(
        tmp_path / "patient.csv", index=False
    )
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False

    with pytest.raises(DatabaseDetectionError) as caught:
        loader._detect_database(None, tmp_path)

    assert caught.value.code == "database_detection_ambiguous"
    assert caught.value.candidates == ("eicu", "miiv")


def test_explicit_database_rejects_an_unrecognized_prepared_path(tmp_path) -> None:
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False

    with pytest.raises(DatabaseDetectionError) as caught:
        loader._setup_data_path(tmp_path, "miiv")

    assert caught.value.code == "database_path_unrecognized"


def test_converter_normalizes_an_explicit_public_database_alias(tmp_path) -> None:
    converter = DataConverter(tmp_path, database="mimic-iv", verbose=False)

    assert converter.database == "miiv"


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


def test_public_source_config_unknown_name_fails_closed() -> None:
    with pytest.raises(KeyError, match="Unknown database profile 'mivv'"):
        load_src_cfg("mivv")


def test_custom_source_config_requires_explicit_registry() -> None:
    from easyicu.config import DataSourceConfig, DataSourceRegistry

    registry = DataSourceRegistry([DataSourceConfig(name="custom", tables={})])

    assert load_src_cfg("custom", registry=registry).name == "custom"


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


@pytest.mark.parametrize(
    ("database", "loader_name", "stay_id_column"),
    [
        ("miiv", "_load_miiv_demographics", "stay_id"),
        ("mimic", "_load_mimic3_demographics", "icustay_id"),
    ],
)
def test_mimic_partial_extract_does_not_claim_patient_global_first_stay(
    monkeypatch,
    database: str,
    loader_name: str,
    stay_id_column: str,
) -> None:
    patient_filter = PatientFilter(database=database, data_path="/unused")
    tables = {
        "icustays": pd.DataFrame(
            {
                "subject_id": [10, 10],
                "hadm_id": [102, 103],
                stay_id_column: [2, 3],
                "intime": ["2021-01-01", "2022-01-01"],
            }
        ),
        "patients": pd.DataFrame({"subject_id": [10]}),
        "admissions": pd.DataFrame(
            {"subject_id": [10, 10], "hadm_id": [102, 103]}
        ),
    }
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda name: tables[name].copy(),
    )

    result = getattr(patient_filter, loader_name)()
    patient_filter._demographics = result

    assert "first_icu_stay" not in result.columns
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(first_icu_stay=True)
    assert caught.value.code == "patient_filter_criterion_unavailable"


@pytest.mark.parametrize(
    ("database", "loader_name", "stay_id_column"),
    [
        ("miiv", "_load_miiv_demographics", "stay_id"),
        ("mimic", "_load_mimic3_demographics", "icustay_id"),
    ],
)
def test_mimic_hospital_survival_preserves_missing_and_unknown_flags(
    monkeypatch,
    database: str,
    loader_name: str,
    stay_id_column: str,
) -> None:
    patient_filter = PatientFilter(database=database, data_path="/unused")
    tables = {
        "icustays": pd.DataFrame(
            {
                "subject_id": [10, 20, 30, 40, 50],
                "hadm_id": [101, 201, 301, 401, 501],
                stay_id_column: [1, 2, 3, 4, 5],
            }
        ),
        "patients": pd.DataFrame({"subject_id": [10, 20, 30, 40, 50]}),
        "admissions": pd.DataFrame(
            {
                "subject_id": [10, 20, 30, 40],
                "hadm_id": [101, 201, 301, 401],
                "hospital_expire_flag": [0, 1, None, 2],
            }
        ),
    }
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda name: tables[name].copy(),
    )

    result = getattr(patient_filter, loader_name)()
    patient_filter._demographics = result

    assert str(result["survived"].dtype) == "boolean"
    assert result["survived"].tolist() == [True, False, pd.NA, pd.NA, pd.NA]
    assert patient_filter.filter(survived=True) == [1]


@pytest.mark.parametrize(
    ("database", "loader_name", "stay_id_column"),
    [
        ("miiv", "_load_miiv_demographics", "stay_id"),
        ("mimic", "_load_mimic3_demographics", "icustay_id"),
    ],
)
def test_mimic_deathtime_does_not_replace_standard_hospital_flag(
    monkeypatch,
    database: str,
    loader_name: str,
    stay_id_column: str,
) -> None:
    patient_filter = PatientFilter(database=database, data_path="/unused")
    tables = {
        "icustays": pd.DataFrame(
            {
                "subject_id": [10, 20],
                "hadm_id": [101, 201],
                stay_id_column: [1, 2],
            }
        ),
        "patients": pd.DataFrame({"subject_id": [10, 20]}),
        "admissions": pd.DataFrame(
            {
                "subject_id": [10, 20],
                "hadm_id": [101, 201],
                "deathtime": [None, "2022-01-01"],
            }
        ),
    }
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda name: tables[name].copy(),
    )

    result = getattr(patient_filter, loader_name)()
    patient_filter._demographics = result

    assert "survived" not in result.columns
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(survived=True)
    assert caught.value.code == "patient_filter_criterion_unavailable"


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


def test_eicu_top_coded_age_publishes_an_open_interval(monkeypatch) -> None:
    patient_filter = PatientFilter(database="eicu", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "patientunitstayid": [1, 2],
                "age": ["> 89", "72"],
            }
        ),
    )

    result = patient_filter._load_eicu_demographics()
    patient_filter._demographics = result

    assert result["age_lower"].tolist() == [90, 72]
    assert result["age_upper"].tolist() == [pd.NA, 72]
    assert result["age_is_censored"].tolist() == [True, False]
    assert result["age_is_grouped"].tolist() == [True, False]
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(age_max=90)
    assert caught.value.code == "patient_filter_grouped_age_indeterminate"


def test_mimic3_top_coded_age_publishes_an_open_interval(monkeypatch) -> None:
    patient_filter = PatientFilter(database="mimic", data_path="/unused")
    tables = {
        "icustays": pd.DataFrame(
            {
                "subject_id": [10, 20],
                "hadm_id": [101, 201],
                "icustay_id": [1, 2],
                "intime": ["2000-01-01", "2000-01-01"],
            }
        ),
        "patients": pd.DataFrame(
            {
                "subject_id": [10, 20],
                "dob": ["1700-01-01", "1930-01-01"],
            }
        ),
        "admissions": pd.DataFrame(
            {"subject_id": [10, 20], "hadm_id": [101, 201]}
        ),
    }
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda name: tables[name].copy(),
    )

    result = patient_filter._load_mimic3_demographics()
    patient_filter._demographics = result

    assert result["age"].tolist() == pytest.approx([90, 70], abs=0.01)
    assert result["age_lower"].tolist() == pytest.approx([90, 70], abs=0.01)
    assert pd.isna(result.loc[0, "age_upper"])
    assert result.loc[1, "age_upper"] == pytest.approx(70, abs=0.01)
    assert result["age_is_censored"].tolist() == [True, False]
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(age_min=91)
    assert caught.value.code == "patient_filter_grouped_age_indeterminate"


def test_mimic4_top_coded_age_preserves_the_shifted_open_interval(monkeypatch) -> None:
    patient_filter = PatientFilter(database="miiv", data_path="/unused")
    tables = {
        "icustays": pd.DataFrame(
            {
                "subject_id": [10, 20],
                "hadm_id": [101, 201],
                "stay_id": [1, 2],
                "intime": ["2022-01-01", "2022-01-01"],
            }
        ),
        "patients": pd.DataFrame(
            {
                "subject_id": [10, 20],
                "anchor_age": [91, 70],
                "anchor_year": [2020, 2020],
            }
        ),
        "admissions": pd.DataFrame(
            {"subject_id": [10, 20], "hadm_id": [101, 201]}
        ),
    }
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda name: tables[name].copy(),
    )

    result = patient_filter._load_miiv_demographics()
    patient_filter._demographics = result

    assert result["age"].tolist() == [93, 72]
    assert result["age_lower"].tolist() == [92, 72]
    assert result["age_upper"].tolist() == [pd.NA, 72]
    assert result["age_is_censored"].tolist() == [True, False]
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(age_max=93)
    assert caught.value.code == "patient_filter_grouped_age_indeterminate"


def test_sicdb_rounded_age_publishes_a_conservative_interval(monkeypatch) -> None:
    patient_filter = PatientFilter(database="sic", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {"CaseID": [1, 2], "AgeOnAdmission": [65, 90]}
        ),
    )

    result = patient_filter._load_sic_demographics()
    patient_filter._demographics = result

    assert result["age_lower"].tolist() == [60, 85]
    assert result["age_upper"].tolist() == [70, pd.NA]
    assert result["age_is_grouped"].tolist() == [True, True]
    assert result["age_is_censored"].tolist() == [False, True]
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(age_min=65)
    assert caught.value.code == "patient_filter_grouped_age_indeterminate"


def test_hirid_binned_age_publishes_conservative_intervals(monkeypatch) -> None:
    patient_filter = PatientFilter(database="hirid", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {"patientid": [1, 2], "age": [70, 90]}
        ),
    )

    result = patient_filter._load_hirid_demographics()

    assert result["age_lower"].tolist() == [65, 85]
    assert result["age_upper"].tolist() == [75, pd.NA]
    assert result["age_is_grouped"].tolist() == [True, True]
    assert result["age_is_censored"].tolist() == [False, True]


def test_hirid_binned_age_thresholds_fail_closed(monkeypatch) -> None:
    patient_filter = PatientFilter(database="hirid", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {"patientid": [1, 2], "age": [70, 90]}
        ),
    )
    patient_filter._demographics = patient_filter._load_hirid_demographics()

    for criterion in ({"age_min": 70}, {"age_max": 90}, {"age_min": 92}):
        with pytest.raises(PatientFilterCriterionError) as caught:
            patient_filter.filter(**criterion)
        assert caught.value.code == "patient_filter_grouped_age_indeterminate"

    assert patient_filter.filter(age_min=80) == [2]
    assert patient_filter.filter(age_max=80) == [1]


def test_filter_criteria_declares_only_supported_filter_arguments() -> None:
    public_filter_arguments = set(signature(PatientFilter.filter).parameters) - {
        "self",
        "return_dataframe",
    }

    assert set(FilterCriteria.__dataclass_fields__) == public_filter_arguments


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


def test_aumc_first_stay_uses_absolute_admission_count_in_partial_extract(
    monkeypatch,
) -> None:
    patient_filter = PatientFilter(database="aumc", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "admissionid": [1, 2, 3, 4, 5, 6, 7],
                "patientid": [10, 10, 20, 30, 40, 50, 60],
                "admittedat": [100, 200, 0, 0, 0, 0, 0],
                "admissioncount": [2, 3, 1, 0, -1, None, "invalid"],
            }
        ),
    )

    result = patient_filter._load_aumc_demographics()

    assert result["first_icu_stay"].tolist() == [
        False,
        False,
        True,
        pd.NA,
        pd.NA,
        pd.NA,
        pd.NA,
    ]


def test_aumc_first_stay_is_unavailable_without_admission_count(monkeypatch) -> None:
    patient_filter = PatientFilter(database="aumc", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame(
            {
                "admissionid": [1, 2],
                "patientid": [10, 10],
                "admittedat": [100, 200],
            }
        ),
    )

    result = patient_filter._load_aumc_demographics()
    patient_filter._demographics = result

    assert "first_icu_stay" not in result.columns
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(first_icu_stay=True)
    assert caught.value.code == "patient_filter_criterion_unavailable"


def test_hirid_first_stay_is_unavailable_without_patient_linkage(monkeypatch) -> None:
    patient_filter = PatientFilter(database="hirid", data_path="/unused")
    monkeypatch.setattr(
        patient_filter,
        "_read_table",
        lambda _name: pd.DataFrame({"patientid": [123, 987]}),
    )

    result = patient_filter._load_hirid_demographics()
    patient_filter._demographics = result

    assert "first_icu_stay" not in result.columns
    with pytest.raises(PatientFilterCriterionError) as caught:
        patient_filter.filter(first_icu_stay=True)
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
