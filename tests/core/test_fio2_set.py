"""Synthetic parity checks for the explicitly selected hourly settings."""

import pandas as pd
import pytest

from easyicu.api import EXTRACT_MODULE_ORDER, EXTRACT_MODULES
from easyicu.api.concepts import clear_global_loader, load_concepts


@pytest.fixture(autouse=True)
def clear_source_cache():
    yield
    clear_global_loader()


def _write(root, table, records, columns):
    frame = pd.DataFrame(records, columns=columns)
    if table == "chartevents":
        extra = {"value": "", "valueuom": "%"}
        if "ICUSTAY_ID" in columns:
            extra = {name.upper(): value for name, value in extra.items()}
        frame = frame.assign(**extra)
    if table == "numericitems":
        frame = frame.assign(unit="%", tag=0)
    frame.to_parquet(root / f"{table}.parquet", index=False)


def _load(root, database, stay=1, **kwargs):
    return load_concepts(
        "fio2_set", database=database, data_path=root, patient_ids=[stay],
        r_compatible=False, concept_workers=1, parallel_workers=1, **kwargs,
    )


def _hour_values(frame):
    return dict(zip(frame["charttime"], frame["fio2_set"]))


def test_miiv_normalizes_before_filtering_on_icu_relative_hours(tmp_path):
    start = pd.Timestamp("2023-01-01 04:27:19")
    _write(tmp_path, "icustays", [(1, start)], ["stay_id", "intime"])
    _write(tmp_path, "chartevents", [
        (1, start - pd.Timedelta(seconds=1), 223835, 0.21),
        (1, start, 223835, 0.2),
        (1, start, 223835, 21),
        (1, start + pd.Timedelta(seconds=3599), 223835, 0.50),
        (1, start + pd.Timedelta(hours=1), 223835, 1),
        (1, start + pd.Timedelta(hours=1), 223835, 100.1),
        (1, start, 224700, 100),
    ], ["stay_id", "charttime", "itemid", "valuenum"])
    assert _hour_values(_load(tmp_path, "miiv")) == {-1.0: 21.0, 0.0: 35.5, 1.0: 100.0}


def test_aumc_pools_all_raw_items_and_keeps_exact_millisecond_boundaries(tmp_path):
    admit = 2_400_000_001
    _write(tmp_path, "admissions", [(1, admit)], ["admissionid", "admittedat"])
    _write(tmp_path, "numericitems", [
        (1, admit - 1, 12282, 21.0),
        (1, admit, 12282, 21.0),
        (1, admit, 12282, 21.0),
        (1, admit + 1, 12282, 21.0),
        (1, admit + 1, 12329, 80.0),
        (1, admit + 1, 12369, 90.0),
        (1, admit + 3_599_999, 6699, 99.0),
        (1, admit + 2, 12279, 21.0),
        (1, admit + 3_600_000, 6699, 100.0),
        (1, admit + 3_600_000, 6699, 0.7),
        (2, admit, 12282, 100.0),
    ], ["admissionid", "measuredat", "itemid", "value"])
    # Pooled median is 50.5. The median of the four item medians is 85.
    assert _hour_values(_load(tmp_path, "aumc")) == {-1.0: 21.0, 0.0: 50.5, 1.0: 100.0}


def test_eicu_numeric_settings_and_frozen_offset_limits(tmp_path):
    _write(tmp_path, "patient", [(1, 5)], ["patientunitstayid", "patienthealthsystemstayid"])
    _write(tmp_path, "respiratorycharting", [
        (1, -1, "FiO2", "respFlowSettings", "0.21"),
        (1, 0, "FiO2", "respFlowSettings", "0.21"),
        (1, 59, "FiO2", "respFlowSettings", "45%"),
        (1, 59, "FiO2", "respFlowSettings", "100"),
        (1, 59, "FiO2", "respFlowCare", "21"),
        (1, 59, "Oxygen", "respFlowSettings", "21"),
        (1, 60, "FiO2", "respFlowSettings", "21"),
        (1, 86399, "FiO2", "respFlowSettings", "1"),
        (1, 86400, "FiO2", "respFlowSettings", "21"),
    ], ["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respcharttypecat", "respchartvalue"])
    assert _hour_values(_load(tmp_path, "eicu")) == {0.0: 60.5, 1.0: 21.0, 1439.0: 100.0}


def test_sic_subtracts_icu_offset_on_the_seconds_clock(tmp_path):
    _write(tmp_path, "cases", [(1, 247)], ["CaseID", "ICUOffset"])
    _write(tmp_path, "data_float_h", [
        (1, 246, 2283, 21.0), (1, 247, 2283, 21.0),
        (1, 3846, 2283, 31.0), (1, 3846, 2283, 0.7),
        (1, 3847, 2283, 100.0), (1, 3847, 2278, 21.0),
    ], ["CaseID", "Offset", "DataID", "Val"])
    assert _hour_values(_load(tmp_path, "sic")) == {-1.0: 21.0, 0.0: 26.0, 1.0: 100.0}


def test_hirid_uses_only_setting_2010_as_recorded(tmp_path):
    start = pd.Timestamp("2023-01-01 04:27:19")
    _write(tmp_path, "general", [(1, start)], ["patientid", "admissiontime"])
    _write(tmp_path, "observations", [
        (1, start, 2010, 21.0), (1, start, 2010, 0.6), (1, start, 4000, 100.0),
        (1, start + pd.Timedelta(hours=1), 2010, 100.0),
    ], ["patientid", "datetime", "variableid", "value"])
    assert _hour_values(_load(tmp_path, "hirid")) == {0.0: 21.0, 1.0: 100.0}


@pytest.mark.parametrize("view_name,other_stay_source", [
    ("mimic_cv_dir_r13", "carevue"),
    ("mimic_cv_dir_r13", "metavision"),
    ("mimic_raw", "carevue"),
])
def test_carevue_view_and_chart_error_gate(tmp_path, view_name, other_stay_source):
    root = tmp_path / view_name
    root.mkdir()
    start = pd.Timestamp("2023-01-01 04:27:19")
    _write(root, "icustays", [(1, start, "carevue"), (2, start, other_stay_source)],
           ["ICUSTAY_ID", "INTIME", "DBSOURCE"])
    _write(root, "chartevents", [
        (1, start, 190, 0.4, None), (1, start, 7570, 60.0, "0"),
        (1, start, 7570, 21.0, "1"), (1, start, 189, 21.0, "0"),
    ], ["ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUENUM", "ERROR"])
    availability = {}
    frame = _load(root, "mimic", availability_sink=availability)
    if view_name == "mimic_cv_dir_r13" and other_stay_source == "carevue":
        assert _hour_values(frame) == {0.0: 50.0}
    else:
        assert frame.empty
        assert availability["fio2_set"].reason_code == "carevue_view_required"


def test_fio2_set_has_its_own_opt_in_module():
    assert EXTRACT_MODULES["vent_settings"] == ["fio2_set"]
    assert "vent_settings" not in EXTRACT_MODULE_ORDER
    assert {module for module, concepts in EXTRACT_MODULES.items() if "fio2_set" in concepts} == {"vent_settings"}
