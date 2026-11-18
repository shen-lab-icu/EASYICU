"""Synthetic parity checks for the explicitly selected hourly settings."""

import json

import pandas as pd
import pytest

from easyicu.api import EXTRACT_MODULE_ORDER, EXTRACT_MODULES
from easyicu.api.concepts import clear_global_loader, load_concepts
from easyicu.base import BaseICULoader
from easyicu.concept.stay_tables import selected_stay_ids
from tests.support.bucket_storage import write_contract
from tests.support.source_mortality import source as mortality_source


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


@pytest.mark.parametrize("missing", ["icustays", "chartevents"])
def test_miiv_refuses_a_missing_fio2_input(tmp_path, missing):
    start = pd.Timestamp("2023-06-12 15:26")
    if missing != "icustays":
        _write(tmp_path, "icustays", [(37, start)], ["stay_id", "intime"])
    if missing != "chartevents":
        _write(tmp_path, "chartevents", [(37, start, 223835, 35.0)],
               ["stay_id", "charttime", "itemid", "valuenum"])

    availability = {}
    assert _load(tmp_path, "miiv", stay=37, availability_sink=availability).empty
    record = availability["fio2_set"]
    assert record.reason == "source_unavailable"
    assert record.evaluation_status == "not_evaluable"
    assert record.reason_code == "required_source_table_missing"
    assert record.missing_tables == (missing,)


def test_miiv_present_chart_with_no_setting_for_requested_stay_is_empty(tmp_path):
    start = pd.Timestamp("2023-06-12 15:26")
    _write(tmp_path, "icustays", [(37, start), (38, start)], ["stay_id", "intime"])
    _write(tmp_path, "chartevents", [(38, start, 223835, 35.0)],
           ["stay_id", "charttime", "itemid", "valuenum"])
    availability = {}

    assert _load(tmp_path, "miiv", stay=37, availability_sink=availability).empty
    assert availability["fio2_set"].reason == "data_missing"


def test_bucketed_miiv_settings_accept_an_explicit_empty_stay_scope(tmp_path):
    start = pd.Timestamp("2022-03-01 13:23")
    _write(tmp_path, "icustays", [(23, start)], ["stay_id", "intime"])
    bucket = tmp_path / "chartevents_bucket"
    part = bucket / "bucket_id=0"
    part.mkdir(parents=True)
    _write(part, "chartevents", [(23, start, 223835, 35.0)],
           ["stay_id", "charttime", "itemid", "valuenum"])
    write_contract(bucket, key="itemid")
    availability = {}

    loader = BaseICULoader(database="miiv", data_path=tmp_path)
    table = loader.load_concepts(
        concepts=["fio2_set"], patient_ids={"stay_id": []},
        merge=False, r_compatible=False, availability_sink=availability,
    )["fio2_set"]

    result = table.data
    assert result.dtypes.astype(str).to_dict() == {
        "stay_id": "int64", "charttime": "float64", "fio2_set": "float64",
    }
    assert result.empty
    assert availability["fio2_set"].reason == "data_missing"


@pytest.mark.parametrize("invalid", [pd.DataFrame({"stay_id": [23]}), 23], ids=["frame", "scalar"])
def test_explicit_stay_scope_rejects_inputs_that_are_not_id_collections(tmp_path, invalid):
    with pytest.raises(ValueError, match="Expected one stay_id identifier list"):
        selected_stay_ids({"stay_id": invalid}, "stay_id")
    with pytest.raises(ValueError, match="Expected one stay_id identifier list"):
        load_concepts("fio2_set", database="miiv", data_path=tmp_path, patient_ids=invalid)


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


def test_mimic_demo_reports_carevue_view_required_even_at_a_view_named_path(tmp_path):
    root = tmp_path / "mimic_cv_dir_r13"
    root.mkdir()
    start = pd.Timestamp("2018-04-18 08:04")
    _write(root, "icustays", [(1, start, "carevue")], ["ICUSTAY_ID", "INTIME", "DBSOURCE"])
    _write(root, "chartevents", [(1, start, 190, 50.0)], ["ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUENUM"])
    availability = {}

    assert _load(root, "mimic_demo", availability_sink=availability).empty
    record = availability["fio2_set"]
    assert record.database == "mimic_demo"
    assert record.reason == "source_unavailable"
    assert record.evaluation_status == "not_evaluable"
    assert record.reason_code == "carevue_view_required"


def _carevue_chart(root):
    start = pd.Timestamp("2012-08-23 07:13")
    _write(root, "icustays", [(17, start, " CareVue\t")], ["ICUSTAY_ID", "INTIME", "DBSOURCE"])
    _write(root, "chartevents", [(17, start, 7570, 62.0, 0)],
           ["ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUENUM", "ERROR"])


@pytest.mark.parametrize("view", ["mimicview/raw", "mimic_cv_dir_r13"])
def test_fio2_accepts_carevue_views_with_padded_source_labels(tmp_path, view):
    root = tmp_path / view
    root.mkdir(parents=True)
    _carevue_chart(root)

    assert _hour_values(_load(root, "mimic", stay=17)) == {0.0: 62.0}


@pytest.mark.parametrize("linked", ["root", "icustays", "chartevents"])
def test_fio2_refuses_sources_linked_outside_carevue(tmp_path, linked):
    foreign = tmp_path / "full-installation"
    foreign.mkdir()
    _carevue_chart(foreign)
    root = tmp_path / "mimic_cv_dir_r13"
    if linked == "root":
        root.symlink_to(foreign, target_is_directory=True)
    else:
        root.mkdir()
        _carevue_chart(root)
        (root / f"{linked}.parquet").unlink()
        (root / f"{linked}.parquet").symlink_to(foreign / f"{linked}.parquet")
    # The origin label must not mask a successful read through a foreign link.
    stays = foreign / "icustays.parquet" if linked in {"root", "icustays"} else root / "icustays.parquet"
    pd.read_parquet(stays).assign(DBSOURCE="carevue").to_parquet(stays, index=False)
    availability = {}

    assert _load(root, "mimic", stay=17, availability_sink=availability).empty
    reason = "carevue_view_required" if linked == "root" else "required_source_table_missing"
    assert availability["fio2_set"].reason_code == reason


@pytest.mark.parametrize("mode", ["whole-module", "streamed", "subprocess-batches"])
@pytest.mark.parametrize(("root_name", "reason"), [
    ("full-installation", "carevue_view_required"),
    ("mimic_cv_dir_r13", "required_source_table_missing"),
])
def test_native_export_retains_fio2_source_refusal(tmp_path, monkeypatch, root_name, reason, mode):
    from easyicu.api.extraction import extract_database
    from easyicu.research_agent.intake.export_package import open_export_package

    source = mortality_source(tmp_path / root_name, "mimic")
    stays = source / "icustays.parquet"
    pd.read_parquet(stays).assign(dbsource="carevue", los=1.0).to_parquet(stays, index=False)
    patients = source / "patients.parquet"
    pd.read_parquet(patients).assign(dod=["2020-01-02", "2020-01-03", None, None]).to_parquet(
        patients, index=False,
    )
    output = tmp_path / "outcome-and-settings"
    batched = mode != "whole-module"
    if mode == "subprocess-batches":
        monkeypatch.setenv("EASYICU_FORCE_SUBPROCESS_BATCH", "1")
    extract_database(
        database="mimic", data_path=source, output_dir=output, modules=["outcome", "vent_settings"],
        patient_ids=[1, 2], batch_size=1 if batched else None, max_workers=1,
        resource_budget_mb=2048 if mode == "streamed" else None,
        stream_output_batches=mode == "streamed",
        native_export_v2=True, verbose=False,
    )

    manifest = json.loads((output / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["unavailable_modules"] == [
        {"module": "vent_settings", "reason": reason, "concept_ids": ["fio2_set"]},
    ]
    with open_export_package(output) as package:
        assert "fio2_set" not in package.concept_index


def test_fio2_set_has_its_own_opt_in_module():
    assert EXTRACT_MODULES["vent_settings"] == ["fio2_set"]
    assert "vent_settings" not in EXTRACT_MODULE_ORDER
    assert {module for module, concepts in EXTRACT_MODULES.items() if "fio2_set" in concepts} == {"vent_settings"}
