"""Synthetic coverage of the thesis support exports and their source clocks."""

import hashlib
import json
import os
import tempfile

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULE_ORDER, EXTRACT_MODULES, extract_database
from easyicu.concept.support_exports import load_stay_identity
from easyicu.synthetic_sources import generate_source
from tests.support.blood_gas_sources import ADMISSION, _mimic_family, _write
from tests.support.eicu_source import _eicu_source


SUPPORT_MODULES = ("stay_identity", "imv_first_episode", "pao2_arterial_hourly_min")


@pytest.fixture(autouse=True)
def _restore_temp_settings(monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for name in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])


@pytest.mark.parametrize("database,stay_key", [("miiv", "stay_id"), ("mimic", "icustay_id")])
def test_stay_identity_ranks_source_history_before_member_filter(tmp_path, database, stay_key):
    stays = pd.DataFrame({
        stay_key: [5, 4, 2, 3, 1],
        "subject_id": [11, 11, 11, 11, 12],
        "hadm_id": [22, 21, 22, 22, 23],
        "intime": ADMISSION + pd.to_timedelta([8, 0, 0, 0, 12], unit="h"),
        "outtime": ADMISSION + pd.Timedelta(hours=24),
    })
    if database == "mimic":
        stays = stays.rename(columns=str.upper)
    _write(stays, tmp_path / "icustays.parquet")

    only = load_stay_identity(database=database, data_path=tmp_path, patient_ids={stay_key: [1, 3, 5]})
    actual = only.set_index(stay_key).sort_index()
    assert actual.person_id.tolist() == ["12", "11", "11"]
    assert actual.hospital_admission_id.tolist() == ["23", "22", "22"]
    assert actual.stay_order.tolist() == ([1, 3, 4] if database == "miiv" else [1, 2, 4])
    assert actual.is_first_icu_stay.tolist() == [True, False, False]
    assert actual.is_first_icu_stay_of_hospital_admission.tolist() == [True, False, False]
    assert actual.order_evidence.str.contains("icustays.intime", regex=False).all()


def test_aumc_uses_absolute_admission_count_and_breaks_ties(tmp_path):
    _write(pd.DataFrame({
        "admissionid": [9, 8, 7, 6], "patientid": [1, 1, 1, 2],
        "admissioncount": [6, 4, 4, 1], "admittedat": [0, 60_000, 0, 0],
        "dischargedat": 3_600_000,
    }), tmp_path / "admissions.parquet")
    actual = load_stay_identity(database="aumc", data_path=tmp_path).set_index("admissionid")
    assert actual.loc[[7, 8, 9], "stay_order"].tolist() == [4, 5, 6]
    assert actual.is_first_icu_stay.sum() == 1
    assert actual.hospital_admission_id.isna().all()
    assert actual.is_first_icu_stay_of_hospital_admission.isna().all()


def test_eicu_resets_visit_number_at_each_hospital_admission(tmp_path):
    _write(pd.DataFrame({
        "patientunitstayid": [7, 6, 5, 4, 3, 2, 1],
        "uniquepid": ["A", "A", "A", "A", "A", "B", "B"],
        "patienthealthsystemstayid": [16, 15, 15, 14, 14, 13, 13],
        "hospitaldischargeyear": [2020, 2019, 2019, 2019, 2019, 2020, 2020],
        "unitvisitnumber": [1, 2, 1, 1, 1, 2, 2],
        "hospitaladmitoffset": [-100, -80, -20, -40, -10, -150, -120],
        "unitdischargeoffset": 1440,
    }), tmp_path / "patient.parquet")
    actual = load_stay_identity(database="eicu", data_path=tmp_path).set_index("patientunitstayid")
    assert actual.loc[[3, 4, 5, 6, 7], "stay_order"].tolist() == [1, 2, 3, 4, 5]
    assert actual.loc[[3, 5], "is_first_icu_stay"].isna().all()
    assert actual.loc[[1, 2, 3, 4, 5, 6, 7], "is_first_icu_stay_of_hospital_admission"].tolist() == [
        False, False, True, False, True, False, True,
    ]


def test_sic_orders_by_relative_admission_and_year(tmp_path):
    _write(pd.DataFrame({
        "CaseID": [3, 2, 4, 1], "PatientID": [11, 11, 11, 11],
        "OffsetAfterFirstAdmission": [0, 0, 9999, 0], "AdmissionYear": [2021, 2020, 2020, 2020],
        "ICUOffset": 0, "TimeOfStay": 3600,
    }), tmp_path / "cases.parquet")
    actual = load_stay_identity(database="sic", data_path=tmp_path).set_index("CaseID")
    assert actual.loc[[1, 2, 3, 4], "stay_order"].tolist() == [1, 2, 3, 4]
    assert actual.loc[[1, 2, 3], "is_first_icu_stay"].isna().all()
    assert not actual.loc[4, "is_first_icu_stay"]
    assert actual.hospital_admission_id.isna().all()


def test_hirid_uses_stay_as_person_without_longitudinal_linkage(tmp_path):
    source = tmp_path / "source"
    generate_source(source, "hirid", stays=3)
    selection = {"patientid": [2]}
    actual = load_stay_identity(database="hirid", data_path=source, patient_ids=selection)
    assert actual.person_id.tolist() == ["2"]
    assert actual.stay_order.tolist() == [1]
    assert actual.is_first_icu_stay.tolist() == [True]
    assert actual.person_linkage_available.tolist() == [False]
    assert actual.is_first_icu_stay_of_hospital_admission.isna().all()
    assert load_stay_identity(
        database="hirid", data_path=source, patient_ids={"patientid": []}
    ).empty


def test_miiv_episode_gap_niv_and_explicit_boundaries(tmp_path):
    _mimic_family(tmp_path, "miiv", [(10, 50821, "80", 1)], [(10, 220274, 7.4)], "specimen_id")
    procedure = pd.DataFrame({
        "itemid": [225794, 225792, 225792, 225792, 225468],
        "starttime": ADMISSION + pd.to_timedelta([0, 75, 90, 121, 112], unit="min"),
        "endtime": ADMISSION + pd.to_timedelta([45, 90, 120, 240, 113], unit="min"),
        "stay_id": 1, "subject_id": 101, "hadm_id": 201, "statusdescription": "Finished",
        "cancelreason": 0, "value": [45., 15., 30., 119., 1.], "valueuom": "min",
    })
    _write(procedure, tmp_path / "procedureevents.parquet")
    root = tmp_path / "out"
    extract_database(
        "miiv", data_path=tmp_path, output_dir=root, modules=["imv_first_episode"],
        patient_ids={"stay_id": [1]}, native_export_v2=True, max_workers=1,
    )
    episode = pd.read_parquet(root / "imv_first_episode.parquet").iloc[0]
    assert episode.imv_start_hours == pytest.approx(75 / 60)
    assert episode.imv_end_hours == pytest.approx(112 / 60)
    assert episode.imv_duration_hours == pytest.approx(37 / 60)
    assert episode.episode_evidence == "mech_vent|vent_start|vent_end"


def test_mimic_iii_episode_uses_owner_eight_hour_chart_gap(tmp_path):
    _mimic_family(
        tmp_path, "mimic", [(10, 50821, "80", None)],
        [(15, 60, 5.), (495, 60, 5.), (976, 60, 5.), (1036, 60, 5.)], "unused",
    )
    _write(pd.DataFrame({
        "itemid": [225794], "starttime": [ADMISSION],
        "endtime": [ADMISSION + pd.Timedelta(minutes=10)],
        "icustay_id": 1, "subject_id": 101, "hadm_id": 201, "value": 10.,
        "valueuom": "min", "statusdescription": "Finished", "cancelreason": 0,
    }), tmp_path / "procedureevents_mv.parquet")
    output = tmp_path / "out"
    extract_database(
        "mimic", data_path=tmp_path, output_dir=output, modules=["imv_first_episode"],
        patient_ids={"icustay_id": [1]}, native_export_v2=True, max_workers=1,
    )
    episode = pd.read_parquet(output / "imv_first_episode.parquet").iloc[0]
    assert episode.imv_start_hours == pytest.approx(.25)
    assert episode.imv_end_hours == pytest.approx(8.25)
    assert episode.imv_duration_hours == pytest.approx(8.)


def test_eicu_episode_refuses_to_infer_duration_from_mixed_points(tmp_path):
    source = _eicu_source(
        tmp_path / "source", "eicu",
        respiratorycharting=pd.DataFrame({
            "respchartvalue": ["Ventilator", "BiPAP"], "respchartvaluelabel": "O2 Device",
            "respchartoffset": [0, 60],
        }),
    )
    output = tmp_path / "out"
    extract_database(
        "eicu", data_path=source.base_path, output_dir=output, modules=["imv_first_episode"],
        patient_ids={"patientunitstayid": [1]}, native_export_v2=True, max_workers=1,
    )
    episode = pd.read_parquet(output / "imv_first_episode.parquet").iloc[0]
    assert episode.episode_status == "not_invasive_specific"
    assert episode[["imv_start_hours", "imv_end_hours", "imv_duration_hours"]].isna().all()


def test_sic_episode_uses_exact_partial_hours_and_rule(tmp_path):
    generate_source(tmp_path / "source", "sic", stays=3)
    source = tmp_path / "source"
    cases = pd.read_parquet(source / "cases.parquet").set_index("CaseID")
    _write(pd.DataFrame({
        "CaseID": [1, 1, 2], "DataID": 720, "Offset": [1200, 3600, 0],
        "OffsetEnd": [3300, 4200, 3600],
    }).assign(
        Offset=lambda df: df.Offset + df.CaseID.map(cases.ICUOffset),
        OffsetEnd=lambda df: df.OffsetEnd + df.CaseID.map(cases.ICUOffset),
    ), source / "data_range.parquet")
    # Remove setting rows from this synthetic fixture to isolate exact ETT intervals.
    _write(pd.read_parquet(source / "data_float_h.parquet").iloc[:0], source / "data_float_h.parquet")
    output = tmp_path / "out"
    extract_database(
        "sic", data_path=source, output_dir=output, modules=["imv_first_episode"],
        patient_ids={"CaseID": [1, 3]}, native_export_v2=True, max_workers=1,
        stream_output_batches=True, batch_size=1,
    )
    rows = pd.read_parquet(output / "imv_first_episode.parquet").set_index("stay_id")
    assert rows.index.tolist() == [1, 3]
    assert rows.loc[1, "imv_start_hours"] == pytest.approx(1200 / 3600)
    assert rows.loc[1, "imv_end_hours"] == pytest.approx(3300 / 3600)
    assert rows.loc[1, "episode_evidence"] == "mech_vent|explicit_ett"
    assert rows.loc[3, "episode_status"] == "no_invasive_episode"


def test_hourly_min_reuses_specimens_and_keeps_default_bytes(tmp_path, monkeypatch):
    lab = [
        (10, 50821, "90", 1), (10, 52033, "ART.", 1),
        (15, 50821, "70", 2), (15, 52033, "ART.", 2),
        (20, 50821, "35", 3), (20, 52033, "VEN.", 3),
        (70, 50821, "45", 4),
        (150, 50821, "105", 5), (150, 52033, "ART.", 5),
    ]
    source, _ = _mimic_family(tmp_path / "source", "miiv", lab, [(10, 220274, 7.4)], "specimen_id")
    stay_path = source.base_path / "icustays.parquet"
    stays = pd.read_parquet(stay_path)
    _write(pd.concat([stays, stays.assign(stay_id=2, subject_id=102, hadm_id=202)]), stay_path)
    lab_path = source.base_path / "labevents" / "part.parquet"
    labs = pd.read_parquet(lab_path)
    _write(pd.concat([labs, labs.assign(subject_id=102, hadm_id=202, specimen_id=labs.specimen_id + 10)]), lab_path)
    assert set(SUPPORT_MODULES) <= EXTRACT_MODULES.keys()
    assert not set(SUPPORT_MODULES) & set(EXTRACT_MODULE_ORDER)
    monkeypatch.setitem(EXTRACT_MODULES, "blood_gas", ["po2"])
    monkeypatch.setitem(EXTRACT_MODULES, "outcome", ["los_icu"])
    before, after = (tmp_path / name for name in ("before", "after"))
    default = ["outcome", "blood_gas"]
    for root, modules in ((before, default), (after, [*default, *SUPPORT_MODULES])):
        result = extract_database(
            "miiv", data_path=source.base_path, output_dir=root, modules=modules,
            patient_ids={"stay_id": [1]}, native_export_v2=True, max_workers=1,
            stream_output_batches=True, batch_size=1,
        )
        assert all(not value.get("errors") for value in result["modules"].values())
    assert not any((before / (name + ".parquet")).exists() for name in SUPPORT_MODULES)
    for module in default:
        filename = module + ".parquet"
        assert hashlib.sha256((before / filename).read_bytes()).digest() == hashlib.sha256(
            (after / filename).read_bytes()
        ).digest()
    default_gases = pd.read_parquet(before / "blood_gas.parquet").dropna(subset=["po2"])
    assert default_gases.po2.tolist() == [80., 105.]
    hourly = pd.read_parquet(after / "pao2_arterial_hourly_min.parquet")
    assert hourly.to_dict("list") == {
        "stay_id": [1, 1], "charttime": [0., 2.], "po2": [70., 105.],
        "po2_specimen_status": ["explicit_arterial"] * 2,
    }
    manifest = json.loads((after / "_manifest.json").read_text())
    files = {file["module"]: file for file in manifest["files"]}
    assert files["stay_identity"]["primary_key"] == ["stay_id"]
    assert files["imv_first_episode"]["primary_key"] == ["stay_id"]
    assert files["pao2_arterial_hourly_min"]["primary_key"] == ["stay_id", "charttime"]
    sidecar = json.loads((after / manifest["column_metadata"]["file"]).read_text())
    metadata = {file["module"]: file["columns"] for file in sidecar["files"]}
    assert metadata["pao2_arterial_hourly_min"]["po2"]["metadata"]["aggregation"] == "min"
    assert metadata["imv_first_episode"]["imv_start_hours"]["metadata"]["time_unit"] == "h"
    assert pd.read_parquet(after / "stay_identity.parquet").stay_id.tolist() == [1]
    assert pd.read_parquet(after / "imv_first_episode.parquet").episode_status.tolist() == [
        "source_unavailable",
    ]
