"""Synthetic coverage of the thesis support exports and their source clocks."""

import hashlib
import json
import math
import os
import tempfile

import pandas as pd
import pytest

from easyicu.api.extraction import EXTRACT_MODULE_ORDER, EXTRACT_MODULES, extract_database
from easyicu.concept import _apply_callback
from easyicu.concept.availability_signal import ConceptAvailabilityRecord
from easyicu.concept.schema import ConceptSource
from easyicu.concept.sicdb_sources import COVERAGE_ATTR
from easyicu.concept.support_exports import load_imv_first_episode, load_stay_identity
from easyicu.synthetic_sources import generate_source
from easyicu.table import TsTbl, WinTbl
from tests.support.aumc_source import aumc_db
from tests.support.blood_gas_sources import ADMISSION, _mimic_family, _write
from tests.support.eicu_source import _eicu_source
from tests.support.hirid_source import hirid_source_root


SUPPORT_MODULES = ("stay_identity", "imv_first_episode", "pao2_arterial_hourly_min")


@pytest.fixture(autouse=True)
def _restore_temp_settings(monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for name in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])


@pytest.fixture
def strict_empty_evidence_merge(monkeypatch):
    """Exercise dtype rejection even when pandas permits an empty-key merge."""
    merge = pd.DataFrame.merge

    def check_keys(left, right, *args, **kwargs):
        key = kwargs.get("on")
        if key in ("icustay_id", "CaseID") and (left.empty or right.empty):
            left_dtype, right_dtype = left[key].dtype, right[key].dtype
            if (pd.api.types.is_integer_dtype(left_dtype) and pd.api.types.is_object_dtype(right_dtype)
                    or pd.api.types.is_object_dtype(left_dtype) and pd.api.types.is_integer_dtype(right_dtype)):
                raise ValueError(f"Incompatible empty evidence join on {key}: {left_dtype} and {right_dtype}")
        return merge(left, right, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "merge", check_keys)


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


@pytest.mark.parametrize("database,stay_key", [("miiv", "stay_id"), ("mimic", "icustay_id")])
def test_undated_stay_hides_rank_but_leaves_other_hospital_admission_rankable(tmp_path, database, stay_key):
    stays = pd.DataFrame({
        stay_key: [18, 40, 73, 68, 91],
        "subject_id": [12, 12, 12, 12, 13],
        "hadm_id": [51, 51, 52, 52, 53],
        "intime": [pd.NaT, ADMISSION, ADMISSION + pd.Timedelta(days=3),
                   ADMISSION + pd.Timedelta(days=5), ADMISSION],
    })
    _write(stays, tmp_path / "icustays.parquet")
    actual = load_stay_identity(
        database=database, data_path=tmp_path, patient_ids={stay_key: [40, 73, 68, 91]},
    ).set_index(stay_key)
    assert actual.loc[[40, 73], ["stay_order", "is_first_icu_stay"]].isna().all(axis=None)
    assert pd.isna(actual.loc[40, "is_first_icu_stay_of_hospital_admission"])
    assert actual.loc[[73, 91], "is_first_icu_stay_of_hospital_admission"].tolist() == [True, True]
    assert pd.isna(actual.loc[68, "stay_order"])
    assert not actual.loc[68, "is_first_icu_stay_of_hospital_admission"]
    assert pd.notna(actual.loc[68, "is_first_icu_stay"])
    assert not actual.loc[68, "is_first_icu_stay"]
    assert actual.loc[91, ["stay_order", "is_first_icu_stay"]].tolist() == [1, True]


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


def test_aumc_only_earliest_dated_count_one_can_be_first_even_after_selection(tmp_path):
    _write(pd.DataFrame({
        "admissionid": [27, 83, 12], "patientid": 21,
        "admissioncount": [1, 1, 5], "admittedat": [240_000, 95_000, 840_000],
    }), tmp_path / "admissions.parquet")
    actual = load_stay_identity(database="aumc", data_path=tmp_path).set_index("admissionid")
    assert actual.loc[[83, 27, 12], "stay_order"].tolist() == [1, 2, 5]
    assert actual.loc[[83, 27, 12], "is_first_icu_stay"].tolist() == [True, False, False]
    selected = load_stay_identity(database="aumc", data_path=tmp_path, patient_ids={"admissionid": [27]})
    assert selected[["stay_order", "is_first_icu_stay"]].to_dict("list") == {
        "stay_order": [2], "is_first_icu_stay": [False],
    }


@pytest.mark.parametrize(
    "unrankable_count,first_timestamp",
    [(None, None), (0, None), (1, None), (1, 120_000), (1.5, None)],
    ids=["missing", "zero", "undated-tie", "same-recorded-admission-time", "fraction"],
)
def test_aumc_unresolved_admission_hides_ordinals_but_a_valid_later_count_still_excludes_first(
    tmp_path, unrankable_count, first_timestamp,
):
    _write(pd.DataFrame({
        "admissionid": [14, 25, 39], "patientid": 20,
        "admissioncount": [unrankable_count, 1, 4], "admittedat": [first_timestamp, 120_000, 720_000],
        "dischargedat": 3_600_000,
    }), tmp_path / "admissions.parquet")
    actual = load_stay_identity(database="aumc", data_path=tmp_path).set_index("admissionid")
    assert actual.stay_order.isna().all()
    assert actual.loc[[14, 25], "is_first_icu_stay"].isna().all()
    assert not actual.loc[39, "is_first_icu_stay"]


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


@pytest.mark.parametrize(
    "field,value",
    [("hospitaldischargeyear", None), ("hospitaldischargeyear", 2014), ("patienthealthsystemstayid", None)],
    ids=["year-unrecorded", "different-years-on-one-admission", "encounter-key-unrecorded"],
)
def test_eicu_uncertain_encounter_coordinates_hide_person_order(tmp_path, field, value):
    stays = pd.DataFrame({
        "patientunitstayid": [36, 48, 59], "uniquepid": "person-q",
        "patienthealthsystemstayid": [14, 14, 14], "hospitaldischargeyear": 2016,
        "unitvisitnumber": [1, 1, 3], "hospitaladmitoffset": [-180, -45, -960],
    })
    stays.loc[0, field] = value
    _write(stays, tmp_path / "patient.parquet")
    actual = load_stay_identity(database="eicu", data_path=tmp_path).set_index("patientunitstayid")
    assert actual.stay_order.isna().all()
    assert pd.isna(actual.loc[48, "is_first_icu_stay"])
    assert actual.loc[48, "is_first_icu_stay_of_hospital_admission"]
    assert not actual.loc[59, "is_first_icu_stay"]
    assert not actual.loc[59, "is_first_icu_stay_of_hospital_admission"]
    if field == "hospitaldischargeyear":
        assert not actual.loc[36, "is_first_icu_stay"]
        assert not actual.loc[36, "is_first_icu_stay_of_hospital_admission"]
    else:
        assert actual.loc[36, ["is_first_icu_stay", "is_first_icu_stay_of_hospital_admission"]].isna().all()


@pytest.mark.parametrize(
    "offsets,rankable",
    [
        ((-40, -85, -140, -320), True),
        ((-40, -85, None, -320), False),
        ((-40, -85, -140, -math.inf), False),
        ((-40, -85, -40, -320), False),
    ],
    ids=[
        "offsets-reorder-all-siblings", "one-duplicate-visit-undated",
        "later-sibling-offset-not-finite", "same-hospital-clock-reading",
    ],
)
def test_eicu_repeated_visit_numbers_require_a_complete_hospital_clock(tmp_path, offsets, rankable):
    _write(pd.DataFrame({
        "patientunitstayid": [82, 27, 65, 94], "uniquepid": "person-l",
        "patienthealthsystemstayid": 61, "hospitaldischargeyear": 2018,
        "unitvisitnumber": [1, 3, 1, 4], "hospitaladmitoffset": list(offsets),
    }), tmp_path / "patient.parquet")
    actual = load_stay_identity(database="eicu", data_path=tmp_path).set_index("patientunitstayid")
    flags = ["is_first_icu_stay", "is_first_icu_stay_of_hospital_admission"]
    if rankable:
        assert actual.loc[[82, 27, 65, 94], "stay_order"].tolist() == [1, 2, 3, 4]
        assert actual.loc[82, flags].tolist() == [True, True]
        assert actual.loc[65, flags].tolist() == [False, False]
    else:
        assert actual.stay_order.isna().all()
        assert actual.loc[[82, 65], flags].isna().all(axis=None)
    assert actual.loc[[27, 94], flags].to_numpy().tolist() == [[False, False], [False, False]]


@pytest.mark.parametrize("later_offset", [-780, None], ids=["rank-by-offset", "missing-offset"])
def test_eicu_incomplete_visit_sequence_uses_a_common_clock(tmp_path, later_offset):
    _write(pd.DataFrame({
        "patientunitstayid": [23, 35], "uniquepid": "person-f",
        "patienthealthsystemstayid": 82, "hospitaldischargeyear": 2017,
        "unitvisitnumber": [None, 2], "hospitaladmitoffset": [-95, later_offset],
    }), tmp_path / "patient.parquet")
    actual = load_stay_identity(database="eicu", data_path=tmp_path).set_index("patientunitstayid")
    flags = ["is_first_icu_stay", "is_first_icu_stay_of_hospital_admission"]
    if later_offset is None:
        assert actual.stay_order.isna().all()
        assert actual.loc[23, flags].isna().all()
    else:
        assert actual.loc[[23, 35], "stay_order"].tolist() == [1, 2]
        assert actual.loc[23, flags].tolist() == [True, True]
    assert actual.loc[35, flags].tolist() == [False, False]


def test_eicu_same_year_tie_leaves_later_hospital_visits_excluded(tmp_path):
    _write(pd.DataFrame({
        "patientunitstayid": [101, 114, 163], "uniquepid": "person-v",
        "patienthealthsystemstayid": [64, 65, 65], "hospitaldischargeyear": 2018,
        "unitvisitnumber": [3, 1, 2], "hospitaladmitoffset": [-880, -180, -570],
    }), tmp_path / "patient.parquet")
    actual = load_stay_identity(database="eicu", data_path=tmp_path).set_index("patientunitstayid")
    assert actual.loc[[101, 163], "is_first_icu_stay"].tolist() == [False, False]
    assert actual.loc[[101, 114, 163], "is_first_icu_stay_of_hospital_admission"].tolist() == [False, True, False]
    assert pd.isna(actual.loc[114, "is_first_icu_stay"])


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


@pytest.mark.parametrize("unrankable_offset", [None, -120])
def test_sic_missing_admission_clock_hides_ordinal_but_preserves_positive_offset_evidence(tmp_path, unrankable_offset):
    _write(pd.DataFrame({
        "CaseID": [28, 50, 62], "PatientID": 19,
        "OffsetAfterFirstAdmission": [unrankable_offset, 0, 7800], "AdmissionYear": 2014,
        "ICUOffset": 0, "TimeOfStay": 3600,
    }), tmp_path / "cases.parquet")
    actual = load_stay_identity(database="sic", data_path=tmp_path).set_index("CaseID")
    assert actual.stay_order.isna().all()
    assert actual.loc[[28, 50], "is_first_icu_stay"].isna().all()
    assert not actual.loc[62, "is_first_icu_stay"]


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


@pytest.mark.parametrize("stop_hour", [None, 4.75], ids=["no-extubation-documented", "rounded-opening-then-stop"])
def test_mimic_iii_episode_uses_owner_eight_hour_chart_gap(tmp_path, stop_hour, strict_empty_evidence_merge):
    _write(pd.DataFrame({"icustay_id": [1], "intime": [ADMISSION]}), tmp_path / "icustays.parquet")
    chart = pd.DataFrame({
        "icustay_id": 1, "charttime": [.25, 8.25, 976 / 60, 1036 / 60],
        "itemid": 60, "value": "5", "error": 0,
    })
    windows = _apply_callback(
        chart, ConceptSource(
            table="chartevents", sub_var="itemid", value_var="value", index_var="charttime",
            callback="mimiciii_chart_ventilation_intervals",
        ), concept_name="mech_vent",
    )

    def read_episode_inputs(*, concepts, patient_ids, availability_sink, **kwargs):
        assert concepts == ["mech_vent", "vent_end"]
        assert patient_ids == {"icustay_id": [1]}
        availability_sink["mech_vent"] = ConceptAvailabilityRecord(
            concept="mech_vent", database="mimic", reason="mapped_present",
        )
        markers = pd.DataFrame({
            "icustay_id": pd.Series(dtype="int64"),
            "charttime": pd.Series(dtype="float64"),
            "vent_end": pd.Series(dtype="bool"),
        })
        if stop_hour is not None:
            markers = pd.DataFrame({
                "icustay_id": 1, "charttime": [math.nextafter(.25, math.inf), stop_hour], "vent_end": True,
            })
        return {
            "mech_vent": WinTbl(windows, "icustay_id", "charttime", "dur_var"),
            "vent_end": TsTbl(markers, "icustay_id", "charttime"),
        }

    episode = load_imv_first_episode(
        read_episode_inputs, database="mimic", data_path=tmp_path, patient_ids={"icustay_id": [1]},
    ).iloc[0]
    assert episode.imv_start_hours == pytest.approx(.25)
    assert episode.imv_end_hours == pytest.approx(8.25 if stop_hour is None else stop_hour)
    assert episode.imv_duration_hours == pytest.approx(8. if stop_hour is None else stop_hour - .25)
    assert episode.episode_evidence == ("mech_vent" if stop_hour is None else "mech_vent|vent_end")


def test_aumc_first_invasive_window_needs_no_boundary_markers(tmp_path):
    source = aumc_db(tmp_path / "source", processes=[
        (12, 37, 10740, "Beademen non-invasief"),
        (49, 105, 10740, "Beademen"),
        (118, 190, 10740, "Beademen"),
    ])
    output = tmp_path / "out"
    extract_database(
        "aumc", data_path=source.base_path, output_dir=output, modules=["imv_first_episode"],
        patient_ids={"admissionid": [1]}, native_export_v2=True, max_workers=1,
    )
    episode = pd.read_parquet(output / "imv_first_episode.parquet").iloc[0]
    assert episode.episode_status == "observed_episode"
    assert episode.imv_start_hours == pytest.approx(49 / 60)
    assert episode.imv_end_hours == pytest.approx(105 / 60)
    assert episode.imv_duration_hours == pytest.approx(56 / 60)
    assert episode.episode_evidence == "mech_vent"


def test_hirid_episode_uses_first_invasive_span_between_airway_transitions(tmp_path):
    source = hirid_source_root(tmp_path, "observations", [
        dict(patientid=1, datetime=hour, variableid=15001552, value=state)
        for hour, state in ((.25, 3.), (1.5, 1.), (4.25, 6.), (9., 1.), (10., 6.))
    ])
    output = tmp_path / "out"
    extract_database(
        "hirid", data_path=source, output_dir=output, modules=["imv_first_episode"],
        patient_ids={"patientid": [1, 2]}, native_export_v2=True, max_workers=1,
    )
    rows = pd.read_parquet(output / "imv_first_episode.parquet").set_index("stay_id")
    episode = rows.loc[1]
    assert episode.episode_status == "observed_episode"
    assert episode.imv_start_hours == pytest.approx(1.5)
    assert episode.imv_end_hours == pytest.approx(4.25)
    assert episode.imv_duration_hours == pytest.approx(2.75)
    assert episode.episode_evidence == "mech_vent"
    assert rows.loc[2, "episode_status"] == "no_invasive_episode"


def test_sic_episode_keeps_covered_span_when_rule_table_is_empty(tmp_path, strict_empty_evidence_merge):
    _write(pd.DataFrame({"CaseID": [57]}), tmp_path / "cases.parquet")
    ventilation = pd.DataFrame({"CaseID": [57], "charttime": [2.], "mech_vent": ["invasive"]})
    ventilation.attrs[COVERAGE_ATTR] = {
        "CaseID": [57], "hour": [2.], "overlap_start_seconds": [7950], "overlap_stop_seconds": [9400],
    }

    def read_episode_inputs(*, concepts, availability_sink, **kwargs):
        assert concepts == ["mech_vent", "mech_vent_rule"]
        availability_sink["mech_vent"] = ConceptAvailabilityRecord(
            concept="mech_vent", database="sic", reason="mapped_present",
        )
        rules = ventilation.iloc[:0][["CaseID", "charttime"]].assign(mech_vent_rule=pd.Series(dtype="string"))
        return {
            "mech_vent": TsTbl(ventilation, "CaseID", "charttime"),
            "mech_vent_rule": TsTbl(rules, "CaseID", "charttime"),
        }

    episode = load_imv_first_episode(read_episode_inputs, database="sic", data_path=tmp_path).iloc[0]
    assert episode.episode_status == "observed_episode"
    assert episode.imv_start_hours == pytest.approx(7950 / 3600)
    assert episode.imv_end_hours == pytest.approx(9400 / 3600)
    assert episode.episode_evidence == "mech_vent"


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
