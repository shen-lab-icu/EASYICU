"""Invented MIMIC-III CSVs exercise full and demo identifier/time semantics."""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from easyicu.base import BaseICULoader
from easyicu.concept.callback_apply import _patient_id_column
from easyicu.concept.callbacks import ConceptCallbackContext, _callback_uo_window
from easyicu.concept.source_duration import normalize_source_dur_var_hours
from easyicu.config import load_src_cfg
from easyicu.datasource import FilterOp, FilterSpec, _close_duckdb_connections
from easyicu.io.data_converter import DataConverter
from easyicu.table import ICUTable
from easyicu.table.duration import get_dur_var_unit, set_dur_var_unit


@pytest.fixture(params=["mimic", "mimic_demo"])
def loader(tmp_path, request, monkeypatch):
    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "1")
    cfg = load_src_cfg(request.param)
    common = {"subject_id": 1, "hadm_id": 11, "icustay_id": 101}
    rows = {
        "patients": [{"subject_id": 1, "gender": "F", "dob": "2060-01-01 00:00:00"}],
        "admissions": [{"subject_id": 1, "hadm_id": 11, "admittime": "2100-01-01 00:00:00",
                        "dischtime": "2100-01-05 00:00:00", "hospital_expire_flag": 0}],
        "icustays": [{**common, "intime": "2100-01-01 01:00:00", "outtime": "2100-01-03 01:00:00", "los": 2}],
        "chartevents": [
            {**common, "charttime": "2100-01-01 02:00:00", "itemid": item,
             "value": value, "valuenum": numeric, "valueuom": unit, "error": 0}
            for item, value, numeric, unit in (
                (220045, "80", 80, "bpm"), (223761, "98.6", 98.6, "F"),
                (223900, "No Response-ETT", 1, ""), (224685, "450", 450, "mL"),
                (223848, "Assist Control", None, ""), (223849, "SIMV", None, ""),
                (720, "Assist Control", None, ""),
            )
        ],
        "labevents": [{"subject_id": 1, "hadm_id": 11, "charttime": "2100-01-01 02:00:00",
                       "itemid": 50912, "value": "1.2", "valuenum": 1.2, "valueuom": "mg/dL"}],
        "outputevents": [{**common, "charttime": "2100-01-01 02:00:00", "itemid": 226559,
                          "value": 120, "valueuom": "mL"}],
        "procedureevents_mv": [{**common, "itemid": 225792, "starttime": "2100-01-01 02:00:00",
                                "endtime": "2100-01-01 03:00:00", "cancelreason": 0,
                                "statusdescription": "FinishedRunning", "value": 60, "valueuom": "min"}],
    }
    raw = tmp_path / "mimic-iii-clinical-database-demo-1.4"
    raw.mkdir()
    for name, table in cfg.tables.items():
        columns = list(table.columns)
        with (raw / f"{name.upper()}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writerow({column: column.upper() for column in columns})
            writer.writerows(rows.get(name, []))
    conversion = DataConverter(tmp_path, database=request.param, parallel_workers=1, verbose=False).convert_all()
    assert all(row["status"] == "completed" for row in conversion.values())
    instance = BaseICULoader(data_path=tmp_path, database=request.param)
    yield instance
    instance.clear_cache()
    _close_duckdb_connections()


@pytest.mark.parametrize("concept,expected", [("hr", 80), ("temp", 37), ("crea", 1.2), ("urine", 120)])
def test_uppercase_plain_csv_measurements_keep_stay_time_and_values(loader, concept, expected):
    frame = loader.load_concepts(concept, concept_workers=1, parallel_workers=1, r_compatible=False)
    assert frame.icustay_id.tolist() == [101]
    assert frame.charttime.tolist() == [1.0]
    assert pd.api.types.is_numeric_dtype(frame[concept])
    assert frame[concept].tolist() == pytest.approx([expected])


@pytest.mark.parametrize("concept", ["ett_gcs", "mech_vent", "vent_mode", "vent_breath_seq"])
def test_mimic_callbacks_keep_the_icu_identifier(loader, concept):
    frame = loader.load_concepts(concept, concept_workers=1, parallel_workers=1, r_compatible=False)
    assert "icustay_id" in frame
    assert frame[concept].notna().any()
    assert set(frame.icustay_id) == {101}


@pytest.mark.parametrize("selector", [[101], {"icustay_id": [101]}])
def test_stay_selection_reaches_hospital_measurements(loader, selector):
    frame = loader.load_concepts("crea", patient_ids=selector, concept_workers=1, parallel_workers=1)
    assert frame.icustay_id.tolist() == [101]
    assert frame.crea.tolist() == pytest.approx([1.2])


@pytest.mark.parametrize("empty_urine", [False, True])
def test_missing_urine_evidence_keeps_the_source_stay_identifier(loader, empty_urine):
    if empty_urine:
        empty = loader.datasource.load_table("outputevents").data.iloc[:0]
        loader.datasource.clear_cache()
        loader.datasource.register_table_source("outputevents", lambda: empty.copy())
    frame = loader.load_concepts("uo_6h", concept_workers=1, parallel_workers=1)
    assert "icustay_id" in frame
    assert "stay_id" not in frame


@pytest.mark.parametrize("concept", ["age", "sex"])
def test_demographics_are_bound_to_the_icu_stay(loader, concept):
    frame = loader.load_concepts(concept, patient_ids={"icustay_id": [101]}, concept_workers=1)
    assert frame.icustay_id.tolist() == [101]
    assert frame[concept].notna().all()


def test_callback_identifier_keeps_repeat_icu_stays_distinct(loader):
    frame = pd.DataFrame({"subject_id": [1, 1], "hadm_id": [11, 11], "icustay_id": [101, 102]})
    assert frame.groupby(_patient_id_column(frame, loader.datasource)).size().tolist() == [1, 1]


@pytest.mark.parametrize("id_column,identifier", [("icustay_id", 101), ("hadm_id", 11), ("subject_id", 1)])
def test_absolute_chart_time_is_anchored_to_the_icu_admission(loader, id_column, identifier):
    frame = pd.DataFrame({id_column: [identifier], "charttime": [pd.Timestamp("2100-01-01 02:00:00")]})
    aligned = loader.concept_resolver._align_time_to_admission(
        frame, loader.datasource, [id_column], "charttime",
    )
    assert aligned.charttime.tolist() == [1.0]
    assert aligned.icustay_id.tolist() == [101]


def test_empty_urine_callback_preserves_the_configured_icu_key(loader):
    tables = {
        concept: ICUTable(pd.DataFrame(columns=["icustay_id", "charttime", concept]),
                          id_columns=["icustay_id"], index_column="charttime", value_column=concept)
        for concept in ("urine", "weight")
    }
    context = ConceptCallbackContext(
        concept_name="uo_6h", target=None, interval=None, resolver=None,
        data_source=loader.datasource, patient_ids=None, kwargs={},
    )
    result = _callback_uo_window(tables, context, window_hours=6, output_col="uo_6h")
    assert result.id_columns == ["icustay_id"]
    assert result.data.columns.tolist() == ["icustay_id", "charttime", "uo_6h"]


def test_hospital_filter_does_not_expand_to_a_second_icu_stay(loader):
    stays = loader.datasource.load_table("icustays").data
    repeated = pd.concat([stays, stays.assign(icustay_id=102)], ignore_index=True)
    loader.datasource.clear_cache()
    loader.datasource.register_table_source("icustays", lambda: repeated.copy())
    table = loader.datasource.load_table(
        "labevents", filters=[FilterSpec(column="icustay_id", op=FilterOp.IN, value=[101])],
    )
    assert table.data.icustay_id.tolist() == [101]


@pytest.mark.parametrize("declared_unit", [None, "minutes"])
def test_empty_duration_source_can_bind_with_hour_normalized_evidence(declared_unit):
    frame = pd.DataFrame(columns=["dur_var"])
    if declared_unit:
        set_dur_var_unit(frame, declared_unit)
    normalized = normalize_source_dur_var_hours(frame, concept_name="synthetic_duration")
    assert pd.api.types.is_numeric_dtype(normalized.dur_var)
    assert get_dur_var_unit(normalized) == "hours"
