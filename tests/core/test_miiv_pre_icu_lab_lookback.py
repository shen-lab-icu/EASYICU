"""MIIV NULL-admission labs in the declared pre-ICU lookback band (invented records only).

A NULL-hadm lab more than 24 h before the subject's next ICU admission used to
be dropped at attribution, so kdigo_creatinine_input (168 h lookback) never saw
pre-ICU outpatient/ED creatinine. Such rows now go to that next stay; the
concept-layer episode clip keeps them only for a declared lookback.
"""

import json
from importlib import resources

import pandas as pd
import pytest

from tests.support.bucket_storage import write_fixture_parquet

from easyicu.concept import ConceptResolver
from easyicu.concept.loader import load_dictionary
from easyicu.config import load_src_cfg
from easyicu.datasource import (
    LAB_PRE_ICU_LOOKBACK_HOURS,
    LAB_WINDOW_HOURS,
    FilterOp,
    FilterSpec,
    ICUDataSource,
    _close_duckdb_connections,
    load_bucketed_table_aggregated,
    load_bucketed_table_multi_aggregated,
)


@pytest.fixture(autouse=True)
def bounded_duckdb(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_MEMORY_LIMIT_GB", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(tmp_path / "spill"))
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "state"))
    yield
    _close_duckdb_connections()


def _stays():
    # Subject 1: an earlier stay 100 and the target stay 101 eleven days later.
    return pd.DataFrame({
        "subject_id": [1, 1, 2], "hadm_id": [10, 11, 21], "stay_id": [100, 101, 201],
        "intime": pd.to_datetime(["2140-01-01", "2140-01-12", "2140-01-12"]),
        "outtime": pd.to_datetime(["2140-01-02", "2140-01-14", "2140-01-14"]),
    })


def _labs():
    rows = [
        (1, None, "2140-01-02 23:00", 50912, 1.0),   # +23 h after stay 100: the 24 h rule keeps it there.
        (1, None, "2140-01-10 12:00", 50912, 0.8),   # -36 h before stay 101: lookback band.
        (1, None, "2140-01-05 00:01", 50912, 0.7),   # -167.98 h: still inside 168 h.
        (1, None, "2140-01-04 23:59", 50912, 9.0),   # -168.02 h: beyond the lookback.
        (1, None, "2140-01-08 00:00", 50912, 5.0),   # Inside a non-ICU admission: never attributed.
        (2, None, "2140-01-10 12:00", 50912, 1.1),   # Another subject's stay only.
    ]
    frame = pd.DataFrame(rows, columns=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"])
    frame["charttime"] = pd.to_datetime(frame.charttime)
    frame["hadm_id"] = frame.hadm_id.astype("Int64")
    frame["valueuom"] = "mg/dL"
    return frame


def _admissions():
    return pd.DataFrame([
        (1, 10, "2140-01-01 00:00", "2140-01-03 00:00"),
        (1, 15, "2140-01-07 00:00", "2140-01-09 00:00"),  # No ICU stay.
        (1, 11, "2140-01-11 06:00", "2140-01-15 00:00"),
    ], columns=["subject_id", "hadm_id", "admittime", "dischtime"])


def _source(tmp_path):
    write_fixture_parquet(_stays(), tmp_path / "icustays.parquet", index=False)
    _admissions().to_parquet(tmp_path / "admissions.parquet", index=False)
    directory = tmp_path / "labevents_bucket" / "bucket_id=0"
    directory.mkdir(parents=True)
    write_fixture_parquet(_labs(), directory / "part.parquet", index=False)
    return ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path, enable_cache=False)


def test_lookback_constant_covers_every_declared_concept_lookback():
    concepts = json.loads(resources.files("easyicu.data").joinpath("concept-dict.json").read_text())
    declared = [v["pre_admission_lookback_hours"] for v in concepts.values()
                if isinstance(v, dict) and v.get("pre_admission_lookback_hours") is not None]
    assert declared and max(declared) == LAB_PRE_ICU_LOOKBACK_HOURS > LAB_WINDOW_HOURS


def test_loader_attributes_the_lookback_band_to_the_next_stay(tmp_path):
    frame = _source(tmp_path).load_table(
        "labevents", filters=[FilterSpec("subject_id", FilterOp.IN, [1])]).data
    got = sorted(frame[["stay_id", "valuenum", "lab_attribution_basis"]].itertuples(index=False, name=None))
    assert got == [
        (100, 1.0, "subject_id_charttime"),
        (101, 0.7, "subject_id_pre_icu_lookback"),
        (101, 0.8, "subject_id_pre_icu_lookback"),
    ]
    audit = frame.attrs["easyicu_miiv_lab_attribution"]
    assert audit["pre_icu_lookback_null_hadm_rows"] == 2
    assert audit["pre_icu_lookback_hours"] == 168
    assert audit["unmatched_null_hadm_rows"] == 2  # -168.02 h row and the non-ICU admission row.


@pytest.mark.parametrize("path", ["bucket", "multi"])
def test_aggregated_loaders_use_the_same_attribution(tmp_path, path):
    source = _source(tmp_path)
    if path == "bucket":
        frame = load_bucketed_table_aggregated(source, "labevents", "valuenum", [50912],
                                               patient_ids=[101], agg_func="sum")
    else:
        frame = load_bucketed_table_multi_aggregated(source, "labevents", {"valuenum": [50912]},
                                                     patient_ids=[101], agg_func="sum")
    assert sorted(frame.valuenum) == pytest.approx([0.7, 0.8])


def _concept(tmp_path, name):
    return ConceptResolver(load_dictionary("miiv")).load_concepts(
        [name], _source(tmp_path), patient_ids=[101], merge=False, r_compatible=False,
        concept_workers=1, interval=pd.Timedelta(hours=1),
    )[name].data


def test_generic_creatinine_keeps_its_24_hour_history(tmp_path):
    assert _concept(tmp_path, "crea").empty


def test_kdigo_input_receives_the_pre_icu_creatinine(tmp_path):
    frame = _concept(tmp_path, "kdigo_creatinine_input")
    value_col = next(c for c in frame.columns if c not in {"stay_id", "charttime"})
    assert sorted(frame[value_col].dropna()) == pytest.approx([0.7, 0.8])
    assert frame.charttime.min() >= -168
