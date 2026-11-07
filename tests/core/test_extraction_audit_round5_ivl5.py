"""Round 5 MIMIC loader regressions, using invented records only."""

from __future__ import annotations

from tests.support.bucket_storage import move_fixture_parquet

from tests.support.bucket_storage import write_fixture_parquet

from tests.support.bucket_storage import synthetic_data_source

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.loader import load_dictionary
from easyicu.concept.schema import ConceptSource
from easyicu.config import DatasetOptions, load_src_cfg
from easyicu.datasource import (
    FilterOp,
    FilterSpec,
    ICUDataSource,
    _close_duckdb_connections,
    load_bucketed_table_aggregated,
    load_bucketed_table_multi_aggregated,
)
from easyicu.table import ICUTable
from easyicu.utils.callback_utils import distribute_amount, distribute_volume_hourly, mimic_rate_mv


@pytest.fixture(autouse=True)
def bounded_duckdb(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_MEMORY_LIMIT_GB", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(tmp_path / "spill"))
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "state"))
    yield
    _close_duckdb_connections()


def _stays(db="miiv"):
    return pd.DataFrame({
        "subject_id": [1, 2, 2, 3, 3], "hadm_id": [11, 21, 22, 31, 31],
        "stay_id" if db == "miiv" else "icustay_id": [101, 201, 202, 301, 302],
        "intime": pd.to_datetime(["2140-01-02", "2140-01-02", "2140-01-03",
                                  "2140-01-02", "2140-01-03"]),
        "outtime": pd.to_datetime(["2140-01-04", "2140-01-04", "2140-01-05",
                                   "2140-01-04", "2140-01-05"]),
    })


def _source(tmp_path, db, table, rows):
    write_fixture_parquet(_stays(db), tmp_path / "icustays.parquet", index=False)
    directory = tmp_path / f"{table}_bucket" / "bucket_id=0"
    directory.mkdir(parents=True)
    write_fixture_parquet(rows, directory / "part.parquet", index=False)
    return ICUDataSource(load_src_cfg(db), base_path=tmp_path, enable_cache=False)


def _labs():
    # Unique NULL lab; a NULL duplicate of a hadm-linked event; ambiguous
    # admissions; ambiguous ICU stays in one admission; outside the window.
    rows = [(1, None, "2140-01-01 12:00", 50912, 1.2),
            (1, None, "2140-01-02 02:00", 50912, 2.0),
            (1, 11, "2140-01-02 02:00", 50912, 2.0),
            (1, None, "2140-01-02 02:00", 50912, 4.0),
            (2, None, "2140-01-03 02:00", 50912, 8.0),
            (3, None, "2140-01-03 02:00", 50912, 9.0),
            (1, None, "2139-12-30 12:00", 50912, 7.0),
            (1, None, "2140-01-02 02:00", 51222, 10.0)]
    frame = pd.DataFrame(rows, columns=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"])
    frame["charttime"] = pd.to_datetime(frame.charttime)
    frame["valueuom"] = ""
    return frame


@pytest.mark.parametrize("path", ["source", "bucket", "batch", "resolver", "resolver_hadm"])
def test_r5c_iv_008_null_hadm_labs_have_bounded_rolling_attribution(tmp_path, path, monkeypatch):
    source = _source(tmp_path, "miiv", "labevents", _labs())
    if path == "source":
        frame = source.load_table("labevents", filters=[
            FilterSpec("hadm_id", FilterOp.IN, [11, 21, 31]),
        ]).data
        assert sorted(frame.loc[frame.itemid.eq(50912), "valuenum"]) == [1.2, 2.0, 4.0, 8.0, 9.0]
        assert set(frame.lab_attribution_basis) == {"hadm_id", "subject_id_charttime"}
        audit = frame.attrs["easyicu_miiv_lab_attribution"]
        assert audit["ambiguous_null_hadm_rows"] == 2
        assert audit["deduplicated_null_hadm_rows"] == 1
        assert audit["unmatched_null_hadm_rows"] == 1
        return
    if path == "bucket":
        frame = load_bucketed_table_aggregated(source, "labevents", "valuenum", [50912],
                                               patient_ids=[101, 201, 301], agg_func="sum")
        values = frame.set_index("charttime").valuenum
    elif path == "batch":
        frame = load_bucketed_table_multi_aggregated(source, "labevents", {"crea": [50912], "hgb": [51222]},
                                                     patient_ids=[101, 201, 301], agg_func="sum")
        values = frame.set_index("charttime").crea
        assert frame.loc[frame.charttime.eq(2), "hgb"].item() == 10
    else:
        original = source.load_table

        def raw_labs(table, **kwargs):
            if table == "labevents":
                raw = _labs()
                for spec in kwargs.get("filters") or []:
                    if spec.column in raw:
                        raw = spec.apply(raw)
                return ICUTable(raw, id_columns=["subject_id"], index_column="charttime",
                                value_column="valuenum", unit_column="valueuom")
            return original(table, **kwargs)

        monkeypatch.setattr(source, "load_table", raw_labs)
        monkeypatch.setattr(source, "resolve_bucket_directory", lambda table: None)
        # Exercise the resolver's raw subject/hadm join independently of the source mapper.
        resolver = ConceptResolver(load_dictionary("miiv"))
        patients = ({"stay_id": [101], "subject_id": [1], "hadm_id": [11]}
                    if path == "resolver_hadm" else [101, 201, 301])
        frame = resolver.load_concepts(["crea"], source, patient_ids=patients,
                                       interval=pd.Timedelta(hours=1), merge=False,
                                       r_compatible=False, concept_workers=1)["crea"].data
        values = frame.set_index("charttime").crea
    assert set(frame.stay_id) == ({101} if path == "resolver_hadm" else {101, 201, 301})
    assert values.loc[-12] == pytest.approx(1.2)
    assert values.loc[2] == pytest.approx(3.0 if path.startswith("resolver") else 6.0)


@pytest.mark.parametrize("path", ["source", "bucket", "batch"])
@pytest.mark.parametrize("uppercase", [False, True])
def test_r5c_iii_008_chart_errors_are_removed_before_every_aggregation(tmp_path, path, uppercase):
    frame = pd.DataFrame({"icustay_id": [101] * 4, "charttime": pd.to_datetime(["2140-01-02 01:00"] * 4),
                          "itemid": [211] * 4, "valuenum": [60., 80., 999., 888.],
                          "value": ["60", "80", "999", "888"], "valueuom": "bpm",
                          "error": [None, 0, 1, "1"]})
    # Mixed CSV-style ERROR tokens have the same interpretation as numeric ones.
    frame["error"] = frame.error.astype("string")
    if uppercase:
        frame.columns = frame.columns.str.upper()
    source = _source(tmp_path, "mimic", "chartevents", frame)
    if path == "source":
        result = source.load_table("chartevents", columns=["valuenum"]).data
        assert result.valuenum.tolist() == [60., 80.]
    elif path == "bucket":
        result = load_bucketed_table_aggregated(source, "chartevents", "valuenum", [211])
        assert result.valuenum.tolist() == [70.]
    else:
        result = load_bucketed_table_multi_aggregated(source, "chartevents", {"hr": [211], "other": [212]})
        assert result.hr.tolist() == [70.]


@pytest.mark.parametrize("name,unit,rate,weight,want", [
    ("fentanyl_rate", "mcg/kg/hour", 2., 50., 100.),
    ("norepi_rate", "mg/hour", .6, 50., .2),
    ("propofol_rate", "mg/hour", 60., 50., 20.),
    ("phn_rate", "mcg/min", 10., 50., .2),
    ("norepi_rate", "mg/kg/min", .001, 50., 1.),
    ("norepi_rate", "mg/kg/min", .001, 9999., None),
    ("fentanyl_rate", "ml/hour", 2., 50., None),
    ("norepi_rate", "mcg/min", 10., 0., None),
    ("norepi_rate", "mcg/min", 10., 9999., None),
    ("norepi_rate", "mcg/min", 10., float("inf"), None),
    ("norepi_rate", "mcg/kg/min", .2, None, .2),
    ("norepi_rate", "mcg/kg/min", -.2, 50., None),
    ("norepi_rate", "mcg/kg/min", float("inf"), 50., None),
])
def test_r5c_iv_010_rates_are_converted_or_rejected_before_bounds(name, unit, rate, weight, want):
    frame = pd.DataFrame({"stay_id": [101], "starttime": [0.25], "endtime": [0.75],
                          "rate": [rate], "rateuom": [unit], "patientweight": [weight]})
    src = ConceptSource.from_mapping({"table": "inputevents", "callback": "mimic_rate_mv", "stop_var": "endtime"})
    out = _apply_callback(frame, src, name)
    if want is None:
        assert out.empty or out[name].isna().all()
    else:
        assert out[name].tolist() == pytest.approx([want])


@pytest.mark.parametrize("datetime_axis", [False, True])
def test_r5c_iv_011_rate_hours_require_positive_interval_overlap(datetime_axis):
    frame = pd.DataFrame({"stay_id": [101] * 4, "starttime": [0., 1.5, 2.2, 4.25],
                          "endtime": [1., 2.2, 2.8, 4.25], "rate": [.1, .2, .3, .4]})
    origins = None
    if datetime_axis:
        origin = pd.Timestamp("2140-01-01 00:17")
        origins = pd.DataFrame({"stay_id": [101], "intime": [origin]})
        for col in ("starttime", "endtime"):
            frame[col] = origin + pd.to_timedelta(frame[col], unit="h")
    out = mimic_rate_mv(frame, val_col="rate", stop_var="endtime", admission_times=origins)
    hours = out.starttime
    if datetime_axis:
        hours = (hours - origins.intime.item()).dt.total_seconds() / 3600
    assert hours.tolist() == [0., 1., 2., 2.]
    # The two changes in hour 2 remain distinct components for the normal rate aggregator.
    assert out.rate.tolist() == [.1, .2, .2, .3]


@pytest.mark.parametrize("datetime_axis", [False, True])
def test_r5c_iv_012_insulin_dose_components_sum_with_true_overlap(datetime_axis):
    # IV infusion 12 U across [0.5, 2.5); IV push 3 U; SC dose 5 U.
    frame = pd.DataFrame({"stay_id": [101] * 3, "starttime": [.5, 1.2, 1.7],
                          "endtime": [2.5, 1.2, 1.7], "ins": [12., 3., 5.]})
    origins = None
    if datetime_axis:
        origin = pd.Timestamp("2140-01-01 00:17")
        origins = pd.DataFrame({"stay_id": [101], "intime": [origin]})
        for col in ("starttime", "endtime"):
            frame[col] = origin + pd.to_timedelta(frame[col], unit="h")
    out = distribute_amount(frame, val_col="ins", index_col="starttime", admission_times=origins)
    assert out.groupby("starttime").ins.median().tolist() == pytest.approx([3., 14., 3.])
    assert out.ins.sum() == pytest.approx(20.)


def test_r5c_iv_014_short_fluid_intervals_use_absolute_tolerance():
    from types import SimpleNamespace
    from easyicu.concept.callbacks import _callback_fluid_balance_cumulative, _callback_fluid_balance_hourly

    # Long-stay clock must not make a genuine 0.01 h interval a bolus.
    frame = pd.DataFrame({"stay_id": [101], "starttime": [1999.995], "endtime": [2000.005], "value": [100.]})
    out = distribute_volume_hourly(frame)
    assert out.starttime.tolist() == [1999., 2000.]
    assert out.value.tolist() == pytest.approx([50., 50.])
    assert out.value.sum() == pytest.approx(100.)
    intake = out.rename(columns={"starttime": "charttime", "value": "total_input_ml"})
    urine = intake[["stay_id", "charttime"]].assign(urine=[10., 20.])
    ctx = SimpleNamespace()
    balance = _callback_fluid_balance_hourly({
        "total_input_ml": ICUTable(intake, id_columns=["stay_id"], index_column="charttime", value_column="total_input_ml"),
        "urine": ICUTable(urine, id_columns=["stay_id"], index_column="charttime", value_column="urine"),
    }, ctx)
    cumulative = _callback_fluid_balance_cumulative({"fluid_balance": balance}, ctx)
    assert balance.data.fluid_balance.tolist() == pytest.approx([40., 30.])
    assert cumulative.data.fluid_balance_cumulative.tolist() == pytest.approx([40., 70.])


def _declared_chartevents_csv(rows: pd.DataFrame, path) -> None:
    """A raw MIMIC-III CHARTEVENTS CSV in its declared schema (the strict CSV reader's contract).

    Every configured column in order; integer columns as integers; timestamps
    in the configured format. Columns the rows lack are empty.
    """
    lower = rows.rename(columns=str.lower)
    table = pd.DataFrame(index=lower.index)
    for column in load_src_cfg("mimic").get_table("chartevents").columns.values():
        name, key = column["name"], column["name"].lower()
        values = lower[key] if key in lower else pd.Series([None] * len(lower), index=lower.index, dtype=object)
        if column["spec"] == "col_integer":
            values = pd.to_numeric(values).astype("Int64")
        elif column["spec"] == "col_datetime":
            values = pd.to_datetime(values).dt.strftime(column.get("format") or "%Y-%m-%d %H:%M:%S")
        table[name] = values
    table.to_csv(path, index=False)


@pytest.mark.parametrize("storage", ["parquet", "csv", "callable", "preloaded", "dataset"])
def test_r5c_iii_008_projection_never_discards_the_error_filter(tmp_path, storage):
    rows = pd.DataFrame({"icustay_id": [101, 101], "charttime": pd.to_datetime(["2140-01-02 01:00"] * 2),
                         "itemid": [211, 211], "valuenum": [70., 900.], "value": ["70", "900"],
                         "valueuom": "bpm", "error": [None, 1.]})
    config = load_src_cfg("mimic").model_copy(deep=True)
    if storage == "dataset":
        config.tables["chartevents"].dataset = DatasetOptions(path="chartevents.parquet")
    source = ICUDataSource(config, base_path=tmp_path, enable_cache=False)
    if storage in {"parquet", "dataset"}:
        write_fixture_parquet(rows, tmp_path / "chartevents.parquet", index=False)
    elif storage == "csv":
        _declared_chartevents_csv(rows, tmp_path / "CHARTEVENTS.csv.gz")
    elif storage == "callable":
        source.register_table_source("chartevents", lambda: rows.copy())
    else:
        source._preloaded_tables["chartevents"] = rows
    out = source.load_table("chartevents", columns=["valuenum"],
                            filters=[FilterSpec("itemid", FilterOp.IN, [211])] if storage == "csv" else None)
    assert out.data.valuenum.tolist() == [70.]


def test_r5c_iv_010_source_projection_keeps_rate_units_and_dosing_weight(tmp_path):
    rows = pd.DataFrame({"stay_id": [101], "hadm_id": [11], "starttime": [0.25], "endtime": [0.75],
                         "itemid": [222168], "amount": [60.], "amountuom": ["mg"],
                         "rate": [600.], "rateuom": ["mg/hour"], "patientweight": [50.], "linkorderid": [1],
                         "statusdescription": "FinishedRunning"})
    write_fixture_parquet(rows, tmp_path / "inputevents.parquet", index=False)
    source = ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path, enable_cache=False)
    result = ConceptResolver(load_dictionary("miiv")).load_concepts(
        ["propofol_rate"], source, patient_ids=[101], r_compatible=False, concept_workers=1, merge=False,
        align_to_admission=False,
    )["propofol_rate"].data
    assert result.propofol_rate.tolist() == pytest.approx([200.])


def test_r5c_iv_012_public_insulin_load_preserves_hourly_total(tmp_path):
    write_fixture_parquet(_stays(), tmp_path / "icustays.parquet", index=False)
    rows = pd.DataFrame({"stay_id": [101] * 3, "hadm_id": [11] * 3, "itemid": [223258, 223258, 223260],
                         "starttime": [.5, 1.2, 1.7], "endtime": [2.5, 1.2, 1.7],
                         "amount": [12., 3., 5.], "amountuom": "units", "rate": [6., None, None],
                         "linkorderid": [1, 2, 3], "statusdescription": "FinishedRunning"})
    for column in ("starttime", "endtime"):
        rows[column] = pd.Timestamp("2140-01-02") + pd.to_timedelta(rows[column], unit="h")
    write_fixture_parquet(rows, tmp_path / "inputevents.parquet", index=False)
    source = ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path, enable_cache=False)
    result = ConceptResolver(load_dictionary("miiv")).load_concepts(
        ["ins"], source, patient_ids=[101], r_compatible=False, concept_workers=1, merge=False,
    )["ins"].data
    assert result.ins.tolist() == pytest.approx([3., 14., 3.])
    assert result.ins.sum() == pytest.approx(20.)


@pytest.mark.parametrize("path", ["source", "bucket", "batch"])
def test_r5c_iv_008_overlap_is_counted_and_resolved_by_rolling_attribution(tmp_path, path):
    rows = _labs().loc[lambda x: x.subject_id.eq(2)]
    source = _source(tmp_path, "miiv", "labevents", rows)
    if path == "source":
        result = source.load_table("labevents", filters=[FilterSpec("stay_id", FilterOp.IN, [201])]).data
    elif path == "bucket":
        result = load_bucketed_table_aggregated(source, "labevents", "valuenum", [50912], patient_ids=[201])
    else:
        result = load_bucketed_table_multi_aggregated(source, "labevents", {"crea": [50912]}, patient_ids=[201])
    assert result.stay_id.tolist() == [201]
    assert result.attrs["easyicu_miiv_lab_attribution"]["ambiguous_null_hadm_rows"] == 1
    assert result.attrs["easyicu_miiv_lab_attribution"]["attributed_null_hadm_rows"] == 1


def test_r5c_iii_008_error_only_csv_is_a_valid_empty_source(tmp_path):
    _declared_chartevents_csv(
        pd.DataFrame({"ICUSTAY_ID": [101], "CHARTTIME": ["2140-01-02"], "ITEMID": [211],
                      "VALUENUM": [900.], "VALUE": ["900"], "VALUEUOM": ["bpm"], "ERROR": [1.]}),
        tmp_path / "CHARTEVENTS.csv.gz",
    )
    source = ICUDataSource(load_src_cfg("mimic"), base_path=tmp_path, enable_cache=False)
    result = source.load_table("chartevents", filters=[FilterSpec("itemid", FilterOp.IN, [211])])
    assert result.data.empty


def test_r5c_iii_008_channel_priorities_and_cvp_derivations_use_valid_rows_only(tmp_path):
    rows = pd.DataFrame({"icustay_id": [101] * 8,
                         "charttime": pd.to_datetime(["2140-01-02 01:00"] * 7 + ["2140-01-02 20:00"]),
                         "itemid": [113, 113, 52, 52, 456, 8368, 220615, 220615],
                         "valuenum": [8., 40., 80., 160., 75., 55., 20., 5.],
                         "value": ["8", "40", "80", "160", "75", "55", "20", "5"],
                         "valueuom": ["mmHg"] * 6 + ["mg/dL"] * 2,
                         "error": [0., 1., None, 1., 0., 0., 1., 1.]})
    source = _source(tmp_path, "mimic", "chartevents", rows)
    write_fixture_parquet(pd.DataFrame({"subject_id": [1], "hadm_id": [11], "itemid": [50912],
                  "charttime": pd.to_datetime(["2140-01-02 01:00"]),
                  "valuenum": [1.2], "valueuom": ["mg/dL"]}), tmp_path / "labevents.parquet", index=False)
    result = ConceptResolver(load_dictionary("mimic")).load_concepts(
        ["map", "cvp", "cvp_raw", "crea"], source, patient_ids=[101], merge=False, r_compatible=False,
        concept_workers=1,
    )
    assert result["cvp_raw"].data.cvp_raw.tolist() == [8.]
    assert result["cvp"].data.cvp.tolist() == [8.]
    assert result["map"].data["map"].tolist() == [80.]
    assert (result["map"].data["map"].iloc[0] - result["cvp"].data.cvp.iloc[0]) == 72.
    # Lab priority keeps its result; an error-only chart hour cannot fill a gap.
    assert result["crea"].data.crea.tolist() == pytest.approx([1.2])
    assert result["crea"].data.charttime.tolist() == [1.]


@pytest.mark.parametrize("path", ["bucket", "batch"])
def test_r5c_iv_008_string_times_obey_the_same_lab_window(tmp_path, path):
    rows = _labs()
    rows["charttime"] = rows.charttime.astype(str)
    source = _source(tmp_path, "miiv", "labevents", rows)
    if path == "bucket":
        result = load_bucketed_table_aggregated(source, "labevents", "valuenum", [50912], patient_ids=[101])
        values = result.valuenum
    else:
        result = load_bucketed_table_multi_aggregated(source, "labevents", {"crea": [50912]}, patient_ids=[101])
        values = result.crea
    assert sorted(values) == pytest.approx([1.2, 3.0])


@pytest.mark.parametrize("path", ["source", "resolver"])
def test_r5c_iv_008_subject_pushdown_preserves_requested_stays(tmp_path, path, monkeypatch):
    stays = _stays().iloc[[0]].copy()
    later = stays.assign(hadm_id=12, stay_id=102, intime=pd.Timestamp("2140-02-02"),
                         outtime=pd.Timestamp("2140-02-04"))
    write_fixture_parquet(pd.concat([stays, later]), tmp_path / "icustays.parquet", index=False)
    rows = _labs().iloc[[0]].copy()
    later_lab = rows.assign(hadm_id=12, charttime=pd.Timestamp("2140-02-02 01:00"), valuenum=3.)
    rows = pd.concat([rows, later_lab], ignore_index=True)
    source = ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path, enable_cache=False)
    source.register_table_source("labevents", lambda: rows.copy())
    if path == "source":
        result = source.load_table("labevents", filters=[FilterSpec(
            "subject_id", FilterOp.IN, [1], metadata={"original_stay_ids": [101]},
        )]).data
    else:
        monkeypatch.setattr(source, "resolve_bucket_directory", lambda table: None)
        result = ConceptResolver(load_dictionary("miiv")).load_concepts(
            ["crea"], source, patient_ids=[101], merge=False, r_compatible=False, concept_workers=1,
        )["crea"].data
    assert result.stay_id.tolist() == [101]


@pytest.mark.parametrize("path", ["source", "bucket", "batch"])
def test_r5c_iv_008_optional_lab_units_do_not_disable_attribution(tmp_path, path):
    source = _source(tmp_path, "miiv", "labevents", _labs().drop(columns="valueuom"))
    if path == "source":
        result = source.load_table("labevents", filters=[FilterSpec("stay_id", FilterOp.IN, [101])]).data
        assert sorted(result.loc[result.itemid.eq(50912), "valuenum"]) == pytest.approx([1.2, 2., 4.])
    elif path == "bucket":
        result = load_bucketed_table_aggregated(source, "labevents", "valuenum", [50912], patient_ids=[101])
        assert sorted(result.valuenum) == pytest.approx([1.2, 3.])
    else:
        result = load_bucketed_table_multi_aggregated(source, "labevents", {"crea": [50912]}, patient_ids=[101])
        assert sorted(result.crea) == pytest.approx([1.2, 3.])


@pytest.mark.parametrize("storage", ["preloaded", "dataset"])
def test_r5c_iii_008_uppercase_projection_keeps_valid_measurements(tmp_path, storage):
    rows = pd.DataFrame({"ICUSTAY_ID": [101] * 3, "CHARTTIME": pd.to_datetime(["2140-01-02 01:00"] * 3),
                         "ITEMID": [211] * 3, "VALUENUM": [60., 80., 900.],
                         "VALUE": ["60", "80", "900"], "VALUEUOM": "bpm", "ERROR": [None, 0., 1.]})
    config = load_src_cfg("mimic").model_copy(deep=True)
    if storage == "dataset":
        write_fixture_parquet(rows, tmp_path / "chartevents.parquet", index=False)
        config.tables["chartevents"].dataset = DatasetOptions(path="chartevents.parquet")
    source = ICUDataSource(config, base_path=tmp_path, enable_cache=False)
    if storage == "preloaded":
        source._preloaded_tables["chartevents"] = rows
    result = source.load_table("chartevents", columns=["icustay_id", "valuenum"],
                               filters=[FilterSpec("icustay_id", FilterOp.IN, [101])]).data
    assert result.valuenum.tolist() == [60., 80.]


@pytest.mark.parametrize("storage", ["parquet", "bucket"])
def test_r5c_iv_012_rewritten_insulin_is_not_an_additional_dose(tmp_path, storage):
    rows = pd.DataFrame({"stay_id": [101] * 3, "hadm_id": [11] * 3, "itemid": [223258] * 3,
                         "starttime": [0., 0., 0.5], "endtime": [1., 1., 0.5],
                         "amount": [12., 8., 3.], "amountuom": "units", "rate": [12., 8., None],
                         "rateuom": "units/hour", "linkorderid": [1, 1, 2],
                         "statusdescription": ["Rewritten", "FinishedRunning", "FinishedRunning"]})
    for column in ("starttime", "endtime"):
        rows[column] = pd.Timestamp("2140-01-02") + pd.to_timedelta(rows[column], unit="h")
    if storage == "parquet":
        write_fixture_parquet(_stays(), tmp_path / "icustays.parquet", index=False)
        write_fixture_parquet(rows, tmp_path / "inputevents.parquet", index=False)
        source = ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path, enable_cache=False)
    else:
        source = _source(tmp_path, "miiv", "inputevents", rows)
    result = ConceptResolver(load_dictionary("miiv")).load_concepts(
        ["ins"], source, patient_ids=[101], r_compatible=False, concept_workers=1, merge=False,
    )["ins"].data
    assert result.ins.tolist() == pytest.approx([11.])


@pytest.mark.parametrize("path", ["source", "bucket", "batch"])
@pytest.mark.parametrize("identity", ["labevent_id", "specimen_id"])
def test_r5c_iv_008_independent_lab_events_are_not_cross_channel_duplicates(tmp_path, path, identity):
    rows = _labs().iloc[[2, 1, 1]].copy().reset_index(drop=True)
    # The second row repeats the first source event; the third is another
    # specimen with an equal measurement at the same recorded time.
    rows[identity] = [1, 1, 2]
    source = _source(tmp_path, "miiv", "labevents", rows)
    if path == "source":
        result = source.load_table("labevents", columns=["valuenum"]).data
        assert result.valuenum.tolist() == [2., 2.]
    elif path == "bucket":
        result = load_bucketed_table_aggregated(source, "labevents", "valuenum", [50912], agg_func="sum")
        assert result.valuenum.tolist() == [4.]
    else:
        result = load_bucketed_table_multi_aggregated(source, "labevents", {"crea": [50912]}, agg_func="sum")
        assert result.crea.tolist() == [4.]
    assert result.attrs["easyicu_miiv_lab_attribution"]["deduplicated_null_hadm_rows"] == 1


def test_r5c_iv_008_lab_attribution_preserves_conflicting_specimen_labels(tmp_path):
    rows = _labs().iloc[[2, 1]].copy().reset_index(drop=True)
    rows["itemid"] = 52033
    rows["valuenum"] = float("nan")
    rows["value"] = ["ART.", "VEN."]
    rows["specimen_id"] = 10
    source = _source(tmp_path, "miiv", "labevents", rows)
    result = source.load_table("labevents", columns=["subject_id", "specimen_id", "itemid", "value"],
                               filters=[FilterSpec("itemid", FilterOp.IN, [52033])]).data
    assert set(result.value) == {"ART.", "VEN."}


@pytest.mark.parametrize("database", ["miiv", "mimic"])
@pytest.mark.parametrize("storage", ["parquet", "bucket"])
def test_r5c_iv_011_public_rate_loads_use_converted_values_and_exclusive_stops(tmp_path, database, storage):
    cases = [
        ("adh_rate", 222315, 3., "units/hour", .05),
        ("dobu_rate", 221653, 2., "mcg/kg/min", 2.),
        ("dopa_rate", 221662, 2., "mcg/kg/min", 2.),
        ("epi_rate", 221289, .1, "mcg/kg/min", .1),
        ("fentanyl_rate", 221744, 2., "mcg/kg/hour", 100.),
        ("midazolam_rate", 221668, 2., "mg/hour", 2.),
        ("norepi_rate", 221906, .6, "mg/hour", .2),
        ("phn_rate", 221749, 10., "mcg/min", .2),
        ("propofol_rate", 222168, 600., "mg/hour", 200.),
    ]
    identifier = "stay_id" if database == "miiv" else "icustay_id"
    table = "inputevents" if database == "miiv" else "inputevents_mv"
    rows = pd.DataFrame({identifier: 101, "hadm_id": 11, "itemid": [r[1] for r in cases],
                         "starttime": pd.Timestamp("2140-01-02 00:15"),
                         "endtime": pd.Timestamp("2140-01-02 01:00"),
                         "rate": [r[2] for r in cases], "rateuom": [r[3] for r in cases],
                         "amount": 1., "amountuom": "unused", "patientweight": 50.,
                         "linkorderid": range(len(cases)), "statusdescription": "FinishedRunning"})
    if storage == "parquet":
        write_fixture_parquet(_stays(database), tmp_path / "icustays.parquet", index=False)
        write_fixture_parquet(rows, tmp_path / f"{table}.parquet", index=False)
        source = ICUDataSource(load_src_cfg(database), base_path=tmp_path, enable_cache=False)
    else:
        source = _source(tmp_path, database, table, rows)
    if database == "mimic":
        source.register_table_source("inputevents_cv", lambda: pd.DataFrame(
            columns=["icustay_id", "charttime", "itemid", "rate", "rateuom", "amount", "linkorderid"]))
    result = ConceptResolver(load_dictionary(database)).load_concepts(
        [r[0] for r in cases], source, patient_ids=[101], merge=False,
        r_compatible=False, concept_workers=1,
    )
    for name, _, _, _, expected in cases:
        assert result[name].data.charttime.tolist() == [0.], name
        assert result[name].data[name].tolist() == pytest.approx([expected]), name


@pytest.mark.parametrize("database,module,concept,item,value,expected", [
    ("miiv", "chemistry", "crea", 50912, 1.2, 1.2),
    ("mimic", "vitals", "hr", 211, 70., 70.),
    ("miiv", "medications", "ins", 223258, 3., 6.),
])
@pytest.mark.parametrize("streamed", [False, True])
def test_r5c_iv_008_iii_008_iv_012_native_modules_use_the_fixed_sources(
    tmp_path, database, module, concept, item, value, expected, streamed,
):
    import json
    from easyicu.api import extraction as api
    from easyicu.api.concepts import clear_global_loader

    raw = tmp_path / "source"
    raw.mkdir()
    identifier = "stay_id" if database == "miiv" else "icustay_id"
    if concept == "ins":
        table = "inputevents"
        rows = pd.DataFrame({identifier: [101, 101], "hadm_id": 11, "itemid": item,
                             "starttime": pd.Timestamp("2140-01-02 01:15"),
                             "endtime": pd.Timestamp("2140-01-02 01:15"),
                             "amount": value, "amountuom": "units", "rate": float("nan"),
                             "rateuom": "units/hour", "linkorderid": [1, 2]})
    else:
        table = "labevents" if database == "miiv" else "chartevents"
        rows = pd.DataFrame({"subject_id": [1, 1], "hadm_id": [None, 11],
                             "charttime": pd.to_datetime(["2140-01-02 01:00"] * 2),
                             "itemid": item, "valuenum": [value, value], "valueuom": ""})
        if database == "miiv":
            rows = rows.iloc[[0]]
        else:
            rows[identifier] = 101
            rows["error"] = [0., 1.]
            rows["valuenum"] = [value, 150.]
            rows["value"] = rows.valuenum.astype(str)
    _source(raw, database, table, rows)
    output = tmp_path / "native"
    output.mkdir()
    try:
        api._run_module_extraction(
            module_name=module, concepts=[concept], database=database, data_path=str(raw),
            patient_ids_filter={identifier: [101]}, batch_size=1, output_dir=str(output),
            use_sofa2=False, stream_output_batches=streamed,
        )
        manifest = json.loads((output / "_manifest.json").read_text())
        assert manifest["errors"] == []
        move_fixture_parquet(output / "_manifest.json", output / f"{module}.manifest.json")
        api._publish_native_export_v2(
            database=database, data_path=str(raw), output_dir=str(output), modules=[module],
            max_patients=None, result={"modules": {module: manifest}},
            concept_projection={module: [concept]},
        )
        result = pd.read_parquet(output / f"{module}.parquet")
        assert result.stay_id.tolist() == [101]
        assert result.charttime.tolist() == [1.]
        assert result[concept].tolist() == pytest.approx([expected])
    finally:
        clear_global_loader()


def test_r5c_iv_008_null_admission_creatinine_enters_the_baseline_history(tmp_path):
    from easyicu.scores.kdigo_aki import kdigo_creatinine

    source = _source(tmp_path, "miiv", "labevents", _labs())
    result = ConceptResolver(load_dictionary("miiv")).load_concepts(
        ["crea"], source, patient_ids=[101], merge=False, r_compatible=False, concept_workers=1,
        interval=pd.Timedelta(hours=1),
    )["crea"].data
    history = kdigo_creatinine(result, time_unit="hours")
    assert history.loc[history.charttime.eq(2), "creat_low_past_7day"].item() == pytest.approx(1.2)
