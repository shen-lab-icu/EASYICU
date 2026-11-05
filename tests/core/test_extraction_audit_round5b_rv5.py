"""Round 5 residue audit: invented void records, no clinical data."""

from __future__ import annotations

from tests.support.bucket_storage import move_fixture_parquet

from tests.support.bucket_storage import write_fixture_parquet

from tests.support.bucket_storage import synthetic_data_source

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import easyicu
from easyicu.concept import ConceptResolver
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.errors import ConceptExtractionUnavailable
from easyicu.concept.schema import ConceptDictionary, ConceptSource
from easyicu.config import DatasetOptions, load_src_cfg
from easyicu.datasource import (
    FilterOp,
    FilterSpec,
    ICUDataSource,
    _close_duckdb_connections,
    load_bucketed_table_aggregated,
    load_bucketed_table_multi_aggregated,
)
from easyicu.utils.callback_utils import _hirid_valid_pharma_rows


@pytest.fixture(autouse=True)
def bounded_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_MEMORY_LIMIT_GB", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(tmp_path / "spill"))
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "state"))
    yield
    _close_duckdb_connections()


def _dictionary(filename="concept-dict.json"):
    return json.loads((Path(easyicu.__file__).parent / "data" / filename).read_text())


def _source(tmp_path, db, table, rows, storage="flat"):
    stay = "stay_id" if db == "miiv" else "icustay_id"
    write_fixture_parquet(pd.DataFrame({
        "subject_id": [1, 2], "hadm_id": [11, 22], stay: [101, 102],
        "intime": pd.to_datetime(["2140-01-02"] * 2),
        "outtime": pd.to_datetime(["2140-01-04"] * 2),
    }), tmp_path / "icustays.parquet", index=False)
    config = load_src_cfg(db).model_copy(deep=True)
    if storage == "bucket":
        directory = tmp_path / f"{table}_bucket" / "bucket_id=0"
        directory.mkdir(parents=True)
        write_fixture_parquet(rows, directory / "part.parquet", index=False)
    elif storage == "csv":
        rows.rename(columns=str.upper).to_csv(tmp_path / f"{table.upper()}.csv.gz", index=False)
    elif storage not in {"callable", "preloaded"}:
        write_fixture_parquet(rows, tmp_path / f"{table}.parquet", index=False)
        if storage == "dataset":
            config.tables[table].dataset = DatasetOptions(path=f"{table}.parquet")
    source = ICUDataSource(config, base_path=tmp_path, enable_cache=False)
    if storage == "csv":
        source.register_table_source(table, tmp_path / f"{table.upper()}.csv.gz")
    elif storage == "callable":
        source.register_table_source(table, lambda: rows.copy())
    elif storage == "preloaded":
        source._preloaded_tables[table] = rows
    return source


def _load(source, payload, names):
    return ConceptResolver(ConceptDictionary.from_payload(payload)).load_concepts(
        names, source, patient_ids=[101, 102], merge=False, r_compatible=False,
        concept_workers=1, interval=pd.Timedelta(hours=1),
    )


_INPUTS = [
    (name, index)
    for name, definition in _dictionary().items()
    for index, source in enumerate(definition.get("sources", {}).get("miiv", []))
    if source.get("table") == "inputevents"
    and "set_val" not in source.get("callback", "")
    and name not in {"abx", "norepi_dur", "epi_dur", "dopa_dur", "dobu_dur"}
]


@pytest.mark.parametrize("name,index", _INPUTS)
@pytest.mark.parametrize("storage", ["flat", "bucket", "batch"])
def test_RV_01_miiv_nonboolean_inputs_exclude_rewritten_before_callbacks(tmp_path, name, index, storage):
    definition = _dictionary()[name]
    spec = definition["sources"]["miiv"][index]
    # This declaration also disables SQL paths that discard raw status fields.
    assert spec.get("status_var") == "statusdescription"
    assert "Rewritten" in spec.get("excluded_statuses", [])
    assert "statusdescription" in spec.get("extra_vars", [])
    item = spec["ids"][0] if isinstance(spec["ids"], list) else spec["ids"]
    times = pd.Timestamp("2140-01-02") + pd.to_timedelta([.25, 1.25, 2.25], unit="h")
    unit = "units/min" if name == "adh_rate" else "mcg/hour" if name == "fentanyl_rate" else "mcg/kg/min"
    rows = pd.DataFrame({
        "stay_id": 101, "hadm_id": 11, "itemid": item, "starttime": times,
        "endtime": times + pd.Timedelta(minutes=30), "rate": [.01, .02, .1] if name == "adh_rate" else [1., 2., 3.],
        "rateuom": unit, "patientweight": 50., "amount": [1., 2., 3.],
        "amountuom": "ml" if name in {"dex", "total_input_ml"} else "units" if name == "ins" else "mg",
        "linkorderid": [1, 2, 3], "statusdescription": [None, "FinishedRunning", "Rewritten"],
    })
    source = _source(tmp_path, "miiv", "inputevents", rows, "bucket" if storage != "flat" else "flat")
    definition["sources"] = {"miiv": [spec]}
    payload = {name: definition}
    names = [name]
    if storage == "batch":
        payload["valid_amount"] = {"sources": {"miiv": [{
            "table": "inputevents", "sub_var": "itemid", "ids": [999001], "val_var": "amount",
        }]}}
        names.append("valid_amount")
    table = _load(source, payload, names)[name]
    frame = table.data
    assert not frame.empty
    assert set(frame[table.index_column]) == {0., 1.}


@pytest.mark.parametrize("db", ["mimic", "mimic_demo"])
@pytest.mark.parametrize("uppercase", [False, True])
@pytest.mark.parametrize("path", [
    "flat", "csv", "callable", "preloaded", "dataset", "bucket", "arrow_bucket",
    "aggregate", "batch", "resolver_flat", "resolver_bucket", "csv_fallback",
])
def test_RV_02_output_iserror_removed_on_all_load_paths(tmp_path, monkeypatch, db, uppercase, path):
    records = [
        (101, item, value, flag)
        for item in (40055, 40048, 40286)
        for value, flag in ((10., None), (20., "0"), (999., "1"), (888., "1.0"))
    ] + [(102, 40286, 999., "1")]
    rows = pd.DataFrame(records, columns=["icustay_id", "itemid", "value", "iserror"])
    rows["charttime"] = pd.Timestamp("2140-01-02 01:00")
    rows["valueuom"] = "ml"
    if uppercase:
        rows.columns = rows.columns.str.upper()
    storage = "bucket" if path in {"bucket", "arrow_bucket", "aggregate", "batch", "resolver_bucket"} else "csv" if path == "csv_fallback" else "flat" if path == "resolver_flat" else path
    source = _source(tmp_path, db, "outputevents", rows, storage)
    if path == "aggregate":
        frame = load_bucketed_table_aggregated(source, "outputevents", "value", [40055], agg_func="sum")
        assert frame.value.tolist() == [30.]
    elif path == "batch":
        frame = load_bucketed_table_multi_aggregated(
            source, "outputevents", {"urine": [40055], "other_output_ml": [40048], "rrt": [40286]},
            agg_func="sum", value_column="value",
        )
        assert frame.icustay_id.tolist() == [101]
        assert frame[["urine", "other_output_ml", "rrt"]].iloc[0].tolist() == [30.] * 3
    elif path.startswith("resolver"):
        result = _load(source, _dictionary(), ["urine", "other_output_ml", "rrt"])
        assert result["urine"].data.urine.sum() == 30.
        assert result["other_output_ml"].data.other_output_ml.sum() == 30.
        assert set(result["rrt"].data.icustay_id) == {101}
    else:
        if path == "arrow_bucket":
            monkeypatch.setattr(source, "_read_partitioned_data_duckdb", lambda *args, **kwargs: source._read_partitioned_data_optimized(*args[:3], itemid_filter_config=kwargs.get("itemid_filter_config")))
        if path == "csv_fallback":
            frame = source._read_mimic3_csv_fallback(
                tmp_path / "OUTPUTEVENTS.csv.gz", ["value"], ("itemid", {40055}),
            )
            assert pd.to_numeric(frame.value).tolist() == [10., 20.]
        else:
            frame = source.load_table("outputevents", columns=["value", "itemid"], filters=[
                FilterSpec("itemid", FilterOp.IN, [40055]),
            ]).data
            assert frame.value.tolist() == [10., 20.]


_PROCEDURES = [
    (filename, db, name, index)
    for filename in ("concept-dict.json", "sofa2-dict.json")
    for name, definition in _dictionary(filename).items()
    for db, sources in definition.get("sources", {}).items()
    if db in {"miiv", "mimic", "mimic_demo"}
    for index, source in enumerate(sources)
    if source.get("table") in {"procedureevents", "procedureevents_mv"}
]


@pytest.mark.parametrize("filename,db,name,index", _PROCEDURES)
@pytest.mark.parametrize("storage", ["callback", "flat", "bucket", "without_cancel_flat", "without_cancel_bucket"])
def test_RV_03_procedure_sources_exclude_rewritten_and_cancelled(tmp_path, filename, db, name, index, storage):
    definition = _dictionary(filename)[name]
    if definition.get("class_name") == "lgl_cncpt":
        assert definition.get("aggregate") == "max"
    spec = definition["sources"][db][index]
    stay = "stay_id" if db == "miiv" else "icustay_id"
    item = spec["ids"][0]
    times = pd.Timestamp("2140-01-02") + pd.to_timedelta(range(7), unit="h")
    rows = pd.DataFrame({
        stay: 101, "hadm_id": 11, "itemid": item, "value": item,
        "starttime": times, "endtime": times + pd.Timedelta(minutes=30),
        "valueuom": "None", "row_id": range(7),
        "statusdescription": [None, "FinishedRunning", "Rewritten", "FinishedRunning", "FinishedRunning", "Cancelled", "Canceled"],
        "cancelreason": [None, 0, 0, 1, 2, 0, 0],
    })
    if storage.startswith("without_cancel_") and db == "miiv":
        # Converted extracts may omit cancelreason; both native schemas declare it.
        rows.loc[rows.cancelreason.gt(0), "statusdescription"] = "Cancelled"
        rows = rows.drop(columns="cancelreason")
    if storage == "callback":
        rows[name] = item
        rows["dur_var"] = 30.
        frame = _apply_callback(rows, ConceptSource.from_mapping(spec), name)
        assert frame.row_id.tolist() == [0, 1]
    else:
        source = _source(tmp_path, db, spec["table"], rows, storage.removeprefix("without_cancel_"))
        definition["sources"] = {db: [spec]}
        table = _load(source, {name: definition}, [name])[name]
        frame = table.data
        assert not frame.empty
        assert set(frame[table.index_column]) == {0., 1.}


@pytest.mark.parametrize("db", ["mimic", "mimic_demo"])
@pytest.mark.parametrize("uppercase", [False, True])
def test_RV_04_weight_id_tbl_duckdb_filters_error_before_median(tmp_path, monkeypatch, db, uppercase):
    from easyicu import datasource

    rows = pd.DataFrame({
        "icustay_id": [101] * 3 + [102] * 2, "itemid": [762] * 3 + [763] * 2,
        "valuenum": [60., 80., 300., 70., 310.], "value": ["60", "80", "300", "70", "310"],
        "charttime": pd.Timestamp("2140-01-02 01:00"), "valueuom": "kg", "error": [None, 0., 1., 0., 1.],
    })
    if uppercase:
        rows.columns = rows.columns.str.upper()
    source = _source(tmp_path, db, "chartevents", rows, "bucket")
    executed = []
    connection = datasource._get_duckdb_connection()

    class RecordingConnection:
        def __getattr__(self, name):
            return getattr(connection, name)

        def execute(self, sql, *args, **kwargs):
            executed.append(sql)
            return connection.execute(sql, *args, **kwargs)

    monkeypatch.setattr(datasource, "_get_duckdb_connection", RecordingConnection)
    definition = _dictionary()["weight"]
    definition["sources"] = {db: [s for s in definition["sources"][db] if s["table"] == "chartevents"]}
    frame = _load(source, {"weight": definition}, ["weight"])["weight"].data
    assert any("MEDIAN(" in sql and "GROUP BY icustay_id" in sql for sql in executed)
    assert frame.set_index("icustay_id").weight.to_dict() == {101: 70., 102: 70.}


@pytest.mark.parametrize("path", ["helper", "callback", "combined", "resolver_flat", "resolver_bucket"])
def test_RV_05_hirid_missing_pharma_status_is_typed_unavailability(tmp_path, path):
    rows = pd.DataFrame({
        "patientid": [101], "givenat": [1.], "pharmaid": [1000724],
        "givendose": [2.], "doseunit": ["units"], "route": ["iv-inf"], "infusionid": [7],
    })
    callback_source = ConceptSource.from_mapping({
        "table": "pharma", "val_var": "givendose", "index_var": "givenat", "unit_var": "doseunit",
        "callback": "combine_callbacks(hirid_rate, convert_unit(binary_op(`*`, 2), 'units/min'))",
    })
    context = SimpleNamespace(config=SimpleNamespace(name="hirid"))
    if path == "combined":
        # Aggregation consumes recordstatus. Later transforms must accept the
        # already-validated result, while the raw source still requires it.
        valid = _apply_callback(rows.assign(recordstatus=0), callback_source, "ins", "doseunit", data_source=context)
        assert not valid.empty
        assert valid.ins.tolist() == pytest.approx([2. / 60 * 2])
    with pytest.raises(ConceptExtractionUnavailable) as caught:
        if path == "helper":
            _hirid_valid_pharma_rows(rows)
        elif path == "callback":
            source = ConceptSource.from_mapping({"table": "pharma", "val_var": "givendose"})
            _apply_callback(rows, source, "ins", data_source=context)
        elif path == "combined":
            _apply_callback(rows, callback_source, "ins", "doseunit", data_source=context)
        else:
            if path.endswith("bucket"):
                directory = tmp_path / "pharma_bucket" / "bucket_id=0"
                directory.mkdir(parents=True)
                write_fixture_parquet(rows, directory / "part.parquet", index=False)
            else:
                write_fixture_parquet(rows, tmp_path / "pharma.parquet", index=False)
            source = ICUDataSource(load_src_cfg("hirid"), base_path=tmp_path, enable_cache=False)
            _load(source, {"ins": _dictionary()["ins"]}, ["ins"])
    assert caught.value.database == "hirid"
    assert caught.value.stage == "pharma_recordstatus"
    assert "recordstatus" in caught.value.detail


@pytest.mark.parametrize("db", ["mimic", "mimic_demo"])
@pytest.mark.parametrize("storage", ["flat", "bucket"])
@pytest.mark.parametrize("status,cancel", [("Rewritten", 0), ("FinishedRunning", 1), ("Cancelled", 0), ("Canceled", 0)])
def test_RV_03_auxiliary_void_extubation_does_not_split_chart_ventilation(tmp_path, db, storage, status, cancel):
    times = pd.Timestamp("2140-01-02") + pd.to_timedelta([0, 4, 10, 12], unit="h")
    charts = pd.DataFrame({
        "icustay_id": 101, "charttime": times, "itemid": 60,
        "value": "5", "valuenum": 5., "valueuom": "cmH2O", "error": 0,
    })
    source = _source(tmp_path, db, "chartevents", charts, storage)
    # The first marker is void; the genuine extubation still stops the episode.
    procedures = pd.DataFrame({
        "icustay_id": 101, "starttime": pd.Timestamp("2140-01-02") + pd.to_timedelta([5, 13], unit="h"),
        "itemid": 227194, "statusdescription": [status, None], "cancelreason": [cancel, 0],
    })
    if storage == "bucket":
        directory = tmp_path / "procedureevents_mv_bucket" / "bucket_id=0"
        directory.mkdir(parents=True)
        write_fixture_parquet(procedures, directory / "part.parquet", index=False)
    else:
        write_fixture_parquet(procedures, tmp_path / "procedureevents_mv.parquet", index=False)
    spec = _dictionary()["mech_vent"]["sources"][db][0]
    result = _apply_callback(charts, ConceptSource.from_mapping(spec), "mech_vent", data_source=source)
    assert result.charttime.tolist() == [times[0]]
    assert result.dur_var.tolist() == [13 * 60.]


@pytest.mark.parametrize("db", ["miiv", "mimic"])
@pytest.mark.parametrize("storage", ["flat", "bucket"])
def test_RV_01_direct_fluid_balance_callback_excludes_rewritten(tmp_path, db, storage):
    from easyicu.concept.callbacks import _callback_fluid_balance_admitted
    from easyicu.table import ICUTable

    stay = "stay_id" if db == "miiv" else "icustay_id"
    table = "inputevents" if db == "miiv" else "inputevents_mv"
    rows = pd.DataFrame({
        stay: 101, "hadm_id": 11, "starttime": [0., 1., 1.], "amount": [20., 40., 999.],
        "endtime": [1., 2., 2.], "itemid": 220970, "rate": [20., 40., 999.], "linkorderid": [1, 2, 3],
        "amountuom": "mL", "statusdescription": [None, "FinishedRunning", "Rewritten"],
    })
    source = _source(tmp_path, db, table, rows, storage)
    urine = ICUTable(pd.DataFrame({stay: 101, "charttime": [0., 1.], "urine": [5., 10.]}),
                     id_columns=[stay], index_column="charttime", value_column="urine")
    result = _callback_fluid_balance_admitted({"urine": urine}, SimpleNamespace(data_source=source))
    assert result.data.fluid_balance_admitted.tolist() == [15., 45.]


def _native_module(tmp_path, db, module, concept, table, rows, storage, streamed):
    from easyicu.api import extraction as api
    from easyicu.api.concepts import clear_global_loader

    raw, output = tmp_path / "source", tmp_path / "native"
    raw.mkdir()
    output.mkdir()
    _source(raw, db, table, rows, storage)
    identifier = "stay_id" if db == "miiv" else "icustay_id"
    try:
        api._run_module_extraction(
            module_name=module, concepts=[concept], database=db, data_path=str(raw),
            patient_ids_filter={identifier: [101, 102]}, batch_size=1, output_dir=str(output),
            use_sofa2=False, stream_output_batches=streamed,
        )
        manifest = json.loads((output / "_manifest.json").read_text())
        assert manifest["errors"] == []
        move_fixture_parquet(output / "_manifest.json", output / f"{module}.manifest.json")
        api._publish_native_export_v2(
            database=db, data_path=str(raw), output_dir=str(output), modules=[module],
            max_patients=None, result={"modules": {module: manifest}},
            concept_projection={module: [concept]},
        )
        return pd.read_parquet(output / f"{module}.parquet")
    finally:
        clear_global_loader()


@pytest.mark.parametrize("storage", ["flat", "bucket"])
@pytest.mark.parametrize("streamed", [False, True])
def test_RV_01_native_rate_export_ignores_rewritten_inputs(tmp_path, storage, streamed):
    times = pd.Timestamp("2140-01-02") + pd.to_timedelta([.25, 1.25, 2.25], unit="h")
    rows = pd.DataFrame({
        "stay_id": 101, "hadm_id": 11, "starttime": times, "endtime": times + pd.Timedelta(minutes=30),
        "itemid": 221906, "rate": [.1, .2, .3], "rateuom": "mcg/kg/min", "patientweight": 50.,
        "amount": [1., 2., 3.], "amountuom": "mg", "linkorderid": [1, 2, 3],
        "statusdescription": [None, "FinishedRunning", "Rewritten"],
    })
    frame = _native_module(tmp_path, "miiv", "vasopressors", "norepi_rate", "inputevents", rows, storage, streamed)
    values = frame.loc[frame.norepi_rate.notna()]
    assert values.charttime.tolist() == [0., 1.]
    assert values.norepi_rate.tolist() == pytest.approx([.1, .2])


@pytest.mark.parametrize("storage", ["flat", "bucket"])
@pytest.mark.parametrize("streamed", [False, True])
def test_RV_02_native_urine_export_ignores_output_errors(tmp_path, storage, streamed):
    rows = pd.DataFrame({
        "icustay_id": 101, "charttime": pd.Timestamp("2140-01-02 01:00"),
        "itemid": 40055, "value": [10., 20., 999.], "valueuom": "ml", "iserror": [None, 0., 1.],
    })
    frame = _native_module(tmp_path, "mimic", "renal", "urine", "outputevents", rows, storage, streamed)
    assert frame.urine.sum() == 30.


@pytest.mark.parametrize("db", ["miiv", "mimic"])
@pytest.mark.parametrize("storage", ["flat", "bucket"])
@pytest.mark.parametrize("streamed", [False, True])
def test_RV_03_native_rrt_export_ignores_void_procedures(tmp_path, db, storage, streamed):
    stay = "stay_id" if db == "miiv" else "icustay_id"
    table = "procedureevents" if db == "miiv" else "procedureevents_mv"
    rows = pd.DataFrame({
        stay: [101, 102, 102], "hadm_id": [11, 22, 22], "itemid": 225441, "value": 1., "valueuom": "hour",
        "starttime": pd.Timestamp("2140-01-02 01:00"), "endtime": pd.Timestamp("2140-01-02 01:30"),
        "statusdescription": [None, "Rewritten", "FinishedRunning"], "cancelreason": [0, 0, 1],
    })
    frame = _native_module(tmp_path, db, "renal", "rrt", table, rows, storage, streamed)
    assert set(frame.loc[frame.rrt.eq(True), "stay_id"]) == {101}


@pytest.mark.parametrize("streamed", [False, True])
def test_RV_04_native_weight_export_filters_error_before_median(tmp_path, streamed):
    rows = pd.DataFrame({
        "icustay_id": 101, "charttime": pd.Timestamp("2140-01-02 01:00"),
        "itemid": 762, "valuenum": [60., 80., 300.], "value": ["60", "80", "300"],
        "valueuom": "kg", "error": [None, 0., 1.],
    })
    frame = _native_module(tmp_path, "mimic", "demographics", "weight", "chartevents", rows, "bucket", streamed)
    assert frame.loc[frame.stay_id.eq(101), "weight"].tolist() == [70.]


@pytest.mark.parametrize("callback", [None, "hirid_rate"])
def test_RV_05_compatibility_loader_preserves_status_and_fails_closed(tmp_path, callback):
    from importlib import import_module

    loader_class = import_module("easyicu.load_concepts").ConceptLoader
    rows = pd.DataFrame({
        "patientid": [101], "givenat": [1.], "enteredentryat": [1.], "pharmaid": [1000724],
        "givendose": [2.], "doseunit": ["units"], "route": ["iv-inf"], "infusionid": [7],
    })
    path = tmp_path / "pharma.parquet"
    spec = ConceptSource.from_mapping({
        "table": "pharma", "sub_var": "pharmaid", "ids": [1000724], "val_var": "givendose",
        "index_var": "givenat", "unit_var": "doseunit", "callback": callback,
    })

    def load():
        loader = loader_class("hirid", data_path=str(tmp_path), low_memory=True)
        return loader._load_concept_source(spec, "ins", None, "patient", None)

    write_fixture_parquet(pd.concat([
        rows.assign(recordstatus=0), rows.assign(givendose=999., recordstatus=2),
        rows.assign(givendose=888., recordstatus=32),
    ], ignore_index=True), path, index=False)
    valid = load()
    column = "ins" if callback else "value"
    assert valid[column].tolist() == pytest.approx([2. / 60 if callback else 2.])
    write_fixture_parquet(rows, path, index=False)
    with pytest.raises(ConceptExtractionUnavailable, match="recordstatus"):
        load()


@pytest.mark.parametrize("streamed", [False, True])
def test_RV_05_native_module_records_missing_status_as_unavailable(tmp_path, streamed):
    from easyicu.api import extraction as api
    from easyicu.api.concepts import clear_global_loader

    raw, output = tmp_path / "source", tmp_path / "native"
    raw.mkdir()
    output.mkdir()
    write_fixture_parquet(pd.DataFrame({
        "patientid": [101], "givenat": [1.], "pharmaid": [1000724],
        "givendose": [2.], "doseunit": ["units"], "route": ["iv-inf"], "infusionid": [7],
    }), raw / "pharma.parquet", index=False)
    write_fixture_parquet(pd.DataFrame({"patientid": [101], "admissiontime": [pd.Timestamp("2140-01-02")]}),
        raw / "general_table.parquet", index=False,
    )
    try:
        api._run_module_extraction(
            module_name="medications", concepts=["ins"], database="hirid", data_path=str(raw),
            patient_ids_filter={"patientid": [101]}, batch_size=1, output_dir=str(output),
            use_sofa2=False, stream_output_batches=streamed,
        )
        manifest = json.loads((output / "_manifest.json").read_text())
        assert any("recordstatus" in error for error in manifest["errors"])
        assert manifest["saved"] == {}
    finally:
        clear_global_loader()


@pytest.mark.parametrize("workers", [1, 2])
def test_RV_05_compatibility_public_loader_propagates_unavailability(tmp_path, workers):
    from importlib import import_module

    write_fixture_parquet(pd.DataFrame({
        "patientid": [101, 101], "givenat": [1., 1.], "enteredentryat": [1., 1.],
        "pharmaid": [1000724, 1000690], "givendose": [2., 3.], "doseunit": ["units", "ml"],
        "route": "iv-inf", "infusionid": [7, 8],
    }), tmp_path / "pharma.parquet", index=False)
    loader = import_module("easyicu.load_concepts").ConceptLoader("hirid", data_path=str(tmp_path), low_memory=True)
    with pytest.raises(ConceptExtractionUnavailable, match="recordstatus"):
        loader.load_concepts(["ins", "dex"], id_type="patient", merge_data=False,
                             verbose=False, concept_workers=workers)
