"""Round-6 loader and CVP rules, using invented observations only."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.callbacks import ConceptCallbackContext, _callback_cvp_artefact_filter
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_dictionary
from easyicu.table import ICUTable


PRESSURES = ["map", "map_invasive", "map_nibp", "sbp", "dbp", "cvp_raw"]


def _write_table(root, database, table, frame, layout="file"):
    # SIC's source schema lists lowercase names; the loader canonicalizes them.
    declared = load_src_cfg(database).get_table(table).columns
    assert {c.lower() for c in frame} <= {c.lower() for c in declared}, (database, table)
    frame = frame.copy()
    for column in declared:
        if column.lower() not in {c.lower() for c in frame}:
            frame[column] = np.nan
    if layout == "bucket":
        path = root / f"{table}_bucket" / "bucket_id=0" / "part.parquet"
        path.parent.mkdir(parents=True)
        (path.parent.parent / "_COMPLETE").write_text("ready")
    elif layout == "flat":
        path = root / table / "part.parquet"
        path.parent.mkdir()
    else:
        path = root / f"{'general_table' if database == 'hirid' and table == 'general' else table}.parquet"
    frame.to_parquet(path, index=False)


def _eicu_pressures(path, database, layout="flat"):
    stays = range(1, 21)
    pd.DataFrame({
        "patientunitstayid": stays, "patienthealthsystemstayid": stays,
        "uniquepid": [f"synthetic-{i}" for i in stays],
        "unitdischargeoffset": 24 * 60, "hospitaladmitoffset": 0,
    }).to_parquet(path / "patient.parquet", index=False)
    periodic, aperiodic, nursing = [], [], []
    labels = [
        ("Invasive BP Mean", "", 100.),
        ("", "Arterial Line MAP (mmHg)", 110.),
        ("Non-Invasive BP Mean", "", 120.),
        ("Non-Invasive BP Systolic", "", 140.),
        ("Non-Invasive BP Diastolic", "", 80.),
        ("", "CVP (mmHg)", 40.),
        ("", "CVP", 45.),
    ]
    for stay in stays:
        delta = stay % 5
        # Hour 0 distinguishes raw pooling from a median of source medians.
        # Hours 1/2 distinguish pre-aggregation upper/lower bounds.
        # Hour 3 has cuff/nursing evidence only, exercising the actual SBP/DBP
        # fallback channel instead of assuming MAP's impact applies to them.
        for hour, n in [(0, 12), (1, 2), (2, 2), (3, 12)]:
            for point in range(n):
                offset = hour * 60 + point + 1
                invalid = point == 1 and hour in {1, 2}
                values = [60. + delta, 90. + delta, 40. + delta, 10. + delta]
                if invalid:
                    values = [1000.] * 4 if hour == 1 else [-100.] * 4
                periodic.append((stay, offset, *(values if hour != 3 else [np.nan] * 3 + [values[3]])))
                aperiodic.append((stay, offset, *values[:3]))
            if hour in {0, 3}:
                for name, label, value in labels:
                    if hour == 3 and (name == "Invasive BP Mean" or label.startswith("Arterial")):
                        continue
                    nursing.append((stay, hour * 60 + 30, name, label, str(value)))
    for name, rows, columns in [
        ("vitalperiodic", periodic, ["patientunitstayid", "observationoffset", "systemicmean", "systemicsystolic", "systemicdiastolic", "cvp"]),
        ("vitalaperiodic", aperiodic, ["patientunitstayid", "observationoffset", "noninvasivemean", "noninvasivesystolic", "noninvasivediastolic"]),
        ("nursecharting", nursing, ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevalname", "nursingchartcelltypevallabel", "nursingchartvalue"]),
    ]:
        frame = pd.DataFrame(rows, columns=columns)
        for column in load_src_cfg(database).get_table(name).columns:
            if column not in frame:
                frame[column] = np.nan
        _write_table(path, database, name, frame, layout)
    return ICUDataSource(load_src_cfg(database), base_path=path, enable_cache=False)


def _load_pressure(dictionary, source, name):
    table = ConceptResolver(dictionary)._load_single_concept(
        name, source, aggregator="median", patient_ids=list(range(1, 21)),
        interval=pd.Timedelta(hours=1), verbose=False,
    )
    return table.data.sort_values(["patientunitstayid", table.index_column]).reset_index(drop=True)


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
@pytest.mark.parametrize("name", PRESSURES)
@pytest.mark.parametrize("single_source", [False, True])
def test_r6_f006_wide_and_ordinary_paths_match_raw_pooling_and_bounds(
    tmp_path, monkeypatch, database, name, single_source,
):
    source = _eicu_pressures(tmp_path, database)
    from easyicu import datasource

    connection = datasource._get_duckdb_connection()
    queries = []

    class RecordingConnection:
        def __getattr__(self, name):
            return getattr(connection, name)

        def execute(self, query, *args, **kwargs):
            queries.append(query)
            return connection.execute(query, *args, **kwargs)

    monkeypatch.setattr(datasource, "_get_duckdb_connection", RecordingConnection)
    dictionary = load_dictionary(source_name=database)
    if single_source:
        definition = dictionary[name]
        dictionary = ConceptDictionary({name: replace(
            definition, sources={database: [definition.sources[database][0]]},
        )})
    fast = _load_pressure(dictionary, source, name)
    wide_queries = [q for q in queries if "GROUP BY patientunitstayid, FLOOR(observationoffset / 60.0)" in q]
    assert bool(wide_queries) == single_source
    with monkeypatch.context() as ordinary:
        # load_table uses the private storage resolvers; disabling their public
        # optimization seam forces the ordinary path over the identical files.
        ordinary.setattr(source, "resolve_bucket_directory", lambda _: None)
        ordinary.setattr(source, "resolve_flat_parquet_directory", lambda _: None)
        raw = _load_pressure(dictionary, source, name)
    pd.testing.assert_frame_equal(fast, raw)
    expected = {"map": 60., "map_invasive": 60., "map_nibp": 60.,
                "sbp": 90., "dbp": 40., "cvp_raw": 10.}[name]
    hours = [0., 1., 2.] if name == "map_invasive" or (single_source and name in {"map", "sbp", "dbp"}) else [0., 1., 2., 3.]
    time = next(c for c in fast.columns if c not in {"patientunitstayid", name})
    # Independent raw replay for every one of the 20 invented stays, then
    # equality of the whole result above. Every valid hour has the same median.
    for stay in range(1, 21):
        assert set(fast.loc[fast.patientunitstayid.eq(stay), name]) == {expected + stay % 5}
        assert fast.loc[fast.patientunitstayid.eq(stay), time].tolist() == hours


@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "eicu_demo", "aumc", "hirid"])
@pytest.mark.parametrize("clock", ["numeric", "datetime"])
def test_r6_f008_cvp_keeps_30_to_50_with_flags_and_bounds(database, clock):
    values = [30., 30.1, 50., 50.1, -5., -5.1, 20., 45., 20., 40., 40.]
    # Patient two must not borrow patient one's low neighbours for spike QC.
    data = pd.DataFrame({"stay_id": [1] * len(values) + [2],
                         "charttime": list(range(len(values))) + [7],
                         "cvp_raw": values + [45.], "dbp": [80.] * 7 + [46., 80., 42., np.nan, 80.]})
    if clock == "datetime":
        data.charttime = pd.Timestamp("2100-01-01") + pd.to_timedelta(data.charttime, unit="h")
    table = ICUTable(data, id_columns=["stay_id"], index_column="charttime", value_column="cvp_raw")
    dbp = ICUTable(data[["stay_id", "charttime", "dbp"]], id_columns=["stay_id"],
                   index_column="charttime", value_column="dbp")
    tables = {"cvp_raw": table, "dbp": dbp}
    ds = SimpleNamespace(config=SimpleNamespace(name=database))
    ctx = ConceptCallbackContext(concept_name="cvp", target="ts_tbl", interval=pd.Timedelta(hours=1),
                                 resolver=None, data_source=ds, patient_ids=None)
    main = _callback_cvp_artefact_filter(tables, ctx).data
    assert main.cvp.tolist() == [v for v in values + [45.] if -5 <= v <= 50]
    ctx.concept_name = "cvp_qc"
    qc = _callback_cvp_artefact_filter(tables, ctx).data
    flags = qc.cvp_qc.tolist()
    assert "above_50" in flags[3] and "below_minus_5" in flags[5]
    assert all("high_30_to_50" in flags[i] for i in [0, 1, 2, 7, 9, 10, 11])
    assert "isolated_spike" in flags[7]
    assert "arterial_diastolic_match" in flags[7]
    assert "arterial_diastolic_match" in flags[9]
    assert "isolated_spike" not in flags[11]
    assert table.data.cvp_raw.tolist() == values + [45.]
    dictionary = load_dictionary(source_name=database)
    assert dictionary["cvp_qc"].class_name == "chr_cncpt"
    assert dictionary["cvp_qc"].sub_concepts == dictionary["cvp"].sub_concepts
    assert dictionary["cvp_qc"].callback == dictionary["cvp"].callback
    from easyicu.concept.catalog import CONCEPT_DB_COVERAGE

    assert CONCEPT_DB_COVERAGE["cvp_qc"] == 6


def test_r6_f008_sic_qc_and_map_minus_cvp_share_retained_values(tmp_path):
    pd.DataFrame({"CaseID": [1], "PatientID": [11], "ICUOffset": [1800],
                  "TimeOfStay": [24 * 3600], "HospitalUnit": [3]}).to_parquet(tmp_path / "cases.parquet")
    rows = [(1, 2018, h, v) for h, v in enumerate([20., 45., 20., 40., 50., 50.1, 30.])]
    rows += [(1, 702, 3, 42.)] + [(1, 703, h, 80.) for h in range(7)]
    frame = pd.DataFrame(rows, columns=["CaseID", "DataID", "Offset", "Val"])
    frame.Offset = frame.Offset * 3600 + 1800
    frame.to_parquet(tmp_path / "data_float_h.parquet")
    source = ICUDataSource(load_src_cfg("sic"), base_path=tmp_path, enable_cache=False)
    results = ConceptResolver(load_dictionary(source_name="sic")).load_concepts(
        ["map", "cvp", "cvp_raw", "cvp_qc"], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"CaseID": [1]}, concept_workers=1, verbose=False,
    )
    main = results["cvp"].data.set_index("Offset").cvp
    assert main.to_dict() == {0.: 20., 1.: 45., 2.: 20., 3.: 40., 4.: 50., 6.: 30.}
    qc = results["cvp_qc"].data.set_index("Offset").cvp_qc
    assert "isolated_spike" in qc.loc[1.]
    assert "arterial_diastolic_match" in qc.loc[3.]
    assert "arterial_evidence_unavailable" in qc.loc[4.]
    assert "high_30_to_50" in qc.loc[6.]
    assert qc.loc[5.] == "above_50"
    perfusion = results["map"].data.set_index("Offset")["map"] - main
    assert perfusion.loc[[1., 3., 4., 6.]].tolist() == [35., 40., 30., 50.]
    assert pd.isna(perfusion.loc[5.])


def _native_vitals(root, database, names, identifier, patients, streamed):
    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader

    output = root / "native"
    output.mkdir()
    try:
        extraction._run_module_extraction(
            module_name="vitals", concepts=names, database=database, data_path=str(root),
            patient_ids_filter={identifier: patients}, batch_size=20, output_dir=str(output),
            use_sofa2=False, stream_output_batches=streamed,
        )
        manifest = json.loads((output / "_manifest.json").read_text())
        assert manifest["errors"] == []
        (output / "_manifest.json").rename(output / "vitals.manifest.json")
        extraction._publish_native_export_v2(
            database=database, data_path=str(root), output_dir=str(output), modules=["vitals"],
            max_patients=None, result={"modules": {"vitals": manifest}},
            concept_projection={"vitals": names},
        )
        return pd.read_parquet(output / "vitals.parquet")
    finally:
        clear_global_loader()


@pytest.mark.parametrize("database", ["eicu", "eicu_demo"])
@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_r6_f006_batch_and_native_vitals_match_each_pressure_channel(tmp_path, database, layout):
    source = _eicu_pressures(tmp_path, database, layout)
    names = PRESSURES + ["cvp", "cvp_qc"]
    tables = ConceptResolver(load_dictionary(source_name=database)).load_concepts(
        names, source, merge=False, r_compatible=False, concept_workers=1, verbose=False,
        patient_ids=list(range(1, 21)), interval=pd.Timedelta(hours=1),
    )
    published = _native_vitals(tmp_path, database, names, "patientunitstayid", list(range(1, 21)), layout == "bucket")
    for name, value in zip(PRESSURES + ["cvp"], [60., 60., 60., 90., 40., 10., 10.]):
        frame = tables[name].data
        assert frame[name].tolist() == (value + frame.patientunitstayid % 5).tolist()
        actual = published[name].dropna()
        assert actual.tolist() == (value + published.loc[actual.index, "stay_id"] % 5).tolist()
        assert len(actual) == len(frame) == (60 if name == "map_invasive" else 80)
    assert set(published.cvp_qc.dropna()) == {"kept"}


@pytest.mark.parametrize("entry", ["single", "batch"])
@pytest.mark.parametrize("aggregate,interval", [
    (False, pd.Timedelta(hours=1)), ("median", None),
    ("median", pd.Timedelta(minutes=30)), ("median", pd.Timedelta(hours=2)),
])
def test_r6_f006_wide_path_respects_raw_requests_and_requested_grid(tmp_path, monkeypatch, entry, aggregate, interval):
    source = _eicu_pressures(tmp_path, "eicu")
    periodic = tmp_path / "vitalperiodic/part.parquet"
    frame = pd.read_parquet(periodic).iloc[:7].copy()
    frame["observationoffset"] = [1, 2, 3, 4, 5, 31, 61]
    frame["cvp"] = [10., 10., 10., 10., 10., 40., 40.]
    frame.to_parquet(periodic, index=False)
    definition = load_dictionary(source_name="eicu")["cvp_raw"]
    dictionary = ConceptDictionary({"cvp_raw": replace(
        definition, sources={"eicu": definition.sources["eicu"][:1]},
    )})

    def load():
        resolver = ConceptResolver(dictionary)
        if entry == "single":
            table = resolver._load_single_concept(
                "cvp_raw", source, patient_ids=[1], aggregator=aggregate, interval=interval, verbose=False,
            )
        else:
            table = resolver.load_concepts(
                ["cvp_raw"], source, patient_ids=[1], aggregate=aggregate, interval=interval,
                merge=False, r_compatible=False, concept_workers=1, verbose=False,
            )["cvp_raw"]
        frame = table.data.rename(columns={table.index_column: "time"})
        return frame[["patientunitstayid", "time", "cvp_raw"]].sort_values("time").reset_index(drop=True)

    fast = load()
    monkeypatch.setattr(source, "resolve_bucket_directory", lambda _: None)
    monkeypatch.setattr(source, "resolve_flat_parquet_directory", lambda _: None)
    # Batch-wide scanning is a separate optimization from the per-source seam.
    from easyicu import datasource

    def force_ordinary(*args, **kwargs):
        raise NotImplementedError("force ordinary loading for comparison")

    monkeypatch.setattr(datasource, "load_wide_table_aggregated", force_ordinary)
    ordinary = load()
    pd.testing.assert_frame_equal(fast, ordinary, check_dtype=False)
    if aggregate is False or (interval is None and entry == "single"):
        assert len(ordinary) == 7


def _cvp_source(root, database, layout):
    """Invented values on each database's declared identifier and clock."""
    values = [30., 30.1, 50., 50.1, 20., 45., 20., 40., 0., -5., -5.1, np.nan]
    start = pd.Timestamp("2100-01-01 00:30")
    hours = np.arange(len(values))
    times = start + pd.to_timedelta(hours, unit="h")
    cfg = load_src_cfg(database)
    identifier = cfg.id_configs["icustay"].id
    if database in {"miiv", "mimic"}:
        anchor = pd.DataFrame({identifier: [1], "subject_id": [1], "hadm_id": [11],
                               "intime": [start], "outtime": [start + pd.Timedelta(days=1)], "los": [1.]})
    elif database == "eicu":
        anchor = pd.DataFrame({identifier: [1], "patienthealthsystemstayid": [11], "uniquepid": ["invented"],
                               "hospitaladmitoffset": [-30], "unitdischargeoffset": [1440]})
    elif database == "aumc":
        anchor = pd.DataFrame({identifier: [1], "patientid": [11], "admittedat": [1800000], "dischargedat": [88200000]})
    elif database == "hirid":
        anchor = pd.DataFrame({identifier: [1], "admissiontime": [str(start)], "age": [60], "sex": ["M"]})
    else:
        anchor = pd.DataFrame({identifier: [1], "PatientID": [11], "ICUOffset": [1800], "TimeOfStay": [88200]})
    _write_table(root, database, cfg.id_configs["icustay"].table, anchor)
    rows = []
    dictionary = load_dictionary(source_name=database)
    for name, observations in [("cvp_raw", values), ("map", [80.] * len(values)),
                               ("dbp", [80.] * 5 + [46., 80., 42., 80., 80., 80., np.nan])]:
        source = dictionary[name].for_data_source(cfg)[0]
        defaults = cfg.get_table(source.table).defaults
        time = source.index_var or defaults.index_var
        value = source.value_var or defaults.val_var
        frame = pd.DataFrame({identifier: 1, value: observations})
        if source.ids:
            frame[source.sub_var] = source.ids[0]
        if database in {"miiv", "mimic"}:
            frame[time] = times
            frame = frame.assign(subject_id=1, hadm_id=11, valueuom="mmHg")
            if database == "mimic":
                frame["error"] = 0
        elif database == "eicu":
            frame[time] = hours * 60
        elif database == "aumc":
            frame[time] = hours * 3600000 + 1800000
            frame = frame.assign(unit="mmHg", tag="+")
        elif database == "hirid":
            frame[time] = times
            frame = frame.assign(entertime=times, status=8, type="N", stringvalue=None)
        else:
            frame[time] = hours * 3600 + 1800
        rows.append(frame)
    if database == "eicu":
        # The three observations are columns of the same real wide table.
        frame = rows[0].merge(rows[1], on=[identifier, time]).merge(rows[2], on=[identifier, time])
    else:
        frame = pd.concat(rows, ignore_index=True)
    _write_table(root, database, source.table, frame, layout)
    return ICUDataSource(cfg, base_path=root, enable_cache=False), identifier


@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "aumc", "hirid", "sic"])
@pytest.mark.parametrize("layout", ["file", "bucket"])
def test_r6_f008_real_columns_keep_cvp_flags_in_single_batch_and_native(tmp_path, database, layout):
    source, identifier = _cvp_source(tmp_path, database, layout)
    dictionary = load_dictionary(source_name=database)
    names = ["map", "cvp", "cvp_raw", "cvp_qc"]
    options = dict(merge=False, r_compatible=False, concept_workers=1, verbose=False,
                   patient_ids={identifier: [1]}, interval=pd.Timedelta(hours=1))
    batch = ConceptResolver(dictionary).load_concepts(names, source, **options)
    for name in ["cvp", "cvp_qc"]:
        single = ConceptResolver(dictionary).load_concepts([name], source, **options)[name]
        pd.testing.assert_frame_equal(single.data, batch[name].data)
    published = _native_vitals(tmp_path, database, names, identifier, [1], layout == "bucket").set_index("charttime")
    expected = {0.: 30., 1.: 30.1, 2.: 50., 4.: 20., 5.: 45., 6.: 20., 7.: 40., 8.: 0., 9.: -5.}
    assert published.cvp.dropna().to_dict() == pytest.approx(expected)
    assert (published["map"] - published.cvp).dropna().to_dict() == pytest.approx({h: 80. - v for h, v in expected.items()})
    for hour in [0., 1., 2., 5., 7.]:
        assert "high_30_to_50" in published.loc[hour, "cvp_qc"]
    assert "arterial_diastolic_match" in published.loc[5., "cvp_qc"]
    assert "isolated_spike" in published.loc[5., "cvp_qc"]
    assert "arterial_diastolic_match" in published.loc[7., "cvp_qc"]
    assert published.loc[8., "cvp_qc"] == "kept"  # A measured zero is not missing.
    assert pd.isna(published.loc[11., "cvp_qc"])
