"""Round-5b residues: synthetic urine collection and ventilation evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import callbacks
from easyicu.resources import load_dictionary
from easyicu.table import ICUTable


DATABASES = ("miiv", "mimic", "eicu", "aumc", "sic", "hirid")


def _table(name, times, values, id_col="stay_id", time_col="charttime"):
    return ICUTable(
        pd.DataFrame({id_col: 1, time_col: times, name: values}),
        id_columns=[id_col], index_column=time_col, value_column=name,
    )


def _context(database, name, resolver=None):
    return callbacks.ConceptCallbackContext(
        name, None, pd.Timedelta(hours=1), resolver,
        SimpleNamespace(config=SimpleNamespace(name=database)), None,
    )


@pytest.mark.parametrize("database", DATABASES)
@pytest.mark.parametrize("gap_axis", ["chart", "source"])
def test_ru_01_assessment_integrates_collection_overlap_and_breaks_gaps(database, gap_axis):
    from easyicu.scores.urine_windows import assess_urine_windows

    rate = database == "hirid"
    boundary = 25. if gap_axis == "chart" else 24.
    urine = _table("urine", [0., 4., 8., 12., boundary, boundary + 4.],
                   [0., 20., 0., 0., 0., 0.] if rate else [0., 80., 0., 0., 0., 0.])
    urine.data["_urine_available_at"] = urine.data.charttime + .25
    if gap_axis == "source":
        urine.data.loc[urine.data.charttime.ge(boundary), "_urine_available_at"] += .5
    urine.data["_urine_gap_h"] = np.r_[0., np.diff(urine.data._urine_available_at)]
    weight = pd.DataFrame({"stay_id": [1], "weight": [10.]})
    result = assess_urine_windows(
        urine.data, weight, id_columns=["stay_id"], time_column="charttime",
        interval=pd.Timedelta(hours=1), source_is_rate=rate,
    ).set_index("charttime")
    assert result.loc[8., "uo_6h_covered_h"] == 6.
    assert result.loc[8., "uo_6h_assessment_rate"] == pytest.approx(40 / 6 / 10)
    assert result.loc[12., "uo_6h_oliguria_gt6h"]
    assert result.loc[boundary, "uo_24h_covered_h"] == 0.
    assert result.loc[boundary + 4., "uo_24h_covered_h"] == 4.
    assert pd.isna(result.loc[boundary + 4., "uo_6h_assessment_rate"])
    # The callbacks used by SOFA-2 and RRT see exactly the same evidence.
    urine.data["urine"] = 0.
    tables = {"urine": urine, "weight": ICUTable(weight, id_columns=["stay_id"], value_column="weight")}
    uo = callbacks._callback_uo_6h(tables, _context(database, "uo_6h"))
    assert uo.data.loc[uo.data.charttime.eq(8.), "uo_6h_assessment_rate"].item() == 0.
    assert uo.data.loc[uo.data.charttime.eq(8.), "uo_6h_oliguria_gt6h"].item()


@pytest.mark.parametrize("database", DATABASES)
def test_ru_02_descriptive_and_assessment_export_source_availability(database, tmp_path):
    urine = _table("urine", np.arange(25, dtype=float), [0.] * 25)
    urine.data["_urine_available_at"] = urine.data.charttime + .75
    weight = ICUTable(pd.DataFrame({"stay_id": [1], "weight": [70.]}),
                      id_columns=["stay_id"], value_column="weight")
    for window in (6, 12, 24):
        name = f"uo_{window}h"
        out = callbacks._callback_uo_window(
            {"urine": urine, "weight": weight}, _context(database, name), window, name,
        ).data
        assert out.iloc[-1][f"{name}_available_at"] == 24.75
        assert out.iloc[-1][f"{name}_assessment_available_at"] == 24.75
        if window == 6:
            assert out.iloc[-1].uo_6h_oliguria_available_at == 24.75
    from easyicu.api import concepts as api
    from tests.core.test_extraction_audit_round5_u5 import _clock_source

    root = tmp_path / database
    _clock_source(root, database)
    try:
        loaded = api.load_concepts(
            ["uo_6h", "uo_12h", "uo_24h"], database=database, data_path=root,
            merge=False, verbose=False, concept_workers=1, parallel_workers=1,
        )
        for name, result in loaded.items():
            frame = result if isinstance(result, pd.DataFrame) else result.data
            assert not frame.empty
            assert frame[f"{name}_available_at"].notna().any()
            assert frame[f"{name}_assessment_available_at"].notna().any()
        merged = api.load_concepts(
            list(loaded), database=database, data_path=root,
            merge=True, verbose=False, concept_workers=1, parallel_workers=1,
        )
        frame = merged if isinstance(merged, pd.DataFrame) else merged.data
        for name in loaded:
            assert frame[f"{name}_available_at"].notna().any()
            assert frame[f"{name}_assessment_available_at"].notna().any()
        assert frame.uo_6h_oliguria_available_at.notna().any()
        from easyicu.api.extraction import _canonicalise_native_export_frame

        native = _canonicalise_native_export_frame(
            frame, module="renal", requested_concepts=list(loaded),
            dictionary=load_dictionary(), database=database,
        )
        companions = [f"{name}{suffix}" for name in loaded for suffix in ("_available_at", "_assessment_available_at")]
        for column in [*companions, "uo_6h_oliguria_available_at"]:
            pd.testing.assert_series_equal(native[column], frame[column].astype("float64"))
    finally:
        api.clear_global_loader()


def test_ru_03_eicu_entry_time_is_loaded_on_both_source_paths(tmp_path):
    for layout in ("raw", "bucketed"):
        _check_source_availability(tmp_path / layout, "eicu", layout)


@pytest.mark.parametrize("legacy", ["callbacks.urine24", "rate", "missing.rrt_criteria", "clinical.urine24"])
def test_ru_04_legacy_urine_apis_cannot_compute_divergent_rules(legacy):
    from easyicu import callbacks as legacy_callbacks, callbacks_missing
    from easyicu.utils import clinical_utils

    if legacy == "rate":
        urine = pd.DataFrame({"stay_id": [1, 1, 1], "charttime": [0., 6., 19.], "urine": [0., 7., 7.]})
        weight = pd.DataFrame({"stay_id": [1], "weight": [70.]})
        result = legacy_callbacks.urine_rate_window_avg_multi(urine, weight, [(6, 3)])["uo_6h"]
        assert result.iloc[1].uo_6h == pytest.approx(.1)
        assert pd.isna(result.iloc[-1].uo_6h)
        assert result.iloc[-1].uo_6h_covered_h == 0.
        return
    call = {
        "callbacks.urine24": lambda: legacy_callbacks.urine24(pd.DataFrame({"urine": [1.]})),
        "missing.rrt_criteria": lambda: callbacks_missing.rrt_criteria(pd.Series([1.])),
        "clinical.urine24": lambda: clinical_utils.urine24(100., 12.),
    }[legacy]
    with pytest.raises(NotImplementedError, match="deprecated.*(?:urine_output|rrt_criteria)"):
        call()


@pytest.mark.parametrize("airway", ["Oral ETT", "Nasal ETT", "Double-Lumen Tube", "Cricothyrotomy", "NIV", "Unknown", "", None])
def test_ru_05_care_intervals_require_invasive_airway_and_half_open_hours(airway):
    from easyicu.concept.eicu_ventilation_windows import bridge_hours, care_intervals, load_care_intervals

    care = pd.DataFrame({
        "patientunitstayid": [1], "airwaytype": [airway],
        "ventstartoffset": [60], "ventendoffset": [120],
        "priorventstartoffset": [-30], "priorventendoffset": [30],
    })
    requested = []

    def load(table, columns, **kwargs):
        if table == "patient":
            return pd.DataFrame({"patientunitstayid": [1], "unitdischargeoffset": [180]})[columns]
        requested.extend(columns)
        return care[columns]

    direct = care_intervals(care)
    loaded = load_care_intervals(SimpleNamespace(load_table=load), [1])
    pd.testing.assert_frame_equal(direct, loaded)
    assert "airwaytype" in requested
    points = pd.DataFrame(columns=["patientunitstayid", "hour"])
    hours = bridge_hours(points, loaded)
    invasive = airway in {"Oral ETT", "Nasal ETT", "Double-Lumen Tube", "Cricothyrotomy"}
    assert hours.hour.tolist() == ([0, 1] if invasive else [])
    assert care_intervals(care.drop(columns="airwaytype")).empty


@pytest.mark.parametrize("code", [12, 13, 14])
def test_ru_06_hirid_niv_mode_blocks_invasive_bridging(code, tmp_path):
    from easyicu.concept.callbacks import _hirid_bridge_ventilator_evidence

    loaded = {n: _table(n, [5.], [8.], "patientid", "datetime") for n in ("peep", "etco2", "tidal_vol")}
    loaded["vent_mode"] = _table("vent_mode", [2.5, 6.], ["pressure", "pressure"], "patientid", "datetime")
    loaded["vent_mode"].data["vent_mode_interface"] = ["niv", "invasive"]
    resolver = SimpleNamespace(load_concepts=lambda *a, **kw: loaded)
    airway = _table("vent_ind", [0.], [True], "patientid", "datetime")
    ctx = _context("hirid", "vent_ind", resolver)
    out = _hirid_bridge_ventilator_evidence(airway, ctx, "patientid", "datetime")
    assert out.data.datetime.tolist() == [0.]
    assert out.data.patientid.dtype == airway.data.patientid.dtype
    loaded["vent_mode"].data["vent_mode_interface"] = "invasive"
    unopposed = _hirid_bridge_ventilator_evidence(airway, ctx, "patientid", "datetime")
    assert unopposed.data.datetime.tolist() == [0., 1., 2., 3., 4., 5.]
    assert unopposed.data.patientid.dtype == airway.data.patientid.dtype
    # Public native mapping preserves the control axis and the NIV interface.
    from easyicu.concept import ConceptSource, _apply_callback

    raw = pd.DataFrame({"patientid": [1], "datetime": [2.5], "value": [code]})
    mapped = _apply_callback(raw, concept_name="vent_mode",
                             source=ConceptSource(table="observations", value_var="value", callback="vent_mode_control"),
                             data_source=ctx.data_source, interval=pd.Timedelta(minutes=1))
    assert mapped.vent_mode.tolist() == ["pressure"]
    assert mapped.vent_mode_interface.tolist() == ["niv"]
    from easyicu.api import concepts as api
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=3)
    path = root / "observations" / "1.parquet"
    frame = pd.concat([pd.read_parquet(path).iloc[[0]]] * 5, ignore_index=True)
    frame["patientid"] = 1
    frame["variableid"] = [15001552, 3845, 3845, 2600, 2200]
    frame["value"] = [1., float(code), 2., 8., 35.]
    origin = pd.read_parquet(root / "general.parquet").set_index("patientid").loc[1, "admissiontime"]
    frame["datetime"] = pd.Timestamp(origin) + pd.to_timedelta([0., 2.5, 6., 5., 5.], unit="h")
    frame.to_parquet(path, index=False)
    try:
        loaded = api.load_concepts(
            ["vent_ind", "vent_mode"], database="hirid", data_path=root,
            merge=False, verbose=False, concept_workers=1, parallel_workers=1,
        )
        loaded = {name: result if isinstance(result, pd.DataFrame) else result.data for name, result in loaded.items()}
        assert loaded["vent_ind"].charttime.tolist() == [0., 1., 2., 3.]
        assert loaded["vent_mode"].vent_mode.eq("pressure").all()
    finally:
        api.clear_global_loader()


def test_ru_07_sofa1_documents_extrapolation_and_preserves_scores():
    from easyicu.callbacks import sofa_renal

    scores = sofa_renal(pd.Series([1., 1., 1.]), pd.Series([600., np.nan, np.nan]),
                        pd.Series([100., 100., np.nan]))
    assert scores.tolist() == [0, 4, 0]
    description = load_dictionary().get("sofa_renal").description
    assert "extrapolat" in description
    assert "18 to less than 24 covered hours" in description


# These are actual schema fields, not fixture-only aliases. The parser at the
# start of the fixture checks both the source table and every availability clock.
SOURCE_CLOCKS = {
    "miiv": ("outputevents", "itemid", 226559, "charttime", None, ("storetime",)),
    "mimic": ("outputevents", "itemid", 226559, "charttime", None, ("storetime",)),
    "eicu": ("intakeoutput", "celllabel", "Urine", "intakeoutputoffset", 60, ("intakeoutputentryoffset",)),
    "aumc": ("numericitems", "itemid", 8794, "measuredat", 3600000, ("registeredat", "updatedat")),
    "hirid": ("observations", "variableid", 10020000, "datetime", None, ("entertime",)),
    "sic": ("data_float_h", "DataID", 725, "Offset", 3600, ()),
}


def _availability_source(root, database, layout, *, gaps=False):
    from tests.core.test_extraction_audit_round5_u5 import _clock_source

    schema = json.loads((Path(callbacks.__file__).parents[1] / "data/data-sources.json").read_text())
    table, selector, code, clock, factor, availability = SOURCE_CLOCKS[database]
    declared = next(d for d in schema if d["name"] == database)["tables"][table]["cols"]
    assert {selector.lower(), clock.lower(), *availability} <= set(declared)
    _clock_source(root, database)
    paths = sorted(set(root.glob(f"{table}*/**/*.parquet")) | set(root.glob(f"{table}.parquet")))
    frames = [pd.read_parquet(path) for path in paths]
    all_rows = pd.concat(frames, ignore_index=True)
    sample = all_rows.loc[all_rows[selector].eq(code)].sort_values(clock).iloc[:1].copy()
    assert not sample.empty
    origin = sample[clock].iloc[0] - (pd.Timedelta(hours=.25) if factor is None else .25 * factor)
    hours = np.array([0., 4., 8., 12., 16., 20., 24., 40., 44.] if gaps else list(np.arange(0., 49., 4.)))
    stamps = hours + .25
    rows = pd.concat([sample] * len(hours), ignore_index=True)
    rows[clock] = origin + (pd.to_timedelta(stamps, unit="h") if factor is None else stamps * factor)
    value = next(d for d in schema if d["name"] == database)["tables"][table]["defaults"]["val_var"]
    rows[value] = (7. if database == "hirid" else 28.)
    if gaps:
        # Nonzero interval clipping at 8 h, complete 12/24 h, then a >12 h gap.
        rows[value] = [0., 240., 0., 0., 0., 0., 0., 0., 0.]
    for index, column in enumerate(availability):
        available = stamps.copy()
        available[hours == 20.] = 40.25 if index == 0 and len(availability) > 1 else 44.25
        available[hours == 24.] = 25.25
        available[hours == 4.] = 3.25  # entry before chart must not lower availability
        available[hours == 8.] = np.nan  # absent per-row entry falls back to chart
        rows[column] = origin + (pd.to_timedelta(available, unit="h") if factor is None else available * factor)
    # Remove original urine for every synthetic stay; preserve adjacent concepts.
    combined = pd.concat([all_rows.loc[~all_rows[selector].eq(code)], rows], ignore_index=True)
    for path in paths:
        path.unlink()
    for directory in sorted({p.parent for p in paths if p.parent != root}, key=lambda p: len(p.parts), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    target = root / f"{table}.parquet"
    if layout != "raw":
        target = root / f"{table}_bucket/bucket_id=0/part.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(target, index=False)
    return 44.25 if availability else 24.25


def _check_source_availability(root, database, layout):
    from easyicu.api import concepts as api
    from easyicu.scores.kdigo_aki import detect_time_col

    expected = _availability_source(root, database, layout)
    options = dict(database=database, data_path=root, verbose=False, concept_workers=1,
                   parallel_workers=1, r_compatible=False)
    names = ["urine24", "urine24_est", "uo_6h", "uo_12h", "uo_24h", "kdigo_uo"]
    try:
        # Explicit raw concepts also exercise the multi-concept batch planner.
        tables = api.load_concepts(["urine", "spo2", *names], merge=False, **options)
        for name in names:
            data = tables[name].data
            row = data.loc[data[detect_time_col(data)].eq(24.)].iloc[0]
            if name == "urine24_est":
                row = data.loc[data[detect_time_col(data)].eq(20.)].iloc[0]
                expected_at = expected if database != "sic" else 20.25
            else:
                expected_at = expected
            column = "aki_stage_uo_available_at" if name == "kdigo_uo" else f"{name}_available_at"
            assert row[column] == pytest.approx(expected_at), (database, layout, name, row)
            if name.startswith("uo_"):
                assert row[f"{name}_assessment_available_at"] == pytest.approx(expected_at)
                assert row[f"{name}_covered_h"] == int(name[3:-1])
        urine = tables["urine"].data
        time = detect_time_col(urine)
        assert urine.loc[urine[time].eq(4.), "_urine_available_at"].item() == 4.25
        assert urine.loc[urine[time].eq(8.), "_urine_available_at"].item() == 8.25
        # Delay metadata must not change coverage or the retrospective values.
        assert tables["uo_6h"].data.loc[lambda d: d[detect_time_col(d)].eq(24.), "uo_6h"].notna().all()
        # A merged public request for the value alone must carry its companion.
        for name in ("urine24", "urine24_est", "uo_6h", "kdigo_uo"):
            frame = api.load_concepts([name], merge=True, **{**options, "r_compatible": True})
            column = "aki_stage_uo_available_at" if name == "kdigo_uo" else f"{name}_available_at"
            assert column in frame.columns
    finally:
        api.clear_global_loader()


@pytest.mark.parametrize("database", DATABASES)
@pytest.mark.parametrize("layout", ["raw", "bucketed"])
def test_ru_08_all_source_clocks_survive_loading_and_window_exports(database, layout, tmp_path):
    _check_source_availability(tmp_path / database, database, layout)


@pytest.mark.parametrize("database", DATABASES)
def test_ru_08_nonmonotone_availability_uses_all_contributing_intervals(database):
    from easyicu.scores.kdigo_aki import kdigo_uo
    from easyicu.scores.urine_output import collection_windows

    times = np.arange(49, dtype=float)
    source = times.copy()
    source[20] = 50.
    urine = _table("urine", times, [7.] * len(times))
    urine.data["_urine_available_at"] = source
    urine.data["_urine_gap_h"] = 1.
    weight = _table("weight", [0.], [70.])
    for window in (6, 12, 24):
        name = f"uo_{window}h"
        frame = callbacks._callback_uo_window(
            {"urine": urine, "weight": weight}, _context(database, name), window, name,
        ).data.set_index("charttime")
        assert frame.loc[24., f"{name}_available_at"] == 50.
        assert frame.loc[24., f"{name}_assessment_available_at"] == 50.
        assert frame.loc[48., f"{name}_available_at"] == 48.
        if window == 6:
            assert frame.loc[24., "uo_6h_oliguria_available_at"] == 50.
    stages = kdigo_uo(urine.data, weight.data, time_unit="hours", source_is_rate=database == "hirid")
    assert stages.loc[stages.charttime.eq(24.), "aki_stage_uo_available_at"].item() == 50.
    daily = callbacks._callback_urine24({"urine": urine}, _context(database, "urine24")).data.set_index("charttime")
    assert daily.loc[24., "urine24_available_at"] == 50.
    assert daily.loc[20., "urine24_est_available_at"] == 50.
    # Even a left endpoint outside the window determines the first interval.
    _, _, available = collection_windows(np.array([0., 4., 8.]), np.array([0., 28., 28.]), [6],
                                         source_times=np.array([30., 4., 8.]), record_gaps=np.array([0., 4., 4.]))[6]
    assert available[-1] == 30.
    endpoint = _table("urine", np.arange(25.), [7.] * 25)
    endpoint.data["_urine_available_at"] = [50., *np.arange(1., 25.)]
    endpoint.data["_urine_gap_h"] = 1.
    daily = callbacks._callback_urine24({"urine": endpoint}, _context(database, "urine24")).data
    assert daily.loc[daily.charttime.eq(24.), "urine24_available_at"].item() == 50.


@pytest.mark.parametrize("layout", ["raw", "bucketed", "native"])
def test_ru_09_aumc_sofa2_renal_matches_shared_collection_windows(layout, tmp_path, monkeypatch):
    from easyicu.api import concepts as api
    from easyicu.scores.kdigo_aki import detect_time_col
    from easyicu.scores.urine_output import collection_windows

    root = tmp_path / "aumc"
    _availability_source(root, "aumc", layout, gaps=True)
    # Isolate urine scoring: the general synthetic source also generates RRT,
    # whose episode/LOCF evidence would correctly dominate the native score.
    for source in load_dictionary(include_sofa2=True).get("rrt").sources["aumc"]:
        for path in [*root.glob(f"{source.table}*/**/*.parquet"), *root.glob(f"{source.table}.parquet")]:
            data = pd.read_parquet(path)
            ids = source.ids if isinstance(source.ids, list) else [source.ids]
            data.loc[~data[source.sub_var].isin(ids)].to_parquet(path, index=False)
    options = dict(database="aumc", data_path=root, verbose=False,
                   concept_workers=1, parallel_workers=1, r_compatible=False)
    try:
        tables = api.load_concepts(["urine", "weight", "uo_6h", "uo_12h", "uo_24h", "sofa2_renal"], merge=False, **options)
        urine = tables["urine"].data
        time = detect_time_col(urine)
        windows = collection_windows(urine[time].to_numpy(), urine.urine.to_numpy(), (6, 12, 24),
                                     record_gaps=urine._urine_gap_h.to_numpy())
        weight = tables["weight"].data.weight.iloc[0]
        for window in (6, 12, 24):
            data = tables[f"uo_{window}h"].data
            volume, covered, _ = windows[window]
            np.testing.assert_allclose(data[f"uo_{window}h_assessment_covered_h"], covered)
            expected = np.where(covered >= window, volume / window / weight, np.nan)
            np.testing.assert_allclose(data[f"uo_{window}h_assessment_rate"], expected, equal_nan=True)
        scored = tables["sofa2_renal"].data
        row = scored.loc[scored[detect_time_col(scored)].eq(24.)].iloc[0]
        assert row.sofa2_renal == 3
        if layout == "native":
            import tempfile
            from easyicu.api.extraction import extract_database

            monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
            for name in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
                monkeypatch.delenv(name, raising=False)
            output = tmp_path / "out"
            result = extract_database("aumc", data_path=str(root), output_dir=str(output),
                                      modules=["outcome", "renal", "sofa2_score"], native_export_v2=True, verbose=False)
            assert not {m: r["errors"] for m, r in result["modules"].items() if r.get("errors")}
            native = pd.read_parquet(output / "sofa2_score.parquet")
            assert native.loc[native.charttime.eq(24.), "sofa2_renal"].eq(3).any()
            renal = pd.read_parquet(output / "renal.parquet")
            assert renal.loc[renal.charttime.eq(24.), "urine24_available_at"].eq(44.25).any()
    finally:
        api.clear_global_loader()


@pytest.mark.parametrize("database", DATABASES)
def test_ru_08_native_renal_export_preserves_true_availability(database, tmp_path, monkeypatch):
    import tempfile
    from easyicu.api import concepts as api
    from easyicu.api.extraction import extract_database

    root = tmp_path / database
    expected = _availability_source(root, database, "bucketed")
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for name in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "out"
    try:
        result = extract_database(database, data_path=str(root), output_dir=str(output),
                                  modules=["outcome", "renal"], native_export_v2=True, verbose=False)
        assert not {m: r["errors"] for m, r in result["modules"].items() if r.get("errors")}
        frame = pd.read_parquet(output / "renal.parquet")
        full = frame.loc[frame.charttime.eq(24.) & frame.urine24.notna()]
        assert len(full) == 1
        for column in ("urine24_available_at", "aki_stage_uo_reference_available_at",
                       "uo_6h_available_at", "uo_12h_available_at", "uo_24h_available_at"):
            assert full[column].item() == pytest.approx(expected)
    finally:
        api.clear_global_loader()


@pytest.mark.parametrize("compatible", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_ru_08_daily_companions_are_unique_when_estimate_is_requested(compatible, explicit, tmp_path):
    from easyicu.api import concepts as api

    root = tmp_path / "eicu"
    _availability_source(root, "eicu", "raw")
    names = ["urine24", "urine24_est"]
    if explicit:
        names += ["urine24_available_at", "urine24_est_available_at"]
    try:
        data = api.load_concepts(names, database="eicu", data_path=root, verbose=False,
                                 concept_workers=1, parallel_workers=1, r_compatible=compatible)
        assert not data.columns.duplicated().any()
        assert data.loc[data.charttime.eq(24.), "urine24_available_at"].item() == 44.25
        assert data.loc[data.charttime.eq(20.), "urine24_est_available_at"].item() == 44.25
    finally:
        api.clear_global_loader()


def test_ru_08_unassessable_long_window_does_not_delay_an_assessable_stage():
    from easyicu.scores.kdigo_aki import kdigo_uo

    urine = _table("urine", np.arange(9.), [7.] * 9).data
    urine["_urine_available_at"] = [100., *np.arange(1., 9.)]
    urine["_urine_gap_h"] = 1.
    out = kdigo_uo(urine, pd.DataFrame({"stay_id": [1], "weight": [70.]}), time_unit="hours")
    assert out.iloc[-1].aki_stage_uo_available_at == 8.


def test_ru_08_entry_delays_do_not_mask_a_true_chart_gap():
    from easyicu.scores.urine_output import collection_windows, source_availability

    # The bins are twelve hours apart, but the source chart gap is 12.8 h.
    # An entry at hour 40 must affect provenance without hiding that gap.
    frame = pd.DataFrame({"stay_id": [1, 1], "charttime": [0., 12.], "urine": [7., 84.],
                          "_urine_source_lag_h": [.1, .9], "_urine_first_lag_h": [.1, .9],
                          "_urine_availability_lag_h": [40., .9]})
    receipt = source_availability(frame, ["stay_id"], "charttime", pd.Timedelta(hours=1))
    assert receipt._urine_available_at.tolist() == [40., 12.9]
    assert receipt._urine_gap_h.iloc[-1] == pytest.approx(12.8)
    _, covered, _ = collection_windows(frame.charttime.to_numpy(), frame.urine.to_numpy(), [12],
                                       record_gaps=receipt._urine_gap_h.to_numpy())[12]
    assert covered[-1] == 0.


def test_ru_08_retrospective_and_source_clock_contract_is_documented():
    dictionary = json.loads((Path(callbacks.__file__).parents[1] / "data/concept-dict.json").read_text())
    for name in ("urine24", "urine24_est", "uo_6h", "uo_12h", "uo_24h", "kdigo_uo"):
        assert "retrospective" in dictionary[name]["description"].lower()
        assert "real-time use must filter availability time" in dictionary[name]["description"]
    note = dictionary["urine"]["_comment"]
    for field in ("intakeoutputentryoffset", "storetime", "entertime", "registeredat", "updatedat"):
        assert field in note
    assert "SICdb" in note and "no entry/store time" in note


@pytest.mark.parametrize("database", DATABASES)
def test_ru_08_unbinned_load_retains_entry_clock(database, tmp_path):
    from easyicu.api import concepts as api
    from easyicu.scores.kdigo_aki import detect_time_col

    root = tmp_path / database
    expected = _availability_source(root, database, "raw")
    try:
        tables = api.load_concepts(["urine", "uo_6h", "urine24"], database=database, data_path=root,
                                   interval=None, merge=False, r_compatible=False, verbose=False,
                                   concept_workers=1, parallel_workers=1)
        for name in ("urine", "uo_6h"):
            frame = tables[name].data
            time = detect_time_col(frame)
            row = frame.loc[frame[time].between(24., 24.25)].iloc[0]
            column = "_urine_available_at" if name == "urine" else "uo_6h_available_at"
            assert row[column] == (25.25 if name == "urine" and database != "sic" else expected)
        daily = tables["urine24"].data
        row = daily.loc[daily[detect_time_col(daily)].eq(24.)].iloc[0]
        assert row.urine24_available_at == expected
    finally:
        api.clear_global_loader()


@pytest.mark.parametrize("database", ["miiv", "mimic"])
@pytest.mark.parametrize("missing", [True, False])
def test_ru_08_optional_partition_clock_never_forces_unfiltered_fallback(database, missing, tmp_path, monkeypatch):
    from easyicu.concept import ConceptDictionary, ConceptResolver
    from tests.core.test_signed_urine_loading import fixture_source

    source, _, _, id_col = fixture_source(tmp_path, database)
    path = tmp_path / "outputevents/part-0.parquet"
    if not missing:
        frame = pd.read_parquet(path)
        frame["storetime"] = np.nan
        frame.to_parquet(path, index=False)
    monkeypatch.setattr(source, "resolve_bucket_directory", lambda table: None)
    monkeypatch.setattr(source, "resolve_flat_parquet_directory", lambda table: None)

    def reject_fallback(*args, **kwargs):
        pytest.fail("Missing entry clock must keep the filtered SQL read")

    monkeypatch.setattr(source, "_read_partitioned_data_optimized", reject_fallback)
    dictionary = Path(callbacks.__file__).parents[1] / "data/concept-dict.json"
    resolver = ConceptResolver(ConceptDictionary.from_json(dictionary))
    try:
        frame = resolver.load_concepts(["urine"], source, merge=False, r_compatible=False,
                                        interval=pd.Timedelta(hours=1), verbose=False,
                                        patient_ids={id_col: [1]}, concept_workers=1)["urine"].data
        assert frame.urine.tolist() == [150., 200., 0., 150.]
        assert frame._urine_available_at.notna().all()
    finally:
        resolver.clear_table_cache()
        source.clear_cache()
