"""Round-5 urine audit: synthetic collection intervals only, all six sources."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import callbacks
from easyicu.scores.kdigo_aki import kdigo_uo
from easyicu.table import ICUTable
from tests.support.urine_clock_source import urine_clock_source as _clock_source


DATABASES = ("miiv", "mimic", "eicu", "aumc", "sic", "hirid")


def _table(name, times, values):
    return ICUTable(
        pd.DataFrame({"stay_id": 1, "charttime": times, name: values}),
        id_columns=["stay_id"], index_column="charttime", value_column=name,
    )


def _context(database, name="urine24", **kwargs):
    return callbacks.ConceptCallbackContext(
        name, None, pd.Timedelta(hours=1), None,
        SimpleNamespace(config=SimpleNamespace(name=database)), None, kwargs=kwargs,
    )


def _daily(database, **kwargs):
    times = [0., 4., 8., 12., 16., 20.25, 24.]
    urine = _table("urine", times, [25.] + [25. if database == "hirid" else 100.] * 6)
    return callbacks._callback_urine24({"urine": urine}, _context(database, **kwargs)).data.set_index("charttime")


def _exact(database):
    times = [-12., 0., 6., 12., 24.]
    volumes = [0., 1200., 0., 0., 0.]
    if database == "hirid":
        volumes = [0., 100., 0., 0., 0.]
    return kdigo_uo(
        _table("urine", times, volumes).data,
        pd.DataFrame({"stay_id": [1], "weight": [50.]}),
        id_col="stay_id", time_col="charttime", time_unit="hours",
        source_is_rate=database == "hirid",
    ).iloc[-1]


def _assert_exact(database):
    row = _exact(database)
    for window in (6, 12, 24):
        assert row[f"uo_rt_{window}hr"] == 0
        assert row[f"uo_rt_{window}hr_covered_h"] == window
    assert row.aki_stage_uo == 3
    assert row.aki_stage_uo_available_at == 24


def _assert_split(database):
    data = _daily(database)
    partial = data.loc[20.25 if database == "hirid" else 18.]
    assert pd.isna(partial.urine24)
    # Event-grid values retain the R4 apportioning. Rates use actual clock time.
    assert partial.urine24_est == pytest.approx(600.)
    assert 18 <= partial.urine24_covered_h < 24
    assert partial.urine24_observed_ml == pytest.approx(25 * partial.urine24_covered_h)
    assert data.loc[24., "urine24"] == pytest.approx(600.)
    assert pd.isna(data.loc[24., "urine24_est"])


def _assert_availability(database):
    data = _daily(database)
    t = 20.25 if database == "hirid" else 18.
    row = data.loc[t]
    assert row.urine24_est_available_at == 20.25
    if database != "hirid":
        assert row.urine24_uses_future
        assert row.urine24_est_available_at > t
        assert data.loc[23., "urine24_available_at"] == 24
        assert data.loc[23., "urine24_uses_future"]
    assert data.loc[24., "urine24_available_at"] == 24


def _assert_descriptive(database):
    times = [0., 4., 8., 12.]
    urine = _table("urine", times, [25. if database == "hirid" else 100.] * 4)
    weight = _table("weight", [0.], [50.])
    result = callbacks._callback_uo_window(
        {"urine": urine, "weight": weight}, _context(database, "uo_6h"), 6, "uo_6h",
    ).data
    row = result.iloc[-1]
    assert row.uo_6h == pytest.approx(0.5)
    assert row.uo_6h_covered_h == 6
    assert not row.uo_6h_partial_window
    first = result.iloc[1]
    assert first.uo_6h_partial_window
    # Actual covered time is the denominator, even on a partial window.
    expected = 0.5 if database == "hirid" else 200 / 5 / 50
    assert first.uo_6h == pytest.approx(expected)


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_exact_kdigo_intersection_and_coverage(database):
    _assert_exact(database)


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_urine24_availability(database):
    _assert_availability(database)


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_measured_and_estimated_daily_volume(database):
    _assert_split(database)


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_descriptive_actual_overlap_and_partial_marker(database):
    _assert_descriptive(database)


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_gap_break_and_daily_minimum(database):
    urine = _table("urine", [0., 20., 26., 32., 38., 44.], [1.] * 6)
    ctx = _context(database)
    daily = callbacks._callback_urine24({"urine": urine}, ctx).data.set_index("charttime")
    assert daily.loc[20., "urine24_covered_h"] == 0
    assert pd.isna(daily.loc[32., "urine24_est"])
    assert daily.loc[38., "urine24_covered_h"] == 18
    assert np.isfinite(daily.loc[38., "urine24_est"])
    strict = callbacks._callback_urine24({"urine": urine}, _context(database, min_win=pd.Timedelta(hours=23))).data.set_index("charttime")
    assert pd.isna(strict.loc[38., "urine24_est"])
    stage = kdigo_uo(urine.data, pd.DataFrame({"stay_id": [1], "weight": [50.]}),
                     time_unit="hours", source_is_rate=database == "hirid")
    after_gap = stage.loc[stage.charttime == 20.].iloc[0]
    assert pd.isna(after_gap.aki_stage_uo)
    assert after_gap.uo_rt_6hr_covered_h == 0


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_nonzero_straddling_interval_is_apportioned(database):
    urine = _table("urine", [0., 4., 8., 12., 16., 20., 24.],
                   [0., 40., 80., 120., 160., 200., 240.])
    if database == "hirid":
        urine.data["urine"] /= 4
    result = kdigo_uo(urine.data, pd.DataFrame({"stay_id": [1], "weight": [50.]}),
                      time_unit="hours", source_is_rate=database == "hirid")
    row = result.iloc[-1]
    # (18,24] contains half of (16,20] and all of (20,24].
    assert row.uo_rt_6hr == pytest.approx((100 + 240) / 6 / 50)
    assert row.uo_rt_12hr == pytest.approx((160 + 200 + 240) / 12 / 50)
    assert row.uo_rt_24hr == pytest.approx(840 / 24 / 50)


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_daily_gap_rejection_uses_source_clock_not_floor(database):
    urine = _table("urine", [0., 12.1, 18.1, 24.1, 30.1], [25.] * 5)
    result = callbacks._callback_urine24({"urine": urine}, _context(database)).data
    row = result.loc[result.charttime == (12.1 if database == "hirid" else 12.)].iloc[0]
    assert row.urine24_covered_h == 0


@pytest.mark.parametrize("database", DATABASES)
def test_source_clock_gap_is_retained_after_hourly_pooling(database):
    urine = _table("urine", [0., 12., 18., 24., 30.], [1.] * 5)
    urine.data["_urine_available_at"] = [0.1, 12.9, 18.9, 24.9, 30.9]
    urine.data["_urine_gap_h"] = np.r_[0., np.diff(urine.data._urine_available_at)]
    tables = {"urine": urine, "weight": _table("weight", [0.], [50.])}
    daily = callbacks._callback_urine24(tables, _context(database)).data.set_index("charttime")
    assert daily.loc[12., "urine24_covered_h"] == 0
    staged = kdigo_uo(urine.data, tables["weight"].data, time_unit="hours",
                      source_is_rate=database == "hirid").set_index("charttime")
    assert staged.loc[12., "uo_rt_6hr_covered_h"] == 0
    descriptive = callbacks._callback_uo_window(tables, _context(database, "uo_6h"), 6, "uo_6h").data.set_index("charttime")
    assert descriptive.loc[12., "uo_6h_covered_h"] == 0
    if database == "hirid":
        assert descriptive.loc[12., "uo_6h_assessment_covered_h"] == 0


def test_source_availability_excludes_missing_values():
    from easyicu.scores.urine_output import source_availability

    frame = _table("urine", [1.25, 1.75], [10., np.nan]).data
    result = source_availability(frame, ["stay_id"], "charttime", pd.Timedelta(hours=1))
    assert result._urine_available_at.item() == 1.25


@pytest.mark.parametrize("database", DATABASES)
def test_hourly_pooling_does_not_invent_a_gap_between_valid_records(database, tmp_path):
    from easyicu.api.concepts import clear_global_loader, load_concepts
    from easyicu.config import load_src_cfg
    from easyicu.scores.kdigo_aki import detect_id_col, detect_time_col

    root = tmp_path / database
    _clock_source(root, database)
    table, selector, code, clock, factor = {
        "miiv": ("outputevents", "itemid", 226559, "charttime", None),
        "mimic": ("outputevents", "itemid", 226559, "charttime", None),
        "eicu": ("intakeoutput", "celllabel", "Urine", "intakeoutputoffset", 60),
        "aumc": ("numericitems", "itemid", 8794, "measuredat", 3600000),
        "sic": ("data_float_h", "DataID", 725, "Offset", 3600),
        "hirid": ("observations", "variableid", 10020000, "datetime", None),
    }[database]
    value = load_src_cfg(database).tables[table].defaults.val_var
    paths = list(root.glob(f"{table}*/**/*.parquet")) + list(root.glob(f"{table}.parquet"))
    placed = False
    for path in paths:
        frame = pd.read_parquet(path)
        mask = frame[selector].eq(code)
        if mask.any() and not placed:
            record = frame.loc[mask].iloc[:1]
            times = np.array([0.1, 12.0, 12.9, 18.9, 24.9])
            origin = record[clock].iloc[0] - (pd.Timedelta(hours=.25) if factor is None else .25 * factor)
            records = pd.concat([record] * len(times), ignore_index=True)
            records[clock] = origin + (pd.to_timedelta(times, unit="h") if factor is None else times * factor)
            records[value] = 25.
            frame = pd.concat([frame.loc[~mask], records], ignore_index=True)
            placed = True
        else:
            frame = frame.loc[~mask]
        frame.to_parquet(path, index=False)
    assert placed
    path = root / f"{table}.parquet"
    if path.exists():
        bucket = root / f"{table}_bucket" / "bucket_id=0"
        bucket.mkdir(parents=True)
        path.rename(bucket / "part.parquet")
    try:
        result = load_concepts(["urine24", "uo_6h", "uo_rt_6hr"], database=database, data_path=root,
                               verbose=False, keep_components=True, r_compatible=False,
                               concept_workers=1, parallel_workers=1)
        axis = detect_time_col(result)
        assert not result.duplicated([detect_id_col(result), axis]).any()
        row = result.loc[result[axis] == 12.].iloc[0]
        # 0.1 -> 12.0 -> 12.9 has no gap over 12 h. Availability is
        # the last record, but the next interval starts at the first record.
        assert row.urine24_covered_h >= 12
        assert row.uo_6h_covered_h == 6
        assert np.isfinite(row.uo_rt_6hr)
        assert result.loc[result[axis] == 24., "urine24"].notna().any()
    finally:
        clear_global_loader()


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_duplicate_channels_share_one_collection_interval(database):
    amounts = [25., 50., 50., 100., 100.]
    if database == "hirid":
        amounts = [25.] * 5
    result = kdigo_uo(
        _table("urine", [0., 4., 4., 8., 12.], amounts).data,
        pd.DataFrame({"stay_id": [1], "weight": [50.]}), time_unit="hours",
        source_is_rate=database == "hirid",
    )
    # The window at 8 h intersects the duplicated collection ending at 4 h.
    assert result.loc[result.charttime == 8., "uo_rt_6hr"].item() == pytest.approx(0.5)
    assert result.iloc[-1].uo_rt_6hr == pytest.approx(0.5)
    assert result.iloc[-1].uo_rt_6hr_covered_h == 6


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_sofa2_does_not_stage_urine_over_a_rejected_gap(database):
    times = list(np.arange(13.)) + [32.]
    tables = {"urine": _table("urine", times, [0.] * len(times)),
              "weight": _table("weight", [0.], [50.])}
    scoring = {"crea": _table("crea", [32.], [1.])}
    for window in (6, 12, 24):
        name = f"uo_{window}h"
        scoring[name] = callbacks._callback_uo_window(tables, _context(database, name), window, name)
    result = callbacks.CALLBACK_REGISTRY["sofa2_renal"](scoring, _context(database, "sofa2_renal")).data
    assert result.loc[result.charttime == 32., "sofa2_renal"].item() == 0
    if database == "hirid":
        assert scoring["uo_6h"].data.iloc[-1].uo_6h_assessment_covered_h == 0


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_kdigo_requires_complete_window_before_publishing_stage(database):
    result = kdigo_uo(
        _table("urine", [0., 4.], [25., 100.]).data,
        pd.DataFrame({"stay_id": [1], "weight": [50.]}), time_unit="hours",
        source_is_rate=database == "hirid",
    )
    assert result.aki_stage_uo.isna().all()
    assert result.aki_stage_uo_available_at.isna().all()
    assert result.iloc[-1].uo_rt_6hr_covered_h == (4 if database == "hirid" else 5)


@pytest.mark.parametrize("database", DATABASES)
@pytest.mark.parametrize("rate,stage", [(1e-10, 3), (2e-9, 2)])
def test_all_databases_preserve_anuria_tolerance(database, rate, stage):
    amount = rate * 50 * (1 if database == "hirid" else 12)
    result = kdigo_uo(
        _table("urine", [0., 12.], [0., amount]).data,
        pd.DataFrame({"stay_id": [1], "weight": [50.]}), time_unit="hours",
        source_is_rate=database == "hirid",
    )
    assert result.iloc[-1].aki_stage_uo == stage


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_bundle_exports_kdigo_coverage_and_stage_availability(database):
    from easyicu.scores.aki_profiles import build_renal_aki_bundle

    result = build_renal_aki_bundle(
        database, urine_df=_table("urine", [0., 6., 12., 24.], [0.] * 4).data,
        weight_df=pd.DataFrame({"stay_id": [1], "weight": [50.]}), time_unit="hours",
    )
    row = result.loc[result.charttime == 24].iloc[0]
    assert row.aki_stage_uo_reference == 3
    assert row.aki_stage_uo_reference_available_at == 24
    for window in (6, 12, 24):
        assert row[f"uo_rt_{window}hr_covered_h"] == window
    assert pd.isna(result.loc[result.charttime == 0, "aki_stage_uo_reference_available_at"].item())


@pytest.mark.parametrize("database", DATABASES)
def test_all_databases_sofa1_uses_daily_estimate_only_when_full_volume_missing(database):
    from easyicu.callbacks import sofa_renal

    data = _daily(database)
    partial_time = 20.25 if database == "hirid" else 18.
    assert data.loc[partial_time, "urine24_est"] == 600
    scores = sofa_renal(pd.Series([1., 1., 1.]), pd.Series([600., np.nan, np.nan]),
                        pd.Series([100., 100., np.nan]))
    assert scores.tolist() == [0, 4, 0]
    daily = _table("urine24", [0., 1., 2.], [600., np.nan, np.nan])
    daily.data["urine24_est"] = [100., 100., np.nan]
    scored = callbacks.CALLBACK_REGISTRY["sofa_renal"](
        {"crea": _table("crea", [0., 1., 2.], [1., 1., 1.]), "urine24": daily},
        _context(database, "sofa_renal"),
    ).data
    assert scored.sofa_renal.tolist() == [0, 4, 0]


@pytest.mark.parametrize("database", ["miiv", "eicu"])
def test_daily_estimate_survives_the_public_loader(database, tmp_path):
    from easyicu.api.concepts import clear_global_loader, load_concepts
    from tests.support.renal_export import write_renal_source

    write_renal_source(tmp_path, database, 2, hours=30)
    try:
        result = load_concepts(
            ["urine24", "urine24_est", "sofa_renal", "urine24_covered_h",
             "urine24_available_at", "urine24_est_available_at", "uo_6h", "uo_6h_covered_h",
             "uo_6h_partial_window"], database=database, data_path=tmp_path,
            verbose=False, concept_workers=1, parallel_workers=1, keep_components=True, r_compatible=False,
        )
        assert result.urine24.notna().any()
        assert result.urine24_est.notna().any()
        assert result.sofa_renal.notna().any()
        assert result.loc[result.urine24.notna(), "urine24_covered_h"].eq(24).all()
        estimated = result.loc[result.urine24_est.notna()]
        assert estimated.urine24.isna().all()
        assert estimated.urine24_covered_h.between(18, 24, inclusive="left").all()
        assert estimated.urine24_est_available_at.notna().all()
    finally:
        clear_global_loader()


def test_urine_coverage_and_availability_are_catalog_bookkeeping():
    from easyicu.webserver.catalog_metadata import concept_audience

    names = ["urine24_covered_h", "urine24_available_at", "urine24_est_available_at",
             "urine24_uses_future", "aki_stage_uo_reference_available_at"]
    for window in (6, 12, 24):
        names += [f"uo_{window}h_covered_h", f"uo_{window}h_partial_window",
                  f"uo_rt_{window}hr_covered_h"]
    for name in names:
        assert concept_audience(name, {}) == "internal", name
    for name in ("urine24", "urine24_est", "urine24_observed_ml"):
        assert concept_audience(name, {}) == "clinical", name


def test_urine_companions_are_exportable_but_absent_from_planner_rosters(monkeypatch):
    from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog
    from easyicu.research_agent.agents.planner import _format_concept_id_allowlist
    from easyicu.resources import load_dictionary

    names = {"urine24_est", "urine24_observed_ml", "urine24_covered_h",
             "urine24_available_at", "urine24_est_available_at", "urine24_uses_future",
             "aki_stage_uo_reference_available_at"}
    for window in (6, 12, 24):
        names.update({f"uo_{window}h_covered_h", f"uo_{window}h_partial_window",
                      f"uo_rt_{window}hr_covered_h"})
    assert load_dictionary().get("urine24_est") is not None
    prompt = _format_concept_id_allowlist()
    assert not any(f"`{name}`" in prompt for name in names)
    from easyicu.research_agent import concept_catalog

    clinical = {"urine24", "uo_6h", "uo_12h", "uo_24h"}
    roster = concept_catalog.load_concept_catalog(restrict_to=names | clinical)
    monkeypatch.setattr(concept_catalog, "load_concept_catalog", lambda: roster)
    for database in DATABASES:
        menu = set(build_database_capability_catalog(database).ids())
        assert not names & menu
        assert clinical <= menu


@pytest.mark.parametrize("prepared", [False, True])
def test_source_metadata_cannot_reintroduce_urine_receipts_to_planner(prepared, tmp_path, monkeypatch):
    import json
    from easyicu.concept.catalog import PLANNER_HIDDEN_CONCEPTS
    from easyicu.research_agent.acquisition import catalog
    from easyicu.webserver.agent_pipeline_runs import _metadata_only_planning_catalog

    source = catalog.AvailableCatalog("synthetic", [
        catalog.CatalogConcept(name) for name in sorted(PLANNER_HIDDEN_CONCEPTS | {"urine24"})
    ])
    monkeypatch.setattr(catalog, "build_database_capability_catalog", lambda _: catalog.AvailableCatalog("base"))
    if prepared:
        (tmp_path / "easyicu_export_manifest.json").write_text(json.dumps({
            "entry_mode": "study_local_prepared_cohort",
        }))
    result = _metadata_only_planning_catalog(
        database="miiv", export_path=tmp_path, _native_metadata={"catalog": source},
    )
    assert result.ids() == ["urine24"]


@pytest.mark.parametrize("database", DATABASES)
def test_rrt_fallback_keeps_descriptive_and_assessment_coverage_separate(database):
    times = [0., 4., 8., 12.]
    tables = {"urine": _table("urine", times, [0.] * 4),
              "weight": _table("weight", [0.], [50.])}
    for name, value in (("crea", 1.), ("potassium", 6.), ("ph", 7.4),
                        ("bicarb", 24.), ("rrt", False)):
        tables[name] = _table(name, [12.], [value])
    callbacks._callback_rrt_criteria(tables, _context(database, "rrt_criteria"))
    for window in (6, 12, 24):
        name = f"uo_{window}h"
        expected = callbacks._callback_uo_window(
            {"urine": tables["urine"], "weight": tables["weight"]},
            _context(database, name), window, name,
        ).data
        pd.testing.assert_frame_equal(tables[name].data, expected)


@pytest.mark.parametrize("database", DATABASES)
@pytest.mark.parametrize("path_kind", ["single", "batch", "bucketed", "native", "legacy", "callback"])
def test_source_record_availability_survives_hourly_loading(database, path_kind, tmp_path, monkeypatch):
    from easyicu.api.concepts import clear_global_loader, load_concepts

    root = tmp_path / database
    _clock_source(root, database)
    if path_kind == "bucketed":
        table = {"miiv": "outputevents", "mimic": "outputevents", "eicu": "intakeoutput",
                 "aumc": "numericitems", "hirid": "observations", "sic": "data_float_h"}[database]
        path = root / f"{table}.parquet"
        if path.exists():
            bucket = root / f"{table}_bucket" / "bucket_id=0"
            bucket.mkdir(parents=True)
            path.rename(bucket / "part.parquet")
    options = dict(database=database, data_path=root, verbose=False, keep_components=True,
                   concept_workers=1, parallel_workers=1, r_compatible=False)
    try:
        if path_kind == "native":
            import tempfile
            from easyicu.api.extraction import extract_database

            monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
            for name in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
                monkeypatch.delenv(name, raising=False)
            output = tmp_path / "out"
            result = extract_database(database, data_path=str(root), output_dir=str(output),
                                      modules=["outcome", "renal"], native_export_v2=True, verbose=False)
            assert not result["modules"]["renal"].get("errors")
            frame = pd.read_parquet(output / "renal.parquet")
            daily = frame
        elif path_kind in {"single", "legacy", "callback"}:
            daily = load_concepts(["urine24"], **options)
            clear_global_loader()
            if path_kind == "legacy":
                from easyicu.scores.kdigo_aki import load_kdigo_aki

                frame = load_kdigo_aki(database, data_path=str(root), verbose=False).rename(columns={
                    "aki_stage_uo": "aki_stage_uo_reference",
                    "aki_stage_uo_available_at": "aki_stage_uo_reference_available_at",
                })
            elif path_kind == "callback":
                loaded = load_concepts(["kdigo_uo", "kdigo_aki"], merge=False, **options)
                direct = loaded["kdigo_uo"].data
                assert direct.aki_stage_uo.notna().any()
                assert np.allclose(direct.aki_stage_uo_available_at.dropna() % 1, .25)
                frame = loaded["kdigo_aki"].data
            else:
                frame = load_concepts(["aki_stage_uo_reference", "aki_stage_uo_reference_available_at"], **options)
        else:
            # SIC urine shares the batch-eligible long table with saturation.
            # Explicit raw concepts reach that branch; recursive daily/stage
            # dependencies alone do not. Other urine sources bypass it.
            batch_inputs = ["urine", "spo2"] if database == "sic" else []
            frame = load_concepts(
                batch_inputs + ["urine24", "urine24_est", "aki_stage_uo_reference", "aki_stage_uo_reference_available_at"],
                **options,
            )
            daily = frame
        full = daily.loc[daily.urine24.notna()]
        assert not full.empty
        assert np.allclose(full.urine24_available_at % 1, .25)
        stages = frame.loc[frame.aki_stage_uo_reference.notna()]
        assert not stages.empty
        assert np.allclose(stages.aki_stage_uo_reference_available_at % 1, .25)
    finally:
        clear_global_loader()


def test_sic_batch_pooled_urine_keeps_availability_for_downstream_callbacks(tmp_path):
    from easyicu.api.concepts import clear_global_loader, load_concepts

    root = tmp_path / "sic"
    _clock_source(root, "sic")
    path = root / "data_float_h.parquet"
    if path.exists():
        bucket = root / "data_float_h_bucket" / "bucket_id=0"
        bucket.mkdir(parents=True)
        path.rename(bucket / "part.parquet")
    try:
        tables = load_concepts(
            ["urine", "spo2"], database="sic", data_path=root, merge=False,
            r_compatible=False, verbose=False, concept_workers=1, parallel_workers=1,
        )
        daily = callbacks._callback_urine24({"urine": tables["urine"]}, _context("sic")).data
        full = daily.loc[daily.urine24.notna()]
        assert not full.empty
        assert np.allclose(full.urine24_available_at % 1, .25)
    finally:
        clear_global_loader()


def test_r5c_aumc_001_exports_availability():
    _assert_availability("aumc")


def test_r5c_aumc_002_clips_collection_intervals():
    _assert_exact("aumc")


def test_r5c_hirid_001_rate_path_obeys_coverage():
    _assert_split("hirid")


def test_r5_eicu_018_clips_collection_intervals():
    _assert_exact("eicu")


def test_r5_eicu_032_exports_availability():
    _assert_availability("eicu")


def test_r5_eicu_033_splits_daily_estimate():
    _assert_split("eicu")


def test_r5_sic_006_clips_collection_intervals():
    _assert_exact("sic")


def test_r5_sic_023_exports_availability():
    _assert_availability("sic")


def test_r5_sic_024_splits_daily_estimate():
    _assert_split("sic")


def test_r5c_iv_001_clips_collection_intervals():
    _assert_exact("miiv")


def test_r5c_iv_002_uses_actual_overlap():
    _assert_descriptive("miiv")


def test_r5c_iv_003_splits_daily_estimate():
    _assert_split("miiv")


def test_r5c_iv_004_exports_availability():
    _assert_availability("miiv")


def test_r5c_iii_001_clips_collection_intervals():
    _assert_exact("mimic")


def test_r5c_iii_002_uses_actual_overlap():
    _assert_descriptive("mimic")


def test_r5c_iii_003_splits_daily_estimate():
    _assert_split("mimic")


def test_r5c_iii_004_exports_availability():
    _assert_availability("mimic")
