"""Synthetic parity and repeated-work bounds for renal export kernels."""

from types import SimpleNamespace
import weakref

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept import callbacks
from easyicu.config import DataSourceConfig
from easyicu.resources import load_dictionary
from easyicu.scores.urine_windows import assess_urine_windows
from easyicu.table import ICUTable
from tests.support import urine_window_reference as old
from tests.support.renal_export import write_renal_source


def test_three_requested_windows_assess_the_shared_source_once(monkeypatch):
    urine = pd.DataFrame({
        "stay_id": np.repeat([11, 12], 30), "charttime": np.tile(np.arange(30.0), 2),
        "urine": 10.0,
    })
    weight = pd.DataFrame({"stay_id": [11, 12], "weight": [70.0, 80.0]})
    source = SimpleNamespace(config=DataSourceConfig(name="miiv"), base_path=None)
    resolver = ConceptResolver(load_dictionary(include_sofa2=True))
    tables = {
        "urine": ICUTable(urine, id_columns=["stay_id"], index_column="charttime", value_column="urine"),
        "weight": ICUTable(weight, id_columns=["stay_id"], value_column="weight"),
    }
    monkeypatch.setattr(resolver, "get_raw_concept", lambda name, *_: tables[name])
    calls = []
    original = callbacks.assess_urine_windows

    def counted(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(callbacks, "assess_urine_windows", counted)
    for hours in (6, 12, 24):
        name = f"uo_{hours}h"
        ctx = callbacks.ConceptCallbackContext(name, None, pd.Timedelta(hours=1), resolver, source, None)
        result = callbacks._callback_uo_window({}, ctx, hours, name)
        assert len(result.data) == len(urine)
    assert calls == [len(urine)]


@pytest.mark.parametrize("change", ["interval", "weight", "cohort", "source_kind", "clear", "budget"])
def test_shared_assessment_respects_scope_and_eviction(monkeypatch, change):
    urine = ICUTable(
        pd.DataFrame({"stay_id": np.repeat([11, 12], 30), "charttime": np.tile(np.arange(30.0), 2), "urine": 14.0}),
        id_columns=["stay_id"], index_column="charttime", value_column="urine",
    )
    weight = ICUTable(pd.DataFrame({"stay_id": [11, 12], "weight": [70.0, 80.0]}), id_columns=["stay_id"], value_column="weight")
    resolver = ConceptResolver(load_dictionary(include_sofa2=True))
    ctx = callbacks.ConceptCallbackContext("uo_6h", None, pd.Timedelta(hours=1), resolver, SimpleNamespace(), None)
    calls = []
    original = callbacks.assess_urine_windows

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(callbacks, "assess_urine_windows", counted)
    if change == "budget":
        resolver._cache_budget_bytes = 1
    callbacks._shared_urine_assessment(urine, weight, ctx, False)
    source_is_rate = False
    if change == "interval":
        ctx.interval = pd.Timedelta(hours=2)
    elif change == "weight":
        weight = ICUTable(weight.data.assign(weight=100.0), id_columns=["stay_id"], value_column="weight")
    elif change == "cohort":
        urine = ICUTable(urine.data[urine.data.stay_id == 11].copy(), id_columns=["stay_id"], index_column="charttime", value_column="urine")
    elif change == "source_kind":
        source_is_rate = True
    elif change == "clear":
        resolver.clear()
    result = callbacks._shared_urine_assessment(urine, weight, ctx, source_is_rate)
    expected = old.assess_urine_windows(urine.data, weight.data, id_columns=["stay_id"], time_column="charttime", interval=ctx.interval, source_is_rate=source_is_rate)
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    assert len(calls) == 2


@pytest.mark.parametrize("source_is_rate", [False, True])
@pytest.mark.parametrize("interval_hours", [0.7, 1.0, 2.0])
@pytest.mark.parametrize("time_kind", ["hours", "timedelta", "datetime"])
@pytest.mark.parametrize("case", ["empty", "single", "dense", "gaps", "invalid", "duplicates"])
def test_window_evidence_matches_original(source_is_rate, interval_hours, time_kind, case):
    times = {
        "empty": [], "single": [1.0], "dense": list(np.arange(36.0)),
        "gaps": [0.0, 1.0, 2.0, 30.0, 31.0, 60.0],
        "invalid": list(np.arange(36.0)), "duplicates": [0.0, 1.0, 1.0, 2.0, 3.0, 30.0],
    }[case]
    frames = []
    for stay in (12, 11):
        t = np.array(times)
        values = np.full(len(t), 12.5)
        if case == "invalid":
            values[:5] = [0, -1, np.nan, np.inf, 0.01]
        if time_kind == "timedelta":
            t = pd.to_timedelta(t, unit="h")
        elif time_kind == "datetime":
            t = pd.Timestamp("2020-01-01") + pd.to_timedelta(t, unit="h")
        frames.append(pd.DataFrame({"stay_id": np.full(len(t), stay, dtype=np.int64), "charttime": t, "urine": values}))
    urine = pd.concat(frames, ignore_index=True)
    weight = pd.DataFrame({"stay_id": [11, 12], "weight": [70.0, np.nan]})
    kwargs = dict(id_columns=["stay_id"], time_column="charttime", interval=pd.Timedelta(hours=interval_hours), source_is_rate=source_is_rate)
    expected = old.assess_urine_windows(urine, weight, **kwargs)
    actual = assess_urine_windows(urine, weight, **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("rate", [False, True])
def test_unequal_stay_lengths_cross_batch_boundaries(rate):
    rng = np.random.default_rng(27)
    frames = []
    for stay in range(2070):
        size = int(rng.integers(1, 70))
        frames.append(pd.DataFrame({
            "stay_id": stay, "charttime": np.cumsum(rng.choice([0.25, 1.0, 2.0, 30.0], size)),
            "urine": rng.uniform(0, 200, size),
        }))
    urine = pd.concat(frames, ignore_index=True)
    weight = pd.DataFrame({"stay_id": np.arange(2070), "weight": rng.uniform(40, 120, 2070)})
    kwargs = dict(id_columns=["stay_id"], time_column="charttime", interval=pd.Timedelta(hours=1), source_is_rate=rate)
    pd.testing.assert_frame_equal(
        assess_urine_windows(urine, weight, **kwargs),
        old.assess_urine_windows(urine, weight, **kwargs), check_exact=True,
    )


def test_each_stays_prefix_starts_at_zero_with_valid_weights():
    urine = pd.DataFrame({
        "stay_id": np.repeat([11, 12], 30), "charttime": np.tile(np.arange(30.0), 2),
        "urine": np.repeat([8.125, 47.125], 30),
    })
    weight = pd.DataFrame({"stay_id": [11, 12], "weight": [70.0, 80.0]})
    kwargs = dict(id_columns=["stay_id"], time_column="charttime", interval=pd.Timedelta(hours=1))
    actual = assess_urine_windows(urine, weight, **kwargs)
    expected = old.assess_urine_windows(urine, weight, **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    first_complete = actual.loc[(actual.stay_id == 12) & (actual.charttime == 5.0)]
    assert first_complete.uo_6h_assessment_rate.item() == 47.125 / 80.0


@pytest.mark.parametrize("database", ["eicu", "miiv"])
@pytest.mark.parametrize("module", ["renal", "sofa2_score"])
@pytest.mark.parametrize("unit_variant", [False, True])
def test_raw_source_loader_frames_match_original(database, module, unit_variant, tmp_path, monkeypatch):
    from easyicu.api import concepts as api
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL
    import easyicu.callbacks as descriptive
    import easyicu.scores.aki_profiles as profiles
    import easyicu.io.ts_utils as ts

    source = tmp_path / database
    counts = write_renal_source(source, database, 32, hours=30)
    assert counts["intakeoutput" if database == "eicu" else "outputevents"] > 3840
    if unit_variant:
        lab_path = source / ("lab.parquet" if database == "eicu" else "labevents.parquet")
        lab = pd.read_parquet(lab_path)
        unit_column = "labmeasurenameinterface" if database == "eicu" else "valueuom"
        creatinine = lab.labname.eq("creatinine") if database == "eicu" else lab.itemid.eq(50912)
        lab.loc[creatinine, unit_column] = np.resize(["mg/dL", "mg/dl", "MG/DL"], creatinine.sum())
        lab.to_parquet(lab_path, index=False)
        if database == "miiv":
            path = source / "outputevents.parquet"
            output = pd.read_parquet(path)
            output["valueuom"] = np.resize(["mL", "ml", "mL "], len(output))
            output.to_parquet(path, index=False)
    kwargs = dict(
        concepts=list(CONCEPT_GROUPS_INTERNAL[module]), database=database, data_path=str(source),
        verbose=False, keep_components=True, concept_workers=1, parallel_workers=1,
    )
    with monkeypatch.context() as legacy:
        legacy.setattr(callbacks, "_concat_component_frames", old._concat_component_frames)
        legacy.setattr(callbacks, "assess_urine_windows", old.assess_urine_windows)
        legacy.setattr(descriptive, "_urine_window_avg", old._urine_window_avg)
        legacy.setattr(descriptive, "_urine_window_avg_multi", old._urine_window_avg_multi)
        legacy.setattr(callbacks, "_urine24_batch", old._urine24_batch)
        legacy.setattr(profiles, "_event_stage_at_exact_rows", old._event_stage_at_exact_rows)
        legacy.setattr(profiles, "_rolling_stage_max", old._rolling_stage_max)
        legacy.setattr(ts, "_slide_vectorized_bulk", old._slide_vectorized_bulk)
        expected = api.load_concepts(**kwargs)
    api.clear_global_loader()
    actual = api.load_concepts(**kwargs)
    api.clear_global_loader()
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    if module == "renal":
        assert actual.rrt.fillna(False).any(), "Fixture must include active dialysis"


@pytest.mark.parametrize("database", ["eicu", "miiv"])
@pytest.mark.parametrize("module", ["renal", "sofa2_score"])
def test_patient_batches_preserve_original_loader_frames(database, module, tmp_path, monkeypatch):
    from easyicu.api import concepts as api
    from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL

    source = tmp_path / database
    write_renal_source(source, database, 32, hours=30)
    kwargs = dict(
        concepts=list(CONCEPT_GROUPS_INTERNAL[module]), database=database, data_path=str(source),
        patient_ids=list(range(1, 33)), verbose=False, keep_components=True,
        concept_workers=1, parallel_workers=1,
    )
    full = api.load_concepts(**kwargs, chunk_size=100)
    api.clear_global_loader()
    batched = api.load_concepts(**kwargs, chunk_size=10)
    api.clear_global_loader()
    import easyicu.callbacks as descriptive
    import easyicu.scores.aki_profiles as profiles
    import easyicu.io.ts_utils as ts

    with monkeypatch.context() as legacy:
        legacy.setattr(callbacks, "_concat_component_frames", old._concat_component_frames)
        legacy.setattr(callbacks, "assess_urine_windows", old.assess_urine_windows)
        legacy.setattr(descriptive, "_urine_window_avg", old._urine_window_avg)
        legacy.setattr(descriptive, "_urine_window_avg_multi", old._urine_window_avg_multi)
        legacy.setattr(callbacks, "_urine24_batch", old._urine24_batch)
        legacy.setattr(profiles, "_event_stage_at_exact_rows", old._event_stage_at_exact_rows)
        legacy.setattr(profiles, "_rolling_stage_max", old._rolling_stage_max)
        legacy.setattr(ts, "_slide_vectorized_bulk", old._slide_vectorized_bulk)
        expected = api.load_concepts(**kwargs, chunk_size=10)
    api.clear_global_loader()
    pd.testing.assert_frame_equal(batched, expected, check_exact=True)
    # The existing batch concatenator changes object-column missing sentinels
    # (pd.NA -> NaN). Preserve that exact old/new behavior above; normalize only
    # this cross-mode comparison, retaining dtypes, column order and all values.
    pd.testing.assert_frame_equal(
        batched.where(batched.notna(), np.nan),
        full.where(full.notna(), np.nan), check_exact=True,
    )


@pytest.mark.parametrize("unit", ["hours", "minutes", "seconds", "timedelta", "datetime"])
@pytest.mark.parametrize("rrt_kind", ["absent", "empty", "active"])
def test_profile_stage_kernels_match_original(unit, rrt_kind):
    from easyicu.scores import aki_profiles as profiles

    frame = pd.DataFrame({
        "stay_id": [12, 11, 11, 12, 11, 11, 12, 12],
        "charttime": [0.0, 0.0, 1.0, 7.0, 1.0, 6.0, 30.0, 36.0],
        "stage": pd.array([0, pd.NA, 1, 0, 3, 0, 2, 0], dtype="Int64"),
    }, index=[8, 2, 4, 7, 6, 1, 5, 3])
    if unit in {"minutes", "seconds"}:
        frame.charttime *= 60 if unit == "minutes" else 3600
    elif unit == "timedelta":
        frame.charttime = pd.to_timedelta(frame.charttime, unit="h")
    elif unit == "datetime":
        frame.charttime = pd.Timestamp("2020-01-01") + pd.to_timedelta(frame.charttime, unit="h")
    rrt = frame[["stay_id", "charttime"]].copy()
    rrt["rrt"] = [True, False, False, True, False, False, False, True]
    if rrt_kind == "absent":
        rrt = None
    elif rrt_kind == "empty":
        rrt = rrt.iloc[:0]
    pd.testing.assert_series_equal(
        profiles._event_stage_at_exact_rows(frame, rrt, "stay_id", "charttime"),
        old._event_stage_at_exact_rows(frame, rrt, "stay_id", "charttime"), check_exact=True,
    )
    kwargs = dict(stage_col="stage", hours=6.0, time_unit=unit)
    pd.testing.assert_series_equal(
        profiles._rolling_stage_max(frame, "stay_id", "charttime", **kwargs),
        old._rolling_stage_max(frame, "stay_id", "charttime", **kwargs), check_exact=True,
    )


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("standard", [False, True])
def test_public_bundle_preserves_projections_with_one_wide_join(monkeypatch, tmp_path, merge, standard):
    from easyicu.api import concepts as api
    from easyicu.scores import aki_profiles as profiles

    names = ["aki_reference", "aki_stage_reference", "aki_reference_status"]
    bundle = pd.DataFrame({
        "stay_id": [11, 11, 12], "charttime": [0.0, 1.0, 1.0],
        "aki_reference": pd.array([False, True, pd.NA], dtype="boolean"),
        "aki_stage_reference": pd.array([0, 2, pd.NA], dtype="Int64"),
        "aki_reference_status": ["evaluated", "evaluated", "unavailable"],
    })
    base = pd.DataFrame({"stay_id": [11, 12], "charttime": [0.0, 0.0], "urine": [70.0, 80.0]})

    def load_standard(*args, **kwargs):
        return base.copy() if merge else {"urine": base.copy()}

    loader = SimpleNamespace(database="miiv", data_path=tmp_path, load_concepts=load_standard)
    monkeypatch.setattr(api, "_get_global_loader", lambda **kwargs: loader)
    monkeypatch.setattr(profiles, "load_renal_aki_bundle", lambda **kwargs: bundle.copy())
    # Reproduce the original set iteration and sequence of narrow outer joins.
    requested = [*names, *(["urine"] if standard else [])]
    outputs = (set(requested) & set(profiles.RENAL_AKI_BUNDLE_OUTPUTS)) & set(profiles.RENAL_AKI_BUNDLE_OUTPUTS)
    expected = base.copy() if standard else pd.DataFrame()
    projections = {}
    for name in sorted(outputs):
        narrow = bundle[["stay_id", "charttime", name]].copy()
        projections[name] = narrow
        expected = expected.merge(narrow, on=["stay_id", "charttime"], how="outer") if not expected.empty else narrow
    joins = []
    original_merge = pd.DataFrame.merge

    def counted(frame, right, *args, **kwargs):
        if any(name in right for name in names):
            joins.append(tuple(right.columns))
        return original_merge(frame, right, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "merge", counted)
    actual = api.load_concepts(
        requested, database="miiv", data_path=tmp_path, patient_ids=[11, 12],
        merge=merge, concept_workers=1, parallel_workers=1, chunk_size=100,
    )
    if merge:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert list(actual.columns[-len(names):]) == sorted(names)
        assert len(joins) == int(standard)
    else:
        assert set(actual) == set(names) | ({"urine"} if standard else set())
        assert [name for name in actual if name in names] == sorted(names)
        for name, frame in projections.items():
            pd.testing.assert_frame_equal(actual[name], frame, check_exact=True)


def test_bundle_releases_consumed_evidence_before_native_profile(monkeypatch):
    from easyicu.scores import aki_profiles as profiles

    consumed = []
    for name in ("apply_reference_aki", "kdigo_stages"):
        original = getattr(profiles, name)

        def track(*args, _original=original, **kwargs):
            frame = _original(*args, **kwargs)
            consumed.append(weakref.ref(frame))
            return frame

        monkeypatch.setattr(profiles, name, track)
    original_native = profiles.apply_source_native_aki

    def native(*args, **kwargs):
        assert all(ref() is None for ref in consumed), "Consumed full-cohort evidence frames remain live"
        return original_native(*args, **kwargs)

    monkeypatch.setattr(profiles, "apply_source_native_aki", native)
    profiles.build_renal_aki_bundle(
        "eicu", crea_df=pd.DataFrame({"stay_id": [11, 11], "charttime": [0.0, 12.0], "crea": [1.0, 1.5]}),
        urine_df=pd.DataFrame({"stay_id": [11] * 30, "charttime": np.arange(30.0), "urine": 70.0}),
        weight_df=pd.DataFrame({"stay_id": [11], "weight": [70.0]}), time_unit="hours",
    )


@pytest.mark.parametrize("kind", ["numeric", "datetime", "datetime_us", "datetime_tz"])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("aggregations", [{"a": "max", "b": "max"}, {"a": "min", "b": "min"}, {"a": "min", "b": "max"}])
def test_bulk_extrema_match_original(kind, full, aggregations):
    from easyicu.io.ts_utils import _slide_vectorized_bulk

    data = pd.DataFrame({
        "stay_id": np.repeat(np.arange(30), 7), "site_id": np.repeat(np.arange(30) % 2, 7),
        "charttime": np.tile([0.0, 0.0, 1.0, 6.0, 6.25, 30.0, 48.0], 30),
        "a": np.tile([np.nan, 2.0, 0.0, -0.0, 3.0, np.inf, 1.0], 30),
        "b": pd.array(np.tile([pd.NA, 1, 0, 0, 1, pd.NA, 0], 30), dtype="Int64"),
    })
    if kind != "numeric":
        data.charttime = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.charttime, unit="h")
        if kind == "datetime_us":
            data.charttime = data.charttime.astype("datetime64[us]")
        elif kind == "datetime_tz":
            data.charttime = data.charttime.dt.tz_localize("UTC")
    kwargs = dict(agg_map=aggregations, window_size_hours=6.0, full_window=full, is_numeric_time=kind == "numeric")
    expected = old._slide_vectorized_bulk(data, ["stay_id"], "charttime", **kwargs)
    actual = _slide_vectorized_bulk(data, ["stay_id"], "charttime", **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    np.testing.assert_array_equal(np.signbit(actual.a), np.signbit(expected.a))


def test_composite_identity_slide_matches_original_fallback(monkeypatch):
    import easyicu.io.ts_utils as ts

    data = pd.DataFrame({
        "stay_id": np.repeat(np.arange(30), 4), "site_id": np.repeat(np.arange(30) % 2, 4),
        "charttime": np.tile([0.0, 1.0, 7.0, 30.0], 30),
        "value": np.tile([0.0, 3.0, np.nan, 1.0], 30),
    })
    kwargs = dict(before=pd.Timedelta(hours=6), after=pd.Timedelta(0), agg_func={"value": "max"})
    with monkeypatch.context() as legacy:
        legacy.setattr(ts, "_slide_vectorized_bulk", old._slide_vectorized_bulk)
        expected = ts.slide(data.copy(), ["stay_id", "site_id"], "charttime", **kwargs)
    actual = ts.slide(data.copy(), ["stay_id", "site_id"], "charttime", **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_bulk_extrema_unequal_lengths_and_missing_values():
    from easyicu.io.ts_utils import _slide_vectorized_bulk

    rng = np.random.default_rng(79)
    lengths = rng.integers(1, 25, 2070)
    data = pd.DataFrame({
        "stay_id": np.repeat(np.arange(len(lengths)), lengths),
        "charttime": np.concatenate([np.cumsum(rng.choice([0.0, 0.25, 1.0, 30.0], n)) for n in lengths]),
        "a": rng.normal(size=lengths.sum()).astype("float32"),
        "b": rng.normal(size=lengths.sum()),
    })
    data.loc[data.index % 7 == 0, ["a", "b"]] = np.nan
    kwargs = dict(agg_map={"a": "min", "b": "max"}, window_size_hours=6.0, full_window=False, is_numeric_time=True)
    pd.testing.assert_frame_equal(
        _slide_vectorized_bulk(data, ["stay_id"], "charttime", **kwargs),
        old._slide_vectorized_bulk(data, ["stay_id"], "charttime", **kwargs), check_exact=True,
    )


def test_bulk_extrema_rejects_nonmonotonic_presorted_input():
    from easyicu.io.ts_utils import _slide_vectorized_bulk

    frame = pd.DataFrame({"stay_id": [11, 11, 11], "charttime": [0.0, 3.0, 1.0], "value": [0.0, 1.0, 2.0]})
    kwargs = dict(agg_map={"value": "max"}, window_size_hours=6.0, full_window=False, is_numeric_time=True, _pre_sorted=True)
    for kernel in (old._slide_vectorized_bulk, _slide_vectorized_bulk):
        with pytest.raises(ValueError, match="monotonic"):
            kernel(frame, ["stay_id"], "charttime", **kwargs)


@pytest.mark.parametrize("window", [1, 6, 12, 24])
@pytest.mark.parametrize("case", ["empty", "single", "gaps", "negative", "weight_times"])
def test_descriptive_window_matches_original(window, case):
    from easyicu.callbacks import _urine_window_avg

    times = [] if case == "empty" else ([1.0] if case == "single" else [0.0, 1.0, 2.0, 30.0, 31.0, 60.0])
    urine = pd.DataFrame({"stay_id": np.repeat([11, 12], len(times)), "charttime": np.tile(times, 2), "urine": np.full(2 * len(times), 14.0)})
    if case == "negative":
        urine.loc[:2, "urine"] = [0, -1, np.nan]
    weight = pd.DataFrame({"stay_id": [11, 12], "weight": [70.0, 80.0]})
    if case == "weight_times":
        weight = pd.concat([weight.assign(charttime=0.0), weight.assign(charttime=2.0)], ignore_index=True)
    kwargs = dict(window_hours=window, min_hours=max(1, window // 2), interval=pd.Timedelta(hours=1))
    expected = old._urine_window_avg(urine, weight, **kwargs)
    actual = _urine_window_avg(urine, weight, **kwargs)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("database", ["eicu", "miiv"])
def test_raw_fixture_has_irregular_long_tailed_stays(database, tmp_path):
    source = tmp_path / database
    write_renal_source(source, database, 32, hours=30)
    frame = pd.read_parquet(source / ("intakeoutput.parquet" if database == "eicu" else "outputevents.parquet"))
    identifier = "patientunitstayid" if database == "eicu" else "stay_id"
    time = "intakeoutputoffset" if database == "eicu" else "charttime"
    counts = frame.groupby(identifier).size().reindex(range(1, 33), fill_value=0)
    assert counts.max() > 3000
    assert (counts == 0).any() and (counts == 1).any()
    differences = frame.groupby(identifier)[time].diff().dropna()
    assert differences.nunique() > 100


def _long_tailed_urine():
    rng = np.random.default_rng(37)
    lengths = np.r_[rng.integers(1, 30, 2050), 8193, 1, 4097]
    times = [np.cumsum(rng.choice([0.25, 1.0, 2.0, 30.0], size)) for size in lengths]
    urine = pd.DataFrame({
        "stay_id": np.repeat(np.arange(len(lengths)), lengths),
        "charttime": np.concatenate(times), "urine": rng.uniform(0.0, 140.0, lengths.sum()),
    })
    urine.loc[urine.index % 127 == 0, "urine"] = 0.0
    urine.loc[urine.index % 149 == 0, "urine"] = -1.0
    weight = pd.DataFrame({"stay_id": np.arange(len(lengths)), "weight": rng.uniform(40.0, 120.0, len(lengths))})
    return urine, weight


@pytest.mark.parametrize("source_is_rate", [False, True])
def test_urine_long_tail_has_no_padded_searches_or_string_allocations(monkeypatch, source_is_rate):
    urine, weight = _long_tailed_urine()
    kwargs = dict(id_columns=["stay_id"], time_column="charttime", interval=pd.Timedelta(hours=1), source_is_rate=source_is_rate)
    expected = old.assess_urine_windows(urine, weight, **kwargs)
    original_full = np.full

    def full(shape, value, *args, **kwargs):
        assert kwargs.get("dtype") is not object, "Reuse scalar reason strings instead of allocating one string per cell"
        return original_full(shape, value, *args, **kwargs)

    def take(*args, **kwargs):
        raise AssertionError("Urine windows must search actual one-dimensional stay rows")

    monkeypatch.setattr(np, "full", full)
    monkeypatch.setattr(np, "take_along_axis", take)
    pd.testing.assert_frame_equal(assess_urine_windows(urine, weight, **kwargs), expected, check_exact=True)


def test_rolling_long_tail_uses_shared_flat_bounds(monkeypatch):
    import easyicu.io.ts_utils as ts

    data, _ = _long_tailed_urine()
    data = data.rename(columns={"urine": "value"})
    data["other"] = -data.value
    kwargs = dict(agg_map={"value": "max", "other": "min"}, window_size_hours=24.0, full_window=False, is_numeric_time=True)
    expected = old._slide_vectorized_bulk(data, ["stay_id"], "charttime", **kwargs)
    calls = []
    original_bounds = ts._grouped_window_bounds

    def bounds(*args, **kwargs):
        calls.append(1)
        return original_bounds(*args, **kwargs)

    def take(*args, **kwargs):
        raise AssertionError("Rolling bounds must not allocate padded sparse tables")

    monkeypatch.setattr(np, "take_along_axis", take)
    monkeypatch.setattr(ts, "_grouped_window_bounds", bounds)
    pd.testing.assert_frame_equal(ts._slide_vectorized_bulk(data, ["stay_id"], "charttime", **kwargs), expected, check_exact=True)
    assert len(calls) == 1


def test_row_batches_bound_actual_rows_and_keep_whole_stays():
    from easyicu.io.ts_utils import _grouped_row_batches

    lengths = np.array([1, 2, 600, 1, 129, 2, 3, 1000, 1])
    codes = np.repeat(np.arange(len(lengths)), lengths)
    positions = []
    for batch, edges in _grouped_row_batches(codes, max_rows=64):
        assert edges[0] == 0 and edges[-1] == batch.stop - batch.start
        assert batch.stop - batch.start <= 64 or len(edges) == 2
        assert batch.start == 0 or codes[batch.start - 1] != codes[batch.start]
        assert batch.stop == len(codes) or codes[batch.stop - 1] != codes[batch.stop]
        positions.extend(range(batch.start, batch.stop))
    assert positions == list(range(len(codes)))


@pytest.mark.parametrize("rate", [False, True])
def test_urine_evidence_crosses_actual_row_batch_boundaries(monkeypatch, rate):
    import easyicu.scores.urine_windows as windows
    from easyicu.io.ts_utils import _grouped_row_batches

    urine, weight = _long_tailed_urine()
    monkeypatch.setattr(windows, "_grouped_row_batches", lambda codes: _grouped_row_batches(codes, max_rows=257))
    kwargs = dict(id_columns=["stay_id"], time_column="charttime", interval=pd.Timedelta(hours=0.7), source_is_rate=rate)
    pd.testing.assert_frame_equal(
        windows.assess_urine_windows(urine, weight, **kwargs),
        old.assess_urine_windows(urine, weight, **kwargs), check_exact=True,
    )


def test_component_merge_avoids_multiindex_union_and_reindex(monkeypatch):
    tables = {
        "a": ICUTable(pd.DataFrame({"stay_id": [12, 11, 11], "charttime": [0.0, 2.0, 2.0], "a": [1.0, 2.0, 3.0]}), id_columns=["stay_id"], index_column="charttime", value_column="a"),
        "b": ICUTable(pd.DataFrame({"stay_id": [11, 12], "charttime": [1.0, 0.0], "b": pd.array([True, False], dtype="boolean")}), id_columns=["stay_id"], index_column="charttime", value_column="b"),
    }
    keys = ["stay_id", "charttime"]
    expected = pd.concat([
        table.data.drop_duplicates(keys, keep="last").set_index(keys)
        for table in tables.values()
    ], axis=1, sort=False).reset_index()

    def forbidden(*args, **kwargs):
        raise AssertionError("Factor component keys once instead of repeatedly unioning/reindexing MultiIndexes")

    monkeypatch.setattr(pd.MultiIndex, "_union", forbidden)
    monkeypatch.setattr(pd.MultiIndex, "get_indexer", forbidden)
    actual, _, _ = callbacks._merge_tables(tables)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("key_kind", ["numeric", "nullable", "categorical", "object", "datetime", "datetime_tz"])
@pytest.mark.parametrize("duplicates", [False, True])
def test_component_merge_matches_original_keys_and_sidecars(key_kind, duplicates):
    a = pd.DataFrame({
        "stay_id": [12, 11, 11, 12], "charttime": [0.0, 2.0, np.nan, 4.0],
        "a": pd.array([1, 2, pd.NA, 4], dtype="Int64"), "receipt": ["x", "y", "z", None],
    })
    b = pd.DataFrame({
        "stay_id": [11, 12, 14, 12], "charttime": [1.0, 0.0, np.nan, 8.0],
        "b": pd.array([True, False, pd.NA, True], dtype="boolean"),
    })
    for frame in (a, b):
        if key_kind == "nullable":
            frame["stay_id"] = frame.stay_id.astype("Int64")
            frame.loc[1, "stay_id"] = pd.NA
        elif key_kind == "categorical":
            frame["stay_id"] = pd.Categorical(frame.stay_id, categories=[11, 12, 14, 19])
        elif key_kind == "object":
            frame["stay_id"] = frame.stay_id.astype(str)
            frame.loc[1, "stay_id"] = None
        elif key_kind.startswith("datetime"):
            frame["charttime"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(frame.charttime, unit="h")
            if key_kind == "datetime_tz":
                frame["charttime"] = frame.charttime.dt.tz_localize("UTC")
    if duplicates:
        a = pd.concat([a, a.iloc[[1]].assign(a=9)], ignore_index=True)
    keys = ["stay_id", "charttime"]
    expected = old._concat_component_frames([a, b], keys)
    actual = callbacks._concat_component_frames([a, b], keys)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
