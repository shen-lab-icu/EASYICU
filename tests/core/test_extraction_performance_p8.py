"""P8 performance contracts, using invented source records only."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.api.extraction import (
    _extract_worker_env_setup,
    _resource_budget_execution_limits,
)
from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.eicu_sedation_scale import RASS_SCALE_CALLBACK
from easyicu.concept.eicu_source_callbacks import cached_auxiliary_source
from easyicu.concept.schema import ConceptSource
from easyicu.datasource import MissingTableSourceError
from easyicu.datasource import FilterOp, FilterSpec
from easyicu.scores.kdigo_aki import _calculate_uo_rates_simple
from easyicu.scores.urine_output import descriptive_windows
from easyicu.table import ICUTable
from tests.support import p8_urine_reference as previous


class _AuxiliarySource:
    def __init__(self):
        self.config = SimpleNamespace(name="eicu")
        self.calls = Counter()
        self.chart = pd.DataFrame({
            "patientunitstayid": [1, 1, 2, 2],
            "nursingchartoffset": [0.0, 60.0, 0.0, 60.0],
            "nursingchartcelltypevalname": ["Sedation Scale"] * 4,
            "nursingchartvalue": pd.array(["RASS", "SAS", "RASS", "RASS"], dtype="string"),
        }, index=[3, 5, 7, 9])
        self.chart.attrs["synthetic_source"] = {"version": [1]}

    def load_table(self, name, *, columns=None, filters=None, verbose=False):
        self.calls[name] += 1
        frame = self.chart.copy()
        for spec in filters or ():
            frame = spec.apply(frame)
        if columns is not None:
            frame = frame[list(columns)]
        return ICUTable(frame, id_columns=["patientunitstayid"],
                        index_column="nursingchartoffset")


def _sedation(source, resolver, stays=(1, 2)):
    frame = pd.DataFrame({
        "patientunitstayid": np.repeat(stays, 2),
        "nursingchartoffset": [0.0, 60.0] * len(stays),
        "rass": [-1.0, -2.0] * len(stays),
    })
    frame.attrs["published"] = {"nested": ["receipt"]}
    definition = ConceptSource.from_mapping({
        "table": "nursecharting", "value_var": "nursingchartvalue",
        "callback": RASS_SCALE_CALLBACK,
    })
    return _apply_callback(frame, definition, "rass", data_source=source,
                           resolver=resolver, interval=pd.Timedelta(hours=1))


def test_b3_scoring_auxiliary_cache_matches_uncached_frames_and_reads_once():
    uncached, cached = _AuxiliarySource(), _AuxiliarySource()
    off = ConceptResolver(ConceptDictionary({}))
    off._cache_budget_bytes = 0
    on = ConceptResolver(ConceptDictionary({}))
    for _ in range(3):
        expected = _sedation(uncached, off)
        actual = _sedation(cached, on)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert actual.attrs == expected.attrs
    assert uncached.calls["nursecharting"] == 3
    assert cached.calls["nursecharting"] == 1
    # A distinct stay filter must never reuse the earlier entry.
    pd.testing.assert_frame_equal(_sedation(cached, on, (2,)),
                                  _sedation(uncached, off, (2,)), check_exact=True)
    assert cached.calls["nursecharting"] == 2
    on.clear_table_cache()
    _sedation(cached, on)
    assert cached.calls["nursecharting"] == 3


def test_b3_keys_cover_projection_selector_clock_dictionary_and_source():
    source = _AuxiliarySource()
    resolver = ConceptResolver(ConceptDictionary({}))
    request = {"columns": ["patientunitstayid", "nursingchartoffset", "nursingchartvalue"],
               "filters": [FilterSpec("patientunitstayid", FilterOp.IN, [1])]}

    def load(ds=source, width=1):
        return cached_auxiliary_source(ds, resolver, pd.Timedelta(hours=width)).load_table(
            "nursecharting", **request,
        )

    baseline = source.load_table("nursecharting", **request)
    miss = load()
    pd.testing.assert_frame_equal(miss.data, baseline.data, check_exact=True)
    miss.data.iloc[0, -1] = "edited on miss"
    miss.data.attrs["synthetic_source"]["version"].append(2)
    actual = load()
    # Cached frames and all nested metadata must be isolated from consumers.
    actual.data.iloc[0, -1] = "edited"
    actual.data.attrs["synthetic_source"]["version"].append(2)
    actual.id_columns.append("edited")
    restored = load()
    pd.testing.assert_frame_equal(restored.data, baseline.data, check_exact=True)
    assert restored.data.attrs == baseline.data.attrs
    assert restored.id_columns == baseline.id_columns
    assert source.calls["nursecharting"] == 2  # one raw baseline + one miss

    request["columns"] = ["nursingchartvalue", "patientunitstayid", "nursingchartoffset"]
    assert list(load().data) == request["columns"]
    request["filters"].append(FilterSpec("nursingchartvalue", FilterOp.IN, ["RASS"]))
    assert load().data.nursingchartvalue.tolist() == ["RASS"]
    load(width=2)
    resolver.dictionary_signature = "another dictionary"
    load(width=2)
    assert source.calls["nursecharting"] == 6
    other = _AuxiliarySource()
    load(other, 2)
    assert other.calls["nursecharting"] == 1
    request["filters"] = [FilterSpec("patientunitstayid", FilterOp.IN, [])]
    assert load().data.empty


def test_b3_filter_operators_and_metadata_have_independent_entries():
    source = _AuxiliarySource()
    resolver = ConceptResolver(ConceptDictionary({}))
    cached = cached_auxiliary_source(source, resolver, None)
    for op, metadata in ((FilterOp.EQ, None), (FilterOp.REGEX, None),
                         (FilterOp.REGEX, {"clock": "a"}), (FilterOp.REGEX, {"clock": "b"}),
                         (FilterOp.REGEX, {"clock": "b"})):
        result = cached.load_table("nursecharting", filters=[
            FilterSpec("nursingchartvalue", op, "RASS", metadata=metadata),
        ])
        assert result.data.nursingchartvalue.tolist() == ["RASS"] * 3
    assert source.calls["nursecharting"] == 4


def test_b3_filter_keys_include_the_effective_in_membership():
    source = _AuxiliarySource()
    cached = cached_auxiliary_source(source, ConceptResolver(ConceptDictionary({})), None)
    earlier = FilterSpec("patientunitstayid", FilterOp.IN, [1])
    # FilterSpec.apply uses the membership captured at construction. Its public
    # value can subsequently differ, and must not alias a fresh filter's key.
    earlier.value[:] = [2]
    fresh = FilterSpec("patientunitstayid", FilterOp.IN, [2])
    first = cached.load_table("nursecharting", filters=[earlier]).data
    second = cached.load_table("nursecharting", filters=[fresh]).data
    assert first.patientunitstayid.tolist() == [1, 1]
    assert second.patientunitstayid.tolist() == [2, 2]
    assert source.calls["nursecharting"] == 2


def test_b3_auxiliary_entries_obey_the_existing_byte_budget_and_lru():
    source = _AuxiliarySource()
    resolver = ConceptResolver(ConceptDictionary({}))
    cached = cached_auxiliary_source(source, resolver, None)
    result = cached.load_table("nursecharting")
    size = resolver._cache_total_bytes
    resolver._cache_budget_bytes = size * 3 // 2
    cached.load_table("another_table")
    assert resolver._cache_total_bytes <= resolver._cache_budget_bytes
    assert resolver._cache_evictions == 1
    reloaded = cached.load_table("nursecharting")
    pd.testing.assert_frame_equal(reloaded.data, result.data, check_exact=True)
    assert source.calls["nursecharting"] == 2
    resolver.drop_source_caches()
    assert resolver._cache_total_bytes == 0
    cached.load_table("nursecharting")
    assert source.calls["nursecharting"] == 3


def test_b3_large_filter_keys_also_consume_the_request_cache_budget():
    source = _AuxiliarySource()
    resolver = ConceptResolver(ConceptDictionary({}))
    resolver._cache_budget_bytes = 2048
    cached = cached_auxiliary_source(source, resolver, None)
    filters = [FilterSpec("patientunitstayid", FilterOp.IN, list(range(1000, 5000)))]
    for _ in range(2):
        assert cached.load_table("nursecharting", filters=filters).data.empty
    assert source.calls["nursecharting"] == 2
    assert resolver._cache_total_bytes <= 2048


def test_b3_unsupported_filter_metadata_bypasses_reuse():
    source = _AuxiliarySource()
    resolver = ConceptResolver(ConceptDictionary({}))
    cached = cached_auxiliary_source(source, resolver, None)
    filters = [FilterSpec("patientunitstayid", FilterOp.IN, [1], metadata={"opaque": object()})]
    first = cached.load_table("nursecharting", filters=filters)
    second = cached.load_table("nursecharting", filters=filters)
    pd.testing.assert_frame_equal(first.data, second.data, check_exact=True)
    assert source.calls["nursecharting"] == 2


def test_b3_nested_callbacks_rebind_their_clock_and_dictionary_owner():
    source = _AuxiliarySource()
    first = ConceptResolver(ConceptDictionary({}))
    second = ConceptResolver(ConceptDictionary({}))
    wrapped = cached_auxiliary_source(source, first, pd.Timedelta(hours=1))
    wrapped.load_table("nursecharting")
    cached_auxiliary_source(wrapped, first, pd.Timedelta(hours=2)).load_table("nursecharting")
    cached_auxiliary_source(wrapped, second, pd.Timedelta(hours=1)).load_table("nursecharting")
    assert source.calls["nursecharting"] == 3


def test_b3_auxiliary_lab_selectors_share_one_union_read():
    class Source:
        config = SimpleNamespace(name="eicu")

        def __init__(self):
            self.calls = Counter()
            self.lab = pd.DataFrame({
                "patientunitstayid": [1] * 4,
                "labresultoffset": [0., 0., 60., 120.],
                "labname": ["paO2", "FiO2", "paCO2", "unrelated"],
                "labresult": [90., 30., 35., 10.],
                "labmeasurenameinterface": ["mm Hg", "%", "mm Hg", "mg/dL"],
            })

        def load_table(self, name, *, columns=None, filters=None, verbose=False):
            self.calls[name] += 1
            if name != "lab":
                raise MissingTableSourceError(name)  # the source's absence signal
            data = self.lab.copy()
            for spec in filters or ():
                data = spec.apply(data)
            return SimpleNamespace(data=data if columns is None else data[list(columns)])

    gas = pd.DataFrame({"patientunitstayid": [1, 1], "labresultoffset": [0., 60.], "ph": [7.4, 7.3]})
    fio2 = pd.DataFrame({"patientunitstayid": [1] * 3, "respchartoffset": [0., 60., 120.],
                         "respchartvaluelabel": ["FiO2"] * 3, "fio2": [45., 70., 80.]})
    sources = [ConceptSource.from_mapping(row) for row in (
        {"table": "lab", "ids": ["pH"], "sub_var": "labname", "value_var": "labresult", "callback": "eicu_arterial_gas"},
        {"table": "respiratorycharting", "value_var": "respchartvalue", "callback": "eicu_fio2_parent"},
    )]
    off, on = Source(), Source()
    cold, warm = ConceptResolver(ConceptDictionary({})), ConceptResolver(ConceptDictionary({}))
    cold._cache_budget_bytes = 0
    for name, frame, source in zip(("ph", "fio2"), (gas, fio2), sources):
        expected = _apply_callback(frame, source, name, data_source=off, resolver=cold)
        actual = _apply_callback(frame, source, name, data_source=on, resolver=warm)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert actual.attrs == expected.attrs
    assert off.calls["lab"] == 2
    assert on.calls["lab"] == 1
    from easyicu.concept.eicu_source_qc import load_optional

    projection = ["labname", "patientunitstayid", "labresultoffset", "labresult"]
    selectors = [FilterSpec("labname", FilterOp.IN, ["FiO2"])]
    stays = on.lab.patientunitstayid.unique()
    expected = load_optional(off, "lab", stays, projection, selectors=selectors)
    actual = load_optional(cached_auxiliary_source(on, warm, None), "lab", stays, projection, selectors=selectors)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert on.calls["lab"] == 1  # different projections/selectors split the union


def test_b3_scoring_request_from_parquet_matches_previous_auxiliary_reads(tmp_path, monkeypatch):
    from copy import deepcopy

    from easyicu.concept import eicu_source_callbacks
    from easyicu.resources import load_dictionary
    from tests.support.eicu_source import _eicu_source

    chart = pd.DataFrame({
        "nursingchartoffset": np.repeat([0., 60., 120.], 2),
        "nursingchartcelltypecat": "Scores",
        "nursingchartcelltypevallabel": "Sedation",
        "nursingchartcelltypevalname": ["Sedation Scale", "Sedation Score"] * 3,
        "nursingchartvalue": ["RASS", "-1", "SAS", "2", "RASS", "-2"],
    })
    prototype = load_dictionary()["rass"]
    definitions = {}
    for name in ("score_a", "score_b", "score_c"):
        definitions[name] = deepcopy(prototype)
        definitions[name].name = name

    def run(label):
        source = _eicu_source(tmp_path / label, "eicu", nursecharting=chart)
        calls = Counter()
        original = source.load_table

        def counted(table, **kwargs):
            calls[table] += 1
            return original(table, **kwargs)

        monkeypatch.setattr(source, "load_table", counted)
        frame = ConceptResolver(ConceptDictionary(definitions)).load_concepts(
            list(definitions), source, merge=True, aggregate="median",
            patient_ids={"patientunitstayid": [1]}, interval=pd.Timedelta(hours=1),
            concept_workers=1, verbose=False,
        )
        source.clear_cache()
        return frame, calls["nursecharting"]

    with monkeypatch.context() as legacy:
        legacy.setattr(eicu_source_callbacks, "cached_auxiliary_source", lambda source, *_: source)
        expected, before_reads = run("before")
    actual, after_reads = run("after")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs == expected.attrs
    assert after_reads < before_reads


_CPU_ENV = ("EASYICU_PARALLEL_MAX_WORKERS", "EASYICU_ARROW_THREADS",
            "EASYICU_DUCKDB_THREADS")


@pytest.fixture
def cpu_environment(monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    # Worker setup writes its environment; keep every write local to this test.
    for name in (*_CPU_ENV, "EASYICU_RESOURCE_BUDGET_MB", "EASYICU_OVERRIDE_MEMORY_GB",
                 "EASYICU_DUCKDB_MEMORY_LIMIT", "EASYICU_CACHE_BUDGET_MB",
                 "EASYICU_DATA_PATH", "EASYICU_FORCE_INPROCESS_BATCH",
                 "TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        monkeypatch.delenv(name, raising=False)
    yield
    from easyicu.runtime.parallel_config import reset_global_config

    reset_global_config()


@pytest.mark.parametrize("budget_gb,tier", [(4, 1), (8, 1), (12, 8), (24, 16), (48, 32), (96, 64)])
def test_b4_no_cpu_limit_preserves_previous_memory_tiers(cpu_environment, budget_gb, tier):
    limits = _resource_budget_execution_limits(budget_gb * 1024)
    assert limits["parallel_max_workers"] == tier
    assert limits["arrow_threads"] == limits["duckdb_threads"] == min(tier, 8)


@pytest.mark.parametrize("budget_gb", [4, 24, 96])
@pytest.mark.parametrize("worker,arrow,duckdb", [(2, 1, 1), (3, 2, 1), (64, 3, 2)])
def test_b4_memory_budget_never_raises_explicit_cpu_limits(
    monkeypatch, cpu_environment, budget_gb, worker, arrow, duckdb,
):
    import os

    for name, value in zip(_CPU_ENV, (worker, arrow, duckdb)):
        monkeypatch.setenv(name, str(value))
    limits = _resource_budget_execution_limits(budget_gb * 1024)
    _extract_worker_env_setup("synthetic/eicu", budget_gb * 1024)
    assert limits["parallel_max_workers"] == min(worker, 1 if budget_gb == 4 else (16 if budget_gb == 24 else 64))
    for name, field, explicit in zip(_CPU_ENV,
                                     ("parallel_max_workers", "arrow_threads", "duckdb_threads"),
                                     (worker, arrow, duckdb)):
        assert int(os.environ[name]) == limits[field] <= explicit
    assert limits["arrow_threads"] <= limits["parallel_max_workers"]
    assert limits["duckdb_threads"] <= limits["parallel_max_workers"]


@pytest.mark.parametrize("argument,environment,expected", [(2, 32, 2), (3, 1, 3), (64, 2, 16)])
def test_b4_explicit_worker_argument_precedes_environment_and_obeys_budget(
    monkeypatch, cpu_environment, argument, environment, expected,
):
    import os

    monkeypatch.setenv("EASYICU_PARALLEL_MAX_WORKERS", str(environment))
    limits = _resource_budget_execution_limits(24 * 1024, max_workers=argument)
    _extract_worker_env_setup("synthetic/eicu", 24 * 1024, max_workers=argument)
    assert limits["parallel_max_workers"] == expected
    assert os.environ["EASYICU_PARALLEL_MAX_WORKERS"] == str(expected)
    assert int(os.environ["EASYICU_ARROW_THREADS"]) == min(8, expected)
    assert int(os.environ["EASYICU_DUCKDB_THREADS"]) == min(8, expected)


@pytest.mark.parametrize("value", ["invalid", "", "0", "-1"])
def test_b4_cpu_environment_parsing_preserves_safe_limits(monkeypatch, cpu_environment, value):
    for name in _CPU_ENV:
        monkeypatch.setenv(name, value)
    limits = _resource_budget_execution_limits(24 * 1024)
    expected = (16, 8, 8) if value in {"invalid", ""} else (1, 1, 1)
    assert tuple(limits[k] for k in ("parallel_max_workers", "arrow_threads", "duckdb_threads")) == expected


@pytest.mark.parametrize("budget", [None, 24 * 1024])
def test_b4_argument_caps_engine_threads_with_or_without_a_memory_budget(
    monkeypatch, cpu_environment, budget,
):
    import os

    for name in _CPU_ENV:
        monkeypatch.setenv(name, "8")
    _extract_worker_env_setup("synthetic/eicu", budget, max_workers=2)
    assert tuple(int(os.environ[name]) for name in _CPU_ENV) == (2, 2, 2)


@pytest.mark.parametrize("budget", [None, 48 * 1024])
def test_b4_api_propagates_argument_and_records_effective_cpu_limits(
    monkeypatch, tmp_path, cpu_environment, budget,
):
    import inspect
    import json

    from easyicu.api import extraction

    captured = {}

    class Process:
        exitcode = 0

        def __init__(self, *, target, args, daemon):
            captured.update(inspect.signature(target).bind(*args).arguments)

        def start(self):
            # Finish the dispatch with an empty synthetic worker receipt.
            for name, _ in captured["module_specs"]:
                path = Path(captured["output_root"]) / name
                path.mkdir(parents=True)
                (path / "_manifest.json").write_text(json.dumps({"saved": {}, "errors": []}))

        def join(self):
            pass

    monkeypatch.setenv("EASYICU_PARALLEL_MAX_WORKERS", "8")
    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "1")
    monkeypatch.setattr(extraction, "_get_extraction_mp_context", lambda _: SimpleNamespace(Process=Process))
    result = extraction.extract_database(
        "eicu", data_path=tmp_path, output_dir=tmp_path / "output", modules=["vitals"],
        patient_ids={"patientunitstayid": [1]}, resource_budget_mb=budget,
        max_workers=2, native_export_v2=False, group_modules=False, verbose=False,
    )
    assert captured["max_workers"] == 2
    assert {k: result["resource_execution_limits"][k] for k in
            ("parallel_max_workers", "arrow_threads", "duckdb_threads")} == {
        "parallel_max_workers": 2, "arrow_threads": 2, "duckdb_threads": 1,
    }


def test_b4_group_worker_installs_the_explicit_argument(monkeypatch, tmp_path, cpu_environment):
    from contextlib import nullcontext
    import os

    from easyicu.api.extraction import _extract_module_group_worker

    for name in _CPU_ENV:
        monkeypatch.setenv(name, "8")
    monkeypatch.setattr("easyicu.api.keep_cache", lambda **_: nullcontext(None))
    _extract_module_group_worker(
        [], [], "eicu", str(tmp_path), None, None, str(tmp_path / "worker"), False,
        resource_budget_mb=48 * 1024, max_workers=2,
    )
    assert tuple(int(os.environ[name]) for name in _CPU_ENV) == (2, 2, 2)


def _urine_stays(clock, *, receipts=True):
    random = np.random.default_rng(734)
    pieces = []
    for stay in range(1, 9):
        # Exact and either-side 12 h gaps; zero, negative and missing channels.
        steps = random.choice([.25, .5, 1., 2., 12., 12. - 1e-7, 12. + 1e-4], 40)
        hours = np.r_[0., 12., 24., 36., 48., 60., 60. + np.cumsum(steps)]
        volumes = random.uniform(0., 200., len(hours))
        volumes[[6, 13, 22]] = [0., -1., np.nan]
        frame = pd.DataFrame({"stay_id": stay, "charttime": hours, "urine": volumes})
        if receipts:
            frame["_urine_available_at"] = hours + random.choice([0., .25, 2.], len(hours))
            frame["_urine_gap_h"] = np.r_[np.nan, np.diff(hours)]
        pieces.append(frame)
    urine = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=19)
    if clock == "minutes":
        urine["charttime"] *= 60.
    elif clock == "seconds":
        urine["charttime"] *= 3600.
    elif clock == "timedelta":
        urine["charttime"] = pd.to_timedelta(urine.charttime, unit="h")
    elif clock == "datetime":
        urine["charttime"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(urine.charttime, unit="h")
    urine.index = np.arange(len(urine)) * 3 + 5
    urine.attrs["published"] = {"source": ["invented"]}
    # Missing, invalid, duplicated-but-identical and wholly absent weights.
    weight = pd.DataFrame({"stay_id": [1, 1, 2, 3, 4, 5, 6],
                           "weight": [80., 80., np.nan, 0., -1., 55., 92.]})
    return urine, weight


@pytest.mark.parametrize("clock", ["hours", "timedelta", "datetime"])
@pytest.mark.parametrize("rate,receipts", [(False, False), (False, True), (True, True)])
def test_b5_descriptive_windows_are_exactly_the_previous_frames(clock, rate, receipts):
    urine, weight = _urine_stays(clock, receipts=receipts)
    args = (urine, weight, [(6, 4.), (12, 8.), (24, 18.)])
    options = {"interval": pd.Timedelta(minutes=30), "source_is_rate": rate}
    expected = previous.descriptive_windows(*args, **options)
    actual = descriptive_windows(*args, **options)
    assert list(actual) == list(expected)
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name], check_exact=True)
        assert actual[name].attrs == expected[name].attrs
    # The 12 h endpoint still contributes coverage, including in a 6 h window.
    assert actual["uo_6h"].loc[1, "uo_6h_covered_h"] == 6.


@pytest.mark.parametrize("clock", ["hours", "minutes", "seconds", "timedelta", "datetime"])
@pytest.mark.parametrize("rate,receipts", [(False, False), (False, True), (True, True)])
def test_b5_kdigo_windows_are_exactly_the_previous_frame(clock, rate, receipts):
    urine, weight = _urine_stays(clock, receipts=receipts)
    args = (urine, weight, "stay_id", "charttime")
    options = {"interval": pd.Timedelta(minutes=30), "source_is_rate": rate, "time_unit": clock}
    expected = previous._calculate_uo_rates_simple(*args, **options)
    actual = _calculate_uo_rates_simple(*args, **options)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs == expected.attrs


@pytest.mark.parametrize("owner", ["descriptive", "kdigo"])
def test_b5_window_outputs_do_not_write_dataframe_cells_per_stay(monkeypatch, owner):
    from pandas.core.indexing import _LocIndexer

    urine, weight = _urine_stays("hours")
    original = _LocIndexer.__setitem__
    writes = []

    def record(self, key, value):
        if isinstance(key, tuple) and isinstance(key[1], str) and key[1].startswith(("uo_", "aki_stage_uo_available")):
            writes.append(key[1])
        return original(self, key, value)

    monkeypatch.setattr(_LocIndexer, "__setitem__", record)
    if owner == "descriptive":
        descriptive_windows(urine, weight, [(6, 4.), (12, 8.), (24, 18.)])
    else:
        _calculate_uo_rates_simple(urine, weight, "stay_id", "charttime", time_unit="hours")
    assert writes == []


def test_b5_three_requested_windows_reuse_the_descriptive_collection_kernel(monkeypatch):
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_uo_window
    from easyicu.scores import urine_output

    urine, weight = _urine_stays("hours")
    tables = {
        "urine": ICUTable(urine, id_columns=["stay_id"], index_column="charttime", value_column="urine"),
        "weight": ICUTable(weight, id_columns=["stay_id"], value_column="weight"),
    }
    source = SimpleNamespace(config=SimpleNamespace(name="miiv"))
    original = urine_output.collection_windows
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(urine_output, "collection_windows", counted)

    def run(resolver):
        frames = {}
        for window in (6, 12, 24):
            name = f"uo_{window}h"
            ctx = ConceptCallbackContext(concept_name=name, patient_ids=None,
                                         interval=pd.Timedelta(hours=1), resolver=resolver,
                                         data_source=source, target=None)
            frames[name] = _callback_uo_window(dict(tables), ctx, window, name).data
        return frames

    expected = run(SimpleNamespace())
    uncached_calls = len(calls)
    calls.clear()
    resolver = ConceptResolver(ConceptDictionary({}))
    actual = run(resolver)
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name], check_exact=True)
        assert actual[name].attrs == expected[name].attrs
    assert len(calls) * 3 == uncached_calls


def test_b5_a_tiny_cache_keeps_single_window_work(monkeypatch):
    from easyicu.concept.callbacks import ConceptCallbackContext, _callback_uo_window
    from easyicu.scores import urine_output

    urine, weight = _urine_stays("hours")
    resolver = ConceptResolver(ConceptDictionary({}))
    resolver._cache_budget_bytes = 1
    tables = {
        "urine": ICUTable(urine, id_columns=["stay_id"], index_column="charttime", value_column="urine"),
        "weight": ICUTable(weight, id_columns=["stay_id"], value_column="weight"),
    }
    ctx = ConceptCallbackContext("uo_6h", None, pd.Timedelta(hours=1), resolver,
                                 SimpleNamespace(config=SimpleNamespace(name="miiv")), None)
    original = urine_output.collection_windows
    calls = []

    def counted(ends, amounts, windows, **kwargs):
        calls.append(tuple(windows))
        return original(ends, amounts, windows, **kwargs)

    monkeypatch.setattr(urine_output, "collection_windows", counted)
    _callback_uo_window(dict(tables), ctx, 6, "uo_6h")
    assert calls and set(calls) == {(6,)}


@pytest.mark.parametrize("change", ["interval", "weight", "cohort", "source_kind", "clear", "budget", "windows"])
def test_b5_shared_descriptive_windows_respect_scope_and_eviction(monkeypatch, change):
    from easyicu.scores import urine_output

    urine, weight = _urine_stays("hours")
    resolver = ConceptResolver(ConceptDictionary({}))
    windows = [(6, 3), (12, 6), (24, 12)]
    options = {"resolver": resolver, "requested_window": 6,
               "interval": pd.Timedelta(hours=1), "source_is_rate": False}
    calls = []
    original = urine_output.descriptive_windows

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(urine_output, "descriptive_windows", counted)
    urine_output.shared_descriptive_windows(urine, weight, windows, **options)
    if change == "interval":
        options["interval"] = pd.Timedelta(hours=2)
    elif change == "weight":
        weight = weight.assign(weight=60.)
    elif change == "cohort":
        urine = urine.loc[urine.stay_id.eq(1)].copy()
    elif change == "source_kind":
        options["source_is_rate"] = True
    elif change == "clear":
        resolver.clear_table_cache()
    elif change == "budget":
        resolver._cache_budget_bytes = 1
    elif change == "windows":
        windows = [(6, 1), (12, 6), (24, 12)]
    actual = urine_output.shared_descriptive_windows(urine, weight, windows, **options)
    expected = previous.descriptive_windows(urine, weight, windows,
                                             interval=options["interval"], source_is_rate=options["source_is_rate"])
    pd.testing.assert_frame_equal(actual["uo_6h"], expected["uo_6h"], check_exact=True)
    assert actual["uo_6h"].attrs == expected["uo_6h"].attrs
    assert len(calls) == 2
