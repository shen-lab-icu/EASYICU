import builtins
import json
from pathlib import Path

import pandas as pd
import pytest

import easyicu
from easyicu.api import extraction as api


def _write_complete_score_dependencies(source: Path, time) -> None:
    """Write component-complete producer artifacts for streamed Sepsis tests."""

    sofa1_components = list(api._SOFA1_COMPONENT_NAMES)
    sofa1 = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": time,
            **{
                component: [0.0, 3.0 if component == "sofa_resp" else 0.0]
                for component in sofa1_components
            },
        }
    )
    sofa1["sofa"] = sofa1[sofa1_components].sum(axis=1)
    sofa1.to_parquet(source / "sofa1_score.parquet", index=False)

    sofa2_components = list(api.SOFA2_COMPONENT_NAMES)
    sofa2 = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": time,
            **{
                component: [0.0, 3.0 if component == "sofa2_resp" else 0.0]
                for component in sofa2_components
            },
        }
    )
    for component in sofa2_components:
        sofa2[f"{component}_available"] = True
    sofa2["sofa2"] = sofa2[sofa2_components].sum(axis=1)
    sofa2.to_parquet(source / "sofa2_score.parquet", index=False)


def test_enforce_concept_bounds_drops_only_numeric_out_of_range(monkeypatch):
    monkeypatch.setattr(api, "_CONCEPT_BOUNDS_CACHE", {"test_signal": (0.0, 10.0)})
    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 1, 1, 1],
            "test_signal": [-1, 0, 5, 10, 11, None, "not_numeric"],
        }
    )

    filtered, dropped = api._enforce_concept_bounds(df, "test_signal")

    assert dropped == 2
    assert filtered["test_signal"].tolist() == [0, 5, 10, None, "not_numeric"]


def test_enforce_concept_bounds_preserves_unbounded_concepts(monkeypatch):
    monkeypatch.setattr(api, "_CONCEPT_BOUNDS_CACHE", {})
    df = pd.DataFrame({"unbounded": [-999, 1, 999]})

    filtered, dropped = api._enforce_concept_bounds(df, "unbounded")

    assert dropped == 0
    assert filtered.equals(df)


def test_enforce_concept_bounds_skips_unit_suspect_batch(monkeypatch):
    monkeypatch.setattr(api, "_CONCEPT_BOUNDS_CACHE", {"temperature": (32.0, 42.0)})
    df = pd.DataFrame({"temperature": [98.6] * 100})

    filtered, dropped = api._enforce_concept_bounds(df, "temperature")

    assert dropped == -1
    assert filtered.equals(df)


def test_load_concept_bounds_map_warns_when_dictionary_cannot_be_read(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no dictionary")

    monkeypatch.setattr(api, "_CONCEPT_BOUNDS_CACHE", None)
    monkeypatch.setattr(builtins, "open", boom)

    with pytest.warns(RuntimeWarning, match="Could not load concept bounds"):
        assert api._load_concept_bounds_map() == {}


def test_bounds_metadata_helpers_preserve_manifest_fields():
    info = {
        "rows": 2,
        "rows_before": 4,
        "bounds_dropped": None,
        "bounds_dropped_post_aggregation": 2,
        "bounds_count_status": "pre_aggregation_count_unavailable",
        "bounds_skipped": False,
        "bounds_status": "enforced",
    }

    public = api._concept_result_info("/tmp/test.parquet", info)
    df = pd.DataFrame({"test_signal": [1, 2]})
    meta = api._attach_bounds_metadata(df, info)

    assert public == {
        "path": "/tmp/test.parquet",
        "rows": 2,
        "rows_before": 4,
        "bounds_dropped": None,
        "bounds_dropped_post_aggregation": 2,
        "bounds_count_status": "pre_aggregation_count_unavailable",
        "bounds_skipped": False,
        "bounds_status": "enforced",
    }
    assert meta == {
        "rows_before": 4,
        "bounds_dropped": None,
        "bounds_dropped_post_aggregation": 2,
        "bounds_count_status": "pre_aggregation_count_unavailable",
        "bounds_skipped": False,
        "bounds_status": "enforced",
    }
    assert df.attrs["easyicu_bounds"] == meta
    assert df.attrs["easyicu_bounds_dropped"] is None


def test_module_extraction_writes_one_wide_module_file(monkeypatch, tmp_path):
    # Unified with the web export path (dataio.py): _run_module_extraction now calls
    # load_concepts(merge=True) and writes ONE wide {module}.parquet (id + time + one column
    # per concept). Concept-bounds enforcement (incl. unit-suspect skip + unbounded retry)
    # is load_concepts' pre-aggregation filter_bounds — covered by the
    # test_enforce_concept_bounds_* unit tests above; the module exporter no longer runs its
    # own per-concept post-guard, so the mock returns an already-enforced wide frame exactly
    # as real load_concepts would.
    def fake_load_concepts(**kwargs):
        assert kwargs.get("merge") is True
        return pd.DataFrame(
            {"stay_id": [1, 1], "charttime": [1, 2], "test_signal": [1.0, 2.0]}
        )

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)

    api._run_module_extraction(
        "test_module", ["test_signal"], "miiv", str(tmp_path), None, None, str(tmp_path)
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["errors"] == []
    assert manifest["warnings"] == []
    saved = manifest["saved"]["test_module"]
    assert saved["rows"] == 2
    assert saved["concepts"] == ["test_signal"]
    exported = pd.read_parquet(saved["path"])
    assert list(exported.columns) == ["stay_id", "charttime", "test_signal"]
    assert exported["test_signal"].tolist() == [1.0, 2.0]


def test_module_extraction_adds_unavailable_concept_as_arrow_null(
    monkeypatch,
    tmp_path,
) -> None:
    """Structural placeholders belong in Arrow, not dense pandas blocks."""
    import pyarrow.parquet as pq

    def fake_load_concepts(**kwargs):
        assert kwargs["_defer_empty_columns_to_arrow"] is True
        return pd.DataFrame(
            {
                "stay_id": [1, 2],
                "charttime": [0.0, 1.0],
                "test_signal": [1.0, 2.0],
            }
        )

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)

    api._run_module_extraction(
        "test_module",
        ["test_signal", "optional_signal"],
        "miiv",
        str(tmp_path),
        None,
        None,
        str(tmp_path),
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    saved = manifest["saved"]["test_module"]
    table = pq.read_table(saved["path"])
    assert manifest["errors"] == []
    assert saved["concepts"] == ["test_signal"]
    assert table.column_names == [
        "stay_id",
        "charttime",
        "test_signal",
        "optional_signal",
    ]
    assert str(table.schema.field("optional_signal").type) == "double"
    assert table["optional_signal"].null_count == table.num_rows
    assert "peak_rss_mb" in manifest
    assert "peak_working_set_mb" in manifest


def test_module_extraction_sanitises_mixed_object_columns(monkeypatch, tmp_path):
    # Indicator concepts (e.g. mech_circ_support) can return object dtype mixing bool/float/
    # NaN, which pyarrow refuses to write to parquet. The exporter losslessly coerces such
    # object columns to numeric before writing; genuine string columns (sex) stay untouched.
    wide = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "charttime": [0, 0, 0],
            "mech_circ_support": pd.Series([True, 1.0, None], dtype=object),
            "sex": pd.Series(["M", "F", None], dtype=object),
        }
    )
    monkeypatch.setattr(easyicu, "load_concepts", lambda **kwargs: wide)

    api._run_module_extraction(
        "circulatory",
        ["mech_circ_support", "sex"],
        "miiv",
        str(tmp_path),
        None,
        None,
        str(tmp_path),
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["errors"] == []
    exported = pd.read_parquet(manifest["saved"]["circulatory"]["path"])
    # mixed bool/float object column -> numeric (True->1.0, 1.0->1.0, None->NaN)
    assert exported["mech_circ_support"].tolist()[:2] == [1.0, 1.0]
    # genuine string column left as-is
    assert exported["sex"].tolist()[:2] == ["M", "F"]


def test_module_extraction_streams_patient_batches_directly_to_parquet(
    monkeypatch, tmp_path
):
    calls = []

    def fake_load_concepts(**kwargs):
        ids = list(kwargs["patient_ids"]["stay_id"])
        calls.append(ids)
        return pd.DataFrame(
            {
                "stay_id": ids,
                "charttime": [0] * len(ids),
                "test_signal": [float(value) for value in ids],
            }
        )

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)

    api._run_module_extraction(
        "test_module",
        ["test_signal"],
        "miiv",
        str(tmp_path),
        {"stay_id": [1, 2, 3, 4, 5]},
        2,
        str(tmp_path),
        stream_output_batches=True,
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["errors"] == []
    assert calls == [[1, 4], [2, 5], [3]]
    assert manifest["patient_partition_strategy"] == (
        "source_order_interleaved_v1"
    )
    assert manifest["initial_planned_partition_count"] == 3
    exported = pd.read_parquet(manifest["saved"]["test_module"]["path"])
    assert exported["stay_id"].tolist() == [1, 4, 2, 5, 3]
    assert not (tmp_path / ".test_module.partial.parquet").exists()


def test_streamed_module_rejects_rows_outside_the_requested_patient_batch(
    monkeypatch, tmp_path
):
    def fake_load_concepts(**kwargs):
        ids = list(kwargs["patient_ids"]["stay_id"])
        return pd.DataFrame(
            {
                "stay_id": [*ids, 999],
                "charttime": [0.0] * (len(ids) + 1),
                "test_signal": [1.0] * (len(ids) + 1),
            }
        )

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)

    api._run_module_extraction(
        "test_module",
        ["test_signal"],
        "miiv",
        str(tmp_path),
        {"stay_id": [1, 2]},
        2,
        str(tmp_path),
        stream_output_batches=True,
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["saved"] == {}
    assert manifest["errors"] == [
        "streamed export(test_module): test_module: streamed batch returned "
        "1 stay_id values outside the requested patient partition"
    ]
    assert not (tmp_path / ".test_module.partial.parquet").exists()


def test_streamed_module_grows_later_batches_from_first_measured_peak(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FixedMemorySampler:
        def start(self):
            return self

        def stop(self):
            return {
                "start_rss_mb": 100.0,
                "peak_rss_mb": 2_100.0,
                "peak_working_set_mb": 2_000.0,
                "available_memory_mb_at_start": 8 * 1024.0,
            }

    def fake_load_concepts(**kwargs):
        ids = list(kwargs["patient_ids"]["stay_id"])
        calls.append((len(ids), kwargs["batch_size"]))
        return pd.DataFrame(
            {
                "stay_id": ids,
                "charttime": [0.0] * len(ids),
                "test_signal": pd.Series(ids, dtype="float32"),
            }
        )

    monkeypatch.setattr(api, "_RSSPeakSampler", FixedMemorySampler)
    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)

    api._run_module_extraction(
        "test_module",
        ["test_signal"],
        "eicu",
        str(tmp_path),
        {"stay_id": list(range(120_000))},
        40_000,
        str(tmp_path),
        stream_output_batches=True,
        adaptive_stream_batches=True,
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["errors"] == []
    assert calls == [
        (40_000, 40_000),
        (67_000, 67_000),
        (13_000, 13_000),
    ]
    assert manifest["initial_batch_size"] == 40_000
    assert manifest["final_planned_batch_size"] == 67_000
    assert manifest["adaptive_batch_growth"] is True
    assert [
        batch["stays"] for batch in manifest["stream_batches"]
    ] == [40_000, 67_000, 13_000]
    assert [
        batch["inner_load_batch_size"]
        for batch in manifest["stream_batches"]
    ] == [40_000, 67_000, 13_000]


def test_streamed_module_preserves_first_schema_without_pandas_reindex(
    monkeypatch, tmp_path
):
    def fake_load_concepts(**kwargs):
        ids = list(kwargs["patient_ids"]["stay_id"])
        frame = pd.DataFrame(
            {
                "stay_id": ids,
                "charttime": [0.0] * len(ids),
                "test_signal": [float(value) for value in ids],
            }
        )
        if ids[0] == 1:
            frame["optional_signal"] = [10.0] * len(ids)
        return frame

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)

    api._run_module_extraction(
        "test_module",
        ["test_signal", "optional_signal"],
        "miiv",
        str(tmp_path),
        {"stay_id": [1, 2, 3]},
        2,
        str(tmp_path),
        stream_output_batches=True,
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["errors"] == []
    exported = pd.read_parquet(manifest["saved"]["test_module"]["path"])
    assert list(exported.columns) == [
        "stay_id",
        "charttime",
        "test_signal",
        "optional_signal",
    ]
    assert exported["optional_signal"].tolist()[:2] == [10.0, 10.0]
    assert pd.isna(exported["optional_signal"].iloc[2])


def test_streamed_module_keeps_later_charttime_when_first_batch_has_none(
    monkeypatch, tmp_path
) -> None:
    def fake_load_concepts(**kwargs):
        ids = list(kwargs["patient_ids"]["stay_id"])
        if ids[0] == 1:
            # This reproduces eICU sepsis_shared when the first stay batch has
            # no timestamped sampling event at all.
            return pd.DataFrame(
                {
                    "stay_id": ids,
                    "susp_inf": [False] * len(ids),
                }
            )
        return pd.DataFrame(
            {
                "stay_id": ids,
                "charttime": [5.0] * len(ids),
                "susp_inf": [True] * len(ids),
            }
        )

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)

    api._run_module_extraction(
        "sepsis_shared",
        ["susp_inf"],
        "eicu",
        str(tmp_path),
        {"stay_id": [1, 2, 3]},
        2,
        str(tmp_path),
        stream_output_batches=True,
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["errors"] == []
    exported = pd.read_parquet(manifest["saved"]["sepsis_shared"]["path"])
    assert list(exported.columns) == ["stay_id", "charttime", "susp_inf"]
    assert exported["charttime"].dtype == "float64"
    assert exported["charttime"].iloc[:2].isna().all()
    assert exported["charttime"].iloc[2] == 5.0


@pytest.mark.parametrize("database", ["eicu", "aumc"])
def test_isolated_stream_batches_preserve_output_and_remove_parts(
    monkeypatch,
    tmp_path,
    database,
) -> None:
    calls = []
    daemon_flags = []
    events = []
    original_append = api._append_isolated_stream_batch

    def tracked_append(*args, **kwargs):
        events.append("append")
        return original_append(*args, **kwargs)

    def fake_load_concepts(**kwargs):
        ids = list(kwargs["patient_ids"]["stay_id"])
        calls.append(ids)
        events.append("extract")
        return pd.DataFrame(
            {
                "stay_id": ids,
                "charttime": [0.0] * len(ids),
                "test_signal": [float(value) for value in ids],
            }
        )

    class InlineProcess:
        def __init__(self, *, target, args, daemon):
            self.target = target
            self.args = args
            self.exitcode = None
            daemon_flags.append(daemon)

        def start(self):
            self.target(*self.args)
            self.exitcode = 0

        def join(self):
            return None

    class InlineContext:
        Process = InlineProcess

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)
    monkeypatch.setattr(api, "_append_isolated_stream_batch", tracked_append)
    monkeypatch.setattr(api, "_extract_worker_env_setup", lambda _path: None)
    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: InlineContext())
    monkeypatch.setattr(
        api,
        "_ISOLATED_STREAM_BATCH_TARGETS",
        {(database, "test_module")},
    )
    monkeypatch.setattr(api, "_DEFERRED_STREAM_MERGE_TARGETS", {("aumc", "test_module")})

    api._run_module_extraction(
        "test_module",
        ["test_signal"],
        database,
        str(tmp_path),
        {"stay_id": [1, 2, 3, 4]},
        2,
        str(tmp_path),
        stream_output_batches=True,
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    exported = pd.read_parquet(manifest["saved"]["test_module"]["path"])
    assert manifest["errors"] == []
    assert manifest["batch_process_isolation"] is True
    assert manifest["deferred_batch_merge"] is (database == "aumc")
    assert events == (
        ["extract", "extract", "append", "append"] if database == "aumc"
        else ["extract", "append", "extract", "append"]
    )
    assert calls == [[1, 3], [2, 4]]
    assert daemon_flags == [False, False]
    assert exported["stay_id"].tolist() == [1, 3, 2, 4]
    assert exported["test_signal"].tolist() == [1.0, 3.0, 2.0, 4.0]
    assert not list(tmp_path.glob(".test_module.batch-*.parquet"))
    assert not (tmp_path / ".test_module.partial.parquet").exists()


def test_batch_process_isolation_is_scoped_to_measured_target(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "_ISOLATED_STREAM_BATCH_TARGETS",
        {("eicu", "sofa2_score")},
    )

    assert api._requires_isolated_stream_batch("eicu", "sofa2_score") is True
    assert api._requires_isolated_stream_batch("eicu_demo", "sofa2_score") is False
    assert api._requires_isolated_stream_batch("mimic", "sofa2_score") is False
    assert api._requires_isolated_stream_batch("eicu", "sofa1_score") is False


def test_aumc_respiratory_uses_measured_batch_process_isolation() -> None:
    assert api._requires_isolated_stream_batch("aumc", "respiratory") is True
    assert api._requires_isolated_stream_batch("aumc", "ventilator") is False
    assert api._requires_isolated_stream_batch("aumc", "other_scores") is False


def test_append_isolated_stream_batch_aligns_to_frozen_schema(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    destination = tmp_path / "destination.parquet"
    schema = pa.schema(
        [
            pa.field("stay_id", pa.int64()),
            pa.field("charttime", pa.float64()),
            pa.field("test_signal", pa.float64()),
            pa.field("first_batch_context", pa.float64()),
        ]
    )
    pq.write_table(
        pa.table(
            {
                "stay_id": pa.array([2], type=pa.int64()),
                "charttime": pa.array([1.0], type=pa.float64()),
                "test_signal": pa.array([2.0], type=pa.float64()),
                "later_only_context": pa.array(["ignored"]),
            }
        ),
        source,
    )
    writer = pq.ParquetWriter(destination, schema)
    try:
        rows = api._append_isolated_stream_batch(
            source,
            writer=writer,
            schema=schema,
            pyarrow_module=pa,
            parquet_module=pq,
        )
    finally:
        writer.close()

    result = pq.read_table(destination)
    assert rows == 1
    assert result.schema == schema
    assert result.column_names == schema.names
    assert result["first_batch_context"].null_count == 1


def test_isolated_stream_batch_failure_removes_atomic_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    class FailedProcess:
        def __init__(self, *, target, args, daemon):
            self.destination = Path(args[-1])
            self.exitcode = None

        def start(self):
            self.destination.write_text("failed child residue")
            self.exitcode = 1

        def join(self):
            return None

    class FailedContext:
        Process = FailedProcess

    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: FailedContext())
    monkeypatch.setattr(
        api,
        "_ISOLATED_STREAM_BATCH_TARGETS",
        {("eicu", "test_module")},
    )

    api._run_module_extraction(
        "test_module",
        ["test_signal"],
        "eicu",
        str(tmp_path),
        {"stay_id": [1, 2]},
        2,
        str(tmp_path),
        stream_output_batches=True,
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["saved"] == {}
    assert "isolated batch worker exited with code 1" in manifest["errors"][0]
    assert not list(tmp_path.glob(".test_module.batch-*.parquet"))
    assert not (tmp_path / ".test_module.partial.parquet").exists()


def test_stream_batch_release_flushes_duckdb_and_arrow_pool(monkeypatch):
    from easyicu import datasource

    released = []
    closed = []
    monkeypatch.setattr(
        datasource,
        "_close_duckdb_connections",
        lambda: closed.append(True),
    )

    class Pool:
        def release_unused(self):
            released.append(True)

    class FakePyArrow:
        @staticmethod
        def default_memory_pool():
            return Pool()

    api._release_stream_batch_memory(
        FakePyArrow,
        trim_native_allocator=False,
    )

    assert closed == [True]
    assert released == [True]


def test_stream_batch_clear_releases_implicit_global_loader(monkeypatch):
    from easyicu.api import concepts as concept_api

    calls = []
    monkeypatch.setattr(
        concept_api,
        "clear_global_loader",
        lambda: calls.append("clear_global_loader"),
    )

    api._clear_stream_loader_caches(None)

    assert calls == ["clear_global_loader"]


def test_streamed_vitals_loads_recursive_concepts_separately():
    calls = []

    def fake_load_concepts(**kwargs):
        concepts = list(kwargs["concepts"])
        calls.append(concepts)
        if concepts == ["hr"]:
            return pd.DataFrame(
                {
                    "stay_id": [1, 1],
                    "charttime": [0.0, 1.0],
                    "hr": [80.0, 90.0],
                }
            )
        concept = concepts[0]
        return pd.DataFrame(
            {
                "stay_id": [1],
                "charttime": [1.0],
                concept: [2.0],
            }
        )

    class Pool:
        def release_unused(self):
            return None

    class FakePyArrow:
        @staticmethod
        def default_memory_pool():
            return Pool()

    result = api._load_stream_module_batch(
        fake_load_concepts,
        module_name="vitals",
        concepts=["hr", "pulse_pressure", "shock_index"],
        load_kwargs={"concepts": ["hr", "pulse_pressure", "shock_index"]},
        patient_ids={"stay_id": [1]},
        loader=None,
        pyarrow_module=FakePyArrow,
    )

    assert calls == [["hr"], ["pulse_pressure"], ["shock_index"]]
    assert result["hr"].tolist() == [80.0, 90.0]
    assert result["pulse_pressure"].tolist()[1] == 2.0
    assert result["shock_index"].tolist()[1] == 2.0


def test_streamed_special_export_uses_published_dependency_parquets(tmp_path):
    source = tmp_path / "published"
    output = tmp_path / "special"
    source.mkdir()
    output.mkdir()
    time = pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T01:00:00"])
    pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [time[0], time[1], pd.NaT],
            "susp_inf": pd.Series([False, True, pd.NA], dtype="boolean"),
            # infection_icd is an explicitly stay-level support field. Its
            # null-time row must not be mistaken for an untimed SI event.
            "infection_icd": pd.Series([True, True, True], dtype="boolean"),
        }
    ).to_parquet(source / "sepsis_shared.parquet", index=False)
    _write_complete_score_dependencies(source, time)

    api._stream_special_extraction_batches(
        ["sepsis3_sofa1", "sepsis3_sofa2"],
        "miiv",
        str(tmp_path),
        {"stay_id": [1]},
        1,
        str(output),
        use_sofa2=True,
        published_output_dir=str(source),
    )

    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["errors"] == []
    assert set(manifest["saved"]) == {"sep3_sofa1", "sep3_sofa2"}
    assert (output / "sep3_sofa1.parquet").is_file()
    assert (output / "sep3_sofa2.parquet").is_file()


def test_streamed_special_export_adapts_sealed_canonical_stay_id(tmp_path):
    """A sealed dependency can be joined to a raw-native eICU score artifact."""

    source = tmp_path / "published"
    output = tmp_path / "special"
    source.mkdir()
    output.mkdir()
    time = pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T01:00:00"])
    pd.DataFrame(
        {
            # Selected refresh stages this file from native-v2, where every
            # database has already been normalized to the public stay_id.
            "stay_id": [1, 1],
            "charttime": time,
            "susp_inf": pd.Series([False, True], dtype="boolean"),
        }
    ).to_parquet(source / "sepsis_shared.parquet", index=False)
    _write_complete_score_dependencies(source, time)
    score_path = source / "sofa2_score.parquet"
    score = pd.read_parquet(score_path).rename(
        columns={"stay_id": "patientunitstayid"}
    )
    score.to_parquet(score_path, index=False)

    api._stream_special_extraction_batches(
        ["sepsis3_sofa2"],
        "eicu",
        str(tmp_path),
        {"patientunitstayid": [1]},
        1,
        str(output),
        use_sofa2=True,
        published_output_dir=str(source),
    )

    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["errors"] == []
    assert set(manifest["saved"]) == {"sep3_sofa2"}
    exported = pd.read_parquet(output / "sep3_sofa2.parquet")
    assert "patientunitstayid" in exported.columns
    assert exported["patientunitstayid"].eq(1).all()


def test_streamed_special_export_rejects_positive_null_time_suspicion(tmp_path):
    source = tmp_path / "published"
    output = tmp_path / "special"
    source.mkdir()
    output.mkdir()
    time = pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T01:00:00"])
    pd.DataFrame(
        {"stay_id": [1], "charttime": [None], "susp_inf": [True]}
    ).to_parquet(source / "sepsis_shared.parquet", index=False)
    _write_complete_score_dependencies(source, time)

    with pytest.raises(
        ValueError,
        match="positive susp_inf rows with null charttime",
    ):
        api._stream_special_extraction_batches(
            ["sepsis3_sofa1", "sepsis3_sofa2"],
            "eicu",
            str(tmp_path),
            {"stay_id": [1]},
            1,
            str(output),
            use_sofa2=True,
            published_output_dir=str(source),
        )

    assert not (output / "sep3_sofa1.parquet").exists()
    assert not (output / "sep3_sofa2.parquet").exists()


def test_streamed_special_export_accepts_declared_empty_infection_dependency(
    tmp_path,
):
    source = tmp_path / "published"
    output = tmp_path / "special"
    source.mkdir()
    output.mkdir()
    (source / "sepsis_shared.manifest.json").write_text(
        json.dumps(
            {
                "module": "sepsis_shared",
                "saved": {},
                "errors": [],
                "warnings": [],
            }
        )
    )

    api._stream_special_extraction_batches(
        ["sepsis3_sofa1", "sepsis3_sofa2"],
        "sic",
        str(tmp_path),
        {"stay_id": [1, 2]},
        2_000,
        str(output),
        use_sofa2=True,
        published_output_dir=str(source),
    )

    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["errors"] == []
    assert manifest["saved"] == {}
    assert not (output / "sep3_sofa1.parquet").exists()
    assert not (output / "sep3_sofa2.parquet").exists()


def test_streamed_special_export_skips_score_reads_without_positive_infection(
    tmp_path,
) -> None:
    """A negative SI batch cannot yield Sepsis and needs no score scan."""

    source = tmp_path / "published"
    output = tmp_path / "special"
    source.mkdir()
    output.mkdir()
    pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [0.0, 0.0],
            "susp_inf": pd.Series([False, False], dtype="boolean"),
        }
    ).to_parquet(source / "sepsis_shared.parquet", index=False)

    api._stream_special_extraction_batches(
        ["sepsis3_sofa1", "sepsis3_sofa2"],
        "eicu",
        str(tmp_path),
        {"stay_id": [1, 2]},
        2,
        str(output),
        use_sofa2=True,
        published_output_dir=str(source),
    )

    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["errors"] == []
    assert manifest["saved"] == {}


def test_streamed_special_export_uses_outer_batch_instead_of_fixed_2000(
    tmp_path,
) -> None:
    source = tmp_path / "published"
    output = tmp_path / "special"
    source.mkdir()
    output.mkdir()
    (source / "sepsis_shared.manifest.json").write_text(
        json.dumps(
            {
                "module": "sepsis_shared",
                "saved": {},
                "errors": [],
                "warnings": [],
            }
        )
    )
    stay_ids = list(range(2_501))

    api._stream_special_extraction_batches(
        ["sepsis3_sofa1", "sepsis3_sofa2"],
        "eicu",
        str(tmp_path),
        {"stay_id": stay_ids},
        len(stay_ids),
        str(output),
        use_sofa2=True,
        published_output_dir=str(source),
    )

    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["errors"] == []
    assert manifest["batch_size"] == 2_501
    assert manifest["batch_count"] == 1


def test_nonstream_special_export_reuses_already_published_scores(
    tmp_path, monkeypatch
):
    source = tmp_path / "published"
    output = tmp_path / "special"
    source.mkdir()
    output.mkdir()
    for module in ("sepsis_shared", "sofa1_score", "sofa2_score"):
        (source / f"{module}.parquet").touch()
    calls = []

    def fake_stream(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(api, "_stream_special_extraction_batches", fake_stream)

    api._run_special_extraction(
        ["sepsis3_sofa1", "sepsis3_sofa2"],
        "miiv",
        str(tmp_path),
        {"stay_id": [1, 2]},
        2_000_000,
        str(output),
        use_sofa2=True,
        stream_output_batches=False,
        published_output_dir=str(source),
    )

    assert len(calls) == 1
    assert calls[0][0][3] == {"stay_id": [1, 2]}
    assert calls[0][0][4] == 2_000_000
    assert calls[0][1]["published_output_dir"] == str(source)


def test_special_sofa1_dependency_collapses_components_before_total() -> None:
    """Same-time component maxima must prevent a false SOFA delta."""

    components = list(api._SOFA1_COMPONENT_NAMES)
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [0.0, 0.0, 1.0],
            **{component: [0.0, 0.0, 0.0] for component in components},
        }
    )
    frame.loc[0, "sofa_resp"] = 4.0
    frame.loc[1, "sofa_coag"] = 4.0
    frame.loc[2, "sofa_resp"] = 4.0
    frame.loc[2, "sofa_coag"] = 2.0

    result = api._consolidate_special_score_dependency(
        frame,
        score_name="sofa",
        id_col="stay_id",
        time_col="charttime",
    )

    assert result["sofa"].tolist() == [8.0, 6.0]


def test_special_sofa2_dependency_respects_component_availability() -> None:
    """Unavailable values cannot enter a total; evidence may unite by hour."""

    components = list(api.SOFA2_COMPONENT_NAMES)
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [0.0, 0.0, 1.0],
            **{component: [0.0, 0.0, 0.0] for component in components},
        }
    )
    for component in components:
        frame[f"{component}_available"] = True
    frame.loc[0, "sofa2_resp"] = 4.0
    frame.loc[1, "sofa2_coag"] = 3.0
    frame.loc[0, "sofa2_coag_available"] = False
    frame.loc[1, "sofa2_resp_available"] = False
    frame.loc[2, "sofa2_renal_available"] = False

    result = api._consolidate_special_score_dependency(
        frame,
        score_name="sofa2",
        id_col="stay_id",
        time_col="charttime",
    )

    assert result.loc[result["charttime"].eq(0.0), "sofa2"].item() == 7.0
    assert result.loc[result["charttime"].eq(1.0), "sofa2"].item() == 0.0
