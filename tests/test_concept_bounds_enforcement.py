import builtins
import json

import pandas as pd
import pytest

import easyicu
from easyicu.api import extraction as api


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
    assert calls == [[1, 2], [3, 4], [5]]
    exported = pd.read_parquet(manifest["saved"]["test_module"]["path"])
    assert exported["stay_id"].tolist() == [1, 2, 3, 4, 5]
    assert not (tmp_path / ".test_module.partial.parquet").exists()


def test_stream_batch_release_flushes_arrow_pool():
    released = []

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
        {"stay_id": [1, 1], "charttime": time, "susp_inf": [False, True]}
    ).to_parquet(source / "sepsis_shared.parquet", index=False)
    pd.DataFrame({"stay_id": [1, 1], "charttime": time, "sofa": [0.0, 3.0]}).to_parquet(
        source / "sofa1_score.parquet", index=False
    )
    pd.DataFrame(
        {"stay_id": [1, 1], "charttime": time, "sofa2": [0.0, 3.0]}
    ).to_parquet(source / "sofa2_score.parquet", index=False)

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


def test_streamed_special_export_broadcasts_stay_level_suspicion(tmp_path):
    source = tmp_path / "published"
    output = tmp_path / "special"
    source.mkdir()
    output.mkdir()
    time = pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T01:00:00"])
    pd.DataFrame({"stay_id": [1], "susp_inf": [True]}).to_parquet(
        source / "sepsis_shared.parquet", index=False
    )
    pd.DataFrame({"stay_id": [1, 1], "charttime": time, "sofa": [0.0, 3.0]}).to_parquet(
        source / "sofa1_score.parquet", index=False
    )
    pd.DataFrame(
        {"stay_id": [1, 1], "charttime": time, "sofa2": [0.0, 3.0]}
    ).to_parquet(source / "sofa2_score.parquet", index=False)

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

    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["errors"] == []
    assert set(manifest["saved"]) == {"sep3_sofa1", "sep3_sofa2"}
    assert pd.read_parquet(output / "sep3_sofa1.parquet")["sep3_sofa1"].tolist() == [1]
    assert pd.read_parquet(output / "sep3_sofa2.parquet")["sep3_sofa2"].tolist() == [1]


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
