import builtins
import json

import pandas as pd
import pytest

import easyicu
from easyicu import api


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


def test_module_extraction_manifest_records_bounds_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_CONCEPT_BOUNDS_CACHE", {"test_signal": (0.0, 10.0)})

    def fake_load_concepts(**kwargs):
        return {
            "test_signal": pd.DataFrame(
                {
                    "stay_id": [1, 1, 1, 1],
                    "charttime": [0, 1, 2, 3],
                    "test_signal": [-1, 1, 2, 99],
                }
            )
        }

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)

    api._run_module_extraction(
        "test_module",
        ["test_signal"],
        "miiv",
        str(tmp_path),
        None,
        None,
        str(tmp_path),
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    saved = manifest["saved"]["test_signal"]

    assert manifest["errors"] == []
    assert manifest["warnings"] == []
    assert saved["rows_before"] == 4
    assert saved["rows"] == 2
    assert saved["bounds_dropped"] is None
    assert saved["bounds_dropped_post_aggregation"] == 2
    assert saved["bounds_count_status"] == "pre_aggregation_count_unavailable"
    assert saved["bounds_skipped"] is False
    assert saved["bounds_status"] == "enforced"


def test_module_extraction_manifest_marks_loader_unit_suspect(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_CONCEPT_BOUNDS_CACHE", {"temperature": (32.0, 42.0)})

    recovered = pd.DataFrame({"stay_id": [1], "charttime": [0], "temperature": [98.6]})
    recovered.attrs["easyicu_bounds_loader"] = {
        "bounds_raw_transformed_non_null": 100,
        "bounds_bounded_transformed_non_null": 0,
        "bounds_bounded_aggregate_non_null": 0,
        "bounds_unit_suspect": True,
        "bounds_unbounded_retry": True,
    }
    monkeypatch.setattr(
        easyicu,
        "load_concepts",
        lambda **kwargs: {"temperature": recovered},
    )

    api._run_module_extraction(
        "vitals",
        ["temperature"],
        "miiv",
        str(tmp_path),
        None,
        None,
        str(tmp_path),
    )

    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    saved = manifest["saved"]["temperature"]
    exported = pd.read_parquet(saved["path"])

    assert exported["temperature"].tolist() == pytest.approx([98.6])
    assert manifest["errors"] == []
    assert manifest["warnings"] == [
        "temperature: BOUNDS SKIPPED (unit-suspect: median outside declared range)"
    ]
    assert saved["bounds_status"] == "skipped_unit_suspect"
    assert saved["bounds_skipped"] is True
    assert saved["bounds_count_status"] == "skipped_unit_suspect"
    assert saved["bounds_dropped"] is None
    assert saved["bounds_raw_transformed_non_null"] == 100
    assert saved["bounds_bounded_transformed_non_null"] == 0
    assert saved["bounds_bounded_aggregate_non_null"] == 0
    assert saved["bounds_unit_suspect"] is True
    assert saved["bounds_unbounded_retry"] is True
