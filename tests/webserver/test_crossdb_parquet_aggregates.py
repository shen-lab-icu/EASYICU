"""Exact whole-parquet feature summaries using synthetic data only."""

import threading
import time

import duckdb
import numpy as np
import pandas as pd
import pytest

from easyicu.webserver import crossdb_review as c
from easyicu.webserver import crossdb_parquet as p


def _describe(path, features):
    return {
        "path": str(path.parent),
        "files": [{"file": path.name, "module": "vitals", "columns": features}],
    }


def test_parquet_summary_aggregates_beyond_pandas_budgets(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.parquet"
    n = 1_000_001
    with duckdb.connect() as conn:
        conn.sql("SELECT i % 101 AS value FROM range(?) t(i)", params=[n]).write_parquet(
            str(path)
        )
    monkeypatch.setattr(c, "_FEATURE_BYTE_LIMIT", 1)

    def no_rows(*args, **kwargs):
        raise AssertionError("Parquet rows must never enter pandas")

    monkeypatch.setattr(pd, "read_parquet", no_rows)
    result = c._source_feature_distributions(_describe(path, ["value"]))
    summary = result["vitals"]["features"][0]
    assert summary["kind"] == "numeric"
    assert summary["n"] == summary["non_null"] == n
    assert (summary["min"], summary["max"]) == (0, 100)
    assert len(summary["points"]) == 32


@pytest.mark.parametrize("values", [
    pytest.param(np.linspace(-20, 73, 201), id="numeric-trim"),
    pytest.param(list(range(26)), id="integer-bin-boundary"),
    pytest.param([1, 9, 2, 8, 3, 7, 4, 6, 5], id="under-ten"),
    pytest.param([0, 1, 2, 3, 4, 5, 6, np.nextafter(1., 0), np.nextafter(1., 2)],
                 id="bin-edges-closed-last"),
    pytest.param(["TRUE", "t", "Yes", "y", "1", "FALSE", "f", "no", "N", "0", " ", None],
                 id="boolean-strings"),
    pytest.param([True, False, None, True], id="native-boolean"),
    pytest.param(["true", "false", "other", "1", "2"], id="boolean-invalid"),
    pytest.param([str(i) for i in range(65)] + ["other"] * 35, id="65-percent"),
    pytest.param([str(i) for i in range(64)] + ["other"] * 36, id="below-65-percent"),
    pytest.param(["2", "3", "4", "5", "a", "b", "c"], id="65-percent-floor"),
    pytest.param(list("lkjihgfedcba") * 2 + ["", " ", None], id="categorical-ties-top8"),
    pytest.param([None, None], id="all-null"),
    pytest.param([np.nan, np.nan], id="all-nan"),
    pytest.param([], id="zero-rows"),
    pytest.param([7], id="single-integer"),
    pytest.param([True], id="single-boolean"),
    pytest.param([3.5] * 12, id="constant-triangle"),
    pytest.param([np.nan, -np.inf, *range(400), np.inf], id="nan-infinities-trim"),
    pytest.param([0] * 98 + [-100, 100], id="reject-constant-trim"),
    pytest.param(["1_0", "0x10", "nan", "3.2", "+4", "\t5\n", ".5"],
                 id="pandas-numeric-grammar"),
])
def test_parquet_payload_matches_pandas(tmp_path, values):
    path = tmp_path / "synthetic ' values.parquet"
    feature = 'quoted " feature'
    pd.DataFrame({feature: values}).to_parquet(path, index=False)
    expected = c._summarize_feature_distribution(pd.read_parquet(path)[feature])
    actual = p.parquet_feature_distributions(path, [feature])[0]
    assert actual == {"feature": feature, **expected}


def test_categories_use_count_then_label_in_both_paths(tmp_path):
    path = tmp_path / "categories.parquet"
    frame = pd.DataFrame({"value": ["z", "b", "a", "b", "a", "", " ", None]})
    frame.to_parquet(path, index=False)
    expected = [
        {"label": "a", "count": 2, "pct": 25.0},
        {"label": "b", "count": 2, "pct": 25.0},
        {"label": "missing", "count": 2, "pct": 25.0},
        {"label": "z", "count": 1, "pct": 12.5},
    ]
    assert p.parquet_feature_distributions(path, ["value"])[0]["categories"] == expected
    assert c._summarize_feature_distribution(frame.value)["categories"] == expected


@pytest.mark.parametrize("values", [
    pytest.param([0, 1, np.inf, -np.inf, np.nan], id="float-infinities"),
    pytest.param(["0", "1", "1e999", "nan"], id="overflowing-decimal-text"),
])
def test_nonfinite_untrimmed_histogram_preserves_failure(tmp_path, values):
    path = tmp_path / "nonfinite.parquet"
    frame = pd.DataFrame({"value": values})
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="not finite"):
        c._summarize_feature_distribution(frame.value)
    with pytest.raises(ValueError, match="not finite"):
        p.parquet_feature_distributions(path, ["value"])


def test_parquet_cannot_enter_row_reader(tmp_path):
    with pytest.raises(ValueError, match="parquet_feature_distributions"):
        c._read_feature_columns(tmp_path / "synthetic.parquet", ["value"])


def _write(path, columns):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    # pyarrow, not pandas, writes the file: no pandas dtype metadata, like a
    # plain export, so pandas reads integers with missing values as float64.
    pq.write_table(pa.table(columns), path)


def _assert_pandas_parity(path, features):
    frame = pd.read_parquet(path)
    actual = p.parquet_feature_distributions(path, features)
    assert [item["feature"] for item in actual] == features
    for item in actual:
        expected = c._summarize_feature_distribution(frame[item["feature"]])
        assert item == {"feature": item["feature"], **expected}, item["feature"]


def test_typed_columns_share_scans_and_match_pandas(tmp_path):
    pa = pytest.importorskip("pyarrow")
    rng = np.random.default_rng(20261022)
    n = 2000
    missing = rng.random(n) < 0.08
    normal = rng.normal(80, 15, n)
    normal[rng.random(n) < 0.05] = np.nan
    skewed = rng.lognormal(1, 1.2, n)
    skewed[:3] = [1e6, -1e5, np.inf]
    single = [None] * n
    single[17] = 7

    def nullable(values, kind):
        return pa.array([None if m else v for v, m in zip(values, missing)], type=kind)

    columns = {
        "dbl_normal": pa.array(normal, mask=missing),
        "dbl_skew_inf_trimmed": pa.array(skewed),
        "int_counts": pa.array(rng.integers(0, 40, n), type=pa.int64()),
        "int_missing": nullable(rng.integers(-5, 300, n).tolist(), pa.int64()),
        "int_bool_like": nullable(rng.integers(0, 2, n).tolist(), pa.int32()),
        "bool_missing": nullable((rng.random(n) < 0.3).tolist(), pa.bool_()),
        "bool_full": pa.array((rng.random(n) < 0.7).tolist()),
        "dbl_constant": nullable([3.5] * n, pa.float64()),
        "dbl_two": pa.array([None] * (n - 2) + [1.0, 2.0], type=pa.float64()),
        "dbl_single": pa.array([None] * (n - 1) + [1e-05], type=pa.float64()),
        "int_single": pa.array(single, type=pa.int64()),
        "bool_single": pa.array([None] * (n - 1) + [True], type=pa.bool_()),
        "dbl_empty": pa.array([None] * n, type=pa.float64()),
        "dbl_edges": pa.array(np.resize(np.linspace(0, 1, 33), n)),
        "dbl_negative_zero": pa.array(np.resize([-0.0, 0.0, 1.5, -2.25], n)),
        "text_beside": pa.array(rng.choice(["a", "b", " ", ""], n).tolist()),
    }
    _write(tmp_path / "typed.parquet", columns)
    _assert_pandas_parity(tmp_path / "typed.parquet", list(reversed(columns)))


@pytest.mark.parametrize("n", [2, 9, 10, 11, 15, 16, 47, 48, 899, 900, 901, 1200])
def test_bin_count_and_trim_boundaries_match_pandas(tmp_path, n):
    rng = np.random.default_rng(n)
    _write(tmp_path / "bins.parquet", {
        "uniform": rng.uniform(-3, 9, n),
        "spiky": np.where(rng.random(n) < 0.97, 1.0, rng.normal(0, 50, n)),
        "integers": rng.integers(0, 6, n),
    })
    _assert_pandas_parity(tmp_path / "bins.parquet", ["uniform", "spiky", "integers"])


def test_repeated_summary_of_an_unchanged_file_is_not_rescanned(tmp_path, monkeypatch):
    path = tmp_path / "cached.parquet"
    _write(path, {"value": np.arange(100, dtype=float)})
    first = p.parquet_feature_distributions(path, ["value"])

    def no_scan(*args, **kwargs):
        raise AssertionError("an unchanged file must come from the cache")

    monkeypatch.setattr(p, "_connection", no_scan)
    assert p.parquet_feature_distributions(path, ["value"]) == first
    first[0]["points"].clear()
    assert p.parquet_feature_distributions(path, ["value"])[0]["points"]

    monkeypatch.undo()
    _write(path, {"value": np.arange(200, dtype=float)})
    assert p.parquet_feature_distributions(path, ["value"])[0]["n"] == 200


def _large_file(tmp_path, rows=3_000_000):
    path = tmp_path / "large.parquet"
    with duckdb.connect() as conn:
        conn.sql(
            "SELECT random() * 100 AS a, (i % 997)::DOUBLE AS b, i % 3 = 0 AS c "
            "FROM range(?) t(i)", params=[rows],
        ).write_parquet(str(path))
    return path


def test_an_exceeded_deadline_is_a_typed_budget_failure_with_its_cause(tmp_path):
    path = _large_file(tmp_path)
    started = time.monotonic()
    with pytest.raises(p.ParquetAggregationError) as caught:
        p.parquet_feature_distributions(path, ["a", "b", "c"], deadline_seconds=0.05)
    assert caught.value.code == "crossdb_feature_read_budget_exceeded"
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert time.monotonic() - started < 5


def test_a_cancel_seen_by_the_watcher_stops_the_summary(tmp_path):
    path = _large_file(tmp_path)

    class Cancelled(Exception):
        pass

    def checkpoint():
        # Only the watcher thread sees the cancel, as when it arrives mid-query.
        if threading.current_thread() is not threading.main_thread():
            raise Cancelled()

    started = time.monotonic()
    with pytest.raises(Cancelled):
        p.parquet_feature_distributions(path, ["a", "b", "c"], checkpoint=checkpoint)
    assert time.monotonic() - started < 5


def test_the_source_reader_keeps_the_typed_code_and_its_cause(tmp_path, monkeypatch):
    path = tmp_path / "values.parquet"
    _write(path, {"value": np.arange(20, dtype=float)})

    def over_budget(*args, **kwargs):
        raise p.ParquetAggregationError("crossdb_feature_read_budget_exceeded") from TimeoutError()

    monkeypatch.setattr(c, "parquet_feature_distributions", over_budget)
    with pytest.raises(c.CrossdbReviewError) as caught:
        c._source_feature_distributions(_describe(path, ["value"]))
    assert caught.value.detail == {
        "error": "crossdb_feature_read_budget_exceeded", "cause_type": "TimeoutError",
    }


@pytest.mark.slow
def test_two_full_size_sources_summarize_within_the_default_deadline(tmp_path):
    """About 15 M rows x 12 columns per source, as a full ICU export module."""
    rows = 15_000_000
    paths = []
    for name in ("first", "second"):
        path = tmp_path / f"{name}.parquet"
        with duckdb.connect() as conn:
            conn.execute("SET threads = 4")
            conn.sql(
                "SELECT " + ", ".join(
                    [f"CASE WHEN i % {7 + k} = 0 THEN NULL ELSE random() * {10 + k} END AS d{k}"
                     for k in range(8)]
                    + [f"i % {k + 2} = 0 AS b{k}" for k in range(3)]
                    + ["CASE i % 4 WHEN 0 THEN 'low' WHEN 1 THEN 'mid' ELSE 'high' END AS t0"]
                ) + " FROM range(?) t(i)", params=[rows],
            ).write_parquet(str(path))
        paths.append(path)
    features = [f"d{k}" for k in range(8)] + [f"b{k}" for k in range(3)] + ["t0"]
    started = time.monotonic()
    for path in paths:
        result = p.parquet_feature_distributions(path, features)
        assert [item["n"] for item in result] == [rows] * len(features)
    assert time.monotonic() - started < 120
