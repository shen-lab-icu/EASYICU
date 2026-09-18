"""Whole-source summary semantics under missing values and bounded reads."""

import threading
import time

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from easyicu.webserver import crossdb_review as c, sources
from easyicu.webserver.app import app
from tests.webserver.test_webserver_workspace_summary import _write_csv_export


def _sources(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "cfg/sources.json")
    monkeypatch.setattr(sources, "_autodiscovered_paths", lambda: [])
    paths = [
        _write_csv_export(tmp_path / name, database=name) for name in ("miiv", "eicu")
    ]
    for i, p in enumerate(paths):
        sources.register_source(str(p), label=p.name, active=i == 0, crossdb=True)
    return paths


@pytest.mark.parametrize(
    "values", [["0", "1", " "], ["yes", "no", " "], ["true", "false", None]]
)
def test_blank_boolean_feature_does_not_break_real_summary(
    tmp_path, monkeypatch, values
):
    paths = _sources(tmp_path, monkeypatch)
    p = paths[0] / "vitals.csv"
    frame = pd.read_csv(p)
    frame["vent_status"] = values
    frame.to_csv(p, index=False)
    response = TestClient(app).post("/api/crossdb-review/summary", json={})
    assert response.status_code == 200, response.text
    data = c._summarize_feature_distribution(pd.Series(values))
    assert data["n"] == 3 and data["non_null"] == 2 and data["kind"] == "numeric"
    assert data["min"] == 0 and data["max"] == 1


@pytest.mark.parametrize("ext", ["csv", "parquet", "xlsx"])
def test_reader_keeps_every_row_or_refuses_budget(tmp_path, monkeypatch, ext):
    p = tmp_path / f"values.{ext}"
    frame = pd.DataFrame({"value": [1, 2, 3, 4, 5], "unused": [99] * 5})
    getattr(frame, "to_" + ("excel" if ext == "xlsx" else ext))(p, index=False)
    monkeypatch.setattr(c, "_FEATURE_BATCH_ROWS", 2)
    actual = c._read_feature_columns(p, ["value"])
    assert pd.to_numeric(actual.value).tolist() == [1, 2, 3, 4, 5]
    assert actual.columns.tolist() == ["value"]
    monkeypatch.setattr(c, "_FEATURE_ROW_LIMIT", 4)
    with pytest.raises(c.CrossdbReviewError) as caught:
        c._read_feature_columns(p, ["value"])
    assert caught.value.detail["error"] == "crossdb_feature_read_budget_exceeded"


def test_excel_reader_keeps_first_sheet_independent_of_saved_active_tab(tmp_path):
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active.append(["value"])
    workbook.active.append([1])
    other = workbook.create_sheet("active_but_not_source")
    other.append(["value"])
    other.append([999])
    workbook.active = 1
    path = tmp_path / "two_sheets.xlsx"
    workbook.save(path)
    workbook.close()
    assert c._read_feature_columns(path, ["value"]).value.tolist() == [1]


def test_one_bad_source_returns_attributable_error_without_partial_comparison(
    tmp_path, monkeypatch
):
    paths = _sources(tmp_path, monkeypatch)
    original = c._read_feature_columns

    def read(path, *a, **kw):
        if path.parent == paths[1]:
            raise OSError("private/path must not appear")
        return original(path, *a, **kw)

    monkeypatch.setattr(c, "_read_feature_columns", read)
    response = TestClient(app).post("/api/crossdb-review/summary", json={})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "crossdb_feature_read_failed"
    assert detail["source"]["label"] == "eicu"
    assert "private/path" not in response.text and "rows" not in detail


def test_sync_deadline_returns_and_bounds_still_blocked_workers(monkeypatch):
    release, started, finished = threading.Event(), threading.Event(), threading.Event()
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(c, "_SYNC_SUMMARY_SLOTS", slots)
    monkeypatch.setattr(c, "_resolve_registered_sources", lambda body: [])

    def work(selected, *, checkpoint):
        started.set()
        try:
            release.wait(5)
            checkpoint()
            pytest.fail("abandoned work must not return success")
        finally:
            finished.set()

    monkeypatch.setattr(c, "_crossdb_review_summary_for_sources", work)
    before = time.monotonic()
    try:
        with pytest.raises(c.CrossdbReviewError) as caught:
            c.crossdb_review_summary({"deadline_seconds": 0.05})
        assert started.is_set()
        assert time.monotonic() - before < 1
        assert caught.value.detail["error"] == c.CrossdbSummaryDeadlineError.code
        with pytest.raises(c.CrossdbReviewError) as busy:
            c.crossdb_review_summary({})
        assert busy.value.detail["error"] == "crossdb_summary_busy"
    finally:
        release.set()
        assert finished.wait(2)
        assert slots.acquire(timeout=2)
        slots.release()


def test_feature_scan_calls_cancel_checkpoint_between_batches(tmp_path, monkeypatch):
    p = tmp_path / "x.csv"
    pd.DataFrame({"x": range(8)}).to_csv(p, index=False)
    monkeypatch.setattr(c, "_FEATURE_BATCH_ROWS", 2)
    seen = []

    def check():
        seen.append(1)
        if len(seen) == 3:
            raise c._CrossdbSummaryAbandoned()

    with pytest.raises(c._CrossdbSummaryAbandoned):
        c._read_feature_columns(p, ["x"], checkpoint=check)
