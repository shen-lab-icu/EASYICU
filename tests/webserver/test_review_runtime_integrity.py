"""Offline regressions for scan, conversion receipts and bounded job replay."""

import json
import pandas as pd
import pytest

from easyicu.webserver import dataio
from easyicu.webserver.jobs import Job


@pytest.mark.parametrize("manifest", ["{broken", "{}", '{"database":"miiv","files":[]}'])
def test_manifest_filename_does_not_establish_readiness(tmp_path, monkeypatch, manifest):
    monkeypatch.setattr(dataio, "_detect_database", lambda path: "miiv")
    (tmp_path / "_manifest.json").write_text(manifest)
    result = dataio.scan_path(str(tmp_path))
    assert result["ready"] is False
    assert result["ok"] is False


def test_manifest_with_missing_member_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(dataio, "_detect_database", lambda path: "miiv")
    (tmp_path / "_manifest.json").write_text(json.dumps({"database": "miiv", "format": "parquet",
        "files": [{"file": "missing.parquet", "rows": 2}]}))
    assert dataio.scan_path(str(tmp_path))["ready"] is False


def test_scan_rejects_conversion_with_dropped_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(dataio, "_detect_database", lambda path: "miiv")
    monkeypatch.setattr(dataio, "_check_data_status", lambda *a: {
        "parquet_count": 3, "csv_count": 0, "ready": True, "missing_tables": []})
    (tmp_path / ".easyicu_conversion_status.json").write_text(json.dumps({
        "patients.csv": {"status": "completed", "bad_rows_skipped": 3}}))
    result = dataio.scan_path(str(tmp_path))
    assert result["ready"] is False
    assert result["conversion_quality"]["bad_rows_skipped"] == 3


@pytest.mark.parametrize("status,emit", [("completed", True), ("skipped", False)])
def test_bad_rows_survive_conversion_job_and_cached_replay(monkeypatch, status, emit):
    from easyicu.io import data_converter
    class FakeConverter:
        def __init__(self, **kw):
            pass
        def convert_all(self, *, force, progress_callback):
            res = {"status": status, "bad_rows_skipped": 7, "row_count": 9}
            if emit:
                progress_callback({"status": status, "file": "synthetic.csv", "result": res})
            return {"synthetic.csv": res}
    monkeypatch.setattr(data_converter, "DataConverter", FakeConverter)
    job = Job("test", "convert")
    result = dataio.make_convert_runner("/unused", "miiv")(job)
    assert result["bad_rows_skipped"] == 7
    assert result["data_quality_status"] == "partial"
    assert result["ready_for_analysis"] is False
    if emit:
        assert job.events[-1]["bad_rows_skipped"] == 7


def test_real_malformed_csv_warning_survives_cached_web_conversion(tmp_path):
    (tmp_path / "sample.csv").write_text(
        "stay_id,value\n1,3\n2,4,extra\n3,5\n", encoding="utf-8"
    )
    runner = dataio.make_convert_runner(str(tmp_path), "miiv")
    first = runner(Job("raw", "convert"))
    second = runner(Job("cached", "convert"))
    assert first["converted"] == 1 and second["skipped"] == 1
    for result in (first, second):
        assert result["bad_rows_skipped"] == 1
        assert result["data_quality_status"] == "partial"
        assert result["ready_for_analysis"] is False
    assert len(pd.read_parquet(tmp_path / "sample.parquet")) == 2
    manifest = json.loads((tmp_path / "conversion_manifest.json").read_text())
    assert manifest["tables"][0]["bad_rows_skipped"] == 1
    assert manifest["tables"][0]["data_quality_status"] == "partial"


@pytest.mark.parametrize("change", ["format", "schema", "rows", "v2_descriptor"])
def test_module_scan_rejects_invalid_manifest_metadata(tmp_path, monkeypatch, change):
    from easyicu.research_agent.intake.export_package import NATIVE_MANIFEST_SCHEMA_V2
    monkeypatch.setattr(dataio, "_detect_database", lambda path: "miiv")
    pd.DataFrame({"stay_id": [1], "value": [3]}).to_parquet(tmp_path / "sample.parquet")
    manifest = {"database": "miiv", "format": "parquet", "files": [
        {"file": "sample.parquet", "rows": 1}]}
    if change == "format":
        manifest["format"] = "csv"
    elif change == "schema":
        manifest["schema_version"] = "unsupported"
    elif change == "rows":
        manifest["files"][0]["rows"] = 2
    else:
        manifest.update(schema_version=NATIVE_MANIFEST_SCHEMA_V2, column_metadata={})
    (tmp_path / "_manifest.json").write_text(json.dumps(manifest))
    assert dataio.scan_path(str(tmp_path))["ready"] is False


def test_job_history_has_bounded_memory_and_absolute_resume_cursor():
    job = Job("stress", "test", max_events=5, max_history_bytes=2048)
    for i in range(1000):
        job.emit({"type": "progress", "current": i})
    job.finish("done", {"ok": True})
    assert len(job.events) <= 5
    replay, status = job.events_since(0)
    assert status == "done"
    assert replay[0]["history_truncated"] is True
    assert replay[-1]["type"] == "end"
    assert replay[-1]["seq"] == 1000
    assert job.events_since(replay[-1]["seq"] + 1)[0] == []
    assert job.snapshot()["events_dropped"] >= 996


def test_oversized_event_is_explicit_and_cannot_bypass_history_budget():
    job = Job("large", "test", max_history_bytes=2048, max_event_bytes=1024)
    job.emit({"type": "progress", "text": "x" * 100000})
    assert len(json.dumps(job.events).encode()) < 2048
    assert job.events[0]["payload_omitted"] is True
    job.finish("done", {"artifact": "x" * 100000})
    assert job.events[-1]["type"] == "end"
    assert job.events[-1]["result_omitted"] is True
    assert job.snapshot()["result"]["artifact"] == "x" * 100000


def test_incomplete_conversion_is_blocked_by_host_before_export(tmp_path):
    (tmp_path / ".easyicu_conversion_status.json").write_text(json.dumps({
        "source.csv": {"status": "completed", "bad_rows_skipped": 1}}))
    output = tmp_path / "never-created"
    with pytest.raises(dataio.ExportCohortError) as caught:
        dataio.make_export_runner(str(tmp_path), "miiv", out_dir=str(output))(Job("export", "test"))
    assert str(caught.value) == "source_conversion_quality_incomplete"
    assert not output.exists()


def test_sse_tail_uses_absolute_sequence_after_history_eviction(monkeypatch):
    import asyncio
    from easyicu.webserver.routes import jobs as routes
    job = Job("sse", "test", max_events=2)
    for i in range(6):
        job.emit({"type": "progress", "current": i})
    monkeypatch.setattr(routes.job_store.MANAGER, "get", lambda key: job)
    async def consume():
        response = await routes.jobs_events("sse")
        iterator = response.body_iterator
        initial = [await anext(iterator) for _ in range(3)]  # gap plus retained 4, 5
        job.finish("done", {"ok": True})
        tail = [value async for value in iterator]
        return initial, tail
    initial, tail = asyncio.run(consume())
    assert '"history_truncated": true' in initial[0]
    assert len(tail) == 1 and '"seq": 6' in tail[0]
    assert '"type": "end"' in tail[0]


@pytest.mark.parametrize("status", ["done", "failed"])
def test_sse_delivers_full_terminal_result_despite_bounded_history(monkeypatch, status):
    import asyncio
    from easyicu.webserver.routes import jobs as routes
    job = Job("large-sse", "test", max_event_bytes=1024)
    result = {"artifact": "x" * 100000}
    error = "synthetic failure" if status == "failed" else None
    job.finish(status, result, error)
    assert job.events[-1]["result_omitted"] is True
    monkeypatch.setattr(routes.job_store.MANAGER, "get", lambda key: job)

    async def consume():
        response = await routes.jobs_events(job.id)
        return [value async for value in response.body_iterator]

    events = [json.loads(value.removeprefix("data: ")) for value in asyncio.run(consume())]
    assert events[-1]["type"] == "end"
    assert events[-1]["result"] == result
    assert events[-1]["error"] == error
    assert not events[-1].get("result_omitted")
    assert job.events[-1]["result_omitted"] is True  # replay does not inflate storage


def test_shutdown_invokes_registered_runner_interrupts():
    from easyicu.webserver.jobs import JobManager
    manager = JobManager()
    job = Job("cleanup", "test")
    manager._jobs[job.id] = job
    calls = []
    job.register_cancel_callback(lambda: calls.append("cleanup"))
    manager.cancel_all()
    manager.cancel_all()
    assert calls == ["cleanup"]
    assert job.cancel_reason == "server_shutdown"
