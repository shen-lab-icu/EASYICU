"""A cancelled conversion job stops starting new files.

11-03 external review: the conversion runner never looked at
``cancel_requested``, and ``convert_all`` swallows progress-callback
exceptions. A cancelled job was shown as cancelled while every remaining CSV
kept converting. The converter now checks a stop request before each file;
a stopped run writes no manifest and is not ready for analysis.
"""

from __future__ import annotations

import time

import pytest

from easyicu.io.data_converter import ConversionStopped, DataConverter
from easyicu.webserver import dataio
from easyicu.webserver.jobs import Job


FILES = 6


@pytest.fixture(autouse=True)
def slow_files(monkeypatch):
    # Each file takes long enough that a stop request lands before the
    # workers have drained the queue.
    original = DataConverter._convert_file

    def slow(self, csv_path):
        time.sleep(0.15)
        return original(self, csv_path)

    monkeypatch.setattr(DataConverter, "_convert_file", slow)


def _folder(tmp_path):
    for index in range(FILES):
        (tmp_path / f"table_{index}.csv").write_text("stay_id,value\n1,2\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("workers", [1, 2])
def test_a_stop_request_skips_every_unstarted_file(tmp_path, workers):
    folder = _folder(tmp_path)
    converter = DataConverter(data_path=str(folder), database="miiv", verbose=False)
    converter.parallel_workers = workers
    finished = []

    def progress(info):
        finished.append(info["file"])

    with pytest.raises(ConversionStopped) as stopped:
        converter.convert_all(progress_callback=progress, should_stop=lambda: bool(finished))
    assert 1 <= len(stopped.value.results) < FILES
    assert all(item["status"] == "completed" for item in stopped.value.results.values())
    assert len(list(folder.glob("*.parquet"))) < FILES
    assert not (folder / "conversion_manifest.json").exists()


def test_the_web_job_stops_when_cancel_is_requested(tmp_path):
    folder = _folder(tmp_path)
    job = Job("convert-cancel", "convert")
    original_emit = job.emit

    def emit(event):
        original_emit(event)
        if event.get("type") == "progress":
            job.cancel_requested = True

    job.emit = emit
    result = dataio.make_convert_runner(str(folder), "miiv")(job)
    assert result["cancelled_at"] == "conversion"
    assert result["ready_for_analysis"] is False
    assert result["converted"] < FILES
    assert len(list(folder.glob("*.parquet"))) < FILES


def test_without_a_stop_request_every_file_converts(tmp_path):
    folder = _folder(tmp_path)
    result = dataio.make_convert_runner(str(folder), "miiv")(Job("convert-all", "convert"))
    assert result["converted"] == FILES and "cancelled_at" not in result
