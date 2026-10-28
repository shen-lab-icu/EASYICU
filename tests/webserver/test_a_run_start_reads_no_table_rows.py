"""A run start lists its source's tables without reading their rows.

Starting a SICdb plan on a converted raw folder took about four minutes
before the run's job existed, so the page showed only "正在启动研究计划生成":
the start described the folder with ``describe_export_source``, which counts
every CSV's rows by reading it through (a 14 GB ``data_float_h.csv``).
"""

from pathlib import Path

import pytest

from easyicu.webserver import dataio, research_launch_scientific, research_run_submission


@pytest.fixture
def raw_folder(tmp_path: Path) -> Path:
    (tmp_path / "cases.csv").write_text("CaseID,AgeOnAdmission\n1,70\n2,55\n", encoding="utf-8")
    (tmp_path / "data_float_h.csv").write_text("CaseID,Val\n1,3.2\n", encoding="utf-8")
    return tmp_path


def _no_rows(_path):
    raise AssertionError("a run start must not count table rows")


def test_source_tables_lists_tables_and_modules_without_rows(raw_folder, monkeypatch):
    monkeypatch.setattr(dataio, "_count_rows", _no_rows)
    monkeypatch.setattr(dataio, "_read_stay_ids", _no_rows)

    described = dataio.describe_source_tables(str(raw_folder))

    assert described["ok"] is True
    assert [item["file"] for item in described["files"]] == ["cases.csv", "data_float_h.csv"]
    assert all("rows" not in item for item in described["files"])
    assert described["files"][0]["columns"] == ["CaseID", "AgeOnAdmission"]


def test_source_tables_keeps_the_describe_refusals(tmp_path):
    assert dataio.describe_source_tables(str(tmp_path / "missing"))["error"] == "not_a_directory"
    assert dataio.describe_source_tables(str(tmp_path))["error"] == "no_export_files"


def test_the_registry_description_still_counts_rows(raw_folder):
    described = dataio.describe_export_source(str(raw_folder))
    assert [item["rows"] for item in described["files"]] == [2, 1]


def test_the_start_and_its_scope_default_use_the_row_free_description():
    # Both start-path owners name the row-free description, never the
    # registry one that reads every CSV through.
    for module in (research_run_submission, research_launch_scientific):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "dataio.describe_source_tables(" in source
        assert "dataio.describe_export_source(" not in source
