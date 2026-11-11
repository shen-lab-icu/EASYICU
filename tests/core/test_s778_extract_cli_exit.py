"""S778: ``easyicu extract`` fails (exit 2) when a requested source is skipped or yields no rows."""

import pandas as pd
import pytest

from easyicu.scripts import extract_features


def _frames(_resolver, _datasource, _groups):
    return {"vitals": pd.DataFrame({"stay_id": [1, 1], "hr": [80.0, 82.0]})}


def _empty(_resolver, _datasource, _groups):
    return {"vitals": pd.DataFrame({"stay_id": [], "hr": []})}


@pytest.fixture
def stub_source(monkeypatch):
    monkeypatch.setattr(extract_features, "resolve_datasource", lambda registry, name, data_dir: object())


def _run(tmp_path, *pairs):
    sources = [name for name, _ in pairs]
    dirs = [str(path) for _, path in pairs]
    out = tmp_path / "exports"
    code = extract_features.main(["extract", "--sources", *sources, "--data-dirs", *dirs, "--output", str(out)])
    written = sorted(p.name for p in out.glob("*.csv")) if out.exists() else []
    return code, written


def test_a_written_source_exits_zero(tmp_path, monkeypatch, stub_source):
    monkeypatch.setattr(extract_features, "extract_groups", _frames)
    assert _run(tmp_path, ("miiv", tmp_path)) == (0, ["miiv_vitals.csv"])


def test_a_missing_data_directory_fails_and_writes_nothing(tmp_path, monkeypatch, stub_source):
    monkeypatch.setattr(extract_features, "extract_groups", _frames)
    assert _run(tmp_path, ("miiv", tmp_path / "absent")) == (2, [])


def test_an_unknown_source_fails(tmp_path, monkeypatch):
    def unknown(registry, name, data_dir):
        raise KeyError(name)

    monkeypatch.setattr(extract_features, "resolve_datasource", unknown)
    monkeypatch.setattr(extract_features, "extract_groups", _frames)
    assert _run(tmp_path, ("not_a_database", tmp_path)) == (2, [])


@pytest.mark.parametrize("extract", [lambda *a: {}, _empty])
def test_an_extraction_without_rows_fails_and_writes_nothing(tmp_path, monkeypatch, stub_source, extract):
    monkeypatch.setattr(extract_features, "extract_groups", extract)
    assert _run(tmp_path, ("miiv", tmp_path)) == (2, [])


def test_one_failed_source_fails_the_run_but_keeps_the_written_one(tmp_path, monkeypatch, stub_source):
    monkeypatch.setattr(extract_features, "extract_groups", _frames)
    assert _run(tmp_path, ("miiv", tmp_path), ("aumc", tmp_path / "absent")) == (2, ["miiv_vitals.csv"])
