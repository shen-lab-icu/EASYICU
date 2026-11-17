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


def test_extract_help_names_the_opt_in_minute_modules(capsys):
    with pytest.raises(SystemExit) as exc:
        extract_features.main(["extract", "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "opt-in minute-resolution" in help_text
    assert all(module in help_text for module in ("subhour_signals", "vent_native", "haemo_native"))


def test_minute_modules_publish_a_manifest_on_the_one_minute_grid(tmp_path):
    import json

    from easyicu.synthetic_sources import generate_source

    source = tmp_path / "hirid"
    output = tmp_path / "out"
    generate_source(source, "hirid", stays=3)
    assert extract_features.main([
        "extract", "--sources", "hirid", "--data-dirs", str(source),
        "--output", str(output), "--modules", "subhour_signals",
    ]) == 0
    manifest = json.loads((output / "hirid" / "_manifest.json").read_text())
    minute = next(row for row in manifest["files"] if row["module"] == "subhour_signals")
    assert minute["time_resolution"]["interval_minutes"] == 1
    assert "hr" in minute["time_resolution"]["signals"]


@pytest.mark.parametrize(
    ("database", "groups", "code"),
    [
        ("sic", '{"vent_native": ["peep_total"]}', "subhour_signal_not_whitelisted"),
        ("hirid", '{"vent_native": ["sofa2"]}', "subhour_signal_not_whitelisted"),
        ("unknown_database", None, "subhour_module_unavailable"),
    ],
)
def test_minute_cli_refuses_outside_whitelisted_cells(tmp_path, caplog, database, groups, code):
    output = tmp_path / "out"
    args = [
        "extract", "--sources", database, "--data-dirs", str(tmp_path),
        "--output", str(output), "--modules", "vent_native",
    ]
    if groups:
        group_file = tmp_path / "groups.json"
        group_file.write_text(groups)
        args += ["--groups-json", str(group_file)]
    assert extract_features.main(args) == 2
    assert code in caplog.text
    assert not output.exists()
