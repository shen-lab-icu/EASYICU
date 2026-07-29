"""Contract tests for the sequential native-v2 re-export launcher."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "reextract_native_export_v2.py"
SPEC = importlib.util.spec_from_file_location("reextract_native_export_v2", TOOL)
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class _Package:
    column_metadata_sha256 = "a" * 64
    concept_index = {"age": object()}
    missing_selected_concepts = ()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_launcher_is_sequential_private_and_uses_adaptive_external_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(launcher, "DEFAULT_DATA_PATHS", {"miiv": str(source)})

    def fake_extract(database, **kwargs):
        calls.append({"database": database, **kwargs})
        assert os.environ["TMPDIR"] == str(tmp_path / "out" / ".runtime_tmp")
        assert os.environ["TMP"] == str(tmp_path / "out" / ".runtime_tmp")
        assert os.environ["TEMP"] == str(tmp_path / "out" / ".runtime_tmp")
        assert os.environ["EASYICU_DUCKDB_TEMP_DIR"] == str(
            tmp_path / "out" / ".runtime_spill"
        )
        assert os.environ["EASYICU_DUCKDB_THREADS"] == "1"
        assert os.environ["EASYICU_DUCKDB_MEMORY_LIMIT"] == "1GB"
        out = Path(kwargs["output_dir"])
        out.mkdir(mode=0o700)
        spill = out / ".easyicu_spill"
        spill.mkdir()
        (spill / "duckdb_temp_storage.tmp").write_text("temporary")
        (out / "_manifest.json").write_text(
            json.dumps({"unavailable_modules": []}), encoding="utf-8"
        )
        return {
            "num_patients": 10,
            "native_export_v2": {"output_validation_reads": 19},
        }

    monkeypatch.setattr(launcher, "extract_database", fake_extract)
    monkeypatch.setattr(launcher, "open_export_package", lambda _path: _Package())
    args = launcher._parse_args(
        ["--output-root", str(tmp_path / "out"), "--databases", "miiv"]
    )

    manifest = launcher.run(args)

    assert manifest["status"] == "verified"
    assert calls == [
        {
            "database": "miiv",
            "data_path": str(source),
            "output_dir": tmp_path / "out" / "miiv",
            "batch_size": None,
            "native_export_v2": True,
            "stream_output_batches": True,
            "verbose": True,
        }
    ]
    persisted = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert persisted["sources"]["miiv"]["status"] == "verified"
    assert persisted["sources"]["miiv"]["spill_directory_removed"] is True
    assert not (tmp_path / "out" / "miiv" / ".easyicu_spill").exists()
    assert not (tmp_path / "out" / ".runtime_tmp").exists()
    assert not (tmp_path / "out" / ".runtime_spill").exists()
    assert oct((tmp_path / "out").stat().st_mode & 0o777) == "0o700"


def test_launcher_rejects_an_existing_output_root(tmp_path: Path) -> None:
    out = tmp_path / "existing"
    out.mkdir()
    args = launcher._parse_args(["--output-root", str(out), "--databases", "miiv"])

    with pytest.raises(ValueError, match="must be new"):
        launcher.run(args)


def test_one_shot_launcher_keeps_external_runtime_but_disables_auto_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(launcher, "DEFAULT_DATA_PATHS", {"miiv": str(source)})

    def fake_extract(database, **kwargs):
        calls.append({"database": database, **kwargs})
        assert os.environ["TMPDIR"] == str(tmp_path / "out" / ".runtime_tmp")
        assert os.environ["EASYICU_DUCKDB_TEMP_DIR"] == str(
            tmp_path / "out" / ".runtime_spill"
        )
        assert "EASYICU_ONESHOT_BUDGET_MB" not in os.environ
        out = Path(kwargs["output_dir"])
        out.mkdir(mode=0o700)
        (out / "_manifest.json").write_text(
            json.dumps({"unavailable_modules": []}), encoding="utf-8"
        )
        return {
            "num_patients": 10,
            "native_export_v2": {"output_validation_reads": 19},
        }

    monkeypatch.setattr(launcher, "extract_database", fake_extract)
    monkeypatch.setattr(launcher, "open_export_package", lambda _path: _Package())
    args = launcher._parse_args(
        ["--output-root", str(tmp_path / "out"), "--databases", "miiv", "--one-shot"]
    )

    manifest = launcher.run(args)

    assert manifest["extraction_mode"] == "one_shot_all_patients"
    assert calls == [
        {
            "database": "miiv",
            "data_path": str(source),
            "output_dir": tmp_path / "out" / "miiv",
            "batch_size": None,
            "native_export_v2": True,
            "stream_output_batches": False,
            "verbose": True,
        }
    ]
