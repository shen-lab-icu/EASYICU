from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "releases"
    / "EX-A02_republish_full6_candidate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("full6_republisher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_run(root: Path, module) -> Path:
    root.mkdir()
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_checkout": {"easyicu_git_commit": "1" * 40},
                "sources": {
                    database: {
                        "module_metrics": {
                            name: {"elapsed_seconds": 1.0}
                            for name in module.MODULES
                        }
                    }
                    for database in module.DATABASES
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "database_extraction_timing.csv").write_text(
        "database,status,total_rows,total_parquet_bytes\n"
        + "".join(
            f"{database},complete,1,1\n" for database in module.DATABASES
        ),
        encoding="utf-8",
    )
    (root / "run_metadata.json").write_text("stale\n", encoding="utf-8")
    (root / "module_extraction_timing.csv").write_text(
        "stale\n", encoding="utf-8"
    )
    (root / "publication_qc").mkdir()
    (root / "publication_qc" / "stale.json").write_text(
        "{}\n", encoding="utf-8"
    )
    for database in module.DATABASES:
        database_root = root / "exports" / database
        database_root.mkdir(parents=True)
        (database_root / "_manifest.json").write_text(
            json.dumps(
                {
                    "runtime_provenance": {
                        "easyicu_git_commit": "1" * 40,
                        "easyicu_git_dirty": False,
                    }
                }
            ),
            encoding="utf-8",
        )
        for name in module.MODULES:
            (database_root / f"{name}.parquet").write_bytes(b"parquet")
            (database_root / f"{name}.manifest.json").write_text(
                json.dumps({"saved": {name: {"concepts": []}}}),
                encoding="utf-8",
            )
    return root


def test_republisher_preserves_source_and_requires_fresh_release_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    source = _source_run(tmp_path / "source", module)
    destination = tmp_path / "destination"
    source_run_manifest_before = (source / "run_manifest.json").read_bytes()
    monkeypatch.setattr(module, "_require_clean_checkout", lambda: "2" * 40)

    def _fake_republish_database(
        database_root: Path,
        *,
        database: str,
        source_receipt: dict,
        publication_commit: str,
    ) -> dict:
        assert database_root.parent.name == "exports"
        assert source_receipt["easyicu_git_commit"] == "1" * 40
        assert publication_commit == "2" * 40
        return {
            "files": [
                {
                    "module": name,
                    "rows": index + 1,
                    "parquet_bytes": index + 2,
                    "parquet_sha256": str(index).zfill(64),
                }
                for index, name in enumerate(module.MODULES)
            ]
        }

    monkeypatch.setattr(module, "_republish_database", _fake_republish_database)
    output = module.republish_candidate(source, destination)

    assert output == destination
    assert (source / "run_manifest.json").read_bytes() == source_run_manifest_before
    assert (source / "run_metadata.json").exists()
    assert not (destination / "run_metadata.json").exists()
    assert not (destination / "module_extraction_timing.csv").exists()
    assert not (destination / "publication_qc").exists()
    run_manifest = json.loads((destination / "run_manifest.json").read_text())
    provenance = run_manifest["release_republication"]
    assert provenance["raw_database_reread"] is False
    assert provenance["timing_runtime_measurements_preserved"] is True
    assert provenance["timing_package_receipts_rebound"] is True
    assert provenance["publication_easyicu_git_commit"] == "2" * 40
    assert provenance["source_run_manifest_sha256"]
    timing = (destination / "database_extraction_timing.csv").read_text()
    assert ",190,209\n" in timing


def test_republisher_refuses_incomplete_source(tmp_path: Path) -> None:
    module = _load_module()
    source = _source_run(tmp_path / "source", module)
    (source / "exports" / "aumc" / "renal.parquet").unlink()

    with pytest.raises(module.RepublicationError, match="Missing regular source file"):
        module._validate_source(source)
