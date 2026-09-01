from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_refresher():
    path = ROOT / "scripts/releases/EX-A03_refresh_selected_modules.py"
    spec = importlib.util.spec_from_file_location("selected_module_refresh", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_module_refresh_is_limited_to_correctness_modules() -> None:
    refresher = _load_refresher()
    assert refresher._validate_modules(["outcome"]) == ("outcome",)
    assert refresher._validate_modules(["renal"]) == ("renal",)
    assert refresher._validate_modules(["respiratory"]) == ("respiratory",)
    assert refresher._validate_modules(["sofa2_score"]) == ("sofa2_score",)
    assert refresher._validate_modules(["renal", "respiratory"]) == (
        "renal",
        "respiratory",
    )
    with pytest.raises(refresher.ModuleRefreshError, match="sofa2_score"):
        refresher._validate_modules(["vitals"])


def test_respiratory_refresh_expands_to_score_and_sepsis_dependencies() -> None:
    refresher = _load_refresher()
    assert refresher._expand_module_dependency_closure(["outcome"]) == ("outcome",)
    assert refresher._expand_module_dependency_closure(["respiratory"]) == (
        "respiratory",
        "sepsis_shared",
        "sofa1_score",
        "sofa2_score",
        "sepsis3_sofa1",
        "sepsis3_sofa2",
    )
    assert refresher._expand_module_dependency_closure(
        ["outcome", "renal", "respiratory"]
    ) == (
        "outcome",
        "respiratory",
        "renal",
        "sepsis_shared",
        "sofa1_score",
        "sofa2_score",
        "sepsis3_sofa1",
        "sepsis3_sofa2",
    )
    assert refresher._expand_module_dependency_closure(["sofa2_score"]) == (
        "sepsis_shared",
        "sofa2_score",
        "sepsis3_sofa2",
    )


def test_selected_module_refresh_rejects_duplicate_data_path_overrides() -> None:
    refresher = _load_refresher()
    with pytest.raises(refresher.ModuleRefreshError, match="Duplicate"):
        refresher._parse_data_path_overrides(["miiv=/tmp/one", "miiv=/tmp/two"])


def test_new_candidate_never_reuses_source_module_just_because_schema_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an explicit resume may reuse a completed selected-module export."""

    refresher = _load_refresher()
    candidate = tmp_path / "candidate"
    destination = candidate / "exports" / "hirid"
    destination.mkdir(parents=True)
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(refresher, "_module_is_canonical_refresh", lambda *_: True)

    def fake_extract_database(*args, **kwargs):
        calls.append(kwargs)
        staging = Path(kwargs["output_dir"])
        staging.mkdir(parents=True)
        (staging / "renal.parquet").write_bytes(b"parquet-placeholder")
        (staging / "renal.manifest.json").write_text(json.dumps({}))
        return {
            "num_patients": 1,
            "batch_size": 1,
            "total_elapsed": 1.0,
            "modules": {
                "renal": {
                    "errors": [],
                    "elapsed": 1.0,
                    "peak_rss_mb": 1.0,
                    "peak_working_set_mb": 1.0,
                }
            },
        }

    monkeypatch.setattr(refresher, "extract_database", fake_extract_database)

    refresher._refresh_one_database(
        database="hirid",
        data_path=str(tmp_path),
        source_database_root=tmp_path / "source" / "hirid",
        candidate_root=candidate,
        modules=("renal",),
        batch_size=None,
        reuse_completed_export=False,
    )

    assert len(calls) == 1


def test_resume_never_treats_destination_schema_as_raw_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume also needs staged evidence or a fresh extraction."""

    refresher = _load_refresher()
    candidate = tmp_path / "candidate"
    destination = candidate / "exports" / "miiv"
    destination.mkdir(parents=True)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(refresher, "_module_is_canonical_refresh", lambda *_: True)

    def fake_extract_database(*args, **kwargs):
        calls.append(kwargs)
        staging = Path(kwargs["output_dir"])
        staging.mkdir(parents=True)
        (staging / "respiratory.parquet").write_bytes(b"parquet-placeholder")
        (staging / "respiratory.manifest.json").write_text(json.dumps({}))
        return {
            "num_patients": 1,
            "batch_size": 1,
            "total_elapsed": 1.0,
            "modules": {
                "respiratory": {
                    "errors": [],
                    "elapsed": 1.0,
                    "peak_rss_mb": 1.0,
                    "peak_working_set_mb": 1.0,
                }
            },
        }

    monkeypatch.setattr(refresher, "extract_database", fake_extract_database)
    refresher._refresh_one_database(
        database="miiv",
        data_path=str(tmp_path),
        source_database_root=tmp_path / "source" / "miiv",
        candidate_root=candidate,
        modules=("respiratory",),
        batch_size=None,
        reuse_completed_export=True,
    )

    assert len(calls) == 1


def test_resume_reuses_only_complete_files_detached_from_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refresher = _load_refresher()
    source = tmp_path / "source" / "aumc"
    candidate_root = tmp_path / "candidate"
    candidate = candidate_root / "exports" / "aumc"
    source.mkdir(parents=True)
    candidate.mkdir(parents=True)
    for suffix in (".parquet", ".manifest.json"):
        (source / f"respiratory{suffix}").write_bytes(b"source")
        (candidate / f"respiratory{suffix}").write_bytes(b"refreshed")
    (candidate / "respiratory.manifest.json").write_text(
        json.dumps({"elapsed_sec": 1, "peak_rss_mb": 2, "peak_working_set_mb": 3})
    )
    monkeypatch.setattr(refresher, "_module_is_canonical_refresh", lambda *_: True)
    monkeypatch.setattr(
        refresher,
        "extract_database",
        lambda *args, **kwargs: pytest.fail(
            "detached completed files were re-extracted"
        ),
    )

    result = refresher._refresh_one_database(
        database="aumc",
        data_path=str(tmp_path),
        source_database_root=source,
        candidate_root=candidate_root,
        modules=("respiratory",),
        batch_size=None,
        reuse_completed_export=True,
    )

    assert result["recovery_mode"].startswith("explicit_resume")
