from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_refresher():
    path = ROOT / "scripts/releases/EX-A03_refresh_selected_modules.py"
    spec = importlib.util.spec_from_file_location("selected_module_refresh", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_module_refresh_is_limited_to_correctness_modules() -> None:
    refresher = _load_refresher()
    assert refresher._validate_modules(["renal"]) == ("renal",)
    assert refresher._validate_modules(["respiratory"]) == ("respiratory",)
    assert refresher._validate_modules(["renal", "respiratory"]) == (
        "renal",
        "respiratory",
    )
    with pytest.raises(refresher.ModuleRefreshError, match="renal and respiratory"):
        refresher._validate_modules(["vitals"])


def test_selected_module_refresh_rejects_duplicate_data_path_overrides() -> None:
    refresher = _load_refresher()
    with pytest.raises(refresher.ModuleRefreshError, match="Duplicate"):
        refresher._parse_data_path_overrides(
            ["miiv=/tmp/one", "miiv=/tmp/two"]
        )


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
        candidate_root=candidate,
        modules=("renal",),
        batch_size=None,
        reuse_completed_export=False,
    )

    assert len(calls) == 1
