from __future__ import annotations

import importlib.util
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


def test_selected_module_refresh_is_deliberately_limited_to_renal() -> None:
    refresher = _load_refresher()
    assert refresher._validate_modules(["renal"]) == ("renal",)
    with pytest.raises(refresher.ModuleRefreshError, match="only renal"):
        refresher._validate_modules(["vitals"])


def test_selected_module_refresh_rejects_duplicate_data_path_overrides() -> None:
    refresher = _load_refresher()
    with pytest.raises(refresher.ModuleRefreshError, match="Duplicate"):
        refresher._parse_data_path_overrides(
            ["miiv=/tmp/one", "miiv=/tmp/two"]
        )
