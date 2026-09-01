from __future__ import annotations

import importlib.util
from pathlib import Path


def _tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "check_agent_runtime.py"
    spec = importlib.util.spec_from_file_location("check_agent_runtime", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_required_packages_is_exact_and_deterministic():
    tool = _tool()

    assert tool.missing_required_packages(
        {"numpy", "statsmodels", "lifelines"},
        ["lifelines", "xgboost", "shap", "xgboost"],
    ) == ("shap", "xgboost")
