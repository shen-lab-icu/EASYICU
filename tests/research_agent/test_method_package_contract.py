"""Boundary tests for static analytical-package declarations."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from easyicu.research_agent.contracts.method_packages import (
    BASELINE_PACKAGES,
    CURATED_METHOD_PACKAGES,
    FINGERPRINT_ONLY_DISTRIBUTIONS,
    OPTIONAL_BASELINE_PACKAGES,
    MethodPackage,
)

ROOT = Path(__file__).resolve().parents[2] / "src" / "easyicu" / "research_agent"


def test_method_package_contract_is_frozen_and_complete() -> None:
    assert BASELINE_PACKAGES == (
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "statsmodels",
        "sklearn",
        "pyarrow",
    )
    assert OPTIONAL_BASELINE_PACKAGES == ("seaborn",)
    assert CURATED_METHOD_PACKAGES == (
        MethodPackage(
            import_name="lifelines",
            pip_name="lifelines",
            capability=(
                "survival analysis — KaplanMeierFitter, CoxPHFitter, "
                "logrank_test, concordance_index"
            ),
            families=("survival",),
            fallback="statsmodels.duration (PHReg, SurvfuncRight)",
        ),
        MethodPackage(
            import_name="shap",
            pip_name="shap",
            capability=(
                "model-agnostic feature attribution — TreeExplainer/Explainer, "
                "beeswarm and waterfall summaries of per-feature contributions"
            ),
            families=("prediction_model", "dynamic_prediction"),
            fallback="sklearn.inspection.permutation_importance or model coefficients",
        ),
        MethodPackage(
            import_name="xgboost",
            pip_name="xgboost",
            capability=(
                "gradient-boosted trees for tabular prediction "
                "(XGBClassifier / XGBRegressor)"
            ),
            families=("prediction_model", "dynamic_prediction"),
            fallback=(
                "sklearn HistGradientBoostingClassifier / " "GradientBoostingClassifier"
            ),
        ),
    )
    # patsy moves computed numbers (statsmodels formulas are built on it) but is
    # deliberately NOT a MethodPackage: declaring it would advertise a direct
    # import the Coder never needs.  It is its own declaration precisely so the
    # literal stops being repeated in every builder of the distribution set.
    assert FINGERPRINT_ONLY_DISTRIBUTIONS == ("patsy",)
    assert not set(FINGERPRINT_ONLY_DISTRIBUTIONS) & {
        *BASELINE_PACKAGES,
        *OPTIONAL_BASELINE_PACKAGES,
        *(package.import_name for package in CURATED_METHOD_PACKAGES),
    }
    with pytest.raises(FrozenInstanceError):
        CURATED_METHOD_PACKAGES[0].import_name = "changed"  # type: ignore[misc]


def test_method_package_contract_imports_only_standard_library() -> None:
    path = ROOT / "contracts" / "method_packages.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imports <= {"__future__", "dataclasses", "typing"}


def test_contracts_package_import_remains_lazy() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.contracts'
module = importlib.import_module(package)
assert module.__name__ == package
assert not {name for name in sys.modules if name.startswith(package + '.')}
"""
    env = dict(os.environ)
    source_root = str(ROOT.parents[1])
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_authority_package_has_no_execution_imports() -> None:
    violations: list[tuple[str, int, str]] = []
    for path in sorted((ROOT / "authority").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = ".".join(
                        (
                            "easyicu",
                            "research_agent",
                            *path.relative_to(ROOT).parts[:-1],
                        )
                    )
                    module = importlib.util.resolve_name(
                        "." * node.level + module,
                        package,
                    )
                if module.startswith("easyicu.research_agent.execution"):
                    violations.append((path.name, node.lineno, module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("easyicu.research_agent.execution"):
                        violations.append((path.name, node.lineno, alias.name))
    assert violations == []


def test_runtime_fingerprint_uses_the_declared_distribution_set(monkeypatch) -> None:
    runtime = importlib.import_module("easyicu.research_agent.authority.step_runtime")
    seen: list[str] = []

    def version(name: str) -> str:
        seen.append(name)
        return "1.0"

    monkeypatch.setattr(runtime.importlib_metadata, "version", version)
    monkeypatch.setattr(runtime.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.13.5")
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")

    digest = runtime.current_execution_runtime_sha256()

    assert len(digest) == 64
    assert seen == sorted(
        {
            "pandas",
            "numpy",
            "scipy",
            "matplotlib",
            "statsmodels",
            "scikit-learn",
            "pyarrow",
            "seaborn",
            "lifelines",
            "shap",
            "xgboost",
            "patsy",
        }
    )
