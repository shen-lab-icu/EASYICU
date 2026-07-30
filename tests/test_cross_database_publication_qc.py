from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


AUDIT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "figures"
    / "QC-A02_easyicu_cross_database_reliability_audit.py"
)
FIGURE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "figures"
    / "QC-A01_cross_database_distributions.py"
)


def _load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_database_specific_commit_overrides_single_run_commit() -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02")
    metadata = {
        "easyicu_commit": "base",
        "database_commits": {"hirid": "corrective"},
    }

    assert module._expected_runtime_commit(metadata, "hirid") == "corrective"
    assert module._expected_runtime_commit(metadata, "eicu") == "base"


def test_single_run_commit_remains_supported() -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02")

    assert (
        module._expected_runtime_commit({"easyicu_commit": "one-commit"}, "miiv")
        == "one-commit"
    )


def test_missing_commit_is_explicit() -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02")

    assert module._expected_runtime_commit({}, "sic") is None


def test_figure_catalog_fills_derived_concept_metadata(tmp_path: Path) -> None:
    module = _load_script(FIGURE_SCRIPT, "easyicu_qc_a01")
    catalog_path = tmp_path / "concept-dict.json"
    catalog_path.write_text("{}\n", encoding="utf-8")

    catalog = module.load_catalog(catalog_path)

    assert catalog["uo_rt_6hr"]["unit"] == "mL/kg/h"
    assert catalog["uo_rt_6hr"]["description"] == (
        "Urine Output Rate (6h rolling window)"
    )
