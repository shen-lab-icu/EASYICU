from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


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
    catalog_path.write_text(
        json.dumps(
            {
                "uo_rt_6hr": {},
                "bmi": {
                    "description": "Explicit BMI description",
                    "unit": "custom BMI unit",
                },
                "urine24": {
                    "description": "Explicit urine description",
                    "unit": None,
                },
                "pafi": {"description": "Explicit P/F description"},
            }
        ),
        encoding="utf-8",
    )

    catalog = module.load_catalog(catalog_path)

    assert catalog["uo_rt_6hr"]["unit"] == "mL/kg/h"
    assert catalog["uo_rt_6hr"]["description"] == (
        "Urine Output Rate (6h rolling window)"
    )
    assert catalog["bmi"]["description"] == "Explicit BMI description"
    assert catalog["bmi"]["unit"] == "custom BMI unit"
    assert catalog["urine24"]["description"] == "Explicit urine description"
    assert catalog["urine24"]["unit"] == "mL/24h"
    assert catalog["pafi"]["description"] == "Explicit P/F description"
    assert catalog["pafi"]["unit"] == "mmHg"


def test_figure_script_prefers_its_checkout_src(monkeypatch, tmp_path: Path) -> None:
    module = _load_script(FIGURE_SCRIPT, "easyicu_qc_a01_checkout")
    checkout_src = FIGURE_SCRIPT.resolve().parents[2] / "src"
    shadow = tmp_path / "old-editable"
    shadow.mkdir()
    monkeypatch.setattr(sys, "path", [str(shadow), *sys.path])

    selected = module._prefer_checkout_src(FIGURE_SCRIPT)

    assert selected == checkout_src
    assert Path(sys.path[0]).resolve() == checkout_src.resolve()


def test_qc_lineage_helpers_bind_exact_run_metadata_bytes(tmp_path: Path) -> None:
    figure_module = _load_script(FIGURE_SCRIPT, "easyicu_qc_a01_lineage")
    audit_module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02_lineage")
    run_metadata = tmp_path / "run_metadata.json"
    run_metadata.write_text(
        '{"run_id":"full6-test","database_commits":{"hirid":"abc"}}\n',
        encoding="utf-8",
    )
    expected_sha256 = hashlib.sha256(run_metadata.read_bytes()).hexdigest()
    expected = {
        "source_run_id": "full6-test",
        "source_run_metadata_sha256": expected_sha256,
    }

    assert figure_module.source_run_lineage(run_metadata) == expected
    assert audit_module._source_run_lineage(run_metadata) == expected


def test_reader_facing_labels_suppress_type_markers_only() -> None:
    module = _load_script(FIGURE_SCRIPT, "easyicu_qc_a01_units")

    for unit in ("boolean", "CATEGORY", " datetime "):
        payload = module.PlotPayload(
            module="demo",
            variable="derived_flag",
            description="Derived flag",
            unit=unit,
            kind="unavailable",
            data=pd.DataFrame(),
            subtitle="",
            footnote="",
        )
        assert module.axis_label(payload, compact=False) == "derived_flag"
        assert module.value_axis_label(payload) == "Value"

    payload = module.PlotPayload(
        module="renal",
        variable="urine24",
        description="24h urine output",
        unit="mL/24h",
        kind="continuous",
        data=pd.DataFrame(),
        subtitle="",
        footnote="",
    )
    assert module.axis_label(payload, compact=False) == "urine24 (mL/24h)"
    assert module.value_axis_label(payload) == "Value (mL/24h)"


def test_render_only_refreshes_catalog_metadata_and_lineage_without_parquet(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script(FIGURE_SCRIPT, "easyicu_qc_a01_render_only")
    output_root = tmp_path / "publication_qc"
    audit_root = output_root / "audit"
    source_root = output_root / "source_data" / "renal"
    audit_root.mkdir(parents=True)
    source_root.mkdir(parents=True)
    audit_path = audit_root / "variable_audit.csv"
    pd.DataFrame(
        [
            {
                "module": "renal",
                "variable": "urine24",
                "description": "stale description",
                "unit": None,
                "catalog_min": None,
                "catalog_max": None,
                "plot_kind": "continuous",
                "database": database,
                "row_count": 1,
                "non_null_or_finite": 1,
            }
            for database in module.DATABASES
        ]
    ).to_csv(audit_path, index=False)
    pd.DataFrame(
        {
            "database": ["aumc"],
            "bin_center": [1.0],
            "density_smoothed": [1.0],
            "total_finite": [1],
        }
    ).to_csv(source_root / "urine24.csv", index=False)
    (audit_root / "run_manifest.json").write_text(
        '{"modules":["renal"],"catalog_sha256":"stale"}\n',
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "urine24": {
                    "description": "Catalog urine output",
                    "min": 0,
                    "max": 10000,
                }
            }
        ),
        encoding="utf-8",
    )
    catalog = module.load_catalog(catalog_path)
    lineage = {
        "source_run_id": "current-test",
        "source_run_metadata_sha256": "a" * 64,
    }
    captured: list[object] = []

    def _capture_atlas(
        module_name,
        payloads,
        output_base,
        dpi,
        panels_per_page,
    ):
        captured.extend(payloads)
        return 1

    monkeypatch.setattr(module, "save_module_atlas", _capture_atlas)
    monkeypatch.setattr(
        module.pq,
        "ParquetFile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("render-only must not scan Parquet")
        ),
    )

    result = module.render_from_source(
        output_root,
        ["renal"],
        72,
        12,
        catalog=catalog,
        catalog_sha256=module.file_sha256(catalog_path),
        lineage=lineage,
    )

    assert result == 0
    refreshed = pd.read_csv(audit_path)
    assert set(refreshed["description"]) == {"Catalog urine output"}
    assert set(refreshed["unit"]) == {"mL/24h"}
    assert set(refreshed["catalog_min"]) == {0.0}
    assert set(refreshed["catalog_max"]) == {10000.0}
    assert len(captured) == 1
    assert captured[0].unit == "mL/24h"
    manifest = json.loads(
        (audit_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_run_id"] == "current-test"
    assert manifest["source_run_metadata_sha256"] == "a" * 64
    assert manifest["catalog_sha256"] == module.file_sha256(catalog_path)
