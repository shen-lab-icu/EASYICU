from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


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


def test_manifest_metadata_coverage_treats_typed_structural_placeholders_as_closed() -> (
    None
):
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02_metadata_coverage")
    manifests = pd.DataFrame(
        [
            {
                "concept_meta_count": 4,
                "availability": "available",
                "concept_metadata_complete": True,
                "structural_placeholder_valid": False,
            },
            {
                "concept_meta_count": 0,
                "availability": "structurally_unavailable",
                "concept_metadata_complete": False,
                "structural_placeholder_valid": True,
            },
            {
                "concept_meta_count": 0,
                "availability": "available",
                "concept_metadata_complete": False,
                "structural_placeholder_valid": False,
            },
        ]
    )

    assert module._manifest_metadata_coverage(manifests) == {
        "manifest_rows_with_concept_meta": 1,
        "manifest_rows_with_complete_concept_meta": 1,
        "manifest_structurally_unavailable_rows": 1,
        "manifest_valid_structural_placeholder_rows": 1,
        "manifest_invalid_structural_placeholder_rows": 0,
        "manifest_rows_with_concept_meta_or_structural_status": 2,
        "manifest_metadata_contract_gap_rows": 1,
        "manifest_rows_missing_concept_meta_without_valid_structural_status": 1,
        "manifest_rows_missing_concept_meta_without_structural_status": 1,
    }


@pytest.mark.parametrize(
    ("module_name", "variable", "database", "evidence_fragment"),
    [
        ("chemistry", "bili_dir", "aumc vs miiv", "375 numeric records"),
        ("chemistry", "tri", "eicu vs mimic", "192,317 numeric rows"),
        ("hematology", "bnd", "mimic vs hirid", "40,560 raw values"),
        ("vasopressors", "epi_dur", "aumc vs mimic", "2,715 order groups"),
    ],
)
def test_distribution_adjudication_preserves_each_exact_flag_and_source_trace(
    module_name: str,
    variable: str,
    database: str,
    evidence_fragment: str,
) -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02_adjudication")
    flags = pd.DataFrame(
        [
            {
                "module": module_name,
                "variable": variable,
                "database": database,
                "flag": "median_scale_shift",
                "severity": "review",
                "evidence": "positive median ratio=63.45",
                "origin_classification": "candidate",
            },
        ]
    )

    result = module._adjudicate_distribution_flags(
        flags,
        source_run_id=module.CURRENT_QC_SOURCE_RUN_ID,
        source_run_metadata_sha256=module.CURRENT_QC_SOURCE_RUN_METADATA_SHA256,
    )

    assert result.shape[0] == flags.shape[0]
    assert result.loc[0, "adjudication_status"] == "source_trace_complete"
    assert (
        result.loc[0, "adjudication_source_run_id"] == module.CURRENT_QC_SOURCE_RUN_ID
    )
    assert result.loc[0, "adjudication_source_run_metadata_sha256"] == (
        module.CURRENT_QC_SOURCE_RUN_METADATA_SHA256
    )
    assert evidence_fragment in result.loc[0, "adjudication_evidence"]


@pytest.mark.parametrize(
    ("source_run_id", "source_sha256", "flag"),
    [
        ("future-run-requiring-a-new-source-trace", "f" * 64, "median_scale_shift"),
        (
            "current_full6_native_v2_hirid_urine24_20260730",
            "f" * 64,
            "median_scale_shift",
        ),
        (
            "current_full6_native_v2_hirid_urine24_20260730",
            "62adfb6f29a05305d687802f0eaa1c98f0ba2c4b888bb122c7e29233b4663d04",
            "above_catalog_range",
        ),
    ],
)
def test_distribution_adjudication_rejects_non_exact_run_or_flag(
    source_run_id: str,
    source_sha256: str,
    flag: str,
) -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02_adjudication_negative")
    flags = pd.DataFrame(
        [
            {
                "module": "chemistry",
                "variable": "bili_dir",
                "database": "aumc vs miiv",
                "flag": flag,
                "severity": "review",
                "evidence": "candidate",
                "origin_classification": "candidate",
            }
        ]
    )

    result = module._adjudicate_distribution_flags(
        flags,
        source_run_id=source_run_id,
        source_run_metadata_sha256=source_sha256,
    )

    assert result.iloc[0]["adjudication_status"] == "unadjudicated"


def test_source_manifest_hashes_are_required_and_verified(tmp_path: Path) -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02_manifest_hashes")
    expected: dict[str, str] = {}
    for database in module.DATABASES:
        database_root = tmp_path / database
        database_root.mkdir()
        manifest = database_root / "_manifest.json"
        manifest.write_text(f'{{"database":"{database}"}}\n', encoding="utf-8")
        expected[database] = hashlib.sha256(manifest.read_bytes()).hexdigest()

    assert (
        module._verify_source_manifest_hashes(
            tmp_path,
            {"source_manifest_sha256": expected},
        )
        == expected
    )

    mismatched = dict(expected)
    mismatched["hirid"] = "f" * 64
    with pytest.raises(ValueError, match="Source manifest SHA-256 mismatch"):
        module._verify_source_manifest_hashes(
            tmp_path,
            {"source_manifest_sha256": mismatched},
        )
    with pytest.raises(ValueError, match="must bind all six"):
        module._verify_source_manifest_hashes(tmp_path, {})


def test_structural_placeholder_requires_zero_rows_typed_schema_and_status() -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02_structural_contract")
    entry = {
        "availability": "structurally_unavailable",
        "rows": 0,
        "concepts": 0,
        "concept_ids": [],
        "physical_concept_ids": ["sep3"],
        "column_metadata_columns": [],
        "concept_status": {
            "sep3": {
                "availability": "structurally_unavailable_placeholder",
                "non_null": 0,
            }
        },
    }
    base = {
        "module": "sepsis3",
        "entry": entry,
        "expected_concepts": ["sep3"],
        "parquet_names": ["stay_id", "charttime", "sep3"],
        "parquet_types": {
            "stay_id": "int64",
            "charttime": "double",
            "sep3": "bool",
        },
        "actual_row_count": 0,
        "manifest_schema_matches_parquet": True,
    }

    assert module._structural_placeholder_checks(**base)["structural_placeholder_valid"]

    nonzero_manifest = json.loads(json.dumps(entry))
    nonzero_manifest["rows"] = 1
    assert not module._structural_placeholder_checks(
        **{**base, "entry": nonzero_manifest}
    )["structural_placeholder_valid"]
    assert not module._structural_placeholder_checks(**{**base, "actual_row_count": 1})[
        "structural_placeholder_valid"
    ]
    assert not module._structural_placeholder_checks(
        **{**base, "parquet_types": {**base["parquet_types"], "sep3": "null"}}
    )["structural_placeholder_valid"]
    wrong_status = json.loads(json.dumps(entry))
    wrong_status["concept_status"]["sep3"]["availability"] = "available"
    assert not module._structural_placeholder_checks(**{**base, "entry": wrong_status})[
        "structural_placeholder_valid"
    ]


def test_row_grain_receipt_binds_primary_key_audit_and_parquet_bytes(
    tmp_path: Path,
) -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02_row_grain")
    parquet = tmp_path / "vitals.parquet"
    pd.DataFrame(
        {"stay_id": [1], "charttime": [0.0], "hr": [70.0]}
    ).to_parquet(parquet, index=False)
    entry = {
        "rows": 1,
        "primary_key": ["stay_id", "charttime"],
        "row_grain": "one_row_per_icu_stay_relative_hour",
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "parquet_bytes": parquet.stat().st_size,
        "row_grain_audit": {
            "row_grain": "one_row_per_icu_stay_relative_hour",
            "primary_key": ["stay_id", "charttime"],
            "null_key_equality": "nulls_equal",
            "source_rows": 1,
            "published_rows": 1,
            "null_charttime_rows_after": 0,
            "duplicate_excess_rows_before": 0,
            "rows_consolidated": 0,
            "duplicate_excess_rows_after": 0,
        },
    }

    checks = module._row_grain_contract_checks(
        module="vitals",
        entry=entry,
        parquet_path=parquet,
        actual_row_count=1,
    )
    assert checks["row_grain_contract_valid"]
    assert checks["parquet_sha256_matches"]
    assert checks["parquet_bytes_matches"]

    parquet.write_bytes(parquet.read_bytes() + b"tampered")
    tampered = module._row_grain_contract_checks(
        module="vitals",
        entry=entry,
        parquet_path=parquet,
        actual_row_count=1,
    )
    assert not tampered["parquet_sha256_matches"]
    assert not tampered["parquet_bytes_matches"]
    assert not tampered["row_grain_contract_valid"]


def test_null_time_contract_allows_only_declared_admission_level_static_values(
    tmp_path: Path,
) -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02_null_time_static")
    parquet = tmp_path / "other_scores.parquet"
    pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [None],
            "qsofa": [None],
            "charlson": [2.0],
        }
    ).to_parquet(parquet, index=False)

    checks, details = module._null_time_concept_contract_checks(
        module="other_scores",
        database="mimic",
        parquet_path=parquet,
        expected_concepts=["qsofa", "charlson"],
        manifest_null_charttime_rows=1,
    )

    assert checks["null_time_concept_contract_valid"]
    assert checks["null_time_observed_rows"] == 1
    assert checks["null_time_allowed_non_null_cells"] == 1
    assert checks["null_time_disallowed_non_null_cells"] == 0
    assert details == [
        {
            "database": "mimic",
            "module": "other_scores",
            "concept": "charlson",
            "null_time_non_null_count": 1,
            "allowed": True,
            "classification": "admission_level_static_score",
            "evidence": "Charlson is derived once per linked hospital/ICU admission.",
        }
    ]


@pytest.mark.parametrize(
    ("module_name", "database", "values", "expected_concepts", "classification"),
    [
        (
            "renal",
            "eicu",
            {"rrt_criteria": False},
            ["rrt_criteria"],
            "unapproved_or_time_dependent_concept",
        ),
        (
            "medications",
            "eicu",
            {"phenytoin": True},
            ["phenytoin"],
            "unapproved_or_time_dependent_concept",
        ),
        (
            "sepsis_shared",
            "aumc",
            {"samp": True},
            ["samp"],
            "unapproved_or_time_dependent_concept",
        ),
        (
            "other_scores",
            "eicu",
            {"qsofa": None},
            ["qsofa"],
            "outer_merge_empty_artifact",
        ),
    ],
)
def test_null_time_contract_fails_closed_on_dynamic_or_empty_rows(
    tmp_path: Path,
    module_name: str,
    database: str,
    values: dict[str, object],
    expected_concepts: list[str],
    classification: str,
) -> None:
    module = _load_script(
        AUDIT_SCRIPT,
        f"easyicu_qc_a02_null_time_negative_{module_name}_{database}",
    )
    parquet = tmp_path / f"{module_name}_{database}.parquet"
    pd.DataFrame(
        {"stay_id": [1], "charttime": [None], **{k: [v] for k, v in values.items()}}
    ).to_parquet(parquet, index=False)

    checks, details = module._null_time_concept_contract_checks(
        module=module_name,
        database=database,
        parquet_path=parquet,
        expected_concepts=expected_concepts,
        manifest_null_charttime_rows=1,
    )

    assert not checks["null_time_concept_contract_valid"]
    assert details[0]["allowed"] is False
    assert details[0]["classification"] == classification


def test_null_time_contract_rejects_manifest_count_mismatch(tmp_path: Path) -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02_null_time_count")
    parquet = tmp_path / "sepsis_shared.parquet"
    pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [None],
            "culture_positive": [False],
        }
    ).to_parquet(parquet, index=False)

    checks, _ = module._null_time_concept_contract_checks(
        module="sepsis_shared",
        database="eicu",
        parquet_path=parquet,
        expected_concepts=["culture_positive"],
        manifest_null_charttime_rows=0,
    )

    assert not checks["null_time_manifest_count_matches"]
    assert not checks["null_time_concept_contract_valid"]


def test_qc_a02_fails_closed_on_end_to_end_metadata_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script(AUDIT_SCRIPT, "easyicu_qc_a02_gap_cli")
    export_root = tmp_path / "exports"
    output_root = tmp_path / "audit"
    modules = [f"module_{index:02d}" for index in range(19)]
    source_manifest_sha256: dict[str, str] = {}
    audit_rows: list[dict[str, object]] = []

    for database in module.DATABASES:
        database_root = export_root / database
        database_root.mkdir(parents=True)
        sidecar = database_root / "column_metadata.json"
        sidecar.write_text("{}\n", encoding="utf-8")
        sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        entries = []
        for module_name in modules:
            parquet_path = database_root / f"{module_name}.parquet"
            pd.DataFrame(
                {"stay_id": [1], "charttime": [0.0], "value": [1.0]}
            ).to_parquet(parquet_path, index=False)
            metadata_columns = ["value"]
            if database == "aumc" and module_name == modules[0]:
                metadata_columns = []
            entries.append(
                {
                    "module": module_name,
                    "availability": "available",
                    "rows": 1,
                    "concepts": 1,
                    "concept_ids": ["value"],
                    "physical_concept_ids": ["value"],
                    "physical_schema": {
                        "stay_id": "int64",
                        "charttime": "double",
                        "value": "double",
                    },
                    "parquet_sha256": hashlib.sha256(
                        parquet_path.read_bytes()
                    ).hexdigest(),
                    "parquet_bytes": parquet_path.stat().st_size,
                    "primary_key": ["stay_id", "charttime"],
                    "row_grain": "one_row_per_icu_stay_relative_hour",
                    "row_grain_audit": {
                        "row_grain": "one_row_per_icu_stay_relative_hour",
                        "primary_key": ["stay_id", "charttime"],
                        "null_key_equality": "nulls_equal",
                        "source_rows": 1,
                        "published_rows": 1,
                        "null_charttime_rows_before": 0,
                        "null_charttime_rows_after": 0,
                        "duplicate_key_rows_before": 0,
                        "duplicate_key_groups_before": 0,
                        "duplicate_excess_rows_before": 0,
                        "rows_consolidated": 0,
                        "duplicate_excess_rows_after": 0,
                    },
                    "concept_status": {
                        "value": {"availability": "available", "non_null": 1}
                    },
                    "column_metadata_columns": metadata_columns,
                }
            )
            audit_rows.append(
                {
                    "module": module_name,
                    "variable": "value",
                    "description": "Synthetic value",
                    "unit": "1",
                    "plot_kind": "continuous",
                    "database": database,
                    "row_count": 1,
                    "non_null_or_finite": 1,
                    "median_sample": 1.0,
                    "minimum": 1.0,
                    "maximum": 1.0,
                    "catalog_min": 0.0,
                    "catalog_max": 2.0,
                }
            )
        root_manifest = database_root / "_manifest.json"
        root_manifest.write_text(
            json.dumps(
                {
                    "schema_version": module.NATIVE_SCHEMA_VERSION,
                    "runtime_provenance": {
                        "easyicu_git_commit": "test-commit",
                        "easyicu_git_dirty": False,
                    },
                    "column_metadata": {
                        "file": sidecar.name,
                        "sha256": sidecar_sha256,
                    },
                    "files": entries,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        source_manifest_sha256[database] = hashlib.sha256(
            root_manifest.read_bytes()
        ).hexdigest()

    run_metadata = tmp_path / "run_metadata.json"
    run_metadata.write_text(
        json.dumps(
            {
                "run_id": "synthetic-gap-run",
                "easyicu_commit": "test-commit",
                "module_concepts": {name: ["value"] for name in modules},
                "source_manifest_sha256": source_manifest_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    figure_audit = tmp_path / "variable_audit.csv"
    pd.DataFrame(audit_rows).to_csv(figure_audit, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(AUDIT_SCRIPT),
            "--export-root",
            str(export_root),
            "--figure-audit",
            str(figure_audit),
            "--run-metadata",
            str(run_metadata),
            "--output-dir",
            str(output_root),
        ],
    )

    with pytest.raises(ValueError, match="metadata contract gaps"):
        module.main()

    summary = json.loads((output_root / "audit_summary.json").read_text())
    assert summary["source_manifest_sha256_verified_rows"] == 6
    assert summary["manifest_metadata_contract_gap_rows"] == 1


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
