from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "figures"
    / "QC-A03_main_figures.py"
)


def _load_script():
    module_name = "easyicu_qc_a03_main_figures"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_bundle():
    module = _load_script()
    modules = [
        "demographics",
        "diagnoses",
        "vitals",
        "respiratory",
        "ventilator",
        "blood_gas",
        "chemistry",
        "hematology",
        "coagulation",
        "liver",
        "renal",
        "medications",
        "vasopressors",
        "other_scores",
        "sepsis3",
        "sepsis_shared",
        "sofa2",
        "kdigo",
        "outcomes",
    ]
    cohort_counts = {
        "aumc": 23_106,
        "eicu": 200_859,
        "hirid": 33_905,
        "mimic": 61_532,
        "miiv": 73_181,
        "sic": 27_386,
    }
    field_rows = []
    variable_rows = []
    availability_rows = []
    concept_specs: dict[str, list[tuple[str, str, str, float]]] = {}
    for module_name in modules:
        concept_specs[module_name] = [
            (f"{module_name}_measure", "continuous", "canonical unit", 0.0),
            (f"{module_name}_flag", "binary", "boolean", 0.0),
        ]
        if module_name == "renal":
            concept_specs[module_name].append(
                ("fluid_balance_cumulative", "continuous", "mL", -100_000.0)
            )

    for module_position, module_name in enumerate(modules):
        for variable, kind, unit, catalog_min in concept_specs[module_name]:
            databases_available = 0
            for database_position, database in enumerate(module.DATABASES):
                structural = module_name == "sepsis3" and database == "sic"
                available = not structural
                databases_available += int(available)
                field_rows.append(
                    {
                        "module": module_name,
                        "database": database,
                        "concept": variable,
                        "available": available,
                        "unit": unit,
                    }
                )
                if variable == "demographics_measure":
                    medians = [1.0, 2.0, 4.0, 8.0, 16.0, 200.0]
                    median = medians[database_position]
                elif variable == "fluid_balance_cumulative":
                    median = -300.0 + database_position * 130.0
                else:
                    median = 1.0 + module_position * 0.01 + database_position * 0.02
                non_null = 0 if structural else 1_000 + module_position * 10
                variable_rows.append(
                    {
                        "module": module_name,
                        "variable": variable,
                        "database": database,
                        "plot_kind": kind,
                        "non_null_or_finite": non_null,
                        "median_sample": median if available else pd.NA,
                        "minimum": (
                            -1_000.0
                            if variable == "fluid_balance_cumulative"
                            else 0.1
                        ),
                        "maximum": 1_000.0,
                        "catalog_min": catalog_min,
                        "catalog_max": 100_000.0,
                        "unit": unit,
                        "description": variable.replace("_", " "),
                        "row_count": 1_500,
                    }
                )
            availability_rows.append(
                {
                    "module": module_name,
                    "variable": variable,
                    "description": variable.replace("_", " "),
                    "unit": unit,
                    "plot_kind": kind,
                    "databases_available": databases_available,
                    "total_non_null": databases_available * 1_000,
                    "total_rows": 9_000,
                    "available_all_six": databases_available == 6,
                }
            )

    module_schema = pd.DataFrame(
        [
            {
                "module": module_name,
                "expected_concept_count": len(concept_specs[module_name]),
                "all_six_parquets_present": True,
                "same_concept_set": True,
                "same_concept_order": True,
                "same_full_physical_schema": True,
                "type_mismatch_field_count": 0,
                "missing_schema_slots": 0,
                "canonical_stay_id_all_six": True,
                "canonical_charttime_all_six": True,
            }
            for module_name in modules
        ]
    )
    manifest_rows = []
    for module_name in modules:
        for database in module.DATABASES:
            structural = module_name == "sepsis3" and database == "sic"
            manifest_rows.append(
                {
                    "module": module_name,
                    "database": database,
                    "availability": (
                        "structurally_unavailable" if structural else "available"
                    ),
                    "actual_parquet_row_count": 0 if structural else 12_345,
                    "saved_path_exists": True,
                    "native_v2": True,
                    "manifest_schema_matches_parquet": True,
                    "parquet_sha256_matches": True,
                    "parquet_bytes_matches": True,
                    "row_grain_contract_valid": True,
                    "null_time_concept_contract_valid": True,
                    "concept_metadata_complete": not structural,
                    "structural_placeholder_valid": structural,
                    "sidecar_sha256_matches": True,
                    "runtime_commit_matches_run": True,
                    "runtime_git_dirty": False,
                }
            )
    run_id = "synthetic-qc-a03-layout"
    source_sha = "a" * 64
    return module, module.AuditBundle(
        lineage=module.SourceLineage(run_id, source_sha),
        run_metadata={"run_id": run_id, "easyicu_commit": "synthetic"},
        qc_a01_manifest={
            "status": "passed",
            "failure_count": 0,
            "module_count": 19,
            "source_run_id": run_id,
            "source_run_metadata_sha256": source_sha,
        },
        qc_a02_summary={
            "source_run_id": run_id,
            "source_run_metadata_sha256": source_sha,
        },
        variable_audit=pd.DataFrame(variable_rows),
        cohort_denominators=pd.DataFrame(
            [
                {
                    "database": database,
                    "database_label": module.DATABASE_LABELS[database],
                    "cohort_stays": cohort_counts[database],
                }
                for database in module.DATABASES
            ]
        ),
        field_contract=pd.DataFrame(field_rows),
        module_schema=module_schema,
        manifest_audit=pd.DataFrame(manifest_rows),
        concept_availability=pd.DataFrame(availability_rows),
        distribution_flags=pd.DataFrame(
            [
                {
                    "module": "demographics",
                    "variable": "demographics_measure",
                    "database": "aumc vs sic",
                    "flag": "median_scale_shift",
                    "severity": "high",
                    "evidence": "positive median ratio=200",
                    "origin_classification": "review trigger",
                    "adjudication_status": "source_trace_complete",
                    "adjudicated_origin": "source_recording_heterogeneity",
                }
            ]
        ),
        verified_issues=pd.DataFrame(
            columns=["issue_id", "severity", "classification", "status"]
        ),
        input_hashes={"synthetic_layout_fixture": "b" * 64},
    )


def test_lineage_mismatch_fails_closed() -> None:
    module = _load_script()
    lineage = module.SourceLineage("run-1", "a" * 64)
    with pytest.raises(ValueError, match="lineage mismatch"):
        module._validate_lineage(
            lineage,
            qc_a01_manifest={
                "source_run_id": "run-1",
                "source_run_metadata_sha256": "a" * 64,
            },
            qc_a02_summary={
                "source_run_id": "other-run",
                "source_run_metadata_sha256": "a" * 64,
            },
        )


def test_support_retains_structural_state_and_exact_denominator() -> None:
    module, bundle = _synthetic_bundle()
    module.validate_audit_bundle(bundle)
    support = module.build_module_support(bundle)

    assert support.shape[0] == 19 * 6
    structural = support[
        support["module"].eq("sepsis3") & support["database"].eq("sic")
    ].iloc[0]
    assert structural["availability_state"] == "structurally_unavailable"
    assert structural["concepts_nonempty"] == 0
    assert structural["concepts_declared"] == 2
    assert structural["cell_label"] == "0/2"


def test_heterogeneity_has_complete_inclusion_audit_and_no_fake_interval() -> None:
    module, bundle = _synthetic_bundle()
    result = module.build_heterogeneity_table(bundle, top_n=10)

    hero = result[
        result["module"].eq("demographics")
        & result["variable"].eq("demographics_measure")
    ].iloc[0]
    assert hero["included"]
    assert hero["displayed"]
    assert hero["max_min_median_ratio"] == pytest.approx(200.0)
    assert hero["adjudication_status"] == "source_trace_complete"
    assert "none; descriptive" in hero["interval_definition"]

    signed = result[result["variable"].eq("fluid_balance_cumulative")].iloc[0]
    assert not signed["included"]
    assert signed["exclusion_reason"] == "signed_scale_ratio_not_meaningful"


def test_publication_gate_detects_one_failed_row_grain_receipt() -> None:
    module, bundle = _synthetic_bundle()
    contract = module.build_contract_matrix(bundle)
    gates = module.build_release_gates(bundle)
    assert module.publication_gate_errors(
        bundle, contract_matrix=contract, release_gates=gates
    ) == []

    failed_bundle = copy.deepcopy(bundle)
    failed_bundle.manifest_audit.loc[0, "row_grain_contract_valid"] = False
    failed_gates = module.build_release_gates(failed_bundle)
    errors = module.publication_gate_errors(
        failed_bundle,
        contract_matrix=contract,
        release_gates=failed_gates,
    )
    assert any("row_grain 113/114" in error for error in errors)


def test_synthetic_layout_bundle_is_watermarked_and_never_publication_eligible(
    tmp_path: Path,
) -> None:
    module, bundle = _synthetic_bundle()
    module.validate_audit_bundle(bundle)
    output_dir = tmp_path / "synthetic_qc_a03"
    manifest = module.render_submission_bundle(
        bundle=bundle,
        output_dir=output_dir,
        source_status="synthetic_layout_qa",
        dpi=100,
        top_heterogeneity=8,
    )

    assert manifest["publication_eligible"] is False
    assert manifest["source_status"] == "synthetic_layout_qa"
    assert manifest["denominators"]["database_module_files"] == 114
    for figure in (
        "QC_Fig1_cross_database_observational_support",
        "QC_Fig2_harmonization_reliability",
    ):
        for suffix in (".svg", ".pdf", ".png", ".tiff"):
            assert (output_dir / "figures" / f"{figure}{suffix}").is_file()

    svg = (
        output_dir
        / "figures"
        / "QC_Fig1_cross_database_observational_support.svg"
    ).read_text(encoding="utf-8")
    assert "SYNTHETIC LAYOUT QA" in svg
    assert "<text" in svg

    source = pd.read_csv(
        output_dir / "source_data" / "QC_Fig1a_module_support.csv"
    )
    assert set(source["source_status"]) == {"synthetic_layout_qa"}
    assert set(source["source_run_id"]) == {bundle.lineage.source_run_id}
    saved_manifest = json.loads(
        (output_dir / "figure_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["figure_roles"]["QC-A01"]["role"].startswith(
        "extended_data"
    )


def test_candidate_formal_output_rejects_sub_600_dpi(tmp_path: Path) -> None:
    module, bundle = _synthetic_bundle()
    with pytest.raises(ValueError, match="at least 600 dpi"):
        module.render_submission_bundle(
            bundle=bundle,
            output_dir=tmp_path,
            source_status="candidate",
            dpi=300,
        )
