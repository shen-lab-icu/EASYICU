from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.supplement_inventory import (
    write_supplement_inventory,
)
from easyicu.research_agent.reporting.supplement_package import (
    write_supplement_package,
)


def _register(store: EvidenceStore, evidence_id: str, filename: str) -> None:
    store.register_text(
        kind="table",
        description=filename,
        text="x\n1\n",
        filename=filename,
        evidence_id=evidence_id,
        producer="pipeline",
        generation_mode="system",
    )


def test_supplement_package_binds_registered_files_and_keeps_missing_explicit(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    _register(store, "cohort_accounting", "cohort_accounting.csv")
    inventory, _findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="association_study"),
        evidence=store,
        per_step_records=[],
        run_dir=tmp_path,
    )

    payload = write_supplement_package(
        inventory=inventory,
        evidence=store,
        run_dir=tmp_path,
    )

    cohort = payload["sections"]["cohort_accounting"]
    assert cohort["all_files_digest_verified"] is True
    assert cohort["artifact_bindings"][0]["evidence_id"] == "cohort_accounting"
    missing = payload["sections"]["primary_results"]
    assert missing["present"] is False
    assert missing["artifact_bindings"] == []
    markdown = (tmp_path / "supplement_package.md").read_text(encoding="utf-8")
    assert "no placeholder result was inserted" in markdown
    assert "not causal or external claims" in markdown
    assert (tmp_path / "supplement_evidence_manifest.json").is_file()


def test_prediction_supplement_inventory_names_missing_scientific_sections(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    for evidence_id, filename in (
        ("cohort_accounting", "cohort_accounting.csv"),
        ("table_one", "table_one.csv"),
        ("missingness", "missingness.csv"),
        ("primary_model", "primary_model.csv"),
        ("robustness", "robustness.csv"),
        ("figure_contract", "figure_contract.json"),
        ("runner_provenance", "runner_provenance.json"),
        ("calibration", "calibration.csv"),
        ("roc", "roc_curve.csv"),
        ("validation", "heldout_validation.csv"),
    ):
        _register(store, evidence_id, filename)

    payload, findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="prediction"),
        evidence=store,
        per_step_records=[],
        run_dir=tmp_path,
    )

    assert payload["supplement_complete"] is False
    assert payload["development_supplement_complete"] is False
    assert payload["top_journal_supplement_complete"] is False
    assert payload["missing_required_sections"] == [
        "clinical_utility",
        "resampling_validation",
        "external_validation",
    ]
    assert findings[0].validator == "supplement_inventory"
    assert (tmp_path / "supplement_inventory.json").is_file()
    assert (tmp_path / "supplement_inventory.md").is_file()
    on_disk = json.loads((tmp_path / "supplement_inventory.json").read_text())
    assert on_disk["missing_required_sections"] == [
        "clinical_utility",
        "resampling_validation",
        "external_validation",
    ]


def test_typed_output_can_close_the_clinical_utility_section(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    for evidence_id, filename in (
        ("cohort", "cohort.csv"),
        ("baseline", "baseline.csv"),
        ("missing", "missing.csv"),
        ("primary", "primary.csv"),
        ("sensitivity", "sensitivity.csv"),
        ("figure", "figure_contract.json"),
        ("provenance", "runner_provenance.json"),
        ("calibration", "calibration.csv"),
        ("performance", "performance.csv"),
        ("validation", "validation.csv"),
        ("resampling", "bootstrap_validation.csv"),
        ("external_validation", "external_validation_cohort.csv"),
    ):
        _register(store, evidence_id, filename)
    records = [
        {
            "step_id": "clinical_utility",
            "step_summary": {
                "method": "decision_curve",
                "output_files": {"table:clinical_utility": "decision_curve.csv"},
            },
        }
    ]

    payload, findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="prediction"),
        evidence=store,
        per_step_records=records,
        run_dir=tmp_path,
    )

    assert payload["supplement_complete"] is True
    assert payload["development_supplement_complete"] is True
    assert payload["top_journal_supplement_complete"] is True
    assert findings == []


def test_prediction_model_alias_uses_prediction_supplement_contract(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    payload, _findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="prediction_model"),
        evidence=store,
        per_step_records=[],
        run_dir=tmp_path,
    )

    assert payload["analysis_family"] == "prediction"
    assert "external_validation" in payload["required_sections"]
    assert "resampling_validation" in payload["required_sections"]
    assert "external_validation" not in payload["development_required_sections"]
    assert "external_validation" in payload["top_journal_required_sections"]


def test_registered_repeated_split_model_output_closes_only_internal_resampling(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    records = [
        {
            "step_id": "primary_prediction",
            "step_summary": {
                "method": (
                    "deterministic_static_prediction_model_with_"
                    "repeated_split_validation"
                ),
                "output_files": {
                    "table:model_performance": "prediction_performance.csv"
                },
            },
        }
    ]

    payload, _findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="prediction"),
        evidence=store,
        per_step_records=records,
        run_dir=tmp_path,
    )

    assert payload["sections"]["resampling_validation"]["present"] is True
    assert payload["sections"]["external_validation"]["present"] is False
    assert "external_validation" in payload["missing_required_sections"]
    assert payload["development_supplement_complete"] is False


def test_external_validation_is_not_a_dev9_supplement_prerequisite(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    for evidence_id, filename in (
        ("cohort", "cohort.csv"),
        ("baseline", "baseline.csv"),
        ("missing", "missing.csv"),
        ("primary", "primary.csv"),
        ("sensitivity", "sensitivity.csv"),
        ("figure", "figure_contract.json"),
        ("provenance", "runner_provenance.json"),
        ("calibration", "calibration.csv"),
        ("performance", "performance.csv"),
        ("validation", "heldout_validation.csv"),
        ("clinical_utility", "decision_curve.csv"),
        ("resampling", "repeated_split_validation.csv"),
    ):
        _register(store, evidence_id, filename)

    payload, _findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="prediction"),
        evidence=store,
        per_step_records=[],
        run_dir=tmp_path,
    )

    assert payload["development_supplement_complete"] is True
    assert payload["top_journal_supplement_complete"] is False
    assert payload["missing_development_required_sections"] == []
    assert payload["missing_top_journal_required_sections"] == [
        "external_validation"
    ]


def test_external_provider_privacy_audit_is_not_external_reproducibility(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    _register(
        store,
        "privacy_audit",
        "external_provider_privacy_audit.json",
    )

    payload, _findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="trajectory_clustering"),
        evidence=store,
        per_step_records=[],
        run_dir=tmp_path,
    )

    assert payload["sections"]["external_reproducibility"]["present"] is False
    assert "external_reproducibility" in payload["missing_required_sections"]


def test_algorithm_agreement_output_closes_only_alternative_algorithm(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    records = [
        {
            "step_id": "cluster_stability",
            "step_summary": {
                "method": "deterministic_cross_sectional_phenotyping_diagnostic",
                "output_files": {
                    "table:cluster_stability": (
                        "cluster_stability_with_algorithm_agreement.csv"
                    )
                },
            },
        }
    ]

    payload, _findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="trajectory_clustering"),
        evidence=store,
        per_step_records=records,
        run_dir=tmp_path,
    )

    assert payload["sections"]["alternative_algorithm"]["present"] is True
    assert payload["sections"]["external_reproducibility"]["present"] is False


def test_association_dev_coverage_does_not_claim_top_journal_replication(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    for evidence_id, filename in (
        ("cohort", "cohort_flow.csv"),
        ("baseline", "table_one.csv"),
        ("missing", "missingness.csv"),
        ("primary", "association_estimate.csv"),
        ("sensitivity", "sensitivity.csv"),
        ("figure", "figure_contract.json"),
        ("provenance", "runner_provenance.json"),
    ):
        _register(store, evidence_id, filename)

    payload, _findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="association_study"),
        evidence=store,
        per_step_records=[],
        run_dir=tmp_path,
    )

    assert payload["development_supplement_complete"] is True
    assert payload["top_journal_supplement_complete"] is False
    assert payload["missing_top_journal_required_sections"] == [
        "external_reproducibility"
    ]


def test_source_feasibility_has_a_terminal_dev_supplement_contract(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    records = [
        {
            "step_id": "source_feasibility",
            "step_summary": {
                "analysis_family": "causal_feasibility",
                "scientific_decision": "blocked_by_source_authority",
                "causal_contrast_authorized": False,
                "effect_estimate": None,
                "output_files": {
                    "table:source_feasibility": "source_feasibility.csv",
                    "log:source_feasibility_receipt": "source_feasibility_receipt.json",
                },
            },
        }
    ]

    payload, _findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="causal_inference"),
        evidence=store,
        per_step_records=records,
        run_dir=tmp_path,
    )

    assert payload["terminal_disposition"] == "source_feasibility_fail_closed"
    assert payload["development_supplement_complete"] is True
    assert payload["top_journal_supplement_complete"] is False
    assert payload["missing_top_journal_required_sections"] == [
        "identified_comparator"
    ]


def test_no_interior_trajectory_solution_uses_a_terminal_dev_contract(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    records = [
        {
            "step_id": "trajectory_candidates",
            "step_summary": {
                "scientific_status": "failed_closed",
                "reason_code": "NO_INTERIOR_OPTIMUM",
                "reportable_result": (
                    "no_interior_solution_in_prespecified_candidate_range"
                ),
                "output_files": {
                    "table:cohort_flow": "cohort_flow.csv",
                    "table:feature_missingness": "feature_missingness.csv",
                    "table:cluster_selection": "cluster_selection.csv",
                    "figure:selection": "selection.figure_contract.json",
                    "log:runtime_receipt": "runtime_receipt.json",
                },
            },
        }
    ]

    payload, _findings = write_supplement_inventory(
        plan=SimpleNamespace(analysis_type="trajectory_clustering"),
        evidence=store,
        per_step_records=records,
        run_dir=tmp_path,
    )

    assert payload["terminal_disposition"] == (
        "prespecified_selection_no_solution"
    )
    assert payload["development_supplement_complete"] is True
    assert payload["top_journal_supplement_complete"] is False
    assert payload["missing_top_journal_required_sections"] == [
        "baseline_characteristics",
        "alternative_algorithm",
        "external_reproducibility",
    ]
