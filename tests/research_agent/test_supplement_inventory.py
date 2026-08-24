from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.supplement_inventory import (
    write_supplement_inventory,
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
