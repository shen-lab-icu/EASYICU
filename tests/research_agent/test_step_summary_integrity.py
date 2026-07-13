from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.audits import (
    StepSummaryFractionValidator,
    StepSummaryIntegrityValidator,
)
from easyicu.research_agent.schema import AnalysisStep


REFERENCE_KEY = "artifact:reference_rows"
SUBSET_KEY = "artifact:subset_rows"


def _step() -> AnalysisStep:
    return AnalysisStep(step_id="04_verify_subset", intent="Verify a typed subset.")


def _resolved_bindings(
    tmp_path: Path,
    *,
    subset_values: tuple[float, float] = (10.0, 30.0),
    subset_secondary_values: tuple[float, float] = (100.0, 300.0),
) -> dict[str, dict[str, str]]:
    reference_path = tmp_path / "reference.parquet"
    subset_path = tmp_path / "subset.parquet"
    pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "measurement": [10.0, 20.0, 30.0],
            "secondary_measurement": [100.0, 200.0, 300.0],
        }
    ).to_parquet(reference_path, index=False)
    pd.DataFrame(
        {
            "record_id": [1, 3],
            "measurement": list(subset_values),
            "secondary_measurement": list(subset_secondary_values),
        }
    ).to_parquet(subset_path, index=False)
    return {
        REFERENCE_KEY: {
            "absolute_path": str(reference_path),
            "evidence_id": "table_reference",
            "sha256": "a" * 64,
        },
        SUBSET_KEY: {
            "absolute_path": str(subset_path),
            "evidence_id": "table_subset",
            "sha256": "b" * 64,
        },
    }


def _truthful_summary() -> dict:
    return {
        "input_bindings": [
            {
                "input_key": REFERENCE_KEY,
                "evidence_id": "table_reference",
                "sha256": "a" * 64,
                "loaded": True,
                "row_count": 3,
            },
            {
                "input_key": SUBSET_KEY,
                "evidence_id": "table_subset",
                "sha256": "b" * 64,
                "loaded": True,
                "row_count": 2,
            },
        ],
        "subset_input": {
            "artifact": SUBSET_KEY,
            "loaded": True,
            "n_rows": 2,
        },
        "subset_reconciliation": {
            "status": "checked",
            "reference_artifact": REFERENCE_KEY,
            "subset_artifact": SUBSET_KEY,
            "key_columns": ["record_id"],
            "value_columns_checked": ["measurement", "secondary_measurement"],
            "value_mismatch_n": 0,
        },
    }


def test_integrity_rejects_nested_loaded_and_row_count_contradiction(
    tmp_path: Path,
) -> None:
    summary = _truthful_summary()
    summary["subset_input"].update({"loaded": False, "n_rows": 7})

    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=summary,
        resolved_input_bindings=_resolved_bindings(tmp_path),
    )

    issues = {finding.detail["issue"] for finding in findings}
    assert "nested_input_declaration_contradiction" in issues
    assert "checked_reconciliation_unloaded_input" in issues


def test_integrity_rejects_checked_reconciliation_without_value_evidence(
    tmp_path: Path,
) -> None:
    summary = _truthful_summary()
    summary["subset_reconciliation"].pop("value_columns_checked")
    summary["subset_reconciliation"].pop("value_mismatch_n")

    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=summary,
        resolved_input_bindings=_resolved_bindings(
            tmp_path, subset_values=(10.0, 999.0)
        ),
    )

    incomplete = [
        finding
        for finding in findings
        if finding.detail["issue"] == "checked_reconciliation_evidence_incomplete"
    ]
    assert len(incomplete) == 1
    assert set(incomplete[0].detail["invalid_fields"]) == {
        "value_columns_checked",
        "value_mismatch_n",
    }


def test_integrity_rejects_self_comparison_as_subset_reconciliation(
    tmp_path: Path,
) -> None:
    summary = _truthful_summary()
    summary["subset_reconciliation"]["subset_artifact"] = REFERENCE_KEY

    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=summary,
        resolved_input_bindings=_resolved_bindings(tmp_path),
    )

    incomplete = [
        finding
        for finding in findings
        if finding.detail["issue"] == "checked_reconciliation_evidence_incomplete"
    ]
    assert len(incomplete) == 1
    assert "distinct_reference_and_subset_artifacts" in incomplete[0].detail[
        "invalid_fields"
    ]


def test_integrity_host_replay_rejects_same_keys_with_wrong_values(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=_truthful_summary(),
        resolved_input_bindings=_resolved_bindings(
            tmp_path, subset_values=(10.0, 999.0)
        ),
    )

    mismatch = [
        finding
        for finding in findings
        if finding.detail["issue"] == "checked_reconciliation_host_value_mismatch"
    ]
    assert len(mismatch) == 1
    assert mismatch[0].detail["value_mismatch_cell_n"] == 1
    assert mismatch[0].detail["value_mismatch_row_n"] == 1


def test_integrity_rejects_a_convenient_subset_of_shared_value_columns(
    tmp_path: Path,
) -> None:
    summary = _truthful_summary()
    summary["subset_reconciliation"]["value_columns_checked"] = ["measurement"]

    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=summary,
        resolved_input_bindings=_resolved_bindings(tmp_path),
    )

    incomplete = [
        finding
        for finding in findings
        if finding.detail["issue"]
        == "checked_reconciliation_value_scope_incomplete"
    ]
    assert len(incomplete) == 1
    assert incomplete[0].detail["omitted_value_columns"] == [
        "secondary_measurement"
    ]


def test_integrity_accepts_truthful_host_verified_reconciliation(
    tmp_path: Path,
) -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary=_truthful_summary(),
        resolved_input_bindings=_resolved_bindings(tmp_path),
    )

    assert findings == []


def test_integrity_does_not_claim_unrelated_checked_qc() -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary={
            "component_qc": {
                "status": "checked",
                "comparison_n": 100,
                "discordant_n": 0,
            }
        },
        resolved_input_bindings={},
    )

    assert findings == []


def test_integrity_does_not_claim_untyped_component_reconciliation() -> None:
    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary={
            "component_reconciliation": {
                "status": "checked",
                "comparison_n": 100,
                "discordant_n": 0,
            }
        },
        resolved_input_bindings={},
    )

    assert findings == []


def test_non_tabular_input_binding_does_not_require_row_count(tmp_path: Path) -> None:
    figure_path = tmp_path / "figure.svg"
    figure_path.write_text("<svg/>", encoding="utf-8")

    findings = StepSummaryIntegrityValidator().audit(
        step=_step(),
        step_summary={
            "input_bindings": [
                {
                    "input_key": "figure:registered_plot",
                    "evidence_id": "figure_registered",
                    "sha256": "c" * 64,
                    "loaded": True,
                }
            ]
        },
        resolved_input_bindings={
            "figure:registered_plot": {
                "absolute_path": str(figure_path),
                "evidence_id": "figure_registered",
                "sha256": "c" * 64,
                "evidence_kind": "figure",
            }
        },
    )

    assert findings == []


def test_probability_ci_roundoff_overflow_is_not_silently_accepted() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=_step(),
        step_summary={"prevalence_ci_high": 1.0000000000000002},
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["issue"] == "bounded_metric_out_of_range"
    assert findings[0].detail["metric_kind"] == "prevalence"
    assert findings[0].detail["roundoff_sized_overflow"] is True


def test_nested_risk_ci_roundoff_overflow_is_not_silently_accepted() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=_step(),
        step_summary={
            "risk": {
                "estimate": 0.2,
                "ci_low": 0.1,
                "ci_high": 1.0000000000000002,
            }
        },
    )

    assert len(findings) == 1
    assert findings[0].detail["summary_path"] == "risk.ci_high"
    assert findings[0].detail["metric_kind"] == "risk"


def test_effect_scale_estimates_and_intervals_are_not_probability_bounded() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=_step(),
        step_summary={
            "risk_ratio": 1.8,
            "risk_ratio_ci_low": 1.4,
            "risk_ratio_ci_high": 2.2,
            "relative_risk": 1.8,
            "relative_risk_ci_high": 2.2,
            "odds_ratio": 2.4,
            "hazard_ratio": 1.7,
            "risk_difference": -0.2,
            "number_at_risk": 120,
            "patients_at_risk": 118,
            "at_risk": "reported",
        },
    )

    assert findings == []


def test_generic_interval_is_not_assigned_to_risk_when_effect_scale_is_present() -> None:
    findings = StepSummaryFractionValidator().audit(
        step=_step(),
        step_summary={
            "absolute_risk": 0.2,
            "risk_ratio": 1.8,
            "ci_low": 1.4,
            "ci_high": 2.2,
        },
    )

    assert findings == []
