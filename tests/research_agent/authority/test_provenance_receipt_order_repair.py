from __future__ import annotations

import ast

from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.repairs.reasons import repair_reason_for_finding
from easyicu.research_agent.schema import ValidationFinding

CODE = """
import pandas as pd
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt

def main(input_path, output_dir):
    frame = pd.read_parquet(input_path)
    table = frame[["value"]].copy()
    table.to_csv(output_dir / "audit.csv", index=False)
    frame.to_parquet(output_dir / "cohort.parquet", index=False)
    step_summary = {
        "measurement_provenance_audit": {
            "source": "COHORT_PARQUET",
            "checks": [measurement_provenance_receipt(
                frame,
                measured_column="value_measured",
                count_column="value_n",
            )],
        }
    }
    return step_summary
"""


def _finding() -> ValidationFinding:
    return ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="receipt runs after outputs",
        detail={"issue_code": "audit_only_companion_row_gating_required"},
    )


def test_receipt_moves_before_first_output_without_changing_arguments() -> None:
    finding = _finding()
    repaired, names = deterministic_concept_audit_repair(
        CODE,
        [finding.message],
        repair_reasons=[repair_reason_for_finding(finding)],
        repair_findings=[finding],
    )

    assert names == ["measurement_provenance_before_outputs_v1"]
    receipt = repaired.index("_easyicu_measurement_provenance_receipt_v1 =")
    assert receipt < repaired.index("table.to_csv(")
    assert receipt < repaired.index("frame.to_parquet(")
    assert '"checks": [_easyicu_measurement_provenance_receipt_v1]' in repaired
    tree = ast.parse(repaired)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "measurement_provenance_receipt"
    ]
    assert len(calls) == 1
    assert isinstance(calls[0].args[0], ast.Name)
    assert calls[0].args[0].id == "frame"
    assert {
        keyword.arg: keyword.value.value
        for keyword in calls[0].keywords
        if keyword.arg and isinstance(keyword.value, ast.Constant)
    } == {
        "measured_column": "value_measured",
        "count_column": "value_n",
    }


def test_receipt_order_repair_requires_exact_structured_finding() -> None:
    repaired, names = deterministic_concept_audit_repair(
        CODE,
        ["receipt runs after outputs"],
    )
    assert repaired == CODE
    assert names == []


def test_receipt_order_repair_refuses_multiple_receipts() -> None:
    ambiguous = CODE.replace(
        '"checks": [measurement_provenance_receipt(',
        '"checks": [measurement_provenance_receipt(frame, measured_column="other_measured", count_column="other_n"), measurement_provenance_receipt(',
    )
    finding = _finding()
    repaired, names = deterministic_concept_audit_repair(
        ambiguous,
        [finding.message],
        repair_reasons=[repair_reason_for_finding(finding)],
        repair_findings=[finding],
    )
    assert repaired == ambiguous
    assert names == []


def test_audit_only_companion_selector_uses_value_owned_mask() -> None:
    code = """
support_mask = valid_pair_mask & (measured_series == 1) & (n_series > 0)
finite_lact_mask = lact_series.notna()
in_range_mask = finite_lact_mask & lact_series.between(0.0, 30.0)
valid_observed_mask = support_mask & in_range_mask
status_masks = {"valid observed": valid_observed_mask}
valid_distribution = lact_series.loc[valid_observed_mask].astype(float)
"""
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="audit-only companions gate the value denominator",
        detail={
            "issue_code": "audit_only_companion_row_gating_required",
            "variables": [
                "support_mask",
                "valid_observed_mask",
                "valid_distribution",
            ],
        },
    )

    repaired, names = deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[repair_reason_for_finding(finding)],
        repair_findings=[finding],
    )

    assert names == ["audit_only_companion_value_selector_v1"]
    assert "lact_series.loc[in_range_mask].astype(float)" in repaired
    assert 'status_masks = {"valid observed": valid_observed_mask}' in repaired


def test_audit_only_companion_selector_refuses_ambiguous_value_masks() -> None:
    code = """
mask_a = lact_series.notna()
mask_b = lact_series.between(0.0, 30.0)
combined_mask = mask_a & mask_b
valid_distribution = lact_series.loc[combined_mask]
"""
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="ambiguous selector",
        detail={
            "issue_code": "audit_only_companion_row_gating_required",
            "variables": ["combined_mask", "valid_distribution"],
        },
    )

    repaired, names = deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[repair_reason_for_finding(finding)],
        repair_findings=[finding],
    )

    assert repaired == code
    assert names == []
