from __future__ import annotations

from easyicu.research_agent.repairs.provenance_summary import (
    patch_companion_audit_frame_alignment,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.schema import ValidationFinding


def _finding(
    *,
    audit_frame: str = "locked_df",
    analysis_frame: str = "typed_df",
) -> ValidationFinding:
    return ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Companion receipt does not use the analyzed rows.",
        detail={
            "issue_code": "audit_only_companion_row_gating_required",
            "audit_frame": audit_frame,
            "analysis_frame": analysis_frame,
            "variables": ["exposure", "exposure_measured", "exposure_n"],
        },
    )


def test_aligns_transparent_receipt_wrapper_to_bound_analysis_frame() -> None:
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

def call_receipt(frame, measured_column, count_column):
    return measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )

def main():
    typed_df, binding, path = load_bound_table(
        manifest, "artifact:analysis_cohort"
    )
    locked_df = pd.read_parquet(cohort_path)
    checks = []
    for measured, count in pairs:
        checks.append(call_receipt(locked_df, measured, count))
    result = typed_df[["exposure"]].copy()
"""

    repaired = patch_companion_audit_frame_alignment(
        code,
        findings=[_finding()],
    )

    assert "call_receipt(typed_df, measured, count)" in repaired
    assert "locked_df = pd.read_parquet(cohort_path)" in repaired
    assert repaired.count("measurement_provenance_receipt(") == 1


def test_refuses_alignment_without_exact_bound_analysis_cohort() -> None:
    code = """
def main():
    typed_df = pd.read_parquet(other_path)
    locked_df = pd.read_parquet(cohort_path)
    measurement_provenance_receipt(
        locked_df,
        measured_column="exposure_measured",
        count_column="exposure_n",
    )
"""

    assert patch_companion_audit_frame_alignment(code, findings=[_finding()]) == code


def test_refuses_nontransparent_local_wrapper() -> None:
    code = """
def call_receipt(frame, measured_column, count_column):
    frame = frame.loc[frame[measured_column] == 1]
    return custom_receipt(frame, measured_column, count_column)

def main():
    typed_df, binding, path = load_bound_table(
        manifest, "artifact:analysis_cohort"
    )
    locked_df = pd.read_parquet(cohort_path)
    call_receipt(locked_df, "exposure_measured", "exposure_n")
"""

    assert patch_companion_audit_frame_alignment(code, findings=[_finding()]) == code


def test_alignment_runs_inside_deterministic_concept_repair() -> None:
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

def call_receipt(frame, measured_column, count_column):
    return measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )

def main():
    typed_df, binding, path = load_bound_table(
        manifest, "artifact:analysis_cohort"
    )
    locked_df = pd.read_parquet(cohort_path)
    call_receipt(locked_df, "exposure_measured", "exposure_n")
"""
    finding = _finding()

    repaired, repair_names = deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_findings=[finding],
    )

    assert "call_receipt(typed_df," in repaired
    assert repair_names == ["audit_only_companion_value_selector_v1"]
