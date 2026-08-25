from __future__ import annotations

from easyicu.research_agent.reporting.manuscript_post import (
    repair_missing_reportable_survival_results,
)


def _record() -> dict:
    return {
        "step_id": "01_survival",
        "step_summary_evidence_id": "statistic_step_summary_survival",
        "step_summary": {
            "reportable_survival_results": {
                "schema_version": "easyicu.survival_reporting/1",
                "execution_owner": "landmark_survival_executor_v1",
                "constant_hazard_ratio_authorized": False,
                "proportional_hazards_status": "violation_block_paper_authorization",
                "rmst": {
                    "method": "unadjusted_kaplan_meier_plugin",
                    "tau_days_from_landmark": 27.0,
                    "exposed_rmst_days": 24.8413608,
                    "comparator_rmst_days": 25.1460979,
                    "difference_days": -0.3047371,
                    "ci_low": -0.4122846,
                    "ci_high": -0.1971897,
                    "p_value": 2.79869e-8,
                },
            }
        },
    }


def test_projects_complete_owner_issued_rmst_when_writer_omits_values() -> None:
    scaffold = """## Abstract

**Results:** Interval-specific associations were reported.

## Results

### Primary association

The proportional-hazards assumption was rejected.

### Sensitivity and subgroup analyses

Interval-specific results were reported.
"""

    repaired, repairs = repair_missing_reportable_survival_results(
        scaffold, per_step_records=[_record()]
    )

    assert "unadjusted Kaplan–Meier plug-in restricted mean" in repaired
    assert "24.841 days in the exposed group" in repaired
    assert "25.146 days in the comparator group" in repaired
    assert "difference of -0.305 days" in repaired
    assert "95% CI, -0.412 to -0.197" in repaired
    assert "p = 2.80e-08" in repaired
    assert "{evidence:statistic_step_summary_survival}" in repaired
    abstract = repaired.split("## Abstract", 1)[1].split("## Results", 1)[0]
    assert "The unadjusted 27-day restricted mean survival time" in abstract
    assert "24.841 versus 25.146 days" in abstract
    assert repairs[0]["reason_code"] == "owner_reportable_survival_rmst_projected"


def test_survival_projection_is_idempotent_and_rejects_untrusted_owner() -> None:
    scaffold = """## Results

### Primary association

Through 27 days, RMST was 24.841 and 25.146 days, a difference of -0.305
(95% CI, -0.412 to -0.197).

### Sensitivity and subgroup analyses
"""
    unchanged, repairs = repair_missing_reportable_survival_results(
        scaffold, per_step_records=[_record()]
    )
    assert unchanged == scaffold
    assert repairs == []

    untrusted = _record()
    untrusted["step_summary"]["reportable_survival_results"][
        "execution_owner"
    ] = "untrusted"
    omitted = scaffold.replace("24.841", "omitted")
    unchanged, repairs = repair_missing_reportable_survival_results(
        omitted, per_step_records=[untrusted]
    )
    assert unchanged == omitted
    assert repairs == []
