from __future__ import annotations

import pytest

from easyicu.research_agent.audits.model_contrast_reporting import model_contrast_reporting_findings


def test_completed_native_projection_requires_report_contract():
    findings = model_contrast_reporting_findings(
        step_record={"deterministic_standard_analysis": "signed_landmark_spline_robustness"},
        step_summary={"status": "ok", "authority_kind": "signed_landmark_spline_robustness"},
    )
    assert len(findings) == 1 and findings[0].severity == "error"
    assert findings[0].detail["scope"] == "aggregate_reporting_projection"
    assert "sealed parent result tables" in findings[0].detail["repair"]


@pytest.mark.parametrize("kind", [None, "signed_landmark_spline_primary", "grouped_table_one", "absolute_risk_context"])
def test_reporting_gate_does_not_retire_unrelated_completed_analysis(kind):
    assert model_contrast_reporting_findings(
        step_record={"deterministic_standard_analysis": kind},
        step_summary={"status": "ok"},
    ) == []


def test_failed_execution_retains_its_original_diagnostic():
    assert model_contrast_reporting_findings(
        step_record={"deterministic_standard_analysis": "signed_landmark_spline_robustness"},
        step_summary={"status": "failed", "error": "unavailable parent table"},
    ) == []


@pytest.mark.parametrize("envelope", [None, {}, {"schema_version": "unknown"}])
def test_malformed_reporting_contract_fails_closed(envelope):
    findings = model_contrast_reporting_findings(
        step_record={"deterministic_standard_analysis": "signed_landmark_spline_robustness"},
        step_summary={"status": "ok", "reportable_model_contrasts": envelope},
    )
    assert findings and findings[0].severity == "error"
