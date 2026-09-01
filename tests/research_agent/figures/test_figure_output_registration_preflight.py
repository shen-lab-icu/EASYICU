from __future__ import annotations

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.reasons import (
    RepairReason,
    repair_reason_for_finding,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair


def _step(ra):
    return ra.AnalysisStep(
        step_id="primary_figure",
        intent="Render the declared figure.",
        inputs=["table:summary"],
        expected_outputs=["figure:primary_figure"],
        method="visualization",
    )


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason")
        == "figure_output_path_missing_extension"
    ]


_BROKEN = """
figure_id = "primary_figure"
figure_files = ["primary_figure.png", "primary_figure.svg"]
summary = {
    "output_files": {"figure:primary_figure": figure_id},
    "figure_files": figure_files,
}
"""


def test_figure_stem_registration_is_flagged(ra):
    findings = _findings(_BROKEN, ra)
    assert len(findings) == 1
    assert repair_reason_for_finding(findings[0]) is (
        RepairReason.FIGURE_OUTPUT_REGISTRATION_INVALID
    )


def test_figure_stem_registration_uses_primary_export(ra):
    findings = _findings(_BROKEN, ra)
    repaired, names = deterministic_concept_audit_repair(
        _BROKEN,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
        step=_step(ra),
    )
    assert names == ["figure_output_registration_v1"]
    assert '"figure:primary_figure": figure_files[0]' in repaired
    assert _findings(repaired, ra) == []


def test_real_figure_filename_registration_is_accepted(ra):
    script = """
figure_files = ["primary_figure.png"]
summary = {
    "output_files": {"figure:primary_figure": figure_files[0]},
    "figure_files": figure_files,
}
"""
    assert _findings(script, ra) == []


def test_nonfigure_output_is_not_overclaimed(ra):
    script = """
figure_id = "primary_figure"
figure_files = ["primary_figure.png"]
summary = {
    "output_files": {"table:summary": figure_id},
    "figure_files": figure_files,
}
"""
    assert _findings(script, ra) == []
