from __future__ import annotations

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.reasons import (
    RepairReason,
    repair_reason_for_finding,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair


def _step(ra):
    return ra.AnalysisStep(
        step_id="figure_summary",
        intent="Render a declared figure and summarize its denominator.",
        inputs=["table:cohort_flow"],
        expected_outputs=["figure:summary"],
        method="visualization",
    )


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason")
        == "literal_mapping_index_then_key"
    ]


_BROKEN = """
flow_values = {
    "n_before": 100,
    "n_excluded": 4,
    "n_remaining": 96,
}
summary = {"cohort_n_remaining": int(flow_values[0]["n_remaining"])}
"""


def test_known_literal_mapping_field_cannot_follow_integer_index(ra):
    findings = _findings(_BROKEN, ra)
    assert len(findings) == 1
    assert repair_reason_for_finding(findings[0]) is RepairReason.INVALID_CONTAINER_ACCESS


def test_literal_mapping_access_is_repaired_without_changing_value(ra):
    findings = _findings(_BROKEN, ra)
    repaired, names = deterministic_concept_audit_repair(
        _BROKEN,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
        step=_step(ra),
    )
    assert names == ["literal_mapping_access_v1"]
    assert "flow_values['n_remaining']" in repaired
    assert _findings(repaired, ra) == []
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    assert namespace["summary"] == {"cohort_n_remaining": 96}


def test_nested_access_on_a_real_sequence_is_not_overclaimed(ra):
    script = """
flow_values = [{"n_remaining": 96}]
summary = {"cohort_n_remaining": flow_values[0]["n_remaining"]}
"""
    assert _findings(script, ra) == []


def test_direct_literal_mapping_key_access_is_accepted(ra):
    script = """
flow_values = {"n_remaining": 96}
summary = {"cohort_n_remaining": flow_values["n_remaining"]}
"""
    assert _findings(script, ra) == []
