from __future__ import annotations

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.reasons import (
    RepairReason,
    repair_reason_for_finding,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair


def _step(ra):
    return ra.AnalysisStep(
        step_id="provenance_audit",
        intent="Validate declared provenance companions.",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:provenance_audit"],
        method="measurement_quality_control",
    )


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "string_suffix_trim_length_mismatch"
    ]


def test_literal_suffix_trim_mismatch_is_repaired_without_provider_call(ra):
    script = """
names = {name[:-10] for name in columns if name.endswith("_measured")}
"""
    findings = _findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "string_suffix_trim_length_mismatch",
        "line": 2,
        "variable": "name",
        "reported": 10,
        "expected": 9,
        "suffix": "_measured",
    }
    assert (
        repair_reason_for_finding(findings[0])
        is RepairReason.STRING_SUFFIX_TRIM_MISMATCH
    )

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["string_suffix_trim_length_v1"]
    assert 'name[:-9] for name in columns if name.endswith("_measured")' in repaired
    assert _findings(repaired, ra) == []


def test_correct_or_dynamic_suffix_trim_is_not_claimed(ra):
    correct = """
names = {name[:-9] for name in columns if name.endswith("_measured")}
"""
    dynamic = """
names = {name[:-10] for name in columns if name.endswith(suffix)}
"""

    assert _findings(correct, ra) == []
    assert _findings(dynamic, ra) == []


def test_ambiguous_literal_suffixes_are_not_host_repaired(ra):
    script = """
names = {
    name[:-10]
    for name in columns
    if name.endswith("_measured") or name.endswith("_measurement")
}
"""

    assert _findings(script, ra) == []


def test_multiple_suffix_trim_mismatches_are_repaired_atomically(ra):
    script = """
measured = {name[:-10] for name in columns if name.endswith("_measured")}
counts = {name[:-3] for name in columns if name.endswith("_n")}
"""
    findings = _findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 2
    assert names == ["string_suffix_trim_length_v1"]
    assert "name[:-9]" in repaired
    assert "name[:-2]" in repaired
    assert _findings(repaired, ra) == []


def test_suffix_trim_repair_registry_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("string_suffix_trim_length_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)
