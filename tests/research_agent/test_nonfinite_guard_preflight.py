"""Mechanical non-finite guard routing and deterministic repair regressions."""

from __future__ import annotations

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.reasons import RepairReason
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.schema import AnalysisStep

_STEP = AnalysisStep(
    step_id="cohort_qc",
    intent="Apply an already planned cohort and validate numeric inputs.",
    inputs=["value_a", "value_b"],
    expected_outputs=["table:cohort_flow"],
    method="cohort_definition_and_attrition",
)

_CONDITIONAL_GUARD = """
import numpy as np

def stop(message):
    raise RuntimeError(message)

def validate(series_by_name):
    for name, values in series_by_name:
        nonfinite = values.notna() & ~np.isfinite(values)
        if int(nonfinite.sum()) > 0:
            if name == "value_b":
                stop(f"non-finite values in {name}")
"""


def test_preflight_rejects_variable_conditional_nonfinite_guard() -> None:
    findings = audit_mechanical_code_contracts(_CONDITIONAL_GUARD, _STEP)

    matching = [
        finding
        for finding in findings
        if finding.detail.get("reason") == "conditional_nonfinite_guard"
    ]
    assert len(matching) == 1
    assert matching[0].detail["guard_line"] == 10


def test_deterministic_repair_makes_proven_nonfinite_guard_unconditional() -> None:
    findings = audit_mechanical_code_contracts(_CONDITIONAL_GUARD, _STEP)

    repaired, names = deterministic_concept_audit_repair(
        _CONDITIONAL_GUARD,
        [finding.message for finding in findings],
        repair_reasons=[RepairReason.NONFINITE_NUMERIC_INPUT],
        repair_findings=findings,
    )

    assert names == ["conditional_nonfinite_fail_closed_guard_v1"]
    assert 'if name == "value_b"' not in repaired
    assert "if int(nonfinite.sum()) > 0:" in repaired
    assert audit_mechanical_code_contracts(repaired, _STEP) == []


def test_unconditional_nonfinite_guard_is_unchanged() -> None:
    safe = _CONDITIONAL_GUARD.replace(
        '            if name == "value_b":\n'
        '                stop(f"non-finite values in {name}")',
        '            stop(f"non-finite values in {name}")',
    )

    findings = audit_mechanical_code_contracts(safe, _STEP)
    repaired, names = deterministic_concept_audit_repair(
        safe,
        [finding.message for finding in findings],
        repair_reasons=[RepairReason.NONFINITE_NUMERIC_INPUT],
        repair_findings=findings,
    )

    assert findings == []
    assert repaired == safe
    assert names == []
