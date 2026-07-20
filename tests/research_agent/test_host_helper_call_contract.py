"""Host-owned helper signatures are checked and repaired before sandbox launch."""

from __future__ import annotations

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.reasons import repair_reason_for_finding
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair


def _step(ra):
    return ra.AnalysisStep(
        step_id="measurement_qc",
        intent="Audit declared measurement provenance.",
        inputs=["value_measured", "value_n"],
        expected_outputs=["table:measurement_qc"],
        method="measurement_quality_control",
    )


def _signature_findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "host_helper_call_signature_invalid"
    ]


def _unpack_findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "local_helper_unpack_arity_mismatch"
    ]


def test_positional_keyword_only_host_arguments_fail_before_execution(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(frame, measured_column, count_column)
"""

    findings = _signature_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail == {
        "reason": "host_helper_call_signature_invalid",
        "helper_name": "measurement_provenance_receipt",
        "line": 3,
        "max_positional": 1,
        "required_keywords": ["measured_column", "count_column"],
        "violations": [
            "keyword_only_parameters_passed_positionally",
            "required_keyword_only_argument_missing",
        ],
    }
    assert repair_reason_for_finding(findings[0]).value == ("INVALID_HELPER_SIGNATURE")


def test_alias_import_is_bound_to_the_same_host_contract(ra):
    script = """
import easyicu.research_agent.methods.descriptive_inputs as host_inputs
receipt = host_inputs.measurement_provenance_receipt(frame, measured, count)
"""

    findings = _signature_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail["helper_name"] == "measurement_provenance_receipt"


def test_function_local_host_import_is_bound_to_the_same_contract(ra):
    script = """
def audit(frame, measured, count):
    from easyicu.research_agent.methods.descriptive_inputs import (
        measurement_provenance_receipt,
    )
    return measurement_provenance_receipt(frame, measured, count)
"""

    findings = _signature_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail["line"] == 6


def test_exact_keyword_host_call_passes(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(
    frame,
    measured_column=measured_column,
    count_column=count_column,
)
"""

    assert _signature_findings(script, ra) == []


def test_local_same_name_without_host_import_is_not_claimed(ra):
    script = """
def measurement_provenance_receipt(frame, measured, count):
    return {}

receipt = measurement_provenance_receipt(frame, measured, count)
"""

    assert _signature_findings(script, ra) == []


def test_outer_host_import_shadowed_by_local_binding_is_not_claimed(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt

def audit(measurement_provenance_receipt, frame, measured, count):
    return measurement_provenance_receipt(frame, measured, count)
"""

    assert _signature_findings(script, ra) == []


def test_deterministic_repair_moves_only_existing_names_to_keyword_slots(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
receipt = measurement_provenance_receipt(frame, measured_name, count_name)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["host_helper_keyword_only_call_v1"]
    assert (
        "measurement_provenance_receipt(frame, "
        "measured_column=measured_name, count_column=count_name)"
    ) in repaired
    assert _signature_findings(repaired, ra) == []


def test_deterministic_repair_refuses_ambiguous_same_line_calls(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import measurement_provenance_receipt
first = measurement_provenance_receipt(frame, measured_a, count_a); second = measurement_provenance_receipt(frame, measured_b, count_b)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 2
    assert repaired == script
    assert names == []


def test_closed_counts_requires_explicit_declared_levels_before_execution(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, levels):
    return closed_categorical_counts(series)
"""

    findings = _signature_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "host_helper_call_signature_invalid",
        "helper_name": "closed_categorical_counts",
        "line": 5,
        "max_positional": 1,
        "required_keywords": ["declared_levels"],
        "violations": ["required_keyword_only_argument_missing"],
    }
    assert repair_reason_for_finding(findings[0]).value == ("INVALID_HELPER_SIGNATURE")


def test_closed_counts_explicit_declared_levels_passes(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, levels):
    return closed_categorical_counts(series, declared_levels=levels)
"""

    assert _signature_findings(script, ra) == []


def test_closed_counts_missing_levels_is_repaired_without_inventing_categories(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, levels):
    return closed_categorical_counts(series)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["closed_counts_declared_levels_binding_v1"]
    assert "closed_categorical_counts(series, declared_levels=levels)" in repaired
    assert _signature_findings(repaired, ra) == []


def test_closed_counts_repair_refuses_ambiguous_local_category_parameter(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, categories):
    return closed_categorical_counts(series)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert repaired == script
    assert names == []


def test_closed_counts_unknown_diagnostic_keyword_is_repaired_without_science_change(
    ra,
):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, variable, levels):
    return closed_categorical_counts(
        series,
        variable=variable,
        declared_levels=levels,
    )
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert findings[0].detail["violations"] == ["unknown_keyword_argument"]
    assert names == ["closed_counts_stable_keywords_v1"]
    assert "closed_categorical_counts(series, declared_levels=levels)" in repaired
    assert "variable=variable" not in repaired
    assert _signature_findings(repaired, ra) == []


def test_closed_counts_unknown_keywords_are_repaired_atomically(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def first(series, variable, first_levels):
    return closed_categorical_counts(
        series, variable=variable, levels=first_levels
    )

def second(series, variable, second_levels):
    return closed_categorical_counts(
        series, variable=variable, declared_levels=second_levels
    )
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 2
    assert names == ["closed_counts_stable_keywords_v1"]
    assert repaired.count("variable=variable") == 0
    assert repaired.count("declared_levels=") == 2
    assert _signature_findings(repaired, ra) == []


def test_closed_counts_stable_keyword_repair_refuses_ambiguous_or_unknown_inputs(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, variable, levels, declared_levels):
    return closed_categorical_counts(
        series,
        variable=variable,
        levels=levels,
        declared_levels=declared_levels,
    )
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert repaired == script
    assert names == []


def test_closed_counts_stable_keyword_repair_does_not_bind_reassigned_levels(ra):
    script = """
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def add_categorical(series, variable, levels):
    levels = infer_levels(series)
    return closed_categorical_counts(series, variable=variable)
"""
    findings = _signature_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert len(findings) == 1
    assert repaired == script
    assert names == []


def test_fixed_local_return_arity_must_match_direct_unpack(ra):
    script = """
def collect(frame):
    left = frame["left"]
    right = frame["right"]
    return left, right

def main(frame):
    receipt, left, right = collect(frame)
"""

    findings = _unpack_findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "local_helper_unpack_arity_mismatch",
        "function_name": "collect",
        "call_line": 8,
        "return_lines": [5],
        "return_arity": 2,
        "target_arity": 3,
    }
    assert repair_reason_for_finding(findings[0]).value == ("INVALID_HELPER_SIGNATURE")


def test_dynamic_or_matching_local_returns_are_not_claimed(ra):
    matching = """
def collect(frame):
    return frame["left"], frame["right"]

def main(frame):
    left, right = collect(frame)
"""
    dynamic = """
def collect(frame):
    return make_result(frame)

def main(frame):
    left, right, extra = collect(frame)
"""

    assert _unpack_findings(matching, ra) == []
    assert _unpack_findings(dynamic, ra) == []


def test_deterministic_repair_threads_discarded_host_receipt(ra):
    script = """
def collect(frame, measured_column, count_column):
    from easyicu.research_agent.methods.descriptive_inputs import (
        measurement_provenance_receipt,
    )
    measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )
    measured = frame[measured_column]
    count = frame[count_column]
    return measured, count

def main(frame, measured_column, count_column):
    receipt, measured, count = collect(frame, measured_column, count_column)
"""
    findings = _unpack_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["local_helper_unpack_receipt_v1"]
    assert "receipt = measurement_provenance_receipt(" in repaired
    assert "return receipt, measured, count" in repaired
    assert _unpack_findings(repaired, ra) == []


def test_discarded_receipt_repair_refuses_unaligned_unpack_tail(ra):
    script = """
def collect(frame, measured_column, count_column):
    from easyicu.research_agent.methods.descriptive_inputs import (
        measurement_provenance_receipt,
    )
    measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    )
    measured = frame[measured_column]
    count = frame[count_column]
    return measured, count

def main(frame, measured_column, count_column):
    receipt, count, measured = collect(frame, measured_column, count_column)
"""
    findings = _unpack_findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert repaired == script
    assert names == []
