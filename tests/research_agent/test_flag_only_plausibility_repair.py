from __future__ import annotations

from easyicu.research_agent.repair_registry import RepairClass, repair_metadata_for
from easyicu.research_agent.repairs.reasons import RepairReason
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.schema import ValidationFinding


def _finding(
    *, variable: str = "age", value_class: str = "finite_outside_plausibility_range"
):
    return ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="A flag-only plausibility range was used as an exclusion rule.",
        detail={
            "issue_code": "plausibility_range_exclusion_required",
            "variable": variable,
            "value_class": value_class,
        },
    )


def _repair(code: str, finding: ValidationFinding):
    return deterministic_concept_audit_repair(
        code,
        [finding.message],
        repair_reasons=[RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION],
        repair_findings=[finding],
    )


def test_exact_flag_only_raise_pair_is_removed_without_changing_cohort_rule():
    code = """
age = strict_numeric(df["age"], "age")
age_out_of_domain = (age < 0.0) | (age > 120.0)
if bool(age_out_of_domain.any()):
    raise ValueError("Age outside plausibility range")
adult_mask = age >= 18.0
"""

    repaired, names = _repair(code, _finding())

    assert names == ["flag_only_plausibility_range_retention_v1"]
    assert "age_out_of_domain" not in repaired
    assert "age >= 18.0" in repaired
    assert "_easyicu_flag_only_plausibility_range_retained_v1" in repaired


def test_repair_is_idempotent():
    code = """
age_out_of_domain = (age < 0.0) | (age > 120.0)
if age_out_of_domain.any():
    raise ValueError("Age outside plausibility range")
next_step()
"""
    once, names = _repair(code, _finding())
    twice, second_names = _repair(once, _finding())

    assert names == ["flag_only_plausibility_range_retention_v1"]
    assert twice == once
    assert second_names == []


def test_wrong_value_class_or_variable_is_not_rewritten():
    code = """
age_out_of_domain = (age < 0.0) | (age > 120.0)
if age_out_of_domain.any():
    raise ValueError("Age outside strict domain")
"""

    assert _repair(code, _finding(value_class="strict_domain_violation")) == (code, [])
    assert _repair(code, _finding(variable="lactate")) == (code, [])


def test_mask_with_another_consumer_is_not_rewritten():
    code = """
age_out_of_domain = (age < 0.0) | (age > 120.0)
audit["outlier_n"] = int(age_out_of_domain.sum())
if age_out_of_domain.any():
    raise ValueError("Age outside plausibility range")
"""

    assert _repair(code, _finding()) == (code, [])


def test_guard_with_side_effect_is_not_rewritten():
    code = """
age_out_of_domain = (age < 0.0) | (age > 120.0)
if age_out_of_domain.any():
    audit["flag"] = True
    raise ValueError("Age outside plausibility range")
"""

    assert _repair(code, _finding()) == (code, [])


def test_repair_is_registered_as_structural():
    metadata = repair_metadata_for("flag_only_plausibility_range_retention_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.classification_source == "exact"
