"""Mechanical integrity gates for descriptive numeric/categorical summaries."""

from __future__ import annotations

import pandas as pd
import pytest

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
from easyicu.research_agent.schema import AnalysisStep

_STEP = AnalysisStep(
    step_id="descriptive_table",
    intent="Summarize the already locked analysis cohort.",
    inputs=["cohort:analysis_set"],
    expected_outputs=["table:baseline_characteristics"],
    method="table_one",
)

_STRICT_NUMERIC = """
import numpy as np
import pandas as pd

def strict_input(series):
    original = series.copy()
    coerced = pd.to_numeric(original, errors="coerce")
    newly_invalid = int((original.notna() & coerced.isna()).sum())
    if newly_invalid > 0:
        raise RuntimeError("lossy coercion")
    return coerced
"""

_CATEGORICAL_SUMMARY = """
def add_categories(rows, series, levels):
    nonmissing = series.dropna()
    denominator = int(nonmissing.shape[0])
    counts = nonmissing.value_counts(dropna=False)
    for level in levels:
        rows.append({
            "category": str(level),
            "count": int(counts.get(level, 0)),
            "denominator": denominator,
        })
"""


def _findings(code: str, reason: str):
    return [
        finding
        for finding in audit_mechanical_code_contracts(code, _STEP)
        if (finding.detail or {}).get("reason") == reason
    ]


def test_strict_numeric_helper_requires_nonfinite_fail_closed_guard() -> None:
    findings = _findings(
        _STRICT_NUMERIC,
        "strict_numeric_nonfinite_unchecked",
    )

    assert len(findings) == 1
    assert (
        repair_reason_for_finding(findings[0]) is RepairReason.NONFINITE_NUMERIC_INPUT
    )


def test_strict_numeric_nonfinite_guard_is_deterministically_repaired() -> None:
    findings = audit_mechanical_code_contracts(_STRICT_NUMERIC, _STEP)
    repaired, names = deterministic_concept_audit_repair(
        _STRICT_NUMERIC,
        [finding.message for finding in findings],
        repair_reasons=[RepairReason.NONFINITE_NUMERIC_INPUT],
        repair_findings=findings,
    )

    assert names == ["strict_numeric_nonfinite_guard_v1"]
    assert audit_mechanical_code_contracts(repaired, _STEP) == []
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    strict_input = namespace["strict_input"]
    assert callable(strict_input)
    output = strict_input(pd.Series([1.0, None]))
    assert output.isna().sum() == 1
    with pytest.raises(RuntimeError, match="non-finite"):
        strict_input(pd.Series([1.0, float("inf")]))


def test_multiple_strict_numeric_helpers_are_repaired_in_one_pass() -> None:
    two_helpers = _STRICT_NUMERIC + _STRICT_NUMERIC.replace(
        "strict_input", "strict_secondary"
    )
    findings = audit_mechanical_code_contracts(two_helpers, _STEP)

    repaired, names = deterministic_concept_audit_repair(
        two_helpers,
        [finding.message for finding in findings],
        repair_reasons=[RepairReason.NONFINITE_NUMERIC_INPUT],
        repair_findings=findings,
    )

    assert names == ["strict_numeric_nonfinite_guard_v1"]
    assert audit_mechanical_code_contracts(repaired, _STEP) == []
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    for helper_name in ("strict_input", "strict_secondary"):
        helper = namespace[helper_name]
        assert callable(helper)
        with pytest.raises(RuntimeError, match="non-finite"):
            helper(pd.Series([1.0, float("inf")]))


def test_unrelated_isfinite_call_does_not_satisfy_strict_helper() -> None:
    decoy = _STRICT_NUMERIC + """

def unrelated(values):
    if (~np.isfinite(values)).any():
        raise RuntimeError("unrelated")
"""

    assert len(_findings(decoy, "strict_numeric_nonfinite_unchecked")) == 1


def test_wrong_direction_nonfinite_guard_does_not_pass() -> None:
    wrong = _STRICT_NUMERIC.replace(
        "    return coerced",
        "    invalid = ~np.isfinite(coerced)\n"
        "    if int(invalid.sum()) == 0:\n"
        "        raise RuntimeError('wrong direction')\n"
        "    return coerced",
    )

    assert len(_findings(wrong, "strict_numeric_nonfinite_unchecked")) == 1


def test_unreachable_nonfinite_guard_after_return_does_not_pass() -> None:
    unreachable = _STRICT_NUMERIC.replace(
        "    return coerced",
        "    return coerced\n"
        "    invalid = ~np.isfinite(coerced)\n"
        "    if invalid.any():\n"
        "        raise RuntimeError('unreachable')",
    )

    assert len(_findings(unreachable, "strict_numeric_nonfinite_unchecked")) == 1


def test_categorical_summary_requires_declared_level_reconciliation() -> None:
    findings = _findings(
        _CATEGORICAL_SUMMARY,
        "categorical_level_accounting_unverified",
    )

    assert len(findings) == 1
    assert (
        repair_reason_for_finding(findings[0])
        is RepairReason.STRUCTURAL_ACCOUNTING_INVALID
    )


def test_wrong_direction_categorical_guard_does_not_pass() -> None:
    wrong = _CATEGORICAL_SUMMARY.replace(
        "    counts = nonmissing.value_counts(dropna=False)",
        "    uncovered = ~nonmissing.isin(levels)\n"
        "    if int(uncovered.sum()) == 0:\n"
        "        raise RuntimeError('wrong direction')\n"
        "    counts = nonmissing.value_counts(dropna=False)",
    )

    assert len(_findings(wrong, "categorical_level_accounting_unverified")) == 1


def test_categorical_level_reconciliation_is_deterministically_repaired() -> None:
    findings = audit_mechanical_code_contracts(_CATEGORICAL_SUMMARY, _STEP)
    repaired, names = deterministic_concept_audit_repair(
        _CATEGORICAL_SUMMARY,
        [finding.message for finding in findings],
        repair_reasons=[RepairReason.STRUCTURAL_ACCOUNTING_INVALID],
        repair_findings=findings,
    )

    assert names == ["categorical_level_reconciliation_guard_v1"]
    assert audit_mechanical_code_contracts(repaired, _STEP) == []
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    add_categories = namespace["add_categories"]
    assert callable(add_categories)
    rows: list[dict[str, object]] = []
    add_categories(rows, pd.Series(["A", "B", None]), ["A", "B"])
    assert sum(int(row["count"]) for row in rows) == 2
    with pytest.raises(RuntimeError, match="declared levels"):
        add_categories([], pd.Series(["A", "unexpected"]), ["A", "B"])


def test_both_descriptive_integrity_repairs_apply_in_one_atomic_pass() -> None:
    code = _STRICT_NUMERIC + "\n" + _CATEGORICAL_SUMMARY
    findings = audit_mechanical_code_contracts(code, _STEP)
    reasons = list(dict.fromkeys(repair_reason_for_finding(item) for item in findings))

    repaired, names = deterministic_concept_audit_repair(
        code,
        [finding.message for finding in findings],
        repair_reasons=reasons,
        repair_findings=findings,
    )

    assert names == [
        "categorical_level_reconciliation_guard_v1",
        "strict_numeric_nonfinite_guard_v1",
    ]
    assert audit_mechanical_code_contracts(repaired, _STEP) == []


@pytest.mark.parametrize(
    "repair_id",
    [
        "strict_numeric_nonfinite_guard_v1",
        "categorical_level_reconciliation_guard_v1",
    ],
)
def test_descriptive_integrity_repairs_are_structural_and_automatic(
    repair_id: str,
) -> None:
    metadata = repair_metadata_for(repair_id)

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert automatic_repair_allowed(repair_id) is True
