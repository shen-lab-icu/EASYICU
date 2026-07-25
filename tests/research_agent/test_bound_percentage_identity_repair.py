"""Regression tests for bound percentage/count identity guards."""

from __future__ import annotations

import pytest

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.percentage_identity import (
    patch_bound_percentage_identity_guards,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.schema import ValidationFinding

_FINDING = ValidationFinding(
    validator="llm_concept_auditor",
    severity="error",
    message="Displayed percentages are not reconciled to counts.",
    detail={
        "issue_code": "other",
        "variables": ["prevalence_pct", "missingness_pct"],
    },
)

_SCRIPT = """
import numpy as np

def finite_numeric(frame, column, input_key, diagnostics):
    return np.asarray(frame[column], dtype=float)

def main(prevalence, missingness):
    diagnostics = {}
    prevalence_counts = finite_numeric(
        prevalence, "count", "table:prevalence", diagnostics
    )
    prevalence_pct = finite_numeric(
        prevalence, "percentage", "table:prevalence", diagnostics
    )
    prevalence_den = finite_numeric(
        prevalence, "denominator", "table:prevalence", diagnostics
    )
    if True:
        missingness_pct = finite_numeric(
            missingness, "missing_pct", "table:missingness", diagnostics
        )
        missingness_n = finite_numeric(
            missingness, "missing_n", "table:missingness", diagnostics
        )
        missingness_total = finite_numeric(
            missingness, "n_total", "table:missingness", diagnostics
        )
    return prevalence_pct, missingness_pct
"""


def test_bound_percentage_repair_inserts_guards_without_replacing_values() -> None:
    repaired = patch_bound_percentage_identity_guards(
        _SCRIPT,
        findings=[_FINDING],
    )

    assert repaired != _SCRIPT
    assert "_easyicu_expected_percentage_0" in repaired
    assert "np.allclose" in repaired
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    good = (
        {"count": [25], "percentage": [25.0], "denominator": [100]},
        {"missing_pct": [10.0], "missing_n": [10], "n_total": [100]},
    )
    observed = namespace["main"](*good)
    assert observed[0].tolist() == [25.0]
    assert observed[1].tolist() == [10.0]

    bad = (
        {"count": [25], "percentage": [99.0], "denominator": [100]},
        good[1],
    )
    with pytest.raises(RuntimeError, match="percentage disagrees"):
        namespace["main"](*bad)


def test_bound_percentage_repair_fails_closed_on_unbound_or_ambiguous_shape() -> None:
    unknown = _FINDING.model_copy(
        update={"detail": {**_FINDING.detail, "variables": ["unknown_pct"]}}
    )
    assert (
        patch_bound_percentage_identity_guards(_SCRIPT, findings=[unknown]) == _SCRIPT
    )
    ambiguous = _SCRIPT.replace(
        'prevalence, "count", "table:prevalence", diagnostics\n    )',
        'prevalence, "count", "table:prevalence", diagnostics\n    )\n'
        "    prevalence_n = finite_numeric(\n"
        '        prevalence, "n", "table:prevalence", diagnostics\n'
        "    )",
    )
    assert (
        patch_bound_percentage_identity_guards(
            ambiguous,
            findings=[_FINDING],
        )
        == ambiguous
    )


def test_percentage_identity_repair_routes_through_concept_gate_and_registry() -> None:
    repaired, names = deterministic_concept_audit_repair(
        _SCRIPT,
        [_FINDING.message],
        repair_findings=[_FINDING],
    )

    assert names == ["bound_percentage_identity_guard_v1"]
    assert repaired != _SCRIPT
    metadata = repair_metadata_for("bound_percentage_identity_guard_v1")
    assert metadata.classification_source == "exact"
    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert automatic_repair_allowed(metadata.repair_id)
