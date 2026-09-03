"""Typed manifest artifacts outrank the generic execution cohort env."""

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

_SCRIPT = """
import json
import os
from pathlib import Path

def load():
    with open(os.environ["EASYICU_RESOLVED_INPUTS_JSON"], encoding="utf-8") as handle:
        manifest = json.load(handle)
    declared = manifest.get("planner_declared_inputs")
    bindings = manifest.get("inputs", {})
    binding = bindings["artifact:analysis_cohort"]
    bound_path = Path(os.environ["EASYICU_RUN_DIR"]) / binding.get("relative_path")
    cohort_env_value = os.environ.get("COHORT_PARQUET")
    cohort_path = Path(cohort_env_value)
    if cohort_path.resolve() != bound_path.resolve():
        raise RuntimeError("generic cohort differs from typed artifact")
    return declared, cohort_path
"""


def _step(ra):
    return ra.AnalysisStep(
        step_id="qc",
        intent="Audit a declared upstream cohort artifact.",
        inputs=["artifact:analysis_cohort", "lact_max"],
        expected_outputs=["table:distribution"],
        method="data_quality_audit",
    )


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason")
        == "resolved_typed_input_shadowed_by_cohort_env"
    ]


def test_manifest_bound_artifact_replaces_generic_cohort_path_before_execution(
    ra,
) -> None:
    findings = _findings(_SCRIPT, ra)

    assert len(findings) == 1
    assert (
        repair_reason_for_finding(findings[0])
        is RepairReason.TYPED_PRODUCT_BINDING_INVALID
    )
    repaired, names = deterministic_concept_audit_repair(
        _SCRIPT,
        [findings[0].message],
        repair_reasons=[RepairReason.TYPED_PRODUCT_BINDING_INVALID],
        repair_findings=findings,
    )

    assert names == ["resolved_typed_input_precedence_v1"]
    assert "cohort_path = bound_path" in repaired
    assert "cohort_path = Path(cohort_env_value)" not in repaired
    assert _findings(repaired, ra) == []


def test_unproven_or_nonconflicting_cohort_reads_are_not_rewritten(ra) -> None:
    no_manifest_authority = _SCRIPT.replace(
        'declared = manifest.get("planner_declared_inputs")',
        "declared = []",
    )
    no_conflict_guard = _SCRIPT.replace(
        """    if cohort_path.resolve() != bound_path.resolve():
        raise RuntimeError("generic cohort differs from typed artifact")
""",
        "",
    )

    assert _findings(no_manifest_authority, ra) == []
    assert _findings(no_conflict_guard, ra) == []


def test_typed_input_precedence_repair_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("resolved_typed_input_precedence_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)
