"""Resolved typed-input paths are rooted once at the run directory."""

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
        step_id="qc",
        intent="Read a host-bound analysis cohort.",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:distribution"],
        method="data_quality_audit",
    )


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason")
        == "resolved_input_relative_path_wrong_root"
    ]


_SCRIPT = """
import json
import os
from pathlib import Path

def main():
    with open(os.environ["EASYICU_RESOLVED_INPUTS_JSON"], encoding="utf-8") as fh:
        manifest = json.load(fh)
    declared = manifest.get("planner_declared_inputs")
    binding = manifest["inputs"]["artifact:analysis_cohort"]
    cohort_path = Path(os.environ["EASYICU_EVIDENCE_DIR"]) / binding["relative_path"]
    return declared, cohort_path
"""


def test_run_relative_binding_is_repaired_before_execution(ra) -> None:
    findings = _findings(_SCRIPT, ra)

    assert len(findings) == 1
    assert findings[0].detail["reason"] == ("resolved_input_relative_path_wrong_root")
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

    assert names == ["resolved_input_run_root_v1"]
    assert 'Path(os.environ["EASYICU_EVIDENCE_DIR"])' not in repaired
    assert "Path(os.environ['EASYICU_RUN_DIR'])" in repaired
    assert _findings(repaired, ra) == []


def test_inputs_mapping_and_get_relative_path_are_supported(ra) -> None:
    script = (
        _SCRIPT.replace(
            'binding = manifest["inputs"]["artifact:analysis_cohort"]',
            'inputs = manifest.get("inputs", {})\n    binding = inputs["artifact:analysis_cohort"]',
        )
        .replace(
            'os.environ["EASYICU_EVIDENCE_DIR"]',
            'os.environ.get("EASYICU_EVIDENCE_DIR")',
        )
        .replace(
            'binding["relative_path"]',
            'binding.get("relative_path")',
        )
    )

    findings = _findings(script, ra)

    assert len(findings) == 1
    repaired, names = deterministic_concept_audit_repair(
        script,
        [findings[0].message],
        repair_reasons=[RepairReason.TYPED_PRODUCT_BINDING_INVALID],
        repair_findings=findings,
    )
    assert names == ["resolved_input_run_root_v1"]
    assert "os.environ.get('EASYICU_RUN_DIR')" in repaired


def test_all_proven_wrong_roots_are_repaired_atomically(ra) -> None:
    script = _SCRIPT.replace(
        "    return declared, cohort_path",
        """    second = Path(os.environ["EASYICU_EVIDENCE_DIR"]) / binding.get("relative_path")
    return declared, cohort_path, second""",
    )
    findings = _findings(script, ra)

    assert len(findings) == 1
    assert len(findings[0].detail["occurrences"]) == 2
    repaired, names = deterministic_concept_audit_repair(
        script,
        [findings[0].message],
        repair_reasons=[RepairReason.TYPED_PRODUCT_BINDING_INVALID],
        repair_findings=findings,
    )
    assert names == ["resolved_input_run_root_v1"]
    assert "EASYICU_EVIDENCE_DIR" not in repaired
    assert repaired.count("EASYICU_RUN_DIR") == 2


def test_correct_run_root_and_unproven_dictionary_are_not_claimed(ra) -> None:
    correct = _SCRIPT.replace("EASYICU_EVIDENCE_DIR", "EASYICU_RUN_DIR")
    unrelated = """
import os
from pathlib import Path

def main(config):
    return Path(os.environ["EASYICU_EVIDENCE_DIR"]) / config["relative_path"]
"""

    assert _findings(correct, ra) == []
    assert _findings(unrelated, ra) == []


def test_run_root_repair_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("resolved_input_run_root_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)
