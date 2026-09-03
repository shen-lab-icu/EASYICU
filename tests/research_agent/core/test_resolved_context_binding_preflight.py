from __future__ import annotations

import hashlib
import json

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


def _step(ra):
    return ra.AnalysisStep(
        step_id="data_quality",
        intent="Audit exposure data quality.",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=["table:data_quality"],
        method="measurement_quality_control",
    )


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "resolved_context_payload_not_loaded"
    ]


def test_inline_resolved_context_payload_is_repaired_to_digest_bound_load(ra):
    script = """
manifest = load_manifest()
declared = manifest["planner_declared_inputs"]
bindings = manifest["inputs"]
context_variables = manifest["context"]["variables"]
"""
    findings = _findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "resolved_context_payload_not_loaded",
        "line": 5,
        "manifest_name": "manifest",
        "target_name": "context_variables",
    }
    assert (
        repair_reason_for_finding(findings[0])
        is RepairReason.TYPED_CONTEXT_BINDING_INVALID
    )

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["resolved_context_digest_load_v1"]
    assert '_easyicu_context_binding = manifest["context"]' in repaired
    assert "_easyicu_context_digest.hexdigest()" in repaired
    assert '_easyicu_context_binding["sha256"]' in repaired
    assert 'context_variables = _easyicu_context_payload.get("variables")' in repaired
    assert _findings(repaired, ra) == []


def test_unproven_or_already_loaded_context_is_not_claimed(ra):
    unrelated = """
manifest = load_other_json()
context_variables = manifest["context"]["variables"]
"""
    loaded = """
manifest = load_manifest()
declared = manifest["planner_declared_inputs"]
bindings = manifest["inputs"]
context_binding = manifest["context"]
context_payload = load_and_verify(context_binding)
context_variables = context_payload["variables"]
"""

    assert _findings(unrelated, ra) == []
    assert _findings(loaded, ra) == []


def test_manifest_identity_proof_does_not_cross_function_scopes(ra) -> None:
    script = """
def resolved_input_reader():
    manifest = load_manifest()
    declared = manifest["planner_declared_inputs"]
    bindings = manifest["inputs"]
    return declared, bindings

def unrelated_reader():
    manifest = load_other_json()
    return manifest["context"]["variables"]
"""

    assert _findings(script, ra) == []


def test_resolved_context_digest_load_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("resolved_context_digest_load_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)


def _minimal_repaired_script(context_sha256: str, ra) -> str:
    script = f"""\
manifest = {{
    "planner_declared_inputs": [],
    "inputs": {{}},
    "context": {{
        "relative_path": "research_context.json",
        "sha256": "{context_sha256}",
    }},
}}
declared = manifest["planner_declared_inputs"]
bindings = manifest["inputs"]
context_variables = manifest["context"]["variables"]
"""
    findings = _findings(script, ra)
    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )
    assert names == ["resolved_context_digest_load_v1"]
    return repaired


def test_resolved_context_digest_load_executes_verified_payload(
    tmp_path, monkeypatch, ra
) -> None:
    payload = {"variables": [{"name": "aki_stage_max"}]}
    payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    (tmp_path / "research_context.json").write_bytes(payload_bytes)
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    namespace: dict[str, object] = {}

    exec(
        _minimal_repaired_script(hashlib.sha256(payload_bytes).hexdigest(), ra),
        namespace,
    )

    assert namespace["context_variables"] == payload["variables"]


def test_resolved_context_digest_load_rejects_tampered_payload(
    tmp_path, monkeypatch, ra
) -> None:
    original_bytes = json.dumps({"variables": []}, sort_keys=True).encode("utf-8")
    (tmp_path / "research_context.json").write_text(
        json.dumps({"variables": [{"name": "tampered"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="ResearchContext digest mismatch"):
        exec(
            _minimal_repaired_script(
                hashlib.sha256(original_bytes).hexdigest(),
                ra,
            ),
            {},
        )
