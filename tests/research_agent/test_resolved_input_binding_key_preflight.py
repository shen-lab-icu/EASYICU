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
from easyicu.research_agent.repairs.source import (
    _deterministic_runner_repair,
    deterministic_concept_audit_repair,
)


def _step(ra):
    return ra.AnalysisStep(
        step_id="render",
        intent="Render an upstream typed table.",
        inputs=["table:upstream"],
        expected_outputs=["figure:panel"],
        method="visualization",
    )


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason") == "resolved_input_key_not_materialized"
    ]


_SCRIPT = """
def read_bound_table(binding):
    path = binding["relative_path"]
    expected_digest = binding.get("sha256")
    product_contract = binding.get("product_contract") or {}
    message = f"Loading {binding['input_key']}"
    return binding["input_key"], path, expected_digest, product_contract, message

def main(manifest):
    declared = manifest.get("planner_declared_inputs")
    bound_inputs = manifest.get("inputs", {})
    loaded = []
    for key in declared:
        binding = bound_inputs[key]
        loaded.append(read_bound_table(binding))
    return loaded
"""


def test_resolved_binding_key_is_repaired_to_identity_row(ra) -> None:
    findings = _findings(_SCRIPT, ra)

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "resolved_input_key_not_materialized",
        "helper_name": "read_bound_table",
        "binding_parameter": "binding",
        "access_lines": [6, 7],
    }
    assert (
        repair_reason_for_finding(findings[0])
        is RepairReason.TYPED_PRODUCT_BINDING_INVALID
    )

    repaired, names = deterministic_concept_audit_repair(
        _SCRIPT,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["resolved_input_identity_key_v1"]
    assert "binding['identity_row']['input_key']" in repaired
    assert _findings(repaired, ra) == []

    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    main = namespace["main"]
    result = main(
        {
            "planner_declared_inputs": ["table:upstream"],
            "inputs": {
                "table:upstream": {
                    "relative_path": "evidence/upstream.csv",
                    "sha256": "abc",
                    "product_contract": {"columns": ["x"]},
                    "identity_row": {"input_key": "table:upstream"},
                }
            },
        }
    )
    assert result[0][0] == "table:upstream"


def test_get_input_key_access_is_reported_without_preflight_crash(ra) -> None:
    script = _SCRIPT.replace(
        "binding['input_key']",
        'binding.get("input_key", "")',
    ).replace(
        'binding["input_key"]',
        'binding.get("input_key", "")',
    ).replace(
        'message = f"Loading {binding.get("input_key", "")}"',
        """message = f'Loading {binding.get("input_key", "")}'""",
    )

    findings = _findings(script, ra)

    assert len(findings) == 1
    assert findings[0].detail["access_lines"] == [6, 7]


def test_unproven_binding_dictionary_is_not_claimed(ra) -> None:
    unrelated = """
def read_config(binding):
    return (
        binding["input_key"],
        binding["relative_path"],
        binding.get("sha256"),
        binding.get("product_contract"),
    )

def main(config):
    return read_config(config)
"""

    assert _findings(unrelated, ra) == []


def test_helper_with_an_unproven_second_call_is_not_claimed(ra) -> None:
    ambiguous = _SCRIPT + "\nresult = read_bound_table(external_binding)\n"

    assert _findings(ambiguous, ra) == []


def test_resolved_input_identity_key_repair_is_syntactic_and_automatic() -> None:
    metadata = repair_metadata_for("resolved_input_identity_key_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)


_DIRECT_SCRIPT = """
manifest = {
    "planner_declared_inputs": ["table:upstream"],
    "inputs": {
        "table:upstream": {
            "relative_path": "evidence/upstream.csv",
            "sha256": "abc",
            "product_contract": {"columns": ["x"]},
            "identity_row": {"input_key": "table:upstream"},
        }
    },
}
typed_bindings = manifest.get("inputs", {})
for expected_key in manifest.get("planner_declared_inputs", []):
    binding = typed_bindings[expected_key]
    binding_input_key = binding.get("input_key")
"""


def test_direct_resolved_binding_key_is_found_and_repaired(ra) -> None:
    findings = _findings(_DIRECT_SCRIPT, ra)

    assert len(findings) == 1
    assert findings[0].detail["binding_name"] == "binding"
    repaired, names = deterministic_concept_audit_repair(
        _DIRECT_SCRIPT,
        [findings[0].message],
        repair_reasons=[repair_reason_for_finding(findings[0])],
        repair_findings=findings,
    )

    assert names == ["resolved_input_identity_key_v1"]
    assert 'binding["identity_row"]["input_key"]' in repaired
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    assert namespace["binding_input_key"] == "table:upstream"


def test_runtime_failure_reuses_direct_identity_key_repair() -> None:
    repair = _deterministic_runner_repair(
        code=_DIRECT_SCRIPT,
        run_log=(
            "ValueError: Typed input binding lacks exact input_key: table:upstream"
        ),
    )

    assert repair is not None
    assert repair[0] == "resolved_input_identity_key_v1"
    assert 'binding["identity_row"]["input_key"]' in repair[1]


def test_direct_unproven_binding_mapping_is_not_claimed(ra) -> None:
    unrelated = """
config = {"input_key": "external"}
binding_input_key = config.get("input_key")
"""

    assert _findings(unrelated, ra) == []


def test_local_binding_summary_shadow_does_not_inherit_typed_binding_origin(ra) -> None:
    script = """
manifest = {
    "planner_declared_inputs": ["table:upstream"],
    "inputs": {
        "table:upstream": {
            "relative_path": "evidence/upstream.csv",
            "sha256": "abc",
            "product_contract": {"columns": ["x"]},
            "identity_row": {"input_key": "table:upstream"},
        }
    },
}
typed_bindings = manifest.get("inputs", {})
summaries = []
for expected_key in manifest.get("planner_declared_inputs", []):
    binding = typed_bindings[expected_key]
    summaries.append({"input_key": expected_key})
for binding in summaries:
    observed_key = binding["input_key"]
"""

    assert _findings(script, ra) == []

    namespace: dict[str, object] = {}
    exec(script, namespace)
    assert namespace["observed_key"] == "table:upstream"


def test_only_accesses_before_local_binding_shadow_are_repaired(ra) -> None:
    script = _DIRECT_SCRIPT + """
summaries = [{"input_key": "local:summary"}]
for binding in summaries:
    local_key = binding["input_key"]
"""

    findings = _findings(script, ra)

    assert len(findings) == 1
    assert len(findings[0].detail["access_occurrences"]) == 1
    repaired, names = deterministic_concept_audit_repair(
        script,
        [findings[0].message],
        repair_reasons=[repair_reason_for_finding(findings[0])],
        repair_findings=findings,
    )

    assert names == ["resolved_input_identity_key_v1"]
    assert 'binding["identity_row"]["input_key"]' in repaired
    assert 'local_key = binding["input_key"]' in repaired
    namespace: dict[str, object] = {}
    exec(repaired, namespace)
    assert namespace["binding_input_key"] == "table:upstream"
    assert namespace["local_key"] == "local:summary"


def test_loop_else_keeps_fail_closed_typed_binding_provenance(ra) -> None:
    script = _DIRECT_SCRIPT + """
for binding in []:
    pass
else:
    fallback_key = binding["input_key"]
"""

    findings = _findings(script, ra)

    assert len(findings) == 1
    assert len(findings[0].detail["access_occurrences"]) == 2
