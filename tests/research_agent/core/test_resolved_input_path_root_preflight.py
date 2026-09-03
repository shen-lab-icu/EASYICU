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


_HELPER_SCRIPT = """
import json
import os
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def resolve_bound_product(input_key, entry, evidence_dir):
    relative_path = entry.get("relative_path")
    path = Path(evidence_dir) / relative_path
    return input_key, path

def main():
    resolved_path = Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"])
    resolved_document = load_json(resolved_path)
    manifest = resolved_document.get("manifest", resolved_document)
    bound_inputs = manifest.get("inputs")
    cohort_entry = bound_inputs["artifact:analysis_cohort"]
    cluster_entry = bound_inputs["dataset:cluster_features"]
    evidence_dir = Path(os.environ["EASYICU_EVIDENCE_DIR"])
    cohort = resolve_bound_product(
        "artifact:analysis_cohort", cohort_entry, evidence_dir
    )
    features = resolve_bound_product(
        "dataset:cluster_features", cluster_entry, evidence_dir
    )
    return cohort, features
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


def test_path_open_json_load_proves_manifest_without_shape_probe(ra) -> None:
    script = """
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"])
with manifest_path.open("r", encoding="utf-8") as manifest_file:
    resolved_manifest = json.load(manifest_file)
binding = resolved_manifest["inputs"]["artifact:analysis_cohort"]
relative_path = binding["relative_path"]
cohort_path = Path(os.environ["EASYICU_EVIDENCE_DIR"]) / relative_path
"""

    findings = _findings(script, ra)

    assert len(findings) == 1
    repaired, names = deterministic_concept_audit_repair(
        script,
        [findings[0].message],
        repair_reasons=[RepairReason.TYPED_PRODUCT_BINDING_INVALID],
        repair_findings=findings,
    )
    assert names == ["resolved_input_run_root_v1"]
    assert "Path(os.environ['EASYICU_RUN_DIR'])" in repaired


def test_direct_host_manifest_read_is_proven_without_key_shape_inference(ra) -> None:
    script = """
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"])
resolved = json.loads(manifest_path.read_text(encoding="utf-8"))
binding = resolved["inputs"]["artifact:analysis_cohort"]
cohort_path = Path(os.environ["EASYICU_EVIDENCE_DIR"]) / binding["relative_path"]
"""

    findings = _findings(script, ra)

    assert len(findings) == 1
    repaired, names = deterministic_concept_audit_repair(
        script,
        [findings[0].message],
        repair_reasons=[RepairReason.TYPED_PRODUCT_BINDING_INVALID],
        repair_findings=findings,
    )
    assert names == ["resolved_input_run_root_v1"]
    assert "EASYICU_EVIDENCE_DIR" not in repaired
    assert "EASYICU_RUN_DIR" in repaired


def test_arbitrary_json_file_is_not_promoted_to_host_manifest_authority(ra) -> None:
    script = """
import json
import os
from pathlib import Path

manifest_path = Path("untrusted.json")
resolved = json.loads(manifest_path.read_text(encoding="utf-8"))
binding = resolved["inputs"]["artifact:analysis_cohort"]
cohort_path = Path(os.environ["EASYICU_EVIDENCE_DIR"]) / binding["relative_path"]
"""

    assert _findings(script, ra) == []


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


def test_helper_parameter_root_is_proven_and_repaired_once(ra) -> None:
    findings = _findings(_HELPER_SCRIPT, ra)

    assert len(findings) == 1
    assert len(findings[0].detail["occurrences"]) == 1
    assert findings[0].detail["occurrences"][0]["kind"] == "root_parameter"

    repaired, names = deterministic_concept_audit_repair(
        _HELPER_SCRIPT,
        [findings[0].message],
        repair_reasons=[RepairReason.TYPED_PRODUCT_BINDING_INVALID],
        repair_findings=findings,
    )

    assert names == ["resolved_input_run_root_v1"]
    assert "Path(evidence_dir) / relative_path" not in repaired
    assert 'Path(os.environ["EASYICU_RUN_DIR"]) / relative_path' in repaired
    assert _findings(repaired, ra) == []


def test_helper_root_with_an_unrelated_use_is_not_claimed(ra) -> None:
    script = _HELPER_SCRIPT.replace(
        "    return input_key, path",
        "    return input_key, path, evidence_dir",
    )

    assert _findings(script, ra) == []


def test_helper_with_one_unproven_call_is_not_claimed(ra) -> None:
    script = _HELPER_SCRIPT.replace(
        "    return cohort, features",
        "    other = resolve_bound_product('other', {}, evidence_dir)\n"
        "    return cohort, features, other",
    )

    assert _findings(script, ra) == []


def test_custom_json_loader_does_not_promote_an_untrusted_path(ra) -> None:
    script = _HELPER_SCRIPT.replace(
        'Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"])',
        'Path("untrusted.json")',
    )

    assert _findings(script, ra) == []


def test_custom_json_loader_with_path_reassignment_is_not_claimed(ra) -> None:
    script = _HELPER_SCRIPT.replace(
        "def load_json(path):\n",
        'def load_json(path):\n    path = Path("untrusted.json")\n',
    )

    assert _findings(script, ra) == []


def test_custom_json_loader_with_caller_path_reassignment_is_not_claimed(ra) -> None:
    script = _HELPER_SCRIPT.replace(
        "    resolved_document = load_json(resolved_path)",
        '    resolved_path = Path("untrusted.json")\n'
        "    resolved_document = load_json(resolved_path)",
    )

    assert _findings(script, ra) == []


def test_helper_with_reassigned_root_alias_is_not_claimed(ra) -> None:
    script = _HELPER_SCRIPT.replace(
        "    cohort = resolve_bound_product(",
        '    evidence_dir = Path("untrusted")\n'
        "    cohort = resolve_bound_product(",
    )

    assert _findings(script, ra) == []


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
