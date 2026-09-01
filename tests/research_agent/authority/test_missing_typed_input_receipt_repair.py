from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from easyicu.research_agent.repairs.source import deterministic_contract_repair


def _findings(input_key: str = "artifact:analysis_cohort") -> list[dict[str, object]]:
    return [
        {
            "validator": "step_summary_integrity",
            "severity": "error",
            "detail": {
                "issue": "input_bindings_missing",
                "resolved_input_keys": [input_key],
            },
        },
        {
            "validator": "step_summary_integrity",
            "severity": "error",
            "detail": {
                "issue": "input_binding_coverage_incomplete",
                "missing_input_keys": [input_key],
                "resolved_input_keys": [input_key],
            },
        },
    ]


def _code(tmp_path: Path) -> str:
    input_path = tmp_path / "analysis_cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2], "age": [60, 70]}).to_parquet(input_path)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    return f"""
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

run_dir = Path({str(tmp_path)!r})
out_dir = run_dir
resolved_inputs_path = Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"])
with open(resolved_inputs_path, "r", encoding="utf-8") as resolved_handle:
    input_document = json.load(resolved_handle)
input_document = {{
    "manifest": {{
        "inputs": {{
            "artifact:analysis_cohort": {{
                "evidence_id": "cohort-evidence",
                "relative_path": "analysis_cohort.parquet",
                "sha256": {digest!r},
            }}
        }}
    }}
}}

def resolve_bound_path(root, relative_path):
    return root / relative_path

def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = input_document.get("manifest", input_document)
typed_inputs = manifest.get("inputs", {{}})
bound_input = typed_inputs["artifact:analysis_cohort"]
input_evidence_id = bound_input.get("evidence_id")
input_relative_path = bound_input.get("relative_path")
input_sha256 = bound_input.get("sha256")
input_path = resolve_bound_path(run_dir, input_relative_path)
if sha256_file(input_path) != input_sha256:
    raise ValueError("digest mismatch")
df_source = pd.read_parquet(input_path)
step_summary = {{
    "status": "completed",
    "output_files": {{"artifact:x": "x.parquet"}},
}}
with open(out_dir / "step_summary.json", "w", encoding="utf-8") as handle:
    json.dump(step_summary, handle)
"""


def test_missing_typed_input_receipt_fails_closed_without_host_sdk_execution(
    tmp_path: Path,
) -> None:
    code = _code(tmp_path)

    repair = deterministic_contract_repair(code=code, findings=_findings())

    assert repair is None


def test_missing_typed_input_receipt_requires_digest_guard(tmp_path: Path) -> None:
    code = _code(tmp_path).replace(
        'if sha256_file(input_path) != input_sha256:\n    raise ValueError("digest mismatch")\n',
        "",
    )

    assert deterministic_contract_repair(code=code, findings=_findings()) is None


def test_missing_typed_input_receipt_declines_multiple_host_keys(
    tmp_path: Path,
) -> None:
    findings = _findings()
    for finding in findings:
        detail = finding["detail"]
        assert isinstance(detail, dict)
        detail["resolved_input_keys"] = [
            "artifact:analysis_cohort",
            "table:another_input",
        ]
    coverage_detail = findings[1]["detail"]
    assert isinstance(coverage_detail, dict)
    coverage_detail["missing_input_keys"] = [
        "artifact:analysis_cohort",
        "table:another_input",
    ]

    code = _code(tmp_path)
    assert deterministic_contract_repair(code=code, findings=findings) is None


def test_missing_typed_input_receipt_declines_untrusted_inputs_mapping(
    tmp_path: Path,
) -> None:
    code = _code(tmp_path).replace(
        'typed_inputs = manifest.get("inputs", {})',
        'config = manifest\ntyped_inputs = config.get("inputs", {})',
    )

    assert deterministic_contract_repair(code=code, findings=_findings()) is None
