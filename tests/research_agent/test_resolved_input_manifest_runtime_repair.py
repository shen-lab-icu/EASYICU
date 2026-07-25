from __future__ import annotations

from easyicu.research_agent.repair_registry import RepairClass, repair_metadata_for
from easyicu.research_agent.repairs.source import _deterministic_runner_repair

_CODE = """
import os
from pathlib import Path

run_dir = Path(os.environ["EASYICU_RUN_DIR"])
manifest_path = run_dir / "resolved_inputs.json"
if not manifest_path.exists():
    raise RuntimeError(f"Resolved input manifest not found: {manifest_path}")
"""


def test_runtime_repair_uses_host_issued_manifest_path():
    repaired = _deterministic_runner_repair(
        code=_CODE,
        run_log=(
            "RuntimeError: Resolved input manifest not found: "
            "/easyicu-run/resolved_inputs.json"
        ),
    )

    assert repaired is not None
    repair_id, patched = repaired
    assert repair_id == "resolved_input_manifest_env_v1"
    assert 'Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"])' in patched
    assert 'run_dir / "resolved_inputs.json"' not in patched


def test_runtime_repair_reads_host_issued_manifest_path_before_json_loads():
    code = """
import json
import os
from pathlib import Path

resolved_inputs_raw = os.environ["EASYICU_RESOLVED_INPUTS_JSON"]
manifest = json.loads(resolved_inputs_raw)
"""

    repaired = _deterministic_runner_repair(
        code=code,
        run_log=(
            "json.decoder.JSONDecodeError: Expecting value: line 1 column 1\n"
            "RuntimeError: EASYICU_RESOLVED_INPUTS_JSON must contain valid JSON"
        ),
    )

    assert repaired is not None
    repair_id, patched = repaired
    assert repair_id == "resolved_input_manifest_env_v1"
    assert (
        'json.loads(Path(resolved_inputs_raw).read_text(encoding="utf-8"))' in patched
    )


def test_runtime_repair_roots_manifest_relative_artifact_at_run_dir():
    code = """
import json
import os
from pathlib import Path

with open(os.environ["EASYICU_RESOLVED_INPUTS_JSON"], encoding="utf-8") as handle:
    manifest = json.load(handle)
binding = manifest["inputs"]["artifact:analysis_cohort"]
expected_path = binding.get("relative_path") or binding.get("path")
candidate_paths = [
    Path(expected_path),
    Path(os.environ.get("EASYICU_EVIDENCE_DIR", "")) / expected_path,
]
existing = next((path for path in candidate_paths if path.exists()), None)
if existing is None:
    raise RuntimeError(f"Bound typed input file is unavailable: {expected_path}")
"""

    repaired = _deterministic_runner_repair(
        code=code,
        run_log=(
            "RuntimeError: Bound typed input file is unavailable: "
            "evidence/table_step_artifact_123__analysis_cohort.parquet"
        ),
    )

    assert repaired is not None
    repair_id, patched = repaired
    assert repair_id == "resolved_input_manifest_env_v1"
    assert 'os.environ.get("EASYICU_EVIDENCE_DIR", "")' not in patched
    assert "os.environ.get('EASYICU_RUN_DIR', \"\")" in patched


def test_runtime_relative_path_repair_requires_manifest_and_unique_run_root_join():
    code = """
import os
from pathlib import Path

expected_path = binding.get("relative_path")
candidate = Path(os.environ["EASYICU_EVIDENCE_DIR"]) / expected_path
"""
    run_log = (
        "RuntimeError: Bound typed input file is unavailable: " "evidence/table.parquet"
    )
    assert _deterministic_runner_repair(code=code, run_log=run_log) is None

    ambiguous = (
        'manifest_path = os.environ["EASYICU_RESOLVED_INPUTS_JSON"]\n'
        + code
        + '\nother = Path(os.environ["EASYICU_EVIDENCE_DIR"]) / expected_path\n'
    )
    assert _deterministic_runner_repair(code=ambiguous, run_log=run_log) is None


def test_runtime_repair_refuses_ambiguous_resolved_manifest_string_uses():
    code = """
import json
import os
from pathlib import Path

resolved_inputs_raw = os.environ["EASYICU_RESOLVED_INPUTS_JSON"]
print(resolved_inputs_raw)
manifest = json.loads(resolved_inputs_raw)
"""

    assert (
        _deterministic_runner_repair(
            code=code,
            run_log=(
                "json.decoder.JSONDecodeError: Expecting value: line 1 column 1\n"
                "RuntimeError: EASYICU_RESOLVED_INPUTS_JSON must contain valid JSON"
            ),
        )
        is None
    )


def test_runtime_repair_requires_exact_failure_and_run_dir_authority():
    assert _deterministic_runner_repair(code=_CODE, run_log="unrelated failure") is None
    unbound = _CODE.replace('Path(os.environ["EASYICU_RUN_DIR"])', 'Path("/tmp/run")')
    assert (
        _deterministic_runner_repair(
            code=unbound,
            run_log="Resolved input manifest not found: /tmp/run/resolved_inputs.json",
        )
        is None
    )


def test_runtime_repair_refuses_ambiguous_manifest_derivations():
    ambiguous = _CODE + '\nother = run_dir / "resolved_inputs.json"\n'

    assert (
        _deterministic_runner_repair(
            code=ambiguous,
            run_log="Resolved input manifest not found: /easyicu-run/resolved_inputs.json",
        )
        is None
    )


def test_runtime_repair_is_registered_as_syntactic():
    metadata = repair_metadata_for("resolved_input_manifest_env_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.classification_source == "exact"
