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
