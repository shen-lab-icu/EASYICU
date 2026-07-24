from __future__ import annotations

import ast

from easyicu.research_agent.repairs.statistic_sidecar import (
    patch_typed_statistic_sidecar_names,
)


def test_typed_statistic_sidecar_names_are_added_without_changing_values() -> None:
    code = """
from pathlib import Path

def write_json(path, payload):
    pass

OUT_DIR = Path(".")
grouped = {"primary_or": 1.25, "converged": True}
write_json(OUT_DIR / "robustness.json", grouped)
write_json(OUT_DIR / "complete_case_n.json", {"complete_case_n": 42})
"""
    repaired = patch_typed_statistic_sidecar_names(
        code,
        step_summary={
            "output_files": {
                "statistic:robustness_summary": "robustness.json",
                "statistic:complete_case_n": "complete_case_n.json",
            }
        },
    )

    assert repaired is not None
    ast.parse(repaired)
    assert "'name': 'robustness_summary'" in repaired
    assert "'name': 'complete_case_n'" in repaired
    assert "'primary_or': 1.25" in repaired
    assert "'complete_case_n': 42" in repaired


def test_typed_statistic_sidecar_repair_refuses_ambiguous_writer() -> None:
    code = """
write_json(OUT_DIR / "metric.json", {"value": 1.0})
write_json(OUT_DIR / "metric.json", {"value": 2.0})
"""

    assert (
        patch_typed_statistic_sidecar_names(
            code,
            step_summary={"output_files": {"statistic:metric": "metric.json"}},
        )
        is None
    )


def test_typed_statistic_sidecar_repair_preserves_explicit_identity() -> None:
    code = """
write_json(
    OUT_DIR / "metric.json",
    {"name": "metric", "value": 1.0},
)
"""

    assert (
        patch_typed_statistic_sidecar_names(
            code,
            step_summary={"output_files": {"statistic:metric": "metric.json"}},
        )
        is None
    )
