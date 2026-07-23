"""Consumer input scope must not be confused with physical Parquet schema."""

from __future__ import annotations

import ast

import pandas as pd

from easyicu.research_agent.repair_registry import (
    RepairClass,
    automatic_repair_allowed,
    repair_metadata_for,
)
from easyicu.research_agent.repairs.source import _deterministic_runner_repair

_ERROR = """
RuntimeError: locked cohort columns do not match the exact declared input scope
"""

_SCRIPT = """
import pandas as pd

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

expected_planner_inputs = ["stay_id", "exposure", "outcome"]
df = FRAME
require(
    len(df.columns) == len(expected_planner_inputs)
    and len(set(df.columns)) == len(df.columns)
    and set(df.columns) == set(expected_planner_inputs),
    "locked cohort columns do not match the exact declared input scope",
)
"""


def _run(script: str, frame: pd.DataFrame) -> None:
    exec(script, {"FRAME": frame})  # noqa: S102 - generated-code regression


def test_runner_repair_accepts_physical_superset_without_dropping_columns() -> None:
    repair = _deterministic_runner_repair(code=_SCRIPT, run_log=_ERROR)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "raw_input_physical_superset_guard_v1"
    assert "set(df.columns) == set(expected_planner_inputs)" not in repaired
    assert "all(_easyicu_input in (df).columns" in repaired
    ast.parse(repaired)

    frame = pd.DataFrame(
        {
            "stay_id": [1],
            "exposure": [0],
            "outcome": [1],
            "unrelated_locked_column": [42],
        }
    )
    original_columns = list(frame.columns)
    _run(repaired, frame)
    assert list(frame.columns) == original_columns


def test_runner_repair_remains_fail_closed_for_missing_declared_column() -> None:
    repair = _deterministic_runner_repair(code=_SCRIPT, run_log=_ERROR)

    assert repair is not None
    _, repaired = repair
    frame = pd.DataFrame({"stay_id": [1], "exposure": [0]})
    try:
        _run(repaired, frame)
    except RuntimeError as exc:
        assert "locked cohort columns" in str(exc)
    else:
        raise AssertionError("missing declared input must still fail closed")


def test_runner_repair_requires_matching_runtime_and_authored_diagnostics() -> None:
    assert (
        _deterministic_runner_repair(
            code=_SCRIPT,
            run_log="RuntimeError: some unrelated failure",
        )
        is None
    )
    forged = _SCRIPT.replace(
        "locked cohort columns do not match the exact declared input scope",
        "unrelated assertion",
    )
    assert _deterministic_runner_repair(code=forged, run_log=_ERROR) is None


def test_raw_input_superset_repair_is_structural_and_automatic() -> None:
    metadata = repair_metadata_for("raw_input_physical_superset_guard_v1")

    assert metadata.repair_class is RepairClass.STRUCTURAL
    assert metadata.introduces_numbers is False
    assert metadata.requires_disclosure is False
    assert automatic_repair_allowed(metadata.repair_id)
