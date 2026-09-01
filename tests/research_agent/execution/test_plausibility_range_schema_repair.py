"""Typed plausibility ranges keep one executable JSON schema."""

from __future__ import annotations

import pytest

from easyicu.research_agent.execution.phase import _untrusted_runtime_repair_allowed
from easyicu.research_agent.repair_registry import RepairClass, repair_metadata_for
from easyicu.research_agent.repairs.source import _deterministic_runner_repair


@pytest.mark.parametrize(
    ("access", "run_log"),
    [
        (
            'float(plausibility[0]), float(plausibility[1])',
            "KeyError: 0",
        ),
        (
            'float(plausibility["lower"]), float(plausibility["upper"])',
            "KeyError: 'lower'",
        ),
    ],
)
def test_runner_repair_uses_sealed_plausibility_range_keys(
    access: str,
    run_log: str,
) -> None:
    code = f"""
def read_bounds(contract):
    plausibility = contract.get("analysis_plausibility_range")
    lower, upper = {access}
    return lower, upper
"""

    repair = _deterministic_runner_repair(code=code, run_log=run_log)

    assert repair is not None
    repair_id, repaired = repair
    assert repair_id == "plausibility_range_schema_keys_v1"
    assert "plausibility['minimum']" in repaired
    assert "plausibility['maximum']" in repaired
    namespace: dict[str, object] = {}
    exec(repaired, namespace)  # noqa: S102 - generated-code regression
    assert namespace["read_bounds"](
        {"analysis_plausibility_range": {"minimum": 0.0, "maximum": 4.0}}
    ) == (0.0, 4.0)
    metadata = repair_metadata_for(repair_id)
    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert _untrusted_runtime_repair_allowed(
        repair_id=repair_id,
        source="deterministic_runner_repair",
    )


def test_plausibility_range_schema_repair_is_traceback_and_field_bound() -> None:
    code = """
plausibility = contract.get("analysis_plausibility_range")
lower = plausibility[0]
"""

    assert _deterministic_runner_repair(code=code, run_log="") is None
    assert (
        _deterministic_runner_repair(
            code=code.replace("analysis_plausibility_range", "confidence_interval"),
            run_log="KeyError: 0",
        )
        is None
    )
    assert (
        _deterministic_runner_repair(
            code=code,
            run_log="KeyError: 'unrelated'",
        )
        is None
    )
