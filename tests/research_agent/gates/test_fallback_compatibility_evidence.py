"""A fallback step proves method compatibility by running the gate-approved script."""

from __future__ import annotations

import pytest

from easyicu.research_agent.reporting.readiness import (
    _fallback_method_compatibility_errors,
)

APPROVED = "a" * 64


def _fallback_record(step_id: str = "robustness_replay", **digests: str) -> dict[str, object]:
    return {
        "step_id": step_id,
        "generation_mode": "fallback",
        "status": "ok",
        "step_summary": {"status": "ok"},
        **digests,
    }


@pytest.mark.parametrize("step_id", ["robustness_replay", "subgroup_interaction"])
def test_a_fallback_step_that_ran_the_gate_approved_script_passes(step_id: str) -> None:
    record = _fallback_record(
        step_id,
        executed_code_sha256=APPROVED,
        concept_approved_code_sha256=APPROVED,
    )

    assert _fallback_method_compatibility_errors(
        per_step_records=[record], context=None, plan=None
    ) == []


@pytest.mark.parametrize(
    "digests",
    [
        {},
        {"executed_code_sha256": APPROVED},
        {"concept_approved_code_sha256": APPROVED},
        {"executed_code_sha256": APPROVED, "concept_approved_code_sha256": "b" * 64},
        {"executed_code_sha256": "approved", "concept_approved_code_sha256": "approved"},
    ],
)
def test_a_fallback_step_without_that_proof_still_fails_closed(
    digests: dict[str, str],
) -> None:
    errors = _fallback_method_compatibility_errors(
        per_step_records=[_fallback_record(**digests)], context=None, plan=None
    )

    assert [error.validator for error in errors] == ["method_compatibility"]
    assert errors[0].detail == {
        "step_id": "robustness_replay",
        "generation_mode": "fallback",
    }


def test_a_non_fallback_step_is_not_asked_for_this_proof() -> None:
    record = _fallback_record()
    record["generation_mode"] = "deterministic_standard"

    assert _fallback_method_compatibility_errors(
        per_step_records=[record], context=None, plan=None
    ) == []
