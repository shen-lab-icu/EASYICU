"""A failed-closed run says what stopped it, as codes.

A finished run fails closed when one of its gate axes is not satisfied: a step
did not finish, the automated validation did not pass, a result lacks its
evidence or a reported number could not be verified.  Its gate detail names
which: the stop its first failed step's executor named, with that stop's
lower-layer cause, or else the first axis the run did not satisfy.  A blocked
reason that replaces the gate's own carries no such detail, and the
conversation reads the detail through one projection, cause code included.

Synthetic run records; no benchmark item.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.pi_copilot import projections, workflow
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    complete_study,
)

_INTERVAL = "continuous_survival_interval_result_not_estimable"
_PASSED = {
    "execution_complete": True,
    "analysis_validated": True,
    "evidence_complete": True,
    "numeric_verified": True,
}


def _failed_step(**codes):
    return {"step_id": "04_suite", "status": "deterministic_standard_blocked", **codes}


def _stopped(cause: str = "interval_without_event"):
    return {
        **_PASSED,
        "execution_complete": False,
        "failed_steps": [_failed_step(reason_code=_INTERVAL, cause_code=cause)],
    }


@pytest.mark.parametrize(
    ("axes", "detail"),
    [
        (_stopped(), {"reason_code": _INTERVAL, "cause_code": "interval_without_event"}),
        # The first failed step decides; a stop a later step named is not the
        # run's cause.
        (
            {
                **_PASSED,
                "execution_complete": False,
                "failed_steps": [
                    _failed_step(),
                    _failed_step(reason_code=_INTERVAL, cause_code="did_not_converge"),
                ],
            },
            {"reason_code": "execution_complete_not_satisfied"},
        ),
        # A code outside the vocabulary names nothing.
        (
            {
                **_PASSED,
                "execution_complete": False,
                "failed_steps": [_failed_step(reason_code="made_up", cause_code="x")],
            },
            {"reason_code": "execution_complete_not_satisfied"},
        ),
        (
            {**_PASSED, "analysis_validated": False, "numeric_verified": False},
            {"reason_code": "analysis_validated_not_satisfied"},
        ),
        ({**_PASSED, "evidence_complete": False}, {"reason_code": "evidence_complete_not_satisfied"}),
        ({**_PASSED, "numeric_verified": False}, {"reason_code": "numeric_verified_not_satisfied"}),
        (_PASSED, None),
    ],
)
def test_a_failed_closed_run_names_its_stop_or_the_first_check_it_missed(axes, detail):
    assert agent_pipeline_runs._failed_closed_detail(axes) == detail


def _project(tmp_path: Path, gates, **kwargs):
    run_dir = tmp_path / "run_failed_closed"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(json.dumps({"gates": gates}), encoding="utf-8")
    return agent_pipeline_runs._write_projection(
        wrapper_dir=tmp_path / "wrapper",
        study=complete_study(),
        provider={"provider": "openai", "model": "test-model"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
        **kwargs,
    )


def test_the_run_projection_carries_the_stop_beside_its_reason(tmp_path):
    gate = _project(tmp_path, _stopped("non_finite_estimate"))["gate"]

    assert gate["reason"] == "research_agent_pipeline_failed_closed"
    assert gate["detail"] == {"reason_code": _INTERVAL, "cause_code": "non_finite_estimate"}
    assert projections.gate_detail_projection(gate["detail"]) == {
        "gate_detail_code": _INTERVAL,
        "gate_missing_concepts": [],
        "gate_detail_cause_code": "non_finite_estimate",
    }


def test_a_blocked_reason_that_replaces_the_gate_s_own_carries_no_such_detail(tmp_path):
    gate = _project(
        tmp_path, _stopped(), blocked_reason="research_pipeline_execution_failed"
    )["gate"]

    assert gate["reason"] == "research_pipeline_execution_failed"
    assert "detail" not in gate


def test_a_run_that_passed_its_checks_carries_no_detail(tmp_path):
    gate = _project(tmp_path, _PASSED)["gate"]

    assert gate["reason"] != "research_agent_pipeline_failed_closed"
    assert "detail" not in gate


def test_the_conversation_reads_one_detail_projection():
    assert workflow.gate_detail_projection is projections.gate_detail_projection
    assert projections.gate_detail_projection(
        {"reason_code": "data_foundation_blocked", "missing_concepts": ["age"]}
    ) == {"gate_detail_code": "data_foundation_blocked", "gate_missing_concepts": ["age"]}
