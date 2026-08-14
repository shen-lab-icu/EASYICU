from __future__ import annotations

import inspect

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.execution.phase import run_execute_phase
from easyicu.research_agent.execution.replan_review import (
    runtime_replan_review_pause,
)
from easyicu.research_agent.execution.run_coordination import (
    RunCoordinator,
    RunExecutionState,
    RunTransition,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _plan(*, revision: int, intent: str) -> AnalysisPlan:
    return AnalysisPlan(
        revision=revision,
        research_question="What is the adjusted ICU association?",
        steps=[
            AnalysisStep(
                step_id="01_primary",
                planned_analysis_role="primary",
                intent=intent,
                method="adjusted_association",
                expected_outputs=["table:primary_estimate"],
            )
        ],
    )


def test_substantive_runtime_replan_pause_binds_both_exact_plan_digests() -> None:
    current = _plan(revision=1, intent="Fit the approved primary model.")
    candidate = _plan(revision=2, intent="Fit a scientifically revised model.")

    pause = runtime_replan_review_pause(
        require_human_plan_review=True,
        current_plan=current,
        candidate_plan=candidate,
        trigger="probe_summary",
    )

    assert pause is not None
    assert pause.current_plan_sha256 == canonical_sha256(
        current.model_dump(mode="json")
    )
    assert pause.candidate_plan_sha256 == canonical_sha256(
        candidate.model_dump(mode="json")
    )
    finding = pause.finding()
    assert finding.severity == "error"
    assert finding.detail["human_review_required"] is True
    assert finding.detail["execution_paused"] is True
    assert finding.detail["review_authority_sha256"] == pause.review_authority_sha256


def test_non_review_gated_runtime_replan_keeps_existing_behavior() -> None:
    assert (
        runtime_replan_review_pause(
            require_human_plan_review=False,
            current_plan=_plan(revision=1, intent="Fit model A."),
            candidate_plan=_plan(revision=2, intent="Fit model B."),
            trigger="01_primary",
        )
        is None
    )


def test_pause_transition_never_applies_or_executes_the_revised_plan() -> None:
    first = AnalysisStep(step_id="01", intent="First step.")
    later = AnalysisStep(step_id="02", intent="Later step.")
    applied: list[object] = []
    executed: list[str] = []
    state = RunExecutionState(
        remaining_steps=[first, later],
        executed_step_ids=set(),
    )

    RunCoordinator().run_sequential(
        state=state,
        execute_step=lambda step: executed.append(step.step_id) or {"status": "ok"},
        resolve_transition=lambda *_: RunTransition.pause(
            "runtime_replan_human_review_required"
        ),
        apply_revised_plan=lambda plan, _executed: applied.append(plan) or [later],
    )

    assert executed == ["01"]
    assert applied == []
    assert state.stop_reason == "runtime_replan_human_review_required"


def test_execute_phase_gates_before_registering_or_applying_candidate_plan() -> None:
    source = inspect.getsource(run_execute_phase)
    review_gate = source.index("review_pause = runtime_replan_review_pause")
    cohort_application = source.index("_resolve_cohort_definition(revised, reason=reason)")
    registration = source.index(
        "plan_path = _register_plan_revision(revised, reason=reason)"
    )

    assert review_gate < cohort_application < registration
    assert '"runtime_replan_human_review_required"' in source
