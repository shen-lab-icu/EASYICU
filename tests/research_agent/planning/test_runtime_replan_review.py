from __future__ import annotations

import inspect

import pytest

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.execution.phase import run_execute_phase
from easyicu.research_agent.execution import phase_support
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
    source = inspect.getsource(phase_support._step_maybe_replan)
    review_gate = source.index("replan_review.record_runtime_replan_review_pause")
    cohort_application = source.index(
        "resolve_cohort_definition(revised, reason=reason)"
    )
    registration = source.index(
        "plan_path = register_plan_revision(revised, reason=reason)"
    )

    assert review_gate < cohort_application < registration
    assert '"runtime_replan_human_review_required"' in inspect.getsource(
        run_execute_phase
    )


def test_runtime_replan_rebinds_signed_science_before_contract_review() -> None:
    """A replan cannot widen a host-owned model step back into generated code."""

    source = inspect.getsource(phase_support._step_maybe_replan)
    normalized = source.index("normalize_replan_candidate(")
    rebound = source.index("_scientific_runtime_authorities.bind_plan(")
    validated = source.index("_scientific_runtime_authorities.validate_plan(revised)")
    contract_review = source.index("replan_candidate_contract_findings(")
    registered = source.index("register_plan_revision(revised, reason=reason)")

    assert normalized < rebound < validated < contract_review < registered
    # The rebind binds the revised plan on the run's source, as the first bind does.
    call = source[rebound:validated].split("bind_plan(", 1)[1]
    assert call.lstrip().startswith("revised,")
    assert "context=context" in call
    assert "replanner_scientific_runtime_rebind_failed" in source


@pytest.mark.parametrize("has_reviewed_delivery", [False, True])
def test_replanner_cannot_add_an_unapproved_delivery_owner(tmp_path, monkeypatch, has_reviewed_delivery):
    from types import SimpleNamespace

    current = _plan(revision=1, intent="Summarize collection shifts")
    reviewed_deliveries = [AnalysisStep(
        step_id="bibliography", intent="Deliver the reviewed source list", host_delivery_kind="references",
    )] if has_reviewed_delivery else []
    current.steps.extend(reviewed_deliveries)
    candidate = _plan(revision=2, intent="Compare aggregate recovery across shifts")
    candidate.steps.append(AnalysisStep(
        step_id="new_pdf", intent="Typeset the shift summary", host_delivery_kind="manuscript_pdf",
    ))
    runtime = SimpleNamespace(bind_plan=lambda plan, **_: (plan, []), validate_plan=lambda _: None)
    pipeline = SimpleNamespace(
        _enable_replanning=True, _planner_strategy="progressive_v2", _max_total_steps=16,
        _scientific_runtime_authorities=runtime, _max_replans=0, _max_consecutive_noop_replans=0,
        _config=SimpleNamespace(require_human_plan_review=False),
    )
    monkeypatch.setattr(phase_support, "ReplannerAgent", lambda _: SimpleNamespace(run=lambda **_: candidate))
    monkeypatch.setattr(phase_support, "robustness_specs_for_execution", lambda **_: {})
    monkeypatch.setattr(
        phase_support, "normalize_replan_candidate",
        lambda **kwargs: SimpleNamespace(plan=kwargs["candidate_plan"], findings=[]),
    )
    monkeypatch.setattr(phase_support, "replan_candidate_contract_findings", lambda **_: [])
    registered = []

    def register(plan, *, reason):
        registered.append(plan)
        return tmp_path / "revised.json"

    result = SimpleNamespace(agent_context=None, allowed_literature_citation_keys=[], direct_comparator_literature_keys=[])
    revised, path = phase_support._step_maybe_replan(
        current_plan=current, reason="aggregate_shape_changed", plan_path=tmp_path / "original.json",
        _replan_state={"disabled": False, "noop_streak": 0, "total": 0}, context=None, findings=[],
        pipeline=pipeline, plan_result=result, role_resolver=lambda _: None, run_dir=tmp_path, skill_obj=None,
        flush_partial_manifest=lambda: None, resolve_cohort_definition=lambda *_args, **_kwargs: None,
        register_plan_revision=register, force=True,
    )
    assert path == tmp_path / "revised.json"
    assert registered == [revised]
    assert revised.steps == [*candidate.execution_steps, *reviewed_deliveries]
