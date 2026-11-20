"""A restarted host seals the reviewed plan's adjustment roster before executing (B27).

A runtime contract may defer its adjustment roster to the reviewed plan; the
planning process seals it in memory.  A host restarted between data
preparation and approval rebuilt the pipeline from its launch configuration
and restored the paused review, so the first step executor read an unsealed
contract ("has not sealed its Planner-selected adjustment roster").  The
durable restore now seals the roster from the restored plan.  Fake authority
only; the pause, checkpoint and restore are real.
"""

from __future__ import annotations

from easyicu.research_agent.contracts.runtime import PlanPhaseResult
from easyicu.research_agent.orchestration.human_review_restore import (
    restore_durable_human_review_pause,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.pipeline import _load_resume_state
from tests.support.durable_human_review import (
    _cohort,
    _force_approvable_plan_review,
    _pipeline,
)

_QUESTION = "Does age describe hospital mortality?"


class _Sealed:
    def seal_adjustment_roster(self, plan):
        return self


class _Deferred:
    """A contract whose roster waits for the reviewed plan."""

    def __init__(self) -> None:
        self.sealed_with: list = []
        self.sealed = _Sealed()

    def seal_adjustment_roster(self, plan):
        self.sealed_with.append(plan)
        return self.sealed


def _restarted(tmp_path, monkeypatch):
    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question=_QUESTION, cohort=_cohort(), target_outcome="death"
    )
    # A fresh object stands in for the host restarted after the pause.
    host = _pipeline(workdir)
    deferred = _Deferred()
    host._scientific_runtime_authorities = ScientificRuntimeAuthorities(
        trajectory=None, current_case=deferred
    )
    return host, pending, deferred


def test_a_restored_review_seals_the_roster_from_its_plan(tmp_path, monkeypatch) -> None:
    host, pending, deferred = _restarted(tmp_path, monkeypatch)

    restore_durable_human_review_pause(
        host,
        run_id=pending.run_id,
        progress_callback=None,
        plan_result_factory=PlanPhaseResult,
        load_resume_state=_load_resume_state,
    )

    assert host._scientific_runtime_authorities.current_case is deferred.sealed
    (plan,) = deferred.sealed_with
    assert plan.research_question == _QUESTION
    assert plan.steps


def test_a_rejection_restore_seals_nothing(tmp_path, monkeypatch) -> None:
    host, pending, deferred = _restarted(tmp_path, monkeypatch)

    restore_durable_human_review_pause(
        host,
        run_id=pending.run_id,
        progress_callback=None,
        plan_result_factory=PlanPhaseResult,
        load_resume_state=_load_resume_state,
        rejection_only=True,
    )

    assert deferred.sealed_with == []
    assert host._scientific_runtime_authorities.current_case is deferred
