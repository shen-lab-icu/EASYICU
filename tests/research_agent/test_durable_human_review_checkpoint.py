"""Cross-process human-review checkpoint regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.orchestration.human_review_checkpoint import (
    HumanReviewCheckpointError,
    load_checkpoint,
    write_checkpoint,
)
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision
from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopLedger,
    ProviderHardStopLimits,
)
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.providers.mocks import MockLLMClient


def _pipeline(root: Path, **overrides) -> ResearchAgentPipeline:
    options = {
        "workdir": root,
        "llm": MockLLMClient(),
        "require_human_plan_review": True,
        "enable_visual_qa": False,
        "enable_publication_figure_skill": False,
        "enable_nature_writing_skill": False,
    }
    options.update(overrides)
    return ResearchAgentPipeline(**options)


def _force_approvable_plan_review(monkeypatch) -> None:
    """Remove unrelated Mock-LLM findings while keeping real review authority."""

    import easyicu.research_agent.orchestration.workflow as workflow_module

    real = workflow_module.human_review_requests_for_plan

    def plan_only(**kwargs):
        request = dict(kwargs)
        request["findings"] = []
        request["require_plan_review"] = True
        return real(**request)

    monkeypatch.setattr(
        workflow_module,
        "human_review_requests_for_plan",
        plan_only,
    )


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [45, 60, 75] * 10,
            "death": [0, 1, 0] * 10,
        }
    )


def test_new_pipeline_instance_resumes_without_running_planner_again(
    monkeypatch, tmp_path
) -> None:
    _force_approvable_plan_review(monkeypatch)
    real_plan = ResearchAgentPipeline._run_plan_phase
    plan_calls = 0

    def counted_plan(self, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        return real_plan(self, **kwargs)

    monkeypatch.setattr(ResearchAgentPipeline, "_run_plan_phase", counted_plan)
    workdir = tmp_path / "runs"
    first = _pipeline(workdir)
    pending = first.run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    assert pending.resume_scope == "durable_checkpoint"
    assert plan_calls == 1

    # A different object stands in for a fresh process. It receives only the
    # checkpoint coordinate and decisions; Planner must not be called again.
    restored = _pipeline(workdir)
    result = restored.resume_human_review(
        [
            HumanReviewDecision(
                review_id=request.review_id,
                authority_sha256=request.authority_sha256,
                decision="approved",
                reviewer="test reviewer",
                decided_at="2026-08-14T03:12:00Z",
            )
            for request in pending.requests
        ],
        run_id=pending.run_id,
    )

    assert result.run_id == pending.run_id
    assert plan_calls == 1
    assert restored._approved_capability_resources == (
        restored._capability_runtime.approved_resources
    )
    checkpoint = load_checkpoint(
        Path(pending.run_dir) / "human_review_checkpoint.json",
        require_pending=False,
    )
    assert checkpoint.state == "completed"
    assert checkpoint.consumed_decision_sha256 is not None
    run_dir = Path(pending.run_dir)
    lineage_paths = list(run_dir.glob("plan_lifecycle_revision_*.json"))
    approved_paths = list(run_dir.glob("approved_executable_plan_revision_*.json"))
    assert len(lineage_paths) == 1
    assert len(approved_paths) == 1
    lineage = json.loads(lineage_paths[0].read_text(encoding="utf-8"))
    approved = json.loads(approved_paths[0].read_text(encoding="utf-8"))
    assert lineage["schema_version"] == "easyicu.normalized_plan/2"
    assert lineage["proposed"]["schema_version"] == "easyicu.proposed_plan/2"
    assert lineage["proposed"]["cohort_concept_ids"]
    assert lineage["transformation_receipts"]
    assert approved["schema_version"] == "easyicu.approved_executable_plan/2"
    assert approved["cohort_concept_ids"] == lineage["proposed"][
        "cohort_concept_ids"
    ]
    assert checkpoint.plan_handoff["cohort_concept_ids"] == lineage["proposed"][
        "cohort_concept_ids"
    ]
    assert approved["plan_sha256"] == lineage["plan_sha256"]
    assert approved["normalized_plan_authority_sha256"] == lineage["authority_sha256"]
    assert approved["decision_set_sha256"] == checkpoint.consumed_decision_sha256


def test_same_pipeline_resumes_its_paused_provider_clock(monkeypatch, tmp_path) -> None:
    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pipeline = _pipeline(workdir)
    limits = ProviderHardStopLimits(
        max_provider_attempts_per_run=100,
        max_provider_attempts_per_batch=100,
        max_total_tokens_per_run=1_000_000,
        max_total_tokens_per_batch=1_000_000,
        max_estimated_cost_usd_per_batch=100.0,
        max_wall_clock_seconds_per_task=600.0,
        input_cost_usd_per_million_tokens=20.0,
        output_cost_usd_per_million_tokens=120.0,
    )
    ledger = ProviderHardStopLedger(
        path=tmp_path / "provider_hard_stop.json",
        task_ids=("task-a",),
        limits=limits,
        batch_id="test-batch",
        declaration_sha256="a" * 64,
    )
    pipeline._provider_hard_stop = ledger.start_task("task-a")

    pending = pipeline.run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    assert ledger.snapshot()["tasks"][0]["status"] == "paused"

    result = pipeline.resume_human_review(
        [
            HumanReviewDecision(
                review_id=request.review_id,
                authority_sha256=request.authority_sha256,
                decision="approved",
                reviewer="test reviewer",
                decided_at="2026-08-14T03:12:00Z",
            )
            for request in pending.requests
        ],
        run_id=pending.run_id,
    )

    assert result.run_id == pending.run_id
    assert ledger.snapshot()["tasks"][0]["status"] != "paused"


def test_exhausted_provider_resume_is_terminal_and_not_masked(
    monkeypatch, tmp_path
) -> None:
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopExceeded,
    )

    _force_approvable_plan_review(monkeypatch)
    pipeline = _pipeline(tmp_path / "runs")
    pending = pipeline.run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )

    class ExhaustedTask:
        def resume(self) -> None:
            raise ProviderHardStopExceeded(
                code="TASK_WALL_CLOCK_EXHAUSTED",
                detail="review resume exhausted the active-time budget",
            )

        def pause(self) -> None:
            raise AssertionError("terminal budget exhaustion must not be re-paused")

    pipeline._provider_hard_stop = ExhaustedTask()
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="approved",
            reviewer="test reviewer",
            decided_at="2026-08-14T03:12:00Z",
        )
        for request in pending.requests
    ]

    with pytest.raises(ProviderHardStopExceeded) as stopped:
        pipeline.resume_human_review(decisions, run_id=pending.run_id)

    assert stopped.value.code == "TASK_WALL_CLOCK_EXHAUSTED"
    assert pipeline.has_resumable_human_review is False


def test_changed_pipeline_configuration_cannot_resume_checkpoint(
    monkeypatch, tmp_path
) -> None:
    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )

    changed = _pipeline(workdir, manuscript_language="zh")
    with pytest.raises(HumanReviewCheckpointError, match="configuration changed"):
        changed.resume_human_review([], run_id=pending.run_id)

    assert load_checkpoint(
        Path(pending.run_dir) / "human_review_checkpoint.json"
    ).state == "pending"


def test_checkpoint_tampering_fails_before_a_pause_is_restored(
    monkeypatch, tmp_path
) -> None:
    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    checkpoint_path = Path(pending.run_dir) / "human_review_checkpoint.json"
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["execution_coordinates"]["database"] = "tampered"
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HumanReviewCheckpointError, match="corrupt or invalid"):
        _pipeline(workdir).resume_human_review([], run_id=pending.run_id)


def test_execution_start_receipt_allows_restart_before_first_execute_side_effect(
    monkeypatch, tmp_path
) -> None:
    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="approved",
            reviewer="test reviewer",
            decided_at="2026-08-14T03:12:00Z",
        )
        for request in pending.requests
    ]
    real_execute = ResearchAgentPipeline._run_execute_phase

    def crash_before_execute_side_effect(self, **kwargs):
        raise SystemExit("simulated process crash")

    monkeypatch.setattr(
        ResearchAgentPipeline,
        "_run_execute_phase",
        crash_before_execute_side_effect,
    )
    with pytest.raises(SystemExit, match="simulated process crash"):
        _pipeline(workdir).resume_human_review(
            decisions,
            run_id=pending.run_id,
        )

    checkpoint_path = Path(pending.run_dir) / "human_review_checkpoint.json"
    interrupted = load_checkpoint(checkpoint_path, require_pending=False)
    assert interrupted.state == "executing"
    assert interrupted.execution_start_receipt is not None
    assert interrupted.approved_decisions
    assert interrupted.approved_decision_records

    monkeypatch.setattr(
        ResearchAgentPipeline,
        "_run_execute_phase",
        real_execute,
    )
    result = _pipeline(workdir).resume_human_review(
        decisions,
        run_id=pending.run_id,
    )

    assert result.run_id == pending.run_id
    assert load_checkpoint(checkpoint_path, require_pending=False).state == "completed"


def test_restart_reconciles_provider_resumed_before_decision_commit(
    monkeypatch, tmp_path
) -> None:
    from easyicu.research_agent.orchestration import human_review_restore

    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="approved",
            reviewer="test reviewer",
            decided_at="2026-08-14T03:12:00Z",
        )
        for request in pending.requests
    ]
    events: list[tuple[str, str]] = []

    class _ProviderTask:
        state = "paused"

        def reconcile_review_pause(self, *, paused_at: str) -> None:
            events.append(("reconcile", paused_at))
            self.state = "paused"

        def resume(self) -> None:
            events.append(("resume", self.state))
            self.state = "running"

        def pause(self) -> None:
            events.append(("pause", self.state))
            self.state = "paused"

        def assert_active(self) -> float:
            return 600.0

        def cap_timeout(self, requested_seconds: float) -> float:
            if self.state == "paused":
                raise AssertionError(
                    "durable runtime preflight must not inspect paused Provider time"
                )
            return requested_seconds

    provider = _ProviderTask()
    real_recorder = human_review_restore.persist_human_review_records
    monkeypatch.setattr(
        human_review_restore,
        "persist_human_review_records",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SystemExit("crash before decision commit")
        ),
    )
    interrupted = _pipeline(workdir)
    interrupted._provider_hard_stop = provider
    with pytest.raises(SystemExit, match="before decision commit"):
        interrupted.resume_human_review(decisions, run_id=pending.run_id)

    checkpoint_path = Path(pending.run_dir) / "human_review_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path, require_pending=False)
    assert checkpoint.state == "pending"
    assert provider.state == "running"

    monkeypatch.setattr(
        human_review_restore, "persist_human_review_records", real_recorder
    )
    restarted = _pipeline(workdir)
    restarted._provider_hard_stop = provider
    result = restarted.resume_human_review(decisions, run_id=pending.run_id)

    assert result.run_id == pending.run_id
    assert events[:4] == [
        ("reconcile", checkpoint.created_at),
        ("resume", "paused"),
        ("reconcile", checkpoint.created_at),
        ("resume", "paused"),
    ]
    assert load_checkpoint(checkpoint_path, require_pending=False).state == "completed"


def test_restart_after_decision_checkpoint_reuses_exact_server_stamped_records(
    monkeypatch, tmp_path
) -> None:
    from easyicu.research_agent.orchestration import human_review_restore

    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="approved",
            reviewer="test reviewer",
            decided_at="2026-08-14T03:12:00Z",
        )
        for request in pending.requests
    ]
    real_recorder = human_review_restore.persist_human_review_records

    def crash_before_decision_evidence(*_args, **_kwargs):
        raise SystemExit("crash after durable decision prepare")

    monkeypatch.setattr(
        human_review_restore,
        "persist_human_review_records",
        crash_before_decision_evidence,
    )
    with pytest.raises(SystemExit, match="durable decision prepare"):
        _pipeline(workdir).resume_human_review(decisions, run_id=pending.run_id)

    checkpoint_path = Path(pending.run_dir) / "human_review_checkpoint.json"
    interrupted = load_checkpoint(checkpoint_path, require_pending=False)
    assert interrupted.state == "pending"
    assert interrupted.approved_decisions
    stamped_records = list(interrupted.approved_decision_records)

    monkeypatch.setattr(
        human_review_restore,
        "persist_human_review_records",
        real_recorder,
    )
    result = _pipeline(workdir).resume_human_review(decisions, run_id=pending.run_id)

    persisted = json.loads(
        (Path(pending.run_dir) / "human_review_decisions.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.run_id == pending.run_id
    assert persisted["decisions"] == stamped_records
    assert load_checkpoint(checkpoint_path, require_pending=False).state == "completed"


def test_rejection_recorder_crash_converges_after_restart(monkeypatch, tmp_path) -> None:
    from easyicu.research_agent.orchestration import human_review_restore
    from easyicu.research_agent.orchestration.workflow import HumanReviewRejected

    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="rejected",
            reviewer="test reviewer",
            decided_at="2026-08-14T03:12:00Z",
        )
        for request in pending.requests
    ]
    real_recorder = human_review_restore.persist_human_review_records
    monkeypatch.setattr(
        human_review_restore,
        "persist_human_review_records",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit("crash")),
    )
    with pytest.raises(SystemExit, match="crash"):
        _pipeline(workdir).resume_human_review(decisions, run_id=pending.run_id)

    checkpoint_path = Path(pending.run_dir) / "human_review_checkpoint.json"
    assert load_checkpoint(checkpoint_path, require_pending=False).state == "pending"
    monkeypatch.setattr(
        human_review_restore,
        "persist_human_review_records",
        real_recorder,
    )

    with pytest.raises(HumanReviewRejected):
        _pipeline(workdir).resume_human_review(decisions, run_id=pending.run_id)

    assert load_checkpoint(checkpoint_path, require_pending=False).state == "rejected"
    status = json.loads(
        (Path(pending.run_dir) / "run_status.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "human_review_rejected"


def test_rejection_restore_needs_no_execution_configuration_or_provider(
    monkeypatch,
    tmp_path,
) -> None:
    from easyicu.research_agent.orchestration import human_review_restore
    from easyicu.research_agent.orchestration.workflow import HumanReviewRejected

    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="rejected",
            reviewer="test reviewer",
            decided_at="2026-08-14T03:12:00Z",
        )
        for request in pending.requests
    ]
    monkeypatch.setattr(
        human_review_restore,
        "build_environment_identity",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("rejection must not inspect the execution environment")
        ),
    )

    with pytest.raises(HumanReviewRejected):
        _pipeline(workdir, llm=None, enable_visual_qa=True).resume_human_review(
            decisions,
            run_id=pending.run_id,
        )

    checkpoint_path = Path(pending.run_dir) / "human_review_checkpoint.json"
    assert load_checkpoint(checkpoint_path, require_pending=False).state == "rejected"
    assert not list(Path(pending.run_dir).glob("approved_executable_plan_revision_*.json"))


def test_durable_restore_rebinds_host_only_declared_levels(
    monkeypatch,
    tmp_path,
) -> None:
    from easyicu.research_agent.orchestration import human_review_restore
    from easyicu.research_agent.orchestration.workflow import HumanReviewRejected

    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    rebound_steps: list[str] = []
    real_bind = human_review_restore.bind_step_declared_levels

    def tracked_bind(step, context) -> None:
        rebound_steps.append(step.step_id)
        real_bind(step, context)

    monkeypatch.setattr(
        human_review_restore,
        "bind_step_declared_levels",
        tracked_bind,
    )
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="rejected",
            reviewer="test reviewer",
            decided_at="2026-08-15T04:30:00Z",
        )
        for request in pending.requests
    ]

    with pytest.raises(HumanReviewRejected):
        _pipeline(workdir, llm=None).resume_human_review(
            decisions,
            run_id=pending.run_id,
        )

    assert rebound_steps


def test_legacy_pending_checkpoint_converges_from_recorded_decision_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    from easyicu.research_agent.orchestration import human_review_restore

    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="approved",
            reviewer="test reviewer",
            decided_at="2026-08-14T03:12:00Z",
        )
        for request in pending.requests
    ]
    checkpoint_path = Path(pending.run_dir) / "human_review_checkpoint.json"
    original_pending = load_checkpoint(checkpoint_path, require_pending=False)
    real_commit = human_review_restore.commit_human_review_decision
    monkeypatch.setattr(
        human_review_restore,
        "commit_human_review_decision",
        lambda **_kwargs: (_ for _ in ()).throw(SystemExit("commit crash")),
    )
    with pytest.raises(SystemExit, match="commit crash"):
        _pipeline(workdir).resume_human_review(decisions, run_id=pending.run_id)
    assert (Path(pending.run_dir) / "human_review_decisions.json").is_file()

    # Recreate the historical ordering where evidence reached disk before the
    # checkpoint learned the exact decision records.
    write_checkpoint(checkpoint_path, original_pending)
    monkeypatch.setattr(
        human_review_restore,
        "commit_human_review_decision",
        real_commit,
    )
    result = _pipeline(workdir).resume_human_review(decisions, run_id=pending.run_id)

    assert result.run_id == pending.run_id
    assert load_checkpoint(checkpoint_path, require_pending=False).state == "completed"


@pytest.mark.parametrize(
    ("method_name", "expected_phase"),
    [
        ("_run_write_phase", "write_in_progress"),
        ("_finalise_success", "finalize_in_progress"),
    ],
)
def test_restart_fails_closed_in_post_analysis_side_effect_phase(
    monkeypatch,
    tmp_path,
    method_name,
    expected_phase,
) -> None:
    _force_approvable_plan_review(monkeypatch)
    workdir = tmp_path / "runs"
    pending = _pipeline(workdir).run(
        question="Does age describe hospital mortality?",
        cohort=_cohort(),
        target_outcome="death",
    )
    decisions = [
        HumanReviewDecision(
            review_id=request.review_id,
            authority_sha256=request.authority_sha256,
            decision="approved",
            reviewer="test reviewer",
            decided_at="2026-08-14T03:12:00Z",
        )
        for request in pending.requests
    ]
    calls = 0

    def crash_after_phase_marker(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise SystemExit("simulated post-analysis crash")

    monkeypatch.setattr(
        ResearchAgentPipeline,
        method_name,
        crash_after_phase_marker,
    )
    with pytest.raises(SystemExit, match="post-analysis crash"):
        _pipeline(workdir).resume_human_review(decisions, run_id=pending.run_id)

    checkpoint_path = Path(pending.run_dir) / "human_review_checkpoint.json"
    assert load_checkpoint(checkpoint_path, require_pending=False).state == expected_phase
    with pytest.raises(HumanReviewCheckpointError, match=expected_phase):
        _pipeline(workdir).resume_human_review(decisions, run_id=pending.run_id)
    assert calls == 1
