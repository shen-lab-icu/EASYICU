"""Cross-process human-review checkpoint regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.orchestration.human_review_checkpoint import (
    HumanReviewCheckpointError,
    load_checkpoint,
)
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision
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
    assert lineage["schema_version"] == "easyicu.normalized_plan/1"
    assert lineage["proposed"]["schema_version"] == "easyicu.proposed_plan/1"
    assert lineage["transformation_receipts"]
    assert approved["schema_version"] == "easyicu.approved_executable_plan/1"
    assert approved["plan_sha256"] == lineage["plan_sha256"]
    assert approved["normalized_plan_authority_sha256"] == lineage["authority_sha256"]
    assert approved["decision_set_sha256"] == checkpoint.consumed_decision_sha256


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
