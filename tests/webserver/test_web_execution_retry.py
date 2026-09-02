from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from tests.support.figure2 import seal_test_run_input_capsule


@pytest.mark.parametrize(
    "gate_reason",
    [
        "research_agent_pipeline_failed_closed",
        "research_pipeline_execution_failed",
    ],
)
def test_completed_approved_run_can_retry_post_execution_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_reason: str,
) -> None:
    root = tmp_path / "projects"
    wrapper = root / "study" / "run-wrapper"
    run_dir = wrapper / "pipeline" / "run-analysis"
    run_dir.mkdir(parents=True)
    (run_dir / "human_review_checkpoint.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "gates": {
                    "execution_complete": True,
                    "evidence_complete": True,
                    "numeric_verified": True,
                    "analysis_validated": False,
                    "failed_steps": [],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        agent_pipeline_runs.agent_runs,
        "list_run_history",
        lambda **_kwargs: {
            "runs": [
                {
                    "run_id": "run-analysis",
                    "scientific_configuration_sha256": "a" * 64,
                    "gate_reason": gate_reason,
                    "run_status": "blocked",
                    "project_dir": str(wrapper),
                }
            ]
        },
    )
    monkeypatch.setattr(
        agent_pipeline_runs.study_context_owner,
        "scientific_configuration_sha256",
        lambda _study: "a" * 64,
    )
    from easyicu.research_agent.orchestration import human_review_checkpoint

    monkeypatch.setattr(
        human_review_checkpoint,
        "load_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            state="completed",
            approved_decisions=[{"decision": "approved"}],
            pipeline_config_sha256="b" * 64,
        ),
    )

    target = agent_pipeline_runs._resolve_execution_resume_wrapper(
        study={"id": "study"},
        project_root=str(root),
        source_run_id="run-analysis",
    )

    assert target.wrapper_dir == wrapper.resolve()
    assert target.pipeline_run_id == "run-analysis"
    assert target.pipeline_config_sha256 == "b" * 64


@pytest.mark.parametrize(
    "blocked_gate",
    [
        "artifact_valid",
        "evidence_complete",
        "numeric_verified",
        "analysis_validated",
        "manuscript_ready",
    ],
)
def test_completed_execution_can_retry_each_downstream_report_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_gate: str,
) -> None:
    root = tmp_path / "projects"
    wrapper = root / "study" / "run-wrapper"
    run_dir = wrapper / "pipeline" / "run-analysis"
    run_dir.mkdir(parents=True)
    (run_dir / "human_review_checkpoint.json").write_text("{}", encoding="utf-8")
    gates = {
        "execution_complete": True,
        "failed_steps": [],
        "artifact_valid": True,
        "evidence_complete": True,
        "numeric_verified": True,
        "analysis_validated": True,
        "manuscript_ready": True,
    }
    gates[blocked_gate] = False
    (run_dir / "run_status.json").write_text(
        json.dumps({"gates": gates}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        agent_pipeline_runs.agent_runs,
        "list_run_history",
        lambda **_kwargs: {
            "runs": [
                {
                    "run_id": "run-analysis",
                    "scientific_configuration_sha256": "a" * 64,
                    "gate_reason": "research_agent_pipeline_failed_closed",
                    "run_status": "blocked",
                    "project_dir": str(wrapper),
                }
            ]
        },
    )
    monkeypatch.setattr(
        agent_pipeline_runs.study_context_owner,
        "scientific_configuration_sha256",
        lambda _study: "a" * 64,
    )
    from easyicu.research_agent.orchestration import human_review_checkpoint

    monkeypatch.setattr(
        human_review_checkpoint,
        "load_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            state="completed",
            approved_decisions=[{"decision": "approved"}],
            pipeline_config_sha256="b" * 64,
        ),
    )

    target = agent_pipeline_runs._resolve_execution_resume_wrapper(
        study={"id": "study"},
        project_root=str(root),
        source_run_id="run-analysis",
    )

    assert target.wrapper_dir == wrapper.resolve()
    assert target.pipeline_run_id == "run-analysis"
    assert target.pipeline_config_sha256 == "b" * 64


def test_execution_retry_reuses_verified_sealed_pipeline_inputs(
    tmp_path: Path,
) -> None:
    wrapper = tmp_path / "projects" / "study" / "run-wrapper"
    run_dir = wrapper / "pipeline" / "run-analysis"
    run_dir.mkdir(parents=True)
    evidence = EvidenceStore(root=run_dir)
    capsule = seal_test_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        research_question="Does lactate predict mortality?",
        primary_exposure="lact",
        target_outcome="death",
    )

    inputs = agent_pipeline_runs._verified_execution_resume_inputs(
        agent_pipeline_runs._ExecutionResumeTarget(
            wrapper_dir=wrapper.resolve(),
            pipeline_run_id="run-analysis",
            pipeline_config_sha256="b" * 64,
        )
    )

    assert inputs.cohort_path == run_dir / capsule.cohort_relative_path
    assert inputs.cohort_authority_ref == capsule.materialized_cohort_authority_ref
    assert inputs.cohort_authority_path == (
        run_dir / capsule.materialized_cohort_authority_ref["file"]
    )
    assert inputs.trajectory_path is None


def test_execution_retry_accepts_missing_seed_only_for_exact_checkpoint_digest(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.orchestration.config import PipelineConfig

    wrapper = tmp_path / "projects" / "study" / "run-wrapper"
    config = PipelineConfig(workdir=wrapper / "pipeline")
    target = agent_pipeline_runs._ExecutionResumeTarget(
        wrapper_dir=wrapper.resolve(),
        pipeline_run_id="run-analysis",
        pipeline_config_sha256=config.canonical_digest(),
    )

    restored = agent_pipeline_runs._validated_execution_retry_config(
        current_config=config,
        target=target,
        recovery_seed=None,
        current_scientific_digest="a" * 64,
        prepared_package_binding=None,
    )

    assert restored is config


def test_execution_retry_rejects_missing_seed_when_checkpoint_digest_drifted(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.orchestration.config import PipelineConfig

    wrapper = tmp_path / "projects" / "study" / "run-wrapper"
    config = PipelineConfig(workdir=wrapper / "pipeline")
    target = agent_pipeline_runs._ExecutionResumeTarget(
        wrapper_dir=wrapper.resolve(),
        pipeline_run_id="run-analysis",
        pipeline_config_sha256="b" * 64,
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        agent_pipeline_runs._validated_execution_retry_config(
            current_config=config,
            target=target,
            recovery_seed=None,
            current_scientific_digest="a" * 64,
            prepared_package_binding=None,
        )

    assert raised.value.code == (
        "research_pipeline_execution_retry_recovery_seed_missing"
    )


def test_execution_retry_reconstructs_approved_live_search_profile(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.profiles import get_submission_profile
    from easyicu.webserver.research_launch_runtime import _submission_profile_ref

    wrapper = tmp_path / "projects" / "study" / "run-wrapper"
    no_search = get_submission_profile(
        _submission_profile_ref(budget_mode="full_reviewed", live_pubmed=False)
    )
    live_search = get_submission_profile(
        _submission_profile_ref(budget_mode="full_reviewed", live_pubmed=True)
    )
    current = PipelineConfig(
        workdir=wrapper / "pipeline",
        **no_search.pipeline_options(),
    )
    approved = PipelineConfig(
        workdir=wrapper / "pipeline",
        **live_search.pipeline_options(),
    )
    target = agent_pipeline_runs._ExecutionResumeTarget(
        wrapper_dir=wrapper.resolve(),
        pipeline_run_id="run-analysis",
        pipeline_config_sha256=approved.canonical_digest(),
    )

    restored = agent_pipeline_runs._validated_execution_retry_config(
        current_config=current,
        target=target,
        recovery_seed=None,
        current_scientific_digest="a" * 64,
        prepared_package_binding=None,
    )

    assert restored.canonical_digest() == approved.canonical_digest()
    assert restored.enable_pubmed is True
    assert restored.bound_preplan_literature is None


def test_failed_projection_keeps_retry_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[Path] = []
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_remove_local_recovery",
        lambda wrapper: removed.append(wrapper),
    )

    agent_pipeline_runs._cleanup_recovery_after_projection(
        wrapper_dir=tmp_path,
        projection={
            "gate": {
                "reason": "research_agent_pipeline_failed_closed",
            }
        },
    )

    assert removed == []


def test_completed_projection_prunes_retry_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed: list[Path] = []
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_remove_local_recovery",
        lambda wrapper: removed.append(wrapper),
    )

    agent_pipeline_runs._cleanup_recovery_after_projection(
        wrapper_dir=tmp_path,
        projection={
            "gate": {
                "reason": (
                    "research_agent_pipeline_complete_human_interpretation_required"
                ),
            }
        },
    )

    assert removed == [tmp_path]
