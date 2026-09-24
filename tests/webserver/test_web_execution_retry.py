from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.run_input import (
    RunInputCapsuleV3,
    build_environment_identity,
    seal_run_input_capsule,
)
from easyicu.research_agent.intake.materialized_metadata import (
    load_verified_materialized_cohort_authority,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    StagedTrajectoryBinding,
    stage_materialized_trajectory_authority,
)
from tests.support.figure2 import seal_test_run_input_capsule
from tests.support.typed_trajectory import (
    trajectory_context_and_evidence,
    trajectory_implementation_sha,
    trajectory_scientific_identity,
    typed_trajectory_bundle,
)


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
    assert target.resume_from_step_id is None


def test_failed_execution_retry_binds_first_failed_step_as_explicit_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
                    "execution_complete": False,
                    "failed_steps": [
                        {"step_id": "02_model", "status": "execution_failed"},
                        {"step_id": "03_figure", "status": "dependency_failed"},
                    ],
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
                    "gate_reason": "research_pipeline_execution_failed",
                    "run_status": "failed",
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

    assert target.resume_from_step_id == "02_model"


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
        primary_exposure="lact_max",
        target_outcome="death",
        source_dir=wrapper / "pipeline_input",
    )

    inputs = agent_pipeline_runs._verified_execution_resume_inputs(
        agent_pipeline_runs._ExecutionResumeTarget(
            wrapper_dir=wrapper.resolve(),
            pipeline_run_id="run-analysis",
            pipeline_config_sha256="b" * 64,
        )
    )

    # The retry declares the approved run's source again, not its staged copy:
    # the pipeline resumes only when the staged copy descends from it.
    source_ref = capsule.scientific_identity["materialized_cohort_authority_ref"]
    assert inputs.cohort_authority_ref == source_ref
    assert inputs.cohort_authority_ref != capsule.materialized_cohort_authority_ref
    assert inputs.cohort_authority_path == (
        wrapper.resolve() / "pipeline_input" / source_ref["file"]
    )
    assert inputs.cohort_path.parent == wrapper.resolve() / "pipeline_input"
    staged = load_verified_materialized_cohort_authority(
        run_dir / capsule.cohort_relative_path
    )
    assert staged is not None
    assert staged.authority.parent_authority_sha256 == source_ref["sha256"]
    assert inputs.trajectory_path is None
    assert inputs.scientific_identity["primary_exposure"] == "lact_max"

    acquisition = agent_pipeline_runs._execution_resume_acquisition_projection(inputs)
    assert acquisition.universe_path == inputs.cohort_path
    assert acquisition.selection.selection_authority == "host_exact"
    assert acquisition.coverage.sufficient is True
    assert acquisition.analysis_columns == {
        "lact_max": "lact_max",
        "death": "death",
    }
    assert acquisition.note.startswith("Digest-verified execution retry projection")


def test_execution_retry_refuses_when_the_source_is_gone(tmp_path: Path) -> None:
    wrapper = tmp_path / "projects" / "study" / "run-wrapper"
    run_dir = wrapper / "pipeline" / "run-analysis"
    run_dir.mkdir(parents=True)
    capsule = seal_test_run_input_capsule(
        run_dir=run_dir,
        evidence=EvidenceStore(root=run_dir),
        research_question="Does lactate predict mortality?",
        primary_exposure="lact_max",
        target_outcome="death",
        source_dir=wrapper / "pipeline_input",
    )
    source_ref = capsule.scientific_identity["materialized_cohort_authority_ref"]
    (wrapper / "pipeline_input" / source_ref["file"]).unlink()

    with pytest.raises(ResearchPipelineRunError) as raised:
        agent_pipeline_runs._verified_execution_resume_inputs(
            agent_pipeline_runs._ExecutionResumeTarget(
                wrapper_dir=wrapper.resolve(),
                pipeline_run_id="run-analysis",
                pipeline_config_sha256="b" * 64,
            )
        )

    assert raised.value.code == "research_pipeline_execution_retry_input_invalid"


def test_execution_retry_redeclares_the_source_trajectory(tmp_path: Path) -> None:
    wrapper = tmp_path / "projects" / "study" / "run-wrapper"
    wrapper.mkdir(parents=True)
    paths, source_cohort, source_trajectory = typed_trajectory_bundle(wrapper)
    (wrapper / "materialized").rename(wrapper / "pipeline_input")
    universe = wrapper / "pipeline_input" / Path(paths["parquet"]).name
    trajectory = wrapper / "pipeline_input" / Path(paths["trajectory"]).name
    run_dir = wrapper / "pipeline" / "run-analysis"
    run_dir.mkdir(parents=True)
    cohort_path = run_dir / "cohort.parquet"
    staged_cohort = stage_materialized_cohort_authority(
        universe,
        cohort_path,
        expected_source_authority=source_cohort.reference,
        producer_implementation_sha256=trajectory_implementation_sha(),
    )
    assert staged_cohort is not None
    staged_trajectory = stage_materialized_trajectory_authority(
        trajectory,
        run_dir / "cohort_trajectory.parquet",
        source_universe_path=universe,
        target_universe_path=cohort_path,
        expected_source_authority=source_trajectory.reference,
        expected_target_universe_authority=staged_cohort.reference,
        producer_implementation_sha256=trajectory_implementation_sha(),
    )
    context_path, evidence = trajectory_context_and_evidence(
        run_dir,
        cohort_path,
        trajectory_binding=StagedTrajectoryBinding(
            path=run_dir / "cohort_trajectory.parquet",
            sha256=staged_trajectory.authority.trajectory_sha256,
            size=staged_trajectory.authority.trajectory_size,
            authority_ref=staged_trajectory.reference,
        ),
    )
    capsule = seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=trajectory_scientific_identity(
            cohort_path=universe,
            cohort_ref=source_cohort.reference,
            trajectory_path=trajectory,
            trajectory_ref=source_trajectory.reference,
        ),
        initial_environment=build_environment_identity(llm_signature="mock"),
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )
    assert isinstance(capsule, RunInputCapsuleV3)

    inputs = agent_pipeline_runs._verified_execution_resume_inputs(
        agent_pipeline_runs._ExecutionResumeTarget(
            wrapper_dir=wrapper.resolve(),
            pipeline_run_id="run-analysis",
            pipeline_config_sha256="b" * 64,
        )
    )

    assert inputs.cohort_path == wrapper.resolve() / "pipeline_input" / universe.name
    assert inputs.cohort_authority_ref == source_cohort.reference.to_dict()
    assert inputs.trajectory_path == (
        wrapper.resolve() / "pipeline_input" / trajectory.name
    )
    assert inputs.trajectory_authority_ref == source_trajectory.reference.to_dict()
    assert (
        staged_trajectory.authority.parent_trajectory_authority
        == source_trajectory.reference
    )


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


def test_execution_retry_projection_checks_secondary_outcome_columns(
    tmp_path: Path,
) -> None:
    import pandas as pd

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"lact_max": [2.0], "death": [0]}).to_parquet(cohort)
    inputs = agent_pipeline_runs._ExecutionResumeInputs(
        cohort_path=cohort,
        cohort_authority_path=None,
        cohort_authority_ref=None,
        trajectory_path=None,
        trajectory_authority_path=None,
        trajectory_authority_ref=None,
        scientific_identity={
            "primary_exposure": "lact_max",
            "target_outcome": "death",
            "outcome_columns": ["death", "los_icu"],
        },
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        agent_pipeline_runs._execution_resume_acquisition_projection(inputs)

    assert raised.value.code == "research_pipeline_execution_retry_input_invalid"
    assert raised.value.details == {"missing_column_count": 1}


@pytest.mark.parametrize("bound_column_sealed", [True, False])
def test_execution_retry_projection_reads_exact_covariates_through_their_bindings(
    tmp_path: Path, bound_column_sealed: bool,
) -> None:
    """Dev9 M1's retry was refused because it looked for ``sofa2_resp``.

    The analysis read the reviewed window summary ``sofa2_resp_max``; a bare
    time-series concept id never exists as a materialized column.
    """

    import pandas as pd

    cohort = tmp_path / "cohort.parquet"
    columns = {"bili_max": [1.0], "death": [0], "age": [70.0]}
    if bound_column_sealed:
        columns["sofa2_resp_max"] = [2.0]
    pd.DataFrame(columns).to_parquet(cohort)
    inputs = agent_pipeline_runs._ExecutionResumeInputs(
        cohort_path=cohort,
        cohort_authority_path=None,
        cohort_authority_ref=None,
        trajectory_path=None,
        trajectory_authority_path=None,
        trajectory_authority_ref=None,
        scientific_identity={
            "primary_exposure": "bili_max",
            "target_outcome": "death",
            "user_preferences": {
                "covariates": ["age", "sofa2_resp"],
                "covariate_operationalizations": {
                    "age": "age",
                    "sofa2_resp": "sofa2_resp_max",
                },
            },
        },
    )

    if not bound_column_sealed:
        with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
            agent_pipeline_runs._execution_resume_acquisition_projection(inputs)
        assert raised.value.code == "research_pipeline_execution_retry_input_invalid"
        assert raised.value.details == {"missing_column_count": 1}
        return

    acquisition = agent_pipeline_runs._execution_resume_acquisition_projection(inputs)

    assert acquisition.analysis_columns == {
        "bili_max": "bili_max",
        "death": "death",
        "age": "age",
        "sofa2_resp_max": "sofa2_resp_max",
    }
    assert acquisition.coverage.sufficient is True


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
