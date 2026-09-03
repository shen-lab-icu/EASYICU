"""Permanent contract checks for the typed pipeline configuration surface."""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest


def test_pipeline_constructor_has_one_config_and_one_service_source(ra) -> None:
    """Keep declarative settings and live collaborators on separate surfaces."""
    parameters = inspect.signature(ra.ResearchAgentPipeline.__init__).parameters

    assert set(parameters) == {"self", "config", "services", "legacy_options"}
    assert parameters["legacy_options"].kind is inspect.Parameter.VAR_KEYWORD

    config_fields = {item.name for item in fields(ra.PipelineConfig)}
    service_fields = {item.name for item in fields(ra.PipelineServices)}
    assert service_fields == {
        "case_plugin_registry",
        "human_review_gate",
        "llm",
        "llm_concept_auditor_client",
        "provider_hard_stop",
        "runner_factory",
        "visual_qa_adapter",
        "vlm_client",
    }
    assert service_fields.isdisjoint(config_fields)


def test_pipeline_config_rejects_unknown_keys(ra, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="task_king"):
        ra.PipelineConfig.from_kwargs(workdir=tmp_path, task_king="prediction")


def test_reportable_capability_requirement_cannot_skip_plan_review(
    ra, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="requires require_human_plan_review"):
        ra.PipelineConfig(
            workdir=tmp_path,
            require_reportable_scientific_capability=True,
        )


def test_task_kind_round_trips_through_config_and_pipeline(ra, tmp_path: Path) -> None:
    config = ra.PipelineConfig.from_kwargs(
        workdir=tmp_path,
        task_kind="subphenotype_clustering",
    )

    kwargs = config.as_kwargs()
    assert kwargs["task_kind"] == "subphenotype_clustering"

    pipeline = ra.ResearchAgentPipeline.from_config(config)
    assert pipeline._config == config
    assert pipeline._benchmark_task_kind == "subphenotype_clustering"


def test_primary_cohort_selection_mode_is_typed_and_bound(ra, tmp_path: Path) -> None:
    config = ra.PipelineConfig(
        workdir=tmp_path,
        required_primary_cohort_selection_mode="all_input_rows",
    )
    pipeline = ra.ResearchAgentPipeline.from_config(config)

    assert pipeline._required_primary_cohort_selection_mode == "all_input_rows"
    with pytest.raises(ValueError, match="required_primary_cohort_selection_mode"):
        ra.PipelineConfig(
            workdir=tmp_path,
            required_primary_cohort_selection_mode="guess",
        )


def test_progressive_planner_strategy_is_typed_and_bound(ra, tmp_path: Path) -> None:
    config = ra.PipelineConfig(
        workdir=tmp_path,
        planner_strategy="progressive_v2",
    )
    pipeline = ra.ResearchAgentPipeline.from_config(config)

    assert pipeline._planner_strategy == "progressive_v2"
    assert config.canonical_payload()["planner_strategy"] == "progressive_v2"
    with pytest.raises(ValueError, match="planner_strategy"):
        ra.PipelineConfig(workdir=tmp_path, planner_strategy="guess")


def test_progressive_resume_is_explicitly_development_only(
    ra, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "progressive_planner_checkpoint_004.json"
    digest = "a" * 64

    with pytest.raises(ValueError, match="configured together"):
        ra.PipelineConfig(
            workdir=tmp_path,
            development_progressive_resume_checkpoint_path=checkpoint_path,
        )
    with pytest.raises(ValueError, match="development_diagnostic=True"):
        ra.PipelineConfig(
            workdir=tmp_path,
            planner_strategy="progressive_v2",
            development_progressive_resume_checkpoint_path=checkpoint_path,
            development_progressive_resume_checkpoint_sha256=digest,
        )
    with pytest.raises(ValueError, match="planner_strategy='progressive_v2'"):
        ra.PipelineConfig(
            workdir=tmp_path,
            development_diagnostic=True,
            development_progressive_resume_checkpoint_path=checkpoint_path,
            development_progressive_resume_checkpoint_sha256=digest,
        )
    with pytest.raises(ValueError, match="fallback plan"):
        ra.PipelineConfig(
            workdir=tmp_path,
            development_diagnostic=True,
            planner_strategy="progressive_v2",
            enable_deterministic_planner_fallback=True,
            development_progressive_resume_checkpoint_path=checkpoint_path,
            development_progressive_resume_checkpoint_sha256=digest,
        )

    config = ra.PipelineConfig(
        workdir=tmp_path,
        development_diagnostic=True,
        planner_strategy="progressive_v2",
        development_progressive_resume_checkpoint_path=checkpoint_path,
        development_progressive_resume_checkpoint_sha256=digest,
    )
    pipeline = ra.ResearchAgentPipeline.from_config(config)

    assert pipeline._config.development_progressive_resume_checkpoint_path == (
        checkpoint_path
    )
    assert (
        pipeline._config.development_progressive_resume_checkpoint_sha256
        == digest
    )
    assert config.canonical_payload()[
        "development_progressive_resume_checkpoint_sha256"
    ] == digest

    profile_config = ra.PipelineConfig(
        workdir=tmp_path,
        planner_strategy="progressive_v2",
        submission_profile_name="npj_dm_e1_canary_dev",
        submission_profile_version="20260817",
        development_progressive_resume_checkpoint_path=checkpoint_path,
        development_progressive_resume_checkpoint_sha256=digest,
    )
    assert profile_config.submission_profile_name == "npj_dm_e1_canary_dev"

    with pytest.raises(ValueError, match="paper-facing submission profile"):
        ra.PipelineConfig(
            workdir=tmp_path,
            development_diagnostic=True,
            planner_strategy="progressive_v2",
            submission_profile_name="npj_dm",
            submission_profile_version="20260719",
            development_progressive_resume_checkpoint_path=checkpoint_path,
            development_progressive_resume_checkpoint_sha256=digest,
        )


def test_locked_analysis_plan_is_digest_bound_and_development_only(
    ra, tmp_path: Path
) -> None:
    plan_path = tmp_path / "analysis_plan.json"
    digest = "b" * 64

    with pytest.raises(ValueError, match="configured together"):
        ra.PipelineConfig(
            workdir=tmp_path,
            development_locked_analysis_plan_path=plan_path,
        )
    with pytest.raises(ValueError, match="development_diagnostic=True"):
        ra.PipelineConfig(
            workdir=tmp_path,
            development_locked_analysis_plan_path=plan_path,
            development_locked_analysis_plan_sha256=digest,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        ra.PipelineConfig(
            workdir=tmp_path,
            development_diagnostic=True,
            planner_strategy="progressive_v2",
            development_progressive_resume_checkpoint_path=tmp_path / "cp.json",
            development_progressive_resume_checkpoint_sha256="a" * 64,
            development_locked_analysis_plan_path=plan_path,
            development_locked_analysis_plan_sha256=digest,
        )

    config = ra.PipelineConfig(
        workdir=tmp_path,
        development_diagnostic=True,
        development_locked_analysis_plan_path=plan_path,
        development_locked_analysis_plan_sha256=digest,
    )
    pipeline = ra.ResearchAgentPipeline.from_config(config)
    assert pipeline._config.development_locked_analysis_plan_path == plan_path
    assert config.canonical_payload()[
        "development_locked_analysis_plan_sha256"
    ] == digest


def test_pipeline_accepts_development_diagnostic_with_dev_profile(
    ra, tmp_path: Path
) -> None:
    from easyicu.research_agent.orchestration.profiles import (
        E1_PROGRESSIVE_PLANNER_CANARY_2026_08_19,
    )

    config = ra.PipelineConfig(
        workdir=tmp_path,
        development_diagnostic=True,
        **E1_PROGRESSIVE_PLANNER_CANARY_2026_08_19.pipeline_options(),
    )

    pipeline = ra.ResearchAgentPipeline.from_config(config)

    assert pipeline._development_diagnostic is True
    assert pipeline._submission_profile_name == "npj_dm_e1_canary_dev"


def test_planner_efficiency_budget_is_complete_and_development_only(
    ra, tmp_path: Path
) -> None:
    limits = {
        "development_planner_efficiency_max_calls": 6,
        "development_planner_efficiency_max_reported_tokens": 100_000,
        "development_planner_efficiency_max_wall_seconds": 600.0,
    }

    with pytest.raises(ValueError, match="configured together"):
        ra.PipelineConfig(
            workdir=tmp_path,
            development_diagnostic=True,
            planner_strategy="progressive_v2",
            development_planner_efficiency_max_calls=6,
        )
    with pytest.raises(ValueError, match="development-only profile"):
        ra.PipelineConfig(
            workdir=tmp_path,
            planner_strategy="progressive_v2",
            **limits,
        )
    with pytest.raises(ValueError, match="planner_strategy='progressive_v2'"):
        ra.PipelineConfig(
            workdir=tmp_path,
            development_diagnostic=True,
            **limits,
        )
    with pytest.raises(ValueError, match="paper-facing submission profile"):
        ra.PipelineConfig(
            workdir=tmp_path,
            development_diagnostic=True,
            planner_strategy="progressive_v2",
            submission_profile_name="npj_dm",
            submission_profile_version="20260719",
            **limits,
        )

    config = ra.PipelineConfig(
        workdir=tmp_path,
        planner_strategy="progressive_v2",
        submission_profile_name="npj_dm_e1_canary_dev",
        submission_profile_version="20260817",
        **limits,
    )

    assert config.canonical_payload()[
        "development_planner_efficiency_max_reported_tokens"
    ] == 100_000
    recovered = ra.PipelineConfig.from_recovery_payload(
        config.recovery_payload(),
        expected_digest=config.canonical_digest(),
    )
    assert recovered.development_planner_efficiency_max_calls == 6
    assert recovered.development_planner_efficiency_max_wall_seconds == 600.0


def test_outline_only_planner_termination_is_strictly_development_design_canary(
    ra, tmp_path: Path
) -> None:
    base = {
        "workdir": tmp_path,
        "development_diagnostic": True,
        "planner_strategy": "progressive_v2",
        "planner_only": True,
        "require_human_plan_review": True,
        "require_literature_design_authority": True,
        "enable_literature": True,
        "development_stop_after_planner_outline": True,
    }

    config = ra.PipelineConfig(**base)
    assert config.development_stop_after_planner_outline is True

    with pytest.raises(ValueError, match="planner_only=True"):
        ra.PipelineConfig(**{**base, "planner_only": False})
    with pytest.raises(ValueError, match="require_literature_design_authority=True"):
        ra.PipelineConfig(
            **{**base, "require_literature_design_authority": False}
        )
    with pytest.raises(ValueError, match="progressive_v2"):
        ra.PipelineConfig(**{**base, "planner_strategy": "monolithic_v1"})


def test_legacy_flat_constructor_is_a_warning_only_adapter(ra, tmp_path: Path) -> None:
    client = ra.MockLLMClient()

    with pytest.warns(DeprecationWarning, match="PipelineConfig"):
        pipeline = ra.ResearchAgentPipeline(
            workdir=tmp_path,
            llm=client,
            task_kind="prediction",
        )

    assert pipeline._config.task_kind == "prediction"
    assert pipeline._services.llm is client
    assert pipeline._llm is client


def test_config_cannot_be_mixed_with_legacy_options(ra, tmp_path: Path) -> None:
    config = ra.PipelineConfig(workdir=tmp_path)

    with pytest.raises(TypeError, match="complete declarative source"):
        ra.ResearchAgentPipeline(config=config, task_kind="prediction")
