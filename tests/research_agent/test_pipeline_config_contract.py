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
