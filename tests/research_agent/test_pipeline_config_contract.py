"""Permanent contract checks for the typed pipeline configuration surface."""

from __future__ import annotations

import inspect
from dataclasses import MISSING, fields
from pathlib import Path

import pytest


def test_pipeline_config_matches_constructor_fields_and_defaults(ra) -> None:
    """Keep PipelineConfig an exact mirror of the legacy constructor."""
    config_fields = {item.name: item for item in fields(ra.PipelineConfig)}
    constructor_params = {
        name: parameter
        for name, parameter in inspect.signature(
            ra.ResearchAgentPipeline.__init__
        ).parameters.items()
        if name != "self"
    }

    assert set(config_fields) == set(constructor_params)

    for name, item in config_fields.items():
        parameter = constructor_params[name]
        if item.default is not MISSING:
            expected_default = item.default
        elif item.default_factory is not MISSING:
            expected_default = item.default_factory()
        else:
            assert parameter.default is inspect.Parameter.empty, name
            continue
        assert parameter.default == expected_default, name


def test_pipeline_config_rejects_unknown_keys(ra, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="task_king"):
        ra.PipelineConfig.from_kwargs(workdir=tmp_path, task_king="prediction")


def test_task_kind_round_trips_through_config_and_pipeline(
    ra, tmp_path: Path
) -> None:
    config = ra.PipelineConfig.from_kwargs(
        workdir=tmp_path,
        task_kind="subphenotype_clustering",
    )

    kwargs = config.as_kwargs()
    assert kwargs["task_kind"] == "subphenotype_clustering"

    pipeline = ra.ResearchAgentPipeline.from_config(config)
    assert pipeline._config == config
    assert pipeline._benchmark_task_kind == "subphenotype_clustering"
