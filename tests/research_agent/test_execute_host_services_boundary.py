"""Architecture locks for the acyclic execute-phase host-service boundary."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from easyicu.research_agent import pipeline, pipeline_execute
from easyicu.research_agent.execution import output_files
from easyicu.research_agent.execution.host_services import ExecutePhaseServices
from easyicu.research_agent.execution import publication_figure


def _parameter_shape(callable_object) -> tuple[tuple[str, object, object], ...]:
    return tuple(
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(callable_object).parameters.items()
    )


def _absolute_import_targets(module) -> set[str]:
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            targets.add(f"{prefix}{node.module or ''}")
    return targets


def test_execute_entrypoint_parameter_shape_is_stable() -> None:
    expected_names = (
        "pipeline",
        "plan_result",
        "cohort_path",
        "trajectory_binding",
        "run_dir",
        "run_id",
        "skill_obj",
        "notes",
        "emit_progress",
        "resume_from_step_id",
        "stop_after_step_id",
    )
    shape = _parameter_shape(pipeline_execute.run_execute_phase)
    assert tuple(item[0] for item in shape) == expected_names
    assert shape[0][1] is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(item[1] is inspect.Parameter.KEYWORD_ONLY for item in shape[1:])

    method_shape = _parameter_shape(pipeline.ResearchAgentPipeline._run_execute_phase)
    assert tuple(item[0] for item in method_shape) == ("self", *expected_names[1:])
    assert method_shape[0][1] is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(item[1] is inspect.Parameter.KEYWORD_ONLY for item in method_shape[1:])


def test_output_helper_legacy_paths_preserve_identity() -> None:
    assert pipeline._clear_output_dir is output_files._clear_output_dir
    assert pipeline_execute._clear_output_dir is output_files._clear_output_dir
    assert pipeline._has_figure_exports is output_files._has_figure_exports
    assert pipeline_execute._has_figure_exports is output_files._has_figure_exports


def test_execute_phase_services_are_fresh_and_resolve_current_host_helpers(
    monkeypatch,
) -> None:
    host = object.__new__(pipeline.ResearchAgentPipeline)
    first = host._execute_phase_services()
    assert isinstance(first, ExecutePhaseServices)

    def replacement(*_args, **_kwargs):
        return {"replacement": True}

    monkeypatch.setattr(pipeline, "_semantic_aliases_for", replacement)
    second = host._execute_phase_services()

    assert second is not first
    assert first.semantic_aliases_for is not replacement
    assert second.semantic_aliases_for is replacement


def test_execute_consumers_do_not_import_pipeline_backwards() -> None:
    for module in (pipeline_execute, publication_figure):
        targets = _absolute_import_targets(module)
        assert ".pipeline" not in targets
        assert "easyicu.research_agent.pipeline" not in targets


def test_publication_figure_authority_uses_injected_host_services() -> None:
    signature = inspect.signature(
        publication_figure._deterministic_publication_figure_code
    )
    assert "authority_services" in signature.parameters
    targets = _absolute_import_targets(publication_figure)
    assert ".host_services" in targets
