"""Architecture locks for the acyclic execute-phase host-service boundary."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import hashlib
import inspect
from pathlib import Path

import pytest

from easyicu.research_agent import pipeline, pipeline_execute
from easyicu.research_agent.execution import output_files
from easyicu.research_agent.execution.host_services import (
    ExecutePhaseServices,
    PublicationFigureAuthorityServices,
)
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
    assert tuple(item[2] for item in shape[:-2]) == (inspect.Parameter.empty,) * 9
    assert tuple(item[2] for item in shape[-2:]) == (None, None)

    method_shape = _parameter_shape(pipeline.ResearchAgentPipeline._run_execute_phase)
    assert tuple(item[0] for item in method_shape) == ("self", *expected_names[1:])
    assert method_shape[0][1] is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(item[1] is inspect.Parameter.KEYWORD_ONLY for item in method_shape[1:])
    assert (
        tuple(item[2] for item in method_shape[:-2]) == (inspect.Parameter.empty,) * 9
    )
    assert tuple(item[2] for item in method_shape[-2:]) == (None, None)


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
    assert first.build_probe_summary is pipeline._build_probe_summary
    assert (
        first.deterministic_figure_family_supported_for_upstream
        is pipeline.deterministic_figure_family_supported_for_upstream
    )
    assert (
        first.promote_prior_publication_bundle
        is pipeline._promote_prior_publication_bundle
    )
    assert (
        first.promote_sibling_figure_exports is pipeline._promote_sibling_figure_exports
    )
    assert (
        first.render_publication_bundle_from_prior_outputs_for_step
        is pipeline._render_publication_bundle_from_prior_outputs_for_step
    )
    assert first.semantic_aliases_for is pipeline._semantic_aliases_for
    assert (
        first.publication_figure_authority.distribution_availability_step_matches_parent
        is pipeline._distribution_availability_figure_step_matches_parent
    )
    assert (
        first.publication_figure_authority.sealed_renderer_step_matches_parent
        is pipeline._sealed_renderer_figure_step_matches_parent
    )
    assert (
        first.publication_figure_authority.sealed_renderer_parent_digest_seal
        is pipeline._sealed_renderer_parent_digest_seal
    )
    assert (
        first.publication_figure_authority.deterministic_repair_id_for_upstream
        is pipeline.deterministic_figure_repair_id_for_upstream
    )

    def replacement(*_args, **_kwargs):
        return {"replacement": True}

    monkeypatch.setattr(pipeline, "_semantic_aliases_for", replacement)
    authority_names = (
        "_distribution_availability_figure_step_matches_parent",
        "_sealed_renderer_figure_step_matches_parent",
        "_sealed_renderer_parent_digest_seal",
        "deterministic_figure_repair_id_for_upstream",
    )

    def replacement_for(name):
        def authority_replacement(*_args, **_kwargs):
            return name

        return authority_replacement

    authority_replacements = {}
    for name in authority_names:
        replacement_callable = replacement_for(name)
        authority_replacements[name] = replacement_callable
        monkeypatch.setattr(pipeline, name, replacement_callable)
    second = host._execute_phase_services()

    assert second is not first
    assert first.semantic_aliases_for is not replacement
    assert second.semantic_aliases_for is replacement
    assert (
        second.publication_figure_authority.distribution_availability_step_matches_parent
        is authority_replacements[
            "_distribution_availability_figure_step_matches_parent"
        ]
    )
    assert (
        second.publication_figure_authority.sealed_renderer_step_matches_parent
        is authority_replacements["_sealed_renderer_figure_step_matches_parent"]
    )
    assert (
        second.publication_figure_authority.sealed_renderer_parent_digest_seal
        is authority_replacements["_sealed_renderer_parent_digest_seal"]
    )
    assert (
        second.publication_figure_authority.deterministic_repair_id_for_upstream
        is authority_replacements["deterministic_figure_repair_id_for_upstream"]
    )

    with pytest.raises(FrozenInstanceError):
        second.semantic_aliases_for = replacement  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        second.publication_figure_authority.deterministic_repair_id_for_upstream = (
            replacement
        )  # type: ignore[misc]

    assert isinstance(
        second.publication_figure_authority, PublicationFigureAuthorityServices
    )


def test_execute_consumers_do_not_import_pipeline_backwards() -> None:
    from easyicu.research_agent.execution import host_services

    for module in (
        pipeline_execute,
        publication_figure,
        host_services,
        output_files,
    ):
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


def test_publication_candidate_template_is_byte_identical() -> None:
    path = Path(publication_figure.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    templates = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "candidate_code"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert len(templates) == 1
    encoded = templates[0].encode("utf-8")
    assert len(encoded) == 3007
    assert hashlib.sha256(encoded).hexdigest() == (
        "175cfd2dbdc2ca5c20ad9b8ecb7d4ed455d1d47ce8b81af9bcd02470da0314e5"
    )


def test_staging_renderer_is_explicit_and_all_production_calls_supply_it() -> None:
    signature = inspect.signature(
        pipeline_execute._repair_publication_figure_in_staging
    )
    assert signature.parameters["renderer"].default is inspect.Parameter.empty

    source_path = Path(pipeline_execute.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    execute_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_execute_phase"
    )
    calls = [
        node
        for node in ast.walk(execute_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_repair_publication_figure_in_staging"
    ]
    assert len(calls) == 2
    for call in calls:
        renderer_keywords = [
            keyword for keyword in call.keywords if keyword.arg == "renderer"
        ]
        assert len(renderer_keywords) == 1
        value = renderer_keywords[0].value
        assert isinstance(value, ast.Name)
        assert value.id == "_render_publication_bundle_from_prior_outputs_for_step"
