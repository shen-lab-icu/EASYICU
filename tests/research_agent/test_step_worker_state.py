from __future__ import annotations

import ast
from dataclasses import asdict
import inspect
from itertools import product
import textwrap


def _legacy_generation_mode(
    *,
    repair_attempts: int,
    fallback_used: bool,
    standard_executor_used: bool,
    runner_repair_name: str | None,
    resumed_code_reuse: bool,
    concept_repair_used: bool,
    llm_repair_used: bool,
) -> str:
    if standard_executor_used:
        return "deterministic_standard"
    if llm_repair_used:
        return "repaired"
    if fallback_used:
        return "fallback"
    if runner_repair_name:
        return "runner_repaired"
    if repair_attempts > 0 or concept_repair_used:
        return "repaired"
    if resumed_code_reuse:
        return "resumed_code_reuse"
    return "llm"


def test_worker_progress_defaults_are_isolated() -> None:
    from easyicu.research_agent.execution.step_worker_state import StepWorkerProgress

    first = StepWorkerProgress()
    second = StepWorkerProgress()

    assert asdict(first) == {
        "resumed_code_reuse_used": False,
        "critic_resume_repair_used": False,
        "deterministic_fallback_used": False,
        "deterministic_standard_executor_used": False,
        "preexecution_runner_repair_name": None,
        "runner_repair_name": None,
        "concept_repair_attempts": 0,
        "concept_audit_error_count": 0,
        "deterministic_concept_repairs": 0,
        "applied_concept_repair_names": [],
        "llm_repair_used": False,
        "repair_attempts": 0,
        "contract_repair_attempts": 0,
        "visual_repair_attempts": 0,
        "runtime_repair_attempts": 0,
    }

    first.applied_concept_repair_names.append("lossy_numeric_coercion_guard")

    assert second.applied_concept_repair_names == []


def test_generation_mode_preserves_existing_priority() -> None:
    from easyicu.research_agent.execution.step_worker_state import StepWorkerProgress

    assert StepWorkerProgress().generation_mode() == "llm"
    assert (
        StepWorkerProgress(resumed_code_reuse_used=True).generation_mode()
        == "resumed_code_reuse"
    )
    assert StepWorkerProgress(concept_repair_attempts=1).generation_mode() == "repaired"
    assert (
        StepWorkerProgress(deterministic_concept_repairs=1).generation_mode()
        == "repaired"
    )
    assert (
        StepWorkerProgress(runner_repair_name="table_one").generation_mode()
        == "runner_repaired"
    )
    assert (
        StepWorkerProgress(
            resumed_code_reuse_used=True,
            repair_attempts=1,
        ).generation_mode()
        == "repaired"
    )
    assert (
        StepWorkerProgress(
            resumed_code_reuse_used=True,
            repair_attempts=1,
            runner_repair_name="table_one",
        ).generation_mode()
        == "runner_repaired"
    )
    assert (
        StepWorkerProgress(
            runner_repair_name="table_one",
            deterministic_fallback_used=True,
        ).generation_mode()
        == "fallback"
    )
    assert (
        StepWorkerProgress(
            deterministic_fallback_used=True,
            llm_repair_used=True,
        ).generation_mode()
        == "repaired"
    )
    assert (
        StepWorkerProgress(
            llm_repair_used=True,
            deterministic_standard_executor_used=True,
        ).generation_mode()
        == "deterministic_standard"
    )


def test_generation_mode_can_describe_non_llm_terminal_branch() -> None:
    from easyicu.research_agent.execution.step_worker_state import StepWorkerProgress

    progress = StepWorkerProgress(
        llm_repair_used=True,
        deterministic_fallback_used=True,
    )

    assert progress.generation_mode(llm_repair_used=False) == "fallback"
    assert progress.llm_repair_used is True


def test_generation_mode_exhaustively_matches_extracted_legacy_projection() -> None:
    from easyicu.research_agent.execution.step_worker_state import StepWorkerProgress

    for (
        repaired,
        fallback,
        standard,
        runner_repaired,
        resumed,
        concept_repaired,
        llm_repaired,
    ) in product((False, True), repeat=7):
        runner_name = "table_one" if runner_repaired else None
        progress = StepWorkerProgress(
            repair_attempts=int(repaired),
            deterministic_fallback_used=fallback,
            deterministic_standard_executor_used=standard,
            runner_repair_name=runner_name,
            resumed_code_reuse_used=resumed,
            concept_repair_attempts=int(concept_repaired),
            llm_repair_used=llm_repaired,
        )

        assert progress.generation_mode() == _legacy_generation_mode(
            repair_attempts=int(repaired),
            fallback_used=fallback,
            standard_executor_used=standard,
            runner_repair_name=runner_name,
            resumed_code_reuse=resumed,
            concept_repair_used=concept_repaired,
            llm_repair_used=llm_repaired,
        )


def test_worker_progress_is_data_only_and_pipeline_uses_single_seam() -> None:
    from easyicu.research_agent import pipeline_execute
    from easyicu.research_agent.execution import step_worker_state

    module_source = inspect.getsource(step_worker_state)
    for forbidden in (
        "AnalysisStep",
        "ResearchContext",
        "validators",
        "EvidenceStore",
        "RepairCoordinator",
        "StepAuthorityCapsule",
        "pipeline_execute",
        "open(",
    ):
        assert forbidden not in module_source

    execute_source = inspect.getsource(pipeline_execute.run_execute_phase)
    assert execute_source.count("StepWorkerProgress()") == 1
    assert "def _script_generation_mode(" not in execute_source
    assert execute_source.count("worker_progress.generation_mode(") == 4

    tree = ast.parse(textwrap.dedent(execute_source))
    execute_one = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_one_step"
    )
    constructors = [
        node
        for node in ast.walk(execute_one)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "StepWorkerProgress"
    ]
    assert len(constructors) == 1

    # The two assignments intentionally preserve the old phase boundaries:
    # Critic repair is projected into the later LLM lineage flag, and a
    # pre-execution runner repair becomes the active runner repair only after
    # the concept loop.
    attribute_bridges = {
        (node.targets[0].attr, node.value.attr)
        for node in ast.walk(execute_one)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "worker_progress"
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "worker_progress"
    }
    assert ("llm_repair_used", "critic_resume_repair_used") in attribute_bridges
    assert (
        "runner_repair_name",
        "preexecution_runner_repair_name",
    ) in attribute_bridges
