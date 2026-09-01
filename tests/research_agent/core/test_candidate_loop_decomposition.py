"""Architecture contract for the candidate execution/repair state machine."""

from __future__ import annotations

import ast
import inspect
import textwrap

from easyicu.research_agent.execution import phase


def _function_node(function) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef))


def test_candidate_loop_stages_remain_bounded() -> None:
    """Keep the former 2,424-line mixed loop split by responsibility."""

    stages = (
        phase._candidate_concept_audit_transition,
        phase._candidate_execute_transition,
        phase._candidate_success_prepare_transition,
        phase._candidate_visual_transition,
        phase._candidate_contract_setup_transition,
        phase._candidate_contract_repair_transition,
        phase._candidate_summary_transition,
        phase._candidate_failure_transition,
    )
    stage_lines = {
        stage.__name__: len(inspect.getsourcelines(stage)[0]) for stage in stages
    }

    assert max(stage_lines.values()) <= 550, stage_lines
    assert len(inspect.getsourcelines(phase._run_candidate_loop)[0]) <= 150


def test_execute_worker_does_not_reabsorb_candidate_state_machine() -> None:
    source = inspect.getsource(phase._execute_step)
    tree = ast.parse(textwrap.dedent(source))
    execute_one = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_step"
    )

    assert execute_one.end_lineno - execute_one.lineno + 1 <= 3000
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_candidate_loop"
            for node in ast.walk(execute_one)
        )
        == 1
    )


def test_candidate_loop_dispatches_each_stage_once_in_scientific_order() -> None:
    node = _function_node(phase._run_candidate_loop)
    stage_names = {
        "_candidate_concept_audit_transition",
        "_candidate_execute_transition",
        "_candidate_success_prepare_transition",
        "_candidate_visual_transition",
        "_candidate_contract_setup_transition",
        "_candidate_contract_repair_transition",
        "_candidate_summary_transition",
        "_candidate_failure_transition",
    }
    stage_references = sorted(
        (
            reference.lineno,
            reference.id,
        )
        for reference in ast.walk(node)
        if isinstance(reference, ast.Name) and reference.id in stage_names
    )

    assert [name for _, name in stage_references] == [
        "_candidate_concept_audit_transition",
        "_candidate_execute_transition",
        "_candidate_success_prepare_transition",
        "_candidate_visual_transition",
        "_candidate_contract_setup_transition",
        "_candidate_contract_repair_transition",
        "_candidate_summary_transition",
        "_candidate_failure_transition",
    ]
