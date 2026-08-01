"""Six plan-time gates and their replan were stapled to the probe step.

They validate the PLAN. They sat inside::

    if pipeline._enable_probe_step and probe_step_id not in resumed_step_ids:

so a run whose probe was already satisfied skipped every one of them, and with
them the ``force``d replan that answers them.

canary36 is the recorded case. Its audit log reads "Skipped step already
completed by pre-execution: 00_probe", the run has no
``analysis_plan_revision_*.json`` at all, and its sealed plan declared a
robustness step promising six outputs against a spec naming two products. The
plan-time gate names that gap exactly -- asked directly, it returns
``robustness_replay_spec.products['complete_case_n']``,
``['missingness_strategy_notes']`` and ``['primary_or']`` -- and the step still
reached execution unrepaired and died, taking its figure with it. Six other
steps in that same run were claimed by their deterministic owners, so the plan
was otherwise the best one recorded.

The previous run, canary35, replanned three times: its probe ran normally.
Whether the gates run at all depended on a step that has nothing to do with
them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PHASE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "easyicu"
    / "research_agent"
    / "execution"
    / "phase.py"
)

#: Every gate that answers "is this PLAN executable", plus the replan that
#: repairs what they find. None of these may depend on the probe.
_PLAN_GATES = (
    "_typed_plan_dag_findings",
    "primary_analysis_cohort_plan_findings",
    "trajectory_plan_dag_findings",
    "declared_raw_input_plan_findings",
    "product_promise_plan_findings",
    "owner_declaration_plan_findings",
    "_maybe_replan",
)


def _source() -> str:
    return _PHASE.read_text(encoding="utf-8")


def _probe_conditional(tree: ast.Module, source: str) -> ast.If:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.get_source_segment(source, node.test) or ""
        if "_enable_probe_step" in test and "resumed_step_ids" in test:
            return node
    raise AssertionError("the probe conditional is no longer recognisable")


def _called_names(node: ast.AST) -> set[str]:
    return {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


@pytest.mark.parametrize("gate", _PLAN_GATES)
def test_no_plan_gate_is_reached_only_when_the_probe_runs(gate: str) -> None:
    source = _source()
    tree = ast.parse(source)

    assert gate not in _called_names(_probe_conditional(tree, source)), (
        f"{gate} is inside the probe branch again; a pre-executed probe would "
        "skip it, which is how canary36 lost a repairable declaration gap"
    )


@pytest.mark.parametrize("gate", _PLAN_GATES)
def test_every_plan_gate_is_still_called_somewhere(gate: str) -> None:
    """Moving a gate out must not mean deleting it."""

    assert gate in _called_names(ast.parse(_source()))


def test_the_replan_still_receives_the_probe_payload_when_there_is_one() -> None:
    """Hoisting must not silently stop feeding the replanner the probe.

    The probe summary is real evidence for a replan when it exists; the point
    of the change is that its ABSENCE no longer cancels the gates.
    """

    source = _source()

    assert "probe_summary_payload=probe_summary" in source
    assert (
        "completed_records=[probe_record] if probe_record is not None else None"
        in source
    )


def test_both_probe_names_are_defined_before_the_branch() -> None:
    """Without this the hoisted call raises NameError on the skip path.

    A NameError here would be worse than the bug: the run would die before its
    first step instead of losing a gate.
    """

    source = _source()
    tree = ast.parse(source)
    probe = _probe_conditional(tree, source)

    for name in ("probe_summary", "probe_record"):
        declared = source.index(f"    {name}: Optional[Dict[str, Any]] = None")
        assert declared < probe.col_offset + source.index(
            "if pipeline._enable_probe_step"
        ), f"{name} must be initialised before the probe branch"


def test_the_forced_replan_still_lists_every_gate() -> None:
    """The ``force`` flag is what makes a found gap cost a replan.

    A gate whose findings are collected but left out of ``force`` reports the
    gap and repairs nothing.
    """

    source = _source()
    start = source.index("force=bool(")
    force_expr = source[start : source.index("),", start)]

    for preflight in (
        "typed_plan_preflight",
        "primary_cohort_preflight",
        "trajectory_preflight",
        "declared_input_preflight",
        "product_promise_preflight",
        "owner_declaration_preflight",
    ):
        assert preflight in force_expr, preflight
