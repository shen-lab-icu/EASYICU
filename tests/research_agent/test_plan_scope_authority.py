"""Architecture contract for the Planner-owned scientific signature kernel."""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from easyicu.research_agent.execution import phase as execution_phase
from easyicu.research_agent.authority import plan_scope
from easyicu.research_agent.schema import AnalysisStep


def test_execution_phase_uses_plan_scope_objects_with_identity() -> None:
    assert plan_scope.__all__
    for name in plan_scope.__all__:
        assert getattr(execution_phase, name) is getattr(plan_scope, name)


def test_plan_scope_has_no_orchestration_or_mutation_dependency() -> None:
    tree = ast.parse(inspect.getsource(plan_scope))
    imported_leaves = {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert imported_leaves.isdisjoint(
        {"pipeline", "pipeline_execute", "gates", "execution", "evidence"}
    )
    assert identifiers.isdisjoint(
        {
            "open",
            "write_text",
            "write_bytes",
            "register",
            "promote",
            "consume",
            "repair",
            "complete",
        }
    )


def test_scientific_signature_uses_typed_role_not_intent_role_words() -> None:
    step = AnalysisStep(
        step_id="01_model",
        intent=(
            "Discuss primary, secondary, sensitivity, and corroborative results "
            "without owning the primary estimand."
        ),
        method="logistic_regression",
        planned_analysis_role="auxiliary",
        expected_outputs=["statistic:adjusted_effect"],
    )

    signature = plan_scope._step_scientific_signature(step)

    assert signature[6] == "auxiliary"
    assert not isinstance(signature[6], tuple)


def test_scientific_signature_changes_only_role_coordinate_for_role_change() -> None:
    primary = AnalysisStep(
        step_id="01_model",
        intent="Fit the prespecified model.",
        method="logistic_regression",
        planned_analysis_role="primary",
        expected_outputs=["statistic:adjusted_effect"],
    )
    secondary = primary.model_copy(update={"planned_analysis_role": "secondary"})

    primary_signature = plan_scope._step_scientific_signature(primary)
    secondary_signature = plan_scope._step_scientific_signature(secondary)

    assert primary_signature != secondary_signature
    assert [
        index
        for index, (left, right) in enumerate(
            zip(primary_signature, secondary_signature, strict=True)
        )
        if left != right
    ] == [6]


@pytest.mark.parametrize("canonical_first", [True, False])
def test_plan_scope_identity_survives_import_order(canonical_first: bool) -> None:
    canonical = "easyicu.research_agent.authority.plan_scope"
    consumer = "easyicu.research_agent.execution.phase"
    first, second = (canonical, consumer) if canonical_first else (consumer, canonical)
    script = f"""
import importlib
first = importlib.import_module({first!r})
second = importlib.import_module({second!r})
canonical = importlib.import_module({canonical!r})
consumer = importlib.import_module({consumer!r})
for name in canonical.__all__:
    assert getattr(consumer, name) is getattr(canonical, name), name
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
