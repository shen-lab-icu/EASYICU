"""Architecture contract for typed-input binding authority."""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from easyicu.research_agent import pipeline_execute
from easyicu.research_agent.authority import typed_binding


def _top_level_function_calls(tree: ast.Module) -> dict[str, set[str]]:
    calls: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls[node.name] = {
            call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, (ast.Attribute, ast.Name))
        }
    return calls


def test_pipeline_execute_reexports_typed_binding_objects_with_identity() -> None:
    assert len(typed_binding.__all__) == 21
    for name in typed_binding.__all__:
        assert getattr(pipeline_execute, name) is getattr(typed_binding, name)


def test_typed_binding_has_no_orchestration_or_scientific_owner_dependency() -> None:
    tree = ast.parse(inspect.getsource(typed_binding))
    imported_leaves = {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert imported_leaves.isdisjoint(
        {"pipeline", "pipeline_execute", "gates", "execution"}
    )
    assert identifiers.isdisjoint(
        {
            "EvidenceStore",
            "LLMConceptAuditor",
            "StepProviderCallBudget",
            "complete_with_provider_budget",
            "consume",
            "promote",
            "register",
            "repair",
            "write_run_checkpoint",
        }
    )


def test_typed_binding_writes_only_its_two_caller_scoped_receipts() -> None:
    calls = _top_level_function_calls(ast.parse(inspect.getsource(typed_binding)))
    writers = {
        name
        for name, function_calls in calls.items()
        if function_calls & {"mkdir", "replace", "write_text", "write_bytes"}
    }
    assert writers == {
        "_write_host_input_binding_receipts",
        "_write_resolved_inputs_manifest",
    }


@pytest.mark.parametrize("canonical_first", [True, False])
def test_typed_binding_identity_survives_import_order(canonical_first: bool) -> None:
    canonical = "easyicu.research_agent.authority.typed_binding"
    legacy = "easyicu.research_agent.pipeline_execute"
    first, second = (canonical, legacy) if canonical_first else (legacy, canonical)
    script = f"""
import importlib
importlib.import_module({first!r})
importlib.import_module({second!r})
canonical = importlib.import_module({canonical!r})
legacy = importlib.import_module({legacy!r})
for name in canonical.__all__:
    assert getattr(legacy, name) is getattr(canonical, name), name
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
