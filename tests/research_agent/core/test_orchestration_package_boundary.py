"""Canonical ownership contracts for run orchestration policy modules."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

ORCHESTRATION_MODULES = (
    "orchestration.config",
    "orchestration.services",
    "orchestration.workflow",
    "orchestration.profiles",
    "orchestration.experiment_spec",
    "orchestration.resume",
    "orchestration.finalize",
)

RETIRED_MODULES = (
    "pipeline_config",
    "pipeline_profiles",
    "experiment_spec",
    "pipeline_resume",
    "pipeline_phases",
    "pipeline_state",
    "pipeline_package",
)

PROFILE_EXPORTS = (
    "SubmissionProfile",
    "NPJ_DM_2026_05",
    "NPJ_DM_2026_06",
    "NPJ_DM_2026_07",
    "NPJ_DM_2026_07_16",
    "NPJ_DM_2026_07_17",
    "NPJ_DM_2026_07_18",
    "NPJ_DM_2026_07_19",
    "DEFAULT_SUBMISSION_PROFILE_REF",
    "SUBMISSION_PROFILE_REGISTRY",
    "get_submission_profile",
)

EXPERIMENT_SPEC_EXPORTS = (
    "ExperimentSpec",
    "CohortInputSpec",
    "RuntimeSpec",
    "load_experiment_spec",
    "dump_experiment_spec",
    "build_pipeline_from_spec",
)


@pytest.mark.parametrize("target", ORCHESTRATION_MODULES)
def test_orchestration_module_has_one_canonical_home(target: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.{target}")
    assert module.__name__ == f"easyicu.research_agent.{target}"


def test_orchestration_package_is_lazy() -> None:
    script = """
import importlib
import sys
package = 'easyicu.research_agent.orchestration'
importlib.import_module(package)
loaded = sorted(name for name in sys.modules if name.startswith(package + '.'))
assert loaded == [], loaded
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


@pytest.mark.parametrize("leaf", RETIRED_MODULES)
def test_retired_orchestration_module_is_absent(leaf: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"easyicu.research_agent.{leaf}")


def test_root_orchestration_api_resolves_to_canonical_modules() -> None:
    root = importlib.import_module("easyicu.research_agent")
    config = importlib.import_module("easyicu.research_agent.orchestration.config")
    services = importlib.import_module("easyicu.research_agent.orchestration.services")
    profiles = importlib.import_module("easyicu.research_agent.orchestration.profiles")
    experiment_spec = importlib.import_module(
        "easyicu.research_agent.orchestration.experiment_spec"
    )

    assert root.PipelineConfig is config.PipelineConfig
    assert root.PipelineServices is services.PipelineServices
    for name in PROFILE_EXPORTS:
        assert getattr(root, name) is getattr(profiles, name)
    for name in EXPERIMENT_SPEC_EXPORTS:
        assert getattr(root, name) is getattr(experiment_spec, name)


def test_pipeline_finalization_helpers_use_canonical_objects() -> None:
    pipeline = importlib.import_module("easyicu.research_agent.pipeline")
    finalize = importlib.import_module("easyicu.research_agent.orchestration.finalize")
    assert (
        pipeline._concept_dictionary_manifest_fields
        is finalize._concept_dictionary_manifest_fields
    )
    assert pipeline._render_cost_summary is finalize._render_cost_summary


@pytest.mark.parametrize("target", ORCHESTRATION_MODULES)
def test_orchestration_policy_does_not_reverse_import_pipeline(target: str) -> None:
    module = importlib.import_module(f"easyicu.research_agent.{target}")
    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(
                importlib.util.resolve_name(
                    "." * node.level + node.module,
                    module.__package__,
                )
                if node.level
                else node.module
            )
    forbidden = {
        "easyicu.research_agent.pipeline",
        "easyicu.research_agent.execution.phase",
    }
    assert imported.isdisjoint(forbidden)
