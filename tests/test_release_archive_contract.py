from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS
from tests.research_agent.test_retired_top_level_imports import (
    RETIRED_TOP_LEVEL_MODULES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_installed_easyicu_source_never_imports_repository_benchmarks() -> None:
    violations: list[str] = []
    source_root = REPO_ROOT / "src" / "easyicu"
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported_roots: tuple[str, ...]
            if isinstance(node, ast.Import):
                imported_roots = tuple(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots = (node.module.split(".", 1)[0],)
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "__import__"
                    or isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                )
            ):
                imported_roots = (node.args[0].value.split(".", 1)[0],)
            else:
                continue
            if "benchmarks" in imported_roots:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not violations, (
        "installed easyicu source must not import repository-only benchmark code: "
        f"{violations}"
    )


def test_installed_easyicu_source_contains_no_canonical9_case_material() -> None:
    source_root = REPO_ROOT / "src" / "easyicu"
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(source_root.rglob("*.py"))
    )
    assert "easyicu_evaluation_protocol_suite" not in combined
    assert all(task_id not in combined for task_id in FIGURE2_TASK_IDS)


def _strip_sdist_root(name: str) -> str:
    parts = Path(name).parts
    if len(parts) <= 1:
        return name
    return str(Path(*parts[1:]))


def _contains_debris(path: str) -> bool:
    parts = Path(path).parts
    return (
        "__pycache__" in parts
        or "__MACOSX" in parts
        or path.endswith(".pyc")
        or path.endswith(".pyo")
        or path.endswith(".DS_Store")
        or any(part.startswith("._") for part in parts)
    )


def test_release_archives_preserve_reviewer_contract_and_package_data(
    tmp_path: Path,
) -> None:
    pytest.importorskip("build.__main__")

    sdist_out_dir = tmp_path / "sdist-dist"
    wheel_out_dir = tmp_path / "wheel-dist"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        sdist_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--outdir",
                str(sdist_out_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert sdist_result.returncode == 0, sdist_result.stdout + sdist_result.stderr

        sdist_path = next(sdist_out_dir.glob("easyicu-*.tar.gz"))
        unpacked_dir = tmp_path / "unpacked-sdist"

        with tarfile.open(sdist_path, "r:gz") as archive:
            sdist_names = {
                _strip_sdist_root(member.name) for member in archive.getmembers()
            }
            archive.extractall(unpacked_dir, filter="data")

        unpacked_roots = [path for path in unpacked_dir.iterdir() if path.is_dir()]
        assert len(unpacked_roots) == 1, (
            "sdist must contain exactly one project root before the wheel build: "
            f"{unpacked_roots}"
        )
        unpacked_root = unpacked_roots[0]
        assert (unpacked_root / "pyproject.toml").is_file()

        wheel_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(wheel_out_dir),
            ],
            cwd=unpacked_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert wheel_result.returncode == 0, wheel_result.stdout + wheel_result.stderr

        wheel_path = next(wheel_out_dir.glob("easyicu-*.whl"))

        with zipfile.ZipFile(wheel_path) as archive:
            wheel_names = set(archive.namelist())

        required_sdist_files = {
            ".github/workflows/ci.yml",
            ".github/workflows/research_agent_ci.yml",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "LICENSE",
            "MANIFEST.in",
            "README.md",
            "README_zh.md",
            "benchmarks/__init__.py",
            "benchmarks/figure2_canonical9/canonical_run_input_bindings_v2.json",
            "benchmarks/figure2_canonical9/evaluator/paper_rubric_v3.py",
            "benchmarks/figure2_canonical9/evaluator/scoring.py",
            "benchmarks/figure2_canonical9/figure2_paper_rubric_v2.json",
            "benchmarks/figure2_canonical9/figure2_paper_rubric_v3.json",
            "benchmarks/meta_generalization/meta_benchmark.jsonl",
            "docs/images/01_mode_selection.jpg",
            "docs/images/06_cross_db_benchmark.jpg",
            "examples/research_agent_mortality_sofa.py",
            "pytest.ini",
            "src/easyicu/data/concept-dict.json",
            "src/easyicu/data/data-sources.json",
            "src/easyicu/research_agent/README.md",
            "src/easyicu/research_agent/acquisition/__init__.py",
            "src/easyicu/research_agent/acquisition/catalog.py",
            "src/easyicu/research_agent/acquisition/foundation.py",
            "src/easyicu/research_agent/agents/agentic_coder.py",
            "src/easyicu/research_agent/agents/core.py",
            "src/easyicu/research_agent/authority/filesystem.py",
            "src/easyicu/research_agent/authority/evidence_snapshot.py",
            "src/easyicu/research_agent/authority/evidence_store.py",
            "src/easyicu/research_agent/authority/lock_contract.py",
            "src/easyicu/research_agent/authority/pipeline_cache.py",
            "src/easyicu/research_agent/authority/provenance.py",
            "src/easyicu/research_agent/authority/run_input.py",
            "src/easyicu/research_agent/authority/run_lock.py",
            "src/easyicu/research_agent/authority/runtime_artifacts.py",
            "src/easyicu/research_agent/authority/step_attempt.py",
            "src/easyicu/research_agent/authority/step_capsule.py",
            "src/easyicu/research_agent/authority/step_runtime.py",
            "src/easyicu/research_agent/cohort/__init__.py",
            "src/easyicu/research_agent/cohort/artifact_facts.py",
            "src/easyicu/research_agent/cohort/materializer.py",
            "src/easyicu/research_agent/cohort/primitives.py",
            "src/easyicu/research_agent/cohort/repair.py",
            "src/easyicu/research_agent/cohort/schema.py",
            "src/easyicu/research_agent/contracts/__init__.py",
            "src/easyicu/research_agent/contracts/runtime.py",
            "src/easyicu/research_agent/contracts/declared_product.py",
            "src/easyicu/research_agent/contracts/method_packages.py",
            "src/easyicu/research_agent/contracts/ordered_stratified.py",
            "src/easyicu/research_agent/contracts/robustness_execution.py",
            "src/easyicu/research_agent/reporting/readiness.py",
            "src/easyicu/research_agent/reporting/side_findings.py",
            "src/easyicu/research_agent/reporting/write_phase.py",
            "src/easyicu/research_agent/reporting/writer_evidence.py",
            "src/easyicu/research_agent/orchestration/finalize.py",
            "src/easyicu/research_agent/replication/report.py",
            "src/easyicu/research_agent/execution/code_hygiene.py",
            "src/easyicu/research_agent/execution/concept_audit_cache.py",
            "src/easyicu/research_agent/execution/method_capabilities.py",
            "src/easyicu/research_agent/execution/run_coordination.py",
            "src/easyicu/research_agent/execution/runner.py",
            "src/easyicu/research_agent/execution/step_execution.py",
            "src/easyicu/research_agent/execution/step_worker_state.py",
            "src/easyicu/research_agent/figures/contracts.py",
            "src/easyicu/research_agent/figures/publication.py",
            "src/easyicu/research_agent/figures/skill.py",
            "src/easyicu/research_agent/gates/visual_qa.py",
            "src/easyicu/research_agent/gates/semantics.py",
            "src/easyicu/research_agent/orchestration/__init__.py",
            "src/easyicu/research_agent/orchestration/config.py",
            "src/easyicu/research_agent/orchestration/experiment_spec.py",
            "src/easyicu/research_agent/orchestration/profiles.py",
            "src/easyicu/research_agent/orchestration/resume.py",
            "src/easyicu/research_agent/planning/cohort_contract.py",
            "src/easyicu/research_agent/planning/analysis_blueprint.py",
            "src/easyicu/research_agent/planning/robustness_contract.py",
            "src/easyicu/research_agent/providers/cost.py",
            "src/easyicu/research_agent/trajectory/bundle.py",
            "src/easyicu/research_agent/trajectory/contract.py",
            "src/easyicu/research_agent/trajectory/plan_contract.py",
            "src/easyicu/research_agent/providers/llm.py",
            "src/easyicu/research_agent/providers/mocks.py",
            "src/easyicu/research_agent/providers/prompts/v1/coder.txt",
            "src/easyicu/research_agent/providers/protocol.py",
            "src/easyicu/research_agent/providers/structured_retry.py",
            "src/easyicu/research_agent/research_context/implementation_identity.py",
            "src/easyicu/research_agent/authority/context_numeric_claims.py",
            "src/easyicu/research_agent/research_context/temporal_semantics.py",
            "src/easyicu/research_agent/replication/metrics.py",
            "src/easyicu/webserver/static/index.html",
            "src/easyicu/webserver/static/css/app.css",
            "src/easyicu/webserver/static/js/app.js",
            "tests/test_release_archive_contract.py",
            "tests/test_release_hardening_p0.py",
            "tests/test_repository_contract.py",
        }
        missing_sdist = sorted(required_sdist_files - sdist_names)
        assert (
            not missing_sdist
        ), f"sdist is missing reviewer/release files: {missing_sdist}"
        benchmark_sources = {
            str(path.relative_to(REPO_ROOT))
            for path in (REPO_ROOT / "benchmarks").rglob("*")
            if path.is_file() and path.suffix in {".py", ".json", ".jsonl", ".md"}
        }
        missing_benchmark_sources = sorted(benchmark_sources - sdist_names)
        assert not missing_benchmark_sources, (
            "sdist is missing repository benchmark authority files: "
            f"{missing_benchmark_sources}"
        )

        assert "easyicu/__init__.py" in wheel_names
        assert "easyicu/data/concept-dict.json" in wheel_names
        assert "easyicu/data/data-sources.json" in wheel_names
        assert "easyicu/webserver/static/index.html" in wheel_names
        assert "easyicu/webserver/static/css/app.css" in wheel_names
        assert "easyicu/webserver/static/js/app.js" in wheel_names
        assert any(name.endswith(".dist-info/entry_points.txt") for name in wheel_names)

        required_canonical_modules = {
            "easyicu/research_agent/acquisition/__init__.py",
            "easyicu/research_agent/acquisition/catalog.py",
            "easyicu/research_agent/acquisition/foundation.py",
            "easyicu/research_agent/agents/__init__.py",
            "easyicu/research_agent/agents/agentic_coder.py",
            "easyicu/research_agent/agents/core.py",
            "easyicu/research_agent/authority/filesystem.py",
            "easyicu/research_agent/authority/evidence_snapshot.py",
            "easyicu/research_agent/authority/evidence_store.py",
            "easyicu/research_agent/authority/lock_contract.py",
            "easyicu/research_agent/authority/pipeline_cache.py",
            "easyicu/research_agent/authority/provenance.py",
            "easyicu/research_agent/authority/run_input.py",
            "easyicu/research_agent/authority/run_lock.py",
            "easyicu/research_agent/authority/runtime_artifacts.py",
            "easyicu/research_agent/authority/step_attempt.py",
            "easyicu/research_agent/authority/step_capsule.py",
            "easyicu/research_agent/authority/step_runtime.py",
            "easyicu/research_agent/cohort/__init__.py",
            "easyicu/research_agent/cohort/artifact_facts.py",
            "easyicu/research_agent/cohort/materializer.py",
            "easyicu/research_agent/cohort/primitives.py",
            "easyicu/research_agent/cohort/repair.py",
            "easyicu/research_agent/cohort/schema.py",
            "easyicu/research_agent/contracts/__init__.py",
            "easyicu/research_agent/contracts/runtime.py",
            "easyicu/research_agent/contracts/declared_product.py",
            "easyicu/research_agent/contracts/method_packages.py",
            "easyicu/research_agent/contracts/ordered_stratified.py",
            "easyicu/research_agent/contracts/robustness_execution.py",
            "easyicu/research_agent/reporting/readiness.py",
            "easyicu/research_agent/reporting/side_findings.py",
            "easyicu/research_agent/reporting/write_phase.py",
            "easyicu/research_agent/reporting/writer_evidence.py",
            "easyicu/research_agent/orchestration/finalize.py",
            "easyicu/research_agent/replication/report.py",
            "easyicu/research_agent/execution/code_hygiene.py",
            "easyicu/research_agent/execution/concept_audit_cache.py",
            "easyicu/research_agent/execution/method_capabilities.py",
            "easyicu/research_agent/execution/run_coordination.py",
            "easyicu/research_agent/execution/runner.py",
            "easyicu/research_agent/execution/step_execution.py",
            "easyicu/research_agent/execution/step_worker_state.py",
            "easyicu/research_agent/figures/contracts.py",
            "easyicu/research_agent/figures/publication.py",
            "easyicu/research_agent/figures/skill.py",
            "easyicu/research_agent/gates/visual_qa.py",
            "easyicu/research_agent/gates/semantics.py",
            "easyicu/research_agent/orchestration/__init__.py",
            "easyicu/research_agent/orchestration/config.py",
            "easyicu/research_agent/orchestration/experiment_spec.py",
            "easyicu/research_agent/orchestration/profiles.py",
            "easyicu/research_agent/orchestration/resume.py",
            "easyicu/research_agent/planning/cohort_contract.py",
            "easyicu/research_agent/planning/analysis_blueprint.py",
            "easyicu/research_agent/planning/robustness_contract.py",
            "easyicu/research_agent/providers/cost.py",
            "easyicu/research_agent/trajectory/bundle.py",
            "easyicu/research_agent/trajectory/contract.py",
            "easyicu/research_agent/trajectory/plan_contract.py",
            "easyicu/research_agent/providers/llm.py",
            "easyicu/research_agent/providers/mocks.py",
            "easyicu/research_agent/providers/prompts/__init__.py",
            "easyicu/research_agent/providers/prompts/v1/coder.txt",
            "easyicu/research_agent/providers/prompts/v1/replanner.txt",
            "easyicu/research_agent/providers/prompts/v1/system.txt",
            "easyicu/research_agent/providers/prompts/v1/writer.txt",
            "easyicu/research_agent/providers/protocol.py",
            "easyicu/research_agent/providers/structured_retry.py",
            "easyicu/research_agent/research_context/implementation_identity.py",
            "easyicu/research_agent/authority/context_numeric_claims.py",
            "easyicu/research_agent/research_context/temporal_semantics.py",
            "easyicu/research_agent/replication/metrics.py",
        }
        missing_canonical_modules = sorted(required_canonical_modules - wheel_names)
        assert not missing_canonical_modules, (
            "wheel built from the sdist is missing canonical research-agent modules: "
            f"{missing_canonical_modules}"
        )

        assert "src/easyicu/research_agent/projection.py" not in sdist_names
        assert "easyicu/research_agent/projection.py" not in wheel_names
        retired_sdist = {
            f"src/easyicu/research_agent/{leaf}.py"
            for leaf in RETIRED_TOP_LEVEL_MODULES
        }
        retired_wheel = {
            f"easyicu/research_agent/{leaf}.py" for leaf in RETIRED_TOP_LEVEL_MODULES
        }
        assert retired_sdist.isdisjoint(sdist_names)
        assert retired_wheel.isdisjoint(wheel_names)
        assert "src/easyicu/research_agent/agents.py" not in sdist_names
        assert "easyicu/research_agent/agents.py" not in wheel_names
        retired_execution_leaves = {
            "code_hygiene",
            "run_coordination",
            "runner",
            "step_execution",
            "step_worker_state",
        }
        assert {
            f"src/easyicu/research_agent/{leaf}.py" for leaf in retired_execution_leaves
        }.isdisjoint(sdist_names)
        assert {
            f"easyicu/research_agent/{leaf}.py" for leaf in retired_execution_leaves
        }.isdisjoint(wheel_names)

        wheel_extract_dir = tmp_path / "installed-wheel"
        with zipfile.ZipFile(wheel_path) as archive:
            archive.extractall(wheel_extract_dir)
        smoke_env = env.copy()
        smoke_env["PYTHONPATH"] = str(wheel_extract_dir)
        smoke_code = """
import importlib
from pathlib import Path

root = Path(__import__('os').environ['EASYICU_WHEEL_ROOT']).resolve()
for name in (
    'easyicu.research_agent.acquisition.catalog',
    'easyicu.research_agent.acquisition.foundation',
    'easyicu.research_agent.agents',
    'easyicu.research_agent.agents.agentic_coder',
    'easyicu.research_agent.agents.core',
    'easyicu.research_agent.authority.filesystem',
    'easyicu.research_agent.authority.evidence_snapshot',
    'easyicu.research_agent.authority.evidence_store',
    'easyicu.research_agent.authority.lock_contract',
    'easyicu.research_agent.authority.pipeline_cache',
    'easyicu.research_agent.authority.provenance',
    'easyicu.research_agent.authority.run_input',
    'easyicu.research_agent.authority.run_lock',
    'easyicu.research_agent.authority.runtime_artifacts',
    'easyicu.research_agent.authority.step_attempt',
    'easyicu.research_agent.authority.step_capsule',
    'easyicu.research_agent.authority.step_runtime',
    'easyicu.research_agent.cohort.artifact_facts',
    'easyicu.research_agent.cohort.materializer',
    'easyicu.research_agent.cohort.primitives',
    'easyicu.research_agent.cohort.repair',
    'easyicu.research_agent.cohort.schema',
    'easyicu.research_agent.contracts',
    'easyicu.research_agent.contracts.runtime',
    'easyicu.research_agent.contracts.declared_product',
    'easyicu.research_agent.contracts.method_packages',
    'easyicu.research_agent.contracts.ordered_stratified',
    'easyicu.research_agent.contracts.robustness_execution',
    'easyicu.research_agent.reporting.readiness',
    'easyicu.research_agent.reporting.side_findings',
    'easyicu.research_agent.reporting.write_phase',
    'easyicu.research_agent.reporting.writer_evidence',
    'easyicu.research_agent.orchestration.finalize',
    'easyicu.research_agent.replication.report',
    'easyicu.research_agent.execution.code_hygiene',
    'easyicu.research_agent.execution.concept_audit_cache',
    'easyicu.research_agent.execution.method_capabilities',
    'easyicu.research_agent.execution.run_coordination',
    'easyicu.research_agent.execution.runner',
    'easyicu.research_agent.execution.step_execution',
    'easyicu.research_agent.execution.step_worker_state',
    'easyicu.research_agent.figures.contracts',
    'easyicu.research_agent.figures.publication',
    'easyicu.research_agent.figures.skill',
    'easyicu.research_agent.gates.visual_qa',
    'easyicu.research_agent.gates.semantics',
    'easyicu.research_agent.orchestration.config',
    'easyicu.research_agent.orchestration.experiment_spec',
    'easyicu.research_agent.orchestration.profiles',
    'easyicu.research_agent.orchestration.resume',
    'easyicu.research_agent.planning.analysis_blueprint',
    'easyicu.research_agent.providers.cost',
    'easyicu.research_agent.trajectory.bundle',
    'easyicu.research_agent.trajectory.contract',
    'easyicu.research_agent.trajectory.plan_contract',
    'easyicu.research_agent.providers.llm',
    'easyicu.research_agent.providers.mocks',
    'easyicu.research_agent.providers.prompts',
    'easyicu.research_agent.providers.structured_retry',
    'easyicu.research_agent.research_context.implementation_identity',
    'easyicu.research_agent.authority.context_numeric_claims',
    'easyicu.research_agent.research_context.temporal_semantics',
):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__

agent = importlib.import_module('easyicu.research_agent')
config = importlib.import_module('easyicu.research_agent.orchestration.config')
profiles = importlib.import_module('easyicu.research_agent.orchestration.profiles')
spec = importlib.import_module('easyicu.research_agent.orchestration.experiment_spec')
assert agent.PipelineConfig is config.PipelineConfig
assert agent.SubmissionProfile is profiles.SubmissionProfile
assert agent.ExperimentSpec is spec.ExperimentSpec

from easyicu.research_agent.providers.prompts import load_prompt_pack
assert set(load_prompt_pack()) == {'system', 'coder', 'replanner', 'writer'}
"""
        smoke_env["EASYICU_WHEEL_ROOT"] = str(wheel_extract_dir)
        smoke_result = subprocess.run(
            [sys.executable, "-c", smoke_code],
            cwd=tmp_path,
            env=smoke_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert smoke_result.returncode == 0, smoke_result.stdout + smoke_result.stderr
        assert not any(
            name.startswith(("tests/", "examples/", "docs/", "benchmarks/"))
            for name in wheel_names
        )
        assert not any(
            name.startswith("easyicu/research_agent/figure2_") for name in wheel_names
        )
        assert "easyicu/research_agent/canonical_input_freeze.py" not in wheel_names
        assert not [name for name in sdist_names if _contains_debris(name)]
        assert not [name for name in wheel_names if _contains_debris(name)]
    finally:
        shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "easyicu.egg-info", ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "src" / "easyicu.egg-info", ignore_errors=True)
