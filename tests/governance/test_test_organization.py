"""Test-suite ownership conventions."""

from __future__ import annotations

import ast
import re
from pathlib import Path


DATE_NAMED_TEST = re.compile(r"(?:^|_)20\d{6}(?:_|\.py$)")
OWNER_DIRECTORIES = {
    "benchmarks",
    "core",
    "governance",
    "research_agent",
    "webserver",
}
LARGE_TEST_MODULE_BASELINES = {
    "benchmarks/figure2_canonical9/test_realrun_authority.py": 2628,
    "research_agent/core/test_idea_mining.py": 3347,
    "research_agent/core/test_materialized_column_metadata.py": 2504,
    "research_agent/execution/test_code_repair.py": 3052,
    "research_agent/execution/test_coder_context_repair_preflight.py": 6826,
    "research_agent/execution/test_docker_runner.py": 2150,
    "research_agent/execution/test_step_result_envelope.py": 2062,
    "research_agent/figures/test_association_figure_rescue.py": 2697,
    "research_agent/figures/test_publication_figures.py": 3266,
    "research_agent/figures/test_validators_figure_source_trace.py": 4417,
    "research_agent/gates/test_declared_product_contract.py": 3206,
    "research_agent/gates/test_execution_phase_contract.py": 2640,
    "research_agent/gates/test_primary_cohort_product_integrity.py": 2046,
    "research_agent/gates/test_validators.py": 5182,
    "research_agent/integration/test_pipeline.py": 11471,
    "research_agent/integration/test_pipeline_typed_artifact_lineage.py": 2408,
    "research_agent/integration/test_resume.py": 4593,
    "research_agent/planning/test_coder_prompt_budget.py": 2498,
    "research_agent/planning/test_plan_scientific_review.py": 2712,
    "research_agent/planning/test_progressive_planner_v2.py": 7393,
    "research_agent/providers/test_primary_model_contract.py": 2120,
    "research_agent/providers/test_provider_budget.py": 2234,
    "research_agent/providers/test_pubmed.py": 2306,
    "webserver/copilot/test_pi_copilot_contract.py": 6935,
    "webserver/copilot/test_pi_copilot_research_workflow.py": 9830,
    "webserver/copilot/test_pi_copilot_static.py": 6788,
    "webserver/test_webserver_static_routes.py": 3934,
    "webserver/test_webserver_workspace_summary.py": 7655,
}


def test_regression_files_use_functional_owner_names() -> None:
    """Review dates belong in docstrings, not in test module ownership."""

    tests_root = Path(__file__).resolve().parents[1]
    offenders = [
        path.relative_to(tests_root).as_posix()
        for path in tests_root.rglob("test_*.py")
        if DATE_NAMED_TEST.search(path.name)
    ]

    assert offenders == []


def test_private_contract_tests_import_the_owner_not_pipeline() -> None:
    """Moved contracts must not be re-exported by the pipeline catch-all."""

    tests_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in tests_root.rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "easyicu.research_agent.pipeline":
                continue
            if any(
                alias.name
                in {"_enforce_advanced_plan_contract", "_step_contract_findings"}
                for alias in node.names
            ):
                offenders.append(path.relative_to(tests_root).as_posix())

    assert sorted(set(offenders)) == []


def test_python_tests_live_under_an_owner_directory() -> None:
    tests_root = Path(__file__).resolve().parents[1]
    root_tests = sorted(path.name for path in tests_root.glob("test_*.py"))
    assert root_tests == []

    unexpected = sorted(
        path.relative_to(tests_root).as_posix()
        for path in tests_root.rglob("test_*.py")
        if path.relative_to(tests_root).parts[0] not in OWNER_DIRECTORIES
    )
    assert unexpected == []


def test_large_test_modules_only_shrink() -> None:
    tests_root = Path(__file__).resolve().parents[1]
    current: dict[str, int] = {}
    for path in tests_root.rglob("test_*.py"):
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count >= 2_000:
            current[path.relative_to(tests_root).as_posix()] = line_count

    assert set(current) <= set(LARGE_TEST_MODULE_BASELINES)
    regressions = {
        path: (line_count, LARGE_TEST_MODULE_BASELINES[path])
        for path, line_count in current.items()
        if line_count > LARGE_TEST_MODULE_BASELINES[path]
    }
    assert regressions == {}


def test_slow_test_manifest_only_names_existing_modules() -> None:
    tests_root = Path(__file__).resolve().parents[1]
    missing: list[str] = []
    for raw_line in (tests_root / "slow_tests.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        module_path = line.split("::", maxsplit=1)[0]
        if not (tests_root.parent / module_path).is_file():
            missing.append(module_path)

    assert sorted(set(missing)) == []
