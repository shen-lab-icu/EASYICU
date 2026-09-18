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
    "research_agent/figures/test_validators_figure_source_trace.py": 4454,
    "research_agent/gates/test_declared_product_contract.py": 3206,
    "research_agent/gates/test_execution_phase_contract.py": 2640,
    "research_agent/gates/test_primary_cohort_product_integrity.py": 2046,
    "research_agent/gates/test_validators.py": 5183,
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
    "webserver/copilot/test_pi_copilot_static.py": 6952,
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


# Split-bypass ratchet (E-P2-2): per-file caps can be dodged by splitting one
# large module into two files each under 2000 lines.  Freeze the split total
# for the known workflow split so the sum cannot grow while each shard stays
# green.  Current tree (2026-09-17): research_workflow 9284 (9252 + E-P2-13
# un-stubbed probe section) + pipeline_route_authority 868 = 10152 (old
# single-file baseline was 9830; the split grew the total, now frozen here).
WORKFLOW_SPLIT_FILES = (
    "webserver/copilot/test_pi_copilot_research_workflow.py",
    "webserver/copilot/test_pi_copilot_pipeline_route_authority.py",
)
WORKFLOW_SPLIT_TOTAL_CAP = 10152

# Grandfathered cross-test importers (E-P2-12): new code must not
# ``from test_* import`` another test module (use conftest/support fixtures
# instead).  The files below predate the rule and are frozen; any importer
# NOT in this list fails the gate.  This pins the 2026-09-17 stock so the
# coupling cannot grow, without forcing a 50-file refactor in one patch.
# Convention going forward (see tests/webserver/copilot/conftest.py):
# new shared fixtures go to conftest.py / tests/support, never
# ``from test_* import``.
GRANDFATHERED_TEST_IMPORTERS = frozenset(
    {
        "governance/test_release_archive_contract.py",
        "research_agent/authority/test_functional_form_effect_products.py",
        "research_agent/authority/test_functional_form_target.py",
        "research_agent/authority/test_plausibility_receipt_gate.py",
        "research_agent/authority/test_step_authority_capsule_integration.py",
        "research_agent/core/test_a_failed_attempt_releases_what_it_never_used.py",
        "research_agent/core/test_a_render_child_is_recognised_by_what_it_declares.py",
        "research_agent/core/test_bench_comparison.py",
        "research_agent/core/test_endpoint_spec.py",
        "research_agent/core/test_reviewed_memory_wiring.py",
        "research_agent/core/test_locked_cohort_spelling_reaches_the_sample.py",
        "research_agent/core/test_ownership_verdict.py",
        "research_agent/core/test_phenotype_comparison_executor.py",
        "research_agent/core/test_phenotyping_feature_roles.py",
        "research_agent/core/test_prediction_model_fit_evidence.py",
        "research_agent/core/test_primary_effect_hazard_ratio.py",
        "research_agent/core/test_the_host_can_adopt_its_own_typed_cohort.py",
        "research_agent/execution/test_prediction_model_fit_persisted_runtime.py",
        "research_agent/execution/test_coder_resource_wiring.py",
        "research_agent/integration/test_capability_workflow_wiring.py",
        "research_agent/figures/test_cross_sectional_phenotyping_contract.py",
        "research_agent/figures/test_run_input_trajectory_authority.py",
        "research_agent/figures/test_runner_trajectory_contract.py",
        "research_agent/figures/test_the_figure_names_the_contrast_its_producer_wrote.py",
        "research_agent/figures/test_visual_repair_governance.py",
        "research_agent/gates/test_owner_declaration_gate.py",
        "research_agent/gates/test_the_declared_windows_reach_the_sealed_cohort.py",
        "research_agent/planning/test_concept_source_support.py",
        "research_agent/planning/test_ordered_trend_parent_semantics.py",
        "research_agent/planning/test_outline_population_choice.py",
        "research_agent/planning/test_plan_shape_costs_the_owner.py",
        "research_agent/planning/test_progressive_baseline_retrieval.py",
        "research_agent/planning/test_progressive_final_repair.py",
        "research_agent/planning/test_progressive_planner_display_labels.py",
        "research_agent/planning/test_research_context_v2_authority_join.py",
        "research_agent/planning/test_research_context_v2_identity.py",
        "research_agent/planning/test_research_context_v2_prompts.py",
        "research_agent/providers/test_real_llm_parsers.py",
        "research_agent/reporting/test_descriptive_report_facts.py",
        "research_agent/reporting/test_landmark_writer_method_receipt.py",
        "research_agent/reporting/test_manuscript_baseline_coverage.py",
        "research_agent/reporting/test_manuscript_policy_composition.py",
        "research_agent/reporting/test_manuscript_reader_assembly.py",
        "research_agent/reporting/test_plan_driven_result_structure.py",
        "research_agent/reporting/test_post_binding_writer_recovery.py",
        "research_agent/reporting/test_writer_only_migration.py",
        "webserver/copilot/test_pi_copilot_literature_reader.py",
        "webserver/copilot/test_pi_copilot_package_revalidation.py",
        "webserver/copilot/test_pi_copilot_plan_supersession.py",
        "webserver/copilot/test_pi_copilot_research_workflow.py",
        "webserver/test_cohort_review_cache_concurrency.py",
        "webserver/test_crossdb_review_boundaries.py",
        "webserver/test_plan_change_requirements.py",
        "webserver/test_research_pipeline_intake_diagnostics.py",
    }
)


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


def test_workflow_split_total_only_shrinks() -> None:
    """E-P2-2: the workflow split total must not grow past its frozen cap."""

    tests_root = Path(__file__).resolve().parents[1]
    total = 0
    missing: list[str] = []
    for rel in WORKFLOW_SPLIT_FILES:
        path = tests_root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        total += len(path.read_text(encoding="utf-8").splitlines())
    assert missing == [], f"workflow split member missing: {missing}"
    assert total <= WORKFLOW_SPLIT_TOTAL_CAP, (
        f"workflow split total grew: {total} > {WORKFLOW_SPLIT_TOTAL_CAP} "
        f"for {list(WORKFLOW_SPLIT_FILES)}"
    )


def test_no_cross_test_module_imports() -> None:
    """E-P2-12: forbid NEW ``from test_* import`` between test modules.

    Shared fixtures belong in ``conftest.py``/``tests/support``.  Files in
    ``GRANDFATHERED_TEST_IMPORTERS`` predate the rule and stay frozen; any
    cross-test import from a file NOT in that set fails this gate.
    """

    def _is_test_module_import(node: ast.ImportFrom) -> bool:
        module = node.module or ""
        if "support" in module or "conftest" in module:
            return False
        head = module.split(".")[-1] if module else ""
        if head.startswith("test_"):
            return True
        return any(
            alias.name.split(".")[-1].startswith("test_")
            or alias.name.startswith("test_")
            for alias in node.names
        ) and (
            module.startswith("tests.")
            or module.startswith("test_")
            or node.level > 0
        )

    tests_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in tests_root.rglob("test_*.py"):
        rel = path.relative_to(tests_root).as_posix()
        if rel in GRANDFATHERED_TEST_IMPORTERS:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and _is_test_module_import(node):
                offenders.append(f"{rel} -> {node.module} (line {node.lineno})")
    assert offenders == [], (
        "new cross-test import(s); move the shared fixture to "
        f"conftest.py/tests/support instead: {offenders}"
    )


def test_grandfathered_test_imports_stay_frozen() -> None:
    """The E-P2-12 allowlist must name only files that still exist."""

    tests_root = Path(__file__).resolve().parents[1]
    missing = sorted(
        rel
        for rel in GRANDFATHERED_TEST_IMPORTERS
        if not (tests_root / rel).is_file()
    )
    assert missing == [], f"grandfathered importer vanished: {missing}"


def test_old_e3_root_module_stays_gone() -> None:
    """E-P2-3: the pre-rename ``tests/test_e3_strict_kdigo_window.py`` stays gone.

    The rename to ``tests/core/test_e3_strict_kdigo_window.py`` landed as R100
    in 170b58189; this pins the git-mv semantics (old path absent, new path
    present) so a half-rename cannot regress.
    """

    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "tests" / "test_e3_strict_kdigo_window.py").exists(), (
        "stale root test module resurrected; the rename to tests/core/ is final"
    )
    assert (repo_root / "tests" / "core" / "test_e3_strict_kdigo_window.py").is_file()


def test_corpus_and_node_skips_are_counted() -> None:
    """E-P2-4: resource-gated skips use markers and count as uncovered.

    ``requires_corpus``/``requires_node``/``requires_docker`` must be
    registered (pytest.ini) and handled in ``tests/conftest.py`` with an
    explicit skip reason, so ``pytest -rs`` surfaces them and coverage
    tooling treats them as uncovered rather than silent passes.  This gate
    counts current marker usages to keep the skip surface visible; a jump
    in the count without a corpus/node/docker justification must update
    this number deliberately.
    """

    repo_root = Path(__file__).resolve().parents[2]
    conftest = (repo_root / "tests" / "conftest.py").read_text(encoding="utf-8")
    for marker in ("requires_corpus", "requires_node", "requires_docker"):
        assert marker in conftest, f"tests/conftest.py must handle {marker}"
    pytest_ini = (repo_root / "pytest.ini").read_text(encoding="utf-8")
    for marker in ("requires_corpus", "requires_node", "requires_docker"):
        assert marker in pytest_ini, f"pytest.ini must register {marker}"

    uses: dict[str, int] = {"requires_corpus": 0, "requires_node": 0, "requires_docker": 0}
    for path in (repo_root / "tests").rglob("test_*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for marker in uses:
            uses[marker] += text.count(marker)
    total = sum(uses.values())
    # Frozen 2026-09-17 after converting the ~10 representative files; the
    # count grows as more files adopt the markers (good), but a drop to zero
    # means the markers were removed and skips went silent again.
    assert total >= 10, f"resource-gated markers vanished: {uses}"
    assert uses["requires_corpus"] >= 7, f"corpus markers too few: {uses}"
    assert uses["requires_node"] >= 3, f"node markers too few: {uses}"


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
