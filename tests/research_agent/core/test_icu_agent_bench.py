from __future__ import annotations


def test_default_icu_agent_bench_suite_has_core_tasks(ra):
    suite = ra.default_icu_agent_bench_suite()
    task_ids = {task.task_id for task in suite.tasks}
    assert {
        "cohort_extraction",
        "aki_staging",
        "sofa_extraction",
        "ventilation_duration",
        "sepsis_onset",
        "mortality_prediction",
        "competing_risk_analysis",
        "survival_analysis",
        "longitudinal_trajectory_analysis",
        "cross_db_sofa_mortality",
    } <= task_ids


def test_icu_agent_bench_markdown_mentions_metrics(ra):
    markdown = ra.icu_agent_bench_markdown()
    assert "correctness" in markdown
    assert "provenance_completeness" in markdown
    assert "icu_semantic_validity" in markdown


def test_icu_agent_bench_includes_cross_database_replication_task(ra):
    suite = ra.default_icu_agent_bench_suite()
    task = next(t for t in suite.tasks if t.task_id == "cross_db_sofa_mortality")

    assert task.kind == "cross_database_replication"
    assert {"miiv", "eicu", "hirid"} <= set(task.target_databases)
    assert "concept availability matrix" in task.expected_outputs
    assert task.gold_answer_status == "planned"


# ---------------------------------------------------------------------------
# Frozen self-checkable task + grading layer
# ---------------------------------------------------------------------------


def test_synthetic_completeness_task_ships_with_frozen_gold_answer(ra):
    """The synthetic-cohort task is the first frozen ICUAgentBench task.

    It declares numeric bounds, a required guardrail warning, and a
    forbidden-output substring — i.e. all three grader axes are
    exercised by a single task.
    """
    suite = ra.default_icu_agent_bench_suite()
    task = next(
        t for t in suite.tasks if t.task_id == "synthetic_cohort_completeness_qc"
    )

    assert task.gold_answer_status == "frozen"
    assert task.gold_answer is not None
    assert task.gold_answer.data_fixture == "synthetic_cohort"
    assert "death_rate" in task.gold_answer.numeric_targets
    assert "sofa2_low_component_frac" in task.gold_answer.numeric_targets
    assert "component_completeness_qc" in task.gold_answer.required_warnings
    assert any(
        "imputed" in s for s in task.gold_answer.forbidden_outputs
    )
    assert "synthetic_cohort_completeness_qc" in suite.frozen_task_ids()


def test_numeric_bound_contains_handles_open_intervals(ra):
    """``ICUAgentBenchNumericBound`` allows open-ended bounds."""
    bound = ra.ICUAgentBenchNumericBound(lower=None, upper=1.0)
    assert bound.contains(-9999.0)
    assert bound.contains(1.0)
    assert not bound.contains(1.0001)

    upper_open = ra.ICUAgentBenchNumericBound(lower=0.5, upper=None)
    assert upper_open.contains(10_000.0)
    assert not upper_open.contains(0.4999)


def test_grade_bench_task_returns_none_correctness_without_gold(ra):
    """Tasks without gold answers must not synthesize correctness scores."""
    suite = ra.default_icu_agent_bench_suite()
    descriptive = next(t for t in suite.tasks if t.task_id == "aki_staging")
    assert descriptive.gold_answer is None

    result = ra.grade_bench_task(
        descriptive,
        observed_metrics={"stage_count": 3},
        observed_warnings=[],
        observed_outputs=["staging table"],
    )
    assert result.correctness is None
    assert result.execution_success_rate == 1.0


def test_grade_bench_task_zero_correctness_when_metrics_missing(ra):
    """If every required metric is missing, correctness collapses to 0."""
    suite = ra.default_icu_agent_bench_suite()
    task = next(
        t for t in suite.tasks if t.task_id == "synthetic_cohort_completeness_qc"
    )

    result = ra.grade_bench_task(
        task,
        observed_metrics={},  # nothing reported
        observed_warnings=["component_completeness_qc"],
        observed_outputs=[],
    )
    # observed_metrics empty → execution_success_rate=0 and correctness=0.0
    assert result.execution_success_rate == 0.0
    assert result.correctness == 0.0
    assert any("missing metric" in n for n in result.notes)


def test_icu_agent_bench_markdown_lists_frozen_count(ra):
    """The rendered markdown surfaces how many tasks have gold answers."""
    md = ra.icu_agent_bench_markdown()
    assert "Frozen tasks:" in md
    assert "synthetic_cohort_completeness_qc" in md
    # the synthetic task declares numeric targets — they must show up
    assert "death_rate" in md


def test_synthetic_tasks_are_marked_self_check(ra):
    """All three frozen synthetic tasks must live in the ``self_check``
    category — otherwise they would silently inflate the evaluation
    headline that a paper cites.
    """
    suite = ra.default_icu_agent_bench_suite()
    self_check_ids = set(suite.self_check_task_ids())
    assert {
        "synthetic_cohort_completeness_qc",
        "synthetic_cohort_table_one",
        "synthetic_cohort_stratified_sofa_mortality",
    } <= self_check_ids

    # No evaluation task should secretly depend on the in-repo
    # synthetic fixture — that would be a misclassification.
    eval_tasks = [t for t in suite.tasks if t.category == "evaluation"]
    for task in eval_tasks:
        if task.gold_answer is not None:
            assert task.gold_answer.data_fixture != "synthetic_cohort", (
                f"Evaluation task {task.task_id!r} declares the in-repo "
                "synthetic fixture as its gold-answer source — this would "
                "mean a CI fixture is being graded as a real-data evaluation."
            )


def test_aggregate_splits_eval_and_self_check_buckets(ra):
    """Mixed eval+self-check results must end up in separate aggregates,
    and the headline must not be polluted by the self-check side.
    """
    suite = ra.default_icu_agent_bench_suite()
    eval_id = suite.evaluation_task_ids()[0]
    self_id = suite.self_check_task_ids()[0]

    results = [
        ra.ICUAgentBenchTaskResult(
            task_id=eval_id, correctness=0.4, execution_success_rate=1.0
        ),
        ra.ICUAgentBenchTaskResult(
            task_id=self_id, correctness=1.0, execution_success_rate=1.0
        ),
    ]
    report = ra.aggregate_bench_report(suite, results)

    # Headline must reflect only the evaluation result.
    assert report.aggregate["correctness"] == 0.4
    # Self-check bucket reflects only the self_check result.
    assert report.self_check_aggregate["correctness"] == 1.0
    # Per-task results keep both rows for inspection.
    assert {r.task_id for r in report.task_results} == {eval_id, self_id}


def test_aggregate_defaults_unknown_task_ids_to_evaluation(ra):
    """Results whose task_id is not in the suite default to evaluation,
    so call sites that fabricate ad-hoc task ids (e.g. in tests) still
    get a headline aggregate — they don't silently disappear."""
    suite = ra.default_icu_agent_bench_suite()
    results = [
        ra.ICUAgentBenchTaskResult(
            task_id="not_in_suite_xyz", correctness=0.7, execution_success_rate=1.0
        ),
    ]
    report = ra.aggregate_bench_report(suite, results)
    assert report.aggregate["correctness"] == 0.7
    assert "correctness" not in report.self_check_aggregate


def test_docs_icu_agent_bench_md_matches_live_renderer(ra):
    """The on-disk paper-appendix doc must stay in sync with the suite.

    ``docs/icu_agent_bench.md`` is rendered from
    ``default_icu_agent_bench_suite()``. If a future PR adds a task or
    changes a bound without regenerating the doc, this test will fail
    and tell the author exactly how to fix it.

    To regenerate after a legitimate change::

        python -c "from easyicu.research_agent import \
            icu_agent_bench_markdown; \
            open('docs/icu_agent_bench.md','w').write(icu_agent_bench_markdown())"
    """
    import pathlib

    import pytest

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    doc_path = repo_root / "docs" / "icu_agent_bench.md"
    if not doc_path.exists():
        # docs/*.md is excluded from the slimmed public repository surface
        # (see 874db1c). When the generated bench doc is absent there is
        # nothing to compare against the live renderer; skip rather than fail.
        pytest.skip(
            f"{doc_path} not present (excluded from the public repo surface); "
            "drift check only runs where the generated doc is kept."
        )
    on_disk = doc_path.read_text()
    live = ra.icu_agent_bench_markdown()
    assert on_disk == live, (
        "docs/icu_agent_bench.md drifted from the live renderer. "
        "Regenerate it by running:\n"
        "  python -c \"from easyicu.research_agent import "
        "icu_agent_bench_markdown; "
        "open('docs/icu_agent_bench.md','w').write(icu_agent_bench_markdown())\""
    )
