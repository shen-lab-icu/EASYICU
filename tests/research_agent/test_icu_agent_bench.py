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
