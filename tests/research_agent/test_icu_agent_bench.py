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
    } <= task_ids


def test_icu_agent_bench_markdown_mentions_metrics(ra):
    markdown = ra.icu_agent_bench_markdown()
    assert "correctness" in markdown
    assert "provenance_completeness" in markdown
    assert "icu_semantic_validity" in markdown
