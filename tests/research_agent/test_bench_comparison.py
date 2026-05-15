"""Tests for the ICUAgentBench A/B comparison harness.

The harness (``compare_bench_reports``) is the sink for ablation
experiments — e.g. "agent with HealthFlow-style experience memory" vs
"agent without memory". The memory implementation itself lives in
another work-stream; this layer is intentionally decoupled so the
two work-streams compose without conflict.

We exercise the harness on hand-built ``ICUAgentBenchReport`` objects
rather than running real agent loops, so these tests stay fast and
hermetic.
"""

from __future__ import annotations

import math


def _make_result(ra, task_id, *, correctness=None, prov=None, halluc=None, exec_=None):
    return ra.ICUAgentBenchTaskResult(
        task_id=task_id,
        correctness=correctness,
        provenance_completeness=prov,
        hallucination_rate=halluc,
        execution_success_rate=exec_,
    )


def _make_report(ra, results):
    return ra.aggregate_bench_report(ra.default_icu_agent_bench_suite(), results)


def test_compare_identical_reports_produces_zero_deltas(ra):
    """Comparing a report to itself yields all-zero aggregate deltas."""
    results = [
        _make_result(ra, "t1", correctness=1.0, prov=1.0, halluc=0.0, exec_=1.0),
        _make_result(ra, "t2", correctness=0.5, prov=0.5, halluc=0.0, exec_=1.0),
    ]
    report = _make_report(ra, results)

    cmp_ = ra.compare_bench_reports(report, report, baseline_name="a", treatment_name="a")

    assert cmp_.aggregate_deltas["correctness"] == 0.0
    assert cmp_.aggregate_deltas["provenance_completeness"] == 0.0
    assert cmp_.tasks_with_correctness_gain == []
    assert cmp_.tasks_with_correctness_loss == []
    assert sorted(cmp_.tasks_unchanged) == ["t1", "t2"]
    assert "matches" in cmp_.verdict


def test_compare_detects_per_task_correctness_gains(ra):
    """A treatment that improves correctness on a subset of tasks must
    surface those tasks by name and bump the aggregate delta in the
    right direction."""
    baseline = _make_report(ra, [
        _make_result(ra, "t1", correctness=0.5, prov=1.0, exec_=1.0),
        _make_result(ra, "t2", correctness=0.8, prov=1.0, exec_=1.0),
    ])
    treatment = _make_report(ra, [
        _make_result(ra, "t1", correctness=1.0, prov=1.0, exec_=1.0),  # +0.5
        _make_result(ra, "t2", correctness=0.6, prov=1.0, exec_=1.0),  # -0.2
    ])

    cmp_ = ra.compare_bench_reports(
        baseline, treatment, baseline_name="no-mem", treatment_name="mem"
    )

    assert cmp_.tasks_with_correctness_gain == ["t1"]
    assert cmp_.tasks_with_correctness_loss == ["t2"]
    # mean correctness delta = (0.5 + (-0.2)) / 2 = 0.15
    assert math.isclose(cmp_.aggregate_deltas["correctness"], 0.15, abs_tol=1e-9)
    assert "improves" in cmp_.verdict
    assert "mem" in cmp_.verdict


def test_compare_handles_missing_metrics_without_synthesising_zero(ra):
    """Tasks present on only one side must report ``delta=None`` and
    must NOT pull the aggregate toward zero.
    """
    baseline = _make_report(ra, [
        _make_result(ra, "shared", correctness=0.5, exec_=1.0),
        _make_result(ra, "baseline_only", correctness=0.9, exec_=1.0),
    ])
    treatment = _make_report(ra, [
        _make_result(ra, "shared", correctness=1.0, exec_=1.0),
        _make_result(ra, "treatment_only", correctness=0.3, exec_=1.0),
    ])

    cmp_ = ra.compare_bench_reports(baseline, treatment)

    # Only the "shared" task contributes to the correctness aggregate.
    assert math.isclose(cmp_.aggregate_deltas["correctness"], 0.5, abs_tol=1e-9)

    one_sided = [
        d for d in cmp_.per_task_deltas
        if d.task_id in {"baseline_only", "treatment_only"}
        and d.metric == "correctness"
    ]
    assert all(d.delta is None for d in one_sided), (
        "one-sided rows should have delta=None, not 0.0"
    )


def test_compare_per_task_deltas_are_diff_stable(ra):
    """Output ordering is deterministic — sorted by (task_id, metric).
    Without this, two runs of the same comparison could produce
    superficially different reports and break paper reproducibility.
    """
    baseline = _make_report(ra, [
        _make_result(ra, "z_task", correctness=0.4, exec_=1.0),
        _make_result(ra, "a_task", correctness=0.6, exec_=1.0),
    ])
    treatment = _make_report(ra, [
        _make_result(ra, "a_task", correctness=0.7, exec_=1.0),
        _make_result(ra, "z_task", correctness=0.5, exec_=1.0),
    ])

    cmp1 = ra.compare_bench_reports(baseline, treatment)
    cmp2 = ra.compare_bench_reports(baseline, treatment)

    ids1 = [(d.task_id, d.metric) for d in cmp1.per_task_deltas]
    ids2 = [(d.task_id, d.metric) for d in cmp2.per_task_deltas]
    assert ids1 == ids2  # deterministic
    # Sorted primarily by task_id
    task_order = [d.task_id for d in cmp1.per_task_deltas]
    assert task_order == sorted(task_order)


def test_compare_verdict_handles_all_none_correctness(ra):
    """When neither arm has any comparable correctness signal the verdict
    must say so explicitly rather than printing ``+0.000`` and pretending
    everything matched.
    """
    baseline = _make_report(ra, [
        _make_result(ra, "t1", correctness=None, exec_=0.0),
    ])
    treatment = _make_report(ra, [
        _make_result(ra, "t1", correctness=None, exec_=0.0),
    ])

    cmp_ = ra.compare_bench_reports(baseline, treatment)
    assert "correctness" not in cmp_.aggregate_deltas
    assert "No comparable evaluation-category correctness signal" in cmp_.verdict


def test_compare_records_hallucination_regression(ra):
    """Hallucination is a *bad* metric — when treatment > baseline that's
    a regression. The harness records the raw delta direction; callers
    decide which direction is "good" per metric.
    """
    baseline = _make_report(ra, [
        _make_result(ra, "t1", correctness=1.0, halluc=0.0, exec_=1.0),
    ])
    treatment = _make_report(ra, [
        _make_result(ra, "t1", correctness=1.0, halluc=1.0, exec_=1.0),
    ])

    cmp_ = ra.compare_bench_reports(baseline, treatment)
    assert cmp_.aggregate_deltas["hallucination_rate"] == 1.0
    assert cmp_.aggregate_deltas["correctness"] == 0.0


def test_compare_with_suite_isolates_self_check_from_headline(ra):
    """When the suite is provided, self-check task deltas must NOT
    contribute to ``aggregate_deltas`` — they go to
    ``self_check_aggregate_deltas`` and the verdict cites the
    evaluation bucket only.
    """
    suite = ra.default_icu_agent_bench_suite()
    eval_id = suite.evaluation_task_ids()[0]
    self_id = suite.self_check_task_ids()[0]

    baseline = _make_report(ra, [
        _make_result(ra, eval_id, correctness=0.5, exec_=1.0),
        _make_result(ra, self_id, correctness=0.5, exec_=1.0),
    ])
    treatment = _make_report(ra, [
        _make_result(ra, eval_id, correctness=0.9, exec_=1.0),  # +0.4 eval
        _make_result(ra, self_id, correctness=1.0, exec_=1.0),  # +0.5 self-check
    ])

    cmp_ = ra.compare_bench_reports(
        baseline, treatment, suite=suite,
        baseline_name="no-mem", treatment_name="mem",
    )

    # Headline = evaluation only, not the inflated combined number.
    assert math.isclose(cmp_.aggregate_deltas["correctness"], 0.4, abs_tol=1e-9)
    assert math.isclose(
        cmp_.self_check_aggregate_deltas["correctness"], 0.5, abs_tol=1e-9
    )
    assert "evaluation correctness by +0.400" in cmp_.verdict
    # Verdict mentions evaluation explicitly so a reader can't mistake
    # the combined-mean for the headline-mean.
    assert "evaluation" in cmp_.verdict

    # Per-task delta rows carry category so callers can post-filter.
    eval_corr = next(
        d for d in cmp_.per_task_deltas
        if d.task_id == eval_id and d.metric == "correctness"
    )
    self_corr = next(
        d for d in cmp_.per_task_deltas
        if d.task_id == self_id and d.metric == "correctness"
    )
    assert eval_corr.category == "evaluation"
    assert self_corr.category == "self_check"


def test_compare_without_suite_falls_back_to_evaluation_default(ra):
    """Back-compat: callers that don't pass ``suite`` still get a
    populated ``aggregate_deltas`` (all tasks defaulting to evaluation).
    """
    baseline = _make_report(ra, [
        _make_result(ra, "ad_hoc_task", correctness=0.5, exec_=1.0),
    ])
    treatment = _make_report(ra, [
        _make_result(ra, "ad_hoc_task", correctness=0.9, exec_=1.0),
    ])
    cmp_ = ra.compare_bench_reports(baseline, treatment)  # no suite=
    assert math.isclose(cmp_.aggregate_deltas["correctness"], 0.4, abs_tol=1e-9)
    assert cmp_.self_check_aggregate_deltas == {}


def test_compare_works_end_to_end_on_synthetic_frozen_suite(ra, synthetic_cohort):
    """Round-trip: grade every frozen synthetic task twice (honest vs
    sabotaged), feed both into ``aggregate_bench_report``, then diff.

    This is the smallest end-to-end exercise of:
      grade_bench_task → aggregate_bench_report → compare_bench_reports

    A failure here means one of the three layers regressed in a way
    the targeted unit tests above missed.
    """
    from tests.research_agent.test_scientific_regression import _compute_cohort_stats

    stats = _compute_cohort_stats(synthetic_cohort)
    suite = ra.default_icu_agent_bench_suite()
    frozen = [
        t for t in suite.tasks
        if t.gold_answer is not None and t.gold_answer.data_fixture == "synthetic_cohort"
    ]

    honest_results = []
    sabotaged_results = []
    for task in frozen:
        observed = {n: stats[n] for n in task.gold_answer.numeric_targets}
        warns = (
            ["\n".join(task.gold_answer.required_warnings)]
            if task.gold_answer.required_warnings else []
        )
        honest_results.append(
            ra.grade_bench_task(
                task,
                observed_metrics=observed,
                observed_warnings=warns,
                observed_outputs=[],
            )
        )
        # Sabotaged: drop required warnings AND inflate first numeric target
        sabotaged_obs = dict(observed)
        first_metric = next(iter(task.gold_answer.numeric_targets))
        bound = task.gold_answer.numeric_targets[first_metric]
        # Push the value 10x past the upper (or below lower if upper is None)
        if bound.upper is not None:
            sabotaged_obs[first_metric] = bound.upper * 10 + 1
        elif bound.lower is not None:
            sabotaged_obs[first_metric] = bound.lower - 1.0
        sabotaged_results.append(
            ra.grade_bench_task(
                task,
                observed_metrics=sabotaged_obs,
                observed_warnings=[],  # required warnings suppressed
                observed_outputs=[],
            )
        )

    honest_report = ra.aggregate_bench_report(suite, honest_results)
    sabotaged_report = ra.aggregate_bench_report(suite, sabotaged_results)

    cmp_ = ra.compare_bench_reports(
        sabotaged_report,
        honest_report,
        baseline_name="sabotaged",
        treatment_name="honest",
    )

    # honest should beat sabotaged across the board
    assert cmp_.aggregate_deltas["correctness"] > 0
    assert cmp_.aggregate_deltas["provenance_completeness"] >= 0
    assert "improves" in cmp_.verdict
    assert len(cmp_.tasks_with_correctness_gain) == len(frozen)
    assert cmp_.tasks_with_correctness_loss == []
