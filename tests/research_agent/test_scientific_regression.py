"""Scientific regression tests.

These tests pin down the *numerical fingerprint* of the
``synthetic_cohort`` fixture and exercise the ICUAgentBench grader
end-to-end against it. Their job is to surface unintended drift
quickly:

* If a refactor changes the synthetic cohort's RNG path, descriptive
  stats will drift outside the bounds and these tests will fail.
* If the grader logic regresses (e.g. silently returns ``correctness``
  when no gold answer is set, or stops penalising out-of-bound metrics),
  these tests will fail.

Why is this layer worth its own file? Most existing tests check
*behaviour* ("the function returns the right shape", "the agent emits
a warning string"). They do not assert that the *numbers* a downstream
paper depends on stayed stable across a refactor — and a silent
numerical drift in a methods paper is the worst kind of regression.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest


# ----------------------------------------------------------------------
# Golden numeric fingerprint of synthetic_cohort
# ----------------------------------------------------------------------
# Values measured from the in-repo synthetic_cohort fixture
# (numpy.random.default_rng(7), n=800). Tolerances are generous (~5%
# of value, or absolute floor) so this does not flap on minor
# floating-point drift across numpy minor versions or platforms.

_GOLDEN = {
    "n_rows": (800, 800),
    "death_rate": (0.10, 0.15),
    "sofa2_low_component_frac": (0.08, 0.11),
    "sofa2_mean": (6.0, 6.5),
    "sofa2_max": (12, 14),
    "age_mean": (62.0, 65.0),
    "vaso_rate": (0.40, 0.48),
    "los_mean": (4.0, 5.5),
}


def _compute_cohort_stats(df: pd.DataFrame) -> dict[str, float]:
    """Compute every metric referenced by a frozen synthetic task.

    Keep this in sync with ``ICUAgentBenchGoldAnswer.numeric_targets``
    across all ``synthetic_cohort_*`` tasks. The
    ``test_all_frozen_synthetic_tasks_pass_grader`` test below acts as
    the safety net: it iterates the frozen suite and would fail if a
    declared target lacks a computation here.
    """

    nonzero = df.loc[df["sofa2"] > 0]
    if not nonzero.empty:
        q33 = nonzero["sofa2"].quantile(0.33)
        q67 = nonzero["sofa2"].quantile(0.67)
        low = nonzero[nonzero["sofa2"] <= q33]
        high = nonzero[nonzero["sofa2"] >= q67]
        death_low = float(low["death"].mean()) if not low.empty else float("nan")
        death_high = float(high["death"].mean()) if not high.empty else float("nan")
    else:
        death_low = death_high = float("nan")

    return {
        # ---- component-completeness QC task ------------------------
        "n_rows": float(len(df)),
        "death_rate": float(df["death"].mean()),
        "sofa2_low_component_frac": float((df["sofa2_n_components"] < 6).mean()),
        "sofa2_mean": float(df["sofa2"].mean()),
        "sofa2_max": float(df["sofa2"].max()),
        "age_mean": float(df["age"].mean()),
        "vaso_rate": float(df["vaso"].mean()),
        "los_mean": float(df["los_icu"].mean()),
        # ---- outcome-blind structural signal locked in its own test -
        "sofa2_low_component_count": float((df["sofa2_n_components"] < 6).sum()),
        # ---- table-one task ----------------------------------------
        "age_median": float(df["age"].median()),
        "age_p25": float(df["age"].quantile(0.25)),
        "age_p75": float(df["age"].quantile(0.75)),
        "los_median": float(df["los_icu"].median()),
        "lact_mean": float(df["lact"].mean()),
        "creat_mean": float(df["creat"].mean()),
        # ---- stratified-mortality task -----------------------------
        "death_low_tertile_excl_zero": death_low,
        "death_high_tertile_excl_zero": death_high,
        "death_delta_hi_minus_lo": death_high - death_low,
    }


def test_synthetic_cohort_locked_statistics(synthetic_cohort):
    """Every locked stat falls inside its golden bound.

    This is the safety net for the fixture itself: changing the RNG,
    the seed, or the generative model would shift these numbers and
    immediately surface a refactor that silently rewrote the smoke
    cohort.
    """

    stats = _compute_cohort_stats(synthetic_cohort)
    for name, (lo, hi) in _GOLDEN.items():
        value = stats[name]
        assert lo <= value <= hi, (
            f"synthetic_cohort.{name}={value} drifted outside locked "
            f"bound [{lo}, {hi}]. If this drift is intentional, update "
            f"_GOLDEN in this file together with the change."
        )


def test_synthetic_cohort_preserves_component_completeness_signal(synthetic_cohort):
    """The fixture keeps an outcome-blind SOFA-2 component-completeness signal."""

    stats = _compute_cohort_stats(synthetic_cohort)
    assert stats["sofa2_low_component_count"] > 0, (
        "The synthetic cohort no longer contains low-component SOFA-2 rows. "
        "That removes the outcome-blind component-completeness signal the "
        "self-check task is supposed to exercise."
    )


# ----------------------------------------------------------------------
# Grader end-to-end on the frozen self-checkable task
# ----------------------------------------------------------------------


def _frozen_synthetic_task(ra):
    suite = ra.default_icu_agent_bench_suite()
    return next(
        t for t in suite.tasks if t.task_id == "synthetic_cohort_completeness_qc"
    )


def test_grade_bench_task_passes_on_well_formed_run(ra, synthetic_cohort):
    """An honest run against synthetic_cohort scores correctness=1.0."""

    task = _frozen_synthetic_task(ra)
    stats = _compute_cohort_stats(synthetic_cohort)
    observed = {
        name: stats[name]
        for name in task.gold_answer.numeric_targets
    }

    result = ra.grade_bench_task(
        task,
        observed_metrics=observed,
        observed_warnings=["component_completeness_qc: low component rows present"],
        observed_outputs=["descriptive stats produced"],
        run_id="regression-honest",
    )

    assert result.task_id == "synthetic_cohort_completeness_qc"
    assert result.correctness == 1.0
    assert result.provenance_completeness == 1.0
    assert result.hallucination_rate == 0.0
    assert result.execution_success_rate == 1.0


def test_grade_bench_task_penalises_out_of_bound_metric(ra, synthetic_cohort):
    """A run that fabricates a too-high death rate must lose correctness."""

    task = _frozen_synthetic_task(ra)
    stats = _compute_cohort_stats(synthetic_cohort)
    observed = {
        name: stats[name]
        for name in task.gold_answer.numeric_targets
    }
    observed["death_rate"] = 0.95  # implausible — outside locked bound

    result = ra.grade_bench_task(
        task,
        observed_metrics=observed,
        observed_warnings=["component_completeness_qc"],
        observed_outputs=[],
    )
    assert result.correctness is not None
    assert result.correctness < 1.0
    assert any("out-of-bound" in note for note in result.notes)


def test_grade_bench_task_flags_missing_guardrail_warning(ra, synthetic_cohort):
    """A run that hides component completeness loses provenance_completeness."""

    task = _frozen_synthetic_task(ra)
    stats = _compute_cohort_stats(synthetic_cohort)
    observed = {
        name: stats[name]
        for name in task.gold_answer.numeric_targets
    }
    result = ra.grade_bench_task(
        task,
        observed_metrics=observed,
        observed_warnings=[],  # required warning suppressed
        observed_outputs=[],
    )
    assert result.provenance_completeness == 0.0
    assert any("missing required warning" in n for n in result.notes)


def test_grade_bench_task_flags_forbidden_output_substring(ra, synthetic_cohort):
    """A run that prints a forbidden phrase trips hallucination_rate."""

    task = _frozen_synthetic_task(ra)
    stats = _compute_cohort_stats(synthetic_cohort)
    observed = {
        name: stats[name]
        for name in task.gold_answer.numeric_targets
    }
    result = ra.grade_bench_task(
        task,
        observed_metrics=observed,
        observed_warnings=["component_completeness_qc"],
        observed_outputs=["values were silently imputed for missing rows"],
    )
    assert result.hallucination_rate == 1.0
    assert any("forbidden output" in n for n in result.notes)


def test_grade_bench_task_neutral_on_descriptive_only_task(ra):
    """Tasks without a gold answer should not synthesize correctness."""

    suite = ra.default_icu_agent_bench_suite()
    descriptive = next(t for t in suite.tasks if t.task_id == "ventilation_duration")
    assert descriptive.gold_answer is None

    result = ra.grade_bench_task(
        descriptive,
        observed_metrics={"duration_mean_hours": 36.4},
        observed_warnings=[],
        observed_outputs=["duration table emitted"],
    )
    assert result.correctness is None
    assert result.provenance_completeness is None
    assert result.hallucination_rate is None
    # execution_success_rate still resolves: the run produced metrics
    assert result.execution_success_rate == 1.0


# ----------------------------------------------------------------------
# Suite self-consistency
# ----------------------------------------------------------------------


def _all_frozen_synthetic_tasks(ra):
    """Every frozen task whose data fixture is the synthetic cohort.

    A pytest ``parametrize`` over this list catches the failure mode
    "someone added a frozen task to the suite but forgot to add the
    metric computation to ``_compute_cohort_stats``" — without it the
    new task would silently grade ``correctness=0`` because every
    metric is "missing" rather than being out of bounds.
    """
    suite = ra.default_icu_agent_bench_suite()
    return [
        task
        for task in suite.tasks
        if task.gold_answer is not None
        and task.gold_answer.data_fixture == "synthetic_cohort"
    ]


def test_suite_has_at_least_three_frozen_synthetic_tasks(ra):
    """If a task is dropped from the frozen subset, surface it loudly.

    Three is the minimum that makes a "mini-benchmark" claim defensible.
    Loosening this assertion later is fine; tightening silently is not.
    """
    tasks = _all_frozen_synthetic_tasks(ra)
    assert len(tasks) >= 3, (
        f"Expected at least 3 frozen synthetic tasks but found {len(tasks)}: "
        f"{[t.task_id for t in tasks]}"
    )


def test_all_frozen_synthetic_tasks_pass_grader(ra, synthetic_cohort):
    """Suite self-consistency: every frozen synthetic task grades cleanly
    when fed the honest statistics computed from its declared fixture.

    A failure here means one of:
      (a) the gold_answer bounds drifted out of sync with the fixture;
      (b) ``_compute_cohort_stats`` is missing a metric the task declares;
      (c) the fixture itself changed numerically (caught upstream by
          ``test_synthetic_cohort_locked_statistics``).
    """
    stats = _compute_cohort_stats(synthetic_cohort)
    failures: list[str] = []
    for task in _all_frozen_synthetic_tasks(ra):
        observed = {name: stats[name] for name in task.gold_answer.numeric_targets}
        # Synthesize a guardrail string that satisfies all required
        # warnings so this self-consistency check focuses on the
        # numeric layer.
        warnings_blob = "\n".join(task.gold_answer.required_warnings)
        result = ra.grade_bench_task(
            task,
            observed_metrics=observed,
            observed_warnings=[warnings_blob] if warnings_blob else [],
            observed_outputs=[],
        )
        if result.correctness != 1.0:
            failures.append(
                f"{task.task_id}: correctness={result.correctness}, notes={result.notes}"
            )
        if task.gold_answer.required_warnings and result.provenance_completeness != 1.0:
            failures.append(
                f"{task.task_id}: provenance_completeness={result.provenance_completeness}"
            )

    assert not failures, "frozen synthetic tasks not self-consistent:\n" + "\n".join(failures)


def test_aggregate_bench_report_averages_only_non_null_metrics(ra, synthetic_cohort):
    """The aggregator ignores missing metrics rather than treating them as 0.

    The two results feed a ``self_check`` task, so they must show up in
    ``self_check_aggregate`` and NOT in the headline ``aggregate`` —
    this is the category-split safety net.
    """

    task = _frozen_synthetic_task(ra)
    stats = _compute_cohort_stats(synthetic_cohort)
    observed = {n: stats[n] for n in task.gold_answer.numeric_targets}

    honest = ra.grade_bench_task(
        task,
        observed_metrics=observed,
        observed_warnings=["component_completeness_qc"],
        observed_outputs=[],
        run_id="honest",
    )
    half = ra.grade_bench_task(
        task,
        observed_metrics={**observed, "death_rate": 0.95},
        observed_warnings=["component_completeness_qc"],
        observed_outputs=[],
        run_id="half",
    )

    suite = ra.default_icu_agent_bench_suite()
    report = ra.aggregate_bench_report(suite, [honest, half])

    # synthetic_cohort_completeness_qc is a self_check task → headline
    # aggregate must stay clean of these results.
    assert "correctness" not in report.aggregate, (
        "Self-check tasks must not pollute the evaluation headline."
    )

    # honest=1.0, half=0.8 (4/5 within bound)  →  mean = 0.9 in the
    # self-check bucket.
    assert math.isclose(
        report.self_check_aggregate["correctness"], 0.9, abs_tol=1e-9
    )
    assert report.self_check_aggregate["provenance_completeness"] == 1.0
    assert report.self_check_aggregate["execution_success_rate"] == 1.0
    # reproducibility was never set — must be absent from any aggregate
    assert "reproducibility" not in report.self_check_aggregate
    assert len(report.task_results) == 2
