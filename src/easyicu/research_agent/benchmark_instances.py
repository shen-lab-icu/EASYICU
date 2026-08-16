"""Versioned one-line access to the frozen ICUAgentBench instance subset.

This is the Biomni-Eval1-style ``evaluate(task_id, ...)`` ergonomic surface
for EasyICU's *prototype* benchmark. It is deliberately not a new benchmark
authority: the scorer and suite remain :mod:`icu_agent_bench`, the frozen
instance set is exactly ``ICUAgentBenchSuite.frozen_task_ids()``, and the
outputs stay prototype framework results — never manuscript scores.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .icu_agent_bench import (
    ICUAgentBenchTaskResult,
    default_icu_agent_bench_suite,
    grade_bench_task,
)

__all__ = [
    "BENCHMARK_INSTANCE_SET_NAME",
    "frozen_instance_ids",
    "evaluate",
    "evaluate_suite",
]

BENCHMARK_INSTANCE_SET_NAME = "icuagentbench-frozen-v1"


def frozen_instance_ids() -> List[str]:
    """Return the versioned list of checkable instance ids."""
    suite = default_icu_agent_bench_suite()
    return suite.frozen_task_ids()


def evaluate(
    task_id: str,
    *,
    observed_metrics: Optional[Dict[str, float]] = None,
    observed_warnings: Optional[Sequence[str]] = None,
    observed_outputs: Optional[Sequence[str]] = None,
    run_id: Optional[str] = None,
) -> ICUAgentBenchTaskResult:
    """Score one observed run against a frozen checkable instance.

    One line, no I/O, no LLM, no pipeline state. ``task_id`` must be one of
    :func:`frozen_instance_ids`; anything else raises ``KeyError`` so a typo
    cannot silently score as a missing metric.
    """
    suite = default_icu_agent_bench_suite()
    by_id = {task.task_id: task for task in suite.tasks}
    task = by_id.get(task_id)
    if task is None or task.gold_answer is None:
        raise KeyError(f"not a frozen checkable benchmark instance: {task_id}")
    return grade_bench_task(
        task,
        observed_metrics=observed_metrics,
        observed_warnings=observed_warnings,
        observed_outputs=observed_outputs,
        run_id=run_id,
    )


def evaluate_suite(
    observed: Dict[str, Dict[str, object]],
) -> List[ICUAgentBenchTaskResult]:
    """Score a batch of ``{task_id: kwargs}`` entries in frozen-id order."""
    return [
        evaluate(task_id, **kwargs)  # type: ignore[arg-type]
        for task_id, kwargs in observed.items()
    ]
