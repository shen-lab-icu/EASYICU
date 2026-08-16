"""One-line frozen benchmark access stays a prototype surface."""

from __future__ import annotations

import pytest

from easyicu.research_agent.benchmark_instances import (
    BENCHMARK_INSTANCE_SET_NAME,
    evaluate,
    frozen_instance_ids,
)
from easyicu.research_agent.icu_agent_bench import default_icu_agent_bench_suite


def test_frozen_ids_match_the_underlying_prototype_suite() -> None:
    suite = default_icu_agent_bench_suite()
    assert suite.maturity == "prototype"
    assert frozen_instance_ids() == suite.frozen_task_ids()
    assert frozen_instance_ids()  # non-empty checkable subset


def test_evaluate_scores_a_frozen_instance_in_one_call() -> None:
    task_id = frozen_instance_ids()[0]
    suite = default_icu_agent_bench_suite()
    task = next(t for t in suite.tasks if t.task_id == task_id)
    assert task.gold_answer is not None

    result = evaluate(
        task_id,
        observed_metrics={"n_rows": 100},
        observed_warnings=[],
        observed_outputs=[],
        run_id="run-facade-smoke",
    )

    assert result.task_id == task_id
    assert result.run_id == "run-facade-smoke"
    assert result.execution_success_rate == 1.0


def test_evaluate_refuses_unknown_or_non_frozen_ids() -> None:
    with pytest.raises(KeyError):
        evaluate("does-not-exist")
    with pytest.raises(KeyError):
        evaluate("not-a-frozen-instance")

    assert BENCHMARK_INSTANCE_SET_NAME == "icuagentbench-frozen-v1"
