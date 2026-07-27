"""The outer benchmark state must follow the run's own execution gates.

The real E1 development run reported ``task status=completed``,
``completed_tasks=1``, ``failed_or_blocked_tasks=0`` and exit 0 while its own
``run_status.json`` recorded 7/12 steps, two failed steps and
``execution_complete=false``.  The outer layer was reading "the call returned"
as "the task succeeded".
"""

from __future__ import annotations

from typing import Any, Dict, List

from tools.run_research_agent_bench import (
    _arm_execution_succeeded,
    _failed_step_ids,
    _finish_task_on_execution_outcome,
    _incomplete_suite_items,
    _score_execution_failures,
)


def _arm(**updates: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "arm": "aware",
        "execution_complete": True,
        "step_scientific_requirements_complete": True,
        "required_step_count": 12,
        "completed_step_count": 12,
        "failed_step_ids": [],
        "missing_step_ids": [],
        # A development diagnostic is expected to end here.
        "paper_authorized": False,
        "publication_ready": False,
        "gate_status": "analysis_only",
    }
    payload.update(updates)
    return payload


def _score(**updates: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"item_key": "e1_sepsis3_prevalence_mortality"}
    payload.setdefault("aware", _arm())
    payload.update(updates)
    return payload


class _RecordingHandle:
    """Stand-in for the provider hard-stop task handle."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def finish(self, *, score: Any = None, error: str | None = None) -> None:
        self.calls.append({"score": score, "error": error})


def test_a_development_diagnostic_that_ran_counts_as_executed() -> None:
    """paper_authorized=false is expected here and must not fail the task."""

    arm = _arm(paper_authorized=False, gate_status="analysis_only")
    assert _arm_execution_succeeded(arm)
    assert _score_execution_failures(_score(aware=arm)) == []

    handle = _RecordingHandle()
    _finish_task_on_execution_outcome(handle, _score(aware=arm))
    assert handle.calls == [{"score": _score(aware=arm), "error": None}]


def test_diagnostic_only_status_alone_never_fails_the_task() -> None:
    """The demotion is a status, not an execution failure."""

    arm = _arm(
        status="diagnostic_only",
        forced_diagnostic_only=True,
        paper_authorized=False,
    )
    assert _arm_execution_succeeded(arm)
    assert _score_execution_failures(_score(aware=arm)) == []


def test_an_incomplete_execution_is_not_a_completed_task() -> None:
    """The exact real-run shape: 7/12 steps with two failed steps."""

    arm = _arm(
        execution_complete=False,
        step_scientific_requirements_complete=False,
        completed_step_count=7,
        failed_step_ids=[
            "05_missingness_measurement_audit_figure",
            "06_primary_adjusted_association",
        ],
        missing_step_ids=[
            "06_primary_adjusted_association_figure",
            "07_robustness_sensitivity",
            "07_robustness_sensitivity_figure",
        ],
    )
    assert not _arm_execution_succeeded(arm)

    failures = _score_execution_failures(_score(aware=arm))
    assert len(failures) == 1
    assert "7/12 steps" in failures[0]
    assert "05_missingness_measurement_audit_figure" in failures[0]
    assert "07_robustness_sensitivity" in failures[0]

    handle = _RecordingHandle()
    _finish_task_on_execution_outcome(handle, _score(aware=arm))
    assert len(handle.calls) == 1
    # The ledger marks a task failed exactly when ``error`` is not None, so the
    # outer totals now move with the run instead of against it.
    assert handle.calls[0]["error"]
    assert "did not complete execution" in handle.calls[0]["error"]


def test_a_failed_step_alone_blocks_completion() -> None:
    """Even a fully-stepped run is not complete while a step is failed."""

    assert not _arm_execution_succeeded(_arm(failed_step_ids=["04_figure"]))
    assert not _arm_execution_succeeded(_arm(missing_step_ids=["07_robustness"]))
    assert not _arm_execution_succeeded(
        _arm(step_scientific_requirements_complete=False)
    )
    assert not _arm_execution_succeeded(_arm(execution_complete=False))


def test_every_scored_arm_must_have_executed() -> None:
    """A second arm cannot ride on the first arm's completion."""

    score = _score(aware=_arm(), naive=_arm(execution_complete=False))
    failures = _score_execution_failures(score)
    assert len(failures) == 1
    assert failures[0].startswith("naive arm")


def test_a_missing_or_unscored_payload_is_not_a_success() -> None:
    assert _score_execution_failures(None) == [
        "benchmark item produced no score payload"
    ]
    assert _score_execution_failures({"item_key": "e1"}) == [
        "benchmark item produced no scored arm"
    ]


def test_failed_step_ids_reads_both_recorded_shapes() -> None:
    """run_status records failed steps as objects; older payloads used strings."""

    assert _failed_step_ids(
        {"failed_steps": [{"step_id": "05_figure", "status": "contract_failed"}]}
    ) == ["05_figure"]
    assert _failed_step_ids({"failed_steps": ["06_primary"]}) == ["06_primary"]
    assert _failed_step_ids({"failed_steps": []}) == []
    assert _failed_step_ids({}) == []


def test_rule_and_analysis_suites_report_their_incomplete_items():
    """The suite path scores runs too, so it can go false green the same way."""

    incomplete_arm = _arm(
        execution_complete=False,
        step_scientific_requirements_complete=False,
        completed_step_count=8,
        failed_step_ids=["06_primary_adjusted_association"],
        missing_step_ids=[
            "06_primary_adjusted_association_figure",
            "07_robustness_sensitivity",
            "07_robustness_sensitivity_figure",
        ],
    )
    runs = [
        {
            "model": "gpt-5.6-luna",
            "scores": [
                _score(
                    item_key="e1_sepsis3_prevalence_mortality", aware=incomplete_arm
                ),
                _score(item_key="e2_other"),
            ],
        }
    ]

    incomplete = _incomplete_suite_items(runs)

    assert incomplete == ["e1_sepsis3_prevalence_mortality (gpt-5.6-luna)"]


def test_a_completed_development_diagnostic_is_not_reported_as_incomplete():
    runs = [{"model": "gpt-5.6-luna", "scores": [_score()]}]

    assert _incomplete_suite_items(runs) == []


def test_every_model_in_a_matrix_run_is_checked():
    runs = [
        {"model": "model-a", "scores": [_score()]},
        {
            "model": "model-b",
            "scores": [_score(aware=_arm(execution_complete=False))],
        },
    ]

    assert _incomplete_suite_items(runs) == [
        "e1_sepsis3_prevalence_mortality (model-b)"
    ]


def test_a_suite_run_with_no_scores_is_not_silently_passed_over():
    """An empty score list is nothing to check; a broken score payload is not."""

    assert _incomplete_suite_items([{"model": "m", "scores": []}]) == []
    assert _incomplete_suite_items([{"model": "m", "scores": [None]}]) == ["? (m)"]


def test_no_benchmark_entry_point_can_exit_zero_without_checking_execution():
    """Fixing one entry point left the other reporting the same false green.

    ``main`` and ``_run_ehrflowbench_jsonl`` are both process exit codes. A new
    entry point that grows its own ``return 0`` must consult the execution axis
    too, so assert the property on the source rather than on the two functions
    that happen to exist today.
    """

    import ast
    import pathlib

    import tools.run_research_agent_bench as bench

    tree = ast.parse(pathlib.Path(bench.__file__).read_text(encoding="utf-8"))
    checkers = {"_incomplete_suite_items", "_score_execution_failures"}
    entry_points = {"main", "_run_ehrflowbench_jsonl"}
    checked: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in entry_points:
            continue
        exits_zero = any(
            isinstance(inner, ast.Return)
            and isinstance(inner.value, ast.Constant)
            and inner.value.value == 0
            for inner in ast.walk(node)
        )
        if not exits_zero:
            continue
        called = {
            inner.func.id
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
        }
        assert (
            called & checkers
        ), f"{node.name} can return 0 without consulting execution completion"
        checked.add(node.name)

    assert checked == entry_points
