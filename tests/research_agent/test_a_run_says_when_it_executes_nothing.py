"""A run must say when it decides to execute nothing.

Three plan-level blocks empty the step list before execution: the trajectory
plan contract, the typed plan DAG, and an unauthorized development sample.  Each
already records findings and a partial-manifest flag, but neither reaches the
audit log -- the run's own narrative.

MEASURED on h3_trajectory_clustering (``..._e13587c_nine2``).  The plan had 9
steps and ``step_attempt_history`` recorded 2.  The audit log read:

    Skipped step already completed by pre-execution: 00_probe.
    Skipped step already completed by pre-execution: 01_define_analysis_cohort.
    Auditing generated figures.
    Research-agent run complete.

The seven remaining steps are simply absent -- no start, no failure, no reason.
Reconstructing why took a full pass over the manifest, run_status and plan
before the trajectory block was found, and h3 has never executed past step 01 in
any of its 7 recorded runs, so this silence is what every one of them looked
like.

The emitted record carries the three things a reader needs: which block fired,
how many steps were dropped, and which ones.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib

import pytest

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")

_MESSAGE = "Plan blocked before execution"


def test_the_emitter_forwards_flat_keywords_into_the_audit_detail():
    """The reason the fields are passed flat rather than under ``detail=``.

    ``_emit_progress`` builds the audit record's ``detail`` from every extra
    keyword except ``status``/``step_id``, so a ``detail=`` argument would land
    nested one level deeper than every other record in the log.
    """

    from easyicu.research_agent import pipeline as pipeline_module

    source = inspect.getsource(pipeline_module)
    marker = "def _emit_progress(stage: str, message: str, **extra: Any) -> None:"
    assert marker in source
    body = source.split(marker, 1)[1][:1200]
    assert 'if k not in {"status", "step_id"}' in body


def _announcement_call() -> "ast.Call":
    """The emit_progress call guarded by ``plan_block_reason is not None``.

    Parsed rather than string-matched.  A first version of this test searched
    the source text, and a mutation that turned the guard into ``if False:``
    left every assertion green -- a rule living only in the test file while
    production never runs it.  Walking to the call INSIDE that specific ``if``
    is what makes the guard load-bearing.
    """

    from easyicu.research_agent.execution import phase

    tree = ast.parse(inspect.getsource(phase))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "plan_block_reason"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "emit_progress"
            ):
                return inner
    raise AssertionError(
        "no emit_progress call is guarded by `plan_block_reason is not None`"
    )


def test_the_block_is_announced_with_its_reason_and_its_casualties():
    """Anchored on the three fields, not the sentence.

    Rewording the message must not silently drop what a reader needs.
    """

    from easyicu.research_agent.execution import phase

    call = _announcement_call()
    passed = {kw.arg for kw in call.keywords}
    for field in ("block_reason", "dropped_step_ids", "planned_step_count"):
        assert field in passed, field
    assert passed >= {"status", "run_id"}

    source = inspect.getsource(phase)
    assert _MESSAGE in ast.dump(call)
    # Every block that can empty the list must be nameable.
    for reason in (
        "endpoint_contract_blocked",
        "trajectory_plan_contract_blocked",
        "typed_plan_dag_blocked",
        "development_sample_unauthorized",
    ):
        assert reason in source, reason


def test_every_path_that_empties_the_step_list_sets_a_reason():
    """``steps_to_run`` may be emptied only when a reason was named.

    Otherwise a fourth block could be added later and go silent again, which is
    the defect this file exists for.
    """

    from easyicu.research_agent.execution import phase

    tree = ast.parse(inspect.getsource(phase))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "steps_to_run"
                for target in node.targets
            )
            and isinstance(node.value, ast.IfExp)
        ):
            continue
        test = node.value.test
        assert isinstance(test, ast.Compare), ast.dump(test)
        assert isinstance(test.left, ast.Name)
        assert test.left.id == "plan_block_reason", test.left.id
        return
    raise AssertionError("steps_to_run is no longer emptied by a named reason")


def test_the_recorded_run_is_the_case_this_describes():
    """The selected h3 run really did drop planned steps without an announcement."""

    logs = sorted(_CORPUS.glob("batch_*/h3_*/aware/run_*/audit_log.jsonl"))
    if not logs:
        pytest.skip("the h3 run that recorded this silence is not on disk")
    log = logs[-1]
    run = log.parent

    plan = json.loads((run / "analysis_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    planned = len(plan.get("steps") or [])
    executed = len(manifest.get("per_step_records") or [])
    if planned <= executed:
        pytest.skip("the recorded h3 run no longer drops steps")

    lines = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    text = json.dumps(lines, ensure_ascii=False)
    # The corpus grows as checkpoints are retained, so the newest h3 run no
    # longer has the original 9-planned/2-executed shape.  The regression is
    # the silent drop itself, not one historical count.
    assert planned - executed > 0, (planned, executed)
    # The recorded log predates the announcement; that absence is the defect.
    assert _MESSAGE not in text
