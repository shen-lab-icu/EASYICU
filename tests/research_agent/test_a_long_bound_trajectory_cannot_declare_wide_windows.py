"""A rule no long-tier plan could ever satisfy.

``trajectory_window_family_not_resolved`` is satisfied ONLY by a declared input
whose variable carries ``fixed_window_trajectory`` -- metadata inferred by
parsing a WIDE column name (``<family>_h<start>_<end>``). A run whose trajectory
is bound in the LONG tier has no such column: ``plan_contract``'s own
applicability docstring records that a run holding 19,067,154 verified
trajectory rows "still presented zero trajectory variables here".

So the rule applied to long-bound runs and could not be satisfied by any plan.
h3 has never passed step 01 in any recorded run, and this is why. It is the
THIRD time the tier flag reached one decision in that file and not another --
the applicability docstring already records two.

The waiver is narrow: the plan must still declare the window manifest, which is
where a long-bound run's windows are recorded and where they are verified after
the representation step actually runs.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.schema import AnalysisPlan, ResearchContext
from easyicu.research_agent.trajectory.plan_contract import (
    evaluate_trajectory_plan_dag,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _recorded_h3(*, declaring_the_manifest: bool = True):
    """h3's real recorded plan and context -- the artifacts that were refused.

    Selection is EXPLICIT, not "the newest": recorded h3 plans differ in whether
    the representation step declares the window manifest, because until this
    branch the planner was never told to. The waiver tests need one that does;
    picking by recency silently swapped the fixture and turned them red.
    """

    for plan_path in sorted(
        _CORPUS.glob("batch_*/h3_*/aware/run_*/analysis_plan.json"), reverse=True
    ):
        context_path = plan_path.parent / "research_context.json"
        if not context_path.exists():
            continue
        raw = json.loads(context_path.read_text())
        # The bound tier is recorded beside the context, not inside the model.
        raw.pop("materialized_inputs", None)
        try:
            plan = AnalysisPlan.model_validate(json.loads(plan_path.read_text()))
            context = ResearchContext.model_validate(raw)
        except Exception:  # noqa: BLE001 - an older schema is not the subject
            continue
        declares = any(
            "window_manifest" in str(product).lower()
            for step in plan.steps
            for product in (step.expected_outputs or [])
        )
        if declares is declaring_the_manifest:
            return plan, context
    return None


def _reasons(plan, context, *, long_bound):
    evaluation = evaluate_trajectory_plan_dag(
        plan=plan, context=context, long_trajectory_bound=long_bound
    )
    return [
        str(finding.message)
        for finding in evaluation.findings
        if getattr(finding, "severity", "error") == "error"
    ]


def test_the_recorded_plan_is_no_longer_refused_for_windows_it_cannot_declare():
    recorded = _recorded_h3()
    if recorded is None:
        pytest.skip("no recorded h3 plan is mounted")
    plan, context = recorded

    # The premise: the context really does expose zero wide-window variables.
    assert not any(
        variable.fixed_window_trajectory is not None
        for variable in (context.variables or [])
    )
    # And the plan really does declare the window manifest.
    declared = {
        str(product).lower()
        for step in plan.steps
        for product in (step.expected_outputs or [])
    }
    assert any("window_manifest" in product for product in declared)

    messages = _reasons(plan, context, long_bound=True)
    assert not any("at least two fixed" in message for message in messages), messages


def test_a_long_bound_plan_without_the_manifest_is_still_refused():
    """The waiver is not unconditional -- it defers to the manifest, not away."""

    recorded = _recorded_h3()
    if recorded is None:
        pytest.skip("no recorded h3 plan is mounted")
    plan, context = recorded

    stripped = plan.model_copy(
        update={
            "steps": [
                step.model_copy(
                    update={
                        "expected_outputs": [
                            product
                            for product in (step.expected_outputs or [])
                            if "window_manifest" not in str(product).lower()
                        ],
                        "inputs": [
                            product
                            for product in (step.inputs or [])
                            if "window_manifest" not in str(product).lower()
                        ],
                    }
                )
                for step in plan.steps
            ]
        }
    )

    messages = _reasons(stripped, context, long_bound=True)
    assert any("at least two fixed" in message for message in messages), messages


def test_the_waiver_is_conditioned_on_the_tier_not_granted_to_everyone():
    """Both cases above are long-bound, so read the condition itself.

    A mutation dropping ``long_trajectory_bound`` from the guard would waive the
    rule for WIDE runs too -- where it is satisfiable and does real work -- and
    two long-tier tests cannot see that.
    """

    import ast
    import inspect

    from easyicu.research_agent.trajectory import plan_contract

    tree = ast.parse(inspect.getsource(plan_contract))
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "long_tier_defers_windows_to_the_manifest"
            for target in node.targets
        )
    ]
    assert guards, "the long-tier waiver is gone"
    for guard in guards:
        assert isinstance(guard.value, ast.BoolOp) and isinstance(
            guard.value.op, ast.And
        ), "the waiver must REQUIRE the tier, not merely mention it"
        names = {
            node.id
            for node in ast.walk(guard.value.values[0])
            if isinstance(node, ast.Name)
        }
        assert "long_trajectory_bound" in names, ast.dump(guard.value.values[0])


def test_the_recorded_plan_now_clears_the_whole_trajectory_contract():
    """h3's real plan, end to end: every wide-topology assumption removed.

    The refusals came in a chain, each presuming the representation either reads
    wide window columns or imports a panel someone else built:
      * it must consume a window manifest from that panel's producer;
      * that manifest must resolve to one upstream producer;
      * it must select two fixed windows from one declared family.
    A long-bound representation reads the bound long trajectory and EMITS the
    manifest itself, so it is the window source -- there is no upstream to
    resolve and no lineage to import.
    """

    recorded = _recorded_h3()
    if recorded is None:
        pytest.skip("no recorded h3 plan is mounted")
    plan, context = recorded

    assert _reasons(plan, context, long_bound=True) == []


def test_a_wide_run_still_needs_its_manifest_lineage():
    """The waiver is keyed to the tier; wide runs keep every rule."""

    import ast
    import inspect

    from easyicu.research_agent.trajectory import plan_contract

    tree = ast.parse(inspect.getsource(plan_contract))
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "representation_emits_the_manifest"
            for target in node.targets
        )
    ]
    assert guards, "the long-tier window-source waiver is gone"
    for guard in guards:
        assert isinstance(guard.value, ast.BoolOp) and isinstance(
            guard.value.op, ast.And
        )
        names = {
            node.id
            for node in ast.walk(guard.value.values[0])
            if isinstance(node, ast.Name)
        }
        assert "long_trajectory_bound" in names


def test_the_guide_tells_a_long_bound_planner_what_it_can_actually_do():
    """The guide described work a long-bound run cannot perform.

    MEASURED: rendered with ``long_trajectory_bound=True`` it mentioned the long
    tier ZERO times -- no charttime, no value_num, no stay_id -- and instructed
    the planner to "list the raw fixed-window columns in its inputs". Those
    columns do not exist in a long-bound run (h3's cohort: 0 of 70). The
    contract then refused the plan for not declaring windows it was never told
    how to declare.

    Confirmed live: verify35's planner produced a representation step whose
    outputs carry no ``manifest:trajectory_window_manifest``, so the tier
    waivers correctly did not apply and both rules fired again.
    """

    from easyicu.research_agent.schema import ConceptDescriptor
    from easyicu.research_agent.trajectory.plan_contract import (
        trajectory_planner_contract_guide,
    )

    recorded = _recorded_h3()
    if recorded is None:
        pytest.skip("no recorded h3 context is mounted")
    _plan, context = recorded

    long_guide = trajectory_planner_contract_guide(
        context=context,
        analysis_type="trajectory_clustering",
        long_trajectory_bound=True,
    )
    assert long_guide
    # It names the tier the run actually has...
    assert "bound as the typed LONG input" in long_guide
    # ...tells it to declare the manifest it is the source of...
    assert "MUST declare `manifest:trajectory_window_manifest`" in long_guide
    # ...and stops describing columns the run does not have.
    assert "raw fixed-window columns" not in long_guide


def test_a_wide_run_is_still_told_the_wide_contract():
    """The wide instructions must survive verbatim for the runs they fit."""

    from easyicu.research_agent.schema import (
        ConceptDescriptor,
        FixedWindowTrajectoryMetadata,
    )
    from easyicu.research_agent.trajectory.plan_contract import (
        trajectory_planner_contract_guide,
    )

    recorded = _recorded_h3()
    if recorded is None:
        pytest.skip("no recorded h3 context is mounted")
    _plan, context = recorded

    wide = context.model_copy(
        update={
            "variables": [
                ConceptDescriptor(
                    name="score_h0_12",
                    dtype="float64",
                    fixed_window_trajectory=FixedWindowTrajectoryMetadata(
                        family="score",
                        window_start_hours=0.0,
                        window_end_hours=12.0,
                        window_width_hours=12.0,
                        representation_kind="discrete_window_state",
                    ),
                )
            ]
        }
    )
    wide_guide = trajectory_planner_contract_guide(
        context=wide,
        analysis_type="trajectory_clustering",
        long_trajectory_bound=False,
    )

    assert wide_guide
    assert "raw fixed-window columns" in wide_guide
    assert "bound as the typed LONG input" not in wide_guide
