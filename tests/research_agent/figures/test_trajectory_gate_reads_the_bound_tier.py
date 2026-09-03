"""A trajectory gate must read the trajectory, not the column names.

H3 (full03, run_20260731T105349) planned 11 steps and executed zero, refused
with "no validated fixed-window trajectory contract" -- while its own
research_context.json bound cohort_trajectory.parquet: 19,067,154 rows of
``stay_id, charttime, concept, value_num, value_str`` with a verified
authority_ref.

The trajectory reaches this host in two representations.  The wide one is
inferred by parsing a column name (``<family>_h<start>_<end>``) and is the only
one ``ResearchContext`` can carry.  The long one is materialised, digested and
bound as a typed run input, and has no column names to parse.  The gate only
knew how to look for the first, so a run holding the second presented zero
trajectory variables and lost its whole plan.

M3 is the control: it is cross-sectional clustering
(benchmarks/figure2_canonical9/resource_preflight.py:69), its
materialized_inputs.trajectory is None, and no file exists on disk.  It must
stay refused -- a run with no trajectory cannot support a trajectory design,
and the flag must not become a way to say otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    FixedWindowTrajectoryMetadata,
    ResearchContext,
)
from easyicu.research_agent.trajectory.plan_contract import (
    trajectory_plan_contract_applies,
)


def _context(*, wide_trajectory: bool) -> ResearchContext:
    variables = []
    if wide_trajectory:
        variables.append(
            ConceptDescriptor(
                name="sofa2_h0_6",
                description="SOFA-2 in the first six hours.",
                role="ordinal_score",
                dtype="float64",
                fixed_window_trajectory=FixedWindowTrajectoryMetadata(
                    family="sofa2",
                    window_start_hours=0.0,
                    window_end_hours=6.0,
                    window_width_hours=6.0,
                    representation_kind="discrete_window_state",
                ),
            )
        )
    return ResearchContext(
        research_question="Do trajectory subgroups differ?",
        cohort=CohortDescriptor(
            cohort_name="trajectory-gate",
            database="test",
            n_patients=20,
            n_stays=20,
            id_columns=["stay_id"],
        ),
        variables=variables,
    )


def _plan(analysis_type: str) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Do trajectory subgroups differ?",
        analysis_type=analysis_type,
        steps=[
            AnalysisStep(
                step_id="01_define_analysis_cohort",
                intent="x",
                method="cohort_definition",
                expected_outputs=["artifact:analysis_cohort", "table:cohort_flow"],
                inputs=[],
            )
        ],
    )


# --- the case that was losing eleven steps ----------------------------------


def test_a_bound_long_trajectory_is_enough_without_any_wide_column():
    """H3's shape: right analysis type, zero wide columns, real trajectory."""

    assert trajectory_plan_contract_applies(
        plan=_plan("trajectory_clustering"),
        context=_context(wide_trajectory=False),
        long_trajectory_bound=True,
    )


def test_without_the_flag_that_same_run_is_still_refused():
    """The defect, pinned: nothing about the plan or context changed."""

    assert not trajectory_plan_contract_applies(
        plan=_plan("trajectory_clustering"),
        context=_context(wide_trajectory=False),
    )


# --- the control: no trajectory means no trajectory design ------------------


def test_no_trajectory_of_either_kind_stays_refused():
    """M3's shape. A run with no trajectory cannot support a trajectory plan."""

    assert not trajectory_plan_contract_applies(
        plan=_plan("trajectory_clustering"),
        context=_context(wide_trajectory=False),
        long_trajectory_bound=False,
    )


@pytest.mark.parametrize("bound", [False, True])
def test_the_flag_never_promotes_a_non_trajectory_analysis_type(bound):
    """The binding says what data exists, never what study this is."""

    assert not trajectory_plan_contract_applies(
        plan=_plan("association_study"),
        context=_context(wide_trajectory=True),
        long_trajectory_bound=bound,
    )


# --- the previous behaviour is the default ----------------------------------


def test_a_wide_column_run_still_applies_with_no_flag_passed():
    """Every caller that cannot see the bound tier keeps today's answer."""

    assert trajectory_plan_contract_applies(
        plan=_plan("trajectory_clustering"),
        context=_context(wide_trajectory=True),
    )


def test_the_default_is_the_old_behaviour_not_the_new_one():
    with_flag = trajectory_plan_contract_applies(
        plan=_plan("trajectory_clustering"),
        context=_context(wide_trajectory=False),
        long_trajectory_bound=True,
    )
    without_flag = trajectory_plan_contract_applies(
        plan=_plan("trajectory_clustering"),
        context=_context(wide_trajectory=False),
    )

    assert with_flag is True
    assert without_flag is False


# --- the execution phase really passes it -----------------------------------


def test_the_execution_phase_derives_the_flag_from_the_verified_authority():
    """Not from a path, an env var, or the presence of a file on disk.

    ``trajectory_authority_sha256`` is None unless a typed authority ref or a
    legacy capsule receipt backs the binding, so an unverified file cannot open
    the gate.
    """

    import ast
    import inspect

    from easyicu.research_agent.execution import phase

    source = inspect.getsource(phase.run_execute_phase)
    tree = ast.parse(source.lstrip())
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "long_trajectory_bound"
            for target in node.targets
        )
    ]

    assert assignments, "run_execute_phase no longer derives long_trajectory_bound"
    rendered = ast.dump(assignments[0])
    assert "trajectory_authority_sha256" in rendered


def test_the_execution_phase_passes_the_flag_to_the_gate():
    import ast
    import inspect

    from easyicu.research_agent.execution import phase

    tree = ast.parse(inspect.getsource(phase.run_execute_phase).lstrip())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "trajectory_plan_contract_applies"
    ]

    assert calls, "the execution phase no longer calls the gate"
    assert any(
        keyword.arg == "long_trajectory_bound"
        for call in calls
        for keyword in call.keywords
    ), "the execution phase calls the gate without the bound-tier flag"


# --- the miss this file did not catch the first time -------------------------


def _stability_spec_plan() -> AnalysisPlan:
    """H3's shape: a stability spec attached, and no wide column anywhere."""

    from easyicu.research_agent.schema import TrajectoryStabilitySpec

    return AnalysisPlan(
        research_question="Do trajectory subgroups differ?",
        analysis_type="trajectory_clustering",
        steps=[
            AnalysisStep(
                step_id="04_trajectory_clustering_and_stability",
                intent="Cluster trajectories and assess stability.",
                method="trajectory_clustering",
                expected_outputs=["table:cluster_assignments"],
                inputs=[],
                trajectory_stability_spec=TrajectoryStabilitySpec(
                    n_resamples=10,
                    sample_fraction=0.8,
                    base_seed=7,
                    minimum_mean_stability=0.7,
                ),
            )
        ],
    )


def test_the_evaluator_stops_raising_the_refusal_h3_actually_got():
    """The regression that shipped: the outer guard had the flag, this did not.

    canary14 ran with the gate change in place and H3 still ended on
    "A trajectory stability spec is attached to a plan that has no validated
    fixed-window trajectory contract" -- because evaluate_trajectory_plan_dag
    re-asks applicability itself and was still using the default.
    """

    from easyicu.research_agent.trajectory.plan_contract import (
        evaluate_trajectory_plan_dag,
    )

    plan = _stability_spec_plan()
    context = _context(wide_trajectory=False)

    def kinds(**kwargs):
        return {
            (finding.detail or {}).get("kind")
            for finding in evaluate_trajectory_plan_dag(
                plan=plan, context=context, **kwargs
            ).findings
        }

    without = kinds()
    with_binding = kinds(long_trajectory_bound=True)

    assert "trajectory_stability_spec_without_trajectory_plan" in without
    assert "trajectory_stability_spec_without_trajectory_plan" not in with_binding


def test_the_bound_tier_moves_the_plan_into_real_role_validation():
    """Not just a different message -- a different stage of the contract.

    Without the binding the evaluator stops at the one stability-spec refusal
    and never inspects roles.  With it, the plan reaches the role/DAG contract
    and comes back with findings a replan can act on.  (``applies`` is not the
    property to assert here: in the not-applicable branch it is set from
    whether any spec step existed to complain about, not from the contract.)
    """

    from easyicu.research_agent.trajectory.plan_contract import (
        evaluate_trajectory_plan_dag,
    )

    plan = _stability_spec_plan()
    context = _context(wide_trajectory=False)

    without = evaluate_trajectory_plan_dag(plan=plan, context=context)
    with_binding = evaluate_trajectory_plan_dag(
        plan=plan, context=context, long_trajectory_bound=True
    )

    without_kinds = {(f.detail or {}).get("kind") for f in without.findings}
    with_kinds = {(f.detail or {}).get("kind") for f in with_binding.findings}

    assert without_kinds == {"trajectory_stability_spec_without_trajectory_plan"}
    assert "trajectory_role_missing" in with_kinds


#: Every entry point that re-asks whether the trajectory contract applies.
#: Missing one is not a cosmetic gap: it is the whole defect, twice over.
_GATE_ENTRY_POINTS = (
    "trajectory_plan_contract_applies",
    "trajectory_plan_dag_findings",
    "evaluate_trajectory_plan_dag",
    "trajectory_bundle_findings",
    "resolve_trajectory_bundle_plan_authority",
)


def _calls_in(source, callee):
    import ast

    tree = ast.parse(source.lstrip())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == callee)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == callee)
        )
    ]


def test_every_gate_call_in_the_execution_phase_carries_the_flag():
    """Enumerated, not enumerated-by-me.

    The first fix threaded one of four call sites and shipped; the second
    threaded two more and shipped; H3 was refused by the fourth both times.
    This walks run_execute_phase for EVERY entry point that re-asks the
    question, so the next missed one fails here instead of in a paid run.
    """

    import inspect

    from easyicu.research_agent.execution import phase

    source = inspect.getsource(phase.run_execute_phase)
    seen = 0
    for callee in _GATE_ENTRY_POINTS:
        for call in _calls_in(source, callee):
            seen += 1
            assert any(
                keyword.arg == "long_trajectory_bound" for keyword in call.keywords
            ), f"a call to {callee} in run_execute_phase drops the bound-tier flag"
    assert seen >= 4, f"expected the phase to consult the gate; found {seen} calls"


def test_the_hops_below_the_phase_carry_it_too():
    import inspect

    from easyicu.research_agent.trajectory import bundle, plan_contract

    hops = {
        "resolve_trajectory_bundle_plan_authority": inspect.getsource(
            bundle.trajectory_bundle_findings
        ),
        "evaluate_trajectory_plan_dag": "\n".join(
            (
                inspect.getsource(bundle.resolve_trajectory_bundle_plan_authority),
                inspect.getsource(plan_contract.trajectory_plan_dag_findings),
            )
        ),
    }
    for callee, source in hops.items():
        calls = _calls_in(source, callee)
        assert calls, f"the chain no longer calls {callee}"
        assert all(
            any(keyword.arg == "long_trajectory_bound" for keyword in call.keywords)
            for call in calls
        ), f"a call to {callee} drops the bound-tier flag"


def test_the_plan_phase_answers_the_same_question_before_the_planner_is_done():
    """The findings have to arrive while a revision can still use them.

    H3 reached revision 2 and only met the role contract during execution, so
    the replan it did get was driven by something else. The plan phase now
    derives the answer from the same predicate, and both gate calls inside it
    carry it.
    """

    import inspect
    import textwrap

    from easyicu.research_agent import pipeline

    source = "\n".join(
        (
            textwrap.dedent(
                inspect.getsource(pipeline.ResearchAgentPipeline._run_plan_phase)
            ),
            textwrap.dedent(
                inspect.getsource(
                    pipeline.ResearchAgentPipeline._validate_and_persist_plan
                )
            ),
        )
    )
    assert "long_trajectory_is_bound(trajectory_binding)" in source
    for callee in ("trajectory_plan_dag_findings", "_enforce_advanced_plan_contract"):
        calls = _calls_in(source, callee)
        assert calls, f"the plan phase no longer calls {callee}"
        assert all(
            any(keyword.arg == "long_trajectory_bound" for keyword in call.keywords)
            for call in calls
        ), f"the plan phase calls {callee} without the bound-tier flag"


def test_both_phases_ask_one_predicate_not_two_spellings_of_it():
    """Two spellings is how a run plans against data it may not use."""

    import inspect

    from easyicu.research_agent.authority.execution_input import (
        ExecutionInputAuthorityState,
    )

    source = inspect.getsource(
        ExecutionInputAuthorityState.trajectory_authority_sha256.fget
    )

    assert "verified_authority_sha256" in source
    assert "authority_ref.sha256" not in source


def test_a_staged_file_without_a_verified_authority_is_not_bound():
    """The answer is about a verified binding, never about a path existing."""

    from easyicu.research_agent.intake.materialized_trajectory import (
        StagedTrajectoryBinding,
        long_trajectory_is_bound,
    )

    unverified = StagedTrajectoryBinding(
        path=Path("cohort_trajectory.parquet"),
        sha256="0" * 64,
        size=1,
    )

    assert unverified.verified_authority_sha256 is None
    assert not long_trajectory_is_bound(unverified)
    assert not long_trajectory_is_bound(None)


def test_the_flag_is_bound_before_the_first_gate_call_that_uses_it():
    """It is derived beside the authority; every consult below must follow it."""

    import ast
    import inspect

    from easyicu.research_agent.execution import phase

    source = inspect.getsource(phase.run_execute_phase)
    tree = ast.parse(source.lstrip())
    bound_at = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "long_trajectory_bound"
            for target in node.targets
        )
    )
    used_at = [
        keyword.value.lineno
        for callee in _GATE_ENTRY_POINTS
        for call in _calls_in(source, callee)
        for keyword in call.keywords
        if keyword.arg == "long_trajectory_bound"
    ]

    assert used_at
    assert bound_at < min(used_at)
