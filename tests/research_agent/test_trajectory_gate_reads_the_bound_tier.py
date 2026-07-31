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
