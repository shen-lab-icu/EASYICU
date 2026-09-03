"""The gate learned about the long trajectory tier; the guide never did.

``h3_trajectory_clustering`` planned 10 steps and executed none.  Its plan was
whole -- the executor's step list was emptied by a trajectory plan-contract
error, because the Planner wrote one combined
``04_cluster_trajectories_and_assess_stability`` step where the contract
requires a candidate owner and a separate later stability owner.

The Planner was never shown that contract.  ``trajectory_planner_contract_guide``
was already wired into the Planner prompt, but it opened only when some
``ResearchContext`` variable carried ``fixed_window_trajectory`` -- which is the
WIDE representation.  h3's trajectory is the long one: 19,067,154 rows over
94,442 stays, digested and bound as a typed run input, with no column names to
parse, so zero variables declared a window and the guide returned "".

``trajectory_plan_contract_applies`` already takes ``long_trajectory_bound`` for
exactly this reason, and its docstring already cites that same row count.  The
gate knew about the tier and the guide did not, so the plan was judged against
a contract nobody published.  Same defect family as the four fixed earlier the
same day: the host enforces a contract it never publishes.

Checked against the recorded contexts: h3 goes 0 -> 3,821 bytes; the
association task stays 0; and ``m3_sepsis_subphenotype`` also stays 0 because
its trajectory really is unbound -- that task is a separate, still-open defect
and this change deliberately does not paper over it.
"""

from __future__ import annotations

from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    FixedWindowTrajectoryMetadata,
    ResearchContext,
)
from easyicu.research_agent.trajectory.plan_contract import (
    _GENERAL_CLUSTER_STABILITY_METHOD,
    non_trajectory_clustering_stability_guide,
    trajectory_planner_contract_guide,
)

TRAJECTORY = "trajectory_clustering"


def _context(*, wide_window: bool) -> ResearchContext:
    variables = []
    if wide_window:
        variables.append(
            ConceptDescriptor(
                name="sofa2_h0_24",
                dtype="float",
                fixed_window_trajectory=FixedWindowTrajectoryMetadata(
                    family="sofa2",
                    window_start_hours=0.0,
                    window_end_hours=24.0,
                    window_width_hours=24.0,
                    representation_kind="continuous_window_summary",
                ),
            )
        )
    return ResearchContext(
        research_question="Do trajectory subgroups differ in outcome?",
        cohort=CohortDescriptor(
            cohort_name="probe",
            database="miiv",
            n_patients=1,
            n_stays=1,
            id_columns=["stay_id"],
        ),
        variables=variables,
    )


def test_a_bound_long_trajectory_gets_the_contract_it_is_judged_by():
    """h3's exact shape: right analysis type, no wide column, bound long tier."""

    context = _context(wide_window=False)

    unaware = trajectory_planner_contract_guide(
        context=context, analysis_type=TRAJECTORY
    )
    aware = trajectory_planner_contract_guide(
        context=context, analysis_type=TRAJECTORY, long_trajectory_bound=True
    )

    assert unaware == "", "the recorded behaviour: the guide said nothing"
    assert aware, "a run holding a bound long trajectory must be told the contract"
    # The two role owners whose absence refused h3's whole plan.
    assert "candidate selection" in aware
    assert "stability/freeze" in aware


def test_the_wide_representation_is_byte_identical_to_before():
    """The default must not change what wide-column runs are told."""

    context = _context(wide_window=True)

    assert trajectory_planner_contract_guide(
        context=context, analysis_type=TRAJECTORY
    ) == trajectory_planner_contract_guide(
        context=context, analysis_type=TRAJECTORY, long_trajectory_bound=False
    )


def test_a_bound_trajectory_on_another_analysis_type_stays_silent():
    """The flag widens one clause, not the whole gate.

    An association study binds cohort data too; being told a clustering
    contract would be noise in a prompt that is already budget-bound.
    """

    assert (
        trajectory_planner_contract_guide(
            context=_context(wide_window=False),
            analysis_type="association_study",
            long_trajectory_bound=True,
        )
        == ""
    )


def test_an_unbound_trajectory_still_says_nothing():
    """No wide column and no bound tier is the case that must stay silent.

    This is m3's real state -- its trajectory is genuinely unbound -- and the
    fix must not invent a contract for it.
    """

    assert (
        trajectory_planner_contract_guide(
            context=_context(wide_window=False),
            analysis_type=TRAJECTORY,
            long_trajectory_bound=False,
        )
        == ""
    )


def test_a_group_discovery_study_without_a_trajectory_is_told_how_to_declare_stability():
    """m3's exact shape: clusters one row per stay, still owes a stability audit.

    Its task requires a cluster-stability audit, and ``trajectory_stability_spec``
    was the only typed stability field it could see.  Declaring that without a
    validated fixed-window trajectory contract is refused, and the refusal
    emptied the whole plan -- 11 planned steps, none executed.
    """

    guide = non_trajectory_clustering_stability_guide(
        context=_context(wide_window=False),
        analysis_type=TRAJECTORY,
        long_trajectory_bound=False,
    )

    assert guide, "the study was asked for stability with no legal way to declare it"
    assert "trajectory_stability_spec` null" in guide
    assert _GENERAL_CLUSTER_STABILITY_METHOD in guide


def test_exactly_one_of_the_two_guides_ever_speaks():
    """They are the two halves of one predicate, so they must not overlap.

    Verified on the recorded contexts as well: h3 gets 3,821 bytes of trajectory
    contract and 0 of this; m3 gets 0 and 627; the association task gets 0 and 0.
    """

    for wide, bound in ((False, False), (False, True), (True, False), (True, True)):
        context = _context(wide_window=wide)
        spoke = [
            bool(
                trajectory_planner_contract_guide(
                    context=context,
                    analysis_type=TRAJECTORY,
                    long_trajectory_bound=bound,
                )
            ),
            bool(
                non_trajectory_clustering_stability_guide(
                    context=context,
                    analysis_type=TRAJECTORY,
                    long_trajectory_bound=bound,
                )
            ),
        ]
        assert sum(spoke) == 1, f"wide={wide} bound={bound} -> {spoke}"


def test_the_stability_guide_stays_out_of_other_families():
    assert (
        non_trajectory_clustering_stability_guide(
            context=_context(wide_window=False),
            analysis_type="association_study",
        )
        == ""
    )


def test_the_named_method_is_the_registry_key_not_a_literal_that_can_drift():
    """The guide may only name a method the Planner is actually allowed to use.

    It is also the registry's own statement that this path is agent-coded:
    ``runner`` is None, so no deterministic owner is being promised.
    """

    from easyicu.research_agent.planning.analysis_method_suite import (
        METHOD_SUITE_REGISTRY,
    )

    methods = {
        method.key: method
        for suite in METHOD_SUITE_REGISTRY
        for method in suite.methods
    }
    assert (
        _GENERAL_CLUSTER_STABILITY_METHOD in methods
    ), "the guide names a method the registry does not define"
    assert methods[_GENERAL_CLUSTER_STABILITY_METHOD].runner is None, (
        "this guide promises agent-coded stability; a runner here would make "
        "that promise false"
    )
