"""A plan that claims trajectory classes is reviewed on what it clusters.

Cross-sectional phenotype discovery and longitudinal trajectory clustering
share one analysis family, so a plan claims trajectories by declaring the
longitudinal trajectory action on its primary step.  When no owner builds a
per-timepoint representation, its classes summarize one value per ICU stay.
The review hands such a plan to the Host when the signed fixed-window owner
can model the plan's own coordinates; otherwise it states the limitation and
does not steer the plan to other variables.  The 9/25 H3 candidate clustered
0-24 h values under a trajectory claim and the review never said so.  The
fixtures are generic variables, not that study.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pytest
from pydantic import BaseModel

from easyicu.research_agent.contracts.trajectory_design import (
    TRAJECTORY_PRIMARY_ACTION,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    plan_revision_blocker_codes,
    remediation_route_for_finding,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from easyicu.research_agent.trajectory.contract import (
    infer_fixed_window_trajectory_metadata,
)

_QUESTION = "Which organ-dysfunction trajectory classes emerge early in the ICU stay?"
_ROLES = {
    "stay_id": VariableRole.ID,
    "age": VariableRole.DEMOGRAPHIC,
    "sofa2_resp": VariableRole.ORDINAL_SCORE,
    "sofa2_cardio": VariableRole.ORDINAL_SCORE,
    # An availability receipt shares the prefix but is not a measurement.
    "sofa2_resp_available": VariableRole.META,
    "sofa_resp": VariableRole.ORDINAL_SCORE,
    "sofa_cardio": VariableRole.ORDINAL_SCORE,
    "lact": VariableRole.LAB,
    "death": VariableRole.OUTCOME,
}


def _context(names: tuple[str, ...] = tuple(_ROLES)) -> ResearchContext:
    return ResearchContext(
        research_question=_QUESTION,
        cohort=CohortDescriptor(
            cohort_name="trajectory_fixture", database="synthetic", n_stays=0,
            id_columns=["stay_id"], outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name=name, role=_ROLES[name], dtype="float64")
            for name in names
        ],
        target_outcome="death",
    )


def _step(inputs: tuple[str, ...], **update: object) -> AnalysisStep:
    return AnalysisStep(
        step_id="trajectory_solution",
        planned_analysis_role="primary",
        intent="Cluster the prespecified coordinates into trajectory classes.",
        inputs=["stay_id", *inputs, "artifact:analysis_cohort"],
        expected_outputs=["table:phenotype_assignments"],
        method="prespecified_trajectory_feature_clustering",
        scientific_action_id=TRAJECTORY_PRIMARY_ACTION,
    ).model_copy(update=update)


def _plan(*steps: AnalysisStep) -> AnalysisPlan:
    return AnalysisPlan(
        research_question=_QUESTION, analysis_type="trajectory_clustering",
        steps=list(steps),
    )


def _review(context: ResearchContext, plan: AnalysisPlan):
    review = build_plan_scientific_review(context=context, plan=plan, literature=None)
    return review, [item for item in review.findings if item.code.startswith("TRAJECTORY_")]


def test_coordinates_the_signed_owner_models_are_handed_to_the_host() -> None:
    review, findings = _review(
        _context(), _plan(_step(("sofa2_resp", "sofa2_cardio", "lact")))
    )

    [finding] = findings
    assert finding.code == "TRAJECTORY_LONGITUDINAL_OWNER_NOT_SEALED"
    assert finding.severity == "blocker"
    assert remediation_route_for_finding(finding) == "runtime_capability"
    # A runtime gap: another Planner turn cannot seal the owner.
    assert finding.code in plan_revision_blocker_codes(review.findings)
    assert not review.approval_allowed
    facts = review.facts["trajectory_representation"]
    assert facts["longitudinal_owner"] is None
    assert facts["executable"] is True
    assert facts["proposed_coordinates"] == ["sofa2_resp", "sofa2_cardio", "lact"]


def test_other_coordinates_keep_the_plan_and_state_the_limitation() -> None:
    review, findings = _review(_context(), _plan(_step(("sofa_resp", "sofa_cardio"))))

    [finding] = findings
    assert finding.code == "TRAJECTORY_REPRESENTATION_NOT_LONGITUDINAL"
    assert finding.severity == "major"
    assert remediation_route_for_finding(finding) == "agent_plan_revision"
    assert finding.code not in plan_revision_blocker_codes(review.findings)
    # The Agent learns which components this study offers, and is told not
    # to swap another score version in for the one the question names.
    assert "this study provides sofa2_cardio, sofa2_resp" in finding.message
    assert "Do not substitute another variable or score version" in finding.remediation
    facts = review.facts["trajectory_representation"]
    assert facts["executable"] is False
    # Only time-varying SOFA-2 variables are offered; the availability
    # receipt shares the prefix but is not a measurement.
    assert facts["study_eligibility_coordinates"] == ["sofa2_cardio", "sofa2_resp"]


def test_a_question_about_other_measurements_is_not_pushed_to_sofa2() -> None:
    context = _context(("stay_id", "lact", "death"))
    _review_, findings = _review(context, _plan(_step(("lact",))))

    [finding] = findings
    assert finding.severity == "major"
    assert "fewer than two time-varying coordinates" in finding.message
    assert "this study's variables include none" in finding.message


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ("death", "it clusters on outcomes (death)"),
        ("age", "it clusters on one-per-stay variables (age)"),
    ],
)
def test_outcome_or_one_per_stay_inputs_are_never_compiled(extra: str, reason: str) -> None:
    review, findings = _review(
        _context(), _plan(_step(("sofa2_resp", "sofa2_cardio", extra)))
    )

    [finding] = findings
    assert finding.code == "TRAJECTORY_REPRESENTATION_NOT_LONGITUDINAL"
    assert reason in finding.message
    facts = review.facts["trajectory_representation"]
    assert facts["executable"] is False
    # The remaining coordinates are not a proposal on their own: the plan
    # clustered on more than them, so the Host must not compile them.
    assert facts["proposed_coordinates"] == ["sofa2_resp", "sofa2_cardio"]


def test_cross_sectional_phenotype_discovery_is_not_a_trajectory_claim() -> None:
    step = _step(
        ("sofa2_resp", "sofa2_cardio"),
        method="cross_sectional_phenotyping",
        scientific_action_id="phenotyping.cluster_solution",
        phenotyping_feature_columns=["sofa2_resp", "sofa2_cardio"],
    )
    review, findings = _review(_context(), _plan(step))

    assert findings == []
    assert review.facts["trajectory_representation"] is None


class _BoundTrajectory(BaseModel):
    trajectory_file: str = "universe_trajectory.parquet"


class _BoundInputs(BaseModel):
    trajectory: Optional[_BoundTrajectory] = None


class _BoundContext(ResearchContext):
    """A context whose run bound a verified long trajectory."""

    materialized_inputs: Optional[_BoundInputs] = None


def _signed_plan() -> AnalysisPlan:
    representation = AnalysisStep(
        step_id="representation", planned_analysis_role="auxiliary",
        intent="Build the sealed representation.",
        method="signed_fixed_window_trajectory_representation",
        expected_outputs=["artifact:trajectory_representation"],
    )
    # The bound suite's primary declares its owner method, not the action.
    candidates = _step(()).model_copy(
        update={
            "inputs": ["artifact:trajectory_representation"],
            "method": "observed_data_diagonal_gaussian_mixture_candidate_selection",
            "scientific_action_id": None,
        }
    )
    return _plan(representation, candidates)


def _fixed_window_case() -> tuple[ResearchContext, AnalysisPlan]:
    columns = ("sofa2_resp_h0_24", "sofa2_resp_h24_48")
    variables = [
        ConceptDescriptor(
            name=name, role=VariableRole.ORDINAL_SCORE, dtype="float64",
            fixed_window_trajectory=infer_fixed_window_trajectory_metadata(
                column_name=name, values=pd.Series([0.0, 1.0]), source_scale="ordinal",
            ),
        )
        for name in columns
    ]
    context = _context().model_copy(
        update={"variables": [*_context().variables, *variables]}
    )
    return context, _plan(_step(columns, method="trajectory_feature_clustering"))


@pytest.mark.parametrize(
    "case", ["signed_fixed_window_suite", "fixed_window_columns", "bound_long_trajectory"]
)
def test_a_longitudinal_owner_is_left_to_its_own_gates(case: str) -> None:
    if case == "signed_fixed_window_suite":
        context, plan = _context(), _signed_plan()
    elif case == "fixed_window_columns":
        context, plan = _fixed_window_case()
    else:
        context = _BoundContext(
            **_context().model_dump(),
            materialized_inputs=_BoundInputs(trajectory=_BoundTrajectory()),
        )
        plan = _plan(_step(("sofa2_resp", "sofa2_cardio")))

    review, findings = _review(context, plan)

    assert findings == []
    assert review.facts["trajectory_representation"]["longitudinal_owner"] == case
