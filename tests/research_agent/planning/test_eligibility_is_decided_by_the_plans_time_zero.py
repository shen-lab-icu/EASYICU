"""Cohort eligibility is decided by the plan's time zero, on every planning path.

The family-spec request refuses an export whose concept population or typed
minimum ICU stay decides membership after the plan's time zero.  A progressive
plan carries no typed time zero, so the scientific review applies the same rule
to the plan it is shown: a trajectory plan's time zero is its window's end;
otherwise the study's declared landmark, the signed runtime authority's, or the
end of the host-bound feature window.  Fixtures are synthetic and vary the
concept population so that no rule keys on one condition.
"""

from __future__ import annotations

import ast
import inspect
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.contracts.trajectory_design import TRAJECTORY_PRIMARY_ACTION
from easyicu.research_agent.planning import scientific_review
from easyicu.research_agent.planning.cohort_eligibility import (
    EligibilityAfterTimeZero,
    eligibility_after_time_zero,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    cohort_eligibility_findings,
    plan_time_zero_hours,
)
from easyicu.research_agent.research_context.concept_population import ConceptCohortWindow
from easyicu.research_agent.research_context.minimum_stay import minimum_icu_stay_hours
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ResearchContext, TimeWindow

from .scientific_review_fixtures import _context, _literature, _plan

_CONCEPTS = ["sep3", "aki", "vent_ind"]
_TRAJECTORY_QUESTION = (
    "Do organ-support trajectories over the first 24 hours from ICU admission "
    "cluster into distinct subgroups?"
)


def _study(
    *,
    concept: tuple[str, float] | None = None,
    minimum_icu_hours: float | None = None,
    landmark_hours: float | None = None,
    question: str | None = None,
    windows: tuple[float, ...] = (),
    constraints: dict | None = None,
) -> ResearchContext:
    base = _context()
    payload: dict = dict(constraints or {})
    if concept is not None:
        payload["concept_cohort_window"] = {
            "definition": concept[0],
            "window_end_hours": concept[1],
        }
    if minimum_icu_hours is not None:
        payload.setdefault("cohort", {})["min_icu_los_hours"] = minimum_icu_hours
    return base.model_copy(
        update={
            "research_question": question or base.research_question,
            "time_windows": [
                TimeWindow(name=f"window_{index}", end_hours=end)
                for index, end in enumerate(windows)
            ],
            "user_preferences": base.user_preferences.model_copy(
                update={
                    "landmark_hours": landmark_hours,
                    "data_constraints": json.dumps(payload) if payload else None,
                }
            ),
        }
    )


def _trajectory_plan() -> AnalysisPlan:
    step = AnalysisStep(
        step_id="trajectory_solution",
        planned_analysis_role="primary",
        intent="Cluster the prespecified coordinates into trajectory classes.",
        inputs=["stay_id", "artifact:analysis_cohort"],
        expected_outputs=["table:phenotype_assignments"],
        method="prespecified_trajectory_feature_clustering",
        scientific_action_id=TRAJECTORY_PRIMARY_ACTION,
    )
    return AnalysisPlan(
        research_question=_TRAJECTORY_QUESTION, analysis_type="trajectory_clustering", steps=[step]
    )


def _trajectory(end: float | None, *, executable: bool = True) -> dict:
    window = {"executable": executable}
    if end is not None:
        window["window_end_hours"] = end
    return {"trajectory_window": window}


# The shared rule ---------------------------------------------------------


@pytest.mark.parametrize("definition", _CONCEPTS)
@pytest.mark.parametrize(("window_end", "refused"), [(72.0, True), (24.0, False), (12.0, False)])
def test_a_concept_window_must_end_by_time_zero(
    definition: str, window_end: float, refused: bool
) -> None:
    found = eligibility_after_time_zero(
        time_zero_hours=24.0,
        minimum_icu_hours=None,
        concept_population=ConceptCohortWindow(definition=definition, window_end_hours=window_end),
    )

    assert bool(found) is refused
    if refused:
        assert found == (
            EligibilityAfterTimeZero(
                criterion="concept_population",
                decided_by_hours=window_end,
                time_zero_hours=24.0,
                definition=definition,
            ),
        )


def test_a_minimum_stay_is_compared_first_and_at_time_zero_passes() -> None:
    concept = ConceptCohortWindow(definition="aki", window_end_hours=48.0)

    found = eligibility_after_time_zero(
        time_zero_hours=24.0, minimum_icu_hours=36.0, concept_population=concept
    )

    assert [item.criterion for item in found] == ["minimum_icu_stay", "concept_population"]
    assert eligibility_after_time_zero(
        time_zero_hours=24.0, minimum_icu_hours=24.0, concept_population=None
    ) == ()
    assert eligibility_after_time_zero(
        time_zero_hours=None, minimum_icu_hours=96.0, concept_population=concept
    ) == ()


def test_the_messages_are_the_family_spec_refusals_word_for_word() -> None:
    minimum, concept = eligibility_after_time_zero(
        time_zero_hours=24.0,
        minimum_icu_hours=48.0,
        concept_population=ConceptCohortWindow(definition="sep3", window_end_hours=72.0),
    )

    assert minimum.message() == (
        "a minimum ICU stay of 48 h ends after the plan's time zero at 24 h after ICU "
        "admission; eligibility would depend on survival after time zero"
    )
    assert concept.message() == (
        "the sep3 population admits a stay on a positive row up to 72 h after ICU "
        "admission, after the plan's time zero at 24 h; eligibility would depend on what "
        "happens after time zero, so the study's cohort window must end by 24 h"
    )


@pytest.mark.parametrize(
    ("value", "hours"),
    [(48, 48.0), (12.5, 12.5), ("36", 36.0), (0, None), (-4, None), (True, None), ("long", None)],
)
def test_the_typed_minimum_stay_reads_as_family_spec_reads_it(value, hours) -> None:
    assert minimum_icu_stay_hours(_study(minimum_icu_hours=value)) == hours


def test_a_study_without_a_minimum_stay_has_none() -> None:
    assert minimum_icu_stay_hours(_study()) is None
    assert minimum_icu_stay_hours(_study(constraints={"cohort": {"age_min": 18}})) is None


# The plan's time zero ----------------------------------------------------


def test_a_trajectory_plans_time_zero_is_its_window_end() -> None:
    context = _study(landmark_hours=48.0, windows=(72.0,))

    assert plan_time_zero_hours(context, _trajectory(24.0), None) == 24.0


def test_a_window_the_design_cannot_execute_falls_back() -> None:
    context = _study(windows=(36.0,))

    assert plan_time_zero_hours(context, _trajectory(None, executable=False), None) == 36.0


def test_otherwise_the_declared_landmark_then_the_signed_one_then_the_feature_window() -> None:
    signed = SimpleNamespace(landmark_hours=12)

    assert plan_time_zero_hours(_study(landmark_hours=24.0, windows=(48.0,)), None, signed) == 24.0
    assert plan_time_zero_hours(_study(windows=(48.0,)), None, signed) == 12.0
    assert plan_time_zero_hours(_study(windows=(48.0, 72.0)), None, None) == 72.0
    assert plan_time_zero_hours(_study(), None, None) is None


# The review --------------------------------------------------------------


@pytest.mark.parametrize("definition", _CONCEPTS)
def test_the_review_refuses_a_concept_window_past_the_trajectory_window(definition: str) -> None:
    findings = cohort_eligibility_findings(
        _study(concept=(definition, 72.0)), _trajectory(24.0), None
    )

    assert [(item.code, item.severity) for item in findings] == [
        ("COHORT_ELIGIBILITY_AFTER_TIME_ZERO", "blocker")
    ]
    finding = findings[0]
    assert f"the {definition} population admits a stay" in finding.message
    # Membership decided inside the window is not what is refused.
    assert "a positive record within the window, is allowed" in finding.message
    assert "by 24 h after ICU admission" in finding.remediation
    assert "to 72 h or later" in finding.remediation
    assert finding.remediation_route == "study_authority_change"


def test_the_review_refuses_a_minimum_stay_past_the_landmark() -> None:
    findings = cohort_eligibility_findings(
        _study(minimum_icu_hours=48.0, landmark_hours=24.0), None, None
    )

    assert [item.code for item in findings] == ["COHORT_ELIGIBILITY_AFTER_TIME_ZERO"]
    assert "a minimum ICU stay of 48 h" in findings[0].message


@pytest.mark.parametrize(
    "context",
    [
        _study(concept=("sep3", 24.0), windows=(24.0,)),
        _study(minimum_icu_hours=24.0, landmark_hours=24.0),
        _study(concept=("aki", 96.0)),
        _study(windows=(24.0,)),
    ],
    ids=["window at time zero", "minimum at time zero", "no time zero", "no criterion"],
)
def test_the_review_passes_eligibility_decided_by_time_zero(context: ResearchContext) -> None:
    assert cohort_eligibility_findings(context, None, None) == []


def test_an_unreadable_concept_record_is_its_own_refusal() -> None:
    findings = cohort_eligibility_findings(
        _study(windows=(24.0,), constraints={"concept_cohort_window": {"definition": "sep3"}}),
        None,
        None,
    )

    assert [(item.code, item.severity) for item in findings] == [
        ("CONCEPT_POPULATION_RECORD_UNREADABLE", "blocker")
    ]
    # Its remedy is the export's record, not the window.
    assert "Prepare the study's export again" in findings[0].remediation
    assert "window" not in findings[0].remediation


def test_a_plan_review_states_the_finding() -> None:
    def codes(context: ResearchContext, plan: AnalysisPlan | None = None) -> set[str]:
        review = build_plan_scientific_review(
            context=context,
            plan=plan or _plan(),
            literature=_literature(),
            figure_strategy=build_article_figure_strategy(context),
        )
        return {item.code for item in review.findings}

    assert "COHORT_ELIGIBILITY_AFTER_TIME_ZERO" in codes(
        _study(concept=("sep3", 72.0), landmark_hours=24.0)
    )
    # The trajectory window (first 24 h) is the time zero, not the 72 h
    # feature window an association plan of the same study would use.
    trajectory = _study(concept=("vent_ind", 48.0), question=_TRAJECTORY_QUESTION, windows=(72.0,))
    assert "COHORT_ELIGIBILITY_AFTER_TIME_ZERO" in codes(trajectory, _trajectory_plan())
    assert "COHORT_ELIGIBILITY_AFTER_TIME_ZERO" not in codes(trajectory)
    assert "COHORT_ELIGIBILITY_AFTER_TIME_ZERO" not in codes(
        _study(concept=("sep3", 24.0), landmark_hours=24.0)
    )


def test_the_review_reads_the_trajectory_facts_and_the_signed_authority() -> None:
    tree = ast.parse(inspect.getsource(scientific_review.build_plan_scientific_review))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cohort_eligibility_findings"
    ]

    assert [[ast.unparse(arg) for arg in call.args] for call in calls] == [
        ["context", "trajectory_representation", "runtime_authority"]
    ]
