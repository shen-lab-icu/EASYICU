"""A cohort that selects stays by the study's outcome is named in review.

The time-zero rule asks when a criterion is decided; this rule asks whether it
tests the outcome.  A criterion on the outcome's own column cuts the outcome
at its threshold whenever it is decided, so the review reports a major
finding for the study to resolve, even when the time-zero rule passes it.
The stays still in the ICU at time zero, with their ICU length of stay as the
outcome, are a landmark cohort instead: a minor finding asking that the
outcome be stated for that population.  A criterion over a concept the
outcome is defined from selects on the outcome only when the time-zero rule
finds the selection made after time zero; decided by then, it is a baseline
value.  A missingness test, a baseline exclusion of an earlier event by its
own time, a predicate outside its column's values and a stay-length entry
with another outcome stay with the rules that already judge them.  Fixtures
are synthetic and vary the outcome's kind so that no rule keys on one concept.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from easyicu.research_agent.planning import cohort_eligibility, scientific_review
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    cohort_concept_id_scope,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    cohort_predicate_domain_findings,
    cohort_predicate_findings,
    outcome_selection_findings,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    ConceptDescriptor,
    ObservationSemantics,
    ResearchContext,
    VariableRole,
)

from .scientific_review_fixtures import _context, _literature, _plan

_SELECTS = "COHORT_SELECTS_ON_OUTCOME"
_LANDMARK = "LANDMARK_STAY_LENGTH_OUTCOME"
_DECIDED = "COHORT_PREDICATE_DECIDED_AFTER_TIME_ZERO"
_IDS = ("organ_score", "marker_value", "marker_stage", "marker_event_new", "death")
_ORGAN = ConceptDescriptor(
    name="organ_score",
    role=VariableRole.ORDINAL_SCORE,
    dtype="int64",
    is_ordinal=True,
    ordinal_levels=[0, 1, 2, 3, 4],
)
_VENTILATION = ConceptDescriptor(name="vent_ind", role=VariableRole.OUTCOME, dtype="int64")


def _los(unit: str = "hours") -> ConceptDescriptor:
    return ConceptDescriptor(name="los_icu", role=VariableRole.OUTCOME, dtype="float64", unit=unit)


def _marker_chain(window: str | None) -> tuple[ConceptDescriptor, ...]:
    """An outcome defined from a stage, itself defined from a measurement."""

    return (
        ConceptDescriptor(
            name="marker_value",
            role=VariableRole.LAB,
            dtype="float64",
            analysis_window=window,
        ),
        ConceptDescriptor(
            name="marker_stage",
            role=VariableRole.ORDINAL_SCORE,
            dtype="int64",
            derived_from_concepts=["marker_value"],
            analysis_window=window,
        ),
        ConceptDescriptor(
            name="marker_event_new",
            role=VariableRole.OUTCOME,
            dtype="int64",
            derived_from_concepts=["marker_stage"],
        ),
    )


def _death_time() -> ConceptDescriptor:
    return ConceptDescriptor(
        name="death_time",
        role=VariableRole.TIME,
        dtype="float64",
        source_concept="death",
        temporal_resolution="relative to icu_admission in h",
        observation_semantics=ObservationSemantics(
            kind="conditional_event_time",
            event_status_column="death",
            representative_column="death_time",
            time_origin="icu_admission",
            time_unit="h",
        ),
    )


def _study(
    *variables: ConceptDescriptor,
    outcomes: tuple[str, ...] = ("los_icu",),
    time_zero: float | None = 72.0,
    minimum_stay: float | None = None,
    materialized: float | None = 24.0,
) -> ResearchContext:
    constraints: dict = {}
    if materialized is not None:
        constraints["materialization_window"] = {
            "role": "outer_observation_window",
            "anchor": "ICU admission",
            "hours": materialized,
        }
    if minimum_stay is not None:
        constraints["cohort"] = {"min_icu_los_hours": minimum_stay}
    base = _context()
    return base.model_copy(
        update={
            "variables": [*base.variables, *variables],
            "cohort": base.cohort.model_copy(update={"requested_outcome_columns": list(outcomes)}),
            "user_preferences": base.user_preferences.model_copy(
                update={
                    "landmark_hours": time_zero,
                    "data_constraints": json.dumps(constraints) if constraints else None,
                }
            ),
        }
    )


def _predicate(concept: str, op: str = ">=", value: object = 48, *, end: float = 24.0) -> dict:
    return {
        "concept_id": concept,
        "time_window": {"anchor": "icu_admission", "start_offset_hours": 0, "end_offset_hours": end},
        "aggregation": "max",
        "op": op,
        "value": value,
    }


def _with_cohort(*inclusion: dict, exclusion: tuple[dict, ...] = ()) -> AnalysisPlan:
    with cohort_concept_id_scope(_IDS):
        cohort = CohortDefinition.from_dict(
            {"name": "primary", "inclusion": list(inclusion), "exclusion": list(exclusion)}
        )
    return _plan().model_copy(update={"cohort": cohort})


def _modeling(outcome: str, plan: AnalysisPlan) -> AnalysisPlan:
    """The plan's primary model estimates ``outcome`` instead."""

    step = plan.steps[0]
    requirement = step.model_requirements[0].model_copy(update={"outcome": outcome})
    modeled = step.model_copy(update={"model_requirements": [requirement]})
    return plan.model_copy(update={"steps": [modeled, *plan.steps[1:]]})


def _named(context: ResearchContext, plan: AnalysisPlan) -> list[tuple[str, str, str]]:
    return [
        (item.code, item.severity, item.remediation_route)
        for item in outcome_selection_findings(context, plan, None, None)
    ]


# The outcome's own column -------------------------------------------------


def test_a_criterion_on_the_outcome_decided_before_time_zero_is_major() -> None:
    context, plan = _study(_los()), _with_cohort(_predicate("los_icu", ">=", 48))

    (finding,) = outcome_selection_findings(context, plan, None, None)

    assert (finding.code, finding.severity, finding.remediation_route) == (
        _SELECTS,
        "major",
        "study_authority_change",
    )
    assert "selects stays by the study's outcome los_icu itself" in finding.message
    assert (
        "decided at 48 h, before the plan's time zero at 72 h after ICU admission, so it "
        "keeps stays whose outcome ended before time zero"
    ) in finding.message
    assert "count the outcome from the analysis time zero" in finding.remediation
    # The time-zero rule alone passes it: it is decided by time zero.
    assert cohort_predicate_findings(context, plan, None, None) == []


@pytest.mark.parametrize(
    ("kind", "op", "value", "unit"),
    [
        ("inclusion", ">=", 72, "hours"),
        ("inclusion", ">", 72, "hours"),
        ("exclusion", "<", 72, "hours"),
        ("exclusion", "<=", 72, "hours"),
        ("inclusion", ">=", 3, "days"),
    ],
)
def test_the_stays_still_in_the_icu_at_time_zero_are_a_landmark_cohort(kind, op, value, unit) -> None:
    predicate = _predicate("los_icu", op, value)
    plan = (
        _with_cohort(predicate) if kind == "inclusion" else _with_cohort(exclusion=(predicate,))
    )

    (finding,) = outcome_selection_findings(_study(_los(unit)), plan, None, None)

    assert (finding.code, finding.severity) == (_LANDMARK, "minor")
    assert finding.message.startswith("The cohort is a landmark cohort:")
    assert "still in the ICU at 72 h after admission" in finding.remediation
    assert "ICU stay remaining after time zero" in finding.remediation


@pytest.mark.parametrize(
    ("variable", "predicate", "decided_after"),
    [
        (_los(), _predicate("los_icu", "<=", 72), False),
        (_los(), _predicate("los_icu", ">=", 96), True),
        (_los(), _predicate("los_icu", "in", [72]), False),
        (_VENTILATION, _predicate("vent_ind", "==", 1), False),
        (_ORGAN, _predicate("organ_score", ">=", 2), False),
    ],
    ids=["short stays", "after time zero", "a list", "binary outcome", "ordinal outcome"],
)
def test_any_other_test_of_the_outcome_is_major_whatever_its_threshold(
    variable: ConceptDescriptor, predicate: dict, decided_after: bool
) -> None:
    context = _study(variable, outcomes=(variable.name,))

    (finding,) = outcome_selection_findings(context, _with_cohort(predicate), None, None)

    assert (finding.code, finding.severity) == (_SELECTS, "major")
    assert f"the study's outcome {variable.name} itself" in finding.message
    assert finding.message.endswith(
        "; it is decided at 96 h, after the plan's time zero at 72 h after ICU admission."
    ) == decided_after


def test_without_a_time_zero_the_outcome_itself_is_still_named() -> None:
    context = _study(_los(), time_zero=None, materialized=None)

    assert _named(context, _with_cohort(_predicate("los_icu", ">=", 72))) == [
        (_SELECTS, "major", "study_authority_change")
    ]
    (finding,) = outcome_selection_findings(
        context, _with_cohort(_predicate("los_icu", ">=", 72)), None, None
    )
    assert "the plan states no time zero to count the outcome from" in finding.message


@pytest.mark.parametrize(
    ("minimum_stay", "code", "severity"),
    [(72.0, _LANDMARK, "minor"), (48.0, _SELECTS, "major"), (96.0, _SELECTS, "major")],
)
def test_the_studys_minimum_icu_stay_is_read_like_a_criterion(
    minimum_stay: float, code: str, severity: str
) -> None:
    context = _study(_los(), minimum_stay=minimum_stay)

    (finding,) = outcome_selection_findings(context, _plan(), None, None)

    assert (finding.code, finding.severity) == (code, severity)
    assert f"of a minimum ICU stay of {minimum_stay:g} h (the study's)" in finding.message


def test_an_outcome_only_the_plan_models_is_an_outcome_too() -> None:
    context = _study(_VENTILATION, outcomes=())
    plan = _with_cohort(_predicate("vent_ind", "==", 1))

    assert _named(context, plan) == []
    assert _named(context, _modeling("vent_ind", plan)) == [
        (_SELECTS, "major", "study_authority_change")
    ]


# What the outcome is defined from ----------------------------------------


@pytest.mark.parametrize("concept", ["marker_stage", "marker_value"])
def test_a_concept_the_outcome_is_defined_from_selects_on_it_after_time_zero(concept: str) -> None:
    context = _study(*_marker_chain("icu_admission[0,72]h"), outcomes=("marker_event_new",), time_zero=24.0)

    (finding,) = outcome_selection_findings(context, _with_cohort(_predicate(concept, ">=", 2)), None, None)

    assert (finding.code, finding.severity) == (_SELECTS, "major")
    assert (
        f"tests {concept}, which the study's outcome marker_event_new is defined from, and "
        "cannot be shown decided by the plan's time zero at 24 h after ICU admission"
    ) in finding.message


def test_a_concept_the_outcome_is_defined_from_is_a_baseline_value_by_time_zero() -> None:
    context = _study(*_marker_chain("icu_admission[0,24]h"), outcomes=("marker_event_new",), time_zero=24.0)

    assert _named(context, _with_cohort(_predicate("marker_value", ">=", 2))) == []
    assert _named(context, _with_cohort(_predicate("marker_stage", ">=", 2))) == []


def test_a_window_to_restate_or_a_record_to_add_stays_with_the_time_zero_rule() -> None:
    restated = _study(*_marker_chain("icu_admission[0,24]h"), outcomes=("marker_event_new",), time_zero=24.0)
    unrecorded = _study(
        *_marker_chain(None), outcomes=("marker_event_new",), time_zero=24.0, materialized=None
    )
    late_window = _with_cohort(_predicate("marker_value", ">=", 2, end=72.0))
    plan = _with_cohort(_predicate("marker_value", ">=", 2))

    assert [item.code for item in cohort_predicate_findings(restated, late_window, None, None)] == [
        "COHORT_PREDICATE_WINDOW_AFTER_TIME_ZERO"
    ]
    assert _named(restated, late_window) == []
    assert [item.code for item in cohort_predicate_findings(unrecorded, plan, None, None)] == [
        "COHORT_PREDICATE_COLUMN_WINDOW_UNRECORDED"
    ]
    assert _named(unrecorded, plan) == []


def test_the_outcome_rule_reads_the_time_zero_rules_own_routing() -> None:
    routed_to_the_study = {
        reason
        for reason, (code, _route) in scientific_review._COHORT_PREDICATE_FINDINGS.items()
        if code == _DECIDED
    }

    assert set(cohort_eligibility._SELECTED_AFTER_TIME_ZERO) == routed_to_the_study


# What other rules judge ---------------------------------------------------


def test_a_death_before_time_zero_is_a_baseline_exclusion_by_its_own_time() -> None:
    context = _study(_death_time(), outcomes=("death",), time_zero=24.0)
    by_time_zero = _with_cohort(exclusion=(_predicate("death", "==", 1, end=24.0),))
    later = _with_cohort(exclusion=(_predicate("death", "==", 1, end=72.0),))

    assert _named(context, by_time_zero) == []
    assert cohort_predicate_findings(context, by_time_zero, None, None) == []
    assert _named(context, later) == []
    assert [item.code for item in cohort_predicate_findings(context, later, None, None)] == [_DECIDED]


def test_a_stay_length_entry_with_another_outcome_is_the_time_zero_rules() -> None:
    context = _study(_los(), outcomes=(), time_zero=24.0)
    plan = _with_cohort(_predicate("los_icu", ">=", 48))

    assert _named(context, plan) == []
    assert [item.code for item in cohort_predicate_findings(context, plan, None, None)] == [_DECIDED]
    assert outcome_selection_findings(_study(_los(), outcomes=(), time_zero=24.0, minimum_stay=48.0), _plan(), None, None) == []


def test_a_missingness_test_of_the_outcome_selects_no_value() -> None:
    assert _named(_study(_los()), _with_cohort(_predicate("los_icu", "not_missing", None))) == []


def test_a_predicate_outside_its_columns_values_is_left_to_that_finding() -> None:
    context = _study(_ORGAN, outcomes=("organ_score",))
    impossible = _with_cohort(_predicate("organ_score", ">=", 5))

    assert [item.code for item in cohort_predicate_domain_findings(context, impossible)] == [
        "COHORT_PREDICATE_OUTSIDE_COLUMN_DOMAIN"
    ]
    assert _named(context, impossible) == []


# The review ---------------------------------------------------------------


def test_the_review_states_the_finding_beside_the_others() -> None:
    context = _study(_los())
    review = build_plan_scientific_review(
        context=context,
        plan=_with_cohort(_predicate("los_icu", ">=", 48)),
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )

    (finding,) = [item for item in review.findings if item.code == _SELECTS]
    assert finding.severity == "major"
    assert _LANDMARK not in {item.code for item in review.findings}


def test_the_review_reads_the_outcome_rule_with_the_plans_time_zero_sources() -> None:
    tree = ast.parse(inspect.getsource(scientific_review.build_plan_scientific_review))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "outcome_selection_findings"
    ]

    assert [[ast.unparse(arg) for arg in call.args] for call in calls] == [
        ["context", "plan", "trajectory_representation", "runtime_authority"]
    ]


@pytest.mark.parametrize(
    ("code", "title", "detail"),
    [
        (_SELECTS, "入组条件按结局本身选人", "不论阈值是多少"),
        (_LANDMARK, "结局是时间零点时仍在 ICU 者的总住院时长", "改报时间零点之后剩余的 ICU 住院时长"),
    ],
)
def test_each_finding_reads_in_chinese(code: str, title: str, detail: str) -> None:
    vocab = (
        Path(scientific_review.__file__).resolve().parents[2]
        / "webserver/static/js/screens-agent-reader-vocab.js"
    ).read_text(encoding="utf-8")

    (entry,) = [line for line in vocab.splitlines() if line.strip().startswith(f"{code}: [")]
    assert f"{code}: ['{title}'" in entry
    assert detail in entry
