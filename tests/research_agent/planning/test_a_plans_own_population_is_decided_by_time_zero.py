"""A plan's own cohort predicates are decided by the plan's time zero.

A predicate filters a column the host materialized before planning, whatever
window the predicate names: a time-varying column was summarized over its own
analysis window, else over the window the Web host records it materialized
every feature column over.  The analysis time windows of a context are not
such a record.  A predicate is decided by time zero only when its column and
its window both end by then.  A value fixed at admission passes; an outcome
the stay records at its end does not, an ICU length of stay of x is known at
x, and a stay-level score carries no time to compare.  The review says who
repairs each: the Planner restates a window, the study decides a selection
made after time zero, and the host records a column window.  Fixtures are
synthetic and vary the concepts so that no rule keys on one condition.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from easyicu.research_agent.planning import scientific_review
from easyicu.research_agent.planning.cohort_contract import CohortDefinition
from easyicu.research_agent.planning.cohort_eligibility import (
    PredicateAfterTimeZero,
    cohort_predicates_after_time_zero,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    cohort_predicate_findings,
    plan_revision_blocker_codes,
)
from easyicu.research_agent.research_context.materialization_window import (
    host_materialization_window_hours,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    TimeWindow,
    UserPreferences,
    VariableRole,
)

from .scientific_review_fixtures import _context, _literature, _plan

_TIME_VARYING = ["lact", "map", "sep3"]
_VARIABLES = (
    ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
    ConceptDescriptor(name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
    ConceptDescriptor(name="lact", role=VariableRole.LAB, dtype="float64"),
    ConceptDescriptor(name="map", role=VariableRole.VITAL, dtype="float64"),
    ConceptDescriptor(name="sep3", role=VariableRole.OTHER, dtype="int64"),
    ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
)
_WINDOW = "COHORT_PREDICATE_WINDOW_AFTER_TIME_ZERO"
_DECIDED = "COHORT_PREDICATE_DECIDED_AFTER_TIME_ZERO"
_UNRECORDED = "COHORT_PREDICATE_COLUMN_WINDOW_UNRECORDED"


def _constraints(materialized: float | None, *, anchor: str = "ICU admission") -> str | None:
    if materialized is None:
        return None
    window = {"role": "outer_observation_window", "anchor": anchor, "hours": materialized}
    return json.dumps({"materialization_window": window})


def _ctx(
    *variables: ConceptDescriptor,
    materialized: float | None = 24.0,
    windows: tuple[float, ...] = (),
    without: tuple[str, ...] = (),
) -> ResearchContext:
    replaced = {variable.name for variable in variables} | set(without)
    return ResearchContext(
        research_question="Describe in-hospital mortality.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="miiv",
            n_stays=100,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[*(v for v in _VARIABLES if v.name not in replaced), *variables],
        time_windows=[TimeWindow(name=f"window_{end:g}", end_hours=end) for end in windows],
        target_outcome="death",
        user_preferences=UserPreferences(data_constraints=_constraints(materialized)),
    )


def _predicate(
    concept: str,
    op: str = ">=",
    value: object = 2.0,
    *,
    end: float = 24.0,
    anchor: str = "icu_admission",
    aggregation: str = "max",
) -> dict:
    return {
        "concept_id": concept,
        "time_window": {"anchor": anchor, "start_offset_hours": 0, "end_offset_hours": end},
        "aggregation": aggregation,
        "op": op,
        "value": value,
    }


def _rule(
    *inclusion: dict,
    context: ResearchContext | None = None,
    time_zero: float | None = 24.0,
    exclusion: tuple[dict, ...] = (),
) -> tuple[PredicateAfterTimeZero, ...]:
    return cohort_predicates_after_time_zero(
        context or _ctx(),
        inclusion=list(inclusion),
        exclusion=list(exclusion),
        time_zero_hours=time_zero,
    )


def _reasons(found: tuple[PredicateAfterTimeZero, ...]) -> list[tuple[str, float | None]]:
    return [(item.reason, item.decided_by_hours) for item in found]


# The materialization record ----------------------------------------------


@pytest.mark.parametrize(
    ("window", "hours"),
    [
        ({"role": "outer_observation_window", "anchor": "ICU admission", "hours": 24}, 24.0),
        ({"role": "outer_observation_window", "anchor": "icu_admission", "observation_hours": 6}, 6.0),
        ({"role": "outer_observation_window", "anchor": "hospital admission", "hours": 24}, None),
        ({"role": "follow_up", "anchor": "icu_admission", "hours": 24}, None),
        ({"role": "outer_observation_window", "anchor": "icu_admission", "hours": True}, None),
        ({"role": "outer_observation_window", "anchor": "icu_admission", "hours": 0}, None),
    ],
)
def test_the_host_records_the_window_it_materialized(window: dict, hours: float | None) -> None:
    context = _ctx().model_copy(
        update={"user_preferences": UserPreferences(data_constraints=json.dumps({"materialization_window": window}))}
    )

    assert host_materialization_window_hours(context) == hours
    assert host_materialization_window_hours(_ctx(materialized=None)) is None


# The shared rule ---------------------------------------------------------


@pytest.mark.parametrize("concept", _TIME_VARYING)
@pytest.mark.parametrize(("end", "refused"), [(72.0, True), (24.0, False), (6.0, False)])
def test_a_predicate_window_must_end_by_time_zero(concept: str, end: float, refused: bool) -> None:
    found = _rule(_predicate(concept, end=end))

    assert _reasons(found) == ([("window", end)] if refused else [])


@pytest.mark.parametrize("anchor", ["hospital_admission", "intubation", ""])
def test_a_window_not_counted_from_icu_admission_cannot_be_compared(anchor: str) -> None:
    assert _reasons(_rule(_predicate("lact", end=6.0, anchor=anchor))) == [("anchor", None)]
    assert _rule(_predicate("lact", end=6.0, anchor="icu_admit")) == ()


@pytest.mark.parametrize("concept", _TIME_VARYING)
def test_the_column_a_predicate_filters_must_end_by_time_zero_too(concept: str) -> None:
    # The host summarized the column over 0-24 h; a 0-6 h predicate filters it.
    (found,) = _rule(_predicate(concept, end=6.0), time_zero=6.0)

    assert (found.reason, found.decided_by_hours) == ("column_window", 24.0)
    assert "the host's materialization window ending at 24 h" in found.message()
    assert _rule(_predicate(concept, end=6.0), context=_ctx(materialized=6.0), time_zero=6.0) == ()


@pytest.mark.parametrize(
    ("label", "refused"),
    [
        ("icu_admission[0,6]h", False),
        ("first_6h", False),
        ("icu_admission[0,48]h", True),
        ("sepsis_onset_0_6h", True),
    ],
)
def test_a_columns_own_window_decides_before_the_host_window(label: str, refused: bool) -> None:
    lact = ConceptDescriptor(name="lact", role=VariableRole.LAB, dtype="float64", analysis_window=label)

    for materialized in (6.0, 24.0, None):
        found = _rule(_predicate("lact", end=6.0), context=_ctx(lact, materialized=materialized), time_zero=6.0)

        assert [(item.reason, item.column_window) for item in found] == (
            [("column_window", label)] if refused else []
        )


@pytest.mark.parametrize("windows", [(24.0,), (24.0, 6.0, 720.0)])
def test_analysis_time_windows_are_not_a_materialization_record(windows: tuple[float, ...]) -> None:
    (found,) = _rule(_predicate("lact"), context=_ctx(materialized=None, windows=windows))

    assert (found.reason, found.decided_by_hours) == ("unrecorded", None)
    assert "whose materialization window the context does not record" in found.message()


def test_the_column_is_judged_before_the_window_a_predicate_states() -> None:
    late = _predicate("lact", end=72.0)

    assert _reasons(_rule(late, context=_ctx(materialized=None))) == [("unrecorded", None)]
    assert _reasons(_rule(late, context=_ctx(materialized=48.0))) == [("column_window", 48.0)]
    assert _reasons(_rule(late, context=_ctx(materialized=24.0))) == [("window", 72.0)]


def test_a_column_only_the_run_knows_is_judged_as_data() -> None:
    derived = _predicate("sep3_sofa1_max", "==", 1, end=6.0, aggregation="any")

    assert _rule(derived) == ()
    assert _reasons(_rule(derived, time_zero=6.0)) == [("column_window", 24.0)]
    assert _reasons(_rule(derived, context=_ctx(materialized=None))) == [("unrecorded", None)]


def test_a_predicate_filters_its_summary_column() -> None:
    summary = ConceptDescriptor(
        name="lact_max", role=VariableRole.LAB, dtype="float64", analysis_window="icu_admission[0,48]h"
    )
    context = _ctx(summary, without=("lact",))

    (found,) = _rule(_predicate("lact", aggregation="max"), context=context)
    assert (found.reason, found.column_window) == ("column_window", "icu_admission[0,48]h")
    # No lact_min column: the bare concept is judged by the host's window.
    assert _rule(_predicate("lact", aggregation="min"), context=context) == ()


def test_an_observation_count_is_windowed_like_its_concept() -> None:
    count = ConceptDescriptor(name="lact_n", role=VariableRole.META, dtype="int64")
    measured = _predicate("lact_n", ">=", 1, end=6.0, aggregation="sum")

    assert _reasons(_rule(measured, context=_ctx(count), time_zero=6.0)) == [("column_window", 24.0)]
    assert _rule(measured, context=_ctx(count, materialized=6.0), time_zero=6.0) == ()


def test_a_stay_level_resolution_is_not_an_admission_value() -> None:
    worst = ConceptDescriptor(
        name="sofa_worst",
        role=VariableRole.COMPOSITE_SCORE,
        dtype="float64",
        temporal_resolution="stay-level",
        analysis_window="icu_admission[0,72]h",
    )

    found = _rule(_predicate("sofa_worst", end=24.0), context=_ctx(worst))

    assert [(item.reason, item.column_window) for item in found] == [
        ("column_window", "icu_admission[0,72]h")
    ]


@pytest.mark.parametrize(
    ("op", "value", "unit", "decided"),
    [
        (">=", 3, "days", 72.0),
        ("<", 2.5, None, 60.0),
        ("in", [1, 2], "days", 48.0),
        (">=", 48, "hours", 48.0),
        (">=", 1, "days", None),
        (">=", 0.5, "days", None),
        (">=", 24, "h", None),
    ],
)
def test_an_icu_length_of_stay_is_known_at_its_threshold(op, value, unit, decided) -> None:
    los = ConceptDescriptor(name="los_icu", role=VariableRole.OUTCOME, dtype="float64", unit=unit)

    # The predicate's own window does not bound a stay-level length of stay.
    found = _rule(_predicate("los_icu", op, value, end=24.0), context=_ctx(los))

    assert _reasons(found) == ([("icu_stay_length", decided)] if decided is not None else [])


def test_a_length_of_stay_without_a_number_is_known_only_at_the_end() -> None:
    assert _reasons(_rule(_predicate("los_icu", "not_missing", None))) == [("stay_outcome", None)]


@pytest.mark.parametrize(
    "variable",
    [
        ConceptDescriptor(name="hospital_death", role=VariableRole.OUTCOME, dtype="int64"),
        ConceptDescriptor(name="icu_readmission", role=VariableRole.OUTCOME, dtype="int64"),
    ],
    ids=lambda variable: variable.name,
)
def test_an_outcome_is_known_only_at_the_stays_end(variable: ConceptDescriptor) -> None:
    (found,) = _rule(_predicate(variable.name, "==", 1, end=6.0), context=_ctx(variable))

    assert found.reason == "stay_outcome"
    assert "records only at its end" in found.message()


def test_the_studys_outcome_is_an_outcome_whatever_its_role() -> None:
    aki = ConceptDescriptor(name="aki", role=VariableRole.ORDINAL_SCORE, dtype="int64")
    context = _ctx(aki)
    targeted = context.model_copy(update={"target_outcome": "aki"})

    assert _rule(_predicate("aki", end=6.0), context=context) == ()
    assert _reasons(_rule(_predicate("aki", end=6.0), context=targeted)) == [("stay_outcome", None)]


@pytest.mark.parametrize(
    "predicate",
    [
        _predicate("age", ">=", 18, end=72.0),
        _predicate("stay_id", "not_missing", None, end=72.0),
        # Stay-level concepts the dictionary files under demographics.
        _predicate("icu_unit_type", "==", "MICU", end=72.0, anchor="hospital_admission"),
        _predicate("bmi", "<", 40, end=72.0),
    ],
    ids=["demographic", "identifier", "unit type", "body mass index"],
)
def test_a_value_fixed_at_admission_passes_whatever_its_window(predicate: dict) -> None:
    # A value fixed at admission needs no materialization record.
    assert _rule(predicate, context=_ctx(materialized=None), time_zero=6.0) == ()


@pytest.mark.parametrize("concept", ["icu_unit_type", "bmi"])
def test_the_dictionary_dates_a_stay_level_value_whatever_the_runs_role(concept: str) -> None:
    other = ConceptDescriptor(name=concept, role=VariableRole.OTHER, dtype="object")

    assert _rule(_predicate(concept, "==", 1, end=72.0), context=_ctx(other), time_zero=6.0) == ()


@pytest.mark.parametrize("concept", ["apache_iv", "saps3"])
def test_a_stay_level_score_carries_no_time_to_compare(concept: str) -> None:
    (found,) = _rule(_predicate(concept, ">=", 20, end=6.0), time_zero=24.0)

    assert found.reason == "stay_level"
    assert "does not date to ICU admission" in found.message()


def test_a_stay_level_outcome_is_known_only_at_the_end() -> None:
    assert _reasons(_rule(_predicate("los_hosp", ">=", 0.5, end=6.0))) == [("stay_outcome", None)]


def test_without_a_time_zero_nothing_is_compared() -> None:
    assert _rule(_predicate("lact", end=720.0), context=_ctx(materialized=None), time_zero=None) == ()


def test_inclusion_and_exclusion_are_both_read_in_order() -> None:
    found = _rule(
        _predicate("lact", end=48.0),
        exclusion=(_predicate("death", "==", 1), _predicate("map", "<", 65, end=12.0)),
    )

    assert [(item.kind, item.reason) for item in found] == [
        ("inclusion", "window"),
        ("exclusion", "stay_outcome"),
    ]


def test_the_messages_name_the_predicate_and_both_hours() -> None:
    (window,) = _rule(_predicate("lact", end=48.0))
    los = ConceptDescriptor(name="los_icu", role=VariableRole.OUTCOME, dtype="float64", unit="days")
    (length,) = _rule(_predicate("los_icu", ">=", 3), context=_ctx(los))
    (anchor,) = _rule(_predicate("lact", anchor="intubation"))

    assert window.message() == (
        "the inclusion predicate lact >= 2.0 (max over 0 to 48 h from ICU admission) is "
        "decided at 48 h after ICU admission, after the plan's time zero at 24 h after "
        "ICU admission"
    )
    assert "by its ICU length of stay at 72 h" in length.message()
    assert "eligibility would depend on survival in the ICU" in length.message()
    assert "is not counted from ICU admission" in anchor.message()


# The review --------------------------------------------------------------


def _study(
    *variables: ConceptDescriptor,
    landmark_hours: float | None = 24.0,
    materialized: float | None = 24.0,
) -> ResearchContext:
    base = _context()
    return base.model_copy(
        update={
            "variables": [*base.variables, *variables],
            "user_preferences": base.user_preferences.model_copy(
                update={"landmark_hours": landmark_hours, "data_constraints": _constraints(materialized)}
            ),
        }
    )


def _with_cohort(*inclusion: dict) -> AnalysisPlan:
    cohort = CohortDefinition.from_dict({"name": "primary", "inclusion": list(inclusion), "exclusion": []})
    return _plan().model_copy(update={"cohort": cohort})


def _routes(findings) -> list[tuple[str, str, str]]:
    return [(item.code, item.severity, item.remediation_route) for item in findings]


@pytest.mark.parametrize("concept", ["lact", "sep3"])
def test_the_planner_restates_a_window_over_a_column_decided_in_time(concept: str) -> None:
    findings = cohort_predicate_findings(_study(), _with_cohort(_predicate(concept, end=72.0)), None, None)

    assert _routes(findings) == [(_WINDOW, "blocker", "agent_plan_revision")]
    assert f"the inclusion predicate {concept} >= 2.0" in findings[0].message
    assert "from ICU admission to 24 h after ICU admission" in findings[0].remediation
    assert plan_revision_blocker_codes(findings) == ()


def test_a_selection_decided_after_time_zero_is_the_studys_to_change() -> None:
    findings = cohort_predicate_findings(
        _study(landmark_hours=6.0), _with_cohort(_predicate("lact", end=6.0)), None, None
    )

    assert _routes(findings) == [(_DECIDED, "blocker", "study_authority_change")]
    assert "the host's materialization window ending at 24 h" in findings[0].message
    assert "(24 h or later)" in findings[0].remediation
    assert plan_revision_blocker_codes(findings) == (_DECIDED,)


def test_an_unrecorded_column_window_is_the_hosts_to_record() -> None:
    findings = cohort_predicate_findings(
        _study(materialized=None), _with_cohort(_predicate("lact", end=24.0)), None, None
    )

    assert _routes(findings) == [(_UNRECORDED, "blocker", "runtime_capability")]
    assert "a plan revision cannot supply that record" in findings[0].remediation
    assert plan_revision_blocker_codes(findings) == (_UNRECORDED,)


def test_the_review_reads_the_runs_own_variables() -> None:
    hours = ConceptDescriptor(name="los_icu", role=VariableRole.OUTCOME, dtype="float64", unit="hours")
    ventilation = ConceptDescriptor(name="vent_ind", role=VariableRole.OUTCOME, dtype="int64")

    (length,) = cohort_predicate_findings(
        _study(hours), _with_cohort(_predicate("los_icu", ">=", 48)), None, None
    )
    assert (length.code, "by its ICU length of stay at 48 h" in length.message) == (_DECIDED, True)
    assert cohort_predicate_findings(_study(hours), _with_cohort(_predicate("los_icu", ">=", 12)), None, None) == []
    assert cohort_predicate_findings(_study(), _with_cohort(_predicate("vent_ind", "==", 1)), None, None) == []
    (outcome,) = cohort_predicate_findings(
        _study(ventilation), _with_cohort(_predicate("vent_ind", "==", 1)), None, None
    )
    assert (outcome.code, "records only at its end" in outcome.message) == (_DECIDED, True)


@pytest.mark.parametrize(
    "plan",
    [
        _with_cohort(_predicate("lact", end=24.0)),
        _with_cohort(_predicate("age", ">=", 18, end=72.0)),
        _plan(),
    ],
    ids=["decided at time zero", "fixed at admission", "no cohort"],
)
def test_the_review_passes_a_population_decided_by_time_zero(plan: AnalysisPlan) -> None:
    assert cohort_predicate_findings(_study(), plan, None, None) == []


def test_without_a_time_zero_the_review_compares_nothing() -> None:
    context = _study(landmark_hours=None, materialized=None)

    assert cohort_predicate_findings(context, _with_cohort(_predicate("lact", end=720.0)), None, None) == []


def test_a_trajectory_plan_says_its_time_zero_is_its_windows_end() -> None:
    trajectory = {"trajectory_window": {"executable": True, "window_end_hours": 24.0}}
    plan = _with_cohort(_predicate("lact", end=48.0))

    (finding,) = cohort_predicate_findings(_study(landmark_hours=None), plan, trajectory, None)
    (association,) = cohort_predicate_findings(_study(), plan, None, None)

    assert finding.message.endswith("A trajectory plan's time zero is the end of its trajectory window.")
    assert "trajectory" not in association.message


def test_a_plan_review_states_the_finding() -> None:
    def codes(plan: AnalysisPlan) -> set[str]:
        context = _study()
        review = build_plan_scientific_review(
            context=context,
            plan=plan,
            literature=_literature(),
            figure_strategy=build_article_figure_strategy(context),
        )
        return {item.code for item in review.findings}

    assert _WINDOW in codes(_with_cohort(_predicate("lact", end=48.0)))
    assert not {_WINDOW, _DECIDED, _UNRECORDED} & codes(_with_cohort(_predicate("lact")))


def test_the_review_reads_the_plan_and_its_time_zero_sources() -> None:
    tree = ast.parse(inspect.getsource(scientific_review.build_plan_scientific_review))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cohort_predicate_findings"
    ]

    assert [[ast.unparse(arg) for arg in call.args] for call in calls] == [
        ["context", "plan", "trajectory_representation", "runtime_authority"]
    ]


@pytest.mark.parametrize(
    ("code", "title"),
    [
        (_WINDOW, "入组谓词的窗口晚于分析时间零点"),
        (_DECIDED, "入组选择晚于分析时间零点"),
        (_UNRECORDED, "入组条件所用列的物化窗口未记录"),
    ],
)
def test_each_finding_reads_in_chinese(code: str, title: str) -> None:
    vocab = (
        Path(scientific_review.__file__).resolve().parents[2]
        / "webserver/static/js/screens-agent-reader-vocab.js"
    ).read_text(encoding="utf-8")

    assert f"{code}: ['{title}'" in vocab
