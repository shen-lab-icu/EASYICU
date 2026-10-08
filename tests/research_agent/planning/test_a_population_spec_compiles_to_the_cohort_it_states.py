"""A study's population spec compiles to the cohort it states.

The Planner states whom a study includes as typed criteria; the host decides
how each one reaches the rows.  A criterion is applied by the source when a
typed record shows every input row meets it, by the plan's predicates when a
column the owners can read expresses it, waits for an extraction when the
input lacks what applies it, and is otherwise not applied with a reason.
Every criterion gets one disposition and every predicate belongs to one
criterion.  A status has no threshold to write, so "a 0/1 diagnosis flag of at
least 2" cannot be stated; stated as a measurement, the column's values refuse
it.  Fixtures are synthetic and vary the concepts, so that no rule keys on one
condition or one benchmark question.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from easyicu.research_agent.cohort.materializer import (
    materialize_cohort,
    materialize_to_parquet,
)
from easyicu.research_agent.intake.materialized_metadata import (
    FIRST_ICU_STAY_RESTRICTION_SCHEMA,
)
from easyicu.research_agent.planning.cohort_contract import concept_id_exists
from easyicu.research_agent.planning.population_compile import (
    NOT_APPLIED_REASONS,
    POPULATION_COMPILE_SCHEMA_VERSION,
    REQUIRES_EXTRACTION_REASONS,
    CompiledCriterion,
    SourceProof,
    compile_population,
)
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ObservationSemantics,
    ResearchContext,
    UserPreferences,
    VariableRole,
)
from tests.support.native_outcome_export import native_outcome, typed_native_export

_BINARY = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}


def _event_time(status: str, *, unit: str = "h") -> ConceptDescriptor:
    return ConceptDescriptor(
        name=f"{status}_time",
        role=VariableRole.OTHER,
        dtype="float64",
        observation_semantics=ObservationSemantics(
            kind="conditional_event_time",
            event_status_column=status,
            representative_column=f"{status}_time",
            time_origin="icu_admission",
            time_unit=unit,
        ),
    )


_VARIABLES = (
    ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
    ConceptDescriptor(
        name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64", unit="years"
    ),
    ConceptDescriptor(
        name="los_icu", role=VariableRole.OUTCOME, dtype="float64", unit="days"
    ),
    ConceptDescriptor(
        name="death", role=VariableRole.OUTCOME, dtype="int64", observed_domain=_BINARY
    ),
    _event_time("death"),
    # A status summarized over its own window.
    ConceptDescriptor(
        name="sep3",
        role=VariableRole.OTHER,
        dtype="int64",
        analysis_window="icu_admission[0,24]h",
        observed_domain=_BINARY,
    ),
    # A status summarized over the host's materialization window.
    ConceptDescriptor(
        name="circ", role=VariableRole.OTHER, dtype="int64", observed_domain=_BINARY
    ),
    # A status read by its own recorded time.
    ConceptDescriptor(
        name="vent",
        role=VariableRole.INTERVENTION,
        dtype="int64",
        observed_domain=_BINARY,
    ),
    _event_time("vent"),
    # An outcome status the stay records over its whole length, with no time.
    ConceptDescriptor(
        name="aki", role=VariableRole.OUTCOME, dtype="int64", observed_domain=_BINARY
    ),
    ConceptDescriptor(
        name="lact_max",
        role=VariableRole.LAB,
        dtype="float64",
        source_concept="lact",
        unit="mmol/L",
    ),
    ConceptDescriptor(
        name="map", role=VariableRole.VITAL, dtype="float64", unit="mmHg"
    ),
    ConceptDescriptor(
        name="organ_max",
        role=VariableRole.ORDINAL_SCORE,
        dtype="int64",
        source_concept="organ",
        is_ordinal=True,
        ordinal_levels=[0, 1, 2, 3, 4],
    ),
)


def _ctx(
    *variables: ConceptDescriptor,
    without: tuple[str, ...] = (),
    constraints: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    n_stays: int = 100,
    materialized: float | None = 24.0,
) -> ResearchContext:
    replaced = {variable.name for variable in variables} | set(without)
    data = dict(constraints or {})
    if materialized is not None:
        data["materialization_window"] = {
            "role": "outer_observation_window",
            "anchor": "ICU admission",
            "hours": materialized,
        }
    return ResearchContext(
        research_question="Which stays does the study include?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="miiv",
            n_stays=n_stays,
            id_columns=["stay_id"],
            outcome_columns=["death"],
            provenance=dict(provenance or {}),
        ),
        variables=[*(v for v in _VARIABLES if v.name not in replaced), *variables],
        target_outcome="death",
        user_preferences=UserPreferences(
            data_constraints=json.dumps(data) if data else None
        ),
    )


def _spec(*criteria: dict[str, Any]) -> PopulationSpec:
    return PopulationSpec.model_validate(
        {
            "criteria": [
                {
                    "id": f"c{index}",
                    "quote": f"criterion number {index}",
                    "source": "question",
                    "role": "include",
                    **item,
                }
                for index, item in enumerate(criteria, start=1)
            ]
        }
    )


def _one(
    criterion: dict[str, Any],
    context: ResearchContext | None = None,
    *,
    time_zero: float | None = None,
) -> CompiledCriterion:
    compiled = compile_population(
        _spec(criterion), context or _ctx(), time_zero_hours=time_zero
    )
    (item,) = compiled.criteria
    return item


def _window(start: float = 0.0, end: float = 24.0) -> dict[str, float]:
    return {"start_hours": start, "end_hours": end}


def _condition(
    *concepts: str,
    start: float = 0.0,
    end: float | None = 24.0,
    **extra,
) -> dict:
    """A condition over ``[start, end)``; ``end=None`` states the whole stay."""

    return {
        "kind": "condition_present",
        "concepts_all_of": list(concepts),
        "window": None if end is None else _window(start, end),
        **extra,
    }


def _absent(concept: str, *, start: float = 0.0, end: float | None = 24.0) -> dict:
    return {
        "kind": "event_absent",
        "concept": concept,
        "window": None if end is None else _window(start, end),
    }


def _measurement(concept: str, summary: str, op: str, value: float, **extra) -> dict:
    return {
        "kind": "measurement",
        "concept": concept,
        "summary": summary,
        "window": _window(),
        "op": op,
        "value": value,
        **extra,
    }


def _triples(item: CompiledCriterion) -> list[tuple]:
    return [
        (
            predicate.concept_id,
            predicate.aggregation,
            predicate.op,
            predicate.value,
            predicate.time_window.start_offset_hours,
            predicate.time_window.end_offset_hours,
        )
        for predicate in item.predicates
    ]


def _recorded(
    demographic: tuple[tuple[str, dict, int], ...] = (),
    *,
    icd: tuple[list[str], list[str]] | None = None,
    executed: dict[str, Any] | None = None,
    basis: str = "export_contract",
) -> tuple[dict[str, Any], int]:
    """A source selection whose count report chains, and the stays it leaves."""

    remaining = 1000
    steps = []
    for criterion, parameters, excluded in demographic:
        steps.append(
            {
                "criterion": criterion,
                "parameters": parameters,
                "n_before": remaining,
                "n_excluded": excluded,
                "n_remaining": remaining - excluded,
                "n_excluded_missing": 0,
            }
        )
        remaining -= excluded
    report: dict[str, Any] = {
        "count_unit": "icu_stay",
        "source_total": 1000,
        "demographic_steps": steps,
        "selected_before_concept_prefilter": remaining,
        "concept_matches": None,
        "selected_before_icd": remaining,
    }
    if icd is not None:
        report["icd"] = {
            "enabled": True,
            "include_tokens": icd[0],
            "exclude_tokens": icd[1],
        }
        remaining -= 50
    report["selected_before_cap"] = remaining
    report["selected"] = remaining
    record: dict[str, Any] = {
        "basis": basis,
        "host_applied": [],
        "export_report": report,
    }
    if executed is not None:
        record["executed_cohort"] = executed
    return {"source_selection": record}, remaining


def _recorded_ctx(*args, **kwargs) -> ResearchContext:
    constraints, remaining = _recorded(*args, **kwargs)
    return _ctx(constraints=constraints, n_stays=remaining)


# The spec states no predicate ------------------------------------------------


def test_a_status_has_no_threshold_to_write() -> None:
    with pytest.raises(ValidationError):
        _spec(_condition("sep3", op=">=", value=2))
    # The condition itself is stated as present, with no comparison.
    assert _spec(_condition("sep3")).criteria[0].kind == "condition_present"


@pytest.mark.parametrize(
    "criterion",
    [
        {"kind": "age_years", "min_years": 18},
        {"kind": "icu_stay_hours", "min_hours": 24},
        {"kind": "first_icu_stay"},
        {"kind": "alive_at", "hours": 24},
        {"kind": "event_absent", "concept": "vent", "window": _window()},
    ],
)
def test_a_criterion_that_states_the_stays_kept_is_an_inclusion(criterion) -> None:
    with pytest.raises(ValidationError, match="role include"):
        _spec({**criterion, "role": "exclude"})
    assert _spec(criterion).criteria[0].role == "include"


def test_an_excluded_condition_names_one_concept() -> None:
    # A stay meeting any exclusion is removed: two exclusions remove either.
    with pytest.raises(ValidationError, match="one concept"):
        _spec(_condition("sep3", "circ", role="exclude"))
    assert _spec(_condition("sep3", role="exclude")).criteria[0].role == "exclude"


@pytest.mark.parametrize(
    "criteria",
    [
        [_condition("sep3", start=24.0, end=24.0)],
        [{"kind": "age_years", "min_years": 80, "max_years": 18}],
        [{"kind": "age_years"}],
        [{"kind": "icu_stay_hours", "min_hours": 0}],
        [{"kind": "diagnosis_codes", "system": "icd10", "codes": ["A40-A41"]}],
        [{"kind": "diagnosis_codes", "system": "icd10", "codes": ["A41.9", "a419"]}],
        [{"kind": "measurement", **_measurement("lact", "max", ">", math.inf)}],
        [{"kind": "not_typed", "why": "short"}],
        [{"kind": "index_event", "concept": "vent"}],
        [{"kind": "age_years", "min_years": 18}] * 9,
    ],
)
def test_a_malformed_spec_is_refused(criteria) -> None:
    with pytest.raises(ValidationError):
        _spec(*criteria)


def test_a_spec_names_each_criterion_once() -> None:
    base = {"source": "question", "role": "include", "kind": "first_icu_stay"}
    with pytest.raises(ValidationError, match="ids"):
        PopulationSpec.model_validate(
            {"criteria": [{**base, "id": "c1", "quote": "first stay"}] * 2}
        )
    # The same restriction stated twice, whatever its words, is refused.
    with pytest.raises(ValidationError, match="same restriction"):
        _spec(
            {"kind": "age_years", "min_years": 18, "quote": "adults"},
            {"kind": "age_years", "min_years": 18, "quote": "aged 18 or older"},
        )
    # One sentence may state several restrictions.
    shared = _spec(
        {"kind": "age_years", "min_years": 18, "quote": "adults with shock"},
        {**_condition("circ"), "quote": "adults with shock"},
    )
    assert [item.quote for item in shared.criteria] == ["adults with shock"] * 2


# Every reason has a case -----------------------------------------------------


def _without_unit(name: str) -> ConceptDescriptor:
    (variable,) = [item for item in _VARIABLES if item.name == name]
    return variable.model_copy(update={"unit": None})


_REASON_CASES: list[tuple[str, dict, Any, float | None]] = [
    (
        "population_kind_not_typed",
        {"kind": "not_typed", "why": "transferred in from another hospital"},
        None,
        None,
    ),
    ("population_concept_unavailable", _condition("made_up_flag"), None, None),
    ("population_identifier_column", _condition("stay_id"), None, None),
    ("population_column_unresolved", _measurement("map", "min", "<", 65), None, None),
    ("population_condition_column_not_status", _condition("map"), None, None),
    (
        "population_unit_mismatch",
        _measurement("lact", "max", ">", 36, unit="mg/dL"),
        None,
        None,
    ),
    (
        "population_unit_unrecorded",
        {"kind": "icu_stay_hours", "min_hours": 24},
        lambda: _ctx(_without_unit("los_icu")),
        None,
    ),
    ("population_window_unreadable", _condition("sep3", end=48.0), None, None),
    ("population_whole_stay_in_finite_window", _condition("aki"), None, None),
    (
        "population_event_time_not_hours",
        {"kind": "alive_at", "hours": 24},
        lambda: _ctx(_event_time("death", unit="d")),
        None,
    ),
    (
        "population_threshold_outside_domain",
        _measurement("sep3", "max", ">=", 2),
        None,
        None,
    ),
    (
        "population_determined_after_time_zero",
        {"kind": "alive_at", "hours": 48},
        None,
        24.0,
    ),
    ("population_concept_not_in_export", _condition("vent_ind"), None, None),
    (
        "population_diagnosis_codes_need_extraction",
        {"kind": "diagnosis_codes", "system": "icd10", "codes": ["A41"]},
        None,
        None,
    ),
    (
        "population_first_icu_stay_not_restricted",
        {"kind": "first_icu_stay"},
        None,
        None,
    ),
]


def test_every_reason_code_has_a_case() -> None:
    assert sorted(case[0] for case in _REASON_CASES) == sorted(
        NOT_APPLIED_REASONS + REQUIRES_EXTRACTION_REASONS
    )


@pytest.mark.parametrize(
    "reason,criterion,context,time_zero",
    _REASON_CASES,
    ids=[case[0] for case in _REASON_CASES],
)
def test_a_criterion_is_not_applied_for_its_reason(
    reason, criterion, context, time_zero
) -> None:
    item = _one(criterion, context() if context else None, time_zero=time_zero)

    assert (item.disposition, item.reason) == (
        "requires_extraction"
        if reason in REQUIRES_EXTRACTION_REASONS
        else "not_applied",
        reason,
    )
    assert item.predicates == () and item.side is None and item.proof is None
    assert item.detail


# Applied by the plan's predicates --------------------------------------------


def test_an_age_range_is_applied_by_the_plans_predicates() -> None:
    item = _one({"kind": "age_years", "min_years": 18, "max_years": 80})

    assert (item.disposition, item.side) == ("applied_by_plan", "inclusion")
    # Age is fixed at admission: no window bounds it.
    assert _triples(item) == [
        ("age", "first", ">=", 18.0, 0.0, math.inf),
        ("age", "first", "<=", 80.0, 0.0, math.inf),
    ]


def test_an_age_recorded_in_another_unit_is_not_compared_in_years() -> None:
    months = _without_unit("age").model_copy(update={"unit": "months"})
    item = _one({"kind": "age_years", "min_years": 18}, _ctx(months))

    assert (item.disposition, item.reason) == (
        "not_applied",
        "population_unit_mismatch",
    )


def test_an_age_with_its_own_window_states_that_window() -> None:
    # A one-value column read through its own window label reads that window.
    labelled = _without_unit("age").model_copy(
        update={"unit": "years", "analysis_window": "icu_admission[0,24]h"}
    )
    item = _one({"kind": "age_years", "min_years": 18}, _ctx(labelled))

    assert _triples(item) == [("age", "first", ">=", 18.0, 0.0, 24.0)]


@pytest.mark.parametrize(
    "unit,value", [("days", 1.0), ("d", 1.0), ("hours", 24.0), ("h", 24.0)]
)
def test_a_stay_length_is_compared_in_its_columns_unit(unit, value) -> None:
    los = _without_unit("los_icu").model_copy(update={"unit": unit})
    item = _one({"kind": "icu_stay_hours", "min_hours": 24}, _ctx(los))

    assert _triples(item) == [("los_icu", "first", ">=", value, 0.0, math.inf)]


def test_a_stay_length_in_an_unread_unit_is_not_applied() -> None:
    weeks = _without_unit("los_icu").model_copy(update={"unit": "weeks"})
    item = _one({"kind": "icu_stay_hours", "min_hours": 24}, _ctx(weeks))

    assert item.reason == "population_unit_mismatch"


def test_a_minimum_stay_is_decided_by_time_zero() -> None:
    stay = {"kind": "icu_stay_hours", "min_hours": 48}

    assert _one(stay, time_zero=48.0).disposition == "applied_by_plan"
    late = _one(stay, time_zero=24.0)
    assert (late.disposition, late.reason) == (
        "not_applied",
        "population_determined_after_time_zero",
    )


def test_a_condition_is_present_when_each_status_equals_one() -> None:
    item = _one(_condition("sep3", "circ"))

    assert (item.disposition, item.side) == ("applied_by_plan", "inclusion")
    assert _triples(item) == [
        ("sep3", "max", "==", 1, 0.0, 24.0),
        ("circ", "max", "==", 1, 0.0, 24.0),
    ]
    excluded = _one(_condition("circ", role="exclude"))
    assert (excluded.side, _triples(excluded)) == (
        "exclusion",
        [("circ", "max", "==", 1, 0.0, 24.0)],
    )


def test_a_condition_with_its_event_time_is_read_inside_the_window_it_states() -> None:
    item = _one(_condition("vent", start=6.0, end=12.0))

    assert item.disposition == "applied_by_plan"
    assert _triples(item) == [("vent", "max", "==", 1, 6.0, 12.0)]


def test_a_condition_without_its_event_time_is_read_only_over_its_columns_window() -> (
    None
):
    # The host summarized circ over 0-24 h: 12 h reads nothing it recorded.
    assert _one(_condition("circ", end=12.0)).reason == "population_window_unreadable"
    assert _one(_condition("circ")).disposition == "applied_by_plan"


def test_a_condition_is_read_from_a_0_1_status_by_an_event_time_in_hours() -> None:
    # An ordinal score is no status; a time in days is no time in hours.
    assert _one(_condition("organ")).reason == "population_condition_column_not_status"
    in_days = _ctx(_event_time("vent", unit="d"))
    assert _one(_condition("vent", end=12.0), in_days).reason == (
        "population_event_time_not_hours"
    )


def test_a_measurement_is_compared_over_the_summary_its_column_holds() -> None:
    item = _one(_measurement("lact", "max", ">", 2, unit="mmol/L"))

    assert (item.disposition, _triples(item)) == (
        "applied_by_plan",
        [("lact", "max", ">", 2.0, 0.0, 24.0)],
    )
    # No column holds the mean; the bare map column holds no stated summary.
    assert _one(_measurement("lact", "mean", ">", 2)).reason == (
        "population_column_unresolved"
    )
    assert _one(_measurement("map", "min", "<", 65)).reason == (
        "population_column_unresolved"
    )


@pytest.mark.parametrize(
    "column,transform,summary,applied",
    [
        # A typed numeric summary, a typed presence and one value per stay.
        ("glucose", "window_numeric_max", "max", True),
        ("glucose", "window_numeric_max", "min", False),
        ("shock", "window_presence_max", "max", True),
        ("shock", "window_presence_first", "max", False),
        ("sofa_day1", "stay_level_unique_value", "mean", True),
        ("sofa_day1", None, "mean", False),
    ],
)
def test_a_bare_column_holds_the_summary_its_typing_states(
    column, transform, summary, applied
) -> None:
    typed = ConceptDescriptor(
        name=column,
        role=VariableRole.OTHER,
        dtype="float64",
        unit_normalization=transform,
    )
    item = _one(_measurement(column, summary, ">=", 1), _ctx(typed))

    assert item.applied is applied
    assert applied or item.reason == "population_column_unresolved"
    # A value fixed at admission is the same under every summary.
    assert _one(_measurement("age", "mean", ">=", 18)).applied
    # A 0/1 status holds its presence, the maximum, never its mean.
    assert _one(_measurement("sep3", "mean", ">=", 1)).reason == (
        "population_column_unresolved"
    )


def test_a_measurement_threshold_is_stated_in_its_columns_unit() -> None:
    assert _one(_measurement("lact", "max", ">", 2)).disposition == "applied_by_plan"
    assert _one(_measurement("lact", "max", ">", 2, unit=" MMOL / l ")).applied
    unitless = _without_unit("lact_max")
    assert (
        _one(_measurement("lact", "max", ">", 2, unit="mmol/L"), _ctx(unitless)).reason
        == "population_unit_unrecorded"
    )


def test_a_threshold_is_judged_against_the_values_its_column_takes() -> None:
    # A 0/1 diagnosis flag stated as a measurement of at least 2 keeps no row.
    assert _one(_measurement("sep3", "max", ">=", 2)).reason == (
        "population_threshold_outside_domain"
    )
    assert _one(_measurement("sep3", "max", ">=", 1)).applied
    assert _one(_measurement("organ", "max", ">=", 5)).reason == (
        "population_threshold_outside_domain"
    )
    assert _one(_measurement("organ", "max", ">=", 2)).applied
    # A bound every declared value meets is applied as stated and removes no stay.
    every = _one(_measurement("organ", "max", ">=", 0))
    assert every.applied and "keeps every row" in every.detail


def test_survival_to_an_hour_excludes_deaths_recorded_before_it() -> None:
    item = _one({"kind": "alive_at", "hours": 24}, time_zero=24.0)

    assert (item.disposition, item.side) == ("applied_by_plan", "exclusion")
    assert _triples(item) == [("death", "max", "==", 1, 0.0, 24.0)]
    assert "no death this input records before 24 h" in item.detail
    # Without its time, a death recorded over the whole stay is not placed.
    assert _one(
        {"kind": "alive_at", "hours": 24}, _ctx(without=("death_time",))
    ).reason == ("population_whole_stay_in_finite_window")


def test_an_absent_event_excludes_the_stays_with_it_in_the_window() -> None:
    item = _one(_absent("vent", end=6.0))

    assert (item.side, _triples(item)) == (
        "exclusion",
        [("vent", "max", "==", 1, 0.0, 6.0)],
    )


@pytest.mark.parametrize("end", [24.0, None])
@pytest.mark.parametrize("concept", ["sep3", "circ", "vent", "aki"])
def test_an_absent_event_compiles_as_its_excluded_condition(concept, end) -> None:
    # A status with its own window, the host's window, its event time, or
    # recorded over the whole stay with no time: one restriction, one result.
    absent = _one(_absent(concept, end=end))
    excluded = _one(_condition(concept, end=end, role="exclude"))

    assert (absent.disposition, absent.reason, absent.side, _triples(absent)) == (
        excluded.disposition,
        excluded.reason,
        excluded.side,
        _triples(excluded),
    )


def test_a_condition_over_the_whole_stay_reads_what_the_stay_records() -> None:
    whole = _one(_condition("aki", end=None))

    assert _triples(whole) == [("aki", "max", "==", 1, 0.0, math.inf)]
    # Recorded at the stay's end, it is known only after a time zero.
    assert _one(_condition("aki", end=None), time_zero=24.0).reason == (
        "population_determined_after_time_zero"
    )
    # A status read by its event's time is read over the whole stay too; one
    # summarized over the first day is not.
    assert _one(_condition("vent", end=None)).applied
    assert _one(_condition("sep3", end=None)).reason == "population_window_unreadable"


# Applied by the source -------------------------------------------------------


def test_an_export_step_shows_the_age_range_it_kept() -> None:
    adults = {"kind": "age_years", "min_years": 18}
    exact = _one(adults, _recorded_ctx((("age", {"age_min": 18}, 100),)))

    assert exact.disposition == "applied_by_source"
    assert exact.proof == SourceProof(
        kind="export_report_step",
        record_ref="source_selection.export_report",
        parameters={"criterion": "age", "age_min": 18},
    )
    # An export that kept a narrower range kept only stays the criterion keeps.
    assert _one(adults, _recorded_ctx((("age", {"age_min": 21}, 120),))).proof
    # One that kept a wider range did not apply it: the plan does.
    wider = _one(adults, _recorded_ctx((("age", {"age_min": 16}, 80),)))
    assert wider.disposition == "applied_by_plan"
    capped = _one(
        {"kind": "age_years", "min_years": 18, "max_years": 65},
        _recorded_ctx((("age", {"age_min": 18}, 100),)),
    )
    assert capped.disposition == "applied_by_plan"


def test_an_export_step_shows_the_minimum_stay_it_kept() -> None:
    context = _recorded_ctx((("los", {"los_min": 24}, 200),))

    assert _one({"kind": "icu_stay_hours", "min_hours": 24}, context).proof
    assert _one({"kind": "icu_stay_hours", "min_hours": 12}, context).proof
    assert (
        _one({"kind": "icu_stay_hours", "min_hours": 48}, context).disposition
        == "applied_by_plan"
    )


def test_the_executed_study_contract_shows_the_ranges_it_kept() -> None:
    context = _recorded_ctx(
        executed={"age_min": 18, "age_max": None, "min_icu_los_hours": 24}
    )
    age = _one({"kind": "age_years", "min_years": 18}, context)
    stay = _one({"kind": "icu_stay_hours", "min_hours": 24}, context)

    assert (age.proof.kind, age.proof.parameters) == (
        "recorded_study_contract",
        {"age_min": 18, "age_max": None},
    )
    assert stay.proof.parameters == {"min_icu_los_hours": 24}
    # A neutral upper bound shows no stated maximum.
    capped = _one({"kind": "age_years", "min_years": 18, "max_years": 65}, context)
    assert capped.disposition == "applied_by_plan"


def test_an_unrecorded_selection_shows_nothing() -> None:
    constraints, remaining = _recorded(
        (("age", {"age_min": 18}, 100),),
        executed={"age_min": 18, "first_icu_stay": True},
        basis="unrecorded",
    )
    context = _ctx(constraints=constraints, n_stays=remaining)

    assert _one({"kind": "age_years", "min_years": 18}, context).proof is None
    assert _one({"kind": "first_icu_stay"}, context).reason == (
        "population_first_icu_stay_not_restricted"
    )


def test_the_hosts_first_stay_step_shows_the_first_icu_stay() -> None:
    constraints, remaining = _recorded()
    restriction = {
        "stays_before": remaining,
        "stays_after": remaining - 30,
        "non_first_icu_stays_removed": 30,
    }
    context = _ctx(
        constraints=constraints,
        n_stays=remaining - 30,
        provenance={"first_icu_stay_restriction": restriction},
    )
    item = _one({"kind": "first_icu_stay"}, context)

    assert (item.proof.kind, item.proof.parameters) == (
        "host_first_icu_stay",
        {"criterion": "first_icu_stay_restriction"},
    )


def test_the_first_icu_stay_is_applied_only_by_a_record_of_it() -> None:
    receipt = {
        "schema_version": FIRST_ICU_STAY_RESTRICTION_SCHEMA,
        "coordinate_sha256": "0" * 64,
    }
    first = {"kind": "first_icu_stay"}

    host = _one(first, _ctx(provenance={"first_icu_stay_restriction": receipt}))
    assert (host.disposition, host.proof.kind) == (
        "applied_by_source",
        "host_first_icu_stay",
    )
    export = _one(
        first, _recorded_ctx((("first_icu_stay", {"first_icu_stay": True}, 90),))
    )
    assert export.proof.kind == "export_report_step"
    contract = _one(first, _recorded_ctx(executed={"first_icu_stay": True}))
    assert contract.proof.kind == "recorded_study_contract"
    stale = _one(
        first,
        _ctx(
            provenance={
                "first_icu_stay_restriction": {**receipt, "schema_version": "x"}
            }
        ),
    )
    assert stale.disposition == "requires_extraction"


def test_diagnosis_codes_are_applied_only_by_an_export_that_applied_them() -> None:
    sepsis = {"kind": "diagnosis_codes", "system": "icd10", "codes": ["A41.9", "a40"]}

    export = _one(sepsis, _recorded_ctx(icd=(["A40", "A419"], [])))
    assert (export.disposition, export.proof.parameters) == (
        "applied_by_source",
        {"criterion": "icd", "include": ["A40", "A419"]},
    )
    contract = _one(sepsis, _recorded_ctx(executed={"icd_include": ["A419", "A40"]}))
    assert contract.proof.kind == "recorded_study_contract"
    # A record that keeps a code as written is read as an export matches it.
    written = _one(sepsis, _recorded_ctx(executed={"icd_include": ["a41.9", " A40"]}))
    assert written.proof.kind == "recorded_study_contract"
    # Another code set, or the codes on the other side, applies something else.
    assert _one(sepsis, _recorded_ctx(icd=(["A41"], []))).reason == (
        "population_diagnosis_codes_need_extraction"
    )
    assert not _one(sepsis, _recorded_ctx(icd=([], ["A40", "A419"]))).applied
    excluded = _one(
        {**sepsis, "role": "exclude"}, _recorded_ctx(icd=([], ["A40", "A419"]))
    )
    assert excluded.disposition == "applied_by_source"
    # E and V begin codes of both versions: the stays kept for an ICD-10 E87
    # may hold ICD-9 E870-E879 instead.  The stays left after removing them
    # hold neither.
    endocrine = {"kind": "diagnosis_codes", "system": "icd10", "codes": ["E87"]}
    assert _one(endocrine, _recorded_ctx(icd=(["E87"], []))).reason == (
        "population_diagnosis_codes_need_extraction"
    )
    removed = _one({**endocrine, "role": "exclude"}, _recorded_ctx(icd=([], ["E87"])))
    assert removed.disposition == "applied_by_source"


# The compiled population ------------------------------------------------------


def _mixed() -> PopulationSpec:
    return _spec(
        {"kind": "age_years", "min_years": 18},
        _condition("sep3", "circ"),
        _condition("aki", role="exclude"),
        {"kind": "alive_at", "hours": 24},
        {"kind": "not_typed", "why": "no prior cardiac surgery this admission"},
        {
            "kind": "diagnosis_codes",
            "system": "icd9",
            "codes": ["995.92"],
            "role": "exclude",
        },
        {"kind": "first_icu_stay"},
    )


def test_every_criterion_has_one_disposition_and_every_predicate_one_criterion() -> (
    None
):
    spec = _mixed()
    compiled = compile_population(spec, _ctx(), time_zero_hours=24.0)
    cohort = compiled.cohort_definition()

    assert [item.criterion for item in compiled.criteria] == list(spec.criteria)
    assert [item.disposition for item in compiled.criteria] == [
        "applied_by_plan",
        "applied_by_plan",
        "not_applied",
        "applied_by_plan",
        "not_applied",
        "requires_extraction",
        "requires_extraction",
    ]
    owned = [predicate for item in compiled.criteria for predicate in item.predicates]
    assert sorted(map(id, owned)) == sorted(
        map(id, (*cohort.inclusion, *cohort.exclusion))
    )
    assert len(cohort.inclusion) == 3 and len(cohort.exclusion) == 1


def test_the_cohort_lists_the_criteria_it_does_not_apply() -> None:
    compiled = compile_population(_mixed(), _ctx(), time_zero_hours=24.0)
    cohort = compiled.cohort_definition()

    assert cohort.selection_mode == "predicate_filtered"
    assert cohort.unapplied_population_criteria == (
        "criterion number 3",
        "criterion number 5",
        "criterion number 6",
        "criterion number 7",
    )
    # Only an inclusion the analysis would not apply waits for the user.
    assert [item.criterion.id for item in compiled.blocking] == ["c5", "c7"]
    # Criteria that share their sentence list it once.
    shared = compile_population(
        _spec(
            {**_condition("made_up_flag"), "quote": "frail older adults"},
            {
                "kind": "not_typed",
                "why": "no frailty concept in this input",
                "quote": "frail older adults",
            },
        ),
        _ctx(),
    )
    assert shared.cohort_definition().unapplied_population_criteria == (
        "frail older adults",
    )


def test_a_population_the_source_selected_keeps_all_input_rows() -> None:
    context = _recorded_ctx(
        (
            ("age", {"age_min": 18}, 100),
            ("first_icu_stay", {"first_icu_stay": True}, 90),
        )
    )
    compiled = compile_population(
        _spec({"kind": "age_years", "min_years": 18}, {"kind": "first_icu_stay"}),
        context,
    )
    cohort = compiled.cohort_definition()

    assert (cohort.selection_mode, cohort.inclusion, cohort.exclusion) == (
        "all_input_rows",
        (),
        (),
    )
    assert cohort.unapplied_population_criteria == () and compiled.blocking == ()


def test_an_empty_spec_selects_all_input_rows() -> None:
    compiled = compile_population(PopulationSpec(), _ctx())

    assert compiled.criteria == ()
    assert compiled.cohort_definition().selection_mode == "all_input_rows"


def test_compiling_is_pure_and_its_record_is_stable() -> None:
    context = _ctx()
    before = context.model_dump(mode="json")
    first = compile_population(_mixed(), context, time_zero_hours=24.0)
    second = compile_population(_mixed(), context, time_zero_hours=24.0)

    assert context.model_dump(mode="json") == before
    # The columns made known for the compile are forgotten after it.
    assert not concept_id_exists("circ")
    assert first.sha256() == second.sha256()
    record = json.loads(json.dumps(first.record()))
    assert record["schema_version"] == POPULATION_COMPILE_SCHEMA_VERSION
    assert [item["criterion"]["id"] for item in record["criteria"]] == [
        f"c{index}" for index in range(1, 8)
    ]
    assert record["cohort"]["unapplied_population_criteria"]
    assert compile_population(_mixed(), context).sha256() != first.sha256()


@pytest.mark.parametrize("hours", [math.inf, math.nan, True])
def test_time_zero_is_a_finite_hour(hours) -> None:
    with pytest.raises(ValueError, match="finite"):
        compile_population(_mixed(), _ctx(), time_zero_hours=hours)


def test_a_compiled_criterion_keeps_its_parts_consistent() -> None:
    (criterion,) = _spec({"kind": "first_icu_stay"}).criteria
    proof = SourceProof("host_first_icu_stay", "cohort.provenance", {})
    with pytest.raises(ValueError):
        CompiledCriterion(criterion, "applied_by_plan", None, "no predicates")
    with pytest.raises(ValueError):
        CompiledCriterion(criterion, "applied_by_source", None, "no proof")
    with pytest.raises(ValueError):
        CompiledCriterion(criterion, "not_applied", None, "no reason")
    with pytest.raises(ValueError):
        CompiledCriterion(
            criterion, "not_applied", "population_concept_not_in_export", "wrong list"
        )
    with pytest.raises(ValueError):
        CompiledCriterion(
            criterion,
            "requires_extraction",
            "population_first_icu_stay_not_restricted",
            "x",
            proof=proof,
        )
    assert CompiledCriterion(
        criterion, "applied_by_source", None, "ok", proof=proof
    ).applied


# Studies of several shapes -----------------------------------------------------


def _metadata_only(*names: str) -> tuple[ConceptDescriptor, ...]:
    """Status columns as a row-free planning catalog types them: declared, not observed."""

    return tuple(
        ConceptDescriptor(
            name=name, role=VariableRole.OTHER, dtype="float64", source_concept=name
        )
        for name in names
    )


def test_an_adult_septic_shock_population_compiles_to_the_predicates_it_states() -> (
    None
):
    # Septic shock as two statuses present in the first ICU day, in adults:
    # one sentence states both restrictions.
    context = _ctx(*_metadata_only("sep3_sofa1", "circ_failure"))
    stated = "adult patients with septic shock in the first 24 h"
    spec = _spec(
        {"kind": "age_years", "min_years": 18, "quote": stated},
        _condition("sep3_sofa1", "circ_failure", quote=stated),
    )
    compiled = compile_population(spec, context, time_zero_hours=24.0)

    assert [item.disposition for item in compiled.criteria] == ["applied_by_plan"] * 2
    assert [
        (p.concept_id, p.op, p.value) for p in compiled.cohort_definition().inclusion
    ] == [("age", ">=", 18.0), ("sep3_sofa1", "==", 1), ("circ_failure", "==", 1)]
    # The same flag with a threshold copied from its description is refused.
    copied = _one(_measurement("sep3_sofa1", "max", ">=", 2), context)
    assert copied.reason == "population_threshold_outside_domain"


def test_a_ventilated_population_alive_at_its_landmark_compiles() -> None:
    # Not a benchmark question: ventilated in the first day, alive at 24 h, in
    # adults up to 90, excluding lactate above 4 mmol/L in the first day.
    adults = {"kind": "age_years", "min_years": 18, "max_years": 90}
    lactate = _measurement("lact", "max", ">", 4, unit="mmol/L", role="exclude")
    spec = _spec(adults, _condition("vent"), {"kind": "alive_at", "hours": 24}, lactate)
    compiled = compile_population(spec, _ctx(), time_zero_hours=24.0)

    assert [item.disposition for item in compiled.criteria] == ["applied_by_plan"] * 4
    assert compiled.blocking == ()
    # With time zero at 6 h, a lactate summarized over 0-24 h is known later.
    early = compile_population(_spec(adults, lactate), _ctx(), time_zero_hours=6.0)
    assert [(item.disposition, item.reason) for item in early.criteria] == [
        ("applied_by_plan", None),
        ("not_applied", "population_determined_after_time_zero"),
    ]
    # An exclusion left unapplied keeps more stays: approval does not wait on it.
    assert early.blocking == ()


def test_a_minimum_stay_beyond_the_landmark_is_not_applied_and_waits_for_the_user() -> (
    None
):
    spec = _spec(
        {"kind": "icu_stay_hours", "min_hours": 72, "quote": "ICU stay of 3 days"},
        _condition("circ"),
    )
    compiled = compile_population(spec, _ctx(), time_zero_hours=24.0)

    assert [item.criterion.id for item in compiled.blocking] == ["c1"]
    assert compiled.cohort_definition().unapplied_population_criteria == (
        "ICU stay of 3 days",
    )


# A typed input, end to end -------------------------------------------------------


def test_a_compiled_population_keeps_the_stays_it_states(tmp_path: Path) -> None:
    # Ages 50, 55, 60, 65; stay 3 dies 12 h after ICU admission and stays 0.6 days.
    export = typed_native_export(
        tmp_path / "export",
        outcome=native_outcome(
            death=[True, False, True, False],
            death_time=[87.0, None, 12.0, None],
            los_icu=[4.0, 1.5, 0.6, 2.0],
            mort_28d=[True, False, True, False],
        ),
        outcome_concepts=["death", "los_icu", "mort_28d"],
    )
    settings = dict(
        feature_concepts=["los_icu", "death"],
        database="miiv",
        data_path=str(export),
        cohort_window=(0.0, 24.0),
        outcome_concepts=["mort_28d"],
        static_concepts=["age"],
    )
    paths = materialize_to_parquet(tmp_path / "universe", **settings)
    context = build_research_context(
        research_question="Who is alive one day after ICU admission?",
        cohort=paths["parquet"],
        cohort_name="synthetic",
        database="miiv",
        target_outcome="mort_28d",
    )
    spec = _spec(
        {"kind": "age_years", "min_years": 55},
        {"kind": "alive_at", "hours": 24},
        {"kind": "icu_stay_hours", "min_hours": 24},
    )
    compiled = compile_population(spec, context, time_zero_hours=24.0)

    assert [item.disposition for item in compiled.criteria] == ["applied_by_plan"] * 3
    cohort, _provenance = materialize_cohort(
        **settings, cohort_definition=compiled.cohort_definition()
    )
    assert sorted(int(stay) for stay in cohort["stay_id"]) == [2, 4]
