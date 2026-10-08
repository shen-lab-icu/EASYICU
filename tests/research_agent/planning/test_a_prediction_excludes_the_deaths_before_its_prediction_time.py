"""A prediction's risk set keeps no stay whose death was recorded before the prediction time.

A static prediction model predicts at the end of its observation window for
the stays still in the ICU after it.  The ICU length of stay decides that on
every source, but a stay can stay in the ICU after its recorded death: MIMIC
records the ICU discharge after the death, so a stay that died at 20 h and
left the ICU at 26 h entered a 24-hour model as a case scored on the vitals of
a dying stay.  Where the export's producer labels its death time as recorded
to the hour, the risk set now also keeps only the stays without a death
recorded before the prediction time, read by that time, as a row of its own in
the cohort's ledger; elsewhere the plan states why it does not.  A context
without the launch's record keeps its request and its words.  Synthetic
contexts and rows only.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from easyicu.research_agent.agents.family_spec_planner import _population_authority
from easyicu.research_agent.cohort.schema import (
    CohortDataError,
    _build_cohort_with_flow,
    build_cohort,
)
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.planning.cohort_eligibility import (
    cohort_predicates_after_time_zero,
)
from easyicu.research_agent.planning.family_spec.contract import spec_from_mapping
from easyicu.research_agent.planning.family_spec.prediction_template import (
    build_prediction_skeleton,
)
from easyicu.research_agent.planning.family_spec.request import _prediction_death_time
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ObservationSemantics,
    ResearchContext,
    VariableRole,
)

from .family_spec_fixtures import (
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
)

FEATURES = ["age", "sex", "hr_max", "lactate_max", "map_min"]
RECORDED = "recorded_deathtime"
OFFSET = "recorded_offset_of_death_for_hospital_discharge_death"
DATE_PROXY = "recorded_dateofdeath_for_72h_post_icu_discharge_death_proxy"
LAST_OBSERVATION = "last_recorded_observation_proxy_for_dead_discharge"
CHINESE = "在成人 ICU 入住中，入 ICU 后前 24 小时的生命体征、化验与人口学特征对院内死亡的预测效果如何？"
CONFLICT = (
    "Planner primary cohort selection mode does not match the caller-bound contract"
)


def _death_time(
    *, status: str = "death", origin: str = "icu_admission", unit: str = "h"
) -> ConceptDescriptor:
    """The death time the export issues beside the status, typed as its time after admission."""

    return ConceptDescriptor(
        name="death_time",
        description="time of death",
        role=VariableRole.TIME,
        dtype="float64",
        source_concept="death",
        temporal_resolution=f"relative to {origin} in {unit}",
        observation_semantics=ObservationSemantics(
            kind="conditional_event_time",
            event_status_column=status,
            representative_column="death_time",
            time_origin=origin,
            time_unit=unit,
        ),
    )


def _recorded(
    context: ResearchContext,
    labels: dict[str, str],
    *,
    companion: ConceptDescriptor | None = None,
    with_companion: bool = True,
) -> ResearchContext:
    """The context a launch writes: the export's labels recorded, its death time typed."""

    constraints = json.loads(context.user_preferences.data_constraints or "{}")
    constraints["event_time_semantics"] = labels
    variables = [item for item in context.variables if item.name != "death_time"]
    if with_companion:
        variables.append(companion or _death_time())
    return context.model_copy(
        update={
            "variables": variables,
            "user_preferences": context.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            ),
        }
    )


def _hourly(context: ResearchContext | None = None) -> ResearchContext:
    return _recorded(context or _prediction_context(), {"death_time": RECORDED})


def _in_chinese(context: ResearchContext) -> ResearchContext:
    return context.model_copy(update={"research_question": CHINESE})


def _design(context: ResearchContext, mode: str | None = None):
    request = _request(context, cohort_mode=mode)
    skeleton = build_prediction_skeleton(
        request, spec_from_mapping(_prediction_payload(request, features=FEATURES))
    )
    return request, skeleton.outline.design_selection.candidates[0], skeleton


def _plan(context: ResearchContext, mode: str | None = None):
    request = _request(context, cohort_mode=mode)
    llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=FEATURES))],
        required_primary_cohort_selection_mode=mode,
    )
    assert len(llm.calls) == 1
    return request, result.output


def _death_row(
    *, op: str = "==", value: object = False, start: float = -24.0, end: float = 24.0
):
    return ConceptPredicate(
        concept_id="death",
        time_window=TimeWindow(
            anchor="icu_admission", start_offset_hours=start, end_offset_hours=end
        ),
        aggregation="any",
        op=op,
        value=value,
    )


def _stays() -> pd.DataFrame:
    """Stays still in the ICU after 24 h but for the first; deaths timed in hours."""

    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6, 7],
            # Left the ICU at 12 h; then still there after 24 h.
            "los_icu": [0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "death": [False, True, True, True, True, False, True],
            # Died at 10 h but left the ICU at 48 h; died exactly at 24 h; at
            # 30 h; without a recorded time; a survivor; recorded 2 h before
            # ICU admission.
            "death_time": [math.nan, 10.0, 24.0, 30.0, math.nan, math.nan, -2.0],
        }
    )


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ({"death_time": RECORDED}, (True, RECORDED, None)),
        ({"death_time": OFFSET}, (True, OFFSET, None)),
        ({"death_time": DATE_PROXY}, (False, DATE_PROXY, "death_time_resolution")),
        (
            {"death_time": LAST_OBSERVATION},
            (False, LAST_OBSERVATION, "death_time_resolution"),
        ),
        (
            {"death_time": "structurally_unavailable"},
            (False, "structurally_unavailable", "death_time_resolution"),
        ),
        (
            {"death_time": "source_event_time"},
            (False, "source_event_time", "death_time_resolution"),
        ),
        ({}, (False, None, "death_time_semantics_unrecorded")),
        (
            {"death_time": "a_label_nobody_wrote"},
            (False, "a_label_nobody_wrote", "death_time_semantics_unrecorded"),
        ),
    ],
    ids=[
        "MIMIC's recorded death time",
        "SICdb's recorded offset",
        "AmsterdamUMCdb's date of death",
        "HiRID's last observation",
        "eICU, which issues none",
        "an export naming no database",
        "an export that labels none",
        "a label the producer never wrote",
    ],
)
def test_deaths_are_read_by_their_time_only_where_it_is_recorded_to_the_hour(
    labels: dict[str, str], expected: tuple
) -> None:
    reading = _request(
        _recorded(_prediction_context(), labels), cohort_mode=None
    ).prediction_death_time

    assert (reading.applied, reading.semantics, reading.reason) == expected
    assert (reading.absent_level is False) is reading.applied


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"with_companion": False}, "death_time_companion_absent"),
        ({"companion": _death_time(status="los_flag")}, "death_time_companion_absent"),
        (
            {"companion": _death_time(origin="hospital_admission")},
            "death_time_companion_absent",
        ),
        ({"companion": _death_time(unit="weeks")}, "death_time_companion_absent"),
        # The builder compares the time with the window's hours.
        ({"companion": _death_time(unit="d")}, "death_time_companion_absent"),
    ],
    ids=[
        "no death time",
        "the time of another event",
        "timed from another origin",
        "an unread unit",
        "counted in days",
    ],
)
def test_an_hourly_death_time_counts_only_as_the_death_status_own_time(
    change, reason
) -> None:
    context = _recorded(_prediction_context(), {"death_time": RECORDED}, **change)

    reading = _prediction_death_time(context)

    assert (reading.applied, reading.reason, reading.semantics) == (
        False,
        reason,
        RECORDED,
    )


@pytest.mark.parametrize(
    "variables",
    [
        lambda items: [item for item in items if item.name != "death"],
        lambda items: [
            item.model_copy(
                update={"observed_domain": {"n_unique": 3, "levels": [0, 1, 2]}}
            )
            if item.name == "death"
            else item
            for item in items
        ],
    ],
    ids=["no death status", "a status of three levels"],
)
def test_a_risk_set_needs_a_death_status_of_two_levels(variables) -> None:
    context = _hourly()
    context = context.model_copy(update={"variables": variables(context.variables)})

    reading = _prediction_death_time(context)

    assert (reading.applied, reading.reason) == (False, "death_status_absent")


def test_a_context_without_the_record_keeps_its_request_and_its_words() -> None:
    """A context written before the launch's record renders what it rendered."""

    for context, population, outcome in (
        (
            _prediction_context(),
            "Analysis rows of the study cohort that are still in the ICU after the prediction "
            "time (24 h after ICU admission); each analysis row is one ICU stay and rows are "
            "not assumed to be distinct patients.",
            "In-hospital death, from the outcome record of these stays; a stay that left the "
            "ICU, alive or dead, by the prediction time is not analyzed.",
        ),
        (
            _in_chinese(_prediction_context()),
            "研究队列中在预测时点（ICU 入院后 24 h）之后仍在 ICU 内的分析行；每行为一次 ICU 入住，"
            "不假定各行来自不同患者。",
            "In-hospital death，取自这些入住的结局记录；在预测时点之前离开 ICU（存活或死亡）的入住"
            "不纳入分析。",
        ),
    ):
        request, design, skeleton = _design(context)

        assert request.prediction_death_time is None
        assert "prediction_death_time" not in request.model_dump(mode="json")
        assert design.reviewable_plan[0] == population
        assert design.reviewable_plan[2] == outcome
        assert design.assumptions[-1] == (
            "A stay leaves the analysis only by leaving the ICU, alive or dead, by the "
            "prediction time; an outcome event that does not end the ICU stay before it is "
            "not excluded."
        )
        assert [
            item.concept_id for item in skeleton.foundation.foundation.cohort.inclusion
        ] == ["los_icu"]
        assert (
            "no_death_recorded_before_prediction_time_hours"
            not in _population_authority(request)["already_applied"]
        )


def test_the_plan_states_the_death_row_it_applies() -> None:
    request, design, skeleton = _design(_hourly())

    assert design.reviewable_plan[0].startswith(
        "Analysis rows of the study cohort that are still in the ICU after the prediction "
        "time (24 h after ICU admission) with no death recorded before it;"
    )
    assert design.reviewable_plan[2] == (
        "In-hospital death, from the outcome record of these stays; a stay that left the ICU, "
        "alive or dead, by the prediction time, or with a death recorded before it, is not "
        "analyzed."
    )
    assert design.assumptions[-1] == (
        "A stay leaves the analysis by leaving the ICU by the prediction time, or by a death "
        "recorded before it; a death without a recorded time, and an outcome event other than "
        "death that does not end the ICU stay before it, are not excluded."
    )
    assert (
        _population_authority(request)["already_applied"][
            "no_death_recorded_before_prediction_time_hours"
        ]
        == 24.0
    )
    [_los, death] = skeleton.foundation.foundation.cohort.inclusion
    assert death.model_dump(mode="json") == {
        "concept_id": "death",
        "anchor": "icu_admission",
        # The export's pre-admission context, up to the prediction time.
        "start_offset_hours": -24.0,
        "end_offset_hours": 24.0,
        "aggregation": "any",
        "op": "==",
        "value": {
            "mode": "boolean",
            "string_value": None,
            "number_value": None,
            "boolean_value": False,
            "string_list": [],
            "number_list": [],
        },
    }

    _request_, chinese, _skeleton = _design(_in_chinese(_hourly()))
    assert chinese.reviewable_plan[0].startswith(
        "研究队列中在预测时点（ICU 入院后 24 h）之后仍在 ICU 内且此前无记录死亡的分析行；"
    )
    assert chinese.reviewable_plan[2] == (
        "In-hospital death，取自这些入住的结局记录；在预测时点之前离开 ICU（存活或死亡）"
        "或在此之前有记录死亡的入住不纳入分析。"
    )


@pytest.mark.parametrize(
    ("labels", "with_companion", "english", "chinese"),
    [
        (
            {"death_time": DATE_PROXY},
            True,
            f"the source's death time is not recorded to the hour ({DATE_PROXY})",
            f"来源的死亡时间未记录到小时（{DATE_PROXY}）",
        ),
        (
            {},
            True,
            "the export does not label what its death time is",
            "导出未标明其死亡时间是什么",
        ),
        (
            {"death_time": RECORDED},
            False,
            "the study's data carry no death time after ICU admission",
            "研究数据中没有 ICU 入院后的死亡时间",
        ),
    ],
    ids=["a date of death", "no label", "no death time"],
)
def test_the_plan_states_why_it_applies_no_death_row(
    labels, with_companion, english, chinese
) -> None:
    context = _recorded(_prediction_context(), labels, with_companion=with_companion)
    request, design, skeleton = _design(context)

    # The rows are what they were; the plan says what is left unchecked.
    assert [
        item.concept_id for item in skeleton.foundation.foundation.cohort.inclusion
    ] == ["los_icu"]
    assert design.reviewable_plan[2] == (
        "In-hospital death, from the outcome record of these stays; a stay that left the "
        "ICU, alive or dead, by the prediction time is not analyzed. A stay still in the ICU "
        f"with a death recorded before the prediction time is not excluded, as {english}."
    )
    assert design.assumptions[-1] == (
        "A stay leaves the analysis only by leaving the ICU, alive or dead, by the "
        "prediction time; an outcome event that does not end the ICU stay before it is "
        "not excluded. Nor is a death recorded before the prediction time while the stay "
        f"remains in the ICU: {english}."
    )
    assert (
        "no_death_recorded_before_prediction_time_hours"
        not in _population_authority(request)["already_applied"]
    )

    _request_, in_chinese, _skeleton = _design(_in_chinese(context))
    assert in_chinese.reviewable_plan[2].endswith(
        f"仍在 ICU 内、但在预测时点之前已有记录死亡的入住未被排除，因为{chinese}。"
    )


def test_the_host_keeps_no_stay_whose_death_was_recorded_before_the_prediction_time() -> (
    None
):
    context = _hourly()
    _request_, plan = _plan(context)

    assert [item.concept_id for item in plan.cohort.inclusion] == ["los_icu", "death"]
    # The plan's own review passes it: decided when the prediction is made.
    assert (
        cohort_predicates_after_time_zero(
            context,
            inclusion=[item.to_dict() for item in plan.cohort.inclusion],
            exclusion=[],
            time_zero_hours=24.0,
        )
        == ()
    )

    kept, flow = _build_cohort_with_flow(plan.cohort, _stays())

    # A death at the prediction time or after it, one without a recorded time,
    # and the survivor stay; the death at 10 h and the one recorded before ICU
    # admission do not.
    assert kept["stay_id"].tolist() == [3, 4, 5, 6]
    [_universe, stay, death] = flow
    assert (stay["concept_id"], stay["n_excluded"]) == ("los_icu", 1)
    assert {
        key: death[key]
        for key in (
            "predicate_kind",
            "concept_id",
            "n_before",
            "n_excluded",
            "n_remaining",
            "n_excluded_missing",
            "event_time_column",
            "event_time_start_hours",
            "event_time_end_hours",
            "event_time_reading",
        )
    } == {
        "predicate_kind": "inclusion",
        "concept_id": "death",
        "n_before": 6,
        "n_excluded": 2,
        "n_remaining": 4,
        "n_excluded_missing": 0,
        "event_time_column": "death_time",
        "event_time_start_hours": -24.0,
        "event_time_end_hours": 24.0,
        "event_time_reading": "absence",
    }


def test_a_source_without_a_death_time_applies_no_death_row_and_builds() -> None:
    """eICU issues no death time: no row is added, and nothing refuses the cohort."""

    context = _recorded(
        _prediction_context(), {"death_time": "structurally_unavailable"}
    )
    _request_, plan = _plan(context)
    rows = _stays().assign(death_time=math.nan)

    assert [item.concept_id for item in plan.cohort.inclusion] == ["los_icu"]
    assert build_cohort(plan.cohort, rows)["stay_id"].tolist() == [2, 3, 4, 5, 6, 7]
    # The row would be refused there: no death in the table has a recorded time.
    forced = CohortDefinition(
        name=plan.cohort.name,
        inclusion=(*plan.cohort.inclusion, _death_row()),
        exclusion=(),
    )
    with pytest.raises(CohortDataError, match="has a recorded time"):
        build_cohort(forced, rows)


def _judged(
    predicate: dict, *, kind: str = "inclusion", context: ResearchContext | None = None
):
    lists = {"inclusion": [], "exclusion": []}
    lists[kind] = [predicate]
    return cohort_predicates_after_time_zero(
        context or _hourly(), time_zero_hours=24.0, **lists
    )


@pytest.mark.parametrize(
    ("kind", "change"),
    [
        ("inclusion", {}),
        ("inclusion", {"op": "!=", "value": True}),
        ("exclusion", {"value": True}),
        ("inclusion", {"start": 0.0}),
    ],
    ids=[
        "no death in the window",
        "not dead in it",
        "excluding a death in it",
        "from admission",
    ],
)
def test_the_time_zero_rule_decides_an_absent_death_when_its_window_ends(
    kind, change
) -> None:
    predicate = _death_row(**change).to_dict()

    assert _judged(predicate, kind=kind) == ()


@pytest.mark.parametrize(
    ("kind", "predicate", "context"),
    [
        ("inclusion", _death_row(end=25.0).to_dict(), None),
        ("inclusion", _death_row(value=True).to_dict(), None),
        ("exclusion", _death_row(value=False).to_dict(), None),
        ("inclusion", {**_death_row().to_dict(), "op": ">", "value": 0}, None),
        (
            "inclusion",
            _death_row().to_dict(),
            _recorded(
                _prediction_context(), {"death_time": RECORDED}, with_companion=False
            ),
        ),
        (
            "inclusion",
            _death_row().to_dict(),
            _recorded(
                _prediction_context(),
                {"death_time": RECORDED},
                companion=_death_time(unit="d"),
            ),
        ),
        (
            "inclusion",
            {
                **_death_row().to_dict(),
                "time_window": {
                    **_death_row().to_dict()["time_window"],
                    "anchor": "hospital_admit",
                },
            },
            None,
        ),
        (
            "inclusion",
            {
                **_death_row().to_dict(),
                "time_window": {
                    "anchor": "icu_admission",
                    "start_offset_hours": 30.0,
                    "end_offset_hours": 24.0,
                },
            },
            None,
        ),
        (
            "inclusion",
            {
                **_death_row().to_dict(),
                "time_window": {
                    "anchor": "icu_admission",
                    "start_offset_hours": "-inf",
                    "end_offset_hours": 24.0,
                },
            },
            None,
        ),
    ],
    ids=[
        "a window ending after time zero",
        "keeping the deaths in it",
        "keeping only those deaths",
        "a magnitude test",
        "no death time in the context",
        "a death time in days",
        "another anchor",
        "a start after the end",
        "an unbounded start",
    ],
)
def test_any_other_test_of_the_outcome_is_still_refused(
    kind, predicate, context
) -> None:
    [found] = _judged(predicate, kind=kind, context=context)

    assert found.reason == "stay_outcome"


def test_the_caller_bound_population_admits_the_risk_set_with_its_death_row() -> None:
    from easyicu.research_agent.planning.family_spec import (
        caller_bound_population_conflict,
    )

    context = _hourly()
    request, plan = _plan(context, mode="all_input_rows")

    assert request.caller_binds_all_input_rows
    assert [item.concept_id for item in plan.cohort.inclusion] == ["los_icu", "death"]
    assert (
        caller_bound_population_conflict(
            plan, context=context, required_selection_mode="all_input_rows"
        )
        is None
    )
    without_death = plan.model_copy(
        update={
            "cohort": CohortDefinition(
                name=plan.cohort.name, inclusion=plan.cohort.inclusion[:1], exclusion=()
            )
        }
    )
    conflict = caller_bound_population_conflict(
        without_death, context=context, required_selection_mode="all_input_rows"
    )
    assert conflict is not None and conflict.startswith(CONFLICT)
    # Where no death row applies, the stay row alone is the risk set, as before.
    unread = _recorded(_prediction_context(), {"death_time": DATE_PROXY})
    assert (
        caller_bound_population_conflict(
            without_death, context=unread, required_selection_mode="all_input_rows"
        )
        is None
    )
