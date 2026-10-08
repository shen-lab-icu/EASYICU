"""A cohort predicate reads only the window its column was summarized over.

The cohort builder filters a column as it was summarized; the window a
predicate states is locked for audit only.  A plan that kept the stays with a
marker above 4 within 48 h, over a column summarized over the first 24 h,
selected by the first 24 h, and nothing said so.  Only a column's own window
label, for a descriptor of the predicate's own concept, was compared with the
predicate's window.

A column's window is now read once (``context_column_windows``): its own
``analysis_window`` label, else, for a summary of observations over a window
(``stay_events.column_kind``, which the time-zero rule reads too), the host's
materialization window.  A predicate states that window; an event the
builder reads by its own ``<concept>_time`` may state any window inside it.
The check runs:

- at planning, for the primary cohort and every robustness override, with a
  correction that restates the window only when the question states none;
- wherever a cohort is built: the locked analysis cohort, every robustness
  override and the materializer's own filter.

An event time the input types in other than hours after ICU admission is
refused wherever the builder would compare it with a window in hours.  A
cohort step's prose keeps the window it states, and otherwise reads its
column's; a window the translator garbles fails the translation.

Synthetic tables and generic concepts, except where a dictionary category is
the subject.
"""

from __future__ import annotations

import inspect
import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.agents.family_spec_planner import FAMILY_SPEC_GUIDE
from easyicu.research_agent.agents.progressive_prompt_contracts import (
    foundation_shape_contract,
)
from easyicu.research_agent.cohort.materializer import materialize_cohort
from easyicu.research_agent.cohort.repair import extract_cohort_definition_from_prose
from easyicu.research_agent.cohort.schema import (
    COHORT_COLUMN_WINDOW_MISMATCH,
    COHORT_EVENT_TIME_NOT_HOURS_FROM_ICU_ADMISSION,
    CohortColumnWindowMismatchError,
    CohortDataError,
    CohortEventTimeNotHoursError,
    _descriptor_window_matches_predicate,
    build_cohort,
    materialize_locked_analysis_cohort,
    require_column_windows_readable,
    validate_plan_typed_bindings_against_context,
)
from easyicu.research_agent.execution.phase import (
    _extract_cohort_definition_with_provider_budget,
)
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    CohortSchemaError,
    ConceptPredicate,
    TimeWindow,
    cohort_concept_id_scope,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.planning.scientific_review import (
    unapplied_population_findings,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_diagnostics import (
    infer_validation_stage,
)
from easyicu.research_agent.research_context.materialization_window import (
    ColumnWindow,
    column_window_from_label,
    context_column_windows,
    host_column_window,
)
from easyicu.research_agent.research_context.stay_events import (
    column_kind,
    event_time_typed_otherwise_than_hours,
    event_times_typed_otherwise_than_hours,
)
from easyicu.research_agent.robustness.estimators import _data_with_predicate_aliases
from tests.support.native_outcome_export import native_outcome, untyped_native_export

_MARKER = "marker_max"
_STATUS = "procedure_done"


@pytest.fixture(autouse=True)
def _generic_concepts():
    with cohort_concept_id_scope(
        ["marker", "marker_alias", _MARKER, "score_first", _STATUS, "age_years"]
    ):
        yield


def _predicate(
    concept: str,
    op: str,
    value,
    *,
    start: float = 0.0,
    end: float = 24.0,
    anchor: str = "icu_admission",
    aggregation: str = "max",
) -> ConceptPredicate:
    return ConceptPredicate(
        concept_id=concept,
        time_window=TimeWindow(
            anchor=anchor, start_offset_hours=start, end_offset_hours=end
        ),
        aggregation=aggregation,
        op=op,
        value=value,
    )


def _cohort(*, inclusion=(), exclusion=(), name: str = "primary") -> CohortDefinition:
    return CohortDefinition(
        name=name, inclusion=tuple(inclusion), exclusion=tuple(exclusion)
    )


def _variable(name: str, role: str = "lab", **fields) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        role=role,
        analysis_window=fields.get("analysis_window"),
        observation_semantics=fields.get("observation_semantics"),
        temporal_resolution=fields.get("temporal_resolution"),
        source_concept=fields.get("source_concept"),
        unit_normalization=fields.get("unit_normalization"),
    )


def _event_time(
    status: str, *, unit: str = "h", origin: str = "icu_admission"
) -> SimpleNamespace:
    return _variable(
        f"{status}_time",
        role="time",
        observation_semantics=SimpleNamespace(
            kind="conditional_event_time",
            event_status_column=status,
            time_origin=origin,
            time_unit=unit,
        ),
    )


def _host_window(hours: float = 24.0, **fields) -> dict:
    return {
        "role": "outer_observation_window",
        "anchor": "icu_admission",
        "hours": hours,
        **fields,
    }


def _context(
    *variables,
    window: dict | None = None,
    constraints: str | None = None,
    sealed: bool = True,
    outcomes=(),
) -> SimpleNamespace:
    """A context whose host summarized its columns over ``window`` (24 h by default)."""

    names = ["stay_id", *(variable.name for variable in variables)]
    if constraints is None:
        constraints = json.dumps(
            {"materialization_window": window if window is not None else _host_window()}
        )
    return SimpleNamespace(
        research_question="Is a raised marker associated with the outcome?",
        cohort=SimpleNamespace(
            outcome_columns=list(outcomes), id_columns=["stay_id"], time_columns=[]
        ),
        target_outcome=None,
        variables=list(variables),
        materialized_inputs=(
            SimpleNamespace(
                cohort=SimpleNamespace(
                    cohort_columns=names,
                    column_bindings={name: name for name in names[1:]},
                )
            )
            if sealed
            else None
        ),
        user_preferences=SimpleNamespace(data_constraints=constraints),
    )


def _marker_context(**kwargs) -> SimpleNamespace:
    return _context(_variable(_MARKER, source_concept="marker"), **kwargs)


def _plan(cohort=None, *, specs=()) -> SimpleNamespace:
    return SimpleNamespace(cohort=cohort, steps=[], robustness_specs=list(specs))


def _universe(**columns) -> pd.DataFrame:
    """The marker each stay reached in the first 24 h."""

    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            _MARKER: [5.0, 3.0, 4.5],
            "exposure": [1.0, 0.0, 1.0],
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


@pytest.mark.parametrize(
    "variable, column, concept, kind",
    [
        (None, "los_icu", "los_icu", "icu_stay_length"),
        (None, "outcome_flag", "outcome_flag", "stay_outcome"),
        (None, "age", "age", "admission"),
        (
            _variable("sex_code", role="demographic"),
            "sex_code",
            "sex_code",
            "admission",
        ),
        (None, "saps3", "saps3", "stay_level"),
        (_event_time(_STATUS), f"{_STATUS}_time", f"{_STATUS}_time", "event_time"),
        (
            _variable(
                "marker_time", temporal_resolution="relative to icu_admission in h"
            ),
            "marker_time",
            "marker_time",
            "event_time",
        ),
        (
            _variable("marker_last_time", unit_normalization="window_last_time"),
            "marker_last_time",
            "marker",
            "window_summary",
        ),
        (_variable(_MARKER), _MARKER, "marker", "window_summary"),
    ],
    ids=[
        "ICU length of stay",
        "outcome",
        "dictionary demographic",
        "demographic role",
        "first-day severity score",
        "typed event time",
        "event time by its resolution",
        "last observation time",
        "summary",
    ],
)
def test_what_a_column_holds_for_each_stay(variable, column, concept, kind) -> None:
    assert (
        column_kind(
            variable,
            column=column,
            concept=concept,
            outcomes=frozenset({"outcome_flag"}),
        )
        == kind
    )


def test_the_window_each_column_was_summarized_over() -> None:
    context = _context(
        _variable(_MARKER, source_concept="marker"),
        _variable("score_first", analysis_window="icu_admission[0,6]h"),
        _variable("free_label", analysis_window="entire_stay"),
        _variable("age", role="demographic"),
        _variable("los_icu", role="covariate"),
        _variable("saps3", role="covariate", source_concept="saps3"),
        _variable("outcome_flag", role="outcome"),
        _variable(_STATUS, role="intervention"),
        _event_time(_STATUS),
    )

    windows = context_column_windows(context)

    assert sorted(windows) == ["free_label", _MARKER, "score_first"]
    assert windows[_MARKER] == host_column_window(0.0, 24.0)
    assert windows[_MARKER].description() == (
        "the host's materialization window icu_admission[0,24]h"
    )
    assert windows["score_first"] == ColumnWindow(
        label="icu_admission[0,6]h",
        anchor="icu_admission",
        start_hours=0.0,
        end_hours=6.0,
    )
    assert windows["score_first"].description() == "its own window icu_admission[0,6]h"
    assert windows["free_label"].anchor is None
    assert "'entire_stay', which names no window" in windows["free_label"].description()


@pytest.mark.parametrize(
    "constraints",
    [
        json.dumps({"materialization_window": _host_window(48)}),
        json.dumps(
            {
                "materialization_window": {
                    "role": "outer_observation_window",
                    "anchor": "ICU admission",
                    "observation_hours": 48,
                }
            }
        ),
        json.dumps({"materialization_window": _host_window(48.0, observation_hours=6)}),
    ],
    ids=["hours", "observation hours", "hours over observation hours"],
)
def test_the_host_records_its_window_in_any_of_its_spellings(constraints) -> None:
    context = _marker_context(constraints=constraints)

    assert context_column_windows(context) == {_MARKER: host_column_window(0.0, 48.0)}


@pytest.mark.parametrize(
    "constraints",
    [
        json.dumps({}),
        json.dumps(
            {"materialization_window": _host_window(48, role="analysis_window")}
        ),
        json.dumps(
            {
                "materialization_window": {
                    **_host_window(48),
                    "anchor": "hospital_admission",
                }
            }
        ),
        "not json",
    ],
    ids=["no record", "another role", "another anchor", "unreadable"],
)
def test_a_column_without_a_recorded_window_is_not_judged(constraints) -> None:
    context = _marker_context(constraints=constraints)
    plan = _plan(_cohort(inclusion=[_predicate("marker", ">", 4, end=48.0)]))

    assert context_column_windows(context) == {}
    validate_plan_typed_bindings_against_context(plan=plan, context=context)


@pytest.mark.parametrize(
    "label",
    [
        "icu_admission[0,24]h",
        "icu_admission[0,24.0]h",
        "icu_admit[0,24]h",
        "ICU admission [0, 24] h",
        "icu_admit_0_24h",
        "icu admission 0-24 h",
        "admission[0,24]h",
    ],
)
def test_a_label_names_the_window_a_predicate_states(label) -> None:
    window = column_window_from_label(label)

    # Read once for both comparisons: the anchor's spellings are one anchor
    # and the bounds are numbers, so "24.0" and "icu_admit" no longer fail a
    # descriptor the plan bound by name.
    for anchor in ("icu_admit", "icu_admission"):
        assert window.is_window(anchor, 0.0, 24.0)
        assert _descriptor_window_matches_predicate(
            label,
            TimeWindow(anchor=anchor, start_offset_hours=0.0, end_offset_hours=24.0),
        )
    assert not window.is_window("icu_admission", 0.0, 48.0)
    assert not window.is_window("hospital_admission", 0.0, 24.0)
    assert column_window_from_label("hospital_admit[0,24]h").is_window(
        "hospital_admission", 0, 24
    )


@pytest.mark.parametrize(
    "label", ["entire_stay", "first_24h", "0_24h", "icu_admission[0,24]hours"]
)
def test_a_label_that_names_no_window_matches_none(label) -> None:
    window = column_window_from_label(label)

    assert window.anchor is None
    assert not window.is_window("icu_admission", 0.0, 24.0)
    assert not window.contains("icu_admission", 0.0, 1.0)
    assert not _descriptor_window_matches_predicate(
        label,
        TimeWindow(
            anchor="icu_admission", start_offset_hours=0.0, end_offset_hours=24.0
        ),
    )
    assert column_window_from_label("  ") is None


def _refused(definition, columns, windows, **kwargs) -> list[str]:
    """The labels of the predicates the builder would read over another window."""

    try:
        require_column_windows_readable(
            definition, columns=columns, column_windows=windows, **kwargs
        )
    except CohortColumnWindowMismatchError as exc:
        return [item.label for item in exc.windows]
    return []


@pytest.mark.parametrize(
    "start, end, anchor, refused",
    [
        (0.0, 24.0, "icu_admission", False),
        (0.0, 24.0, "icu_admit", False),
        (0.0, 48.0, "icu_admission", True),
        (0.0, 12.0, "icu_admission", True),
        (6.0, 24.0, "icu_admission", True),
        (0.0, math.inf, "icu_admission", True),
        (0.0, 24.0, "hospital_admission", True),
    ],
    ids=[
        "its window",
        "its window (alias)",
        "wider",
        "narrower",
        "later start",
        "whole stay",
        "another anchor",
    ],
)
def test_a_value_reads_exactly_its_columns_window(start, end, anchor, refused) -> None:
    definition = _cohort(
        inclusion=[_predicate("marker", ">", 4, start=start, end=end, anchor=anchor)]
    )

    assert _refused(
        definition, [_MARKER], {_MARKER: host_column_window(0.0, 24.0)}
    ) == (["cohort.inclusion[0]"] if refused else [])


@pytest.mark.parametrize(
    "end, with_time, time_label, refused",
    [
        (12.0, True, None, False),
        (24.0, True, None, False),
        (48.0, True, None, True),
        (12.0, False, None, True),
        (24.0, False, None, False),
        (48.0, False, None, True),
        (12.0, True, "icu_admission[0,12]h", False),
        (24.0, True, "icu_admission[0,12]h", True),
    ],
    ids=[
        "inside, by its time",
        "its window, by its time",
        "wider, by its time",
        "inside, no time",
        "its window, no time",
        "wider, no time",
        "inside the time's own window",
        "beyond the time's own window",
    ],
)
def test_an_event_read_by_its_time_reads_any_window_inside_its_record(
    end, with_time, time_label, refused
) -> None:
    status = column_window_from_label("icu_admission[0,24]h")
    windows = {_STATUS: status}
    if time_label is not None:
        windows[f"{_STATUS}_time"] = column_window_from_label(time_label)
    columns = [_STATUS, *([f"{_STATUS}_time"] if with_time else [])]
    definition = _cohort(exclusion=[_predicate(_STATUS, "==", 1, end=end)])

    assert _refused(definition, columns, windows) == (
        ["cohort.exclusion[0]"] if refused else []
    )
    # Its missingness is no event: it reads the status as it was recorded.
    missing = _cohort(exclusion=[_predicate(_STATUS, "missing", None, end=12.0)])
    assert _refused(missing, columns, windows) == ["cohort.exclusion[0]"]


def test_the_refusal_says_what_each_predicate_reads() -> None:
    definition = _cohort(
        inclusion=[_predicate("marker", ">", 4, end=48.0)],
        exclusion=[
            _predicate("age_years", "<", 18),
            _predicate(_STATUS, "==", 1, end=48.0),
        ],
    )

    with pytest.raises(CohortColumnWindowMismatchError) as caught:
        require_column_windows_readable(
            definition,
            columns=[_MARKER, "age_years", _STATUS, f"{_STATUS}_time"],
            column_windows={
                _MARKER: host_column_window(0.0, 24.0),
                _STATUS: column_window_from_label("icu_admission[0,24]h"),
            },
            label="cohort_override",
        )

    error = caught.value
    assert str(error) == (
        f"{COHORT_COLUMN_WINDOW_MISMATCH}: cohort_override.inclusion[0] reads "
        f"'{_MARKER}' within icu_admission[0, 48) h, but this input summarizes "
        f"'{_MARKER}' over the host's materialization window icu_admission[0,24]h; "
        f"cohort_override.exclusion[1] reads the event of '{_STATUS}' within "
        f"icu_admission[0, 48) h by its time, but this input records '{_STATUS}' "
        "only over its own window icu_admission[0,24]h"
    )
    assert error.code == COHORT_COLUMN_WINDOW_MISMATCH
    assert isinstance(error, CohortDataError)
    assert str(pickle.loads(pickle.dumps(error))) == str(error)


def test_before_the_builder_selected_by_the_columns_window() -> None:
    # A stay whose marker rose above 4 only after 24 h is not in this column:
    # the builder keeps stays 1 and 3 whatever window the predicate states.
    within_two_days = _cohort(inclusion=[_predicate("marker", ">", 4, end=48.0)])

    selected = build_cohort(within_two_days, _universe())

    assert sorted(selected["stay_id"]) == [1, 3]


def test_the_planner_is_refused_a_window_its_column_was_not_summarized_over() -> None:
    plan = _plan(_cohort(inclusion=[_predicate("marker", ">", 4, end=48.0)]))

    with pytest.raises(CohortSchemaError) as caught:
        validate_plan_typed_bindings_against_context(
            plan=plan, context=_marker_context()
        )

    message = str(caught.value)
    assert (
        f"Invalid references: {COHORT_COLUMN_WINDOW_MISMATCH}: cohort.inclusion[0] "
        f"reads '{_MARKER}' within icu_admission[0, 48) h, but this input summarizes "
        f"'{_MARKER}' over the host's materialization window icu_admission[0,24]h"
    ) in message
    # The correction restates the window only when the question states none.
    assert (
        "When the question states no window for the criterion and the plan "
        "chose this one, restate the predicate's time_window as the column's "
        "window.  When the question states the window, do not restate it."
    ) in message
    assert "population_criteria with no concepts" in message
    assert "cohort.unapplied_population_criteria" in message
    assert "the user's to extract" in message
    assert "Use an executable dictionary concept" not in message
    assert infer_validation_stage(caught.value) == "typed_context_binding"


def test_the_general_correction_stays_for_an_issue_without_its_own() -> None:
    unbound = RobustnessSpec(
        spec_id="unbound",
        axis="cohort",
        description="Keep the stays with a raised marker.",
        cohort_override=_cohort(
            name="unbound", inclusion=[_predicate("marker_alias", ">", 4)]
        ),
    )
    plan = _plan(
        _cohort(inclusion=[_predicate("marker", ">", 4, end=48.0)]), specs=[unbound]
    )

    with pytest.raises(CohortSchemaError) as caught:
        validate_plan_typed_bindings_against_context(
            plan=plan, context=_marker_context()
        )

    message = str(caught.value)
    assert f"For each {COHORT_COLUMN_WINDOW_MISMATCH}:" in message
    assert "has no exact or uniquely bound sealed column" in message
    assert "Use an executable dictionary concept" in message


def test_the_planner_keeps_the_windows_its_columns_were_summarized_over() -> None:
    context = _context(
        _variable(_MARKER, source_concept="marker"),
        _variable("score_first", analysis_window="icu_admission[0,6]h"),
        _variable("age_years", role="demographic"),
    )
    plan = _plan(
        _cohort(
            inclusion=[
                _predicate("marker", ">", 4, anchor="icu_admit"),
                _predicate("score_first", ">=", 2, end=6.0),
                _predicate("age_years", ">=", 18, end=72.0),
            ]
        )
    )

    validate_plan_typed_bindings_against_context(plan=plan, context=context)
    # Its own label takes precedence over the host's window.
    plan.cohort = _cohort(inclusion=[_predicate("score_first", ">=", 2)])
    with pytest.raises(CohortSchemaError, match="over its own window icu_admission"):
        validate_plan_typed_bindings_against_context(plan=plan, context=context)


def test_the_planner_is_refused_an_override_its_columns_cannot_read() -> None:
    override = RobustnessSpec(
        spec_id="two_days",
        axis="cohort",
        description="Keep the stays with a raised marker within 48 h.",
        cohort_override=_cohort(
            name="two_days", inclusion=[_predicate("marker", ">", 4, end=48.0)]
        ),
    )

    with pytest.raises(CohortSchemaError) as caught:
        validate_plan_typed_bindings_against_context(
            plan=_plan(specs=[override]), context=_marker_context()
        )

    assert (
        f"{COHORT_COLUMN_WINDOW_MISMATCH}: robustness_specs[two_days]"
        f".cohort_override.inclusion[0] reads '{_MARKER}'"
    ) in str(caught.value)


def test_a_context_without_a_sealed_roster_is_judged_when_its_cohort_is_built() -> None:
    plan = _plan(_cohort(inclusion=[_predicate("marker", ">", 4, end=48.0)]))

    validate_plan_typed_bindings_against_context(
        plan=plan, context=_marker_context(sealed=False)
    )


def test_the_locked_analysis_cohort_reads_no_other_window(tmp_path: Path) -> None:
    universe_path = tmp_path / "universe.parquet"
    _universe().to_parquet(universe_path, index=False)
    context = _marker_context(sealed=False)

    refused = materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_plan(_cohort(inclusion=[_predicate("marker", ">", 4, end=48.0)])),
        universe_path=universe_path,
        context=context,
    )
    applied = materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_plan(_cohort(inclusion=[_predicate("marker", ">", 4)])),
        universe_path=universe_path,
        context=context,
    )

    assert refused["status"] == "error"
    assert (
        f"CohortColumnWindowMismatchError: {COHORT_COLUMN_WINDOW_MISMATCH}: "
        "cohort.inclusion[0]"
    ) in refused["error"]
    assert (applied["status"], applied["n_cohort"]) == ("applied", 2)


@pytest.mark.parametrize("concept", [_MARKER, "marker_alias"], ids=["named", "aliased"])
def test_a_robustness_override_reads_no_other_window(concept) -> None:
    context = _marker_context(sealed=False)

    with pytest.raises(
        CohortColumnWindowMismatchError,
        match=r"cohort_override\.inclusion\[0\] reads .* over the host's",
    ):
        _data_with_predicate_aliases(
            data=_universe(),
            cohort_definition=_cohort(
                name="two_days", inclusion=[_predicate(concept, ">", 4, end=48.0)]
            ),
            exposure="exposure",
            context=context,
        )
    readable = _data_with_predicate_aliases(
        data=_universe(),
        cohort_definition=_cohort(
            name="first_day", inclusion=[_predicate(concept, ">", 4)]
        ),
        exposure="exposure",
        context=context,
    )
    assert sorted(
        build_cohort(
            _cohort(name="first_day", inclusion=[_predicate(concept, ">", 4)]), readable
        )["stay_id"]
    ) == [1, 3]


def _antibiotic_export(tmp_path: Path) -> Path:
    """An older export: antibiotics at 9 h and 30 h (stay 1), 4 h (2) and 30 h (3)."""

    medications = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 3],
            "charttime": [9.0, 30.0, 4.0, 30.0],
            "abx": [True, True, False, True],
        }
    )
    return untyped_native_export(
        tmp_path / "export",
        outcome=native_outcome(death=[False, False, False, False]),
        outcome_concepts=["death"],
        longitudinal=medications,
        longitudinal_concepts=["abx"],
    )


def test_the_materializer_reads_a_concept_over_the_window_it_derived_it_over(
    tmp_path: Path,
) -> None:
    export = _antibiotic_export(tmp_path)
    options = dict(
        feature_concepts=[],
        database="miiv",
        outcome_concepts=["death"],
        static_concepts=["age"],
        data_path=str(export),
    )
    first_day = _predicate("abx", "==", 1)
    two_days = _predicate("abx", "==", 1, end=48.0)

    kept_first_day, _ = materialize_cohort(
        cohort_definition=_cohort(exclusion=[first_day]), **options
    )
    kept_two_days, _ = materialize_cohort(
        cohort_definition=_cohort(exclusion=[two_days]), **options
    )
    # Both predicates read one column, derived over the first one's window:
    # the 30 h antibiotic of stay 3 was read as absent within 48 h.
    with pytest.raises(CohortColumnWindowMismatchError) as caught:
        materialize_cohort(
            cohort_definition=_cohort(exclusion=[first_day, two_days]), **options
        )

    assert sorted(kept_first_day["stay_id"]) == [2, 3, 4]
    assert sorted(kept_two_days["stay_id"]) == [2, 4]
    assert str(caught.value) == (
        f"{COHORT_COLUMN_WINDOW_MISMATCH}: cohort.exclusion[1] reads 'abx' within "
        "icu_admission[0, 48) h, but this input summarizes 'abx' over its own "
        "window icu_admission[0,24]h"
    )


@pytest.mark.parametrize(
    "time_variable, typed",
    [
        (_event_time(_STATUS), None),
        (_event_time(_STATUS, unit="d"), ("icu_admission", "d")),
        (_event_time(_STATUS, unit="min"), ("icu_admission", "min")),
        (
            _event_time(_STATUS, origin="hospital_admission"),
            ("hospital_admission", "h"),
        ),
        (_event_time(_STATUS, origin=None), (None, "h")),
        (_event_time(_STATUS, unit=None), ("icu_admission", None)),
        (_variable(f"{_STATUS}_time", role="time"), None),
        (
            _variable(
                f"{_STATUS}_time", temporal_resolution="relative to icu_admission in d"
            ),
            ("icu_admission", "d"),
        ),
        (
            _variable(
                f"{_STATUS}_time", temporal_resolution="relative to ICU admission in h"
            ),
            None,
        ),
    ],
    ids=[
        "hours after ICU admission",
        "days",
        "minutes",
        "from hospital admission",
        "a unit and no origin",
        "an origin and no unit",
        "untyped",
        "days by its resolution",
        "hours by its resolution",
    ],
)
def test_an_event_time_typed_otherwise_than_hours(time_variable, typed) -> None:
    assert event_time_typed_otherwise_than_hours(time_variable) is (typed is not None)
    assert event_times_typed_otherwise_than_hours(
        _context(_variable(_STATUS, role="intervention"), time_variable)
    ) == ({f"{_STATUS}_time": typed} if typed else {})


def _days_context(**kwargs) -> SimpleNamespace:
    return _context(
        _variable(_STATUS, role="intervention"),
        _event_time(_STATUS, unit="d"),
        **kwargs,
    )


def test_an_event_time_in_days_is_refused_wherever_it_would_be_read(
    tmp_path: Path,
) -> None:
    within_a_day = _cohort(exclusion=[_predicate(_STATUS, "==", 1)])
    universe = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            _STATUS: [1, 1, 0],
            f"{_STATUS}_time": [0.5, 3.0, None],
            "exposure": [1.0, 0.0, 1.0],
        }
    )
    universe_path = tmp_path / "universe.parquet"
    universe.to_parquet(universe_path, index=False)
    expected = (
        f"{COHORT_EVENT_TIME_NOT_HOURS_FROM_ICU_ADMISSION}: cohort.exclusion[0] "
        f"reads the event of '{_STATUS}' within its window in hours after ICU "
        f"admission by '{_STATUS}_time', which this input times in 'd' from "
        "'icu_admission'"
    )

    with pytest.raises(CohortSchemaError) as planned:
        validate_plan_typed_bindings_against_context(
            plan=_plan(within_a_day), context=_days_context()
        )
    with pytest.raises(CohortEventTimeNotHoursError) as override:
        _data_with_predicate_aliases(
            data=universe,
            cohort_definition=within_a_day,
            exposure="exposure",
            context=_days_context(sealed=False),
        )
    locked = materialize_locked_analysis_cohort(
        run_dir=tmp_path,
        plan=_plan(within_a_day),
        universe_path=universe_path,
        context=_days_context(sealed=False),
    )

    assert expected in str(planned.value)
    # A Planner with strict structured output cannot write an unbounded end.
    assert 'end_offset_hours "inf"' not in str(planned.value)
    assert "An event time in hours after ICU admission is the user's to extract" in (
        str(planned.value)
    )
    assert str(override.value) == expected.replace("cohort.", "cohort_override.")
    assert override.value.code == COHORT_EVENT_TIME_NOT_HOURS_FROM_ICU_ADMISSION
    assert str(pickle.loads(pickle.dumps(override.value))) == str(override.value)
    assert locked["status"] == "error" and expected in locked["error"]
    # The whole stay is read by the status, not by its time.
    whole_stay = _cohort(exclusion=[_predicate(_STATUS, "==", 1, end=math.inf)])
    validate_plan_typed_bindings_against_context(
        plan=_plan(whole_stay), context=_days_context()
    )


@pytest.mark.parametrize(
    "time_variable, typed",
    [
        (_event_time(_STATUS, origin=None), "in 'h' from an origin it does not state"),
        (
            _event_time(_STATUS, unit=None),
            "from 'icu_admission' in a unit it does not state",
        ),
        (
            _event_time(_STATUS, origin="hospital_admission"),
            "in 'h' from 'hospital_admission'",
        ),
    ],
    ids=["a unit alone", "an origin alone", "another origin"],
)
def test_the_refusal_says_how_the_input_times_the_event(time_variable, typed) -> None:
    context = _context(_variable(_STATUS, role="intervention"), time_variable)
    plan = _plan(_cohort(exclusion=[_predicate(_STATUS, "==", 1)]))

    with pytest.raises(CohortSchemaError) as caught:
        validate_plan_typed_bindings_against_context(plan=plan, context=context)

    message = str(caught.value)
    assert f"by '{_STATUS}_time', which this input times {typed}" in message
    assert "or with only its origin or only its unit" in message


def test_an_untyped_event_time_is_read_as_hours_after_icu_admission() -> None:
    context = _context(
        _variable(_STATUS, role="intervention"),
        _variable(f"{_STATUS}_time", role="time"),
    )

    validate_plan_typed_bindings_against_context(
        plan=_plan(_cohort(exclusion=[_predicate(_STATUS, "==", 1, end=12.0)])),
        context=context,
    )


def _prose_context() -> SimpleNamespace:
    return _context(
        _variable("age_years", role="demographic"),
        _variable(_MARKER, source_concept="marker"),
        _variable("outcome_flag", role="outcome"),
        window=_host_window(48),
        sealed=False,
        outcomes=("outcome_flag",),
    )


_PROSE_REPLY = json.dumps(
    {
        "inclusion": [
            {"concept_id": "age_years", "op": ">=", "value": 18},
            {"concept_id": _MARKER, "op": ">", "value": 4},
        ],
        "exclusion": [
            {"concept_id": "outcome_flag", "op": "==", "value": 1},
            {
                "concept_id": _MARKER,
                "op": ">",
                "value": 9,
                "time_window": {
                    "anchor": "icu_admission",
                    "start_offset_hours": 0,
                    "end_offset_hours": 24,
                },
            },
        ],
    }
)
_PROSE = (
    "Adults with a raised marker; exclude stays with the outcome and stays "
    "with a marker above 9 within 24 h of ICU admission."
)
_PROSE_COLUMNS = ["stay_id", "age_years", _MARKER, "outcome_flag"]


def _windows(definition: CohortDefinition) -> list[tuple]:
    return [
        (
            predicate.concept_id,
            predicate.time_window.anchor,
            predicate.time_window.start_offset_hours,
            predicate.time_window.end_offset_hours,
        )
        for predicate in (*definition.inclusion, *definition.exclusion)
    ]


def test_a_cohort_in_prose_reads_the_window_it_states_else_its_columns() -> None:
    llm = ScriptedMockLLMClient([_PROSE_REPLY])
    context = _prose_context()

    definition = extract_cohort_definition_from_prose(
        cohort_prose=_PROSE, universe_columns=_PROSE_COLUMNS, llm=llm, context=context
    )

    assert _windows(definition) == [
        # A value fixed at admission: no window summarizes it.
        ("age_years", "icu_admit", 0.0, 24.0),
        # The host's window, which the column was summarized over.
        (_MARKER, "icu_admission", 0.0, 48.0),
        # An outcome the stay records whole.
        ("outcome_flag", "icu_admission", 0.0, math.inf),
        # The window the prose states, carried and checked like any plan's.
        (_MARKER, "icu_admission", 0.0, 24.0),
    ]
    assert (
        "When the prose states the time window a criterion reads"
        in llm.calls[0][0][-1].content
    )
    assert _refused(definition, _PROSE_COLUMNS, context_column_windows(context)) == [
        "cohort.exclusion[1]"
    ]


@pytest.mark.parametrize(
    "window",
    [
        {
            "anchor": "icu_admission",
            "start_offset_hours": "<hours>",
            "end_offset_hours": 24,
        },
        {"anchor": "icu_admission", "start_offset_hours": 24, "end_offset_hours": 24},
        {"anchor": "icu_admission", "start_offset_hours": 0},
        {"start_offset_hours": 0, "end_offset_hours": 24},
        "within 24 h of ICU admission",
    ],
    ids=[
        "a bound that is no number",
        "an end not after its start",
        "no end",
        "no anchor",
        "no object",
    ],
)
def test_a_window_the_translator_garbles_fails_the_translation(window) -> None:
    # Dropping that criterion alone would apply the others: a wider cohort
    # than the prose states, and nothing would say so.
    reply = json.dumps(
        {
            "inclusion": [
                {"concept_id": "age_years", "op": ">=", "value": 18},
                {"concept_id": _MARKER, "op": ">", "value": 4, "time_window": window},
            ],
            "exclusion": [],
        }
    )

    definition = extract_cohort_definition_from_prose(
        cohort_prose=_PROSE,
        universe_columns=_PROSE_COLUMNS,
        llm=ScriptedMockLLMClient([reply]),
        context=_prose_context(),
    )

    assert definition is None


def test_the_host_hands_the_translator_its_context(tmp_path: Path) -> None:
    definition, _snapshot = _extract_cohort_definition_with_provider_budget(
        run_dir=tmp_path,
        budget_owner_step_id="01_cohort_definition",
        configured_limit=1,
        cohort_prose=_PROSE,
        universe_columns=_PROSE_COLUMNS,
        llm=ScriptedMockLLMClient([_PROSE_REPLY]),
        name="primary",
        context=_prose_context(),
    )

    assert _windows(definition)[1] == (_MARKER, "icu_admission", 0.0, 48.0)


def test_the_planners_are_told_a_criterion_is_expressed_over_its_window() -> None:
    foundation = foundation_shape_contract(outline_sha256="a" * 64, host_cohort=None)

    assert (
        "Give a criterion no concepts only when no allowed concept expresses it "
        "over the window the criterion states,"
    ) in foundation
    assert (
        "Give a criterion no concepts only when no offered concept expresses it "
        "over the window the criterion states."
    ) in FAMILY_SPEC_GUIDE


def test_a_criterion_no_predicate_applies_is_reported_without_a_cause() -> None:
    plan = SimpleNamespace(
        cohort=SimpleNamespace(
            unapplied_population_criteria=("a raised marker within 48 h",)
        )
    )

    (finding,) = unapplied_population_findings(plan)

    assert finding.message == (
        "The plan states the population criterion 'a raised marker within 48 h', "
        "but no predicate applies it: the analysis keeps every input row that "
        "the other cohort criteria keep, a broader population than the one stated."
    )
    assert "(a concept that expresses it, recorded over the window it states)" in (
        finding.remediation
    )
    vocab = (
        Path(inspect.getfile(unapplied_population_findings)).resolve().parents[2]
        / "webserver"
        / "static"
        / "js"
        / "screens-agent-reader-vocab.js"
    ).read_text(encoding="utf-8")
    (entry,) = [
        line
        for line in vocab.splitlines()
        if "POPULATION_CRITERION_NOT_APPLIED:" in line
    ]
    assert "但没有谓词施加它" in entry
    assert "能表达它、并按它写明的窗口记录的概念" in entry
    assert "没有可用的队列概念能表达它" not in entry
