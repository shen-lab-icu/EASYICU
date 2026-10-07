"""A windowed predicate on a whole-stay event is read by the event's own time.

A study universe carries an outcome as its whole-stay status (``death`` is 1
whenever the stay died) beside the event's time (``death_time``, hours from ICU
admission).  The cohort builder read a predicate's window by that time only for
a truthy ``==`` occurrence check, so:

- "survived the first 24 h", written as ``death == 0`` (or ``!= 1``) over hours
  0-24, read the whole stay and kept only the stays that never died: every
  later death left the cohort, and with it the mortality outcome;
- the same exclusion in other words (``death != 0``, ``in [1]``) removed every
  death, not only the early ones;
- a source that records no time of death (every time empty) made "death within
  24 h" exclude nobody;
- a window from hospital admission was read against hours from ICU admission.

A predicate naming one event level is now read by the event time whichever way
it asks; a window no recorded time can place is refused.  An event the universe
records with no time column at all is still read for the whole stay.  Synthetic
tables.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.cohort.schema import (
    CohortDataError,
    _build_cohort_with_flow,
    materialize_locked_analysis_cohort,
)
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from tests.support.native_outcome_export import native_outcome, typed_native_export


def _universe(death_time=(2.0, 30.0, None, None, None, None)) -> pd.DataFrame:
    """Deaths at 2 h (stay 1), 30 h (stay 2) and at no recorded time (stay 3)."""

    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "death": [1, 1, 1, 0, 0, 0],
            "death_time": list(death_time),
        }
    )


def _predicate(
    concept: str, op: str, value, end: float = 24.0, *, anchor: str = "icu_admission"
) -> ConceptPredicate:
    return ConceptPredicate(
        concept_id=concept,
        time_window=TimeWindow(
            anchor=anchor, start_offset_hours=0.0, end_offset_hours=end
        ),
        aggregation="max",
        op=op,
        value=value,
    )


def _kept(universe, *, inclusion=(), exclusion=(), **kwargs) -> list[int]:
    cohort, _ = _build_cohort_with_flow(
        CohortDefinition(
            name="primary", inclusion=tuple(inclusion), exclusion=tuple(exclusion)
        ),
        universe,
        **kwargs,
    )
    return sorted(int(stay) for stay in cohort["stay_id"])


@pytest.mark.parametrize(
    "op, value",
    [("==", 0), ("==", False), ("!=", 1), ("not_in", [1])],
)
def test_survival_through_the_window_keeps_the_later_deaths(op, value) -> None:
    kept = _kept(_universe(), inclusion=[_predicate("death", op, value)])

    # Before: only the stays that never died (4, 5, 6) were kept.
    assert kept == [2, 3, 4, 5, 6]


@pytest.mark.parametrize(
    "op, value",
    [("==", 1), ("==", True), ("!=", 0), ("in", [1])],
)
def test_an_early_death_exclusion_reads_the_same_in_any_words(op, value) -> None:
    kept = _kept(_universe(), exclusion=[_predicate("death", op, value)])

    # Before: written as ``!= 0`` or ``in [1]`` it removed every death.
    assert kept == [2, 3, 4, 5, 6]


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_a_death_with_no_recorded_time_lies_outside_every_window(dtype) -> None:
    # Stay 3 died at no recorded time: neither reading places it in the window,
    # so excluding the deaths within it and keeping the survivors of it agree,
    # whether the table holds the missing time as NaN or as a nullable NA.
    universe = _universe()
    universe["death_time"] = universe["death_time"].astype(dtype)
    occurrence = _kept(universe, exclusion=[_predicate("death", "==", 1)])
    absence = _kept(universe, inclusion=[_predicate("death", "==", 0)])

    assert 3 in occurrence and 3 in absence
    assert occurrence == absence


@pytest.mark.parametrize("kind", ["inclusion", "exclusion"])
def test_a_window_no_recorded_time_can_place_is_refused(kind) -> None:
    untimed = _universe(death_time=(None,) * 6)
    predicate = _predicate("death", "==", 0 if kind == "inclusion" else 1)

    # Before: every death fell outside the window, so the exclusion removed
    # nobody (and the inclusion kept every death).
    with pytest.raises(CohortDataError, match="has a recorded time"):
        _kept(untimed, **{kind: [predicate]})


def test_a_universe_without_events_needs_no_time() -> None:
    universe = _universe(death_time=(None,) * 6).assign(death=0)

    assert _kept(universe, exclusion=[_predicate("death", "==", 1)]) == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_a_window_from_another_anchor_is_refused() -> None:
    with pytest.raises(CohortDataError, match="hours from ICU admission"):
        _kept(
            _universe(),
            exclusion=[_predicate("death", "==", 1, anchor="hospital_admit")],
        )


@pytest.mark.parametrize(
    "predicate",
    [
        _predicate("death", ">=", 1),
        _predicate("death", "missing", None),
        _predicate("death", "==", 2),
        _predicate("death", "in", [0, 1]),
    ],
    ids=["magnitude", "missingness", "no level", "both levels"],
)
def test_a_predicate_naming_no_single_level_reads_no_window(predicate) -> None:
    _, flow = _build_cohort_with_flow(
        CohortDefinition(name="primary", exclusion=(predicate,)), _universe()
    )

    assert flow[-1]["event_time_column"] is None
    assert flow[-1]["event_time_reading"] is None


def test_the_ledger_publishes_how_the_window_was_read() -> None:
    _, flow = _build_cohort_with_flow(
        CohortDefinition(
            name="primary",
            inclusion=(_predicate("death", "!=", 1),),
            exclusion=(_predicate("death", "in", [1], 48.0),),
        ),
        _universe(),
    )

    assert [row["event_time_reading"] for row in flow] == [
        None,
        "absence",
        "occurrence",
    ]


def test_the_host_reads_a_sealed_universe_the_same_way(tmp_path: Path) -> None:
    root = typed_native_export(
        tmp_path / "export",
        outcome=native_outcome(
            death=[True, False, True, False],
            death_time=[87.0, None, 12.0, None],
        ),
        outcome_concepts=["death"],
    )
    # The run directory holds the sealed universe the analysis cohort descends from.
    paths = materialize_to_parquet(
        tmp_path / "run",
        stem="universe",
        feature_concepts=[],
        database="miiv",
        data_path=str(root),
        cohort_window=(0.0, 24.0),
        outcome_concepts=["death"],
        static_concepts=["age"],
    )
    definition = CohortDefinition(
        name="primary", inclusion=(_predicate("death", "==", 0),)
    )

    survived = materialize_locked_analysis_cohort(
        run_dir=tmp_path / "run",
        plan=SimpleNamespace(cohort=definition, steps=[]),
        universe_path=Path(paths["parquet"]),
    )

    # The death at 87 h survived the first 24 h; the one at 12 h did not.
    assert survived["status"] == "applied"
    assert pd.read_parquet(survived["path"])["stay_id"].tolist() == [1, 2, 4]


def test_the_coder_is_told_how_an_absence_row_was_read() -> None:
    from easyicu.research_agent.resources import coder as coder_resources

    source = Path(coder_resources.__file__).read_text(encoding="utf-8")
    start = source.index("deterministically resolved the Planner-owned predicates")
    guidance = source[start : start + 2500]

    assert "`event_time_reading`" in guidance
    assert "for `absence`" in guidance
    assert "lies outside that window" in guidance
