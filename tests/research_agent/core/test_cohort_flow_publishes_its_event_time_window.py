"""The attrition ledger must describe the mask its own owner applied.

MEASURED on the H1 canary run of 2026-07-30
(``run_20260730T143739_8f97b6``, image ``dev-d7d1687-20260730``). The Planner
locked the exclusion ``death == 1`` over ICU hours 0-24 -- a landmark exclusion
against immortal-time bias, and exactly what that benchmark item's guardrails
ask for. ``_refine_occurrence_mask_by_event_time`` applied the window through
the ``death_time`` sibling and excluded 2,060 of 94,458 stays; the ledger row it
wrote said only ``death == 1``, which over the same table is 9,466. The Coder,
told to "use every ``resolved_column`` and operation in order, and assert the
recorded before/excluded/remaining counts", wrote correct code and raised::

    ValueError: Observed predicate accounting [... (94458, 84992, 9466)] does
    not match host receipt [... (94458, 92398, 2060)]

Step 01 died, steps 02 and 03 blocked on its evidence, steps 04-07 never ran:
0 of 7. Across 114 recorded plans, 87 carry a predicate-filtered cohort and
exactly one declares any exclusion at all -- this one. The feature had been
exercised once and had failed every time it was exercised.

These tests hold the ledger to the mask: whatever the owner consults to build a
predicate's mask, the same call reports, and the receipt and the Coder prompt
carry it through.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.cohort.schema import (
    AppliedEventTimeWindow,
    _build_cohort_with_flow,
)
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)


def _universe() -> pd.DataFrame:
    """Six stays: three die, at hours 2, 30 and -1 from ICU admission.

    The negative event time is the shape that separated 2,088 from 2,060 in the
    real run, so it is kept here rather than idealised away.
    """

    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "age": [40, 55, 61, 72, 33, 80],
            "death": [1, 1, 1, 0, 0, 0],
            "death_time": [2.0, 30.0, -1.0, None, None, None],
        }
    )


def _early_death_exclusion() -> CohortDefinition:
    return CohortDefinition(
        name="landmark",
        inclusion=(
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow(
                    anchor="icu_admission",
                    start_offset_hours=0.0,
                    end_offset_hours=1.0,
                ),
                aggregation="first",
                op=">=",
                value=18,
            ),
        ),
        exclusion=(
            ConceptPredicate(
                concept_id="death",
                time_window=TimeWindow(
                    anchor="icu_admission",
                    start_offset_hours=0.0,
                    end_offset_hours=24.0,
                ),
                aggregation="any",
                op="==",
                value=1,
            ),
        ),
    )


def test_the_windowed_exclusion_really_does_drop_fewer_rows_than_its_op_says():
    """The premise: op and value alone give a different answer."""

    universe = _universe()
    cohort, flow = _build_cohort_with_flow(_early_death_exclusion(), universe)

    exclusion = flow[-1]
    assert exclusion["predicate_kind"] == "exclusion"
    # Only stay 1 dies inside [0h, 24h]. Reading `death == 1` without the
    # window would drop three, and a window that admitted the negative event
    # time would drop two -- both of those are the failure this test guards.
    assert exclusion["n_excluded"] == 1
    assert list(cohort["stay_id"]) == [2, 3, 4, 5, 6]


def test_the_ledger_names_the_window_and_the_column_it_consulted():
    _, flow = _build_cohort_with_flow(_early_death_exclusion(), _universe())

    exclusion = flow[-1]
    assert exclusion["event_time_column"] == "death_time"
    assert exclusion["event_time_start_hours"] == 0.0
    assert exclusion["event_time_end_hours"] == 24.0


def _replay_ledger(universe: pd.DataFrame, flow: list) -> None:
    """Reproduce every recorded count using only the published fields.

    This is the assertion the real generated code was making, and losing. It
    walks the ledger the way that code did -- cumulative masks, each row's
    ``n_before`` inherited from the last row's ``n_remaining`` -- with no access
    to the definition, the predicate objects, or the refinement rule inside the
    host. Anything the host consults but does not publish shows up here as a
    count that does not reconcile.
    """

    import operator

    ops = {
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }

    kept = pd.Series(True, index=universe.index)
    assert flow[0]["n_remaining"] == int(kept.sum())

    for row in flow[1:]:
        matches = ops[row["op"]](universe[row["resolved_column"]], row["value"])
        event_time_column = row["event_time_column"]
        if event_time_column is not None:
            event_time = universe[event_time_column]
            matches = matches & (
                (event_time >= row["event_time_start_hours"])
                & (event_time <= row["event_time_end_hours"])
            ).fillna(False)
        before = int(kept.sum())
        kept = kept & (matches if row["predicate_kind"] == "inclusion" else ~matches)
        assert before == row["n_before"]
        assert before - int(kept.sum()) == row["n_excluded"]
        assert int(kept.sum()) == row["n_remaining"]


def _windowed_universe() -> pd.DataFrame:
    """One table carrying an event-time sibling for both concepts under test."""

    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "age": [40, 55, 61, 72, 33, 17],
            "death": [1, 1, 1, 0, 0, 0],
            "death_time": [2.0, 30.0, -1.0, None, None, None],
            "lact_max": [1.0, 4.0, 9.0, 0.5, 3.0, 6.0],
            "lact_time": [3.0, 500.0, 500.0, 1.0, 2.0, 700.0],
        }
    )


def _predicate(concept_id, aggregation, op, value, window_end):
    return ConceptPredicate(
        concept_id=concept_id,
        time_window=TimeWindow(
            anchor="icu_admission",
            start_offset_hours=0.0,
            end_offset_hours=window_end,
        ),
        aggregation=aggregation,
        op=op,
        value=value,
    )


# One scenario per surviving branch of _refine_occurrence_mask_by_event_time, so
# a new unpublished refinement on any of them breaks the replay rather than
# hiding. There is no "predicate without a window" case because the schema
# refuses to build one -- see the invariant test below.
_LEDGER_SHAPES = {
    "windowed occurrence exclusion (the shape that killed the real run)": (
        (),
        (_predicate("death", "any", "==", 1, 24.0),),
    ),
    "occurrence exclusion over an unbounded window": (
        (),
        (_predicate("death", "any", "==", 1, math.inf),),
    ),
    "magnitude filter whose concept does have an event time": (
        (_predicate("lact", "max", ">=", 2.0, 24.0),),
        (),
    ),
    "windowed predicate on a concept with no event-time sibling": (
        (_predicate("age", "first", ">=", 18, 1.0),),
        (),
    ),
    "an inclusion and a windowed exclusion together": (
        (_predicate("age", "first", ">=", 18, 1.0),),
        (_predicate("death", "any", "==", 1, 24.0),),
    ),
}


@pytest.mark.parametrize("shape", sorted(_LEDGER_SHAPES))
def test_a_reader_of_the_ledger_alone_reproduces_every_recorded_count(shape):
    inclusion, exclusion = _LEDGER_SHAPES[shape]
    universe = _windowed_universe()
    _, flow = _build_cohort_with_flow(
        CohortDefinition(name=shape, inclusion=inclusion, exclusion=exclusion),
        universe,
    )

    _replay_ledger(universe, flow)


def test_an_unwindowed_predicate_says_so_instead_of_omitting_the_fields():
    """A ledger whose columns depend on the plan is not a schema."""

    definition = CohortDefinition(
        name="plain",
        inclusion=(
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow(
                    anchor="icu_admission",
                    start_offset_hours=0.0,
                    end_offset_hours=1.0,
                ),
                aggregation="first",
                op=">=",
                value=18,
            ),
        ),
        exclusion=(),
    )
    _, flow = _build_cohort_with_flow(definition, _universe())

    for row in flow:
        assert row["event_time_column"] is None
        assert row["event_time_start_hours"] is None
        assert row["event_time_end_hours"] is None


def test_a_magnitude_filter_is_not_windowed_even_when_it_could_be():
    """Only occurrence predicates are refined.

    A summary column such as ``lact_max`` was already aggregated within its own
    window, so re-windowing it by ``lact_time`` would filter twice. The
    universe here gives that predicate an event-time sibling on purpose: the
    operator is then the only thing keeping the row unrefined, which is what
    this test is for.
    """

    universe = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "lact_max": [1.0, 4.0, 9.0],
            "lact_time": [3.0, 500.0, 500.0],
        }
    )
    definition = CohortDefinition(
        name="magnitude",
        inclusion=(
            ConceptPredicate(
                concept_id="lact",
                time_window=TimeWindow(
                    anchor="icu_admission",
                    start_offset_hours=0.0,
                    end_offset_hours=24.0,
                ),
                aggregation="max",
                op=">=",
                value=2.0,
            ),
        ),
        exclusion=(),
    )
    _, flow = _build_cohort_with_flow(definition, universe)

    inclusion = flow[1]
    assert inclusion["resolved_column"] == "lact_max"
    assert inclusion["event_time_column"] is None
    # And the behaviour that goes with it: both high-lactate stays are kept,
    # including the one whose event time sits outside the declared window.
    assert inclusion["n_remaining"] == 2


def test_the_occurrence_predicate_is_refined_beside_an_unrefined_one():
    _, flow = _build_cohort_with_flow(_early_death_exclusion(), _universe())

    assert flow[1]["concept_id"] == "age"
    assert flow[1]["event_time_column"] is None
    assert flow[-1]["event_time_column"] == "death_time"


def test_the_universe_row_carries_the_same_keys_as_every_other_row():
    _, flow = _build_cohort_with_flow(_early_death_exclusion(), _universe())

    assert flow[0]["predicate_kind"] == "universe"
    assert set(flow[0]) == set(flow[-1])


def test_the_ledger_survives_the_csv_the_host_writes_it_to(tmp_path):
    """``<stem>_flow.csv`` is ``pd.DataFrame(flow).to_csv(...)``.

    A nested object would land there as a repr string, so the fields are flat
    and must come back as real scalars a reader can compare against.
    """

    _, flow = _build_cohort_with_flow(_early_death_exclusion(), _universe())
    flow_path = tmp_path / "cohort_analysis_flow.csv"
    pd.DataFrame(flow).to_csv(flow_path, index=False)
    restored = pd.read_csv(flow_path)

    exclusion = restored.iloc[-1]
    assert exclusion["event_time_column"] == "death_time"
    assert float(exclusion["event_time_start_hours"]) == 0.0
    assert float(exclusion["event_time_end_hours"]) == 24.0
    assert pd.isna(restored.iloc[1]["event_time_column"])
    assert "{" not in flow_path.read_text(encoding="utf-8")


def test_the_schema_guarantees_the_window_the_refiner_assumes():
    """Why the refiner no longer guards a missing window or a missing end.

    Both guards were unreachable: the predicate refuses to exist without a
    window, and the window's end is a required float. A guard that cannot fire
    is not protection, it is a place for a broken invariant to hide as a
    silently unrefined mask. If this ever starts passing, restore the guard --
    with this test inverted to prove it fires.
    """

    from easyicu.research_agent.planning.cohort_contract import CohortSchemaError

    with pytest.raises(CohortSchemaError, match="time_window is required"):
        ConceptPredicate(
            concept_id="death",
            time_window=None,  # type: ignore[arg-type]
            aggregation="any",
            op="==",
            value=1,
        )
    with pytest.raises(Exception):
        TimeWindow(
            anchor="icu_admission",
            start_offset_hours=0.0,
            end_offset_hours=None,  # type: ignore[arg-type]
        )


def test_an_unbounded_window_publishes_no_bound_rather_than_infinity():
    """The one guard that does fire, and the reason it has to.

    ``"inf"`` is an accepted window end, meaning 'ever'. Refining by it would
    change nothing, and publishing it would put a non-finite number in a ledger
    that is serialized to JSON and to CSV.
    """

    definition = CohortDefinition(
        name="unbounded",
        inclusion=(),
        exclusion=(_predicate("death", "any", "==", 1, math.inf),),
    )
    universe = _universe()
    cohort, flow = _build_cohort_with_flow(definition, universe)

    exclusion = flow[-1]
    assert exclusion["event_time_column"] is None
    assert exclusion["event_time_end_hours"] is None
    # Unrefined means every death is excluded, not just the early ones.
    assert exclusion["n_excluded"] == 3
    assert list(cohort["stay_id"]) == [4, 5, 6]


def test_the_refinement_record_is_immutable():
    """It crosses a boundary; a consumer must not be able to edit it."""

    window = AppliedEventTimeWindow(
        event_time_column="death_time",
        start_offset_hours=0.0,
        end_offset_hours=24.0,
    )
    with pytest.raises(Exception):
        window.end_offset_hours = 48.0  # type: ignore[misc]


def test_the_event_time_column_is_authorized_as_a_predicate_coordinate():
    """The Coder cannot read a column it holds no contract for."""

    from easyicu.research_agent.research_context.typed import (
        raw_contract_inputs_for_step,
    )

    receipt = {
        "ordered_predicate_flow": [
            {"predicate_kind": "universe", "resolved_column": None},
            {
                "predicate_kind": "exclusion",
                "resolved_column": "death",
                "event_time_column": "death_time",
            },
        ]
    }
    names = raw_contract_inputs_for_step(
        planner_declared_inputs=["age"],
        primary_cohort_execution_receipt=receipt,
    )

    assert "death" in names
    assert "death_time" in names


def test_an_unrefined_row_authorizes_nothing_extra():
    from easyicu.research_agent.research_context.typed import (
        raw_contract_inputs_for_step,
    )

    receipt = {
        "ordered_predicate_flow": [
            {"predicate_kind": "universe", "resolved_column": None},
            {
                "predicate_kind": "inclusion",
                "resolved_column": "age",
                "event_time_column": None,
            },
        ]
    }
    names = raw_contract_inputs_for_step(
        planner_declared_inputs=["age"],
        primary_cohort_execution_receipt=receipt,
    )

    assert names == ("age",)


def test_a_typed_artifact_name_is_still_refused_in_the_event_time_slot():
    """The new field is a raw coordinate, held to the same rule as the old one."""

    from easyicu.research_agent.intake.materialized_metadata import (
        MaterializedMetadataError,
    )
    from easyicu.research_agent.research_context.typed import (
        raw_contract_inputs_for_step,
    )

    receipt = {
        "ordered_predicate_flow": [
            {"predicate_kind": "universe", "resolved_column": None},
            {
                "predicate_kind": "exclusion",
                "resolved_column": "death",
                "event_time_column": "table:not_raw",
            },
        ]
    }
    with pytest.raises(MaterializedMetadataError) as excinfo:
        raw_contract_inputs_for_step(
            planner_declared_inputs=["age"],
            primary_cohort_execution_receipt=receipt,
        )

    assert "event-time column" in str(excinfo.value)


def test_the_coder_is_told_how_to_read_a_windowed_row():
    """A field the host publishes but never explains is not a contract."""

    from easyicu.research_agent.resources import coder as coder_resources

    source = Path(coder_resources.__file__).read_text(encoding="utf-8")
    start = source.index("deterministically resolved the Planner-owned predicates")
    guidance = source[start : start + 2500]

    assert "`event_time_column` is not null" in guidance
    assert "event_time_start_hours <= " in guidance
    assert "event_time_end_hours" in guidance
    # The rule that is not in any row: what a missing event time means.
    assert "missing event time as outside" in guidance
    # And why it matters, so the instruction is not read as optional colour.
    assert "gives a different count" in guidance
