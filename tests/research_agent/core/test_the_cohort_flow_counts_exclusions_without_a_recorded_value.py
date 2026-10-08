"""A cohort step's exclusions say how many it read without a recorded value.

The attrition ledger said how many stays each step excluded, never why. A stay
with no recorded lactate fails ``lact >= 2`` exactly as a stay with a lactate
of 1 does, so a step that removed 300 stays could not say whether they had low
values or none -- and the receipt, the figure and the reader all saw only the
300. ``n_excluded_missing`` is that split: of the stays a step excluded, those
its predicate read without a recorded value, counted from the same mask that
excluded them. For a predicate read over an event-time window the event's time
is read too, so an event without a recorded time counts as well.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.cohort.schema import (
    _build_cohort_with_flow,
    load_materialized_analysis_cohort_result,
    materialize_locked_analysis_cohort,
)
from easyicu.research_agent.gates import figure_privacy, publication_disclosure
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.schema import AnalysisPlan


def _window(end: float = 24.0) -> TimeWindow:
    return TimeWindow(anchor="icu_admit", start_offset_hours=0.0, end_offset_hours=end)


def _lactate(op: str, value: object) -> ConceptPredicate:
    return ConceptPredicate(
        concept_id="lact",
        time_window=_window(),
        aggregation="first",
        op=op,
        value=value,
    )


def _lactate_universe() -> pd.DataFrame:
    """Six stays; stays 4 and 5 have no recorded lactate."""

    return pd.DataFrame(
        {"stay_id": [1, 2, 3, 4, 5, 6], "lact": [1.0, 5.0, 9.0, None, None, 5.0]}
    )


@pytest.mark.parametrize(
    ("op", "value", "kind", "excluded", "missing"),
    [
        ("==", 5, "inclusion", 4, 2),
        ("==", 5, "exclusion", 2, 0),
        # A missing value is not 5: ``!=`` and ``not_in`` hold on it, as
        # pandas reads them. An inclusion keeps such a stay; an exclusion
        # removes it, and the count says so.
        ("!=", 5, "inclusion", 2, 0),
        ("!=", 5, "exclusion", 4, 2),
        ("<", 5, "inclusion", 5, 2),
        ("<", 5, "exclusion", 1, 0),
        ("<=", 5, "inclusion", 3, 2),
        ("<=", 5, "exclusion", 3, 0),
        (">", 5, "inclusion", 5, 2),
        (">", 5, "exclusion", 1, 0),
        (">=", 5, "inclusion", 3, 2),
        (">=", 5, "exclusion", 3, 0),
        ("in", [5], "inclusion", 4, 2),
        ("in", [5], "exclusion", 2, 0),
        ("not_in", [5], "inclusion", 2, 0),
        ("not_in", [5], "exclusion", 4, 2),
        # A missingness check reads nothing but the absence of a value.
        ("missing", None, "inclusion", 4, 0),
        ("missing", None, "exclusion", 2, 2),
        ("not_missing", None, "inclusion", 2, 2),
        ("not_missing", None, "exclusion", 4, 0),
    ],
)
def test_every_operator_counts_the_excluded_it_read_without_a_value(
    op: str, value: object, kind: str, excluded: int, missing: int
) -> None:
    definition = CohortDefinition(name="lactate", **{kind: (_lactate(op, value),)})
    cohort, flow = _build_cohort_with_flow(definition, _lactate_universe())

    universe, step = flow
    assert (universe["n_excluded"], universe["n_excluded_missing"]) == (0, 0)
    assert (step["n_excluded"], step["n_excluded_missing"]) == (excluded, missing)
    assert len(cohort) == 6 - excluded


def _death_universe() -> pd.DataFrame:
    """Stay 3 died without a recorded time; stay 5 has no recorded status.

    Stay 4 did not die and has no time, which is no missing value: a stay
    without the event has no event time to record.
    """

    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "death": [1.0, 1.0, 1.0, 0.0, None, 0.0],
            "death_time": [2.0, 30.0, None, None, None, None],
        }
    )


@pytest.mark.parametrize(
    ("level", "kind", "excluded", "missing"),
    [
        # Died within 24 h: only stay 1. Stay 3's death lies outside every
        # window, so the exclusion keeps it and counts nothing.
        (1, "exclusion", 1, 0),
        # Of the five stays it drops, stay 3 (no time) and stay 5 (no
        # status) were read without a value; stay 4 needs no time.
        (1, "inclusion", 5, 2),
        # Alive through 24 h keeps stay 3 and drops stay 5, for its status.
        (0, "inclusion", 2, 1),
        (0, "exclusion", 4, 1),
    ],
)
def test_a_windowed_event_counts_an_event_without_a_recorded_time(
    level: int, kind: str, excluded: int, missing: int
) -> None:
    predicate = ConceptPredicate(
        concept_id="death",
        time_window=_window(),
        aggregation="any",
        op="==",
        value=level,
    )
    definition = CohortDefinition(name="landmark", **{kind: (predicate,)})
    _, flow = _build_cohort_with_flow(definition, _death_universe())

    assert flow[1]["event_time_column"] == "death_time"
    assert (flow[1]["n_excluded"], flow[1]["n_excluded_missing"]) == (excluded, missing)


_OPS = ("==", "!=", "<", "<=", ">", ">=", "in", "not_in", "missing", "not_missing")


def _holds(value: float, op: str, target: float) -> bool:
    """One stay's predicate, written out by hand from pandas' reading."""

    missing = math.isnan(value)
    if op == "missing":
        return missing
    if op == "not_missing":
        return not missing
    if missing:
        return op in {"!=", "not_in"}
    return {
        "==": value == target,
        "!=": value != target,
        "<": value < target,
        "<=": value <= target,
        ">": value > target,
        ">=": value >= target,
        "in": value == target,
        "not_in": value != target,
    }[op]


@pytest.mark.parametrize("seed", range(12))
def test_the_count_matches_a_stay_by_stay_recount_and_stays_inside_its_exclusion(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    columns = ("lact", "crea", "map")
    universe = pd.DataFrame(
        {
            column: np.where(
                rng.random(60) < 0.3, np.nan, rng.integers(0, 6, 60).astype(float)
            )
            for column in columns
        }
    )
    steps = []
    for _ in range(4):
        op = str(rng.choice(_OPS))
        target = float(rng.integers(0, 6))
        value = None if "missing" in op else [target] if "in" in op else target
        steps.append(
            (
                str(rng.choice(["inclusion", "exclusion"])),
                str(rng.choice(columns)),
                op,
                target,
                value,
            )
        )
    definition = CohortDefinition(
        name="random",
        inclusion=tuple(
            ConceptPredicate(column, _window(), "first", op, value)
            for kind, column, op, _target, value in steps
            if kind == "inclusion"
        ),
        exclusion=tuple(
            ConceptPredicate(column, _window(), "first", op, value)
            for kind, column, op, _target, value in steps
            if kind == "exclusion"
        ),
    )
    _, flow = _build_cohort_with_flow(definition, universe)

    ordered = [step for step in steps if step[0] == "inclusion"] + [
        step for step in steps if step[0] == "exclusion"
    ]
    remaining = list(range(len(universe)))
    assert flow[0]["n_excluded_missing"] == 0
    for row, (kind, column, op, target, _value) in zip(flow[1:], ordered):
        values = universe[column]
        kept = [
            index
            for index in remaining
            if _holds(values[index], op, target) == (kind == "inclusion")
        ]
        dropped = sorted(set(remaining) - set(kept))
        assert row["n_excluded_missing"] == sum(
            math.isnan(values[index]) for index in dropped
        )
        assert 0 <= row["n_excluded_missing"] <= row["n_excluded"]
        assert row["n_excluded"] == row["n_before"] - row["n_remaining"] == len(dropped)
        remaining = kept


def _plan(universe_path: Path, run_dir: Path) -> tuple[AnalysisPlan, dict]:
    plan = AnalysisPlan(
        research_question="Apply the declared eligibility predicate.",
        cohort=CohortDefinition(name="lactate", inclusion=(_lactate(">=", 2),)),
        robustness_specs=[],
        steps=[],
    )
    result = materialize_locked_analysis_cohort(
        run_dir=run_dir, plan=plan, universe_path=universe_path
    )
    return plan, result


def test_the_published_ledger_carries_the_count_and_is_adopted_again(
    tmp_path: Path,
) -> None:
    universe_path = tmp_path / "cohort.parquet"
    _lactate_universe().to_parquet(universe_path, index=False)
    plan, result = _plan(universe_path, tmp_path)

    flow = pd.read_csv(result["flow_path"])
    # ``lact >= 2`` drops stay 1 on its value and stays 4 and 5 on none.
    assert flow["n_excluded"].tolist() == [0, 3]
    assert flow["n_excluded_missing"].tolist() == [0, 2]
    provenance = json.loads(
        (tmp_path / "cohort_analysis_provenance.json").read_text(encoding="utf-8")
    )
    assert [row["n_excluded_missing"] for row in provenance["cohort_flow"]] == [0, 2]
    assert load_materialized_analysis_cohort_result(run_dir=tmp_path, plan=plan)


def test_a_ledger_written_before_the_count_is_still_adopted(tmp_path: Path) -> None:
    universe_path = tmp_path / "cohort.parquet"
    _lactate_universe().to_parquet(universe_path, index=False)
    plan, result = _plan(universe_path, tmp_path)
    provenance_path = tmp_path / "cohort_analysis_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for row in provenance["cohort_flow"]:
        del row["n_excluded_missing"]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    pd.read_csv(result["flow_path"]).drop(columns="n_excluded_missing").to_csv(
        result["flow_path"], index=False
    )

    adopted = load_materialized_analysis_cohort_result(run_dir=tmp_path, plan=plan)

    assert adopted is not None and adopted["n_cohort"] == 3


def test_the_count_is_a_subject_count_under_both_disclosure_rules(
    tmp_path: Path,
) -> None:
    """Publication review lists it from 1 to 10; figure upload refuses it below 20."""

    assert publication_disclosure.is_subject_count_name("n_excluded_missing")
    assert publication_disclosure.small_cell_value(3) == 3
    ledger = tmp_path / "flow.csv"
    pd.DataFrame(
        [[0, "universe", 500, 0, 500, 0], [1, "inclusion", 500, 40, 460, 3]],
        columns=[
            "step_order",
            "predicate_kind",
            "n_before",
            "n_excluded",
            "n_remaining",
            "n_excluded_missing",
        ],
    ).to_csv(ledger, index=False)

    reasons = figure_privacy._inspect_delimited(ledger, delimiter=",")["reasons"]

    assert [reason for reason in reasons if "below" in reason] == [
        "group size(s) or subject count(s) below 20: n_excluded_missing=3"
    ]
