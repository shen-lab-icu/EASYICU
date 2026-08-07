"""Something has to produce the columns the trajectory contract reads.

``trajectory/contract.py`` parses ``<family>_h<start>_<end>`` into
:class:`FixedWindowTrajectoryMetadata`, and the trajectory plan contract's
primary path requires two such columns from one family.  Nothing produced them:
over the recorded corpus that parser answered ``None`` for every column of all
234 sealed research contexts, so ``fixed_window_trajectory`` was populated
exactly zero times and h3_trajectory_clustering has never executed past step 01
in any of its 7 recorded runs.

MEASURED on h3's own sealed long table (19,067,154 rows x 5 columns, 94,442
stays, ``charttime`` already in ICU-hours from a fixed anchor over 0-72 h):
a 12-hour grid pivots it to 94,442 x 48 in about 3 s, and the parser reads back
48 of 48.

The grid is the caller's declaration, and h3's case protocol declares it:
"Build fixed-anchor ICU-hour trajectories over hours 0-72 ... Use a common time
grid, make missingness explicit."  This module chooses no family, width,
horizon or aggregate.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.trajectory.contract import (
    infer_fixed_window_trajectory_metadata,
)
from easyicu.research_agent.trajectory.panel import (
    FixedWindowGrid,
    fixed_window_column_name,
    fixed_window_panel,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def _long(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["stay_id", "charttime", "concept", "value_num"]
    )


def test_every_emitted_column_is_one_the_parser_reads_back():
    """The producer and the parser share one convention; this is that link."""

    panel = fixed_window_panel(
        _long(
            [
                (1, 0.0, "sofa2", 3),
                (1, 13.0, "sofa2", 5),
                (2, 1.0, "sofa2", 1),
            ]
        ),
        grid=FixedWindowGrid(width_hours=12.0, horizon_hours=24.0),
        id_column="stay_id",
        time_column="charttime",
    )

    assert list(panel.columns) == ["sofa2_h0_12", "sofa2_h12_24"]
    for column in panel.columns:
        metadata = infer_fixed_window_trajectory_metadata(
            column_name=column, values=panel[column], source_scale="ordinal"
        )
        assert metadata is not None, column
        assert metadata.family == "sofa2"


def test_the_last_window_closes_on_the_horizon():
    """A grid over "hours 0-72" has to contain hour 72.

    MEASURED on h3: 159,095 of 19,067,154 rows sit at exactly hour 72.0, across
    67,162 of 94,442 stays. A half-open final window dropped all of them, and
    for a stay whose only late observation is the endpoint that turns a real
    value into missing.
    """

    panel = fixed_window_panel(
        _long([(1, 72.0, "sofa2", 9), (1, 61.0, "sofa2", 2)]),
        grid=FixedWindowGrid(width_hours=12.0, horizon_hours=72.0),
        id_column="stay_id",
        time_column="charttime",
    )

    assert panel.loc[1, "sofa2_h60_72"] == 9


def test_beyond_the_horizon_is_dropped_not_folded_in():
    """A trajectory declared over 0-72 h that absorbed hour 96 is a different one."""

    panel = fixed_window_panel(
        _long([(1, 5.0, "sofa2", 1), (1, 96.0, "sofa2", 9)]),
        grid=FixedWindowGrid(width_hours=12.0, horizon_hours=72.0),
        id_column="stay_id",
        time_column="charttime",
    )

    assert panel.loc[1, "sofa2_h0_12"] == 1
    assert "sofa2_h60_72" not in panel.columns


def test_a_window_with_no_observation_stays_missing():
    """Filling it would erase the length-biased sampling the analysis must reason about.

    Two stays, because a column only exists when SOMEONE was observed in that
    window.  A first version used one stay, so the late windows produced no
    columns at all and a mutation that filled missing cells with 0 survived --
    on organ-dysfunction scores that reads a discharged patient as perfectly
    healthy, which inverts the meaning rather than merely blurring it.
    """

    panel = fixed_window_panel(
        _long(
            [
                (1, 1.0, "sofa2", 4),
                (2, 1.0, "sofa2", 6),
                (2, 25.0, "sofa2", 7),
            ]
        ),
        grid=FixedWindowGrid(width_hours=12.0, horizon_hours=36.0),
        id_column="stay_id",
        time_column="charttime",
    )

    assert panel.loc[1, "sofa2_h0_12"] == 4
    assert panel.loc[2, "sofa2_h24_36"] == 7
    # Stay 1 left before the last window. The column exists because stay 2 was
    # there; stay 1's cell must be missing, not zero.
    assert pd.isna(panel.loc[1, "sofa2_h24_36"])
    # And the window nobody was observed in is not invented for either of them.
    assert "sofa2_h12_24" not in panel.columns


def test_the_aggregate_is_declared_not_assumed():
    rows = _long([(1, 1.0, "lact", 2.0), (1, 5.0, "lact", 7.0)])
    grid_kwargs = dict(width_hours=12.0, horizon_hours=12.0)

    worst = fixed_window_panel(
        rows,
        grid=FixedWindowGrid(aggregate="max", **grid_kwargs),
        id_column="stay_id",
        time_column="charttime",
    )
    earliest = fixed_window_panel(
        rows,
        grid=FixedWindowGrid(aggregate="first", **grid_kwargs),
        id_column="stay_id",
        time_column="charttime",
    )

    assert worst.loc[1, "lact_h0_12"] == 7.0
    assert earliest.loc[1, "lact_h0_12"] == 2.0


def test_a_fractional_bound_round_trips_through_the_column_name():
    """The parser spells a decimal point ``p``; the producer must match it."""

    name = fixed_window_column_name("sofa2", 0.0, 1.5)
    assert name == "sofa2_h0_1p5"
    metadata = infer_fixed_window_trajectory_metadata(
        column_name=name, values=pd.Series([1.0]), source_scale="ordinal"
    )
    assert metadata is not None
    assert metadata.window_end_hours == 1.5


def test_a_grid_narrower_than_one_window_is_refused():
    with pytest.raises(ValueError):
        FixedWindowGrid(width_hours=24.0, horizon_hours=12.0)
    with pytest.raises(ValueError):
        FixedWindowGrid(width_hours=0.0, horizon_hours=12.0)


def test_a_missing_column_is_named_rather_than_guessed():
    with pytest.raises(ValueError, match="missing column"):
        fixed_window_panel(
            pd.DataFrame({"stay_id": [1]}),
            grid=FixedWindowGrid(width_hours=12.0, horizon_hours=24.0),
            id_column="stay_id",
            time_column="charttime",
        )


def test_the_real_sealed_trajectory_pivots_onto_the_declared_grid():
    """h3's own 19-million-row table, on the grid its protocol declares."""

    runs = sorted(_CORPUS.glob("batch_*/h3_*/aware/run_*/cohort_trajectory.parquet"))
    if not runs:
        pytest.skip("h3's sealed trajectory table is not on disk")
    frame = pd.read_parquet(
        runs[-1], columns=["stay_id", "charttime", "concept", "value_num"]
    )

    panel = fixed_window_panel(
        frame,
        grid=FixedWindowGrid(width_hours=12.0, horizon_hours=72.0),
        id_column="stay_id",
        time_column="charttime",
    )

    families: dict[str, int] = {}
    for column in panel.columns:
        metadata = infer_fixed_window_trajectory_metadata(
            column_name=column, values=panel[column], source_scale="ordinal"
        )
        assert metadata is not None, column
        families[metadata.family] = families.get(metadata.family, 0) + 1

    # The contract's primary path needs two windows from one family.
    assert families, panel.columns.tolist()
    assert min(families.values()) >= 2, families
    # Clustering needs stays that actually have a trajectory to cluster.
    total = [column for column in panel.columns if column.startswith("sofa2_h")]
    assert total, panel.columns.tolist()
    covered = (panel[total].notna().sum(axis=1) >= 2).mean()
    assert covered > 0.9, covered
    # The first window is the anchored one and must be the best covered.
    ordered = sorted(total, key=lambda c: float(c.split("_h")[1].split("_")[0]))
    coverage = [panel[c].notna().mean() for c in ordered]
    assert coverage == sorted(coverage, reverse=True), dict(zip(ordered, coverage))
    assert np.isclose(coverage[0], 1.0, atol=0.01), coverage[0]
    # Patients leave, so the late windows CANNOT be complete. This is the
    # length-biased sampling h3's own guardrails name, and a panel that filled
    # missing cells would report it as full follow-up.
    assert coverage[-1] < 0.9, coverage[-1]
