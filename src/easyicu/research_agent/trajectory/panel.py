"""Produce the fixed-window columns the trajectory contract reads.

``contract.py`` parses ``<family>_h<start>_<end>`` and turns it into
:class:`FixedWindowTrajectoryMetadata`; nothing produced those columns.  Over
the recorded corpus that parser answered ``None`` for every column of all 234
sealed research contexts, so ``fixed_window_trajectory`` was populated exactly
zero times and the trajectory plan contract -- whose primary path requires two
such columns from one family -- could never be satisfied.

The producer lives beside the parser deliberately: the two must agree on one
naming convention, and the convention is this module's only contract with the
rest of the system.  It chooses no family, no width, no horizon and no
aggregate; a caller declares the grid, and a case protocol declares it to the
caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import pandas as pd

from .contract import infer_fixed_window_trajectory_metadata

__all__ = [
    "FixedWindowGrid",
    "fixed_window_column_name",
    "fixed_window_panel",
]

#: How a window's value is reduced from the observations inside it.  ``max`` is
#: the ICU convention for organ-dysfunction scores and peak lactate ("worst in
#: window"); the alternatives exist so a caller whose concept is not
#: worst-is-highest does not have to misuse it.
WindowAggregate = Literal["max", "min", "mean", "first", "last"]


def _format_bound(hours: float) -> str:
    """Render a bound the way :mod:`.contract` parses it back.

    The parser accepts ``12`` and ``1p5`` (its ``p`` stands in for a decimal
    point, since a dot cannot appear in a column name).  Whole hours are written
    without a fractional part so the common grid reads as ``h0_12``.
    """

    value = float(hours)
    if value.is_integer():
        return str(int(value))
    return repr(value).replace(".", "p")


def fixed_window_column_name(family: str, start_hours: float, end_hours: float) -> str:
    """The one spelling this package produces and parses."""

    return f"{family}_h{_format_bound(start_hours)}_{_format_bound(end_hours)}"


@dataclass(frozen=True)
class FixedWindowGrid:
    """A uniform grid over ``[0, horizon_hours)``, declared by the caller.

    ``width_hours`` and ``horizon_hours`` are the whole contract: the grid is
    uniform because the trajectory contract compares windows within a family,
    and a family whose windows have different widths is not one trajectory.
    """

    width_hours: float
    horizon_hours: float
    aggregate: WindowAggregate = "max"

    def __post_init__(self) -> None:
        if not (self.width_hours > 0):
            raise ValueError("width_hours must be positive")
        if not (self.horizon_hours > 0):
            raise ValueError("horizon_hours must be positive")
        if self.horizon_hours < self.width_hours:
            raise ValueError("horizon_hours must be at least one window wide")

    @property
    def edges(self) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        start = 0.0
        while start < self.horizon_hours - 1e-9:
            end = min(start + self.width_hours, self.horizon_hours)
            bounds.append((start, end))
            start = end
        return bounds

    def as_manifest(self) -> dict:
        """What a caller records so a reader can replay the grid."""

        return {
            "schema_version": "easyicu.fixed_window_grid/1",
            "width_hours": float(self.width_hours),
            "horizon_hours": float(self.horizon_hours),
            "aggregate": str(self.aggregate),
            "windows": [
                {"start_hours": start, "end_hours": end} for start, end in self.edges
            ],
        }


def fixed_window_panel(
    long_frame: pd.DataFrame,
    *,
    grid: FixedWindowGrid,
    id_column: str,
    time_column: str,
    concept_column: str = "concept",
    value_column: str = "value_num",
    concepts: Optional[Sequence[str]] = None,
    ids: Optional[Iterable] = None,
) -> pd.DataFrame:
    """Pivot a long per-timepoint table onto ``grid``, one column per window.

    ``time_column`` holds hours from the cohort's own anchor, so the caller owns
    alignment and this function never guesses one.

    Windows are half-open except the last, which CLOSES on ``horizon_hours``:
    ``[0,12) [12,24) ... [60,72]``.  A grid declared over "hours 0-72" has to
    contain hour 72, and the long tables this reads are themselves clipped to
    that closed bound.  MEASURED on h3_trajectory_clustering: 159,095 of
    19,067,154 rows sit at exactly hour 72.0, across 67,162 of the 94,442 stays.
    A half-open final window dropped every one of them, which for a stay whose
    only late observation is the endpoint turns a real value into missing --
    silently discarding the bound the export deliberately included.  Anything
    beyond ``horizon_hours`` is still dropped rather than folded in: a
    trajectory declared over 0-72 h that absorbed hour 96 would not be the
    trajectory the protocol described.

    Missing windows stay missing.  A stay that left the ICU has no observation
    to summarize, and filling that with anything would erase the length-biased
    sampling a trajectory analysis has to reason about.

    Returns a frame indexed by ``id_column`` whose every column is one the
    parser in :mod:`.contract` recognises.
    """

    required = {id_column, time_column, concept_column, value_column}
    missing = sorted(required - set(long_frame.columns))
    if missing:
        raise ValueError(f"long frame is missing column(s): {missing}")

    frame = long_frame.loc[:, [id_column, time_column, concept_column, value_column]]
    if ids is not None:
        frame = frame.loc[frame[id_column].isin(set(ids))]
    if concepts is not None:
        frame = frame.loc[frame[concept_column].isin(list(dict.fromkeys(concepts)))]

    hours = pd.to_numeric(frame[time_column], errors="coerce")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    keep = hours.notna() & values.notna() & (hours >= 0) & (hours <= grid.horizon_hours)
    frame = frame.loc[keep].assign(
        _easyicu_hours=hours.loc[keep], _easyicu_value=values.loc[keep]
    )
    edges = grid.edges
    if frame.empty:
        return pd.DataFrame(index=pd.Index([], name=id_column))

    index = (frame["_easyicu_hours"] // grid.width_hours).astype(int)
    frame = frame.assign(_easyicu_window=index.clip(upper=len(edges) - 1))

    reduced = (
        frame.groupby(
            [id_column, concept_column, "_easyicu_window"], observed=True, sort=False
        )["_easyicu_value"]
        .agg(grid.aggregate)
        .reset_index()
    )
    reduced["_easyicu_column"] = [
        fixed_window_column_name(str(family), *edges[int(window)])
        for family, window in zip(
            reduced[concept_column], reduced["_easyicu_window"], strict=True
        )
    ]
    panel = reduced.pivot(
        index=id_column, columns="_easyicu_column", values="_easyicu_value"
    )
    panel.columns.name = None
    # Every emitted column must be one the parser reads back; a family whose
    # name cannot round-trip (a concept id with a hyphen, say) would otherwise
    # ship a column the contract silently ignores.
    unreadable = [
        column
        for column in panel.columns
        if infer_fixed_window_trajectory_metadata(
            column_name=column, values=panel[column], source_scale="unknown"
        )
        is None
    ]
    if unreadable:
        raise ValueError(
            "these window columns are not readable by the trajectory contract, "
            f"so the concept ids cannot be used as family names: {sorted(unreadable)}"
        )
    return panel.reindex(sorted(panel.columns), axis=1)
