"""Validate time semantics before the resolver aligns index column names."""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

from ..table import ICUTable, TsTbl


RELATIVE_HOURS = "icu_admission_hours"
MergeTimeReason = Literal[
    "time_basis_unknown",
    "time_basis_mismatch",
    "time_basis_not_relative",
    "time_interval_unknown",
    "time_interval_mismatch",
    "time_grid_mismatch",
    "time_index_collision",
]


class MergeTimeIndexError(ValueError):
    """An index rename would merge incompatible or unproven time coordinates."""

    def __init__(self, reason_code: MergeTimeReason, concept: str) -> None:
        self.reason_code = reason_code
        self.concept = concept
        super().__init__(f"Cannot align concept time index: {reason_code} (concept={concept})")


def declare_resolved_time_index(
    table: ICUTable | TsTbl,
    *,
    align_to_admission: bool,
    interval: pd.Timedelta | None,
) -> None:
    """Record the resolver's admission alignment and resampling postcondition.

    Raw tables are never declared here. Explicit producer metadata wins over
    the request, including WinTbl's independently declared interval. Actual
    coordinates are checked again before any differing-name merge.
    """
    index = getattr(table, "index_column", None)
    if not align_to_admission or interval is None or not index:
        return
    values = table.data[index]
    if not values.empty and not pd.api.types.is_numeric_dtype(values):
        return
    if getattr(table, "time_basis", None) is None:
        table.time_basis = RELATIVE_HOURS
    if getattr(table, "interval", None) is None:
        table.interval = interval


def merge_index_name(tables: Mapping[str, ICUTable | TsTbl]) -> str | None:
    """Select the first time index; allow aliases only on one proven grid.

    Equal names and id-only tables retain the resolver's existing handling.
    Sparse observations need not have equal timestamps; every timestamp must
    fall on the declared admission-anchored grid. No values are rescaled,
    rounded, expanded or interpolated here.
    """
    indexed = [
        (name, table, table.index_column)
        for name, table in tables.items()
        if getattr(table, "index_column", None)
    ]
    if not indexed:
        return None
    target = indexed[0][2]
    if len({index for _, _, index in indexed}) == 1:
        return target

    expected_ids = list(next(iter(tables.values())).id_columns)
    for table in tables.values():
        if list(table.id_columns) != expected_ids:
            raise ValueError("All concepts must share identical identifier columns to merge")

    first_basis = getattr(indexed[0][1], "time_basis", None)
    first_interval = getattr(indexed[0][1], "interval", None)
    for name, table, index in indexed:
        basis = getattr(table, "time_basis", None)
        if basis is None:
            raise MergeTimeIndexError("time_basis_unknown", name)
        if basis != first_basis:
            raise MergeTimeIndexError("time_basis_mismatch", name)
        if basis != RELATIVE_HOURS:
            raise MergeTimeIndexError("time_basis_not_relative", name)
        interval = getattr(table, "interval", None)
        if interval is None or pd.isna(interval) or interval <= pd.Timedelta(0):
            raise MergeTimeIndexError("time_interval_unknown", name)
        if interval != first_interval:
            raise MergeTimeIndexError("time_interval_mismatch", name)
        if index != target and target in table.data:
            raise MergeTimeIndexError("time_index_collision", name)
        values = table.data[index]
        if values.empty:
            continue
        if not pd.api.types.is_numeric_dtype(values):
            raise MergeTimeIndexError("time_basis_not_relative", name)
        step = interval / pd.Timedelta(hours=1)
        # Bound validation scratch space even for full-database loads.
        for offset in range(0, len(values), 65_536):
            ticks = values.iloc[offset:offset + 65_536].dropna().to_numpy(dtype=float) / step
            if not np.all(np.isfinite(ticks) & np.isclose(ticks, np.rint(ticks), rtol=0, atol=1e-8)):
                raise MergeTimeIndexError("time_grid_mismatch", name)
    return target
