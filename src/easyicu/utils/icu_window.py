"""The ICU episode window that both extraction paths apply to raw event times.

A concept row is kept when its ICU-relative event hour lies in
``[-pre_admission_hours, upper]``, where ``upper`` is the stay's discharge
hour plus the post-discharge allowance (or the 366-day sanity bound when
the stay has none).  The pandas path applies the window in
``ConceptResolver._align_time_to_admission``; the DuckDB aggregation path
applies the same window inside SQL, on the raw event hour *before* the
hour floor.  Flooring first and clipping the bin afterwards kept rows up to
59 minutes past the bound (lead-r13 parity census), so both paths take the
window from this one object.

The per-database upper definitions stay with their owner in
``easyicu.concept`` (``ConceptResolver._icu_episode_window``); this module
only carries the resolved bounds and renders them for pandas or SQL.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = ["IcuEpisodeWindow", "IcuWindowSql"]

_VIEW_COUNTER = itertools.count()


def _sql_double(value: float) -> str:
    # A string cast parses the shortest round-trip repr exactly; a bare
    # decimal literal would pass through DuckDB's DECIMAL type first.
    return f"CAST('{float(value)!r}' AS DOUBLE)"


@dataclass(frozen=True)
class IcuEpisodeWindow:
    """Resolved ICU episode bounds in ICU-relative hours.

    ``upper_by_stay`` maps a stay key to its upper hour; stays missing from
    it use ``default_upper_hours``.  ``keep_post_discharge`` drops only the
    pre-admission side (outcomes recorded at hospital discharge).
    """

    pre_admission_hours: float
    default_upper_hours: float
    upper_by_stay: Optional[pd.Series] = None
    keep_post_discharge: bool = False

    def __post_init__(self) -> None:
        pre = float(self.pre_admission_hours)
        if not math.isfinite(pre) or pre < 0:
            raise ValueError(
                "pre_admission_hours must be a finite non-negative number"
            )
        if self.upper_by_stay is not None:
            upper = self.upper_by_stay
            if not upper.index.is_unique:
                raise ValueError("upper_by_stay must have one bound per stay")
            if upper.isna().any():
                raise ValueError("upper_by_stay must not contain missing bounds")

    @property
    def lower_hours(self) -> float:
        return -float(self.pre_admission_hours)

    def upper_for(self, stay_keys: pd.Series) -> pd.Series:
        """Return each row's upper hour, aligned to ``stay_keys``."""

        default = float(self.default_upper_hours)
        if self.upper_by_stay is None:
            return pd.Series(default, index=stay_keys.index, dtype="float64")
        keys = pd.to_numeric(stay_keys, errors="coerce").astype("Int64")
        upper = keys.map(self.upper_by_stay).astype("float64").fillna(default)
        upper.index = stay_keys.index
        return upper

    def outside(self, event_hours: pd.Series, stay_keys: pd.Series) -> pd.Series:
        """Boolean mask of rows outside the window (missing hours are kept)."""

        hours = pd.to_numeric(event_hours, errors="coerce")
        outside = hours < self.lower_hours
        if not self.keep_post_discharge:
            outside = outside | (hours > self.upper_for(stay_keys).to_numpy())
        return (outside & hours.notna()).astype(bool)


class IcuWindowSql:
    """Render an :class:`IcuEpisodeWindow` as a DuckDB row predicate.

    The per-stay bounds are registered on ``conn`` on first use and removed
    by :meth:`close`.  ``window=None`` renders no predicate (the concept is
    not clipped, e.g. a community release or ``align_to_admission=False``).
    """

    def __init__(self, conn: Any, window: Optional[IcuEpisodeWindow]) -> None:
        self._conn = conn
        self._window = window
        self._view: Optional[str] = None

    @property
    def active(self) -> bool:
        return self._window is not None

    def _relation(self) -> Optional[str]:
        window = self._window
        if window is None or window.upper_by_stay is None:
            return None
        if self._view is None:
            upper = window.upper_by_stay
            frame = pd.DataFrame(
                {
                    "stay": pd.to_numeric(upper.index, errors="coerce").astype(
                        np.int64
                    ),
                    "upper_hours": upper.to_numpy(dtype="float64"),
                }
            )
            name = f"__easyicu_icu_window_{next(_VIEW_COUNTER)}"
            self._conn.register(name, frame)
            self._view = name
        return self._view

    def condition(self, hours_expr: str, stay_expr: str) -> str:
        """Return ``" AND (...)"`` keeping rows inside the window, or ``""``."""

        window = self._window
        if window is None:
            return ""
        hours = f"({hours_expr})"
        parts = [f"{hours} >= {_sql_double(window.lower_hours)}"]
        if not window.keep_post_discharge:
            default = _sql_double(window.default_upper_hours)
            relation = self._relation()
            upper = (
                default
                if relation is None
                else (
                    f"COALESCE((SELECT __w.upper_hours FROM {relation} __w "
                    f"WHERE __w.stay = {stay_expr}), {default})"
                )
            )
            parts.append(f"{hours} <= {upper}")
        return f" AND ({hours} IS NULL OR ({' AND '.join(parts)}))"

    def close(self) -> None:
        if self._view is not None:
            try:
                self._conn.unregister(self._view)
            finally:
                self._view = None
