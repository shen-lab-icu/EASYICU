"""Explicit unit contract for the ``dur_var`` window-duration column.

Historically every producer of a numeric ``dur_var`` picked its own unit —
``ts_to_win_tbl`` emits hours when the index is numeric, the source-level
callback dispatcher emits minutes, ``grp_mount_to_rate`` emits hours for
HiRID's float clock — and every consumer re-guessed the unit from the value
distribution.  A distribution guess cannot separate "10 minutes" from
"10 hours", so a short infusion could be inflated 60x.

This module makes the unit a declared property of the frame instead of
something inferred from its values:

* producers call :func:`set_dur_var_unit` right where they write the column;
* consumers call :func:`resolve_dur_var_hours`, which converts exactly when
  the unit is known and only falls back to the legacy heuristic (loudly, and
  never under strict mode) when it is not.

``timedelta64`` durations are self-describing and always take the exact path.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: ``DataFrame.attrs`` key carrying the declared unit of ``dur_var``.
DUR_VAR_UNIT_ATTR = "easyicu_dur_var_unit"

UNIT_MINUTES = "minutes"
UNIT_HOURS = "hours"
UNIT_TIMEDELTA = "timedelta"

VALID_DUR_VAR_UNITS = frozenset({UNIT_MINUTES, UNIT_HOURS, UNIT_TIMEDELTA})

#: Env flag that turns an undeclared numeric ``dur_var`` into a hard error.
STRICT_ENV_VAR = "EASYICU_STRICT_DUR_VAR_UNIT"


class DurationUnitError(ValueError):
    """Raised when a ``dur_var`` unit is required but cannot be determined."""


def strict_dur_var_units() -> bool:
    """Return True when an undeclared numeric ``dur_var`` must fail closed."""

    raw = str(os.environ.get(STRICT_ENV_VAR, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def set_dur_var_unit(frame: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Declare the unit of ``frame``'s ``dur_var`` column.

    Returns the same frame so producers can use it inline. The unit lives in
    ``DataFrame.attrs``, which pandas carries across ``copy``/slice/assign.
    """

    if unit not in VALID_DUR_VAR_UNITS:
        raise DurationUnitError(
            f"unknown dur_var unit {unit!r}; expected one of "
            + ", ".join(sorted(VALID_DUR_VAR_UNITS))
        )
    if isinstance(frame, pd.DataFrame):
        frame.attrs[DUR_VAR_UNIT_ATTR] = unit
    return frame


def get_dur_var_unit(frame: pd.DataFrame) -> Optional[str]:
    """Return the declared ``dur_var`` unit, or None when undeclared."""

    if not isinstance(frame, pd.DataFrame):
        return None
    unit = frame.attrs.get(DUR_VAR_UNIT_ATTR)
    return unit if unit in VALID_DUR_VAR_UNITS else None


def clear_dur_var_unit(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop a stale unit declaration (e.g. after the column is removed)."""

    if isinstance(frame, pd.DataFrame):
        frame.attrs.pop(DUR_VAR_UNIT_ATTR, None)
    return frame


def _heuristic_unit_is_hours(values: pd.Series) -> bool:
    """Legacy distribution guess, kept only as an undeclared-unit fallback."""

    sample = values.dropna()
    if sample.empty:
        return False
    return float(sample.quantile(0.95)) <= 48.0 and float(sample.median()) <= 24.0


def resolve_dur_var_hours(
    frame: pd.DataFrame,
    *,
    column: str = "dur_var",
    concept: Optional[str] = None,
    strict: Optional[bool] = None,
) -> pd.Series:
    """Return ``frame[column]`` converted to float hours.

    The conversion is exact whenever the dtype is ``timedelta64`` or the frame
    declares a unit. Otherwise it falls back to the legacy distribution guess
    and logs a warning naming the concept — unless strict mode is on, in which
    case it raises :class:`DurationUnitError` rather than guessing.
    """

    if column not in frame.columns:
        raise DurationUnitError(f"frame has no {column!r} column")

    series = frame[column]
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds().div(3600.0).fillna(0.0)

    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    unit = get_dur_var_unit(frame)
    if unit == UNIT_HOURS:
        return numeric
    if unit == UNIT_MINUTES:
        return numeric.div(60.0)
    if unit == UNIT_TIMEDELTA:
        # Declared timedelta but already coerced to a numeric dtype upstream:
        # the values are nanoseconds, which is never a plausible duration.
        raise DurationUnitError(
            f"{column!r} declares unit 'timedelta' but holds a numeric dtype; "
            "the producer must convert before the unit is consumed"
        )

    if strict is None:
        strict = strict_dur_var_units()
    if strict:
        raise DurationUnitError(
            f"{column!r} has no declared unit for concept {concept or '<unknown>'!r}; "
            f"set it with set_dur_var_unit(frame, '{UNIT_MINUTES}'|'{UNIT_HOURS}') "
            f"at the producer, or unset {STRICT_ENV_VAR} to allow the legacy guess"
        )

    is_hours = _heuristic_unit_is_hours(numeric)
    logger.warning(
        "dur_var for concept %r has no declared unit; guessing %s from the value "
        "distribution. This guess cannot distinguish 10 minutes from 10 hours — "
        "declare the unit at the producer with set_dur_var_unit().",
        concept or "<unknown>",
        UNIT_HOURS if is_hours else UNIT_MINUTES,
    )
    return numeric if is_hours else numeric.div(60.0)


__all__ = [
    "DUR_VAR_UNIT_ATTR",
    "UNIT_MINUTES",
    "UNIT_HOURS",
    "UNIT_TIMEDELTA",
    "VALID_DUR_VAR_UNITS",
    "STRICT_ENV_VAR",
    "DurationUnitError",
    "strict_dur_var_units",
    "set_dur_var_unit",
    "get_dur_var_unit",
    "clear_dur_var_unit",
    "resolve_dur_var_hours",
]
