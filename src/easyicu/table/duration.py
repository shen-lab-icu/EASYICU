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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: ``DataFrame.attrs`` key carrying the declared unit of ``dur_var``.
DUR_VAR_UNIT_ATTR = "easyicu_dur_var_unit"

UNIT_MINUTES = "minutes"
UNIT_HOURS = "hours"
UNIT_TIMEDELTA = "timedelta"

VALID_DUR_VAR_UNITS = frozenset({UNIT_MINUTES, UNIT_HOURS, UNIT_TIMEDELTA})

#: Opt-OUT flag. Guessing is off by default: a warning does not stop a wrong
#: number from reaching a manuscript, so an undeclared unit is an error unless
#: the caller explicitly accepts the legacy guess.
ALLOW_GUESS_ENV_VAR = "EASYICU_ALLOW_DUR_VAR_UNIT_GUESS"

#: Retired opt-IN flag, still honoured so existing scripts keep working.
STRICT_ENV_VAR = "EASYICU_STRICT_DUR_VAR_UNIT"


class DurationUnitError(ValueError):
    """Raised when a ``dur_var`` unit is required but cannot be determined."""


class DurationValueError(ValueError):
    """Raised when a ``dur_var`` value cannot be a real window duration."""


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def strict_dur_var_units() -> bool:
    """Return True when an undeclared numeric ``dur_var`` must fail closed.

    Strict is the default. ``EASYICU_ALLOW_DUR_VAR_UNIT_GUESS=1`` re-enables the
    legacy distribution guess for a transitional path that has no declaration
    yet; the old ``EASYICU_STRICT_DUR_VAR_UNIT=1`` remains a no-op-compatible
    way to ask for the same strict behaviour.
    """

    if _env_flag(ALLOW_GUESS_ENV_VAR):
        return False
    return True


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


def _validate_hours(
    hours: pd.Series,
    *,
    column: str,
    concept: Optional[str],
) -> pd.Series:
    """Reject durations that cannot describe a real window.

    Previously every one of these was coerced to ``0.0``, and a zero-length
    window still emits its start point — so a corrupt or unparseable record
    became a *valid* exposure point for a vasopressor, infusion or ventilation
    concept. Negative and infinite values are always corrupt and raise; NaN is
    plausible in real source data, so those rows are dropped (as NaN) and
    counted in a warning rather than silently becoming zero-length exposures.
    """

    label = concept or "<unknown>"

    infinite = np.isinf(hours.to_numpy(dtype="float64", na_value=np.nan))
    if bool(infinite.any()):
        raise DurationValueError(
            f"{column!r} for concept {label!r} contains non-finite durations "
            f"({int(infinite.sum())} row(s)); the source record is corrupt"
        )

    negative = hours < 0
    if bool(negative.any()):
        raise DurationValueError(
            f"{column!r} for concept {label!r} contains {int(negative.sum())} "
            "negative duration(s); a window cannot end before it starts"
        )

    missing = int(hours.isna().sum())
    if missing:
        logger.warning(
            "dropping %d row(s) with a missing %s for concept %r; a missing "
            "duration is not a zero-length window.",
            missing,
            column,
            label,
        )
    return hours


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
        return _validate_hours(
            series.dt.total_seconds().div(3600.0), column=column, concept=concept
        )

    numeric = pd.to_numeric(series, errors="coerce")
    unit = get_dur_var_unit(frame)
    if unit == UNIT_HOURS:
        return _validate_hours(numeric, column=column, concept=concept)
    if unit == UNIT_MINUTES:
        return _validate_hours(numeric.div(60.0), column=column, concept=concept)
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
            f"at the producer. Set {ALLOW_GUESS_ENV_VAR}=1 to fall back to the "
            "legacy distribution guess, which cannot distinguish 10 minutes from "
            "10 hours and may inflate a window 60x"
        )

    is_hours = _heuristic_unit_is_hours(numeric.fillna(0.0))
    logger.warning(
        "dur_var for concept %r has no declared unit; %s=1 so guessing %s from the "
        "value distribution. This guess cannot distinguish 10 minutes from 10 hours "
        "— declare the unit at the producer with set_dur_var_unit().",
        concept or "<unknown>",
        ALLOW_GUESS_ENV_VAR,
        UNIT_HOURS if is_hours else UNIT_MINUTES,
    )
    guessed = numeric if is_hours else numeric.div(60.0)
    return _validate_hours(guessed, column=column, concept=concept)


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
