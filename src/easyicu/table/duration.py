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
  the unit is known and raises :class:`DurationUnitError` when it is not.
  The legacy distribution guess and its ``EASYICU_ALLOW_DUR_VAR_UNIT_GUESS``
  opt-in were removed (lead-r13, units are declared, never inferred from
  values).

``timedelta64`` durations are self-describing and always take the exact path.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: ``DataFrame.attrs`` key carrying the declared unit of ``dur_var``.
DUR_VAR_UNIT_ATTR = "easyicu_dur_var_unit"

UNIT_SECONDS = "seconds"
UNIT_MINUTES = "minutes"
UNIT_HOURS = "hours"
UNIT_DAYS = "days"
UNIT_TIMEDELTA = "timedelta"

VALID_DUR_VAR_UNITS = frozenset(
    {UNIT_SECONDS, UNIT_MINUTES, UNIT_HOURS, UNIT_DAYS, UNIT_TIMEDELTA}
)

#: Hours per numeric unit, so every conversion is one multiplication.
_HOURS_PER_UNIT = {
    UNIT_SECONDS: 1.0 / 3600.0,
    UNIT_MINUTES: 1.0 / 60.0,
    UNIT_HOURS: 1.0,
    UNIT_DAYS: 24.0,
}

#: Retired opt-out flag. It used to re-enable a distribution guess for an
#: undeclared ``dur_var``; it is no longer read, and an undeclared unit is
#: always an error (lead-r13 magnitude-inference sweep, #9).
ALLOW_GUESS_ENV_VAR = "EASYICU_ALLOW_DUR_VAR_UNIT_GUESS"

#: Retired opt-IN flag, still honoured so existing scripts keep working.
STRICT_ENV_VAR = "EASYICU_STRICT_DUR_VAR_UNIT"


class DurationUnitError(ValueError):
    """Raised when a ``dur_var`` unit is required but cannot be determined."""


class DurationUnitUndeclaredError(DurationUnitError):
    """A numeric duration reached a consumer without a declared unit."""

    reason_code = "duration_unit_undeclared"


class DurationValueError(ValueError):
    """Raised when a ``dur_var`` value cannot be a real window duration."""


class WindowContractError(ValueError):
    """Raised when window tables being combined do not describe one window."""


def assert_window_contract(
    specs: Sequence[tuple],
    *,
    column: Optional[str],
) -> None:
    """Check that every window input agrees on what its window *is*.

    ``specs`` is a sequence of ``(label, dur_var, index_var, id_vars)``, one per
    window table taking part in the combine. Kept as plain tuples so this
    module stays free of any table-class dependency.

    The unit check alone was not enough. It is driven by one column name — the
    first input's ``dur_var`` — so a right-hand table whose duration column is
    called something else simply has no column to check: it is skipped, its
    unit declaration is dropped, and its duration survives into the result as
    an ordinary numeric column with its window meaning gone. Nothing raises,
    and the combined table claims to be a window table over the *other*
    column. Binding a duration to the wrong index or the wrong id is the same
    class of silent error, so all three are checked here.
    """

    windows = [spec for spec in specs if spec is not None]
    if len(windows) < 2:
        return

    def _mismatch(field: str, position: int, expected: Any, found: Any) -> None:
        raise WindowContractError(
            f"cannot combine window tables with different {field}: input 0 has "
            f"{expected!r} and input {position} has {found!r}. A window table's "
            f"{field} carries its meaning; combining them would keep the first "
            "table's declaration and silently demote the other's duration to an "
            "ordinary column. Align them before combining."
        )

    _, base_dur, base_index, base_ids = windows[0]
    if column is not None and base_dur is not None and str(base_dur) != str(column):
        raise WindowContractError(
            f"window combine was asked to check column {column!r} but the first "
            f"window table declares dur_var {base_dur!r}"
        )
    for label, dur_var, index_var, id_vars in windows[1:]:
        if str(dur_var) != str(base_dur):
            _mismatch("dur_var", label, base_dur, dur_var)
        if str(index_var) != str(base_index):
            _mismatch("index_var", label, base_index, index_var)
        if tuple(id_vars or ()) != tuple(base_ids or ()):
            _mismatch("id_vars", label, base_ids, id_vars)


def strict_dur_var_units() -> bool:
    """Return True: an undeclared numeric ``dur_var`` always fails closed.

    Kept for callers that ask. The ``EASYICU_ALLOW_DUR_VAR_UNIT_GUESS`` opt-out
    that once returned False is no longer read.
    """

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


def convert_dur_var_unit(
    frame: pd.DataFrame,
    *,
    column: str,
    from_unit: str,
    to_unit: str,
) -> pd.DataFrame:
    """Rescale a numeric ``dur_var`` column from one unit to another.

    Returns a copy with the converted column and the new unit declared. A
    ``timedelta`` column is self-describing and is never rescaled — mixing it
    with a numeric column is a contract error the caller must resolve.
    """

    if from_unit == to_unit:
        return frame
    if UNIT_TIMEDELTA in {from_unit, to_unit}:
        raise DurationUnitError(
            f"cannot convert {column!r} between {from_unit!r} and {to_unit!r}: a "
            "timedelta duration and a numeric duration are different "
            "representations, not different scales"
        )
    if from_unit not in _HOURS_PER_UNIT or to_unit not in _HOURS_PER_UNIT:
        raise DurationUnitError(
            f"cannot convert {column!r} from {from_unit!r} to {to_unit!r}"
        )
    scale = _HOURS_PER_UNIT[from_unit] / _HOURS_PER_UNIT[to_unit]
    converted = frame.copy()
    if column in converted.columns:
        converted[column] = converted[column] * scale
    return set_dur_var_unit(converted, to_unit)


def combine_dur_var_units(
    frames: Sequence[pd.DataFrame],
    *,
    column: str,
    declared: Sequence[Optional[str]] = (),
) -> Optional[str]:
    """Return the single unit a combined ``dur_var`` column may carry.

    Row-binding a minutes table onto an hours table produced a column whose
    values were 60× apart with the *first* table's unit label attached — a
    silent 60× error in every downstream window, and exactly the class of bug
    the declared-unit contract exists to prevent.

    Rules:

    * every input agreeing (or exactly one declaring) → that unit;
    * numeric units disagreeing → the caller must convert, so this raises with
      both units named rather than picking one;
    * a ``timedelta`` mixed with a numeric unit → always an error;
    * nothing declared → ``None``, preserving the undeclared path;
    * some declared, some not → an error.
    """

    # Only inputs that actually carry the duration column are party to the
    # contract. A frame joined column-wise (a covariate table, a label table)
    # has no duration to declare, and counting it as "undeclared" would block
    # every ordinary merge.
    declared_units = list(declared) + [None] * max(0, len(frames) - len(declared))
    units: list = []
    for frame, declared_unit in zip(frames, declared_units):
        if not isinstance(frame, pd.DataFrame) or column not in frame.columns:
            continue
        units.append(declared_unit or get_dur_var_unit(frame))
    seen = {unit for unit in units if unit}
    if not seen:
        return None
    if len(seen) == 1:
        unit = next(iter(seen))
        undeclared = sum(1 for item in units if not item)
        if undeclared:
            message = (
                f"combining {column!r} across {len(units)} inputs where "
                f"{undeclared} declare no unit and the rest declare {unit!r}; "
                "an undeclared duration is not assumed to share the declared "
                "unit — declare it at the producer with set_dur_var_unit()"
            )
            raise DurationUnitError(message)
        return unit
    raise DurationUnitError(
        f"cannot combine {column!r} across inputs declaring different units "
        f"({', '.join(sorted(seen))}); convert them to one unit first with "
        "convert_dur_var_unit() — row-binding them would leave a column whose "
        "values differ by a scale factor under a single unit label"
    )


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


def duration_series_to_timedelta(
    values: pd.Series,
    unit: Optional[str],
    *,
    column: str = "dur_var",
    concept: Optional[str] = None,
) -> pd.Series:
    """Return a duration series as ``timedelta64`` using its declared unit.

    A ``timedelta`` series is returned unchanged. A numeric series is scaled
    by ``unit`` (seconds, minutes, hours or days); it never goes through a
    unit-less ``pd.to_timedelta``, which reads numbers as nanoseconds. Without
    a declared unit it raises :class:`DurationUnitUndeclaredError`. Values
    are not validated here: callers keep their own window rules.
    """

    if pd.api.types.is_timedelta64_dtype(values):
        return values
    label = concept or "<unknown>"
    if unit in _HOURS_PER_UNIT:
        hours = pd.to_numeric(values, errors="coerce") * _HOURS_PER_UNIT[unit]
        return pd.to_timedelta(hours, unit="h")
    if unit == UNIT_TIMEDELTA:
        raise DurationUnitError(
            f"{column!r} for concept {label!r} declares unit 'timedelta' but "
            "holds a numeric dtype; the producer must convert before the unit "
            "is consumed"
        )
    raise DurationUnitUndeclaredError(
        f"{column!r} for concept {label!r} is numeric with no declared unit "
        f"(declared: {unit!r}); a number is not a duration until its unit is "
        "declared, and EasyICU does not guess one"
    )


def resolve_dur_var_hours(
    frame: pd.DataFrame,
    *,
    column: str = "dur_var",
    concept: Optional[str] = None,
    strict: Optional[bool] = None,
) -> pd.Series:
    """Return ``frame[column]`` converted to float hours.

    The conversion is exact whenever the dtype is ``timedelta64`` or the frame
    declares a unit. Otherwise it raises :class:`DurationUnitError`: a value
    distribution cannot tell 10 minutes from 10 hours. ``strict`` is accepted
    for compatibility and has no effect.
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
    if unit in _HOURS_PER_UNIT:
        return _validate_hours(
            numeric.mul(_HOURS_PER_UNIT[unit]), column=column, concept=concept
        )
    if unit == UNIT_TIMEDELTA:
        # Declared timedelta but already coerced to a numeric dtype upstream:
        # the values are nanoseconds, which is never a plausible duration.
        raise DurationUnitError(
            f"{column!r} declares unit 'timedelta' but holds a numeric dtype; "
            "the producer must convert before the unit is consumed"
        )

    del strict  # no guess remains to switch on
    raise DurationUnitError(
        f"{column!r} has no declared unit for concept {concept or '<unknown>'!r}; "
        f"set it with set_dur_var_unit(frame, '{UNIT_MINUTES}'|'{UNIT_HOURS}') "
        "at the producer. A value distribution cannot distinguish 10 minutes "
        "from 10 hours, so EasyICU does not guess"
    )


__all__ = [
    "DUR_VAR_UNIT_ATTR",
    "UNIT_MINUTES",
    "UNIT_HOURS",
    "UNIT_TIMEDELTA",
    "VALID_DUR_VAR_UNITS",
    "STRICT_ENV_VAR",
    "DurationUnitError",
    "DurationUnitUndeclaredError",
    "WindowContractError",
    "assert_window_contract",
    "strict_dur_var_units",
    "set_dur_var_unit",
    "get_dur_var_unit",
    "clear_dur_var_unit",
    "combine_dur_var_units",
    "convert_dur_var_unit",
    "resolve_dur_var_hours",
    "duration_series_to_timedelta",
]
