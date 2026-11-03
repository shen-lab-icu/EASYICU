"""Time placement of MIMIC-III ``prescriptions`` rows (``mimic_abx_presc``).

MIMIC-III prescriptions carry only a calendar date (``startdate`` is always
00:00).  ricu's ``mimic_abx_presc`` therefore moves each order to the middle of
its day, ``startdate + mins(720L)``, before the row is placed on the ICU clock.

The shift is twelve hours whatever representation the index has when the
callback runs.  The loader hands this callback absolute ``datetime64`` values;
adding the bare number 720 to their integer form moved the order by 720
microseconds, kept it at midnight, and turned the column into an epoch integer
that only a later guard converted back.  ICU-relative numeric indexes in
EasyICU are hours.
"""

from __future__ import annotations

import pandas as pd

PRESCRIPTION_DAY_OFFSET_MINUTES = 720


def shift_index_minutes(values: pd.Series, minutes: int = PRESCRIPTION_DAY_OFFSET_MINUTES) -> pd.Series:
    """Return ``values`` moved forward by ``minutes``, keeping their type.

    * ``datetime64`` (naive or tz-aware) and ``timedelta64`` move by a
      :class:`pandas.Timedelta`;
    * numeric values are ICU-relative hours and move by ``minutes / 60``;
    * object columns are parsed as datetimes; values that do not parse raise
      instead of being silently reinterpreted as numbers.
    """
    offset = pd.Timedelta(minutes=minutes)
    if pd.api.types.is_datetime64_any_dtype(values) or pd.api.types.is_timedelta64_dtype(values):
        return values + offset
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return values + minutes / 60.0
    parsed = pd.to_datetime(values, errors="coerce")
    unparsed = values.notna() & parsed.isna()
    if bool(unparsed.any()):
        raise ValueError(
            "mimic_abx_presc: prescription index values are neither datetimes nor "
            f"ICU-relative hours ({int(unparsed.sum())} unparseable values)"
        )
    return parsed + offset
