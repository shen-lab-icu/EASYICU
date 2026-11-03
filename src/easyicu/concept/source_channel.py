"""Per-hour source-channel preference for concepts with ranked measurement channels.

Some measurements are recorded through physically different channels that a
clinician does not average.  Mean arterial pressure is the reference case: an
arterial line and an oscillometric cuff are both "MAP", but when an arterial
line is in place its value is the one used, and the cuff value of the same
hour is a cross-check rather than a second sample of the same quantity
(2026-11-03 extraction audit #16: pooling both in one hourly median shifted
MIMIC-III MAP by >= 3 mmHg in 203,649 hours).

A concept opts in with a concept-level ``source_channel_priority`` list and a
``source_channel`` on every source of a database, for example::

    "source_channel_priority": ["invasive", "nibp"],
    "sources": {"miiv": [
        {"ids": [220052, 225312], ..., "source_channel": "invasive"},
        {"ids": [220181], ..., "source_channel": "nibp"}]}

Within each patient and clock hour since ICU admission, only rows of the
best-ranked channel that has a usable (non-missing, in-bounds) value are kept;
the usual aggregation then runs on those rows.  The rule always uses one-hour
buckets, whatever interval the caller requests, so a native-resolution load
and an hourly load make the same selection and coarser intervals aggregate
already-selected hours.

A database whose sources declare no channel at all is loaded unchanged (a
community database with one undifferentiated MAP column, for example).
Declaring a channel on some sources but not on others, or a channel that is
not in the priority list, is a dictionary contract error.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Source parameter naming a source's measurement channel.
SOURCE_CHANNEL_PARAM = "source_channel"
#: Private per-row column carrying the channel rank between concat and selection.
RANK_COLUMN = "__source_channel_rank"
#: The selection bucket, in hours since ICU admission.
BUCKET_HOURS = 1.0


class SourceChannelContractError(ValueError):
    """A concept's channel declarations cannot be evaluated."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def channel_priority(definition: object) -> Tuple[str, ...]:
    """Return the concept's ranked channels (best first), or ``()``."""

    raw = getattr(definition, "source_channel_priority", None)
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    channels = tuple(str(item) for item in raw)
    if len(set(channels)) != len(channels):
        raise SourceChannelContractError(
            "source_channel_priority_duplicate", repr(channels)
        )
    return channels


def source_channel(source: object) -> Optional[str]:
    params = getattr(source, "params", None) or {}
    value = params.get(SOURCE_CHANNEL_PARAM)
    return None if value is None else str(value)


def validate_sources(definition: object, sources: Sequence[object]) -> bool:
    """Check one database's sources; return whether selection applies."""

    priority = channel_priority(definition)
    declared = [source_channel(source) for source in sources]
    if not priority:
        if any(channel is not None for channel in declared):
            raise SourceChannelContractError(
                "source_channel_without_priority",
                str(getattr(definition, "name", "")),
            )
        return False
    if all(channel is None for channel in declared):
        return False
    if any(channel is None for channel in declared):
        raise SourceChannelContractError(
            "source_channel_undeclared",
            f"{getattr(definition, 'name', '')}: every source of a database "
            "with ranked channels must declare one",
        )
    unknown = sorted({c for c in declared if c not in priority})
    if unknown:
        raise SourceChannelContractError(
            "source_channel_unknown",
            f"{getattr(definition, 'name', '')}: {unknown} not in {list(priority)}",
        )
    return True


def one_source_per_channel(sources: Sequence[object]) -> bool:
    """Whether every declared channel is exactly one source.

    Then a per-source hourly aggregate is the channel's hourly value, so the
    loader may pre-aggregate each source (for example in DuckDB) before the
    selection without forming a cross-channel median.
    """

    channels = [source_channel(source) for source in sources]
    return None not in channels and len(set(channels)) == len(channels)


def stamp_rank(frame: pd.DataFrame, definition: object, source: object) -> pd.DataFrame:
    """Attach the source's channel rank to every row of its frame."""

    channel = source_channel(source)
    if channel is None:
        return frame
    rank = channel_priority(definition).index(channel)
    return frame.assign(**{RANK_COLUMN: np.int16(rank)})


def _hour_bucket(times: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_timedelta64_dtype(times):
        return np.floor(times.dt.total_seconds() / 3600.0 / BUCKET_HOURS)
    if pd.api.types.is_datetime64_any_dtype(times):
        return times.dt.floor(f"{int(BUCKET_HOURS * 60)}min")
    if pd.api.types.is_numeric_dtype(times):
        return np.floor(pd.to_numeric(times, errors="coerce") / BUCKET_HOURS)
    return None


def select_preferred_channel(
    frame: pd.DataFrame,
    *,
    value_column: str,
    id_columns: Iterable[str],
    time_column: Optional[str],
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    time_in_hours: bool = True,
) -> pd.DataFrame:
    """Keep, per patient and hour, only the best channel with a usable value.

    ``time_in_hours`` states that a numeric ``time_column`` is hours since
    admission (the aligned case).  For an unaligned numeric clock, whose unit
    is database-specific, the channel cannot be bucketed and a contract error
    is raised rather than silently returning pooled channels.
    """

    if RANK_COLUMN not in frame.columns:
        return frame
    if frame.empty:
        return frame.drop(columns=[RANK_COLUMN])
    if not time_column or time_column not in frame.columns:
        raise SourceChannelContractError("source_channel_time_missing", str(time_column))
    times = frame[time_column]
    if pd.api.types.is_numeric_dtype(times) and not time_in_hours:
        raise SourceChannelContractError(
            "source_channel_unaligned_numeric_time",
            "channel preference needs admission-aligned hours",
        )
    bucket = _hour_bucket(times)
    if bucket is None:
        raise SourceChannelContractError("source_channel_time_type", str(times.dtype))
    keys = [col for col in id_columns if col in frame.columns]
    if not keys:
        raise SourceChannelContractError("source_channel_id_missing", repr(list(id_columns)))

    values = pd.to_numeric(frame[value_column], errors="coerce")
    usable = values.notna()
    if minimum is not None:
        usable &= values >= minimum
    if maximum is not None:
        usable &= values <= maximum
    rank = frame[RANK_COLUMN].astype("float64")
    work = frame[keys].copy()
    work["__bucket"] = bucket.to_numpy()
    work["__usable_rank"] = rank.where(usable.to_numpy())
    best = work.groupby(keys + ["__bucket"], dropna=False)["__usable_rank"].transform("min")
    # Rows of a bucket without any usable value keep their rank-free fate to
    # the caller's bounds filter; otherwise only the best usable channel stays.
    keep = best.isna().to_numpy() | (rank.to_numpy() == best.to_numpy())
    return frame.loc[keep].drop(columns=[RANK_COLUMN])


def drop_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the private rank column from a frame that skipped selection."""

    if isinstance(frame, pd.DataFrame) and RANK_COLUMN in frame.columns:
        return frame.drop(columns=[RANK_COLUMN])
    return frame


__all__ = [
    "BUCKET_HOURS",
    "RANK_COLUMN",
    "SOURCE_CHANNEL_PARAM",
    "SourceChannelContractError",
    "channel_priority",
    "drop_rank",
    "one_source_per_channel",
    "select_preferred_channel",
    "source_channel",
    "stamp_rank",
    "validate_sources",
]
