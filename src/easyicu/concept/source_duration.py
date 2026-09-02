"""Source-duration semantics owned outside the monolithic concept resolver."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "declare_dur_var_hours",
    "drop_negative_source_end_durations",
    "normalize_source_dur_var_hours",
    "source_duration_is_end",
]


def declare_dur_var_hours(frame: Any) -> None:
    """Record that ``frame``'s ``dur_var`` is now expressed in hours."""

    from ..table.duration import UNIT_HOURS, set_dur_var_unit

    set_dur_var_unit(frame, UNIT_HOURS)


def normalize_source_dur_var_hours(
    frame: Any,
    *,
    concept_name: str,
    source_frame: Any = None,
) -> Any:
    """Normalize one source frame before binding multiple concept sources."""

    if "dur_var" not in frame.columns:
        return frame

    from ..table.duration import (
        UNIT_HOURS,
        get_dur_var_unit,
        resolve_dur_var_hours,
        set_dur_var_unit,
    )

    declared = get_dur_var_unit(frame) or get_dur_var_unit(source_frame)
    if declared:
        set_dur_var_unit(frame, declared)
    if pd.api.types.is_numeric_dtype(
        frame["dur_var"]
    ) or pd.api.types.is_timedelta64_dtype(frame["dur_var"]):
        frame["dur_var"] = resolve_dur_var_hours(frame, concept=concept_name)
        set_dur_var_unit(frame, UNIT_HOURS)
    return frame


def drop_negative_source_end_durations(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    source_table: str,
    column: str = "dur_var",
) -> pd.DataFrame:
    """Quarantine raw end-before-start rows before strict duration validation."""

    if column not in frame.columns or frame.empty:
        return frame
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = values < 0
    count = int(invalid.sum())
    if not count:
        return frame
    logger.warning(
        "dropping %d raw end-before-start row(s) for concept %r from table %r",
        count,
        concept_name,
        source_table,
    )
    return frame.loc[~invalid].copy()


def source_duration_is_end(source: Any) -> bool:
    """Return whether the schema declares a source duration column as an end."""

    params = getattr(source, "params", None) or {}
    explicit = params.get("dur_is_end")
    if isinstance(explicit, str):
        normalized = explicit.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    elif explicit is not None:
        return bool(explicit)

    name = re.sub(r"[^a-z0-9]+", "", str(getattr(source, "dur_var", "")).lower())
    return "end" in name or "stop" in name
