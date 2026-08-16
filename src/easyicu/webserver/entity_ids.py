"""Shared ICU entity-identifier boundary.

EasyICU exports retain the source database's ICU-stay key when that is part of
the concept output contract. Patient Review, Cohort Statistics, and Cross-DB
use one canonical in-memory name (``stay_id``) so downstream review logic never
needs database-specific branches.
"""

from __future__ import annotations

from typing import Any, Iterable

CANONICAL_ENTITY_ID = "stay_id"

_ENTITY_ID_KEY_PRIORITY = (
    "stayid",
    "icustayid",
    "patientunitstayid",
    "admissionid",
    "caseid",
)


def resolve_entity_id_column(columns: Iterable[object]) -> str | None:
    """Return the source column that uniquely identifies one ICU stay."""

    by_key = {_column_key(column): str(column) for column in columns}
    for key in _ENTITY_ID_KEY_PRIORITY:
        if key in by_key:
            return by_key[key]
    return None


def normalize_entity_id(value: Any) -> str:
    """Return the comparable string form of one stored stay id.

    Prepared exports store the stay key as int64, float64 or string depending
    on the source database and the converter path, so `1`, `1.0` and `"1"` all
    have to compare equal. A whole float is rendered through `int` so it does
    not arrive as `"1.0"` or, for a large id, in scientific notation. A missing
    value becomes the empty string rather than `"nan"`, which would otherwise
    match a literal `nan` id.

    Values that are not numbers are returned as their own text, so an opaque
    key such as `"stay_7"` survives unchanged.
    """

    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return "" if value is None else str(value)


def canonicalize_entity_frame(frame: Any, source_column: str) -> Any:
    """Rename a verified source stay key to the Patient Review canonical name."""

    if frame is None or getattr(frame, "empty", True):
        return frame
    if source_column == CANONICAL_ENTITY_ID:
        return frame
    if source_column not in getattr(frame, "columns", []):
        return frame
    if CANONICAL_ENTITY_ID in getattr(frame, "columns", []):
        return frame
    return frame.rename(columns={source_column: CANONICAL_ENTITY_ID})


def _column_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())
