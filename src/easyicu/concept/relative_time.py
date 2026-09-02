"""Coercion of a mixed number/timestamp time column to relative ICU hours.

Why this is its own module: the concept resolver concatenates per-table frames
that do not agree on how time is spelled. DuckDB-backed reads return relative
hours as ``float64``; the non-DuckDB path returns absolute ``datetime64``.
``pd.concat`` of the two yields ``object``, and everything downstream — window
alignment, ``change_interval``, aggregation — assumes a single numeric scale.

Unifying them means anchoring the absolute timestamps against each stay's
``intime``. That anchor can be absent: the frame may not carry ``intime`` and
the ``icustays`` table may be unreadable. The tempting behaviour is to leave
those rows as NaN and continue, which is what this code used to do.

That is the one thing it must not do. A NaN in a time column is
indistinguishable downstream from "this patient has no measurement at this
point". Coercing a recorded, readable timestamp to NaN therefore reports a
missing *anchor table* as missing *clinical data* — silently, for every
affected row, in a value that then flows into windows, scores and the
manuscript. So the anchor lookup is fail-closed: recover the anchor, or refuse
the concept with ``ConceptExtractionUnavailable``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .errors import ConceptExtractionUnavailable


__all__ = ["coerce_mixed_time_column"]


def _naive(series: pd.Series) -> pd.Series:
    """Drop a timezone if one is attached; subtraction needs both sides naive."""
    if hasattr(series.dt, "tz") and series.dt.tz is not None:
        return series.dt.tz_localize(None)
    return series


def coerce_mixed_time_column(
    combined: pd.DataFrame,
    key: str,
    *,
    concept_id: str,
    database: str,
    id_columns: Optional[List[str]] = None,
    data_source: Any = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return ``combined`` and the ``key`` column unified as relative hours.

    ``combined`` is returned alongside the values because resolving the anchor
    may merge ``intime`` into it; callers must keep the returned frame.

    Raises:
        ConceptExtractionUnavailable: a non-null value in ``key`` could not be
            converted, so writing the result back would silently turn recorded
            observations into missing measurements.
    """
    original = combined[key]
    numeric_vals = pd.to_numeric(original, errors="coerce")
    dt_mask = numeric_vals.isna() & original.notna()
    if not dt_mask.any():
        return combined, numeric_vals

    dt_vals = pd.to_datetime(original[dt_mask], errors="coerce")
    anchor_failure: Optional[str] = None

    if dt_vals.isna().any():
        anchor_failure = (
            f"{int(dt_vals.isna().sum())} of {int(dt_mask.sum())} value(s) in "
            f"column {key!r} are neither a number nor a parseable timestamp"
        )
    else:
        intime_col: Optional[pd.Series] = None
        if "intime" in combined.columns:
            intime_col = pd.to_datetime(combined.loc[dt_mask, "intime"], errors="coerce")
        elif data_source is not None and id_columns:
            icu_df = None
            try:
                icu_tbl = data_source.load_table(
                    "icustays", columns=[id_columns[0], "intime"], verbose=False
                )
                icu_df = icu_tbl.data if hasattr(icu_tbl, "data") else icu_tbl
                if pd.api.types.is_datetime64_any_dtype(icu_df["intime"]):
                    icu_df["intime"] = _naive(icu_df["intime"])
            except Exception as exc:  # noqa: BLE001 - re-raised as a typed failure below
                anchor_failure = (
                    f"the icustays table that anchors {key!r} to relative hours "
                    f"could not be read ({exc})"
                )
            if icu_df is not None:
                merged = combined.merge(
                    icu_df[[id_columns[0], "intime"]], on=id_columns[0], how="left"
                )
                if len(merged) != len(combined):
                    # ``dt_mask`` selects rows positionally, so a fan-out would
                    # pair timestamps with the wrong stay's anchor and produce
                    # plausible-looking hours instead of an error.
                    raise ConceptExtractionUnavailable(
                        concept_id=concept_id,
                        database=database or "unknown",
                        stage="relative_time_anchor",
                        detail=(
                            "the icustays table lists more than one 'intime' "
                            f"for some {id_columns[0]!r}, so the anchor for "
                            f"{key!r} is ambiguous"
                        ),
                    )
                combined = merged
                intime_col = pd.to_datetime(
                    combined.loc[dt_mask, "intime"], errors="coerce"
                )

        if anchor_failure is not None:
            pass
        elif intime_col is None:
            anchor_failure = (
                f"no 'intime' anchor is available to convert {key!r} to "
                "relative hours"
            )
        elif intime_col.isna().any():
            anchor_failure = (
                f"{int(intime_col.isna().sum())} of {int(dt_mask.sum())} row(s) "
                f"needing {key!r} converted have no usable 'intime' anchor"
            )
        else:
            rel_hours = np.floor(
                (_naive(dt_vals) - _naive(intime_col)).dt.total_seconds() / 3600.0
            )
            numeric_vals.loc[dt_mask] = rel_hours

    if anchor_failure is not None:
        raise ConceptExtractionUnavailable(
            concept_id=concept_id,
            database=database or "unknown",
            stage="relative_time_anchor",
            detail=(
                f"{anchor_failure}. Coercing them to NaN would report recorded "
                "observations as missing measurements"
            ),
        )
    return combined, numeric_vals
