"""Fractional excretion of a solute from a spot urine sample.

FE_x (%) = 100 * (U_x * P_cr) / (P_x * U_cr)

The urine solute and urine creatinine must come from the same time bucket
(the same spot sample).  Plasma values are the most recent observation in the
same or an earlier bucket, no older than ``lookback``.  A plasma value charted
after the urine bucket is never used, and a urine pair with no plasma value in
the lookback has no result rather than an imputed one.

The units cancel only when each urine/plasma pair shares a unit: sodium
mmol/L (mEq/L) against mmol/L, urea nitrogen mg/dL against BUN mg/dL, and
creatinine mg/dL against mg/dL.  The concept dictionary normalises every
source to those canonical units before this owner sees the values.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np
import pandas as pd

from ..utils.thresholds import ge, le

logger = logging.getLogger(__name__)

DEFAULT_PLASMA_LOOKBACK = pd.Timedelta(hours=24)
RECEIPT_ATTR = "fractional_excretion"


@dataclass(frozen=True)
class FractionalExcretionSpec:
    """Column roles for one fractional-excretion concept."""

    value_name: str
    urine_solute: str
    plasma_solute: str
    urine_creatinine: str = "urine_crea"
    plasma_creatinine: str = "crea"
    lower: float = 0.0
    upper: float = 100.0

    @property
    def inputs(self) -> tuple[str, str, str, str]:
        return (
            self.urine_solute,
            self.urine_creatinine,
            self.plasma_solute,
            self.plasma_creatinine,
        )


FENA = FractionalExcretionSpec(value_name="fena", urine_solute="urine_na", plasma_solute="na")
FEUREA = FractionalExcretionSpec(value_name="feurea", urine_solute="urine_urea", plasma_solute="bun")


def _hours(values: pd.Series) -> np.ndarray:
    """Return a time column as float hours (numeric hours, timedelta or datetime)."""

    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    if pd.api.types.is_timedelta64_dtype(values):
        return values.dt.total_seconds().to_numpy(dtype="float64") / 3600.0
    stamps = pd.to_datetime(values, errors="coerce")
    origin = pd.Timestamp("2000-01-01", tz=stamps.dt.tz)
    return (stamps - origin).dt.total_seconds().to_numpy(dtype="float64") / 3600.0


def _last_observation(
    frame: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    hours: np.ndarray,
    column: str,
    lookback_hours: float,
) -> pd.Series:
    """Most recent non-missing ``column`` at or before each row, within the lookback."""

    values = pd.to_numeric(frame[column], errors="coerce")
    seen_at = pd.Series(np.where(values.notna(), hours, np.nan), index=frame.index)
    keys = [frame[c] for c in id_columns]
    carried = values.groupby(keys, sort=False, dropna=False).ffill()
    carried_at = seen_at.groupby(keys, sort=False, dropna=False).ffill()
    age = hours - carried_at.to_numpy(dtype="float64")
    fresh = np.isfinite(age) & (age >= 0) & (age <= lookback_hours)
    return carried.where(fresh)


def fractional_excretion(
    frame: pd.DataFrame,
    *,
    spec: FractionalExcretionSpec,
    id_columns: Sequence[str],
    index_column: str,
    lookback: pd.Timedelta = DEFAULT_PLASMA_LOOKBACK,
) -> pd.DataFrame:
    """Compute one fractional excretion from an outer-merged frame.

    ``frame`` holds one row per (id, time bucket) with the four input columns
    named by ``spec``.  The result keeps ``id_columns``, ``index_column`` and
    ``spec.value_name`` for urine pairs whose value is defined and inside the
    inclusive bounds.  ``result.attrs[RECEIPT_ATTR]`` counts why pairs were
    dropped.
    """

    id_columns = list(id_columns)
    keep = id_columns + [index_column]
    missing = [c for c in spec.inputs if c not in frame.columns]
    if missing or frame.empty:
        out = pd.DataFrame(columns=keep + [spec.value_name])
        out[spec.value_name] = out[spec.value_name].astype("float64")
        out.attrs[RECEIPT_ATTR] = {"urine_pairs": 0, "missing_inputs": missing}
        return out

    lookback_hours = pd.Timedelta(lookback).total_seconds() / 3600.0
    ordered = frame.sort_values(keep, kind="mergesort").reset_index(drop=True)
    hours = _hours(ordered[index_column])
    plasma = {
        name: _last_observation(
            ordered, id_columns=id_columns, hours=hours, column=name, lookback_hours=lookback_hours
        )
        for name in (spec.plasma_solute, spec.plasma_creatinine)
    }
    urine_x = pd.to_numeric(ordered[spec.urine_solute], errors="coerce")
    urine_cr = pd.to_numeric(ordered[spec.urine_creatinine], errors="coerce")
    plasma_x = plasma[spec.plasma_solute]
    plasma_cr = plasma[spec.plasma_creatinine]

    pair = urine_x.notna() & urine_cr.notna()
    has_plasma = plasma_x.notna() & plasma_cr.notna()
    denominator = (plasma_x * urine_cr).where(lambda d: d > 0)
    value = 100.0 * urine_x * plasma_cr / denominator
    value = value.where(pair & has_plasma).replace([np.inf, -np.inf], np.nan)
    in_bounds = ge(value, spec.lower) & le(value, spec.upper)

    out = ordered.loc[value.notna() & in_bounds, keep].copy()
    out[spec.value_name] = value[out.index].astype("float64")
    receipt = {
        "urine_pairs": int(pair.sum()),
        "pairs_without_plasma_in_lookback": int((pair & ~has_plasma).sum()),
        "pairs_with_nonpositive_denominator": int((pair & has_plasma & denominator.isna()).sum()),
        "values_outside_bounds": int((value.notna() & ~in_bounds).sum()),
        "values": int(len(out)),
        "lookback_hours": lookback_hours,
    }
    out = out.reset_index(drop=True)
    out.attrs[RECEIPT_ATTR] = receipt
    logger.info("%s fractional excretion receipt: %s", spec.value_name, receipt)
    return out
