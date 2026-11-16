"""Declared alternative Sepsis-3 definition options.

The canonical ``sep3`` / ``susp_inf`` concepts follow R ricu.  Published
Sepsis-3 variants differ from it in three places, each declared here as an
explicit, validated keyword whose default reproduces the canonical result
exactly (the default path never calls into this module):

``baseline_sofa`` (``sep3``)
    ``"window"`` (default): the SOFA rise is measured against the baseline
    chosen by ``delta_fun`` inside the suspected-infection window.
    ``"zero"``: the baseline SOFA is taken as 0, so the criterion is an
    absolute SOFA >= ``sofa_thresh`` inside the window (MIT-LCP mimic-code
    ``mimic-iv/concepts/sepsis/sepsis3.sql``: ``sofa_24hours >= 2``).

``onset_scope`` (``sep3``)
    ``"all"`` (default): every suspected-infection event and SOFA hour of the
    loaded stay can define the onset.
    ``"icu"``: only suspected-infection events and onsets inside the ICU stay,
    the closed interval [ICU admission, ICU discharge] in hours since ICU
    admission, count.

``abx_min_days`` / ``abx_max_gap`` (``susp_inf``)
    Off by default.  When set, an antibiotic administration counts toward
    suspected infection only if antibiotics continue for ``abx_min_days``
    days from it (YAIB ``abx_cont``; see :func:`continuous_antibiotic_anchors`).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

BASELINE_SOFA_CHOICES = ("window", "zero")
ONSET_SCOPE_CHOICES = ("all", "icu")
ABX_CONTINUITY_MODES = ("and", "or", "abx")

_SCOPE_LWR = "_scope_lwr"
_SCOPE_UPR = "_scope_upr"


def declared_choice(name: str, value: object, choices: Sequence[str]) -> str:
    """Validate a declared option; ``None`` selects the default (first) value."""
    if value is None:
        return choices[0]
    text = str(value).strip()
    if text not in choices:
        raise ValueError(
            f"{name}={value!r} is not a declared value; allowed: {sorted(choices)}"
        )
    return text


def declared_min_days(value: object) -> Optional[int]:
    """``abx_min_days``: ``None`` (off) or a positive whole number of days."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"abx_min_days={value!r} must be a positive integer or None")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"abx_min_days={value!r} must be a positive integer or None"
        ) from None
    if not number.is_integer() or number < 1:
        raise ValueError(f"abx_min_days={value!r} must be a positive integer or None")
    return int(number)


def delta_zero_baseline(x: pd.Series) -> pd.Series:
    """SOFA change against a baseline of 0: the SOFA value itself (NA stays NA)."""
    return x - 0


def _hours(values: pd.Series) -> np.ndarray:
    """Time values as float hours (numeric hours pass through)."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    stamps = pd.to_datetime(values, errors="coerce")
    out = stamps.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / 3.6e12
    out[stamps.isna().to_numpy()] = np.nan
    return out


def truthy(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(
        values.map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v),
        errors="coerce",
    )
    return numeric.gt(0).fillna(False).to_numpy(dtype=bool)


def continuous_antibiotic_anchors(
    abx: pd.DataFrame,
    id_cols: Sequence[str],
    index_col: str,
    *,
    min_days: int,
    max_gap: pd.Timedelta = pd.Timedelta(hours=24),
    death: Optional[pd.DataFrame] = None,
    value_col: str = "abx",
) -> pd.DataFrame:
    """Keep antibiotic rows that start a continuous course (YAIB ``abx_cont``).

    YAIB (van de Water et al., ICLR 2024, App. D.3; YAIB-cohorts
    ``ricu-extensions/callbacks/callback-sepsis.R::abx_cont``) counts an
    antibiotic at time ``t`` when, over the window [t, t + min_days x 24 h]
    (ended earlier by death), no gap between consecutive antibiotic
    administrations, nor between the last one and the window end, exceeds
    ``max_gap`` (24 h).  Antibiotics recorded after death are ignored.

    Days are rolling 24-hour blocks from each administration, not calendar
    days.  EasyICU's harmonized ``abx`` concept carries administration times
    without durations, so each administration is a point (YAIB's
    ``abx_duration`` coverage is taken as zero).

    ``death`` (optional) holds ``id_cols`` + ``index_col`` rows giving each
    stay's death time on the same time axis; stays without a row are alive.
    """
    id_cols = list(id_cols)
    if abx.empty:
        return abx
    span = float(min_days) * 24.0
    gap = max_gap.total_seconds() / 3600.0
    present = truthy(abx[value_col]) if value_col in abx.columns else np.ones(len(abx), bool)
    times = _hours(abx[index_col])
    rows = abx.loc[present & ~np.isnan(times), id_cols].copy()
    rows["_t"] = times[present & ~np.isnan(times)]

    if death is not None and not death.empty:
        dead = death[id_cols].copy()
        dead["_death"] = _hours(death[index_col])
        dead = dead.dropna(subset=["_death"]).groupby(id_cols, as_index=False)["_death"].min()
        rows = rows.merge(dead, on=id_cols, how="left")
    else:
        rows["_death"] = np.nan
    rows = rows[rows["_death"].isna() | (rows["_t"] <= rows["_death"])]

    keep = []
    for key, group in rows.groupby(id_cols, sort=False):
        t = np.unique(group["_t"].to_numpy(dtype=float))
        died = group["_death"].iloc[0]
        diffs = np.diff(t)
        last = np.searchsorted(t, t + span, side="right") - 1
        for i, start in enumerate(t):
            k = last[i]
            end = start + span if np.isnan(died) else min(died, start + span)
            gaps = np.append(diffs[i:k], end - t[k])
            if gaps.max() <= gap:
                keep.append((*(key if isinstance(key, tuple) else (key,)), start))
    anchors = pd.DataFrame(keep, columns=id_cols + ["_t"])
    slim = abx[id_cols].copy()
    slim["_t"] = times
    slim["_pos"] = np.arange(len(abx))
    hit = slim.merge(anchors, on=id_cols + ["_t"], how="inner")["_pos"].to_numpy()
    mask = np.zeros(len(abx), dtype=bool)
    mask[hit] = True
    return abx[mask & present]


def icu_stay_bounds(
    los: pd.DataFrame, id_cols: Iterable[str], los_col: str = "los_icu",
) -> pd.DataFrame:
    """[0, ICU length of stay] in hours since ICU admission, per stay.

    ``los`` is the ``los_icu`` concept (days).  A stay without a finite,
    non-negative length of stay gets no bounds, so nothing of it is in scope.
    """
    id_cols = list(id_cols)
    hours = pd.to_numeric(los[los_col], errors="coerce") * 24.0
    bounds = los[id_cols].copy()
    bounds[_SCOPE_LWR] = 0.0
    bounds[_SCOPE_UPR] = hours.to_numpy(dtype=float)
    bounds = bounds[np.isfinite(bounds[_SCOPE_UPR]) & (bounds[_SCOPE_UPR] >= 0)]
    return bounds.groupby(id_cols, as_index=False).agg(
        {_SCOPE_LWR: "min", _SCOPE_UPR: "max"}
    )


def within_bounds(
    frame: pd.DataFrame, id_cols: Sequence[str], time_col: str, bounds: pd.DataFrame,
) -> np.ndarray:
    """Row mask: ``time_col`` lies in the stay's closed [lower, upper] bounds."""
    id_cols = list(id_cols)
    joined = frame[id_cols].merge(bounds, on=id_cols, how="left")
    t = _hours(frame[time_col])
    lwr = joined[_SCOPE_LWR].to_numpy(dtype=float)
    upr = joined[_SCOPE_UPR].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return (t >= lwr) & (t <= upr)
