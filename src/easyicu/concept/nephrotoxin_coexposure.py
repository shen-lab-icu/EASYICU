"""Vancomycin plus piperacillin-tazobactam co-exposure (S732 B3).

Vancomycin with piperacillin-tazobactam is the nephrotoxic antibiotic pair of
interest. Concomitant exposure is defined on administration hours: an hour is
flagged when one agent is given in that hour and the other agent was given in
the same hour or in the preceding ``WINDOW_HOURS - 1`` hours. The flag is
therefore emitted only at administration hours, never after the last dose, and
a stay has any flagged hour exactly when two administrations of the different
agents lie less than ``WINDOW_HOURS`` hours apart.

Inputs are the hourly Boolean exposure concepts ``vancomycin_iv`` and
``piperacillin_tazobactam``; only rows whose value is TRUE are administrations.
Concept times are relative to ICU admission: numeric hours (the axis concept
callbacks receive, see ``callbacks._ensure_time_column_type``) or timedeltas.
Absolute datetimes or other types are refused rather than guessed. The output
uses the input time type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CALLBACK = "vpt_coexposure"
VANCOMYCIN = "vancomycin_iv"
PIPTAZO = "piperacillin_tazobactam"
#: Co-administration window (hours). Both agents within 24 h of each other.
WINDOW_HOURS = 24


class CoexposureDerivationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _events(table, ids, index):
    """TRUE administration hours of one agent as (ids..., _h) rows."""
    frame = table.data
    if frame.empty:
        return pd.DataFrame({**{c: pd.Series(dtype=frame[c].dtype) for c in ids}, "_h": pd.Series(dtype=float)})
    value = table.value_column
    given = frame[value].eq(True).fillna(False).to_numpy(dtype=bool)
    out = frame.loc[given, ids].copy()
    out["_h"] = _hours(frame[index])[given]
    return out.dropna(subset=["_h"]).drop_duplicates()


def _hours(times):
    if pd.api.types.is_timedelta64_dtype(times):
        return (times.dt.total_seconds() / 3600.0).to_numpy(dtype=float)
    if pd.api.types.is_numeric_dtype(times) and not pd.api.types.is_bool_dtype(times):
        return times.to_numpy(dtype=float)
    raise CoexposureDerivationError("coexposure_time_unit_unavailable", "ICU-relative hours or timedeltas required")


def _flag(left, right, ids):
    """Rows of ``left`` with an event of ``right`` in (h - WINDOW, h]."""
    if left.empty or right.empty:
        return left.iloc[0:0]
    left = left.sort_values("_h", kind="mergesort")
    prior = right.rename(columns={"_h": "_prior"}).sort_values("_prior", kind="mergesort")
    matched = pd.merge_asof(left, prior, left_on="_h", right_on="_prior", by=ids, direction="backward")
    gap = matched["_h"] - matched["_prior"]
    keep = np.isfinite(gap.to_numpy(dtype=float)) & (gap.to_numpy(dtype=float) < WINDOW_HOURS)
    return matched.loc[keep, ids + ["_h"]]


def coexposure(vanco: pd.DataFrame, piptazo: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Co-exposure hours from two event frames with columns ``ids + ['_h']``."""
    rows = pd.concat([_flag(vanco, piptazo, ids), _flag(piptazo, vanco, ids)], ignore_index=True)
    return rows.drop_duplicates().sort_values(ids + ["_h"], kind="mergesort").reset_index(drop=True)


def vpt_coexposure_callback(tables, ctx):
    """Concept callback: TRUE at hours of vancomycin/pip-tazo co-exposure."""
    from ..table import ICUTable

    vanco, piptazo = tables.get(VANCOMYCIN), tables.get(PIPTAZO)
    if vanco is None or piptazo is None:
        raise CoexposureDerivationError("coexposure_input_unavailable", f"{VANCOMYCIN} and {PIPTAZO} required")
    ids, index = list(vanco.id_columns), vanco.index_column
    if list(piptazo.id_columns) != ids or not index or piptazo.index_column != index:
        raise CoexposureDerivationError("concept_identity_mismatch", "same id and time columns required")
    rows = coexposure(_events(vanco, ids, index), _events(piptazo, ids, index), ids)
    out = rows[ids].copy()
    hours = rows["_h"].to_numpy(dtype=float)
    timedelta = any(pd.api.types.is_timedelta64_dtype(t.data[index]) for t in (vanco, piptazo) if not t.data.empty)
    out[index] = pd.to_timedelta(hours, unit="h") if timedelta else hours
    out[ctx.concept_name] = True
    return ICUTable(out, id_columns=ids, index_column=index, value_column=ctx.concept_name)


__all__ = ["CALLBACK", "WINDOW_HOURS", "CoexposureDerivationError", "coexposure", "vpt_coexposure_callback"]
