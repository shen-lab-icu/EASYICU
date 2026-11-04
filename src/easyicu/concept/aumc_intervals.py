"""AUMC half-open intervals, anchored to the exact ICU admission clock.

Producers and the central resolver exchange absolute minutes. Quantization
happens only after subtraction of admittedat; output returns to that clock
so the resolver still performs admission alignment exactly once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def admission_origins(frame, data_source):
    if "admittedat" in frame:
        origins = frame[["admissionid", "admittedat"]]
    elif data_source is not None:
        table = data_source.load_table(
            "admissions", columns=["admissionid", "admittedat"], verbose=False,
        )
        origins = table.data if hasattr(table, "data") else table
    else:
        # Direct callback callers may supply an already-relative minute clock.
        origins = frame[["admissionid"]].drop_duplicates().assign(admittedat=0.0)
    origins = origins[["admissionid", "admittedat"]].drop_duplicates()
    if origins["admissionid"].duplicated().any():
        raise ValueError("AUMC intervals require one admission origin per stay")
    return origins


def expand_intervals(frame, *, start_col, stop_col, value_col, data_source=None,
                     union=False, allocate=False):
    """Occupancy, or overlap-weighted interval totals, in ICU hours [h,h+1)."""
    columns = ["admissionid", start_col, value_col]
    units = [c for c in ("doseunit", "unit", "doserateunit") if c in frame]
    columns += [c for c in units if c not in columns]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    origins = admission_origins(out, data_source).set_index("admissionid")["admittedat"]
    origin = pd.to_numeric(out.admissionid.map(origins), errors="coerce")
    if origin.isna().any():
        raise ValueError("AUMC interval has no usable admission origin")
    # Subtract before division to preserve exact boundaries on large clocks.
    out["__start"] = (pd.to_numeric(out[start_col], errors="coerce") - origin) / 60.0
    out["__stop"] = (pd.to_numeric(out[stop_col], errors="coerce") - origin) / 60.0
    valid = np.isfinite(out["__start"]) & np.isfinite(out["__stop"])
    valid &= out["__stop"].gt(out["__start"]) & out[value_col].notna()
    out = out.loc[valid].copy()
    if union and not out.empty:
        keys = ["admissionid", value_col, *units]
        out = out.sort_values(keys + ["__start", "__stop"], kind="stable")
        running = out.groupby(keys, dropna=False)["__stop"].cummax()
        previous = running.groupby([out[k] for k in keys], dropna=False).shift()
        new = previous.isna() | out["__start"].gt(previous)
        out["__episode"] = new.groupby([out[k] for k in keys], dropna=False).cumsum()
        out = out.groupby(keys + ["__episode"], as_index=False, dropna=False).agg(
            __start=("__start", "min"), __stop=("__stop", "max"),
        )
    pieces = []
    # Bounded source chunks keep long synthetic or clinical batches from
    # multiplying every original table column in memory.
    for lo in range(0, len(out), 20_000):
        chunk = out.iloc[lo:lo + 20_000]
        start = chunk["__start"].to_numpy(dtype=float)
        stop = chunk["__stop"].to_numpy(dtype=float)
        first = np.floor(start).astype(np.int64)
        counts = (np.ceil(stop).astype(np.int64) - first).clip(min=0)
        rows = np.repeat(np.arange(len(chunk)), counts)
        offsets = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
        hours = first[rows] + offsets
        piece = chunk.iloc[rows][["admissionid", value_col, *units]].reset_index(drop=True)
        if allocate:
            overlap = np.minimum(stop[rows], hours + 1) - np.maximum(start[rows], hours)
            piece[value_col] = piece[value_col].to_numpy(dtype=float) * overlap / (stop[rows] - start[rows])
        piece[start_col] = hours * 60.0 + piece.admissionid.map(origins).to_numpy(dtype=float)
        pieces.append(piece)
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
    if allocate:
        result = result.groupby(["admissionid", start_col], as_index=False)[value_col].sum(min_count=1)
    elif union:
        result = result.drop_duplicates(["admissionid", start_col, value_col])
    return result


def expand_source_windows(frame, source, concept_name, index_col, data_source, interval):
    """Expand explicit AUMC windows before the generic floor/dedup path."""
    if getattr(data_source.config, "name", "") != "aumc" or interval is None:
        return frame
    if not index_col or index_col not in frame:
        return frame
    if frame.empty:
        # Empty window sources must not donate stop/duration columns to
        # unrelated point sources at concat: the generic path would try to
        # expand their all-NA durations and discard the points.
        return frame.drop(columns=["dur_var", "duration", "stop"], errors="ignore")
    if "dur_var" in frame:
        from ..table.duration import resolve_dur_var_hours

        frame = frame.copy()
        frame["__end"] = frame[index_col] + resolve_dur_var_hours(frame) * 60.0
        stop_col = "__end"
    elif source.dur_var and source.dur_var in frame:
        stop_col = source.dur_var
    else:
        return frame
    return expand_intervals(
        frame, start_col=index_col, stop_col=stop_col, value_col=concept_name,
        data_source=data_source, union=True,
    )


def vaso_indicator(tables, ctx):
    """Union exact duration episodes on the same hour grid as AUMC rates."""
    from ..table import ICUTable

    pieces = []
    for name, table in tables.items():
        frame = table.data
        index = table.index_column
        if frame.empty or not index:
            continue
        frame = frame[["admissionid", index, name]].rename(columns={index: "start"}).copy()
        frame["stop"] = (frame.start + frame[name]) * 60.0
        frame["start"] *= 60.0
        frame["vaso_ind"] = True
        pieces.append(expand_intervals(frame, start_col="start", stop_col="stop", value_col="vaso_ind", union=True))
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["admissionid", "start", "vaso_ind"])
    out["start"] /= 60.0
    out = out.drop_duplicates(["admissionid", "start"])
    return ICUTable(out, id_columns=["admissionid"], index_column="start", value_column="vaso_ind")


def collapse_window_points(frame, *, ids, index_col, value_col, logical, interval):
    """Combine overlapping window sources after alignment, without first-row loss."""
    out = frame.copy()
    hours = interval.total_seconds() / 3600.0
    out[index_col] = np.floor(out[index_col] / hours) * hours
    keys = [*ids, index_col]
    if logical:
        return out.groupby(keys, as_index=False, dropna=False)[value_col].max()
    return out.drop_duplicates([*keys, value_col])


def ventilation_indicator(table, interval):
    """Preserve exact invasive occupancy; never stretch hourly points into spells."""
    from ..table import ICUTable
    from ..table.duration import resolve_dur_var_hours

    index = table.index_column
    frame = table.data
    # WinTbl.value_column can resolve to the start column; the canonical
    # categorical measurement is still named mech_vent in its payload.
    value = "mech_vent"
    frame = frame.loc[frame[value].astype("string").eq("invasive").fillna(False)].copy()
    frame["vent_ind"] = True
    if "dur_var" in frame:
        # Native-time dependency loads still contain admission-relative windows.
        frame["__stop"] = (frame[index] + resolve_dur_var_hours(frame)) * 60.0
        frame[index] *= 60.0
        frame = expand_intervals(
            frame, start_col=index, stop_col="__stop", value_col="vent_ind", union=True,
        )
        frame[index] /= 60.0
    frame = collapse_window_points(
        frame, ids=["admissionid"], index_col=index, value_col="vent_ind",
        logical=True, interval=interval,
    )
    return ICUTable(frame, id_columns=["admissionid"], index_column=index, value_column="vent_ind")
