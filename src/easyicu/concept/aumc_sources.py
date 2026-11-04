"""Source-specific AUMC value and evidence rules from extraction audit round 5."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .aumc_intervals import expand_intervals
from .blood_gas_specimen import SPECIMEN_RULES, _load_specimen_records, normalize_specimen_token

logger = logging.getLogger(__name__)


def tidal_volume(frame, value_col):
    out = frame.copy()
    raw = pd.to_numeric(out[value_col], errors="coerce")
    # 8851 has both litre and millilitre populations despite its ml label.
    # The parked 2 L knob maximum and the gap between populations are unknown.
    litres = raw.between(.1, 1.5)
    ml = raw.between(100, 1500)
    out[value_col] = raw.where(ml, raw.where(litres) * 1000)
    rejected = int((raw.notna() & ~(litres | ml)).sum())
    out.attrs["aumc_tidal_unit_rejected"] = rejected
    logger.info("AUMC 8851 unit normalization rejected %d ambiguous row(s)", rejected)
    return out


def rrt_evidence(frame, value_col):
    out = frame.copy()
    raw = pd.to_numeric(out[value_col], errors="coerce")
    pressure = out.itemid.isin([7666, 7667, 7668, 7671, 14838])
    positive = np.isfinite(raw) & ((pressure & raw.ne(0)) | (~pressure & raw.gt(0)))
    # Absence of a positive treatment reading does not document no treatment.
    out[value_col] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[positive, value_col] = True
    return out


def central_saturation(frame, value_col, data_source):
    if frame.empty:
        return frame
    rule = SPECIMEN_RULES["aumc"]
    site = _load_specimen_records(frame, rule.also[0], data_source)
    extra = _load_specimen_records(frame, rule, data_source)
    keys = list(rule.keys)
    tokens = site["value"].map(normalize_specimen_token)
    sites = site[keys].assign(recorded=tokens.ne(""), conflict=tokens.ne("") & tokens.ne("CENTRALE LIJN"))
    status = sites.groupby(keys, dropna=True)[["recorded", "conflict"]].any()
    allowed = status.index[status.recorded & ~status.conflict]
    # Conservative cross-era agreement: any nonempty auxiliary specimen
    # description is excluded until its central-site semantics are reviewed.
    forbidden = pd.MultiIndex.from_frame(extra.loc[extra.value.map(normalize_specimen_token).ne(""), keys])
    index = pd.MultiIndex.from_frame(frame[keys])
    out = frame.loc[index.isin(allowed) & ~index.isin(forbidden)].copy()
    out.loc[out.itemid.eq(12311), value_col] *= 100.0
    return out


def dextrose(frame, value_col, data_source):
    """D10-equivalent interval totals; never stretch a bag's start or stop."""
    if "doseunit" not in frame:
        raise ValueError("aumc_dextrose requires its declared volume-unit column")
    out = frame.copy()
    unit = out.doseunit.astype("string").str.strip().str.casefold()
    rate_unit = out.doserateunit.astype("string").str.strip().str.casefold()
    factor = out.itemid.map({7254: 1., 7255: 2., 7256: 3., 8940: 4., 9571: 4.})
    per_hour = rate_unit.map({"uur": 1., "hour": 1., "hr": 1., "min": 60., "minute": 60., "dag": 1. / 24})
    rate = pd.to_numeric(out[value_col], errors="coerce") * per_hour
    minutes = pd.to_numeric(out.stop, errors="coerce") - pd.to_numeric(out.start, errors="coerce")
    flush = out.action.astype("string").str.contains(r"(?i)^\s*flush\s*$", na=False)
    total = rate * minutes / 60.0
    # A labelled single-bag administered volume is an amount, not a rate.
    if {"administered", "administeredunit"} <= set(out.columns):
        intermittent = pd.to_numeric(out.iscontinuous, errors="coerce").eq(0)
        volume_unit = out.administeredunit.astype("string").str.strip().str.casefold().eq("ml")
        amount = pd.to_numeric(out.administered, errors="coerce")
        total = total.where(~(intermittent & volume_unit & amount.notna()), amount)
    out[value_col] = (total * factor).where(unit.eq("ml") & ~flush & minutes.gt(0) & np.isfinite(total) & total.gt(0))
    # Only values and exact source times are repeated; totals are summed once
    # across concurrent bags after overlap allocation.
    return expand_intervals(
        out[["admissionid", "start", "stop", value_col]], start_col="start",
        stop_col="stop", value_col=value_col, data_source=data_source, allocate=True,
    ).assign(doseunit="ml/hr")


def respiratory_modes(frame, value_col, data_source=None):
    """A set PEEP alone cannot establish active respiratory support.

    Original-time mode evidence is retained even when another device says
    standby. Such conflicts are counted before hour aggregation.
    """
    out = frame.copy()
    tokens = out[value_col].astype("string").str.strip().str.casefold()
    standby = tokens.str.fullmatch(r"stand\s*by", na=False)
    keys = ["admissionid", "measuredat"]
    # The dictionary identifies distinct Servo, Evita 4 and Evita 1 channels.
    device = out.itemid.map({12290: "servo", 6685: "evita4", 9534: "evita1"})
    grouped = out[keys].assign(device=device, standby=standby)
    same_device_standby = grouped.groupby(keys + ["device"]).standby.transform("any")
    active = tokens.notna() & tokens.ne("") & ~standby & ~same_device_standby
    times_with_standby = pd.MultiIndex.from_frame(out.loc[standby, keys])
    conflict = active & pd.MultiIndex.from_frame(out[keys]).isin(times_with_standby)
    process_conflicts = _standby_process_conflicts(out.loc[standby, keys], data_source)
    out = out.loc[active].copy()
    out[value_col] = True
    out.attrs["aumc_active_standby_conflicts"] = int(conflict.sum()) + process_conflicts
    logger.info("AUMC active/standby mode conflicts at original times: %d", out.attrs["aumc_active_standby_conflicts"])
    return out


def _standby_process_conflicts(standby, data_source):
    """Count explicit processes active at original standby timestamps."""
    if standby.empty or data_source is None:
        return 0
    from ..datasource import FilterOp, FilterSpec

    table = data_source.load_table(
        "processitems", columns=["admissionid", "itemid", "start", "stop"],
        filters=[FilterSpec("admissionid", FilterOp.IN, standby.admissionid.unique().tolist()),
                 FilterSpec("itemid", FilterOp.IN, [9328, 10740])], verbose=False,
    )
    processes = table.data if hasattr(table, "data") else table
    processes = processes.dropna(subset=["start", "stop"])
    processes = processes.loc[processes.stop.gt(processes.start)]
    if processes.empty:
        return 0
    processes = processes.sort_values(["admissionid", "start"]).copy()
    processes["stop"] = processes.groupby("admissionid").stop.cummax()
    matched = pd.merge_asof(
        standby.drop_duplicates().sort_values("measuredat"),
        processes.sort_values("start"), left_on="measuredat", right_on="start",
        by="admissionid", direction="backward",
    )
    return int(matched.measuredat.lt(matched.stop).sum())


def apply_source_rule(frame, source, concept_name, data_source, unit_column):
    value_col = concept_name if concept_name in frame else (source.value_var or concept_name)
    callback = source.callback
    if callback == "aumc_positive_dose":
        dose = pd.to_numeric(frame[value_col], errors="coerce")
        out = frame.loc[np.isfinite(dose) & dose.gt(0)].copy()
        out[value_col] = True
        return out
    if callback == "aumc_tidal_volume_set":
        return tidal_volume(frame, value_col)
    if callback == "aumc_rrt_evidence":
        return rrt_evidence(frame, value_col)
    if callback == "aumc_central_saturation":
        return central_saturation(frame, value_col, data_source)
    if callback == "aumc_dextrose":
        return dextrose(frame, value_col, data_source)
    if callback == "aumc_respiratory_modes":
        return respiratory_modes(frame, value_col, data_source)
    raise ValueError(f"Unknown AUMC source callback: {callback}")
