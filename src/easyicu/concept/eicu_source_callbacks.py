"""Dispatch for explicit eICU source-quality callbacks (audit round 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import eicu_medication_evidence as medication
from . import eicu_source_qc as qc
from .eicu_io import systemic_intake

CALLBACKS = frozenset({
    "eicu_arterial_gas", "eicu_height_qc", "eicu_infection_evidence",
    "eicu_systemic_intake", "eicu_fio2_source", "eicu_fio2_parent",
    "eicu_loop_dose", "eicu_mode_target", "eicu_breath_sequence",
    "eicu_iv_exposure", "eicu_positive_infusion", "eicu_positive_value",
    "eicu_dex_order", "eicu_advanced_device", "eicu_circulatory_support",
    "eicu_ionized_calcium",
    "eicu_insulin_units",
})


def _circulatory_support(frame, column):
    work = frame.copy()
    text = work[column].astype("string").str.casefold()
    work["_device"] = text.str.extract(r"(iabp|intra.?aortic|balloon pump|lvad|impella|tandem|ventricular assist|\bvad\b)", expand=False)
    work["_device"] = work._device.str.replace(r"intra.?aortic|balloon pump", "iabp", regex=True)
    stop = text.str.contains(r"\b(?:remov\w*|discontinu\w*|stop\w*|off)\b", na=False)
    negative = text.str.contains(r"\b(?:no|not|consider\w*|plan\w*|declin\w*|refus\w*)\b", na=False)
    work["_active"] = ~stop & ~negative & work._device.notna()
    work["_stop"] = stop
    # A plan is not a completed transition; it cannot close an active span.
    uncertain = text.str.contains(r"\b(?:consider\w*|plan\w*|declin\w*|refus\w*)\b", na=False)
    work = work.loc[~uncertain].copy()
    work = work.sort_values([qc.STAY, "_device", "treatmentoffset"])
    rows = []
    for _, group in work.groupby([qc.STAY, "_device"], dropna=False):
        records = group.to_dict("records")
        for i, row in enumerate(records):
            start = float(row["treatmentoffset"])
            if not np.isfinite(start):
                continue
            row[column] = bool(row["_active"])
            end = start + 60
            if row["_active"] and i + 1 < len(records):
                following = records[i + 1]
                # An explicit same-device termination verifies the interval;
                # other repeated treatment points bridge at most six hours.
                end = float(following["treatmentoffset"])
                if not following["_stop"]:
                    end = min(end, start + 360)
            # Every overlapped hour counts; a stop exactly on the boundary
            # contributes no active evidence to the following hour.
            times = np.arange(np.floor(start / 60) * 60, end, 60) if row["_active"] else [start]
            for time in times:
                rows.append({**row, "treatmentoffset": time})
    out = pd.DataFrame(rows, columns=[*work.columns]).drop(columns=["_device", "_active", "_stop"])
    out.attrs["eicu_circulatory_support_audit"] = {"removal_rows": int(stop.sum()), "negative_rows": int(negative.sum()), "interval_rule": "same device; removal terminates; unclosed evidence at most six hours"}
    return out


def apply(frame, source, concept_name, data_source=None):
    expr = source.callback
    column = concept_name if concept_name in frame else source.value_var
    if not column or column not in frame:
        raise KeyError(f"{expr}: value column missing for {concept_name}")
    if expr == "eicu_arterial_gas":
        return qc.arterial_gases(frame, data_source)
    if expr == "eicu_height_qc":
        return qc.height_qc(frame, column)
    if expr == "eicu_infection_evidence":
        return qc.infection_evidence(frame, column)
    if expr == "eicu_systemic_intake":
        return systemic_intake(frame, column)
    if expr == "eicu_fio2_source":
        return qc.fio2_values(frame, column, source.table)
    if expr == "eicu_fio2_parent":
        return qc.fio2_parent(frame, column, source.table, data_source)
    if expr == "eicu_loop_dose":
        return medication.loop_dose(frame, column, data_source)
    if expr in {"eicu_mode_target", "eicu_breath_sequence"}:
        return qc.mode_values(frame, column, expr == "eicu_breath_sequence")
    if expr == "eicu_iv_exposure":
        return medication.iv_exposure(frame, column)
    if expr == "eicu_positive_infusion":
        return medication.positive_infusion(frame, column)
    if expr == "eicu_insulin_units":
        return medication.insulin_units(frame, column)
    if expr == "eicu_positive_value":
        out = frame.copy()
        values = pd.to_numeric(frame[column], errors="coerce")
        out[column] = values.gt(0).astype("boolean").mask(~np.isfinite(values) | values.lt(0))
        return out
    if expr == "eicu_dex_order":
        return medication.dex_dose(frame, column, data_source)
    if expr == "eicu_advanced_device":
        out = frame.copy()
        text = frame[column].astype("string").str.casefold().str.strip()
        hfnc = text.str.contains(r"\bhfnc\b|high[- ]?flow(?: nasal)?(?: cannula)?", na=False)
        vent = text.isin(["ventilator", "mechanical ventilator", "ett", "bi-pap", "bipap", "cpap", "niv"])
        out = out.loc[hfnc | vent].copy()
        out[column] = True
        out["eicu_respiratory_support_class"] = np.where(hfnc.loc[out.index], "hfnc", "mechanical")
        return out
    if expr == "eicu_circulatory_support":
        return _circulatory_support(frame, column)
    if expr == "eicu_ionized_calcium":
        out = frame.copy()
        values = pd.to_numeric(frame[column], errors="coerce")
        mgdl = frame.labmeasurenameinterface.astype("string").str.fullmatch(r"\s*mg\s*/\s*dl\s*", case=False, na=False)
        out[column] = values.where(~(values.gt(2.5) | (mgdl & values.gt(2))), values / 4.008)
        return out
    raise ValueError(f"unsupported eICU source callback: {expr}")
