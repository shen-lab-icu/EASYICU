"""eICU vasopressor exposure from medication orders.

In eICU, ``infusiondrug`` is the charted infusion and the only source of a
vasopressor dose.  Not every hospital charts vasopressors there: 11,454 of
the 24,591 stays with a norepinephrine order in ``medication`` have no
norepinephrine row in ``infusiondrug`` (2026-11-03 extraction audit round 2,
V1).  For those stays the order is the only evidence of the drug.

A source with this callback turns the orders of stays without any matching
infusion row into exposure durations: the row sits at ``drugstartoffset`` and
the value is the order span ``drugstopoffset - drugstartoffset`` in hours.
The span is when the order was active, not when the pump ran, and it never
gives a dose.  An order without a later stop gives a zero-hour exposure at its
start.  Stays that do chart the infusion keep their infusion-based durations
only, so the two sources are never pooled.  Cancelled orders are dropped by
the source's ``excluded_statuses`` before this callback.

An order is exposure only while the patient is in the unit: its stop is
clipped to ``patient.unitdischargeoffset`` and an order starting after
discharge is dropped.  Unclipped spans reached 8,784 h (2026-11-04 audit
round 3, R3-04).

So that an analysis can keep charted infusions only, ``norepi_order_dur``
reads the same order sources with this callback and repeats the order-derived
spans alone (round-12 ruling): a ``norepi_dur`` row whose ``norepi_order_dur``
is missing comes from a charted infusion.

Dictionary use::

    "table": "medication", "sub_var": "drugname", "regex": "...",
    "class_name": "rgx_itm", "extra_vars": ["drugstopoffset", ...],
    "callback": "eicu_order_exposure_duration",
    "infused_regex": "(?i)^(?:norepi|levophed)"
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

CALLBACK = "eicu_order_exposure_duration"
INFUSED_REGEX_PARAM = "infused_regex"
ID_COLUMN = "patientunitstayid"
START_COLUMN = "drugstartoffset"
STOP_COLUMN = "drugstopoffset"
DISCHARGE_COLUMN = "unitdischargeoffset"
#: Attribute naming how many stays came from orders only.
ORDER_ONLY_STAYS_ATTR = "eicu_order_only_stays"


def order_exposure_durations(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    infused_stays: Iterable,
    discharge_offsets: Mapping[Any, float] | None = None,
) -> pd.DataFrame:
    """Order spans (hours) of stays without an infusion row, inside the unit stay.

    ``discharge_offsets`` maps a stay to its ``unitdischargeoffset`` (minutes);
    a stay without one keeps its order span.
    """
    missing = {ID_COLUMN, START_COLUMN, STOP_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"eicu_order_exposure_columns_missing:{sorted(missing)}")
    rows = frame.loc[~frame[ID_COLUMN].isin(set(infused_stays))]
    start = pd.to_numeric(rows[START_COLUMN], errors="coerce")
    stop = pd.to_numeric(rows[STOP_COLUMN], errors="coerce")
    if discharge_offsets is not None:
        discharge = pd.to_numeric(rows[ID_COLUMN].map(discharge_offsets), errors="coerce")
        inside = ~(start >= discharge)
        rows, start, stop, discharge = rows[inside], start[inside], stop[inside], discharge[inside]
        stop = stop.where(~(stop > discharge), discharge)
    hours = ((stop - start) / 60.0).where(stop > start, 0.0)
    out = pd.DataFrame(
        {
            ID_COLUMN: rows[ID_COLUMN].to_numpy(),
            START_COLUMN: start.to_numpy(dtype="float64"),
            concept_name: hours.to_numpy(dtype="float64"),
        }
    )
    out = out.loc[np.isfinite(out[START_COLUMN])].reset_index(drop=True)
    out.attrs[ORDER_ONLY_STAYS_ATTR] = int(out[ID_COLUMN].nunique())
    return out


def apply_order_exposure(
    frame: pd.DataFrame, *, source: Any, concept_name: str, data_source: Any
) -> pd.DataFrame:
    """Source callback: find the stays that chart the infusion, then span."""
    from ..datasource import FilterOp, FilterSpec

    params = getattr(source, "params", None) or {}
    pattern = params.get(INFUSED_REGEX_PARAM)
    if not pattern:
        raise ValueError(f"{CALLBACK}_requires_{INFUSED_REGEX_PARAM}")
    if frame.empty:
        return order_exposure_durations(
            pd.DataFrame(columns=[ID_COLUMN, START_COLUMN, STOP_COLUMN]),
            concept_name=concept_name, infused_stays=(),
        )
    if data_source is None:
        raise ValueError(f"{CALLBACK}_requires_data_source")
    stays = pd.unique(frame[ID_COLUMN].dropna()).tolist()
    table = data_source.load_table(
        "infusiondrug",
        columns=[ID_COLUMN, "drugname"],
        filters=[FilterSpec(ID_COLUMN, FilterOp.IN, stays)],
        verbose=False,
    )
    infusions = table.data if hasattr(table, "data") else table
    matched = infusions["drugname"].astype("string").str.contains(
        re.compile(pattern), na=False
    )
    infused = set(infusions.loc[matched.fillna(False).to_numpy(), ID_COLUMN])
    patients = data_source.load_table(
        "patient",
        columns=[ID_COLUMN, DISCHARGE_COLUMN],
        filters=[FilterSpec(ID_COLUMN, FilterOp.IN, stays)],
        verbose=False,
    )
    patients = patients.data if hasattr(patients, "data") else patients
    discharge = dict(zip(patients[ID_COLUMN], pd.to_numeric(patients[DISCHARGE_COLUMN], errors="coerce")))
    return order_exposure_durations(
        frame, concept_name=concept_name, infused_stays=infused, discharge_offsets=discharge
    )


__all__ = [
    "CALLBACK",
    "ORDER_ONLY_STAYS_ATTR",
    "apply_order_exposure",
    "order_exposure_durations",
]
